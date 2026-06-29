"""Transcribe audio to SRT using faster-whisper (local GPU).

Requires nvidia-cublas-cu12 and nvidia-cudnn-cu12 installed via pip
to provide the CUDA runtime libraries for ctranslate2.
"""

import ctypes
import logging
import os
import subprocess
import threading
import time
from pathlib import Path

# ── Limit CPU thread count before any C-library import ─────────────────
# ctranslate2 / ONNX / OpenMP can spawn dozens of threads, which
# multiplies memory usage on memory-constrained WSL2 systems.
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")

logger = logging.getLogger(__name__)

# ── Preload CUDA libs ──────────────────────────────────────────────────
# ctranslate2 (used by faster-whisper) links libcublas.so.12 at import
# time via dlopen.  On Debian/Ubuntu without the system CUDA toolkit,
# those .so files live in the pip-installed nvidia-cublas-cu12 package
# under ~/.local.  Setting LD_LIBRARY_PATH in Python is too late because
# the dynamic linker already resolved; instead we use ctypes to preload
# the library before importing faster_whisper.

_CUDA_LIBS = os.path.expanduser(
    "~/.local/lib/python3.12/site-packages/nvidia/cublas/lib"
)
_CUBLAS_SO = os.path.join(_CUDA_LIBS, "libcublas.so.12")
if os.path.exists(_CUBLAS_SO):
    try:
        ctypes.CDLL(_CUBLAS_SO, ctypes.RTLD_GLOBAL)
        logger.debug("Preloaded %s", _CUBLAS_SO)
    except Exception as e:
        logger.warning("Could not preload CUDA libs: %s", e)

from faster_whisper import WhisperModel

# ── Model cache ────────────────────────────────────────────────────────
_MODELS = {}
_DIAG_LOCK = threading.Lock()
_DIAG_PATH = None          # set by transcribe_audio() before any work

# ═══════════════════════════════════════════════════════════════════════
# Crash-safe diagnostic capture
#
# Every log line emitted by our logger is mirrored to a
# _transcribe_diag.log file with immediate fsync.  If the Python
# process is killed (OOM, segfault, WSL crash), the diagnostic file
# survives with everything up to the instant of death.
#
# We do NOT attempt to capture C-level stderr (CUDA runtime messages
# etc.) because doing so requires os.dup2 manipulation of fd 2, which
# is too risky — it can leave the terminal in a bad state if the
# process crashes mid-redirect.
# ═══════════════════════════════════════════════════════════════════════


def _diag_path(output_dir: str) -> str:
    return os.path.join(output_dir, "_transcribe_diag.log")


def _diag_write(msg: str):
    """Thread-safe fsynced append to the current diag file."""
    path = _DIAG_PATH
    if not path:
        return
    try:
        Path(os.path.dirname(path)).mkdir(parents=True, exist_ok=True)
        with _DIAG_LOCK:
            # Re-open every time so a crash mid-write never corrupts an
            # open file handle — the OS closes it atomically on death.
            with open(path, "a") as f:
                f.write(f"{time.strftime('%H:%M:%S')} {msg}\n")
                f.flush()
                os.fsync(f.fileno())
    except Exception:
        pass  # diagnostics must never block the pipeline


# ── 1. Logging handler — captures every logger.info/warning/error ─────

class _DiagLogHandler(logging.Handler):
    """Mirror all log records to the fsynced diag file."""

    def __init__(self):
        super().__init__(level=logging.DEBUG)

    def emit(self, record):
        try:
            msg = self.format(record)
            _diag_write(f"LOG [{record.levelname}] {msg}")
        except Exception:
            pass


_DIAG_LOG_HANDLER = _DiagLogHandler()
_DIAG_LOG_HANDLER.setFormatter(logging.Formatter(
    "%(name)s: %(message)s"
))
logging.getLogger(__name__).addHandler(_DIAG_LOG_HANDLER)


def _get_model(model_size: str = "base"):
    """Get or create a cached WhisperModel instance."""
    if model_size not in _MODELS:
        logger.info("Loading faster-whisper model '%s' on cuda...", model_size)
        _MODELS[model_size] = WhisperModel(
            model_size, device="cuda", compute_type="int8",
        )
        logger.info("Model '%s' loaded", model_size)
    return _MODELS[model_size]


def _srt_timestamp(sec: float) -> str:
    """Format a float seconds as SRT timestamp (HH:MM:SS,mmm)."""
    h = int(sec // 3600)
    m = int((sec % 3600) // 60)
    s = int(sec % 60)
    ms = int((sec - int(sec)) * 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


# ── 3. Watchdog — heartbeat during the blocking transcribe() call ─────

_WATCHDOG_STOP = threading.Event()


def _watchdog_loop(diag_path: str, interval: float = 30.0):
    """Daemon thread: log heartbeat + GPU/system memory every *interval* s.

    Runs until ``_WATCHDOG_STOP`` is set (by the main thread after
    ``model.transcribe()`` returns).  Uses ``_diag_write`` so every
    heartbeat is fsynced immediately.
    """
    tick = 0
    while not _WATCHDOG_STOP.wait(interval):
        tick += 1
        _diag_write(f"WATCHDOG still alive (tick={tick}, interval={interval:.0f}s)")

        # GPU memory every 2 ticks (60 s)
        if tick % 2 == 0:
            try:
                result = subprocess.run(
                    ["nvidia-smi", "--query-gpu=memory.used,memory.total,memory.free",
                     "--format=csv,noheader,nounits"],
                    capture_output=True, text=True, timeout=5,
                )
                if result.returncode == 0:
                    parts = result.stdout.strip().split(",")
                    if len(parts) == 3:
                        used, total, free = [p.strip() for p in parts]
                        _diag_write(
                            f"WATCHDOG GPU mem: used={used} free={free} total={total} MiB"
                        )
            except Exception as exc:
                _diag_write(f"WATCHDOG nvidia-smi failed: {exc}")

        # System memory every 4 ticks (120 s)
        if tick % 4 == 0:
            try:
                with open("/proc/meminfo") as f:
                    raw = f.read()
                lines = [
                    ln for ln in raw.splitlines()
                    if any(ln.startswith(k) for k in
                           ["MemTotal:", "MemFree:", "MemAvailable:",
                            "SwapTotal:", "SwapFree:", "Buffers:", "Cached:"])
                ]
                _diag_write("WATCHDOG Sys mem: " + " | ".join(lines))
            except Exception as exc:
                _diag_write(f"WATCHDOG meminfo failed: {exc}")


def _run_watchdog(diag_path: str):
    """Start (or restart) the watchdog daemon thread."""
    _WATCHDOG_STOP.clear()
    t = threading.Thread(
        target=_watchdog_loop,
        args=(diag_path,),
        daemon=True,
        name="transcribe-watchdog",
    )
    t.start()
    return t


def _stop_watchdog():
    """Signal the watchdog to exit."""
    _WATCHDOG_STOP.set()


def _log_gpu_memory(label: str):
    """Log GPU memory usage from nvidia-smi (fsynced)."""
    try:
        result = subprocess.run(
            [
                "nvidia-smi", "--query-gpu=memory.used,memory.total,memory.free",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode == 0:
            parts = result.stdout.strip().split(",")
            if len(parts) == 3:
                used, total, free = [p.strip() for p in parts]
                msg = f"GPU mem [{label}]: used={used} free={free} total={total} MiB"
                logger.info("📊 %s", msg)
    except Exception as exc:
        logger.debug("nvidia-smi failed: %s", exc)


def _log_system_memory(label: str):
    """Log system RAM / swap from /proc/meminfo (fsynced)."""
    try:
        with open("/proc/meminfo") as f:
            raw = f.read()
        lines = [
            ln for ln in raw.splitlines()
            if any(ln.startswith(k) for k in
                   ["MemTotal:", "MemFree:", "MemAvailable:",
                    "SwapTotal:", "SwapFree:", "Buffers:", "Cached:"])
        ]
        logger.info("📊 Sys mem [%s]: %s", label, " | ".join(lines))
    except Exception as exc:
        logger.debug("sys mem read failed: %s", exc)


def _log_gpu_processes(label: str):
    """Log GPU compute processes running now (fsynced)."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory",
             "--format=csv,noheader"],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode == 0 and result.stdout.strip():
            for line in result.stdout.strip().splitlines():
                _diag_write(f"GPU proc [{label}]: {line.strip()}")
        else:
            _diag_write(f"GPU proc [{label}]: (none)")
    except Exception as exc:
        _diag_write(f"GPU proc [{label}]: nvidia-smi error — {exc}")


def _log_audio_info(audio_path: str):
    """Log audio file size and duration via ffprobe (fsynced)."""
    try:
        size_mb = os.path.getsize(audio_path) / (1024 * 1024)
        result = subprocess.run(
            [
                "ffprobe", "-v", "error", "-show_entries",
                "format=duration", "-of", "csv=p=0",
                audio_path,
            ],
            capture_output=True, text=True, timeout=10,
        )
        dur = result.stdout.strip()
        logger.info("📊 Audio: %s MiB, duration=%ss", size_mb, dur)
    except Exception as exc:
        logger.debug("audio info failed: %s", exc)


def transcribe_audio(
    audio_path: str,
    output_dir: str,
    model_size: str = "base",
    beam_size: int = 1,
    vad_filter: bool = False,
    condition_on_previous_text: bool = False,
) -> str:
    """Transcribe audio to SRT using faster-whisper.

    Writes ``source_subtitles.srt`` to *output_dir*.

    Resumable: if the target SRT already exists, returns immediately
    without re-transcribing.

    Kwargs default to memory-conservative settings (beam_size=1,
    vad_filter=False) to avoid OOM on 8 GB / 6 GB GPU hardware.
    Increase beam_size to 3-5 and enable vad_filter when running on
    a machine with 16+ GB RAM / 8+ GB VRAM.

    Returns:
        Path to the output SRT file.
    """
    global _DIAG_PATH

    output_srt = os.path.join(output_dir, "source_subtitles.srt")
    if os.path.exists(output_srt):
        logger.info("Transcription checkpoint HIT — %s exists, reusing", output_srt)
        return output_srt

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # ── Set global diag path ──────────────────────────────────────────
    _DIAG_PATH = _diag_path(output_dir)

    # ── Retry detection ──────────────────────────────────────────────
    diag_path = _DIAG_PATH
    if os.path.exists(diag_path):
        try:
            with open(diag_path) as f:
                prev = f.read().strip()
            if prev:
                last_line = prev.splitlines()[-1]
                logger.info("📊 DIAG RETRY — previous run last diagnostic: %s", last_line)
                _diag_write("=== RETRY of previous run ===")
                _diag_write(f"Previous last line: {last_line}")
        except Exception:
            pass

    # ── Crash-safe diagnostics before transcription ──────────────────
    _diag_write("=== TRANSCRIBE_START ===")
    _log_system_memory("before-transcribe")
    _log_audio_info(audio_path)
    _log_gpu_memory("before-transcribe")
    _log_gpu_processes("before-transcribe")

    # ── Model load (cached globally; first call does real loading) ────
    _diag_write("MODEL_LOAD_START")
    t_model = time.time()
    model = _get_model(model_size)
    _diag_write(f"MODEL_LOAD_DONE in {time.time() - t_model:.1f}s")
    _log_gpu_memory("after-model-load")

    t_start = time.time()
    _diag_write("CALLING model.transcribe()")
    logger.info("Starting faster-whisper transcribe() call...")

    # ── Start watchdog (heartbeat while transcribe blocks) ──────────
    # Stays running through BOTH the transcribe() call AND the segment
    # iteration — faster-whisper's generator does on-the-fly decode,
    # so CUDA is active the whole time.
    watchdog_thread = _run_watchdog(diag_path)

    try:
        # Memory-conservative settings:
        #   beam_size=1          — no multi-hypothesis overhead
        #   vad_filter=False      — no VAD pre-pass (biggest memory spike)
        #   condition_on_previous_text=False  — prevents memory leak from
        #                                       growing n-gram cache on
        #                                       very long audio
        segments, info = model.transcribe(
            audio_path,
            beam_size=beam_size,
            language="en",
            vad_filter=vad_filter,
            condition_on_previous_text=condition_on_previous_text,
        )
    except Exception as exc:
        _stop_watchdog()
        _diag_write(f"TRANSCRIBE_EXCEPTION: {type(exc).__name__}: {exc}")
        import traceback as _tb
        _diag_write(f"TRANSCRIBE_EXCEPTION traceback: {_tb.format_exc()}")
        logger.exception("model.transcribe() raised an exception")
        raise

    elapsed_init = time.time() - t_start
    _diag_write(
        f"Transcribe() RETURNED generator after {elapsed_init:.1f}s — "
        f"language={info.language} prob={info.language_probability*100:.0f}%",
    )
    logger.info(
        "Transcribe() returned generator after %.1fs — language %s (%.0f%%), iterating...",
        elapsed_init,
        info.language, info.language_probability * 100,
    )

    count = 0
    log_every = 100
    try:
        with open(output_srt, "w", encoding="utf-8") as f:
            for seg in segments:
                count += 1
                f.write(f"{count}\n")
                f.write(f"{_srt_timestamp(seg.start)} --> {_srt_timestamp(seg.end)}\n")
                f.write(f"{seg.text.strip()}\n\n")

                if count % log_every == 0:
                    elapsed = time.time() - t_start
                    seg_dur = seg.end - seg.start
                    msg = (
                        f"Progress: seg={count} elapsed={elapsed:.0f}s "
                        f"current_seg={seg.start:.1f}–{seg.end:.1f} dur={seg_dur:.1f}s"
                    )
                    logger.info("📊 %s", msg)
    except Exception as exc:
        _stop_watchdog()
        _diag_write(f"SEGMENT_ITER_EXCEPTION: {type(exc).__name__}: {exc}")
        import traceback as _tb
        _diag_write(f"SEGMENT_ITER_EXCEPTION traceback: {_tb.format_exc()}")
        logger.exception("Segment iteration crashed")
        raise

    _stop_watchdog()
    total_time = time.time() - t_start
    _log_gpu_memory("after-transcribe")
    _diag_write(f"=== TRANSCRIBE_DONE: {count} segments in {total_time:.0f}s ===")
    logger.info(
        "Transcription complete: %d segments -> %s in %.0fs",
        count, output_srt, total_time,
    )
    # Clear global model cache to release CUDA context before next episode.
    # Prevents accumulated driver state corruption across many transcriptions
    # in the same Python process (known WSL2 issue with ctranslate2).
    _MODELS.clear()
    _diag_write("MODEL_CACHE_CLEARED")
    return output_srt
