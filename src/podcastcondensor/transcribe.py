"""Transcribe audio to SRT using faster-whisper (local GPU).

Requires nvidia-cublas-cu12 and nvidia-cudnn-cu12 installed via pip
to provide the CUDA runtime libraries for ctranslate2.
"""

import ctypes
import logging
import os
import subprocess
import time
from pathlib import Path

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


# ── Crash-safe diagnostics ──────────────────────────────────────────────
# logger.info() uses buffered I/O — if we crash (OOM, segfault, WSL kill),
# the last log lines are lost.  We write all diagnostic data to a dedicated
# _transcribe_diag.log file with immediate fsync so it survives any crash.
# This doubles the data: once to the logger (user sees in terminal) and
# once to the fsynced file (survives crash).


def _diag_path(output_dir: str) -> str:
    return os.path.join(output_dir, "_transcribe_diag.log")


def _diag(output_dir: str, msg: str):
    """Crash-safe diagnostic write — fsynced immediately."""
    path = _diag_path(output_dir)
    try:
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        with open(path, "a") as f:
            f.write(f"{time.strftime('%H:%M:%S')} {msg}\n")
            f.flush()
            os.fsync(f.fileno())
    except Exception:
        pass  # diagnostics must never block the pipeline


def _log_gpu_memory(output_dir: str, label: str):
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
                _diag(output_dir, msg)
    except Exception as exc:
        logger.debug("nvidia-smi failed: %s", exc)


def _log_audio_info(output_dir: str, audio_path: str):
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
        msg = f"Audio: {size_mb:.0f} MiB, duration={dur}s"
        logger.info("📊 %s", msg)
        _diag(output_dir, msg)
    except Exception as exc:
        logger.debug("audio info failed: %s", exc)


def transcribe_audio(
    audio_path: str,
    output_dir: str,
    model_size: str = "base",
) -> str:
    """Transcribe audio to SRT using faster-whisper.

    Writes ``source_subtitles.srt`` to *output_dir*.

    Resumable: if the target SRT already exists, returns immediately
    without re-transcribing.

    Returns:
        Path to the output SRT file.
    """
    output_srt = os.path.join(output_dir, "source_subtitles.srt")
    if os.path.exists(output_srt):
        logger.info("Transcription checkpoint HIT — %s exists, reusing", output_srt)
        return output_srt

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # ── Crash-safe diagnostics before transcription ──────────────────
    _diag(output_dir, "=== TRANSCRIBE_START ===")
    _log_audio_info(output_dir, audio_path)
    _log_gpu_memory(output_dir, "before-transcribe")

    model = _get_model(model_size)

    t_start = time.time()
    _diag(output_dir, "CALLING model.transcribe()")
    logger.info("Starting faster-whisper transcribe() call...")
    segments, info = model.transcribe(
        audio_path, beam_size=3, language="en", vad_filter=True,
    )
    elapsed_init = time.time() - t_start
    _diag(
        output_dir,
        f"TRanscribe() RETURNED generator after {elapsed_init:.1f}s — "
        f"language={info.language} prob={info.language_probability*100:.0f}%",
    )
    logger.info(
        "Transcribe() returned generator after %.1fs — language %s (%.0f%%), iterating...",
        elapsed_init,
        info.language, info.language_probability * 100,
    )

    count = 0
    log_every = 100
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
                _diag(output_dir, msg)

    total_time = time.time() - t_start
    _log_gpu_memory(output_dir, "after-transcribe")
    _diag(output_dir, f"=== TRANSCRIBE_DONE: {count} segments in {total_time:.0f}s ===")
    logger.info(
        "Transcription complete: %d segments -> %s in %.0fs",
        count, output_srt, total_time,
    )
    return output_srt
