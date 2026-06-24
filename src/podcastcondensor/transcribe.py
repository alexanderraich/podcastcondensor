"""Transcribe audio to SRT using faster-whisper (local GPU).

Requires nvidia-cublas-cu12 and nvidia-cudnn-cu12 installed via pip
to provide the CUDA runtime libraries for ctranslate2.
"""

import ctypes
import logging
import os
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

    model = _get_model(model_size)

    segments, info = model.transcribe(
        audio_path, beam_size=5, language="en", vad_filter=True,
    )
    logger.info(
        "Transcribing: detected language %s (%.0f%%)",
        info.language, info.language_probability * 100,
    )

    count = 0
    with open(output_srt, "w", encoding="utf-8") as f:
        for seg in segments:
            count += 1
            f.write(f"{count}\n")
            f.write(f"{_srt_timestamp(seg.start)} --> {_srt_timestamp(seg.end)}\n")
            f.write(f"{seg.text.strip()}\n\n")

    logger.info("Transcription complete: %d segments -> %s", count, output_srt)
    return output_srt
