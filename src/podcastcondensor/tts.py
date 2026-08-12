"""Text-to-speech — edge-tts (Microsoft neural voices).

``synthesize_narration`` renders plain narration prose to an MP3 via
``edge-tts``. edge-tts has a per-request character limit, so the prose is
split into sentence-bounded chunks (never mid-sentence), each chunk is
rendered, and the parts are concatenated with ffmpeg into one file.

edge-tts requires internet access to Microsoft's TTS service (the quality
choice; no API key). It is not installed by default — see the CLI command's
prerequisite note.
"""

import logging
import os
import re
import subprocess
import sys
from typing import List

logger = logging.getLogger(__name__)

# edge-tts SSML limit is ~4096 chars per request; stay comfortably under it.
_CHUNK_MAX_CHARS = 3500

# Sentence-ending punctuation — a chunk never splits inside a sentence.
_SENTENCE_END_RE = re.compile(r"(?<=[.!?])\s+")

_DEFAULT_VOICE = "en-US-AndrewNeural"


def _split_sentences(text: str) -> List[str]:
    """Split prose into sentences at sentence-ending punctuation."""
    text = text.strip()
    if not text:
        return []
    parts = _SENTENCE_END_RE.split(text)
    return [p.strip() for p in parts if p.strip()]


def _chunk_sentences(sentences: List[str], max_chars: int = _CHUNK_MAX_CHARS) -> List[str]:
    """Pack sentences into chunks of at most ``max_chars`` chars.

    Sentence-bounded: a sentence is never split. A single sentence longer
    than ``max_chars`` is emitted whole (edge-tts caps those internally;
    rare in narration prose).
    """
    chunks: List[str] = []
    current = ""
    for s in sentences:
        candidate = f"{current} {s}".strip() if current else s
        if len(candidate) <= max_chars:
            current = candidate
        else:
            if current:
                chunks.append(current)
            current = s
    if current:
        chunks.append(current)
    return chunks


def _render_chunk(chunk: str, out_path: str, voice: str) -> None:
    """Render one chunk to ``out_path`` via edge-tts."""
    # edge-tts is a CLI: `edge-tts --voice V --text "..." --write-media out.mp3`.
    # Invoke via `python -m edge_tts` so it works inside a venv (the console
    # script may not be on PATH when running with the venv's interpreter).
    proc = subprocess.run(
        [sys.executable, "-m", "edge_tts", "--voice", voice,
         "--text", chunk, "--write-media", out_path],
        capture_output=True, text=True, timeout=180,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"edge-tts failed (exit {proc.returncode}): {proc.stderr.strip()[:300]}"
        )


def synthesize_narration(
    text: str,
    out_path: str,
    voice: str = _DEFAULT_VOICE,
    keep_parts: bool = False,
) -> str:
    """Render ``text`` to an MP3 at ``out_path``; return the path.

    Splits into sentence-bounded chunks, renders each, concatenates with
    ffmpeg (silent gaps between chunks are naturally filled by the audio).
    ``keep_parts`` keeps the per-chunk temp files for debugging.

    Raises RuntimeError if edge-tts is missing or fails.
    """
    import tempfile

    chunks = _chunk_sentences(_split_sentences(text))
    if not chunks:
        raise RuntimeError("No text to synthesize")

    with tempfile.TemporaryDirectory(prefix="los-tts-") as tmpdir:
        part_paths = []
        for i, chunk in enumerate(chunks):
            part_path = os.path.join(tmpdir, f"part-{i:03d}.mp3")
            logger.info("TTS chunk %d/%d (%d chars)", i + 1, len(chunks), len(chunk))
            _render_chunk(chunk, part_path, voice)
            part_paths.append(part_path)

            if keep_parts:
                saved = os.path.join(os.path.dirname(out_path), f"tts_part-{i:03d}.mp3")
                subprocess.run(["cp", part_path, saved], check=True)

        # Concatenate the parts with ffmpeg concat demuxer.
        list_path = os.path.join(tmpdir, "parts.txt")
        with open(list_path, "w", encoding="utf-8") as f:
            for p in part_paths:
                f.write(f"file '{os.path.abspath(p)}'\n")
        os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
        ff = subprocess.run(
            ["ffmpeg", "-y", "-f", "concat", "-safe", "0",
             "-i", list_path, "-c", "copy", out_path],
            capture_output=True, text=True,
        )
        if ff.returncode != 0:
            raise RuntimeError(f"ffmpeg concat failed: {ff.stderr.strip()[:300]}")

    logger.info("Wrote %s (%d chunks)", out_path, len(chunks))
    return out_path
