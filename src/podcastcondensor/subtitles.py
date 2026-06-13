"""Subtitle parsing — normalize .srt and .vtt to clean chunk objects."""

import logging
import os
import re
from typing import List

logger = logging.getLogger(__name__)

TIMESTAMP_RE = re.compile(
    r"(\d{1,2}):(\d{2}):(\d{2})[.,](\d{1,3})"
)
SRT_BLOCK_RE = re.compile(
    r"(\d+)\s*\n"
    r"(\d{2}:\d{2}:\d{2}[.,]\d{1,3})\s*-->\s*(\d{2}:\d{2}:\d{2}[.,]\d{1,3})\s*\n"
    r"((?:(?!\n\n|\n$).+\n?)*)",
    re.MULTILINE,
)


def _ts_to_seconds(ts_str: str) -> float:
    """Convert HH:MM:SS.mmm or HH:MM:SS,mmm to seconds."""
    m = TIMESTAMP_RE.match(ts_str.strip())
    if not m:
        raise ValueError(f"Cannot parse timestamp: {ts_str}")
    h, mi, s, ms = int(m.group(1)), int(m.group(2)), int(m.group(3)), int(m.group(4))
    # ms can be 2 or 3 digits
    if len(str(ms)) == 2:
        ms = ms * 10
    return h * 3600 + mi * 60 + s + ms / 1000.0


def _normalize_vtt(text: str) -> str:
    """Remove WEBVTT header and metadata lines, return clean SRT-like blocks."""
    # Remove header up to first blank line
    lines = text.split("\n")
    clean = []
    found_first = False
    for line in lines:
        if line.strip() == "":
            if found_first:
                clean.append("")
            continue
        if line.strip() == "WEBVTT":
            found_first = True
            continue
        if line.startswith("Kind:") or line.startswith("Language:"):
            continue
        if re.match(r"^\d{2}:\d{2}", line.strip()):
            # Convert VTT timestamp separator to SRT style
            line = line.replace(".", ",")
        clean.append(line)
    return "\n".join(clean)


def _remove_duplicates(chunks: List[dict]) -> List[dict]:
    """Remove consecutive duplicate text entries (caption carryover artifacts)."""
    result = []
    prev_text = ""
    for c in chunks:
        text = c["text"].strip().lower()
        # Also skip if text is fully contained in previous
        if text == prev_text or (prev_text and text in prev_text):
            continue
        # Skip empty
        if not text:
            continue
        result.append(c)
        prev_text = text
    return result


def parse_srt_text(text: str) -> List[dict]:
    """Parse SRT text into list of chunk dicts."""
    # If it's VTT, normalize first
    if text.strip().startswith("WEBVTT"):
        text = _normalize_vtt(text)

    chunks = []
    for m in SRT_BLOCK_RE.finditer(text):
        seq = int(m.group(1))
        start = _ts_to_seconds(m.group(2))
        end = _ts_to_seconds(m.group(3))
        raw_text = m.group(4).strip().replace("\n", " ").replace("  ", " ")
        # Remove HTML tags sometimes present in VTT -> SRT
        clean_text = re.sub(r"<[^>]+>", "", raw_text).strip()
        if clean_text:
            chunks.append({
                "id": seq,
                "start": round(start, 3),
                "end": round(end, 3),
                "text": clean_text,
            })

    chunks = _remove_duplicates(chunks)

    # Re-number after dedup
    for i, c in enumerate(chunks, 1):
        c["id"] = i
        c["uid"] = f"chunk-{i:04d}"

    return chunks


def load_subtitles(filepath: str) -> List[dict]:
    """Load subtitles from .srt or .vtt file, return normalized chunk list."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Subtitle file not found: {filepath}")

    with open(filepath, "r", encoding="utf-8", errors="replace") as f:
        text = f.read()

    chunks = parse_srt_text(text)
    logger.info("Parsed %d subtitle entries from %s", len(chunks), filepath)
    return chunks
