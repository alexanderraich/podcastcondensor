"""Subtitle parsing — normalize .srt and .vtt to clean, deduplicated entries."""

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
TIMESTAMP_LEAK_RE = re.compile(
    r"\d+\s+\d{2}:\d{2}:\d{2}[.,]\d{1,3}\s*-->\s*\d{2}:\d{2}:\d{2}[.,]\d{1,3}"
)


def _ts_to_seconds(ts_str: str) -> float:
    """Convert HH:MM:SS.mmm or HH:MM:SS,mmm to seconds."""
    m = TIMESTAMP_RE.match(ts_str.strip())
    if not m:
        raise ValueError(f"Cannot parse timestamp: {ts_str}")
    h, mi, s, ms = int(m.group(1)), int(m.group(2)), int(m.group(3)), int(m.group(4))
    if len(str(ms)) == 2:
        ms = ms * 10
    return h * 3600 + mi * 60 + s + ms / 1000.0


def _normalize_vtt(text: str) -> str:
    """Remove WEBVTT header and metadata lines, return clean SRT-like blocks."""
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
            line = line.replace(".", ",")
        clean.append(line)
    return "\n".join(clean)


def parse_srt_text(text: str) -> List[dict]:
    """Parse SRT text into list of raw entry dicts.

    Each entry: {index, start, end, text}
    Indices are sequential from the SRT file.
    """
    if text.strip().startswith("WEBVTT"):
        text = _normalize_vtt(text)

    entries = []
    for m in SRT_BLOCK_RE.finditer(text):
        seq = int(m.group(1))
        start = _ts_to_seconds(m.group(2))
        end = _ts_to_seconds(m.group(3))
        raw_text = m.group(4).strip().replace("\n", " ").replace("  ", " ")
        clean_text = re.sub(r"<[^>]+>", "", raw_text).strip()
        if clean_text:
            entries.append({
                "index": seq,
                "start": round(start, 3),
                "end": round(end, 3),
                "text": clean_text,
            })

    logger.info("Parsed %d raw entries from SRT", len(entries))
    return entries


# ---------------------------------------------------------------------------
# Cleanup pass (applied after raw parsing, before chunking/segmentation)
# ---------------------------------------------------------------------------

def _strip_timestamp_leak(text: str) -> str:
    """Remove residual SRT timestamp + sequence numbers that leak into text fields."""
    return TIMESTAMP_LEAK_RE.sub("", text).strip()


def _classify_entry(text: str) -> str:
    """Classify a subtitle entry as 'speech', 'music', or 'noise'."""
    t = text.strip().lower()
    if not t:
        return "noise"
    if re.match(r"^\[music\]", t) or re.match(r"^♪", t):
        return "music"
    if len(t) < 3 or re.match(r"^[\s\-_*#]+$", t):
        return "noise"
    return "speech"


def _is_echo(entry: dict, prev_entry: dict) -> bool:
    """True if this entry is a <=50ms echo of the prior entry (auto-caption carryover)."""
    duration = entry["end"] - entry["start"]
    if duration > 0.05:
        return False
    prev_text = prev_entry.get("text", "").lower().strip()
    this_text = entry["text"].lower().strip()
    if not this_text or not prev_text:
        return False
    return this_text in prev_text or prev_text.endswith(this_text)


def _partial_dedup(entries: List[dict]) -> List[dict]:
    """Remove near-duplicate consecutive speech entries using word-overlap."""
    result = []
    for entry in entries:
        if entry["type"] != "speech":
            result.append(entry)
            continue
        if not result:
            result.append(entry)
            continue
        prev = result[-1]
        if prev["type"] != "speech":
            result.append(entry)
            continue

        prev_words = set(prev["text"].lower().split())
        curr_words = set(entry["text"].lower().split())
        if not prev_words or not curr_words:
            result.append(entry)
            continue

        # Measure how much of the current entry's content is NEW
        # (not already present in the previous entry)
        new_words = curr_words - prev_words
        novelty = len(new_words) / len(curr_words) if curr_words else 0

        # If current entry adds almost nothing new, it's a duplicate
        if novelty < 0.25:
            prev_dur = prev["end"] - prev["start"]
            curr_dur = entry["end"] - entry["start"]
            if curr_dur <= prev_dur:
                continue  # drop current (shorter or equal, adds nothing)
            # Current is longer but mostly repeats — keep the longer one
            result[-1] = entry
            continue

        result.append(entry)
    return result


def clean_entries(entries: List[dict]) -> List[dict]:
    """Clean raw subtitle entries: strip artifacts, dedup, remove echoes.

    Each returned entry:
      {index, start, end, text, type}
    where type is "speech", "music", or "noise".
    """
    if not entries:
        return []

    cleaned = []
    echo_removed = 0
    noise_removed = 0
    leak_fixes = 0

    for entry in entries:
        # Step 1: Strip timestamp leaks
        text = _strip_timestamp_leak(entry["text"])
        if text != entry["text"]:
            leak_fixes += 1

        if not text:
            noise_removed += 1
            continue

        entry["text"] = text
        entry["type"] = _classify_entry(text)

        # Step 2: Skip noise (non-speech, non-music)
        if entry["type"] == "noise":
            noise_removed += 1
            continue

        # Step 3: Remove echo entries
        if cleaned and _is_echo(entry, cleaned[-1]):
            echo_removed += 1
            continue

        cleaned.append(dict(entry))

    # Step 4: Partial dedup
    deduped = _partial_dedup(cleaned)

    logger.info(
        "Cleanup: %d leaks fixed, %d echoes removed, %d noise removed, "
        "%d deduped -> %d entries",
        leak_fixes, echo_removed, noise_removed,
        len(cleaned) - len(deduped), len(deduped),
    )

    # Re-index
    for i, entry in enumerate(deduped):
        entry["index"] = i + 1

    return deduped


def load_subtitles(filepath: str) -> List[dict]:
    """Load subtitles from .srt or .vtt file, return CLEANED entries.

    This is the primary entry point. Output is cleaned and deduplicated.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Subtitle file not found: {filepath}")

    with open(filepath, "r", encoding="utf-8", errors="replace") as f:
        text = f.read()

    raw = parse_srt_text(text)
    cleaned = clean_entries(raw)
    logger.info("Loaded %s: %d entries after cleaning", filepath, len(cleaned))
    return cleaned
