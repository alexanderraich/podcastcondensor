"""Chunking — merge small subtitle entries into larger semantic chunks."""

import logging
from typing import List

logger = logging.getLogger(__name__)


def merge_chunks(
    entries: List[dict],
    max_chars: int = 600,
    merge_gap: float = 0.5,
) -> List[dict]:
    """Merge small consecutive subtitle entries into semantic chunks.

    Each entry has: id, uid, start, end, text.

    Merging strategy:
    - Consecutive entries within `merge_gap` seconds get merged if the
      combined text is under `max_chars`.
    - A new chunk starts when gap exceeds threshold or combined text
      would exceed max_chars.
    """
    if not entries:
        return []

    merged = []
    current = dict(entries[0])  # shallow copy

    for entry in entries[1:]:
        gap = entry["start"] - current["end"]
        combined = current["text"] + " " + entry["text"]

        if gap <= merge_gap and len(combined) <= max_chars:
            # Merge
            current["end"] = entry["end"]
            current["text"] = combined
        else:
            # Finalize current, start new
            merged.append(_finalize(current))
            current = dict(entry)

    merged.append(_finalize(current))

    # Re-uid
    for i, c in enumerate(merged, 1):
        c["uid"] = f"chunk-{i:04d}"
        c.pop("id", None)

    logger.info("Merged %d entries into %d chunks", len(entries), len(merged))
    return merged


def _finalize(chunk: dict) -> dict:
    """Clean up a chunk before storing."""
    return {
        "uid": chunk.get("uid", ""),
        "start": round(chunk["start"], 3),
        "end": round(chunk["end"], 3),
        "text": chunk["text"].strip(),
    }
