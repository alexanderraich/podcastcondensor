"""Interval building — convert kept chunk decisions to audio cut intervals."""

import logging
from typing import List

logger = logging.getLogger(__name__)


def build_intervals(
    chunks: List[dict],
    decisions: List[dict],
    merge_gap: float = 2.0,
    pad_before: float = 0.35,
    pad_after: float = 0.5,
    audio_duration: float = 0.0,
) -> List[dict]:
    """Build condensed audio intervals from kept chunks.

    Steps:
    1. Collect time ranges for kept chunks.
    2. Merge adjacent ranges where gap <= merge_gap.
    3. Apply padding.
    4. Clamp to audio_duration.
    5. Remove overlaps.

    Returns list of {start, end, kept_uids} dicts.
    """
    uid_to_label = {d["id"]: d["label"] for d in decisions}
    uid_to_chunk = {c["uid"]: c for c in chunks}

    # Step 1: Collect kept intervals
    kept = []
    for chunk in chunks:
        label = uid_to_label.get(chunk["uid"], "drop")
        if label in ("keep",):
            kept.append(chunk)

    if not kept:
        logger.warning("No chunks kept — nothing to condense")
        return []

    # Step 2: Merge adjacent
    merged = [dict(kept[0])]
    for chunk in kept[1:]:
        gap = chunk["start"] - merged[-1]["end"]
        if gap <= merge_gap:
            merged[-1]["end"] = chunk["end"]
            merged[-1]["text"] += " " + chunk["text"]
        else:
            merged.append(dict(chunk))

    # Step 3 & 4 & 5: Apply padding, clamp, and resolve overlaps
    intervals = []
    for seg in merged:
        start = max(0, seg["start"] - pad_before)
        end = seg["end"] + pad_after
        if audio_duration > 0:
            end = min(end, audio_duration)
            start = min(start, audio_duration)

        intervals.append({
            "start": round(start, 3),
            "end": round(end, 3),
            "kept_uids": seg.get("uid", ""),
        })

    # Remove overlaps between consecutive intervals
    cleaned = [intervals[0]]
    for seg in intervals[1:]:
        prev = cleaned[-1]
        if seg["start"] < prev["end"]:
            # Overlap — merge by extending prev
            if seg["end"] > prev["end"]:
                prev["end"] = seg["end"]
        else:
            cleaned.append(seg)

    logger.info(
        "Built %d intervals from %d kept chunks (merged from %d)",
        len(cleaned), len(kept), len(merged),
    )
    return cleaned


def compute_stats(
    chunks: List[dict],
    decisions: List[dict],
    intervals: List[dict],
) -> dict:
    """Compute summary statistics for review."""
    uid_to_label = {d["id"]: d["label"] for d in decisions}
    labels = list(uid_to_label.values())

    keep_count = labels.count("keep")
    drop_count = labels.count("drop")
    maybe_count = labels.count("maybe")

    total_original = sum(c["end"] - c["start"] for c in chunks)
    total_condensed = sum(i["end"] - i["start"] for i in intervals)

    maybe_chunks = [
        {"uid": c["uid"], "text": c["text"][:120],
         "start": c["start"], "end": c["end"]}
        for c in chunks
        if uid_to_label.get(c["uid"]) == "maybe"
    ]

    return {
        "total_chunks": len(chunks),
        "keep_count": keep_count,
        "drop_count": drop_count,
        "maybe_count": maybe_count,
        "original_duration_sec": round(total_original, 1),
        "condensed_duration_sec": round(total_condensed, 1),
        "compression_ratio": (
            round(total_condensed / total_original, 3) if total_original > 0 else 0
        ),
        "maybe_chunks": maybe_chunks,
    }
