"""Interval building — convert kept segment decisions to audio cut intervals."""

import logging
from typing import List

logger = logging.getLogger(__name__)


def build_intervals(
    segments: List[dict],
    decisions: List[dict],
    merge_gap: float = 2.0,
    pad_before: float = 0.35,
    pad_after: float = 0.5,
    audio_duration: float = 0.0,
) -> List[dict]:
    """Build condensed audio intervals from kept segments.

    Steps:
    1. Collect time ranges for kept segments.
    2. Merge adjacent ranges where gap <= merge_gap.
    3. Apply padding.
    4. Clamp to audio_duration.
    5. Remove overlaps.

    Returns list of {start, end, kept_ids} dicts.
    """
    sid_to_label = {d["id"]: d["label"] for d in decisions}
    sid_to_seg = {s["segment_id"]: s for s in segments}

    # Step 1: Collect kept intervals
    kept = []
    for seg in segments:
        label = sid_to_label.get(seg["segment_id"], "drop")
        if label == "keep":
            kept.append(seg)

    if not kept:
        logger.warning("No segments kept — nothing to condense")
        return []

    # Step 2: Merge adjacent
    merged = [dict(kept[0])]
    for seg in kept[1:]:
        gap = seg["start"] - merged[-1]["end"]
        if gap <= merge_gap:
            merged[-1]["end"] = seg["end"]
            merged[-1]["text"] = merged[-1].get("text", "") + " " + seg["text"]
        else:
            merged.append(dict(seg))

    # Step 3-5: Padding + clamp + overlap removal
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
            "kept_ids": seg.get("segment_id", ""),
        })

    # Remove overlaps
    cleaned = [intervals[0]]
    for seg in intervals[1:]:
        prev = cleaned[-1]
        if seg["start"] < prev["end"]:
            if seg["end"] > prev["end"]:
                prev["end"] = seg["end"]
        else:
            cleaned.append(seg)

    logger.info(
        "Built %d intervals from %d kept segments (merged from %d)",
        len(cleaned), len(kept), len(merged),
    )
    return cleaned


def compute_stats(
    segments: List[dict],
    decisions: List[dict],
    intervals: List[dict],
) -> dict:
    """Compute summary statistics for review."""
    sid_to_label = {d["id"]: d["label"] for d in decisions}
    labels = list(sid_to_label.values())

    keep_count = labels.count("keep")
    drop_count = labels.count("drop")
    maybe_count = labels.count("maybe")

    total_original = sum(s["end"] - s["start"] for s in segments)
    total_condensed = sum(i["end"] - i["start"] for i in intervals)

    maybe_segments = [
        {
            "id": s["segment_id"],
            "text": s["text"][:120],
            "start": s["start"],
            "end": s["end"],
        }
        for s in segments
        if sid_to_label.get(s["segment_id"]) == "maybe"
    ]

    return {
        "total_segments": len(segments),
        "keep_count": keep_count,
        "drop_count": drop_count,
        "maybe_count": maybe_count,
        "original_duration_sec": round(total_original, 1),
        "condensed_duration_sec": round(total_condensed, 1),
        "compression_ratio": (
            round(total_condensed / total_original, 3) if total_original > 0 else 0
        ),
        "maybe_segments": maybe_segments,
    }
