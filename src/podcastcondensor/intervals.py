"""Interval building — convert kept segment decisions to audio cut intervals.

No clustering (cluster_gap=0). Kept segments are collected individually,
padded, and overlapping padding is merged. Decisions are treated as final.
"""

import logging
from typing import List, Set

logger = logging.getLogger(__name__)


def build_intervals(
    segments: List[dict],
    decisions: List[dict],
    merge_gap: float = 2.0,
    pad_before: float = 0.35,
    pad_after: float = 0.5,
    audio_duration: float = 0.0,
    cluster_gap: float = 1.5,
) -> List[dict]:
    """Build condensed audio intervals from kept segments.

    Steps:
    1. Collect kept segments and merge into clusters.
    2. Apply padding around each cluster.
    3. Merge overlapping/adjancent clusters.
    4. Clamp to audio_duration.

    Compared to the original implementation, this version applies
    *cluster-aware merging* — short gaps between kept segments are
    merged more aggressively (controlled via ``cluster_gap``) so
    that isolated keep decisions don't produce fragmented audio.

    Returns list of ``{start, end, kept_ids}`` dicts.
    """
    sid_to_label = {d["id"]: d["label"] for d in decisions}

    # Step 1: Collect kept intervals
    kept = []
    for seg in segments:
        label = sid_to_label.get(seg["segment_id"], "drop")
        if label == "keep":
            kept.append(seg)

    if not kept:
        logger.warning("No segments kept — nothing to condense")
        return []

    # Step 2: Cluster-aware merge — merge segments within cluster_gap of each other
    clusters = _cluster_segments(kept, cluster_gap)

    logger.info(
        "Interval clustering: %d segments → %d clusters (gap≤%.1fs)",
        len(kept), len(clusters), cluster_gap,
    )

    # Step 3: Padding + clamp + overlap removal
    intervals = []
    for cl in clusters:
        # Compute cluster span
        cluster_start = min(s["start"] for s in cl)
        cluster_end = max(s["end"] for s in cl)

        start = max(0, cluster_start - pad_before)
        end = cluster_end + pad_after
        if audio_duration > 0:
            end = min(end, audio_duration)
            start = min(start, audio_duration)

        kept_ids = ",".join(s["segment_id"] for s in cl)
        intervals.append({
            "start": round(start, 3),
            "end": round(end, 3),
            "kept_ids": kept_ids,
        })

    # Step 4: Merge overlapping intervals from padding
    cleaned = [intervals[0]]
    for iv in intervals[1:]:
        prev = cleaned[-1]
        if iv["start"] < prev["end"]:
            if iv["end"] > prev["end"]:
                prev["end"] = iv["end"]
            # Merge kept_ids too
            prev["kept_ids"] = _merge_kept_ids(prev.get("kept_ids", ""), iv.get("kept_ids", ""))
        else:
            cleaned.append(iv)

    logger.info(
        "Built %d intervals from %d kept segments in %d clusters",
        len(cleaned), len(kept), len(clusters),
    )
    return cleaned


def compute_stats(
    segments: List[dict],
    decisions: List[dict],
    intervals: List[dict],
) -> dict:
    """Compute summary statistics for review, including fragmentation metrics."""
    sid_to_label = {d["id"]: d["label"] for d in decisions}
    labels = list(sid_to_label.values())

    keep_count = labels.count("keep")
    drop_count = labels.count("drop")
    maybe_count = labels.count("maybe")

    total_original = sum(max(0, s["end"] - s["start"]) for s in segments)
    total_condensed = sum(max(0, i["end"] - i["start"]) for i in intervals)

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

    # Fragmentation analysis
    frag = analyze_fragmentation(segments, decisions, intervals)

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
        "fragmentation": frag,
    }


def analyze_fragmentation(
    segments: List[dict],
    decisions: List[dict],
    intervals: List[dict],
) -> dict:
    """Analyze fragmentation of the keep decisions.

    Returns a dict with:
      - ``num_intervals`` — total intervals after clustering
      - ``num_islands`` — intervals containing exactly 1 kept segment
      - ``island_ratio`` — fraction of intervals that are islands
      - ``avg_segments_per_interval`` — mean cluster size
      - ``keep_density`` — fraction of time-axis where audio is kept
      - ``status`` — ``"ok"``, ``"fragmented"``, or ``"very_fragmented"``
    """
    if not intervals or not segments:
        return {
            "num_intervals": 0,
            "num_islands": 0,
            "island_ratio": 0,
            "avg_segments_per_interval": 0,
            "keep_density": 0,
            "status": "ok",
        }

    # Count kept segments per interval
    total_kept = 0
    islands = 0
    for iv in intervals:
        ids_str = iv.get("kept_ids", "")
        kept_ids = [x for x in ids_str.split(",") if x]
        n = len(kept_ids)
        total_kept += n
        if n == 1:
            islands += 1

    num_intervals = len(intervals)
    island_ratio = islands / num_intervals if num_intervals > 0 else 0
    avg_per_interval = total_kept / num_intervals if num_intervals > 0 else 0

    # Keep density along time axis
    if segments:
        total_duration = max(s["end"] for s in segments)
        kept_duration = sum(iv["end"] - iv["start"] for iv in intervals)
        keep_density = kept_duration / total_duration if total_duration > 0 else 0
    else:
        keep_density = 0

    # Status thresholds
    if island_ratio > 0.5 and num_intervals > 5:
        status = "very_fragmented"
    elif island_ratio > 0.3:
        status = "fragmented"
    else:
        status = "ok"

    return {
        "num_intervals": num_intervals,
        "num_islands": islands,
        "island_ratio": round(island_ratio, 3),
        "avg_segments_per_interval": round(avg_per_interval, 1),
        "keep_density": round(keep_density, 3),
        "status": status,
    }


def check_quality_guardrails(stats: dict, min_keep_ratio: float = 0.20) -> List[str]:
    """Check compression / fragmentation results against quality thresholds.

    Returns a list of warning strings.  Empty list means all checks passed.
    """
    warnings: List[str] = []

    # Compression ratio too low → overly aggressive cutting
    cr = stats.get("compression_ratio", 0)
    if cr < min_keep_ratio:
        warnings.append(
            f"Compression ratio {cr:.1%} is below minimum {min_keep_ratio:.0%} — "
            f"result may be too sparse and choppy"
        )

    # Keep rate too low
    total = stats.get("total_segments", 0)
    kept = stats.get("keep_count", 0)
    if total > 0 and kept / total < min_keep_ratio:
        warnings.append(
            f"Keep rate {kept}/{total} ({kept/max(total,1):.0%}) is very low — "
            f"suspiciously aggressive"
        )

    # Fragmentation
    frag = stats.get("fragmentation", {})
    frag_status = frag.get("status", "ok")
    if frag_status == "very_fragmented":
        warnings.append(
            f"Audio is very fragmented: {frag.get('num_islands', 0)} isolated islands "
            f"out of {frag.get('num_intervals', 0)} intervals "
            f"(island ratio {frag.get('island_ratio', 0):.0%})"
        )
    elif frag_status == "fragmented":
        warnings.append(
            f"Audio is fragmented: island ratio {frag.get('island_ratio', 0):.0%}"
        )

    return warnings


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _cluster_segments(
    kept_segments: List[dict],
    cluster_gap: float,
) -> List[List[dict]]:
    """Group kept segments into clusters where gap between them ≤ *cluster_gap*.

    Unlike the old ``merge_gap`` (which only merged if gap ≤ threshold),
    this clusters segments greedily and always merges within the gap window,
    producing larger, more coherent spans.
    """
    if not kept_segments:
        return []

    clusters: List[List[dict]] = [[kept_segments[0]]]

    for seg in kept_segments[1:]:
        gap = seg["start"] - clusters[-1][-1]["end"]
        if gap <= cluster_gap:
            clusters[-1].append(seg)
        else:
            clusters.append([seg])

    return clusters


def _merge_kept_ids(a: str, b: str) -> str:
    """Merge two comma-separated kept_id strings, deduplicating."""
    seen: Set[str] = set()
    parts: List[str] = []
    for raw in (a, b):
        for item in raw.split(","):
            item = item.strip()
            if item and item not in seen:
                seen.add(item)
                parts.append(item)
    return ",".join(parts)
