"""Classification — classify segments using global context.

Phase B: classify each segment (with block summary + global outline context)
Phase C: global cleanup / dedup
"""

import json
import logging
import os
import re
from typing import List, Optional, Tuple

from podcastcondensor.ollama_client import generate_batch, generate

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Keywords suggestive of off-topic administrative / procedural content
# that typically appears at the tail of episodes.
# ---------------------------------------------------------------------------
_OFF_TOPIC_KEYWORDS = [
    "subscribe", "patreon", "donate", "donation", "next week",
    "next time", "support us", "check out", "sign up",
    "merch", "book plug", "please rate", "review us",
    "closing", "outro", "announcements", "sponsor",
    "thanks for listening", "join us next",
]


def _has_off_topic_keywords(text: str) -> bool:
    """Check whether *text* contains off-topic administrative keywords."""
    lower = text.lower()
    for kw in _OFF_TOPIC_KEYWORDS:
        if kw in lower:
            return True
    return False


def _load_prompt(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


# ---------------------------------------------------------------------------
# Phase B: classify segments with global context
# ---------------------------------------------------------------------------

def classify_segments(
    segments: List[dict],
    model: str,
    prompt_path: str,
    global_outline: str,
    block_summaries: List[dict],
    max_segments_per_batch: int = 3,
    host: str = "http://localhost:11434",
    ollama_timeout: int = 600,
    output_path: Optional[str] = None,
    universe_state_context: str = "",
) -> List[dict]:
    """Classify all segments using global context (block summaries + outline + universe state).

    Each segment dict must have:
      - segment_id, block_id, start, end, text, word_count

    Process is resumable: saves incrementally to output_path.
    """
    prompt_template = _load_prompt(prompt_path)
    all_decisions = []
    kept_claims_so_far: List[str] = []
    start_batch = 0

    # Resume from saved progress
    if output_path and os.path.exists(output_path):
        try:
            with open(output_path) as f:
                all_decisions = json.load(f)
            completed_ids = {d["id"] for d in all_decisions}
            start_batch = len(completed_ids) // max_segments_per_batch
            for d in all_decisions:
                if d.get("label") == "keep" and "reason" in d:
                    kept_claims_so_far.append(d["reason"])
            logger.info(
                "Resuming: %d decisions, %d kept claims, batch %d+",
                len(all_decisions), len(kept_claims_so_far), start_batch + 1,
            )
        except (json.JSONDecodeError, KeyError):
            logger.warning("Corrupted decisions file, starting fresh")
            all_decisions = []

    total_batches = (
        len(segments) + max_segments_per_batch - 1
    ) // max_segments_per_batch

    for i in range(
        start_batch * max_segments_per_batch,
        len(segments),
        max_segments_per_batch,
    ):
        batch_num = i // max_segments_per_batch + 1
        batch = segments[i:i + max_segments_per_batch]

        existing_ids = {d["id"] for d in all_decisions}
        batch_ids = {s["segment_id"] for s in batch}
        if batch_ids.issubset(existing_ids):
            logger.info("Batch %d/%d done, skip", batch_num, total_batches)
            continue

        # Normalize segment dicts for the model: add "id" = segment_id
        # The prompt expects chunks with "id", "text", "start", "end"
        batch_for_model = []
        for s in batch:
            entry = dict(s)
            entry["id"] = s["segment_id"]
            batch_for_model.append(entry)

        # Get block summary for first segment in batch
        first_sid = batch[0]["segment_id"]
        block_id = batch[0]["block_id"]
        block_summary = ""
        for bs in block_summaries:
            if bs["block_id"] == block_id:
                block_summary = bs["summary"]
                break

        # Previous decision
        prev_decision = None
        if all_decisions:
            prev_decision = {
                "id": all_decisions[-1]["id"],
                "label": all_decisions[-1]["label"],
                "reason": all_decisions[-1].get("reason", ""),
            }

        # Next segment text
        next_seg_text = None
        next_idx = i + len(batch)
        if next_idx < len(segments):
            nct = segments[next_idx]["text"]
            next_seg_text = nct[:200] if len(nct) > 200 else nct

        payload_parts = {
            "chunks": batch_for_model,
            "block_summary": block_summary,
            "global_outline": global_outline,
            "previous_decision": prev_decision,
            "next_chunk_text": next_seg_text,
            "kept_claims_so_far": kept_claims_so_far[-20:],
        }
        if universe_state_context:
            payload_parts["universe_state"] = universe_state_context
        payload_json = json.dumps(payload_parts, ensure_ascii=False, indent=2)
        payload = payload_json + '\n\n{"decisions": ['

        prompt_len = len(prompt_template) + len(payload) + 2
        logger.info(
            "Batch %d/%d (%d segments, block %d, %d chars)",
            batch_num, total_batches, len(batch), block_id, prompt_len,
        )

        decisions = generate_batch(
            prompt_template=prompt_template,
            chunks=batch,
            model=model,
            host=host,
            timeout=ollama_timeout,
            retries=2,
            payload_override=payload,
        )
        all_decisions.extend(decisions)

        for d in decisions:
            if d.get("label") == "keep" and "reason" in d:
                kept_claims_so_far.append(d["reason"])

        if output_path:
            _save_decisions(output_path, all_decisions)

    return all_decisions


# ---------------------------------------------------------------------------
# Phase C: global cleanup / deduplication
# ---------------------------------------------------------------------------

def global_cleanup(
    segments: List[dict],
    decisions: List[dict],
) -> List[dict]:
    """Global deduplication and cleanup pass on segment decisions.

    Deduplicates text-similar kept segments.  No longer protects opening
    segments — that was overriding the classifier's judgment on intro/setup
    material that should be evaluated by the same criteria as everything
    else.
    """
    seg_id_to_label = {d["id"]: d["label"] for d in decisions}
    seg_id_to_seg = {s["segment_id"]: s for s in segments}

    if not decisions:
        return decisions

    # Pass 1: Text-similarity dedup
    kept_texts = []
    for decision in decisions:
        sid = decision["id"]
        seg = seg_id_to_seg.get(sid)
        if not seg:
            continue
        label = seg_id_to_label.get(sid, "drop")
        if label != "keep":
            continue

        text = seg["text"].lower().strip()
        is_duplicate = False
        for prev_text in kept_texts:
            words = set(text.split())
            prev_words = set(prev_text.split())
            if len(words) > 5 and len(prev_words) > 5:
                overlap = len(words & prev_words) / min(len(words), len(prev_words))
                if overlap > 0.7:
                    is_duplicate = True
                    logger.info("Dedup dropping seg %s (%.0f%% overlap)", sid, overlap * 100)
                    seg_id_to_label[sid] = "drop"
                    break
        if not is_duplicate:
            kept_texts.append(text)

    # Rebuild
    result = []
    for d in decisions:
        entry = dict(d)
        new_label = seg_id_to_label.get(entry["id"])
        if new_label:
            entry["label"] = new_label
        result.append(entry)

    changed = sum(
        1 for d in result
        if d["label"] != next(
            (od["label"] for od in decisions if od["id"] == d["id"]),
            d["label"],
        )
    )
    if changed:
        logger.info("Global cleanup: %d segments changed", changed)

    return result


# ---------------------------------------------------------------------------
# Maybe resolution
# ---------------------------------------------------------------------------

def resolve_maybe(
    maybe_segments: List[dict],
    all_segments: List[dict],
    all_decisions: List[dict],
    model: str,
    prompt_path: str,
    host: str = "http://localhost:11434",
    ollama_timeout: int = 120,
) -> List[dict]:
    """Resolve maybe segments into keep or drop using contextual LLM pass."""
    prompt_template = _load_prompt(prompt_path)
    sid_to_label = {d["id"]: d["label"] for d in all_decisions}
    sid_to_seg = {s["segment_id"]: s for s in all_segments}

    for ms in maybe_segments:
        sid = ms["segment_id"]
        idx = next(
            (i for i, s in enumerate(all_segments) if s["segment_id"] == sid), None,
        )
        if idx is None:
            sid_to_label[sid] = "drop"
            continue

        prev_kept = next_kept = None
        for i in range(idx - 1, -1, -1):
            if sid_to_label.get(all_segments[i]["segment_id"]) == "keep":
                prev_kept = all_segments[i]
                break
        for i in range(idx + 1, len(all_segments)):
            if sid_to_label.get(all_segments[i]["segment_id"]) == "keep":
                next_kept = all_segments[i]
                break

        nearby = []
        for j in range(max(0, idx - 3), min(len(all_segments), idx + 4)):
            if j != idx:
                nearby.append(all_segments[j]["text"][:200])

        payload = json.dumps({
            "target_chunk": ms,
            "previous_kept_chunk": prev_kept,
            "next_kept_chunk": next_kept,
            "nearby_context": "\n".join(nearby),
        }, ensure_ascii=False, indent=2)
        full_prompt = prompt_template.strip() + "\n\n" + payload

        try:
            raw = generate(
                prompt=full_prompt,
                model=model,
                host=host,
                timeout=ollama_timeout,
                temperature=0.1,
                force_json=True,
            )
            result = _parse_resolve_response(raw)
            if result and result.get("label") in ("keep", "drop"):
                sid_to_label[sid] = result["label"]
                logger.info("Resolved maybe %s -> %s", sid, result["label"])
            else:
                logger.info("Resolve returned no valid label for %s, defaulting drop", sid)
                sid_to_label[sid] = "drop"
        except Exception as e:
            logger.warning("Failed to resolve %s, defaulting drop: %s", sid, e)
            sid_to_label[sid] = "drop"

    result = []
    for d in all_decisions:
        entry = dict(d)
        if entry["label"] == "maybe":
            entry["label"] = sid_to_label.get(entry["id"], "keep")
        result.append(entry)
    return result


def _save_decisions(path: str, decisions: list):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(decisions, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)


def _parse_resolve_response(raw: str) -> Optional[dict]:
    text = raw.strip()
    if "```" in text:
        for part in text.split("```"):
            part = part.strip()
            if part.startswith("{"):
                text = part
                break
    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        try:
            return json.loads(text[start:end + 1])
        except json.JSONDecodeError:
            pass
    return None


# ---------------------------------------------------------------------------
# Listenability helpers — called after classification & cleanup
# ---------------------------------------------------------------------------


def apply_continuity_bias(
    segments: List[dict],
    decisions: List[dict],
    bridge_gap_sec: float = 3.0,
    min_cluster_size: int = 1,
) -> List[dict]:
    """Add bridging and context padding to reduce fragmentation.

    1. **Bridge pass** — if a dropped segment sits *between* two kept
       segments with only a small gap on each side, keep it as a bridge.

    2. **Context pass** — isolated kept segments without nearby neighbours
       get their immediate predecessor / successor kept for context.

    3. **Cluster pass** — any dropped segment that directly touches a kept
       segment (adjacent in segment sequence) and is very short (<50 words)
       gets kept as minimal context.

    Returns an updated decisions list.
    """
    sid_to_label = {d["id"]: d["label"] for d in decisions}
    sid_to_seg = {s["segment_id"]: s for s in segments}

    changes = 0

    # --- Pass 1: Bridge dropped segments that connect two kept segments ---
    for i, seg in enumerate(segments):
        sid = seg["segment_id"]
        if sid_to_label.get(sid) != "drop":
            continue

        prev_kept_idx = None
        next_kept_idx = None
        for j in range(i - 1, -1, -1):
            if sid_to_label.get(segments[j]["segment_id"]) == "keep":
                prev_kept_idx = j
                break
        for j in range(i + 1, len(segments)):
            if sid_to_label.get(segments[j]["segment_id"]) == "keep":
                next_kept_idx = j
                break

        if prev_kept_idx is not None and next_kept_idx is not None:
            gap_prev = seg["start"] - segments[prev_kept_idx]["end"]
            gap_next = segments[next_kept_idx]["start"] - seg["end"]
            if gap_prev <= bridge_gap_sec and gap_next <= bridge_gap_sec:
                sid_to_label[sid] = "keep"
                changes += 1
                logger.debug("Bridge keep: %s (gap %.1f/%.1f)", sid, gap_prev, gap_next)

    # --- Pass 2: Context padding around isolated kept segments ---
    for i, seg in enumerate(segments):
        sid = seg["segment_id"]
        if sid_to_label.get(sid) != "keep":
            continue

        # Check if this kept segment has kept neighbours nearby
        has_kept_before = any(
            sid_to_label.get(segments[j]["segment_id"]) == "keep"
            for j in range(max(0, i - 3), i)
        )
        has_kept_after = any(
            sid_to_label.get(segments[j]["segment_id"]) == "keep"
            for j in range(i + 1, min(len(segments), i + 4))
        )

        if not has_kept_before and i > 0:
            prev_sid = segments[i - 1]["segment_id"]
            if sid_to_label.get(prev_sid) == "drop":
                sid_to_label[prev_sid] = "keep"
                changes += 1
                logger.debug("Context before isolated %s: kept %s", sid, prev_sid)

        if not has_kept_after and i < len(segments) - 1:
            next_sid = segments[i + 1]["segment_id"]
            if sid_to_label.get(next_sid) == "drop":
                sid_to_label[next_sid] = "keep"
                changes += 1
                logger.debug("Context after isolated %s: kept %s", sid, next_sid)

    # --- Pass 3: Keep very short adjacent neighbours of kept segments ---
    for i, seg in enumerate(segments):
        sid = seg["segment_id"]
        if sid_to_label.get(sid) != "keep":
            continue
        # Check neighbours (one on each side) — keep if very short
        for neighbour_idx in (i - 1, i + 1):
            if neighbour_idx < 0 or neighbour_idx >= len(segments):
                continue
            n_sid = segments[neighbour_idx]["segment_id"]
            if sid_to_label.get(n_sid) == "drop":
                n_words = segments[neighbour_idx].get("word_count", 0)
                if n_words < 30:
                    sid_to_label[n_sid] = "keep"
                    changes += 1
                    logger.debug("Short neighbour keep: %s (%d words)", n_sid, n_words)

    if changes:
        logger.info("Continuity bias: %d segments promoted to keep", changes)

    # Rebuild decisions list
    result = []
    for d in decisions:
        entry = dict(d)
        new_label = sid_to_label.get(entry["id"])
        if new_label:
            entry["label"] = new_label
        result.append(entry)
    return result


def detect_tail_block(
    segments: List[dict],
    decisions: List[dict],
    tail_fraction: float = 0.12,
    min_keep_fraction: float = 0.03,
) -> List[str]:
    """Detect off-topic trailing material and return IDs to force-drop.

    Uses three signals:

    1. **Block identity** — if the last N segments are in a different block
       from the main discussion, flag them.

    2. **Keyword match** — if the tail contains administrative/off-topic
       keywords, flag with higher confidence.

    3. **Min-content heuristic** — if the last block has very little kept
       content (< ``min_keep_fraction`` of total), it's likely off-topic.

    Returns a set of ``segment_id`` strings to force-drop.
    """
    if not segments or len(segments) < 10:
        return []

    sid_to_label = {d["id"]: d["label"] for d in decisions}

    # Measure kept content per block
    block_kept_sec: dict = {}
    for seg in segments:
        if sid_to_label.get(seg["segment_id"]) == "keep":
            bid = seg.get("block_id", 0)
            block_kept_sec[bid] = block_kept_sec.get(bid, 0) + (seg["end"] - seg["start"])

    if not block_kept_sec:
        return []

    total_kept = sum(block_kept_sec.values())

    # Find segments in the tail region (last tail_fraction)
    tail_start = int(len(segments) * (1.0 - tail_fraction))
    tail_segs = segments[tail_start:]
    if not tail_segs:
        return []

    # Determine dominant block of the tail
    tail_block_counts: dict = {}
    for seg in tail_segs:
        bid = seg.get("block_id", 0)
        tail_block_counts[bid] = tail_block_counts.get(bid, 0) + 1

    if not tail_block_counts:
        return []

    tail_block = max(tail_block_counts, key=tail_block_counts.get)

    # Heuristic 1: tail block has very little kept content
    tail_kept_sec = block_kept_sec.get(tail_block, 0)
    is_min_content_tail = (
        total_kept > 0 and tail_kept_sec / total_kept < min_keep_fraction
    )

    # Heuristic 2: keyword check
    # Check if ANY of the tail segments have off-topic keywords
    has_off_topic_kw = any(
        _has_off_topic_keywords(seg.get("text", ""))
        for seg in tail_segs
    )

    # Heuristic 3: there exists a main block before the tail block
    main_blocks = [b for b in block_kept_sec if b != tail_block]
    has_main_content_before = bool(main_blocks)

    if (is_min_content_tail or has_off_topic_kw) and has_main_content_before:
        # Flag ALL segments in the tail region that belong to the tail block
        flagged = [
            seg["segment_id"]
            for seg in tail_segs
            if seg.get("block_id", 0) == tail_block
        ]
        if flagged:
            logger.info(
                "Tail detection: %d segments flagged in block %s "
                "(tail_kept=%.0fs/%.0fs total, kw=%s)",
                len(flagged), tail_block, tail_kept_sec, total_kept, has_off_topic_kw,
            )
        return flagged

    return []
