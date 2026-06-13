"""Classification — classify segments using global context.

Phase B: classify each segment (with block summary + global outline context)
Phase C: global cleanup / dedup
"""

import json
import logging
import os
from typing import List, Optional

from podcastcondensor.ollama_client import generate_batch, generate

logger = logging.getLogger(__name__)


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
) -> List[dict]:
    """Classify all segments using global context (block summaries + outline).

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

        payload = json.dumps({
            "chunks": batch_for_model,
            "block_summary": block_summary,
            "global_outline": global_outline,
            "previous_decision": prev_decision,
            "next_chunk_text": next_seg_text,
            "kept_claims_so_far": kept_claims_so_far[-20:],
        }, ensure_ascii=False, indent=2)
        full_prompt = prompt_template.strip() + "\n\n" + payload

        logger.info(
            "Batch %d/%d (%d segments, block %d, %d chars)",
            batch_num, total_batches, len(batch), block_id, len(full_prompt),
        )

        decisions = generate_batch(
            prompt_template=prompt_template,
            chunks=batch,
            model=model,
            host=host,
            timeout=ollama_timeout,
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

    1. Protect opening segments (first ~10%) from dropping.
    2. Deduplicate text-similar kept segments.
    3. Ensure transitions between blocks are preserved.
    """
    seg_id_to_label = {d["id"]: d["label"] for d in decisions}
    seg_id_to_seg = {s["segment_id"]: s for s in segments}

    if not decisions:
        return decisions

    # Pass 1: Protect opening zone
    protect_count = max(3, len(segments) // 10)
    for i in range(min(protect_count, len(segments))):
        sid = segments[i]["segment_id"]
        if sid in seg_id_to_label and seg_id_to_label[sid] == "drop":
            seg_id_to_label[sid] = "keep"

    # Pass 2: Text-similarity dedup
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
                sid_to_label[sid] = "keep"
        except Exception as e:
            logger.warning("Failed to resolve %s, defaulting keep: %s", sid, e)
            sid_to_label[sid] = "keep"

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
