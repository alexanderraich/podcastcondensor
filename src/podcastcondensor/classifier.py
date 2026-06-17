"""Classification — single DeepSeek call for all segments."""

import json
import logging
from typing import List, Optional

logger = logging.getLogger(__name__)


def classify_segments(
    segments: List[dict],
    client,
    global_outline: str,
    block_summaries: List[dict],
    model: str = "deepseek-chat",
    prompt_path: str = "",
    timeout: int = 300,
    universe_state_context: str = "",
) -> List[dict]:
    """Classify all segments in one DeepSeek call.

    Args:
        segments: List of segment dicts with segment_id, block_id, start, end, text, word_count.
        client: DeepSeekClient instance.
        global_outline: Episode outline string.
        block_summaries: List of {block_id, summary, ...}.
        model: DeepSeek model name.
        prompt_path: Path to classification prompt template.
        timeout: API timeout.
        universe_state_context: Optional context from universe state.

    Returns:
        List of {id, label, reason} dicts.
    """
    prompt_template = _load_prompt(prompt_path)

    # Build payload
    batch_for_model = []
    for seg in segments:
        entry = dict(seg)
        entry["id"] = seg["segment_id"]
        batch_for_model.append(entry)

    payload_parts = {
        "chunks": batch_for_model,
        "block_summaries": block_summaries,
        "global_outline": global_outline,
    }
    if universe_state_context:
        payload_parts["universe_state"] = universe_state_context

    payload_json = json.dumps(payload_parts, ensure_ascii=False, indent=2)
    payload = payload_json + '\n\n{"decisions": ['
    full_prompt = prompt_template.strip() + "\n\n" + payload

    logger.info(
        "DeepSeek classification: %d segments, %d blocks, %d chars",
        len(segments), len(block_summaries), len(full_prompt),
    )

    raw = client.generate(
        prompt=full_prompt,
        model=model,
        timeout=timeout,
        temperature=0.1,
        max_tokens=12000,
        force_json=True,
    )

    decisions = _parse_decision_response(raw)
    if not decisions:
        # Retry once
        raw = client.generate(
            prompt=full_prompt,
            model=model,
            timeout=timeout,
            temperature=0.1,
            max_tokens=12000,
            force_json=True,
        )
        decisions = _parse_decision_response(raw)

    if not decisions:
        raise RuntimeError("Classification failed after retry — empty/invalid response")

    logger.info("Classified %d segments (%d keep, %d drop, %d maybe)",
                len(decisions),
                sum(1 for d in decisions if d.get("label") == "keep"),
                sum(1 for d in decisions if d.get("label") == "drop"),
                sum(1 for d in decisions if d.get("label") == "maybe"))

    return decisions


def resolve_maybe(
    maybe_segments: List[dict],
    all_segments: List[dict],
    all_decisions: List[dict],
    client,
    model: str = "deepseek-chat",
    prompt_path: str = "",
    timeout: int = 120,
) -> List[dict]:
    """Resolve maybe segments into keep/drop via a DeepSeek call per segment."""
    prompt_template = _load_prompt(prompt_path)
    sid_to_label = {d["id"]: d["label"] for d in all_decisions}

    failures = 0
    for ms in maybe_segments:
        sid = ms["segment_id"]
        idx = next((i for i, s in enumerate(all_segments) if s["segment_id"] == sid), None)
        if idx is None:
            sid_to_label[sid] = "drop"
            continue

        prev_kept = next_kept = None
        for j in range(idx - 1, -1, -1):
            if sid_to_label.get(all_segments[j]["segment_id"]) == "keep":
                prev_kept = all_segments[j]
                break
        for j in range(idx + 1, len(all_segments)):
            if sid_to_label.get(all_segments[j]["segment_id"]) == "keep":
                next_kept = all_segments[j]
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
            raw = client.generate(
                prompt=full_prompt, model=model,
                timeout=min(timeout, 120), temperature=0.1, force_json=True,
            )
            result = _parse_resolve_response(raw)
            if result and result.get("label") in ("keep", "drop"):
                sid_to_label[sid] = result["label"]
            else:
                sid_to_label[sid] = "drop"
                failures += 1
        except Exception:
            sid_to_label[sid] = "drop"
            failures += 1

    result = []
    for d in all_decisions:
        entry = dict(d)
        if entry["label"] == "maybe":
            entry["label"] = sid_to_label.get(entry["id"], "drop")
        result.append(entry)
    return result


def global_cleanup(
    segments: List[dict],
    decisions: List[dict],
) -> List[dict]:
    """Deduplicate text-similar kept segments."""
    sid_to_label = {d["id"]: d["label"] for d in decisions}

    kept_texts = []
    for decision in decisions:
        sid = decision["id"]
        seg = next((s for s in segments if s["segment_id"] == sid), None)
        if not seg or sid_to_label.get(sid, "drop") != "keep":
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
                    sid_to_label[sid] = "drop"
                    break
        if not is_duplicate:
            kept_texts.append(text)

    result = []
    for d in decisions:
        entry = dict(d)
        new_label = sid_to_label.get(entry["id"])
        if new_label:
            entry["label"] = new_label
        result.append(entry)
    return result


def apply_continuity_bias(
    segments: List[dict],
    decisions: List[dict],
    bridge_gap_sec: float = 1.5,
) -> List[dict]:
    """Bridge dropped segments that sit between two kept segments within gap."""
    sid_to_label = {d["id"]: d["label"] for d in decisions}
    sid_to_reason = {d["id"]: d.get("reason", "") for d in decisions}
    changes = 0

    for i, seg in enumerate(segments):
        sid = seg["segment_id"]
        if sid_to_label.get(sid) != "drop":
            continue

        prev_kept = next_kept = None
        for j in range(i - 1, -1, -1):
            if sid_to_label.get(segments[j]["segment_id"]) == "keep":
                prev_kept = segments[j]
                break
        for j in range(i + 1, len(segments)):
            if sid_to_label.get(segments[j]["segment_id"]) == "keep":
                next_kept = segments[j]
                break

        if prev_kept and next_kept:
            gap_prev = seg["start"] - prev_kept["end"]
            gap_next = next_kept["start"] - seg["end"]
            if gap_prev <= bridge_gap_sec and gap_next <= bridge_gap_sec:
                sid_to_label[sid] = "keep"
                sid_to_reason[sid] = f"bridge (gap {gap_prev:.1f}s/{gap_next:.1f}s)"
                changes += 1

    if changes:
        logger.info("Continuity bias: %d segments bridged", changes)

    result = []
    for d in decisions:
        entry = dict(d)
        new_label = sid_to_label.get(entry["id"])
        if new_label:
            entry["label"] = new_label
        new_reason = sid_to_reason.get(entry["id"])
        if new_reason:
            entry["reason"] = new_reason
        result.append(entry)
    return result


def detect_tail_block(
    segments: List[dict],
    decisions: List[dict],
    tail_fraction: float = 0.12,
    min_keep_fraction: float = 0.03,
) -> List[str]:
    """Detect off-topic trailing content."""
    if not segments or len(segments) < 10:
        return []

    sid_to_label = {d["id"]: d["label"] for d in decisions}
    block_kept_sec = {}
    for seg in segments:
        if sid_to_label.get(seg["segment_id"]) == "keep":
            bid = seg.get("block_id", 0)
            block_kept_sec[bid] = block_kept_sec.get(bid, 0) + (seg["end"] - seg["start"])

    if not block_kept_sec:
        return []

    total_kept = sum(block_kept_sec.values())
    tail_start = int(len(segments) * (1.0 - tail_fraction))
    tail_segs = segments[tail_start:]

    tail_block_counts = {}
    for seg in tail_segs:
        bid = seg.get("block_id", 0)
        tail_block_counts[bid] = tail_block_counts.get(bid, 0) + 1

    if not tail_block_counts:
        return []
    tail_block = max(tail_block_counts, key=tail_block_counts.get)
    tail_kept_sec = block_kept_sec.get(tail_block, 0)

    is_min_content = total_kept > 0 and tail_kept_sec / total_kept < min_keep_fraction
    has_off_topic = _has_off_topic_keywords(" ".join(seg.get("text", "") for seg in tail_segs))
    has_main = any(b != tail_block for b in block_kept_sec)

    if (is_min_content or has_off_topic) and has_main:
        flagged = [seg["segment_id"] for seg in tail_segs if seg.get("block_id", 0) == tail_block]
        if flagged:
            logger.info("Tail detection: %d segments flagged in block %s", len(flagged), tail_block)
        return flagged
    return []


_OFF_TOPIC_KEYWORDS = [
    "subscribe", "patreon", "donate", "donation", "next week", "next time",
    "support us", "check out", "sign up", "merch", "book plug",
    "please rate", "review us", "closing", "outro", "announcements",
    "sponsor", "thanks for listening", "join us next",
]


def _has_off_topic_keywords(text: str) -> bool:
    lower = text.lower()
    return any(kw in lower for kw in _OFF_TOPIC_KEYWORDS)


def _load_prompt(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def _parse_decision_response(raw: str) -> Optional[list]:
    """Extract decisions list from LLM JSON response."""
    if not raw:
        return None
    text = raw.strip()
    # Strip markdown fences
    if "```" in text:
        lines = text.split("\n")
        clean = []
        in_block = False
        for line in lines:
            if line.strip().startswith("```"):
                in_block = not in_block
                continue
            if in_block:
                clean.append(line)
        if clean:
            text = "\n".join(clean).strip()

    for start_char, end_char, parser_fn in [
        ("{", "}", lambda t: _classify_parse_dict(t)),
        ("[", "]", json.loads),
    ]:
        start = text.find(start_char)
        if start >= 0:
            end = text.rfind(end_char)
            if end > start:
                candidate = text[start:end + 1]
                try:
                    result = parser_fn(candidate)
                    if isinstance(result, list):
                        return result
                except json.JSONDecodeError:
                    continue
    return None


def _classify_parse_dict(text: str) -> Optional[list]:
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return None
    if not isinstance(data, dict):
        return None

    decisions = data.get("decisions")
    if isinstance(decisions, list):
        normalized = []
        for item in decisions:
            if not isinstance(item, dict):
                continue
            item["id"] = item.get("id") or item.get("chunk_id") or item.get("segment_id", "?")
            item["label"] = item.get("label") or item.get("classification", "maybe")
            normalized.append(item)
        return normalized if normalized else None

    label = data.get("label") or data.get("classification")
    if label in ("keep", "drop", "maybe"):
        return [{
            "id": data.get("id", data.get("chunk_id", "?")),
            "label": label,
            "reason": data.get("reason", ""),
        }]
    return None


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
