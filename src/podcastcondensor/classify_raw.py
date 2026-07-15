"""Raw SRT classifier — one DeepSeek call, per-entry decisions."""

import json
import logging
import os

logger = logging.getLogger(__name__)

_CLASSIFY_PROMPT = ""


def _load_prompt(path):
    global _CLASSIFY_PROMPT
    if not _CLASSIFY_PROMPT:
        with open(path, "r", encoding="utf-8") as f:
            _CLASSIFY_PROMPT = f.read()
        logger.info("Loaded classify prompt from %s (%d chars)", path, len(_CLASSIFY_PROMPT))
    return _CLASSIFY_PROMPT


def classify_raw(srt_path, client, global_outline, universe_state_context="",
                 model="deepseek-chat", timeout=600, prompt_path=""):
    prompt_template = _load_prompt(prompt_path)

    with open(srt_path, "r", encoding="utf-8") as f:
        raw_srt = f.read()

    payload = json.dumps({
        "episode_outline": global_outline,
        "universe_state": universe_state_context if universe_state_context else None,
        "srt": raw_srt,
    }, ensure_ascii=False, indent=2)
    full_prompt = prompt_template.strip() + "\n\n" + payload

    logger.info("Raw classifier: %d chars in prompt", len(full_prompt))

    raw = client.generate(
        prompt=full_prompt, model=model,
        timeout=timeout, temperature=0.1,
        max_tokens=16000, force_json=True,
    )

    decisions = _parse_json(raw)
    if not decisions:
        logger.info("Retrying raw classifier...")
        raw = client.generate(
            prompt=full_prompt, model=model,
            timeout=timeout, temperature=0.1,
            max_tokens=16000, force_json=True,
        )
        decisions = _parse_json(raw)

    if not decisions:
        raise RuntimeError("Raw classifier returned no decisions after retry")

    logger.info("Classifier returned %d decisions", len(decisions))
    return decisions


# ---------------------------------------------------------------------------
# New: one-shot compression — direct timestamp segments
# ---------------------------------------------------------------------------

_COMPRESS_PROMPT = ""


def _load_compress_prompt(prompt_path: str) -> str:
    """Load and cache the compression prompt."""
    global _COMPRESS_PROMPT
    if not _COMPRESS_PROMPT:
        with open(prompt_path, "r", encoding="utf-8") as f:
            _COMPRESS_PROMPT = f.read()
        logger.info("Loaded compress prompt from %s (%d chars)", prompt_path, len(_COMPRESS_PROMPT))
    return _COMPRESS_PROMPT


def _format_timestamped_transcript(entries: list) -> str:
    """Format subtitle entries as ``[INDEX] START-END: TEXT`` lines."""
    lines = []
    for e in entries:
        lines.append(
            f"[{e['index']:4d}] {e['start']:7.1f}-{e['end']:7.1f}: {e['text']}"
        )
    return "\n".join(lines)


def _snap_segment(seg: dict, srt_entries: list):
    """Snap a segment's start/end to SRT entry boundaries.

    Returns None if the segment can't be mapped to SRT entries or is
    too short to be valid (hallucination guard). Warnings are logged
    for segments that start mid-sentence or are on the short side.
    The segment is mutated in-place and also returned for convenience.
    """
    start = seg.get("start", 0)
    end = seg.get("end", 0)
    if end <= start:
        logger.warning("Segment %.1f-%.1f: end <= start, dropped", start, end)
        return None

    # Find containing SRT entry for start
    start_entry = None
    for e in srt_entries:
        if e["start"] <= start <= e["end"]:
            start_entry = e
            break
    if start_entry is None:
        # Try nearest within 15s
        for e in srt_entries:
            if abs(e["start"] - start) < 15.0:
                start_entry = e
                break
    if start_entry is None:
        logger.warning("Segment %.1f-%.1f: start not found in any SRT entry, dropped", start, end)
        return None

    # Find containing SRT entry for end
    end_entry = None
    for e in reversed(srt_entries):
        if e["start"] <= end <= e["end"]:
            end_entry = e
            break
    if end_entry is None:
        for e in reversed(srt_entries):
            if abs(e["end"] - end) < 15.0:
                end_entry = e
                break
    if end_entry is None:
        logger.warning("Segment %.1f-%.1f: end not found in any SRT entry, dropped", start, end)
        return None

    snapped_start = start_entry["start"]
    snapped_end = end_entry["end"]

    # Hallucination guard: drop segments under 3s
    duration = snapped_end - snapped_start
    if duration < 3.0:
        logger.warning("Segment %.1f-%.1f: too short (%.1fs), dropped", snapped_start, snapped_end, duration)
        return None

    # Warn on mid-sentence boundary at start
    first_word = start_entry["text"].strip().split()[0] if start_entry["text"].strip() else ""
    if first_word and first_word[0].islower():
        logger.warning(
            "Segment %.1f-%.1f: starts mid-sentence ('%s...') — may need widening",
            snapped_start, snapped_end, start_entry["text"][:80],
        )

    seg["start"] = snapped_start
    seg["end"] = snapped_end
    return seg


def compress_episode(
    srt_entries: list,
    *,
    episode_title: str = "",
    episode_number: int = 0,
    client=None,
    model: str = "deepseek-chat",
    timeout: int = 600,
    prompt_path: str = "",
) -> dict:
    """Single LLM call: timestamped transcript → core idea + 1-2 segments.

    Unlike the old two-call pipeline (global_state + classify_raw), this
    is a single shot that returns direct timestamp segments for audio
    cutting. No cue numbers, no word indices, no universe state.

    Args:
        srt_entries: Cleaned subtitle entries with index, start, end, text.
        episode_title: Title for LLM context.
        episode_number: Episode number for context.
        client: DeepSeek LLM client.
        model: Model name.
        timeout: LLM request timeout.
        prompt_path: Path to compress prompt file.

    Returns:
        Dict with:
          - ``core_idea``: one-sentence summary of the episode's main argument
          - ``segments``: list of ``{"start": float, "end": float, "reason": str}``
            snapped to SRT entry boundaries
    """
    prompt_template = _load_compress_prompt(prompt_path)
    timestamped = _format_timestamped_transcript(srt_entries)

    ep_display = episode_title or f"Episode {episode_number}"
    full_prompt = (
        prompt_template.strip()
        + "\n\n"
        + f"Episode: \"{ep_display}\"\n"
        + f"Episode number: {episode_number or 0}\n\n"
        + "Transcript entries with timestamps:\n"
        + timestamped
    )

    logger.info(
        "Compress episode: '%s' (%d entries, %d chars)",
        ep_display, len(srt_entries), len(full_prompt),
    )

    raw = client.generate(
        prompt=full_prompt, model=model,
        timeout=timeout, temperature=0.1,
        max_tokens=8192, force_json=True,
    )

    data = _parse_json(raw)
    if not data:
        logger.warning("Compress: LLM returned empty/unparseable — falling back to empty result")
        return {"core_idea": "", "segments": []}

    core_idea = data.get("core_idea", "")
    raw_segments = data.get("segments", [])

    if not isinstance(raw_segments, list):
        raw_segments = []

    # Validate and snap each segment
    validated = []
    for seg in raw_segments:
        if not isinstance(seg, dict):
            continue
        result = _snap_segment(seg, srt_entries)
        if result is not None:
            validated.append({
                "start": result["start"],
                "end": result["end"],
                "reason": seg.get("reason", ""),
            })

    logger.info(
        "Compress complete: core_idea='%s', %d segment(s) (%.0fs total)",
        core_idea[:80] if core_idea else "(none)",
        len(validated),
        sum(s["end"] - s["start"] for s in validated),
    )
    return {"core_idea": core_idea, "segments": validated}


def _parse_json(raw):
    if not raw:
        return None
    text = raw.strip()
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
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end <= start:
        return None
    try:
        return json.loads(text[start:end + 1])
    except json.JSONDecodeError:
        return None
