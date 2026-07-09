"""Global state — single DeepSeek call: episode outline + universe knowledge.

Architecture note — audio position back-references:

Phase 2 produces ``word_ranges`` on each concept/entity/claim/glossary entry
alongside its definition. A deterministic post-process converts word indices
to timestamps via the SRT entries (same logic as ``map_blocks_to_segments``).

The universe state stores ``segments: [{episode, start, end}]`` on each item.
When theme extraction groups concepts into themes, the segments come along as
immutable baggage — no secondary keyword grep or fragile text matching.
"""

import json
import logging
import os
import re
from typing import List, Optional

logger = logging.getLogger(__name__)

_DEFAULT_PROMPT = """You are an expert podcast transcript analyst.

Given the FULL transcript of a single episode of the "Lord of Spirits" podcast
with SRT entry timestamps, your job is to produce **structured knowledge**
(entities, concepts, claims, etc.) — including direct timestamp segments for
audio cutting.

Return ONLY valid JSON — no markdown, no extra text.

Input format — transcript as timestamped SRT entries:

  Episode: "{title}"
  Episode number: {episode}

  [INDEX] START-END: TEXT
  [Index]   00.0-  05.0: Sample entry text

Output format — with timestamp segments:

{
  "summary": "2-3 paragraph narrative summary of the episode's content, themes, and key arguments.",
  "concepts": [
    {
      "id": "kebab-case-id",
      "title": "Concept Name",
      "summary": "Brief explanation",
      "segments": [{"start": 320.0, "end": 394.0}]
    }
  ],
  "entities": [
    {
      "id": "kebab-case-id",
      "title": "Entity Name",
      "category": "person|place|theological|historical|other",
      "summary": "Brief description",
      "segments": [{"start": 600.0, "end": 669.0}]
    }
  ],
  "claims": [
    {
      "id": "kebab-case-id",
      "text": "The claim being made (max 300 chars)",
      "topic": "Theology|Scripture|History|Other",
      "segments": [{"start": 800.0, "end": 950.0}]
    }
  ],
  "scriptural_links": [
    {
      "id": "kebab-case-id",
      "reference": "Book Chapter:Verse",
      "summary": "How it is used",
      "segments": [{"start": 4300.0, "end": 4500.0}]
    }
  ],
  "glossary": [
    {
      "id": "kebab-case-id",
      "term": "Term",
      "definition": "Definition",
      "segments": [{"start": 2000.0, "end": 2100.0}]
    }
  ]
}

SEGMENT RULES (critical for audio cutting):
- "segments" replaces "word_ranges" from earlier versions. Use direct timestamps.
- Each segment = {"start": SECONDS, "end": SECONDS} matching input timestamps.
- CAPTURE COMPLETE THOUGHTS: start where the speaker begins explaining the
  concept, end where that explanation wraps up. Do NOT cut mid-sentence.
- If the concept is mentioned briefly in passing (the speaker is actually
  talking about something else), do NOT include a segment for it — only
  substantive discussion qualifies.
- If the transcript around the concept starts with a sentence fragment
  or a trailing clause (e.g. "because theosis" or "and so"), widen the
  window to where the full sentence began.
- Multiple segments per item allowed (discussed in separate parts of episode).
- Return [] if you cannot confidently identify clean segments."""


def _load_prompt(prompt_path: str = "") -> str:
    if prompt_path and os.path.exists(prompt_path):
        with open(prompt_path, "r", encoding="utf-8") as f:
            return f.read()
    return _DEFAULT_PROMPT


def _format_timestamped_transcript(entries: List[dict]) -> str:
    """Format cleaned subtitle entries as timestamped lines for the LLM prompt.

    Each line: ``[INDEX] START-END: TEXT`` so the LLM can reference exact
    timestamps in its output segments.
    """
    lines = []
    for e in entries:
        lines.append(
            f"[{e['index']:4d}] {e['start']:7.1f}-{e['end']:7.1f}: {e['text']}"
        )
    return "\n".join(lines)


def _validate_segments(
    items: List[dict],
    episode_number: int,
    srt_entries: List[dict],
) -> List[dict]:
    """Validate and snap LLM-returned segments to SRT entry boundaries.

    Mutates items in-place, replacing their ``segments`` with snapped +
    filtered versions.

    Validation steps:
      1. Snap each segment's start/end to the containing SRT entry boundaries.
      2. Drop segments < 3 seconds (hallucination guard).
      3. Log a warning for segments that start mid-sentence (entry text
         begins with a lowercase letter — indicates bad boundary).

    Each mutated segment::

        {"episode": N, "start": 123.4, "end": 567.8}

    Items without segments or with only filtered-out segments get ``[]``.
    """
    for item in items:
        if not isinstance(item, dict):
            continue
        raw_segs = item.pop("segments", [])
        if not raw_segs:
            item["segments"] = []
            continue

        cleaned = []
        for seg in raw_segs:
            if not isinstance(seg, dict):
                continue
            start = seg.get("start", 0)
            end = seg.get("end", 0)
            if end <= start:
                continue

            # Snapping: find SRT entry CONTAINING start, and entry CONTAINING end
            start_entry = None
            for e in srt_entries:
                if e["start"] <= start <= e["end"]:
                    start_entry = e
                    break
            # If no containing entry, find nearest start within 10s
            if start_entry is None:
                for e in srt_entries:
                    if abs(e["start"] - start) < 10.0:
                        start_entry = e
                        break

            end_entry = None
            for e in reversed(srt_entries):
                if e["start"] <= end <= e["end"]:
                    end_entry = e
                    break
            if end_entry is None:
                for e in reversed(srt_entries):
                    if abs(e["end"] - end) < 10.0:
                        end_entry = e
                        break

            if start_entry is None or end_entry is None:
                logger.debug("Segment %.1f-%.1f: no matching entries, dropped", start, end)
                continue

            snapped_start = start_entry["start"]
            snapped_end = end_entry["end"]

            # Drop too-short segments (hallucination guard)
            if snapped_end - snapped_start < 3.0:
                logger.debug("Segment %.1f-%.1f: too short (%.1fs), dropped",
                             snapped_start, snapped_end, snapped_end - snapped_start)
                continue

            # Log warning for mid-sentence boundaries
            first_word = start_entry["text"].strip().split()[0] if start_entry["text"].strip() else ""
            if first_word and first_word[0].islower():
                logger.info("Segment %.1f-%.1f: starts mid-sentence ('%s...') — may need widening",
                            snapped_start, snapped_end, start_entry["text"][:60])

            cleaned.append({
                "episode": episode_number,
                "start": snapped_start,
                "end": snapped_end,
            })

        item["segments"] = cleaned

    return items


def build_global_state(
    transcript_text: str = "",
    *,
    episode_title: str = "",
    episode_number: Optional[int] = None,
    client=None,
    model: str = "deepseek-chat",
    prompt_path: str = "",
    timeout: int = 300,
    srt_entries: Optional[List[dict]] = None,
) -> dict:
    """Single DeepSeek call: timestamped transcript → structured knowledge.

    The LLM receives a timestamped SRT transcript (``[INDEX] START-END: TEXT``)
    and returns direct timestamp segments on each knowledge item. No word
    indices, no post-conversion.

    Args:
        transcript_text: Ignored (kept for backward compatibility).
        episode_title: Episode title for LLM context.
        episode_number: Episode number for provenance tracking.
        client: DeepSeek LLM client.
        model: Model name.
        prompt_path: Path to custom prompt file.
        timeout: LLM request timeout.
        srt_entries: **Required.** Cleaned subtitle entries with timestamps.

    Returns:
        Dict with keys: summary, entities, concepts, claims,
        scriptural_links, glossary. Each knowledge item has a ``segments``
        array snapped to SRT entry boundaries.

    Raises:
        RuntimeError if LLM response is empty or unparseable, or if
        srt_entries is not provided.
    """
    if not srt_entries:
        raise RuntimeError("srt_entries is required — pass cleaned subtitle entries with timestamps")

    prompt_template = _load_prompt(prompt_path)

    # Build timestamped transcript from SRT entries
    timestamped_transcript = _format_timestamped_transcript(srt_entries)

    ep_display = episode_title or f"Episode {episode_number or ''}"

    full_prompt = (
        prompt_template.strip()
        + "\n\n"
        + f"Episode: \"{ep_display}\"\n"
        + f"Episode number: {episode_number or 0}\n\n"
        + "Transcript entries with timestamps:\n"
        + timestamped_transcript
    )

    logger.info(
        "Global state: '%s' (%d entries, %d chars total)",
        ep_display, len(srt_entries), len(full_prompt),
    )

    raw = client.generate(
        prompt=full_prompt, model=model,
        timeout=timeout, temperature=0.1,
        max_tokens=8192, force_json=True,
    )

    data = _parse_json_response(raw)
    if not data:
        raise RuntimeError("Global state: empty or unparseable LLM response")

    result = {
        "summary": data.get("summary", ""),
        "entities": data.get("entities", []),
        "concepts": data.get("concepts", []),
        "claims": data.get("claims", []),
        "scriptural_links": data.get("scriptural_links", []),
        "glossary": data.get("glossary", []),
    }

    # Validate + snap segments for each knowledge category
    ep_num = episode_number or 0
    for category in ("entities", "concepts", "claims", "scriptural_links", "glossary"):
        if result.get(category):
            _validate_segments(result[category], ep_num, srt_entries)
            seg_count = sum(
                len(item.get("segments", [])) for item in result[category]
            )
            if seg_count:
                logger.info("  %s: %d timestamp segments", category, seg_count)

    logger.info(
        "Global state complete: %d entities, %d concepts, %d claims, %d glossary",
        len(result["entities"]),
        len(result["concepts"]),
        len(result["claims"]),
        len(result["glossary"]),
    )
    return result


def map_blocks_to_segments(
    segments: List[dict],
    block_summaries: List[dict],
    transcript_text: str,
) -> dict:
    """Map segmentation segments to topic blocks by word-index overlap.

    Args:
        segments: list of segment dicts with 'text' key.
        block_summaries: list of {block_id, start_word_index, end_word_index, …}.
        transcript_text: the full transcript used for word-offset calculation.

    Returns:
        dict mapping segment_id (or uid) → block_id.
    """
    chunk_word_ranges = []
    offset = 0
    for s in segments:
        wc = len(s["text"].split())
        chunk_word_ranges.append((offset, offset + wc))
        offset += wc

    chunk_to_block = {}
    for block in block_summaries:
        bid = block["block_id"]
        swi = block.get("start_word_index", 0)
        ewi = block.get("end_word_index", 0)
        for seg, (cw_start, cw_end) in zip(segments, chunk_word_ranges):
            if cw_start < ewi and cw_end > swi:
                uid = seg.get("uid", seg["segment_id"])
                chunk_to_block[uid] = bid

    return chunk_to_block


def _parse_json_response(raw: str) -> Optional[dict]:
    """Parse JSON from LLM response, handling fences and common issues."""
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
    if start < 0:
        return None
    end = text.rfind("}")
    if end <= start:
        return None
    candidate = text[start:end + 1]

    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        pass

    repaired = re.sub(r",\s*([}\]])", r"\1", candidate)
    try:
        return json.loads(repaired)
    except json.JSONDecodeError:
        pass

    logger.warning("Failed to parse global state JSON (first 200): %s", candidate[:200])
    return None
