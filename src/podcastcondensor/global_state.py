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

Given the FULL cleaned transcript of a single episode of the "Lord of Spirits" podcast,
your job is to produce both an **episode outline** (topic blocks + summary) and
**structured knowledge** (entities, concepts, claims, etc.) in a single response.

Return ONLY valid JSON — no markdown, no extra text.

Input:
{
  "episode_title": "...",
  "episode_number": N,
  "transcript": "Full cleaned transcript text..."
}

Output format — with word_ranges added to each knowledge item:

{
  "topic_segments": [
    {
      "segment_id": 1,
      "title": "Short title",
      "start_word_index": 0,
      "end_word_index": 800,
      "summary": "1-3 sentence summary of this topic block"
    }
  ],
  "global_outline": "- Bullet point 1\\n- Bullet point 2\\n- Bullet point 3",
  "summary": "2-3 paragraph narrative summary of the episode's content, themes, and key arguments.",
  "concepts": [
    {"id": "kebab-case-id", "title": "Concept Name", "summary": "Brief explanation",
     "word_ranges": [{"start_word": 300, "end_word": 520}]}
  ],
  "entities": [
    {"id": "kebab-case-id", "title": "Entity Name", "category": "person|place|theological|historical|other",
     "summary": "Brief description",
     "word_ranges": [{"start_word": 100, "end_word": 350}]}
  ],
  "claims": [
    {"id": "kebab-case-id", "text": "The claim being made (max 300 chars)", "topic": "Theology|Scripture|History|Other",
     "word_ranges": [{"start_word": 800, "end_word": 950}]}
  ],
  "scriptural_links": [
    {"id": "kebab-case-id", "reference": "Book Chapter:Verse", "summary": "How it is used in the episode",
     "word_ranges": [{"start_word": 1200, "end_word": 1350}]}
  ],
  "glossary": [
    {"id": "kebab-case-id", "term": "Term", "definition": "Definition",
     "word_ranges": [{"start_word": 2000, "end_word": 2100}]}
  ]
}

IMPORTANT — word_ranges:
For concepts, entities, claims, scriptural_links, and glossary entries,
also include a "word_ranges" field — a list of one or more
{"start_word": N, "end_word": N} pairs indicating WHICH PORTIONS of the
transcript discuss or reference this item. Use the same word-indexing
scheme as topic_segments (0-based word offset from transcript start).

- An item may have multiple word_ranges if it's discussed in several
  separate portions of the episode.
- Ranges should be faithful — only cover the actual discussion, not
  tangential mentions.
- Return an empty array [] if you cannot confidently identify ranges.
- This is CRITICAL for audio extraction — the word_ranges will be
  converted to exact audio timestamps."""


def _load_prompt(prompt_path: str = "") -> str:
    if prompt_path and os.path.exists(prompt_path):
        with open(prompt_path, "r", encoding="utf-8") as f:
            return f.read()
    return _DEFAULT_PROMPT


def convert_word_ranges_to_segments(
    items: List[dict],
    episode_number: int,
    srt_entries: List[dict],
) -> List[dict]:
    """Convert word_ranges on each item to timestamp segments.

    For each item that has a ``word_ranges`` array, computes overlapping
    SRT entries' timestamps and stores them as ``segments`` on the item.

    Each segment dict::

        {"episode": N, "start": 123.4, "end": 567.8,
         "word_start": 300, "word_end": 520}

    Mutates items in-place. Items without word_ranges get an empty list.
    """
    # Compute word range for each SRT entry
    entry_word_ranges = []
    offset = 0
    for entry in srt_entries:
        wc = len(entry["text"].split())
        entry_word_ranges.append((offset, offset + wc))
        offset += wc

    for item in items:
        word_ranges = item.pop("word_ranges", []) if isinstance(item, dict) else []
        if not word_ranges:
            item["segments"] = []
            continue

        timestamp_segments = []
        for wr in word_ranges:
            if not isinstance(wr, dict):
                continue
            swi = wr.get("start_word", 0)
            ewi = wr.get("end_word", 0)
            if ewi <= swi:
                continue

            start_time = None
            end_time = None
            for entry, (cw_start, cw_end) in zip(srt_entries, entry_word_ranges):
                if cw_start < ewi and cw_end > swi:
                    if start_time is None:
                        start_time = entry["start"]
                    end_time = max(end_time or 0, entry["end"])

            if start_time is not None and end_time is not None:
                timestamp_segments.append({
                    "episode": episode_number,
                    "start": round(start_time, 3),
                    "end": round(end_time, 3),
                    "word_start": swi,
                    "word_end": ewi,
                })

        item["segments"] = timestamp_segments

    return items


def build_global_state(
    transcript_text: str,
    *,
    episode_title: str = "",
    episode_number: Optional[int] = None,
    client=None,
    model: str = "deepseek-chat",
    prompt_path: str = "",
    timeout: int = 300,
    srt_entries: Optional[List[dict]] = None,
) -> dict:
    """Single DeepSeek call: full transcript → outline + structured knowledge.

    If ``srt_entries`` is provided, also converts word_ranges on each
    knowledge item to timestamp segments (via ``convert_word_ranges_to_segments``).

    Returns a dict with keys:
        blocks, block_summaries, global_outline, chunk_to_block,
        summary, entities, concepts, claims, scriptural_links, glossary

    Raises RuntimeError if the LLM response is empty or cannot be parsed.
    """
    prompt_template = _load_prompt(prompt_path)
    total_words = len(transcript_text.split())

    payload = json.dumps({
        "episode_title": episode_title,
        "episode_number": episode_number,
        "transcript": transcript_text,
    }, ensure_ascii=False, indent=2)
    full_prompt = prompt_template.strip() + "\n\n" + payload

    logger.info(
        "Global state: '%s' (%d words, %d chars total)",
        episode_title, total_words, len(full_prompt),
    )

    raw = client.generate(
        prompt=full_prompt, model=model,
        timeout=timeout, temperature=0.1,
        max_tokens=8192, force_json=True,
    )

    data = _parse_json_response(raw)
    if not data:
        raise RuntimeError("Global state: empty or unparseable LLM response")

    raw_topic_segs = data.get("topic_segments", [])
    if not raw_topic_segs:
        raise RuntimeError("Global state: no topic_segments in response")

    # Build block_summaries
    block_summaries = []
    for ts in raw_topic_segs:
        block_summaries.append({
            "block_id": ts.get("segment_id", len(block_summaries) + 1),
            "title": ts.get("title", ""),
            "summary": ts.get("summary", ""),
            "word_count": ts.get("end_word_index", 0) - ts.get("start_word_index", 0),
            "start_word_index": ts.get("start_word_index", 0),
            "end_word_index": ts.get("end_word_index", 0),
        })
        logger.info(
            "  Topic %d (%s): %s",
            block_summaries[-1]["block_id"],
            block_summaries[-1]["title"][:40],
            block_summaries[-1]["summary"][:100],
        )

    # Normalise outline
    outline_raw = data.get("global_outline", "")
    if isinstance(outline_raw, list):
        global_outline = "\n".join(f"- {b}" for b in outline_raw)
    else:
        global_outline = outline_raw.strip()
    logger.info("Global outline:\n%s", global_outline[:500])

    result = {
        "blocks": block_summaries,
        "block_summaries": block_summaries,
        "global_outline": global_outline,
        "chunk_to_block": {},
        "summary": data.get("summary", ""),
        "entities": data.get("entities", []),
        "concepts": data.get("concepts", []),
        "claims": data.get("claims", []),
        "scriptural_links": data.get("scriptural_links", []),
        "glossary": data.get("glossary", []),
    }

    # Convert word_ranges to timestamp segments if SRT entries provided
    ep_num = episode_number or 0
    for category in ("entities", "concepts", "claims", "scriptural_links", "glossary"):
        if srt_entries and result.get(category):
            result[category] = convert_word_ranges_to_segments(
                result[category], ep_num, srt_entries,
            )
            seg_count = sum(
                len(item.get("segments", [])) for item in result[category]
            )
            if seg_count:
                logger.info("  %s: %d timestamp segments", category, seg_count)

    logger.info(
        "Global state complete: %d blocks, %d entities, %d concepts, %d claims",
        len(block_summaries),
        len(result["entities"]),
        len(result["concepts"]),
        len(result["claims"]),
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
