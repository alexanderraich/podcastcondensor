"""Global state — single DeepSeek call: episode outline + universe knowledge."""

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

Output format:
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
    {"id": "kebab-case-id", "title": "Concept Name", "summary": "Brief explanation"}
  ],
  "entities": [
    {"id": "kebab-case-id", "title": "Entity Name", "category": "person|place|theological|historical|other", "summary": "Brief description"}
  ],
  "claims": [
    {"id": "kebab-case-id", "text": "The claim being made (max 300 chars)", "topic": "Theology|Scripture|History|Other"}
  ],
  "scriptural_links": [
    {"id": "kebab-case-id", "reference": "Book Chapter:Verse", "summary": "How it is used in the episode"}
  ],
  "glossary": [
    {"id": "kebab-case-id", "term": "Term", "definition": "Definition"}
  ]
}"""


def _load_prompt(prompt_path: str = "") -> str:
    if prompt_path and os.path.exists(prompt_path):
        with open(prompt_path, "r", encoding="utf-8") as f:
            return f.read()
    return _DEFAULT_PROMPT


def build_global_state(
    transcript_text: str,
    *,
    episode_title: str = "",
    episode_number: Optional[int] = None,
    client=None,
    model: str = "deepseek-chat",
    prompt_path: str = "",
    timeout: int = 300,
) -> dict:
    """Single DeepSeek call: full transcript → outline + structured knowledge.

    Returns a dict with keys:
        blocks, block_summaries, global_outline, chunk_to_block (empty dict,
            filled later when segments exist),
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

    # Build block -> chunk mapping placeholder (filled later by map_blocks_to_segments)
    result = {
        "blocks": block_summaries,
        "block_summaries": block_summaries,
        "global_outline": global_outline,
        "chunk_to_block": {},  # filled later
        "summary": data.get("summary", ""),
        "entities": data.get("entities", []),
        "concepts": data.get("concepts", []),
        "claims": data.get("claims", []),
        "scriptural_links": data.get("scriptural_links", []),
        "glossary": data.get("glossary", []),
    }

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
    # Compute word ranges for each segment
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

    # Strip fences
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

    # Repair trailing commas
    repaired = re.sub(r",\s*([}\]])", r"\1", candidate)
    try:
        return json.loads(repaired)
    except json.JSONDecodeError:
        pass

    logger.warning("Failed to parse global state JSON (first 200): %s", candidate[:200])
    return None
