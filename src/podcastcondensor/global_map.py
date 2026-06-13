"""Global episode map — hierarchical transcript analysis.

Phase A of the condensation pipeline:
1. Split transcript into large semantic blocks (~500-1000 words)
2. Summarize each block via LLM
3. Synthesize a global episode outline from block summaries
"""

import json
import logging
import os
from typing import List, Optional

from podcastcondensor.ollama_client import generate

logger = logging.getLogger(__name__)

# Estimated average English word length in chars
_WORDS_PER_CHUNK_ESTIMATE = 12


def _load_prompt(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def split_into_blocks(
    chunks: List[dict],
    block_size_words: int = 800,
) -> List[dict]:
    """Split normalized chunks into semantic blocks of roughly block_size_words.

    Each block preserves chunk boundaries (no mid-chunk splits).

    Returns list of blocks, each with:
      - block_id: int
      - chunks: list of chunk dicts
      - word_count: int
      - start_time: float (first chunk start)
      - end_time: float (last chunk end)
    """
    blocks = []
    current = []
    current_words = 0

    for chunk in chunks:
        chunk_words = max(1, len(chunk["text"]) // _WORDS_PER_CHUNK_ESTIMATE)

        # Start new block if current is non-empty AND adding this chunk
        # would exceed block_size AND we have at least 1 chunk
        if current and current_words + chunk_words > block_size_words:
            blocks.append(_seal_block(current, len(blocks) + 1))
            current = []
            current_words = 0

        current.append(chunk)
        current_words += chunk_words

    if current:
        blocks.append(_seal_block(current, len(blocks) + 1))

    logger.info(
        "Split %d chunks into %d blocks (target %d words/block)",
        len(chunks), len(blocks), block_size_words,
    )
    return blocks


def _seal_block(chunks: list, block_id: int) -> dict:
    texts = [c["text"] for c in chunks]
    return {
        "block_id": block_id,
        "chunks": chunks,
        "chunk_ids": [c["uid"] for c in chunks],
        "word_count": sum(max(1, len(c["text"]) // _WORDS_PER_CHUNK_ESTIMATE) for c in chunks),
        "start_time": chunks[0]["start"],
        "end_time": chunks[-1]["end"],
        "text": " ".join(texts),
    }


def summarize_block(
    block: dict,
    model: str,
    prompt_path: str,
    host: str = "http://localhost:11434",
    timeout: int = 300,
) -> str:
    """Send a block transcript to the LLM and get a summary.

    Returns summary string (1-3 sentences).
    """
    prompt_template = _load_prompt(prompt_path)
    payload = json.dumps({
        "block_id": block["block_id"],
        "transcript": block["text"],
        "word_count": block["word_count"],
        "time_range": f"{block['start_time']:.0f}s - {block['end_time']:.0f}s",
    }, ensure_ascii=False, indent=2)
    full_prompt = prompt_template.strip() + "\n\n" + payload

    raw = generate(
        prompt=full_prompt,
        model=model,
        host=host,
        timeout=timeout,
        temperature=0.1,
        max_tokens=512,
        force_json=True,
    )

    summary = _parse_summary_response(raw, block["block_id"])
    return summary


def _parse_summary_response(raw: str, block_id: int) -> str:
    """Parse block summary JSON response, return summary text or fallback."""
    try:
        data = json.loads(raw)
        if isinstance(data, dict):
            return data.get("summary", str(data))
        return str(data)
    except json.JSONDecodeError:
        # Try to extract from code fences
        if "```" in raw:
            for part in raw.split("```"):
                part = part.strip()
                if part.startswith("{"):
                    try:
                        data = json.loads(part)
                        return data.get("summary", str(data))
                    except json.JSONDecodeError:
                        pass
        # Fallback: return raw text truncated
        logger.warning("Could not parse summary for block %d, using raw text", block_id)
        return raw.strip()[:500]


def synthesize_outline(
    block_summaries: List[dict],
    model: str,
    prompt_path: str,
    host: str = "http://localhost:11434",
    timeout: int = 300,
) -> str:
    """Synthesize a global episode outline from all block summaries.

    Returns outline text (5-15 bullet points).
    """
    prompt_template = _load_prompt(prompt_path)
    payload = json.dumps({
        "block_summaries": block_summaries,
        "total_blocks": len(block_summaries),
    }, ensure_ascii=False, indent=2)
    full_prompt = prompt_template.strip() + "\n\n" + payload

    raw = generate(
        prompt=full_prompt,
        model=model,
        host=host,
        timeout=timeout,
        temperature=0.1,
        max_tokens=1024,
        force_json=True,
    )

    outline = _parse_outline_response(raw)
    return outline


def _parse_outline_response(raw: str) -> str:
    """Parse outline JSON response, return outline text."""
    try:
        data = json.loads(raw)
        if isinstance(data, dict):
            bullets = data.get("outline", data.get("points", []))
            if isinstance(bullets, list):
                return "\n".join(f"- {b}" for b in bullets)
            return str(bullets)
        return str(data)
    except json.JSONDecodeError:
        if "```" in raw:
            for part in raw.split("```"):
                part = part.strip()
                if part.startswith("{"):
                    try:
                        data = json.loads(part)
                        bullets = data.get("outline", data.get("points", []))
                        if isinstance(bullets, list):
                            return "\n".join(f"- {b}" for b in bullets)
                        return str(data)
                    except json.JSONDecodeError:
                        pass
        # Fallback: extract bullet points from raw text
        lines = [l.strip().lstrip("- ") for l in raw.split("\n") if l.strip().startswith("-")]
        if lines:
            return "\n".join(f"- {l}" for l in lines)
        return raw.strip()[:1000]


def build_global_map(
    chunks: List[dict],
    model: str,
    block_prompt_path: str,
    outline_prompt_path: str,
    block_size_words: int = 800,
    host: str = "http://localhost:11434",
    timeout: int = 300,
) -> dict:
    """Full Phase A: split, summarize, outline.

    Returns:
        blocks: list of block dicts with summaries
        block_summaries: list of {block_id, summary, start, end}
        global_outline: string of bullet points
        chunk_to_block: dict mapping chunk uid -> block_id
    """
    blocks = split_into_blocks(chunks, block_size_words)

    block_summaries = []
    for block in blocks:
        logger.info(
            "Summarizing block %d/%d (%d words, %.0fs-%.0fs)",
            block["block_id"], len(blocks),
            block["word_count"], block["start_time"], block["end_time"],
        )
        summary = summarize_block(block, model, block_prompt_path, host, timeout)
        block_summaries.append({
            "block_id": block["block_id"],
            "summary": summary,
            "start_time": block["start_time"],
            "end_time": block["end_time"],
            "word_count": block["word_count"],
        })
        logger.info("  Summary: %s", summary[:120])

    logger.info(
        "Synthesizing global outline from %d block summaries...",
        len(block_summaries),
    )
    global_outline = synthesize_outline(
        block_summaries, model, outline_prompt_path, host, timeout,
    )
    logger.info("Global outline:\n%s", global_outline[:500])

    # Map chunk uid -> block_id
    chunk_to_block = {}
    for block in blocks:
        for cid in block["chunk_ids"]:
            chunk_to_block[cid] = block["block_id"]

    return {
        "blocks": blocks,
        "block_summaries": block_summaries,
        "global_outline": global_outline,
        "chunk_to_block": chunk_to_block,
    }
