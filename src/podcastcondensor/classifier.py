"""Classification — send chunks to local LLM, collect decisions."""

import json
import logging
import os
from typing import List, Optional

from podcastcondensor.ollama_client import generate_batch, generate

logger = logging.getLogger(__name__)


def _load_prompt(path: str) -> str:
    """Load a prompt template from file."""
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def classify_chunks(
    chunks: List[dict],
    model: str,
    prompt_path: str,
    max_chunks_per_batch: int = 20,
    host: str = "http://localhost:11434",
    ollama_timeout: int = 120,
    max_chars_per_chunk: int = 600,
) -> List[dict]:
    """Classify all chunks into keep/drop/maybe.

    Returns list of decisions, one per chunk uid.
    """
    prompt_template = _load_prompt(prompt_path)
    all_decisions = []

    # Split into batches
    for i in range(0, len(chunks), max_chunks_per_batch):
        batch = chunks[i:i + max_chunks_per_batch]
        logger.info(
            "Classifying batch %d/%d (%d chunks)",
            i // max_chunks_per_batch + 1,
            (len(chunks) + max_chunks_per_batch - 1) // max_chunks_per_batch,
            len(batch),
        )

        # Truncate chunk text to max_chars_per_chunk
        for c in batch:
            if len(c["text"]) > max_chars_per_chunk:
                c["text"] = c["text"][:max_chars_per_chunk] + "..."

        decisions = generate_batch(
            prompt_template=prompt_template,
            chunks=batch,
            model=model,
            host=host,
            timeout=ollama_timeout,
        )
        all_decisions.extend(decisions)

    return all_decisions


def resolve_maybe_chunks(
    maybe_chunks: List[dict],
    all_chunks: List[dict],
    all_decisions: List[dict],
    model: str,
    prompt_path: str,
    host: str = "http://localhost:11434",
    ollama_timeout: int = 120,
) -> List[dict]:
    """Second pass: resolve maybe chunks into keep or drop.

    Args:
        maybe_chunks: The chunks that were classified as "maybe"
        all_chunks: All original chunks (for context)
        all_decisions: Original decisions mapping (uid -> label)
        model: Ollama model name
        prompt_path: Path to resolve-maybe prompt template
        host: Ollama host
        ollama_timeout: Request timeout

    Returns:
        Updated list of decisions (all now keep or drop)
    """
    prompt_template = _load_prompt(prompt_path)
    uid_to_label = {d["id"]: d["label"] for d in all_decisions}
    uid_to_chunk = {c["uid"]: c for c in all_chunks}

    for mc in maybe_chunks:
        uid = mc["uid"]
        idx = next(
            (i for i, c in enumerate(all_chunks) if c["uid"] == uid),
            None,
        )
        if idx is None:
            uid_to_label[uid] = "drop"
            continue

        # Find nearest kept chunks for context
        prev_kept = None
        next_kept = None
        for i in range(idx - 1, -1, -1):
            if uid_to_label.get(all_chunks[i]["uid"]) == "keep":
                prev_kept = all_chunks[i]
                break
        for i in range(idx + 1, len(all_chunks)):
            if uid_to_label.get(all_chunks[i]["uid"]) == "keep":
                next_kept = all_chunks[i]
                break

        # Nearby context
        nearby = []
        for j in range(max(0, idx - 3), min(len(all_chunks), idx + 4)):
            if j != idx:
                nearby.append(all_chunks[j]["text"][:200])

        payload = json.dumps({
            "target_chunk": mc,
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
            )
            result = _parse_resolve_response(raw)
            if result and result.get("label") in ("keep", "drop"):
                uid_to_label[uid] = result["label"]
                logger.info("Resolved maybe %s -> %s: %s", uid, result["label"], result.get("reason", ""))
            else:
                uid_to_label[uid] = "keep"  # conservative default on parse failure
        except Exception as e:
            logger.warning("Failed to resolve %s, defaulting to keep: %s", uid, e)
            uid_to_label[uid] = "keep"

    # Rebuild decisions with resolved labels
    result = []
    for d in all_decisions:
        entry = dict(d)
        if entry["label"] == "maybe":
            entry["label"] = uid_to_label.get(entry["id"], "keep")
        result.append(entry)

    return result


def _parse_resolve_response(raw: str) -> Optional[dict]:
    """Parse a resolve-maybe JSON response."""
    text = raw.strip()
    # Strip code fences if present
    if "```" in text:
        parts = text.split("```")
        for p in parts:
            p = p.strip()
            if p.startswith("{") and p.endswith("}"):
                text = p
                break

    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        try:
            return json.loads(text[start:end + 1])
        except json.JSONDecodeError:
            pass
    return None
