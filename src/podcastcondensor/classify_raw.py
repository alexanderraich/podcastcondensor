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
