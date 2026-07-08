"""Extract core themes from the accumulated universe state.

One DeepSeek call over the full universe_state.json identifies the ~10–20
core themes that span the entire podcast series, groups related items under
each theme, and assigns importance weights.
"""

import json
import logging
import os
import re
from dataclasses import dataclass, field, asdict
from typing import List, Optional

logger = logging.getLogger(__name__)

PROMPT_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "prompts", "extract_themes.txt",
)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class Theme:
    """A core theme spanning the podcast series."""
    id: str
    title: str
    description: str
    importance: float = 0.5
    related_item_ids: List[str] = field(default_factory=list)
    natural_intro_items: List[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Build the universe-state text for the prompt
# ---------------------------------------------------------------------------


def _build_universe_text(universe_data: dict) -> str:
    """Format universe state as structured text for the LLM prompt.

    Includes episode count, item counts, and per-item summaries with
    frequency hints. Truncates categories to stay within context limits
    (~50k chars max for 64k-token context window).
    """
    meta = universe_data.get("metadata", {})
    total_eps = meta.get("last_built_episode", "?")
    lines = [
        f"Total episodes: {total_eps}",
        "",
    ]

    # Episode summaries (first 15 + last 5 if many, all if few)
    summaries = universe_data.get("episode_summaries", [])
    if summaries:
        lines.append(f"=== Episode summaries ({len(summaries)}) ===")
        if len(summaries) > 20:
            shown = summaries[:15] + summaries[-5:]
        else:
            shown = summaries
        for s in shown:
            ep = s.get("episode_number", "?")
            summary = s.get("summary", "")[:150].replace("\n", " ")
            lines.append(f"  Ep {ep}: {summary}")
        if len(summaries) > 20:
            lines.append(f"  ... ({len(summaries) - 20} more episodes not shown)")
        lines.append("")

    # Structured categories — sorted by frequency (most widespread first),
    # truncated so the total prompt stays manageable.
    category_config = [
        ("concepts", "Concepts", 50),
        ("entities", "Entities", 50),
        ("claims", "Claims", 40),
        ("scriptural_links", "Scriptural references", 30),
        ("glossary", "Glossary terms", 30),
    ]

    for category, label, max_items in category_config:
        items = universe_data.get(category, [])
        if not items:
            continue

        # Sort by frequency descending (items in most episodes first)
        sorted_items = sorted(
            items,
            key=lambda it: len(it.get("episode_numbers", [])),
            reverse=True,
        )

        shown = sorted_items[:max_items]
        truncated = len(sorted_items) - len(shown)

        lines.append(f"=== {label} ({len(items)} total, showing {len(shown)}) ===")
        for item in shown:
            item_id = item.get("id", "?")
            title = (
                item.get("title")
                or item.get("term")
                or item.get("reference")
                or item.get("id", "?")
            )
            summary = (
                item.get("summary")
                or item.get("definition")
                or item.get("text", "")
            )
            eps = item.get("episode_numbers", [])
            freq = len(eps)
            summary_short = summary[:120].replace("\n", " ")
            lines.append(f"  [{item_id}] \"{title}\" (freq={freq})")
            if summary_short and summary_short != title:
                lines.append(f"      {summary_short}")
        if truncated > 0:
            lines.append(f"  ... ({truncated} more items not shown)")
        lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Theme extraction (one DeepSeek call)
# ---------------------------------------------------------------------------


def extract_themes(
    universe_data: dict,
    client=None,
    model: str = "deepseek-chat",
    timeout: int = 600,
    prompt_path: str = "",
) -> List[Theme]:
    """Extract core themes from the accumulated universe state.

    Args:
        universe_data: The full universe_state.json data dict.
        client: DeepSeek LLM client.
        model: Model name.
        timeout: LLM request timeout.
        prompt_path: Path to custom prompt file (empty = default).

    Returns:
        List of Theme dataclass instances, or empty list on failure.
    """
    # Load prompt
    prompt_file = prompt_path if prompt_path else PROMPT_PATH
    try:
        with open(prompt_file, "r", encoding="utf-8") as f:
            base_prompt = f.read().strip()
    except FileNotFoundError:
        logger.error("Theme extraction prompt not found: %s", prompt_file)
        return []

    if not base_prompt:
        logger.error("Empty theme extraction prompt")
        return []

    # Build the formatted universe text
    universe_text = _build_universe_text(universe_data)
    full_prompt = base_prompt + "\n\n" + universe_text

    # Truncate to avoid hitting token limits (model has 16k output, but
    # input should stay reasonable — 12k chars is ~3k tokens, fine)
    # DeepSeek Chat has 64k context, so this is well within limits for
    # the full universe state (~20k chars max).
    logger.info(
        "Extracting themes from universe state (%d chars prompt)",
        len(full_prompt),
    )

    try:
        raw = client.generate(
            prompt=full_prompt,
            model=model,
            timeout=timeout,
            temperature=0.3,
            max_tokens=16000,
            force_json=True,
        )
    except Exception as e:
        logger.error("Theme extraction LLM call failed: %s", e)
        return []

    # Parse response
    themes = _parse_themes_response(raw)
    if not themes:
        logger.warning("Theme extraction returned no themes")
        return []

    logger.info("Extracted %d themes from universe state", len(themes))
    for t in themes:
        logger.debug(
            "  Theme: %s (importance=%.2f, items=%d)",
            t.id, t.importance, len(t.related_item_ids),
        )
    return themes


# ---------------------------------------------------------------------------
# Response parsing
# ---------------------------------------------------------------------------


def _try_parse_json(text: str) -> Optional[dict]:
    """Attempt to parse JSON, trying several repair strategies."""
    # Strategy 1: strict parse
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Strategy 2: trailing comma repair
    repaired = re.sub(r",\s*([}\]])", r"\1", text)
    try:
        return json.loads(repaired)
    except json.JSONDecodeError:
        pass

    return None


def _repair_truncated_json(text: str) -> Optional[str]:
    """Try to repair truncated JSON by closing unclosed brackets and strings.

    The LLM response may get cut off mid-stream, leaving the JSON
    without its closing braces and possibly mid-string. This function
    tracks bracket state (LIFO stack) and appends the necessary closers.
    """
    stack: List[str] = []  # tracks open bracket type
    in_string = False
    escape = False

    for ch in text:
        if escape:
            escape = False
            continue
        if ch == '\\' and in_string:
            escape = True
            continue
        if ch == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch == '{':
            stack.append('}')
        elif ch == '[':
            stack.append(']')
        elif ch == '}':
            if stack and stack[-1] == '}':
                stack.pop()
        elif ch == ']':
            if stack and stack[-1] == ']':
                stack.pop()

    if not stack and not in_string:
        return None  # not truncated

    closers: List[str] = []
    if in_string:
        closers.append('"')  # close the open string
    while stack:
        closers.append(stack.pop())

    return text + ''.join(closers)


def _parse_themes_response(raw: str) -> List[Theme]:
    """Parse JSON from LLM response, return list of Theme objects.

    Handles markdown fences, trailing commas, and truncated JSON
    (LLM response cut off mid-stream without closing braces).
    """
    if not raw:
        return []

    text = raw.strip()

    # Strip markdown code fences
    if "```" in text:
        lines = text.split("\n")
        clean = []
        in_block = False
        for line in lines:
            stripped = line.strip()
            if stripped.startswith("```"):
                in_block = not in_block
                continue
            if in_block:
                clean.append(line)
        if clean:
            text = "\n".join(clean).strip()

    # Find JSON start
    start = text.find("{")
    if start < 0:
        logger.warning("No JSON object found in theme extraction response")
        return []
    candidate = text[start:]

    # Try parsing directly
    data = _try_parse_json(candidate)
    if data is None:
        # Try repairing truncated JSON (missing closing braces)
        repaired = _repair_truncated_json(candidate)
        if repaired and repaired != candidate:
            data = _try_parse_json(repaired)
            if data:
                logger.info("Repaired truncated JSON (added %d closing brackets)",
                           len(repaired) - len(candidate))

    if data is None:
        logger.warning(
            "Failed to parse theme extraction JSON (first 200 chars): %s",
            candidate[:200],
        )
        return []

    if not isinstance(data, dict):
        return []

    raw_themes = data.get("themes", [])
    if not raw_themes:
        return []

    themes = []
    for rt in raw_themes:
        try:
            theme = Theme(
                id=str(rt.get("id", "unknown")),
                title=str(rt.get("title", "Untitled")),
                description=str(rt.get("description", "")),
                importance=float(rt.get("importance", 0.5)),
                related_item_ids=list(rt.get("related_item_ids", [])),
                natural_intro_items=list(rt.get("natural_intro_items", [])),
            )
            themes.append(theme)
        except (ValueError, TypeError, KeyError) as e:
            logger.warning("Skipping malformed theme: %s", e)
            continue

    return themes
