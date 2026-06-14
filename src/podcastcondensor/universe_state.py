"""Cross-episode knowledge base for podcastcondensor.

Tracks entities, concepts, claims, scriptural links, and glossary terms
established across episodes so the classifier can drop re-explanations
and focus on new content.
"""

import json
import logging
import os
from datetime import datetime
from typing import Any, Dict, List, Optional

from podcastcondensor.ollama_client import generate

logger = logging.getLogger(__name__)

DEFAULT_STATE = {
    "metadata": {
        "source_playlist": "",
        "episodes_built_from": [],
        "last_built_episode": 0,
        "updated_at": "",
    },
    "entities": [],
    "concepts": [],
    "claims": [],
    "scriptural_links": [],
    "historical_links": [],
    "glossary": [],
    "open_threads": [],
    "canonical_repetitions": [],
}


def _now() -> str:
    return datetime.now().strftime("%Y-%m-%dT%H:%M:%S")


def _deduplicate_by_id(items: List[dict]) -> List[dict]:
    """Deduplicate a list of dicts by their 'id' field, keeping the last occurrence."""
    seen = {}
    for item in items:
        item_id = item.get("id")
        if item_id:
            seen[item_id] = item
    return list(seen.values())


def _make_item(category: str, item_id: str, text: str, episode_num: int) -> dict:
    """Wrap a plain string into a structured knowledge dict."""
    base = {"id": item_id, "episode_numbers": [episode_num], "tags": []}
    if category == "entities":
        base["title"] = text[:80]
        base["summary"] = text[:200]
        base["category"] = "theological_category"
    elif category == "concepts":
        base["title"] = text[:80]
        base["summary"] = text[:200]
    elif category == "claims":
        base["title"] = text[:80]
        base["claim"] = text[:200]
    elif category == "scriptural_links":
        base["reference"] = text[:80]
        base["summary"] = text[:200]
    elif category == "historical_links":
        base["title"] = text[:80]
        base["summary"] = text[:200]
    elif category == "glossary":
        base["term"] = text[:80]
        base["definition"] = text[:200]
        base["language"] = "English"
    elif category == "open_threads":
        base["title"] = text[:80]
        base["summary"] = text[:200]
    elif category == "canonical_repetitions":
        base["title"] = text[:80]
        base["summary"] = text[:200]
    return base


def _merge_episode_numbers(items: list, episode_num: int) -> list:
    """Merge an episode number into each item's episode_numbers list if not present."""
    result = []
    for item in items:
        if not isinstance(item, dict):
            logger.warning("Skipping non-dict item: %s", str(item)[:80])
            continue
        item = dict(item)  # shallow copy
        ep_nums = item.get("episode_numbers", [])
        if isinstance(ep_nums, list) and episode_num not in ep_nums:
            ep_nums = ep_nums + [episode_num]
            item["episode_numbers"] = sorted(ep_nums)
        result.append(item)
    return result


class UniverseState:
    """Persistent cross-episode knowledge base.

    Loads from / saves to a JSON file. Provides context for LLM prompts
    and accumulates new knowledge after each episode.
    """

    def __init__(self, path: str):
        self.path = path
        self.data: Dict[str, Any] = dict(DEFAULT_STATE)
        self.load()

    # ------------------------------------------------------------------
    # I/O
    # ------------------------------------------------------------------

    def load(self):
        if os.path.exists(self.path):
            try:
                with open(self.path, "r", encoding="utf-8") as f:
                    loaded = json.load(f)
                for key in DEFAULT_STATE:
                    if key not in loaded:
                        loaded[key] = (
                            []
                            if isinstance(DEFAULT_STATE[key], list)
                            else dict(DEFAULT_STATE[key])
                        )
                self.data = loaded
                meta = self.data.get("metadata", {})
                logger.info(
                    "Loaded universe state: %d episodes, %d concepts, %d entities",
                    meta.get("last_built_episode", 0),
                    len(self.data.get("concepts", [])),
                    len(self.data.get("entities", [])),
                )
            except (json.JSONDecodeError, KeyError) as e:
                logger.warning("Corrupted universe state at %s: %s — starting fresh", self.path, e)
                self.data = dict(DEFAULT_STATE)
        else:
            logger.info("No universe state at %s — starting fresh", self.path)

    def save(self):
        self.data["metadata"]["updated_at"] = _now()
        os.makedirs(os.path.dirname(os.path.abspath(self.path)) or ".", exist_ok=True)
        tmp = self.path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(self.data, f, ensure_ascii=False, indent=2)
        os.replace(tmp, self.path)
        meta = self.data.get("metadata", {})
        logger.info(
            "Saved universe state: %d episodes, %d concepts, %d entities",
            meta.get("last_built_episode", 0),
            len(self.data.get("concepts", [])),
            len(self.data.get("entities", [])),
        )

    # ------------------------------------------------------------------
    # Context for prompts
    # ------------------------------------------------------------------

    def get_context(self, max_items: int = 8, max_chars: int = 3000) -> str:
        """Format universe state as a concise string for LLM prompt context.

        Args:
            max_items: Maximum number of items to include per category.
            max_chars: Maximum total output length. Truncated if exceeded.
        """
        parts = []

        for label, items, formatter in [
            ("Core concepts already established:", self.data.get("concepts", []),
             lambda c: f"- {c.get('title', c.get('id', '?'))}: {c.get('summary', '')}" if c.get('summary') else f"- {c.get('title', c.get('id', '?'))}"),
            ("Key entities:", self.data.get("entities", []),
             lambda e: f"- {e.get('title', e.get('id', '?'))}: {e.get('summary', '')}" if e.get('summary') else f"- {e.get('title', e.get('id', '?'))}"),
            ("Glossary:", self.data.get("glossary", []),
             lambda g: f"- {g.get('term', g.get('id', '?'))}: {g.get('definition', '')}" if g.get('definition') else f"- {g.get('term', g.get('id', '?'))}"),
            ("Key claims:", self.data.get("claims", []),
             lambda c: f"- {c.get('title', c.get('id', '?'))}: {c.get('claim', '')}" if c.get('claim') else f"- {c.get('title', c.get('id', '?'))}"),
            ("Frequent repetitions (drop on repeat):", self.data.get("canonical_repetitions", []),
             lambda c: f"- {c.get('title', c.get('id', '?'))}: {c.get('summary', '')}" if c.get('summary') else f"- {c.get('title', c.get('id', '?'))}"),
        ]:
            selected = items[:max_items]
            if selected:
                block = label + "\n" + "\n".join(formatter(it) for it in selected if formatter(it))
                if len(block) > max_chars // 3:
                    block = block[:max_chars // 3] + "..."
                parts.append(block)

        if not parts:
            return "(no prior episodes processed — universe state is empty)"

        result = "\n\n".join(parts)
        if len(result) > max_chars:
            result = result[:max_chars - 100] + "\n...(truncated)"
        return result

    # ------------------------------------------------------------------
    # Add knowledge from an episode
    # ------------------------------------------------------------------

    def add_episode_knowledge(self, episode_num: int, knowledge: dict):
        """Merge structured knowledge from an episode into the state.

        Accepts both dict items (with id, title, summary) and string items
        (which get auto-wrapped into dict format).

        Args:
            episode_num: 1-based episode number.
            knowledge: dict with keys matching DEFAULT_STATE (entities, concepts, etc.)
        """
        for category in ["entities", "concepts", "claims", "scriptural_links",
                          "historical_links", "glossary", "open_threads",
                          "canonical_repetitions"]:
            items = knowledge.get(category, [])
            if not items:
                continue

            # Normalize string items into dict format
            normalized = []
            for item in items:
                if isinstance(item, str):
                    item_text = item.strip().rstrip(".")
                    if not item_text:
                        continue
                    item_id = item_text.lower().replace(" ", "_").replace("'", "")[:60]
                    normalized.append(_make_item(category, item_id, item_text, episode_num))
                elif isinstance(item, dict):
                    item["episode_numbers"] = list(set(
                        item.get("episode_numbers", []) + [episode_num]
                    ))
                    normalized.append(item)

            if not normalized:
                continue

            # Merge with existing (dedup by id)
            existing = self.data.get(category, [])
            existing_ids = {e.get("id") for e in existing if isinstance(e, dict)}
            deduped = []
            for item in normalized:
                item_id = item.get("id") or item.get("term", "").lower().replace(" ", "_") or item.get("title", "").lower().replace(" ", "_")
                if item_id in existing_ids:
                    continue
                existing_ids.add(item_id)
                if not item.get("id"):
                    item["id"] = item_id
                deduped.append(item)

            self.data[category] = existing + deduped

            logger.info(
                "  %s: %d new items (total: %d)",
                category, len(deduped), len(self.data[category]),
            )

        # Update metadata
        meta = self.data.setdefault("metadata", {})
        meta.setdefault("episodes_built_from", [])
        if episode_num not in meta["episodes_built_from"]:
            meta["episodes_built_from"].append(episode_num)
            meta["episodes_built_from"].sort()
        meta["last_built_episode"] = max(
            meta.get("last_built_episode", 0), episode_num
        )
        self.save()

    def add_episode_concepts(self, episode_num: int, title: str, video_id: str,
                              concepts: List[str]):
        """Simpler variant: add only concept strings (no structured knowledge).

        Used for lightweight updates after single-episode processing.
        """
        new_entries = []
        existing_ids = {c.get("id", "") for c in self.data.get("concepts", [])}

        for concept_text in concepts:
            concept_text = concept_text.strip().rstrip(".")
            if not concept_text:
                continue
            cid = concept_text.lower().replace(" ", "_").replace("'", "")[:60]
            if cid in existing_ids:
                continue
            new_entries.append({
                "id": cid,
                "title": concept_text[:80],
                "summary": concept_text[:200],
                "episode_numbers": [episode_num],
                "related_entities": [],
                "tags": [],
            })
            existing_ids.add(cid)

        if new_entries:
            self.data.setdefault("concepts", []).extend(new_entries)
            logger.info("Added %d new concepts from episode %d", len(new_entries), episode_num)

        meta = self.data.setdefault("metadata", {})
        meta.setdefault("episodes_built_from", [])
        if episode_num not in meta["episodes_built_from"]:
            meta["episodes_built_from"].append(episode_num)
            meta["episodes_built_from"].sort()
        meta["last_built_episode"] = max(
            meta.get("last_built_episode", 0), episode_num
        )

        # Record episode info
        meta.setdefault("episodes", {})
        meta["episodes"][str(episode_num)] = {
            "title": title,
            "video_id": video_id,
            "concept_count": len(new_entries),
        }

        self.save()

    # ------------------------------------------------------------------
    # LLM-driven knowledge extraction
    # ------------------------------------------------------------------

    @staticmethod
    def extract_knowledge(
        block_summaries: List[dict],
        global_outline: str,
        episode_title: str = "",
        episode_number: Optional[int] = None,
        model: str = "qwen2.5:7b",
        prompt_path: str = "",
        host: str = "http://localhost:11434",
        timeout: int = 600,
    ) -> dict:
        """Call the LLM to extract structured knowledge from episode data.

        Args:
            block_summaries: List of {block_id, summary, start_time, end_time, word_count}
            global_outline: Episode outline string
            episode_title: The episode title
            episode_number: Optional episode number
            model: Ollama model name
            prompt_path: Path to the extraction prompt template
            host: Ollama host URL
            timeout: Request timeout

        Returns:
            dict with keys matching DEFAULT_STATE, or empty dict if extraction fails.
        """
        prompt_template = _load_prompt(prompt_path)

        payload = json.dumps({
            "episode_title": episode_title,
            "episode_number": episode_number,
            "block_summaries": block_summaries,
            "global_outline": global_outline,
        }, ensure_ascii=False, indent=2)

        full_prompt = prompt_template.strip() + "\n\n" + payload

        logger.info(
            "Extracting knowledge from '%s' (%d blocks, %d chars)",
            episode_title, len(block_summaries), len(full_prompt),
        )

        try:
            raw = generate(
                prompt=full_prompt,
                model=model,
                host=host,
                timeout=timeout,
                temperature=0.1,
                force_json=True,
            )
            debug_tag = f"ep{episode_number or ''}"
            knowledge = _parse_structured_response(raw, debug_tag=debug_tag)
            if knowledge:
                logger.info(
                    "Extracted: %d entities, %d concepts, %d claims, %d gloss terms",
                    len(knowledge.get("entities", [])),
                    len(knowledge.get("concepts", [])),
                    len(knowledge.get("claims", [])),
                    len(knowledge.get("glossary", [])),
                )
                return knowledge
            else:
                logger.warning("Knowledge extraction returned empty result")
                return {}
        except Exception as e:
            logger.error("Knowledge extraction failed: %s", e)
            return {}

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------

    def reset(self):
        """Clear all state (useful for testing or rebuilding)."""
        self.data = dict(DEFAULT_STATE)
        self.data["metadata"]["updated_at"] = _now()
        self.save()
        logger.info("Universe state reset to empty")


# ------------------------------------------------------------------
# Internal helpers
# ------------------------------------------------------------------

def _load_prompt(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def _parse_structured_response(raw: str, debug_tag: str = "") -> Optional[dict]:
    """Parse the JSON response from knowledge extraction.

    Accepts:
    - ```json ... ```
    - ``` ... ```
    - unfenced raw JSON
    - leading/trailing prose (extracts first JSON object found)
    - common malformed JSON (repair pass)

    Debug: if non-empty debug_tag, saves raw text and parse status to output/debug/.
    """
    if not raw:
        return None

    text = raw.strip()
    raw_len = len(raw)
    prefix = raw[:200]

    # Save raw for debugging
    if debug_tag:
        _save_debug_raw(debug_tag, "raw", raw)

    # Step 1: Strip markdown code fences (both ```json and ```)
    stripped_any_fence = False
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
            stripped_any_fence = True

    # Step 2: Find first JSON object
    start = text.find("{")
    if start < 0:
        logger.warning("_parse_structured_response(%s): no { found in response", debug_tag)
        return None
    end = text.rfind("}")
    if end <= start:
        logger.warning("_parse_structured_response(%s): no matching }} found", debug_tag)
        return None
    candidate = text[start:end + 1]

    # Step 3: Try strict parse
    try:
        result = json.loads(candidate)
        if isinstance(result, dict):
            return result
        return None
    except json.JSONDecodeError as e:
        logger.debug("_parse_structured_response(%s): strict parse failed: %s", debug_tag, e)

    # Step 4: Repair common malformed JSON (trailing commas, single quotes)
    import re as _re
    repaired = candidate
    repaired = _re.sub(r",\s*([}\]])", r"\1", repaired)
    repaired = _re.sub(r",\s*}", "}", repaired)
    repaired = _re.sub(r",\s*]", "]", repaired)
    # Replace single quotes with double quotes (but not inside already-escaped strings)
    # Simple approach: if strict parse failed due to single quotes, try replacing
    if "'" in repaired:
        # Only do this if the string looks like it has single-quoted keys
        repaired = _re.sub(r"(?<=[{, ])'([^']+?)'(?=\s*:)", r'"\1"', repaired)

    try:
        result = json.loads(repaired)
        if isinstance(result, dict):
            return result
        return None
    except json.JSONDecodeError as e:
        logger.debug("_parse_structured_response(%s): repair parse failed: %s", debug_tag, e)

    # Step 5: Last resort — extract first { ... } that forms valid JSON via bracket matching
    depth = 0
    start_idx = -1
    for i, ch in enumerate(candidate):
        if ch == "{":
            if depth == 0:
                start_idx = i
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0 and start_idx >= 0:
                try:
                    result = json.loads(candidate[start_idx:i + 1])
                    if isinstance(result, dict):
                        return result
                except json.JSONDecodeError:
                    pass
                start_idx = -1

    logger.warning("_parse_structured_response(%s): ALL parses failed. raw_len=%d first_200=%s",
                   debug_tag, raw_len, raw[:200])
    return None


import os as _os
_DEBUG_DIR = _os.path.join(
    _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))), "output", "debug"
)


def _save_debug_raw(tag: str, stage: str, content: str):
    """Save raw model output to output/debug/ for inspection."""
    _os.makedirs(_DEBUG_DIR, exist_ok=True)
    path = _os.path.join(_DEBUG_DIR, f"{tag}_{stage}.txt")
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
    logger.debug("Saved debug raw: %s", path)
