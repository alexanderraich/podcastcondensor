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


def _merge_episode_numbers(items: List[dict], episode_num: int) -> List[dict]:
    """Merge an episode number into each item's episode_numbers list if not present."""
    result = []
    for item in items:
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

    def get_context(self, max_items: int = 30) -> str:
        """Format universe state as a concise string for LLM prompt context.

        Args:
            max_items: Maximum number of items to include per category.
        """
        parts = []

        concepts = self.data.get("concepts", [])[:max_items]
        if concepts:
            lines = ["Core concepts already established in prior episodes:"]
            for c in concepts:
                title = c.get("title", c.get("id", "?"))
                summary = c.get("summary", "")
                if summary:
                    lines.append(f"- {title}: {summary}")
                else:
                    lines.append(f"- {title}")
            parts.append("\n".join(lines))

        entities = self.data.get("entities", [])[:max_items]
        if entities:
            lines = ["Key entities discussed in prior episodes:"]
            for e in entities:
                title = e.get("title", e.get("id", "?"))
                cat = e.get("category", "")
                cat_tag = f" ({cat})" if cat else ""
                summary = e.get("summary", "")
                if summary:
                    lines.append(f"- {title}{cat_tag}: {summary}")
                else:
                    lines.append(f"- {title}{cat_tag}")
            parts.append("\n".join(lines))

        glossary = self.data.get("glossary", [])[:max_items]
        if glossary:
            lines = ["Glossary of key terms (already defined):"]
            for g in glossary:
                term = g.get("term", g.get("id", "?"))
                definition = g.get("definition", "")
                if definition:
                    lines.append(f"- {term}: {definition}")
            parts.append("\n".join(lines))

        claims = self.data.get("claims", [])[:max_items]
        if claims:
            lines = ["Established claims (already covered):"]
            for c in claims:
                title = c.get("title", c.get("id", "?"))
                claim = c.get("claim", "")
                if claim:
                    lines.append(f"- {title}: {claim}")
            parts.append("\n".join(lines))

        canonical = self.data.get("canonical_repetitions", [])[:max_items // 2]
        if canonical:
            lines = ["Frequently repeated explanations (can usually be dropped):"]
            for c in canonical:
                title = c.get("title", c.get("id", "?"))
                summary = c.get("summary", "")
                if summary:
                    lines.append(f"- {title}: {summary}")
            parts.append("\n".join(lines))

        if not parts:
            return "(no prior episodes processed — universe state is empty)"

        return "\n\n".join(parts)

    # ------------------------------------------------------------------
    # Add knowledge from an episode
    # ------------------------------------------------------------------

    def add_episode_knowledge(self, episode_num: int, knowledge: dict):
        """Merge structured knowledge from an episode into the state.

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

            # Assign episode number to each item
            items = _merge_episode_numbers(items, episode_num)

            # Merge with existing (dedup by id)
            existing = self.data.get(category, [])
            merged = existing + items
            self.data[category] = _deduplicate_by_id(merged)

            logger.info(
                "  %s: %d new items (total: %d)",
                category, len(items), len(self.data[category]),
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
            knowledge = _parse_structured_response(raw)
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


def _parse_structured_response(raw: str) -> Optional[dict]:
    """Parse the JSON response from knowledge extraction.

    Handles markdown code fences and surrounding text.
    """
    if not raw:
        return None

    text = raw.strip()

    # Remove markdown code fences
    if text.startswith("```"):
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
            text = "\n".join(clean)

    # Find JSON object
    start = text.find("{")
    if start >= 0:
        end = text.rfind("}")
        if end > start:
            candidate = text[start:end + 1]
            try:
                result = json.loads(candidate)
                if isinstance(result, dict) and any(
                    k in result for k in DEFAULT_STATE
                ):
                    return result
                return result
            except json.JSONDecodeError:
                pass
    return None
