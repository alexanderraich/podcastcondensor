"""Cross-episode knowledge base for podcastcondensor.

Rolling structured knowledge accumulation. Each episode produces a
structured summary (entities, concepts, claims, glossary, etc.) via a
single-shot DeepSeek call over the full transcript. The state grows by
merging these per-episode extractions — no raw transcript dump.

``get_context()`` formats the accumulated knowledge for the classifier
so it knows what has already been covered.
"""

import json
import logging
import os
import re
from datetime import datetime
from typing import List, Optional

logger = logging.getLogger(__name__)

DEFAULT_STATE = {
    "metadata": {
        "source_playlist": "",
        "episodes_built_from": [],
        "last_built_episode": 0,
        "updated_at": "",
    },
    "episode_summaries": [],
    "entities": [],
    "concepts": [],
    "claims": [],
    "scriptural_links": [],
    "glossary": [],
}


def _fresh_default() -> dict:
    return {
        "metadata": dict(DEFAULT_STATE["metadata"]),
        "episode_summaries": [],
        "entities": [],
        "concepts": [],
        "claims": [],
        "scriptural_links": [],
        "glossary": [],
    }


def _now() -> str:
    return datetime.now().strftime("%Y-%m-%dT%H:%M:%S")


# ------------------------------------------------------------------
# Prompt for the per-episode single-shot extraction
# ------------------------------------------------------------------

_EXTRACT_PROMPT = """You are an expert podcast transcript analyst.

Given the FULL transcript of a single episode of the "Lord of Spirits" podcast,
produce a structured summary. Return ONLY valid JSON — no markdown, no extra text.

Input:
{
  "episode_title": "...",
  "episode_number": N,
  "transcript": "Full cleaned transcript text..."
}

Output format:
{
  "summary": "2-3 paragraph narrative summary of the episode's content, themes, and key arguments.",
  "concepts": [
    {"id": "short-kebab-case-id", "title": "Concept Name", "summary": "Brief explanation"}
  ],
  "entities": [
    {"id": "short-kebab-case-id", "title": "Entity Name", "category": "person|place|theological|historical|other", "summary": "Brief description"}
  ],
  "claims": [
    {"id": "short-kebab-case-id", "text": "The claim being made (max 300 chars)", "topic": "Theology|Scripture|History|Other"}
  ],
  "scriptural_links": [
    {"id": "short-kebab-case-id", "reference": "Book Chapter:Verse", "summary": "How it is used in the episode"}
  ],
  "glossary": [
    {"id": "short-kebab-case-id", "term": "Term", "definition": "Definition"}
  ]
}

Rules:
- Every item must have a unique, stable-looking `id` (kebab-case).
- Items are episode-specific — the caller merges across episodes.
- Return an empty array [] for any category with nothing to report.
- Be precise and faithful to the transcript — no hallucination.
- Prioritise distinctive content (new concepts or developments) over generic statements."""


# ==================================================================


class UniverseState:
    """Persistent cross-episode knowledge base.

    Accumulates per-episode structured knowledge (entities, concepts, claims,
    glossary) plus episode summaries.  ``get_context()`` formats the
    accumulated knowledge for the classifier.
    """

    def __init__(self, path: str):
        self.path = path
        self.data = _fresh_default()
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
                logger.warning(
                    "Corrupted universe state at %s: %s — starting fresh",
                    self.path, e,
                )
                self.data = _fresh_default()
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
    # Per-episode knowledge extraction (single DeepSeek call)
    # ------------------------------------------------------------------

    @staticmethod
    def extract_knowledge_from_transcript(
        transcript_text: str,
        *,
        episode_title: str = "",
        episode_number: Optional[int] = None,
        client=None,
        model: str = "deepseek-chat",
        prompt: str = "",
        timeout: int = 300,
    ) -> dict:
        """Single-shot extraction: full transcript → structured knowledge.

        Returns a dict with keys matching DEFAULT_STATE (episode_summaries
        excluded here — handled by the caller), or an empty dict on failure.
        """
        effective_prompt = (prompt.strip() if prompt else _EXTRACT_PROMPT.strip())

        payload = json.dumps({
            "episode_title": episode_title,
            "episode_number": episode_number,
            "transcript": transcript_text,
        }, ensure_ascii=False, indent=2)

        full_prompt = effective_prompt + "\n\n" + payload

        logger.info(
            "Extracting knowledge: '%s' (%d chars total)",
            episode_title, len(full_prompt),
        )

        try:
            raw = client.generate(
                prompt=full_prompt,
                model=model,
                timeout=timeout,
                temperature=0.1,
                max_tokens=8192,
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
                logger.warning("Extraction returned empty")
                return {}
        except Exception as e:
            logger.error("Knowledge extraction failed: %s", e)
            return {}

    # ------------------------------------------------------------------
    # Merge extracted knowledge into the state
    # ------------------------------------------------------------------

    def add_episode_knowledge(self, episode_num: int, knowledge: dict):
        """Merge structured knowledge from an episode into the state.

        Accumulates episode summaries and merges entities/concepts/claims
        etc. with dedup by id.
        """
        # Episode summary
        ep_summary = knowledge.get("summary", "").strip()
        if ep_summary:
            eps = self.data.setdefault("episode_summaries", [])
            eps.append({
                "episode_number": episode_num,
                "summary": ep_summary,
            })

        # Structured items
        for category in ["entities", "concepts", "claims",
                         "scriptural_links", "glossary"]:
            items = knowledge.get(category, [])
            if not items:
                continue

            existing = self.data.get(category, [])
            existing_ids = {
                e.get("id") for e in existing if isinstance(e, dict) and e.get("id")
            }

            deduped = []
            for item in items:
                if not isinstance(item, dict):
                    continue
                item_id = item.get("id")
                if item_id and item_id in existing_ids:
                    continue
                item["episode_numbers"] = [episode_num]
                if item_id:
                    existing_ids.add(item_id)
                deduped.append(item)

            self.data[category] = existing + deduped
            if deduped:
                logger.info(
                    "  %s: %d new (total: %d)",
                    category, len(deduped), len(self.data[category]),
                )

        # Update metadata
        meta = self.data.setdefault("metadata", {})
        meta.setdefault("episodes_built_from", [])
        if episode_num not in meta["episodes_built_from"]:
            meta["episodes_built_from"].append(episode_num)
            meta["episodes_built_from"].sort()
        meta["last_built_episode"] = max(
            meta.get("last_built_episode", 0), episode_num,
        )
        self.save()

    # ------------------------------------------------------------------
    # Context for prompts
    # ------------------------------------------------------------------

    def get_context(self, exclude_episode_gte: Optional[int] = None,
                    max_chars: int = 3000) -> str:
        """Format accumulated universe knowledge as classifier context.

        Args:
            exclude_episode_gte: If set, exclude items whose episode_numbers
                include any episode >= this value.
            max_chars: Rough character budget for the output.
        """
        parts = []

        # Episode summaries (most recent first, capped)
        summaries = list(self.data.get("episode_summaries", []))
        if exclude_episode_gte is not None:
            summaries = [
                s for s in summaries
                if s.get("episode_number", 0) < exclude_episode_gte
            ]
        if summaries:
            block = "Recent episode summaries:\n"
            for s in summaries[-3:]:  # last 3
                block += f"- Ep {s['episode_number']}: {s['summary'][:300]}...\n"
            parts.append(block)

        # Structured items
        for label, items, formatter in [
            ("Core concepts already established:",
             self.data.get("concepts", []),
             lambda c: f"- {c.get('title', '?')}: {c.get('summary', '')}" if c.get('summary') else f"- {c.get('title', '?')}"),
            ("Key entities:",
             self.data.get("entities", []),
             lambda e: f"- {e.get('title', '?')}: {e.get('summary', '')}" if e.get('summary') else f"- {e.get('title', '?')}"),
            ("Glossary:",
             self.data.get("glossary", []),
             lambda g: f"- {g.get('term', '?')}: {g.get('definition', '')}" if g.get('definition') else f"- {g.get('term', '?')}"),
            ("Key claims:",
             self.data.get("claims", []),
             lambda c: f"- {c.get('text', '?')}" if c.get('text') else f"- {c.get('id', '?')}"),
            ("Scriptural references:",
             self.data.get("scriptural_links", []),
             lambda s: f"- {s.get('reference', '?')}: {s.get('summary', '')}" if s.get('summary') else f"- {s.get('reference', '?')}"),
        ]:
            if not items:
                continue
            if exclude_episode_gte is not None:
                items = [
                    it for it in items
                    if not any(ep >= exclude_episode_gte
                               for ep in it.get("episode_numbers", []))
                ]
            selected = items[:6]
            if selected:
                block = label + "\n" + "\n".join(formatter(it) for it in selected)
                parts.append(block)

        if not parts:
            return "(no prior episodes processed — universe state is empty)"

        result = "\n\n".join(parts)
        if len(result) > max_chars:
            result = result[:max_chars - 100] + "\n…(truncated)"
        return result

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------

    def reset(self):
        """Clear all state."""
        self.data = _fresh_default()
        self.data["metadata"]["updated_at"] = _now()
        self.save()
        logger.info("Universe state reset to empty")


# ------------------------------------------------------------------
# JSON parsing helper
# ------------------------------------------------------------------


def _parse_structured_response(raw: str) -> Optional[dict]:
    """Parse JSON from the LLM response, handling common formatting issues."""
    if not raw:
        return None

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

    # Find first JSON object
    start = text.find("{")
    if start < 0:
        return None
    end = text.rfind("}")
    if end <= start:
        return None
    candidate = text[start:end + 1]

    # Try strict parse
    try:
        result = json.loads(candidate)
        return result if isinstance(result, dict) else None
    except json.JSONDecodeError:
        pass

    # Try repair: trailing commas
    repaired = re.sub(r",\s*([}\]])", r"\1", candidate)
    try:
        result = json.loads(repaired)
        return result if isinstance(result, dict) else None
    except json.JSONDecodeError:
        pass

    logger.warning("Failed to parse extraction response (first 200 chars): %s",
                   candidate[:200])
    return None
