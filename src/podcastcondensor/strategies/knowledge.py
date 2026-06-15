"""Strategy implementations for Phase D: knowledge extraction.

Two strategies:
  - ``OllamaKnowledgeExtractionStrategy`` — wraps the existing
    ``UniverseState.extract_knowledge`` static method.
  - ``DeepSeekKnowledgeExtractionStrategy`` — reimplements extraction
    using the DeepSeek LLM client.
"""

import json
import logging
from typing import List, Optional

from podcastcondensor.strategies.base import KnowledgeExtractionStrategy

logger = logging.getLogger(__name__)

# Schema version — bump when the extraction prompt output format changes.
# This is part of the knowledge cache fingerprint so stale data is
# automatically invalidated.
EXTRACTION_SCHEMA_VERSION = "1"


class OllamaKnowledgeExtractionStrategy(KnowledgeExtractionStrategy):
    """Delegates to the existing ``UniverseState.extract_knowledge``.

    Preserves exact current behaviour — same prompt construction, same
    JSON parsing, same error handling.
    """

    def __init__(
        self,
        model: str,
        prompt_path: str,
        host: str = "http://localhost:11434",
        timeout: int = 600,
    ):
        self._model = model
        self._prompt_path = prompt_path
        self._host = host
        self._timeout = timeout

    def extract(
        self,
        block_summaries: List[dict],
        global_outline: str,
        episode_title: str = "",
        episode_number: Optional[int] = None,
    ) -> dict:
        from podcastcondensor.universe_state import UniverseState

        return UniverseState.extract_knowledge(
            block_summaries=block_summaries,
            global_outline=global_outline,
            episode_title=episode_title,
            episode_number=episode_number,
            model=self._model,
            prompt_path=self._prompt_path,
            host=self._host,
            timeout=self._timeout,
        )

    def name(self) -> str:
        return "ollama"

    @property
    def model(self) -> str:
        return self._model


class DeepSeekKnowledgeExtractionStrategy(KnowledgeExtractionStrategy):
    """Knowledge extraction via DeepSeek chat completions.

    Builds the same prompt shape as the Ollama path but sends it
    through the DeepSeek client.
    """

    def __init__(
        self,
        client,  # DeepSeekClient
        prompt_path: str,
        model: Optional[str] = None,
        timeout: int = 300,
    ):
        self._client = client
        self._prompt_path = prompt_path
        self._model = model or client.model
        self._timeout = timeout

    def extract(
        self,
        block_summaries: List[dict],
        global_outline: str,
        episode_title: str = "",
        episode_number: Optional[int] = None,
    ) -> dict:
        prompt_template = self._load_prompt(self._prompt_path)

        payload = json.dumps({
            "episode_title": episode_title,
            "episode_number": episode_number,
            "block_summaries": block_summaries,
            "global_outline": global_outline,
        }, ensure_ascii=False, indent=2)

        full_prompt = prompt_template.strip() + "\n\n" + payload

        logger.info(
            "DeepSeek knowledge extraction: '%s' (%d blocks, %d chars)",
            episode_title, len(block_summaries), len(full_prompt),
        )

        try:
            raw = self._client.generate(
                prompt=full_prompt,
                model=self._model,
                timeout=self._timeout,
                temperature=0.1,
                force_json=True,
            )
            knowledge = self._parse_structured_response(
                raw, debug_tag=f"ep{episode_number or ''}",
            )
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
                logger.warning("DeepSeek extraction returned empty result")
                return {}
        except Exception as e:
            logger.error("DeepSeek extraction failed: %s", e)
            return {}

    def name(self) -> str:
        return "deepseek"

    @property
    def model(self) -> str:
        return self._model

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _load_prompt(path: str) -> str:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()

    @staticmethod
    def _parse_structured_response(
        raw: str, debug_tag: str = "",
    ) -> Optional[dict]:
        """Parse the JSON response from knowledge extraction.

        Mirrors the logic in ``universe_state._parse_structured_response``.
        """
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
            logger.warning("_parse_structured_response(%s): no { found", debug_tag)
            return None
        end = text.rfind("}")
        if end <= start:
            logger.warning("_parse_structured_response(%s): no } found", debug_tag)
            return None
        candidate = text[start:end + 1]

        # Strict parse
        try:
            result = json.loads(candidate)
            return result if isinstance(result, dict) else None
        except json.JSONDecodeError:
            pass

        # Repair: trailing commas, single-quoted keys
        import re
        repaired = candidate
        repaired = re.sub(r",\s*([}\]])", r"\1", repaired)
        repaired = re.sub(r",\s*}", "}", repaired)
        repaired = re.sub(r",\s*]", "]", repaired)
        if "'" in repaired:
            repaired = re.sub(r"(?<=[{, ])'([^']+?)'(?=\s*:)", r'"\1"', repaired)

        try:
            result = json.loads(repaired)
            return result if isinstance(result, dict) else None
        except json.JSONDecodeError:
            pass

        logger.warning(
            "_parse_structured_response(%s): ALL parses failed. len=%d",
            debug_tag, len(raw),
        )
        return None
