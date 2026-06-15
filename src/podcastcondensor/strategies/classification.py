"""Strategy implementations for Phase B/C: classification.

Two strategies:
  - ``OllamaClassifierStrategy`` — wraps the existing ``classifier.py`` functions.
  - ``DeepSeekClassifierStrategy`` — reimplements the same logic using the
    DeepSeek LLM client.
"""

from typing import List, Optional

from podcastcondensor.strategies.base import ClassifierStrategy, ClassificationFailedError

# ---------------------------------------------------------------------------
# Ollama classifier (current local path)
# ---------------------------------------------------------------------------

_OLLAMA_MODEL_SENTINEL = "__ollama__"


class OllamaClassifierStrategy(ClassifierStrategy):
    """Delegates to the existing ``classifier`` module functions.

    This preserves the exact current behavior — same batching, same
    resumability, same JSON parsing.
    """

    def __init__(
        self,
        model: str,
        prompt_path: str,
        host: str = "http://localhost:11434",
        ollama_timeout: int = 600,
        resolve_maybe_prompt_path: str = "",
    ):
        self._model = model
        self._prompt_path = prompt_path
        self._host = host
        self._ollama_timeout = ollama_timeout
        self._resolve_maybe_prompt_path = resolve_maybe_prompt_path

    # ------------------------------------------------------------------
    # ClassifierStrategy interface
    # ------------------------------------------------------------------

    def classify_segments(
        self,
        segments: List[dict],
        global_outline: str,
        block_summaries: List[dict],
        max_segments_per_batch: int = 3,
        output_path: Optional[str] = None,
        universe_state_context: str = "",
        kept_claims_so_far: Optional[List[str]] = None,
    ) -> List[dict]:
        # Defer to the existing module-level function
        from podcastcondensor.classifier import classify_segments as _cls

        return _cls(
            segments=segments,
            model=self._model,
            prompt_path=self._prompt_path,
            global_outline=global_outline,
            block_summaries=block_summaries,
            max_segments_per_batch=max_segments_per_batch,
            host=self._host,
            ollama_timeout=self._ollama_timeout,
            output_path=output_path,
            universe_state_context=universe_state_context,
        )

    def resolve_maybe(
        self,
        maybe_segments: List[dict],
        all_segments: List[dict],
        all_decisions: List[dict],
    ) -> List[dict]:
        from podcastcondensor.classifier import resolve_maybe as _rs

        return _rs(
            maybe_segments=maybe_segments,
            all_segments=all_segments,
            all_decisions=all_decisions,
            model=self._model,
            prompt_path=self._resolve_maybe_prompt_path,
            host=self._host,
            ollama_timeout=self._ollama_timeout,
        )

    def name(self) -> str:
        return "ollama"

    @property
    def model(self) -> str:
        return self._model


# ---------------------------------------------------------------------------
# DeepSeek classifier
# ---------------------------------------------------------------------------


class DeepSeekClassifierStrategy(ClassifierStrategy):
    """Classification via DeepSeek (OpenAI-compatible chat API).

    Reimplements batching, resumability, and JSON parsing using the
    DeepSeek LLM client.  Kept closely parallel to the Ollama path so
    behaviour is comparable.
    """

    def __init__(
        self,
        client,  # DeepSeekClient
        prompt_path: str,
        resolve_maybe_prompt_path: str = "",
        model: Optional[str] = None,
        timeout: int = 600,
        max_segments_per_batch: int = 3,
    ):
        self._client = client
        self._prompt_path = prompt_path
        self._resolve_maybe_prompt_path = resolve_maybe_prompt_path
        self._model = model or client.model
        self._timeout = timeout
        self._max_segments_per_batch = max_segments_per_batch

    # ------------------------------------------------------------------
    # ClassifierStrategy interface
    # ------------------------------------------------------------------

    def classify_segments(
        self,
        segments: List[dict],
        global_outline: str,
        block_summaries: List[dict],
        max_segments_per_batch: int = 3,
        output_path: Optional[str] = None,
        universe_state_context: str = "",
        kept_claims_so_far: Optional[List[str]] = None,
    ) -> List[dict]:
        """Classify all segments in a single API call.

        DeepSeek has a 128K context window, so batching is unnecessary.
        All segments, block summaries, and context are sent in one prompt.
        """
        import json
        import os
        import logging
        logger = logging.getLogger(__name__)

        prompt_template = self._load_prompt(self._prompt_path)
        all_decisions: List[dict] = []

        # Resume from saved progress
        if output_path and os.path.exists(output_path):
            try:
                with open(output_path) as f:
                    all_decisions = json.load(f)
                if len(all_decisions) == len(segments):
                    logger.info(
                        "Resuming: %d decisions already saved",
                        len(all_decisions),
                    )
                    return all_decisions
                logger.info(
                    "Partial save: %d/%d decisions, re-classifying all",
                    len(all_decisions), len(segments),
                )
            except (json.JSONDecodeError, KeyError):
                logger.warning("Corrupted decisions file, starting fresh")
                all_decisions = []

        kept_claims: List[str] = kept_claims_so_far or []

        # Build context — all segments, all block summaries, full outline
        batch_for_model = []
        for seg in segments:
            entry = dict(seg)
            entry["id"] = seg["segment_id"]
            batch_for_model.append(entry)

        payload_parts = {
            "chunks": batch_for_model,
            "block_summaries": block_summaries,
            "global_outline": global_outline,
            "kept_claims_so_far": kept_claims[-20:],
        }
        if universe_state_context:
            payload_parts["universe_state"] = universe_state_context

        payload_json = json.dumps(payload_parts, ensure_ascii=False, indent=2)
        # Append the JSON prefix the prompt expects
        payload = payload_json + '\n\n{"decisions": ['
        full_prompt = prompt_template.strip() + "\n\n" + payload

        logger.info(
            "DeepSeek single-batch: %d segments, %d blocks, %d chars",
            len(segments), len(block_summaries), len(full_prompt),
        )

        decisions = self._call_with_retry(full_prompt, logger)
        if decisions is None:
            raise ClassificationFailedError(
                "DeepSeek classification failed after retries — "
                "pipeline must fall back or abort; 'maybe' decisions "
                "must NOT be emitted to avoid spurious keep/drop flow."
            )

        all_decisions.extend(decisions)

        if output_path:
            self._save_decisions(output_path, all_decisions)

        return all_decisions

    def resolve_maybe(
        self,
        maybe_segments: List[dict],
        all_segments: List[dict],
        all_decisions: List[dict],
    ) -> List[dict]:
        import json
        import logging
        logger = logging.getLogger(__name__)

        prompt_template = self._load_prompt(self._resolve_maybe_prompt_path)
        sid_to_label = {d["id"]: d["label"] for d in all_decisions}
        sid_to_seg = {s["segment_id"]: s for s in all_segments}

        failures = 0
        for ms in maybe_segments:
            sid = ms["segment_id"]
            idx = next(
                (i for i, s in enumerate(all_segments) if s["segment_id"] == sid),
                None,
            )
            if idx is None:
                sid_to_label[sid] = "drop"
                continue

            prev_kept = next_kept = None
            for j in range(idx - 1, -1, -1):
                if sid_to_label.get(all_segments[j]["segment_id"]) == "keep":
                    prev_kept = all_segments[j]
                    break
            for j in range(idx + 1, len(all_segments)):
                if sid_to_label.get(all_segments[j]["segment_id"]) == "keep":
                    next_kept = all_segments[j]
                    break

            nearby = []
            for j in range(max(0, idx - 3), min(len(all_segments), idx + 4)):
                if j != idx:
                    nearby.append(all_segments[j]["text"][:200])

            payload = json.dumps({
                "target_chunk": ms,
                "previous_kept_chunk": prev_kept,
                "next_kept_chunk": next_kept,
                "nearby_context": "\n".join(nearby),
            }, ensure_ascii=False, indent=2)
            full_prompt = prompt_template.strip() + "\n\n" + payload

            try:
                raw = self._client.generate(
                    prompt=full_prompt,
                    model=self._model,
                    timeout=min(self._timeout, 120),
                    temperature=0.1,
                    force_json=True,
                )
                result = self._parse_resolve_response(raw)
                if result and result.get("label") in ("keep", "drop"):
                    sid_to_label[sid] = result["label"]
                    logger.info("Resolved maybe %s -> %s", sid, result["label"])
                else:
                    sid_to_label[sid] = "drop"
                    failures += 1
            except Exception as e:
                logger.warning("Failed to resolve %s, defaulting drop: %s", sid, e)
                sid_to_label[sid] = "drop"
                failures += 1

        # If a large fraction of resolve attempts failed systemically, do not
        # silently produce a degraded result — raise so the pipeline can fall back.
        if len(maybe_segments) > 5 and failures / len(maybe_segments) > 0.5:
            raise ClassificationFailedError(
                f"DeepSeek resolve_maybe failed for {failures}/{len(maybe_segments)} "
                f"segments — systemic failure, must not silently drop all maybes."
            )

        result = []
        for d in all_decisions:
            entry = dict(d)
            if entry["label"] == "maybe":
                entry["label"] = sid_to_label.get(entry["id"], "keep")
            result.append(entry)
        return result

    def name(self) -> str:
        return "deepseek"

    @property
    def model(self) -> str:
        return self._model

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _call_with_retry(self, full_prompt: str, logger, retries: int = 2) -> Optional[list]:
        """Send prompt, parse JSON decisions, retry on failure.

        Returns a list of decision dicts, or ``None`` if all retries fail.
        """
        from podcastcondensor.ollama_client import _parse_json_response as _parse_ollama

        for attempt in range(retries + 1):
            try:
                raw = self._client.generate(
                    prompt=full_prompt,
                    model=self._model,
                    timeout=self._timeout,
                    temperature=0.1,
                    max_tokens=12000,
                    force_json=True,
                )
                decisions = _parse_ollama(raw)
                if decisions:
                    return decisions
                logger.warning(
                    "DeepSeek attempt %d/%d: empty/invalid response",
                    attempt + 1, retries + 1,
                )
            except Exception as e:
                logger.warning(
                    "DeepSeek attempt %d/%d failed: %s",
                    attempt + 1, retries + 1, e,
                )
        return None

    @staticmethod
    def _load_prompt(path: str) -> str:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()

    @staticmethod
    def _save_decisions(path: str, decisions: list):
        import json, os
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(decisions, f, ensure_ascii=False, indent=2)
        os.replace(tmp, path)

    @staticmethod
    def _parse_resolve_response(raw: str) -> Optional[dict]:
        import json
        text = raw.strip()
        if "```" in text:
            for part in text.split("```"):
                part = part.strip()
                if part.startswith("{"):
                    text = part
                    break
        start = text.find("{")
        end = text.rfind("}")
        if start >= 0 and end > start:
            try:
                return json.loads(text[start:end + 1])
            except json.JSONDecodeError:
                pass
        return None
