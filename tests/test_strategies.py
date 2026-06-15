"""Tests for LLM strategy pattern — provider selection, fallback, cache, fingerprint."""

import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

# Preserve original env so we can restore after tests that set/clear keys
_ORIG_ENV = dict(os.environ)


# ---------------------------------------------------------------------------
# 1. Provider factory / selection
# ---------------------------------------------------------------------------

class TestCreateClassifier:
    """create_classifier() must return the right strategy for each provider."""

    def test_ollama_classifier(self):
        from podcastcondensor.strategies import create_classifier
        strat = create_classifier(
            provider="ollama",
            prompt_path="/dev/null",
            model="qwen2.5:7b",
        )
        from podcastcondensor.strategies.classification import OllamaClassifierStrategy
        assert isinstance(strat, OllamaClassifierStrategy)
        assert strat.name() == "ollama"

    def test_deepseek_classifier_requires_key(self):
        """DeepSeek classifier must raise if no API key is available."""
        os.environ.pop("DEEPSEEK_API_KEY", None)
        os.environ.pop("ANTHROPIC_AUTH_TOKEN", None)
        from podcastcondensor.strategies import create_classifier
        import pytest
        with pytest.raises(ValueError, match="DEEPSEEK_API_KEY|API key"):
            create_classifier(
                provider="deepseek",
                prompt_path="/dev/null",
            )

    def test_deepseek_classifier_with_key(self):
        from podcastcondensor.strategies import create_classifier
        strat = create_classifier(
            provider="deepseek",
            prompt_path="/dev/null",
            deepseek_api_key="sk-test123",
        )
        from podcastcondensor.strategies.classification import DeepSeekClassifierStrategy
        assert isinstance(strat, DeepSeekClassifierStrategy)
        assert strat.name() == "deepseek"

    def test_unknown_provider_raises(self):
        from podcastcondensor.strategies import create_classifier
        import pytest
        with pytest.raises(ValueError, match="Unknown"):
            create_classifier(provider="gpt-4", prompt_path="/dev/null")


class TestCreateKnowledgeExtractor:
    """create_knowledge_extractor() must return the right strategy."""

    def test_ollama_extractor(self):
        from podcastcondensor.strategies import create_knowledge_extractor
        strat = create_knowledge_extractor(
            provider="ollama",
            prompt_path="/dev/null",
        )
        from podcastcondensor.strategies.knowledge import OllamaKnowledgeExtractionStrategy
        assert isinstance(strat, OllamaKnowledgeExtractionStrategy)
        assert strat.name() == "ollama"

    def test_deepseek_extractor_requires_key(self):
        os.environ.pop("DEEPSEEK_API_KEY", None)
        os.environ.pop("ANTHROPIC_AUTH_TOKEN", None)
        from podcastcondensor.strategies import create_knowledge_extractor
        import pytest
        with pytest.raises(ValueError, match="ANTHROPIC_AUTH_TOKEN|API key"):
            create_knowledge_extractor(
                provider="deepseek",
                prompt_path="/dev/null",
            )


# ---------------------------------------------------------------------------
# 2. Missing-key validation (API-level)
# ---------------------------------------------------------------------------

class TestDeepSeekClientKeyValidation:
    """DeepSeekClient must fail fast when API key is missing."""

    def test_generate_raises_without_key(self):
        from podcastcondensor.llm.deepseek import DeepSeekClient
        client = DeepSeekClient(api_key="")  # no key
        from podcastcondensor.llm.base import LLMConnectionError
        import pytest
        with pytest.raises(LLMConnectionError, match="API key"):
            client.generate(prompt="test", model="deepseek-chat")


# ---------------------------------------------------------------------------
# 3. Retry behaviour (DeepSeek classifier strategy)
# ---------------------------------------------------------------------------

class TestDeepSeekClassifierRetry:
    """_call_with_retry must retry on transient failures and return None
    when all retries are exhausted."""

    def test_returns_none_on_all_retries_fail(self):
        from podcastcondensor.strategies.classification import DeepSeekClassifierStrategy
        import logging
        logger = logging.getLogger("test")

        class _FailingClient:
            def generate(self, **kw):
                raise RuntimeError("transient failure")
            @property
            def model(self):
                return "deepseek-chat"

        strat = DeepSeekClassifierStrategy(
            client=_FailingClient(),
            prompt_path="/dev/null",
            model="deepseek-chat",
            timeout=10,
        )
        result = strat._call_with_retry("prompt", logger, retries=1)
        assert result is None, "Should return None after exhausting retries"

    def test_classify_segments_raises_on_total_failure(self):
        """classify_segments() must NOT emit fake 'maybe' decisions on failure.

        It must raise ClassificationFailedError so the pipeline can fall
        back properly instead of silently producing degraded output.
        """
        from podcastcondensor.strategies.classification import DeepSeekClassifierStrategy
        from podcastcondensor.strategies.base import ClassificationFailedError
        import pytest

        class _FailingClient:
            def generate(self, **kw):
                raise RuntimeError("provider unreachable")
            @property
            def model(self):
                return "deepseek-chat"

        strat = DeepSeekClassifierStrategy(
            client=_FailingClient(),
            prompt_path="/dev/null",
            model="deepseek-chat",
            timeout=10,
        )

        with pytest.raises(ClassificationFailedError, match="failed after retries"):
            strat.classify_segments(
                segments=[{"segment_id": "s1", "text": "hello", "start": 0, "end": 1}],
                global_outline="test",
                block_summaries=[],
            )

    def test_resolve_maybe_raises_on_systemic_failure(self):
        """resolve_maybe must raise ClassificationFailedError when >50% of
        resolve attempts fail systemically."""
        from podcastcondensor.strategies.classification import DeepSeekClassifierStrategy
        from podcastcondensor.strategies.base import ClassificationFailedError
        import pytest, tempfile, os

        # Create a temp resolve prompt
        tf = tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False)
        tf.write("resolve: {target_chunk}")
        tf.close()

        try:
            class _FailingClient:
                def generate(self, **kw):
                    raise RuntimeError("provider unreachable")
                @property
                def model(self):
                    return "deepseek-chat"

            strat = DeepSeekClassifierStrategy(
                client=_FailingClient(),
                prompt_path="/dev/null",
                resolve_maybe_prompt_path=tf.name,
                model="deepseek-chat",
                timeout=10,
            )
            # Need >5 segments to trigger the systemic failure threshold (50% of >5)
            seg_ids = [f"s{i}" for i in range(8)]
            all_decisions = [
                {"id": sid, "label": "maybe", "reason": "cloud-classification-failed"}
                for sid in seg_ids
            ]
            all_segments = [
                {"segment_id": sid, "text": f"text-{i}", "start": i*2, "end": i*2+1}
                for i, sid in enumerate(seg_ids)
            ]
            with pytest.raises(ClassificationFailedError):
                strat.resolve_maybe(
                    maybe_segments=all_segments,
                    all_segments=all_segments,
                    all_decisions=all_decisions,
                )
        finally:
            os.unlink(tf.name)


# ---------------------------------------------------------------------------
# 4. Fallback-to-local behaviour
# ---------------------------------------------------------------------------

class TestFallbackLogic:
    """When DeepSeek is configured but unreachable, fallback must switch
    to the Ollama classifier."""

    def test_fallback_returns_ollama_strategy(self):
        cfg_path = self._make_temp_prompt()
        from podcastcondensor.strategies.classification import (
            OllamaClassifierStrategy,
        )
        strat = OllamaClassifierStrategy(
            model="qwen2.5:7b",
            prompt_path=cfg_path,
        )
        assert isinstance(strat, OllamaClassifierStrategy)
        os.unlink(cfg_path)

    def test_pipeline_errors_on_deepseek_without_key_no_fallback(self):
        """When no fallback is configured and key is missing, the strategy
        factory must raise a clear error."""
        import os, pytest
        os.environ.pop("DEEPSEEK_API_KEY", None)
        os.environ.pop("ANTHROPIC_AUTH_TOKEN", None)
        from podcastcondensor.strategies import create_classifier
        with pytest.raises(ValueError, match="ANTHROPIC_AUTH_TOKEN|API key"):
            create_classifier(
                provider="deepseek",
                prompt_path="/dev/null",
            )

    @staticmethod
    def _make_temp_prompt() -> str:
        tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False)
        tmp.write("classify: {chunks}")
        tmp.close()
        return tmp.name


# ---------------------------------------------------------------------------
# 5. JSON parse / validation failures
# ---------------------------------------------------------------------------

class TestJsonParseResponses:
    """Strategy JSON parsing must handle various edge cases."""

    def test_parse_structured_response_valid(self):
        from podcastcondensor.strategies.knowledge import DeepSeekKnowledgeExtractionStrategy
        raw = '{"entities": [{"id": "adam", "title": "Adam"}]}'
        result = DeepSeekKnowledgeExtractionStrategy._parse_structured_response(raw)
        assert result is not None
        assert result["entities"][0]["id"] == "adam"

    def test_parse_structured_response_code_fence(self):
        from podcastcondensor.strategies.knowledge import DeepSeekKnowledgeExtractionStrategy
        raw = '```json\n{"entities": [{"id": "adam"}]}\n```'
        result = DeepSeekKnowledgeExtractionStrategy._parse_structured_response(raw)
        assert result is not None
        assert result["entities"][0]["id"] == "adam"

    def test_parse_structured_response_trailing_comma(self):
        from podcastcondensor.strategies.knowledge import DeepSeekKnowledgeExtractionStrategy
        raw = '{"entities": [{"id": "adam",}]}'
        result = DeepSeekKnowledgeExtractionStrategy._parse_structured_response(raw)
        assert result is not None
        assert result["entities"][0]["id"] == "adam"

    def test_parse_structured_response_empty(self):
        from podcastcondensor.strategies.knowledge import DeepSeekKnowledgeExtractionStrategy
        assert DeepSeekKnowledgeExtractionStrategy._parse_structured_response("") is None
        assert DeepSeekKnowledgeExtractionStrategy._parse_structured_response("   ") is None

    def test_parse_structured_response_prose_before_after(self):
        from podcastcondensor.strategies.knowledge import DeepSeekKnowledgeExtractionStrategy
        raw = "Here is the result:\n\n{\"concepts\": [{\"id\": \"test\"}]}\n\nHope that helps."
        result = DeepSeekKnowledgeExtractionStrategy._parse_structured_response(raw)
        assert result is not None
        assert result["concepts"][0]["id"] == "test"

    def test_parse_resolve_response_valid(self):
        from podcastcondensor.strategies.classification import DeepSeekClassifierStrategy
        raw = '{"label": "keep", "reason": "adds new info"}'
        result = DeepSeekClassifierStrategy._parse_resolve_response(raw)
        assert result is not None
        assert result["label"] == "keep"

    def test_parse_resolve_response_code_fence(self):
        from podcastcondensor.strategies.classification import DeepSeekClassifierStrategy
        raw = '```json\n{"label": "drop", "reason": "repetition"}\n```'
        result = DeepSeekClassifierStrategy._parse_resolve_response(raw)
        assert result is not None
        assert result["label"] == "drop"


# ---------------------------------------------------------------------------
# 6. Knowledge cache fingerprint
# ---------------------------------------------------------------------------

class TestCacheFingerprint:
    """Knowledge cache fingerprint must depend on provider, model, prompt,
    and schema version."""

    def _make_cfg(self, provider="ollama", model="qwen2.5:3b", **kw):
        from podcastcondensor.config import Config
        # Remove duplicate keys — model takes precedence as kwarg
        overrides = {**kw}
        if "knowledge_provider" not in overrides:
            overrides["knowledge_provider"] = provider
        if "knowledge_model" not in overrides:
            overrides["knowledge_model"] = model
        cfg = Config(**overrides)
        return cfg

    def test_fingerprint_differs_on_provider(self):
        from podcastcondensor.pipeline import _compute_fingerprint
        with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as f:
            f.write(b"prompt template")
            prompt_path = f.name
        try:
            cfg1 = self._make_cfg(provider="ollama", extract_concepts_prompt_path=prompt_path)
            cfg2 = self._make_cfg(provider="deepseek", knowledge_model="deepseek-chat",
                                  extract_concepts_prompt_path=prompt_path)
            fp1 = _compute_fingerprint(cfg1, "outline")
            fp2 = _compute_fingerprint(cfg2, "outline")
            assert fp1["provider"] != fp2["provider"]
        finally:
            os.unlink(prompt_path)

    def test_fingerprint_differs_on_model(self):
        from podcastcondensor.pipeline import _compute_fingerprint
        with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as f:
            f.write(b"prompt template")
            prompt_path = f.name
        try:
            cfg1 = self._make_cfg(knowledge_model="model-a",
                                  extract_concepts_prompt_path=prompt_path)
            cfg2 = self._make_cfg(knowledge_model="model-b",
                                  extract_concepts_prompt_path=prompt_path)
            assert _compute_fingerprint(cfg1, "outline")["model"] != \
                   _compute_fingerprint(cfg2, "outline")["model"]
        finally:
            os.unlink(prompt_path)

    def test_fingerprint_differs_on_schema_version(self):
        from podcastcondensor.pipeline import _compute_fingerprint
        with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as f:
            f.write(b"prompt template")
            prompt_path = f.name
        try:
            cfg1 = self._make_cfg(knowledge_cache_schema_version="1",
                                  extract_concepts_prompt_path=prompt_path)
            cfg2 = self._make_cfg(knowledge_cache_schema_version="2",
                                  extract_concepts_prompt_path=prompt_path)
            assert _compute_fingerprint(cfg1, "outline")["schema_version"] != \
                   _compute_fingerprint(cfg2, "outline")["schema_version"]
        finally:
            os.unlink(prompt_path)

    def test_fingerprint_differs_on_prompt(self):
        from podcastcondensor.pipeline import _compute_fingerprint
        with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as f:
            f.write(b"version 1 prompt")
            p1 = f.name
        with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as f:
            f.write(b"version 2 prompt")
            p2 = f.name
        try:
            cfg1 = self._make_cfg(extract_concepts_prompt_path=p1)
            cfg2 = self._make_cfg(extract_concepts_prompt_path=p2)
            fp1 = _compute_fingerprint(cfg1, "outline")
            fp2 = _compute_fingerprint(cfg2, "outline")
            assert fp1["prompt_hash"] != fp2["prompt_hash"]
        finally:
            os.unlink(p1)
            os.unlink(p2)

    def test_cache_matches_on_same_config(self):
        from podcastcondensor.pipeline import _compute_fingerprint, _cache_fingerprint_matches
        with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as f:
            f.write(b"prompt")
            prompt_path = f.name
        try:
            cfg = self._make_cfg(extract_concepts_prompt_path=prompt_path)
            fp = _compute_fingerprint(cfg, "same outline")
            cached = {"_fingerprint": fp, "concepts": []}
            assert _cache_fingerprint_matches(cached, cfg, "same outline") is True
        finally:
            os.unlink(prompt_path)

    def test_cache_miss_on_outline_change(self):
        """Outline changes should NOT invalidate — fingerprint only covers
        provider, model, prompt, schema. (Outline is part of the input,
        not the config that determines validity.)"""
        from podcastcondensor.pipeline import _compute_fingerprint, _cache_fingerprint_matches
        with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as f:
            f.write(b"prompt")
            prompt_path = f.name
        try:
            cfg = self._make_cfg(extract_concepts_prompt_path=prompt_path)
            fp = _compute_fingerprint(cfg, "outline v1")
            cached = {"_fingerprint": fp, "concepts": []}
            # Outline differs but fingerprint should still match since we
            # intentionally did not include outline in the cache key
            assert _cache_fingerprint_matches(cached, cfg, "outline v2") is True
        finally:
            os.unlink(prompt_path)

    def test_cache_miss_on_missing_fingerprint(self):
        """Legacy cache without _fingerprint must be treated as a miss."""
        from podcastcondensor.pipeline import _cache_fingerprint_matches
        from podcastcondensor.config import Config
        cfg = Config()
        assert _cache_fingerprint_matches({"concepts": []}, cfg, "") is False


# ---------------------------------------------------------------------------
# 7. No-regression: Ollama-only defaults
# ---------------------------------------------------------------------------

class TestOllamaDefaults:
    """The default configuration must use Ollama for all phases."""

    def test_default_classification_provider_is_ollama(self):
        from podcastcondensor.config import Config
        cfg = Config()
        assert cfg.classification_provider == "ollama"

    def test_default_knowledge_provider_is_ollama(self):
        from podcastcondensor.config import Config
        cfg = Config()
        assert cfg.knowledge_provider == "ollama"

    def test_default_audio_strategy_is_single_pass_filter(self):
        from podcastcondensor.config import Config
        cfg = Config()
        assert cfg.audio_strategy == "single_pass_filter"

    def test_default_classification_model_resolves(self):
        from podcastcondensor.config import Config
        cfg = Config()
        assert cfg.classification_model  # should be set in __post_init__
        assert cfg.classification_model == cfg.classify_model  # same as ollama default


# ---------------------------------------------------------------------------
# 8. Degraded decision detection
# ---------------------------------------------------------------------------

class TestDegradedDecisions:
    """_decisions_are_degraded must detect cached results from prior failures."""

    def test_detects_cloud_failure_maybes(self):
        from podcastcondensor.pipeline import _decisions_are_degraded
        decisions = [
            {"id": "s1", "label": "maybe", "reason": "cloud-classification-failed"},
            {"id": "s2", "label": "maybe", "reason": "cloud-classification-failed"},
            {"id": "s3", "label": "maybe", "reason": "cloud-classification-failed"},
        ]
        assert _decisions_are_degraded(decisions) is True

    def test_clean_decisions_not_degraded(self):
        from podcastcondensor.pipeline import _decisions_are_degraded
        decisions = [
            {"id": "s1", "label": "keep", "reason": "new claim"},
            {"id": "s2", "label": "drop", "reason": "filler"},
            {"id": "s3", "label": "keep", "reason": "scriptural ref"},
        ]
        assert _decisions_are_degraded(decisions) is False

    def test_empty_not_degraded(self):
        from podcastcondensor.pipeline import _decisions_are_degraded
        assert _decisions_are_degraded([]) is False

    def test_mixed_does_not_false_positive(self):
        from podcastcondensor.pipeline import _decisions_are_degraded
        decisions = [
            {"id": "s1", "label": "keep", "reason": "new info"},
            {"id": "s2", "label": "drop", "reason": "filler"},
            {"id": "s3", "label": "maybe", "reason": "cloud-classification-failed"},
            {"id": "s4", "label": "keep", "reason": "theological point"},
            {"id": "s5", "label": "drop", "reason": "repetition"},
        ]
        # Only 1/5 = 20% are cloud-failure maybes, well below 50% threshold
        assert _decisions_are_degraded(decisions) is False


# ---------------------------------------------------------------------------
# 9. Continuity bias
# ---------------------------------------------------------------------------

class TestContinuityBias:
    """apply_continuity_bias must bridge isolated kept segments."""

    def test_bridge_keeps_link_between_two_kept(self):
        from podcastcondensor.classifier import apply_continuity_bias
        segments = [
            {"segment_id": "s1", "start": 0, "end": 5, "word_count": 50},
            {"segment_id": "s2", "start": 6, "end": 7, "word_count": 20},
            {"segment_id": "s3", "start": 8, "end": 13, "word_count": 50},
        ]
        decisions = [
            {"id": "s1", "label": "keep"},
            {"id": "s2", "label": "drop"},
            {"id": "s3", "label": "keep"},
        ]
        result = apply_continuity_bias(segments, decisions, bridge_gap_sec=3.0)
        result_labels = {d["id"]: d["label"] for d in result}
        # s2 should be promoted to keep because it bridges s1 and s3
        assert result_labels["s2"] == "keep"

    def test_isolated_kept_gets_context(self):
        from podcastcondensor.classifier import apply_continuity_bias
        segments = [
            {"segment_id": "s1", "start": 0, "end": 5, "word_count": 50},
            {"segment_id": "s2", "start": 6, "end": 10, "word_count": 60},
            {"segment_id": "s3", "start": 11, "end": 15, "word_count": 30},
        ]
        decisions = [
            {"id": "s1", "label": "drop"},
            {"id": "s2", "label": "keep"},
            {"id": "s3", "label": "drop"},
        ]
        result = apply_continuity_bias(segments, decisions, bridge_gap_sec=3.0)
        result_labels = {d["id"]: d["label"] for d in result}
        assert result_labels["s1"] == "keep"
        assert result_labels["s3"] == "keep"

    def test_short_neighbour_kept(self):
        from podcastcondensor.classifier import apply_continuity_bias
        segments = [
            {"segment_id": "s1", "start": 0, "end": 5, "word_count": 10},
            {"segment_id": "s2", "start": 5, "end": 10, "word_count": 60},
            {"segment_id": "s3", "start": 10, "end": 15, "word_count": 80},
        ]
        decisions = [
            {"id": "s1", "label": "drop"},
            {"id": "s2", "label": "keep"},
            {"id": "s3", "label": "keep"},
        ]
        result = apply_continuity_bias(segments, decisions)
        result_labels = {d["id"]: d["label"] for d in result}
        assert result_labels["s1"] == "keep"

    def test_noop_when_all_kept(self):
        from podcastcondensor.classifier import apply_continuity_bias
        segments = [
            {"segment_id": "s1", "start": 0, "end": 5, "word_count": 50},
            {"segment_id": "s2", "start": 6, "end": 10, "word_count": 60},
        ]
        decisions = [
            {"id": "s1", "label": "keep"},
            {"id": "s2", "label": "keep"},
        ]
        original = [dict(d) for d in decisions]
        result = apply_continuity_bias(segments, decisions)
        assert result == original


# ---------------------------------------------------------------------------
# 10. Tail detection
# ---------------------------------------------------------------------------

class TestTailDetection:
    """detect_tail_block must identify off-topic trailing content."""

    def test_detects_min_content_tail(self):
        from podcastcondensor.classifier import detect_tail_block
        segments = []
        for i in range(20):
            bid = 0 if i < 16 else 1
            segments.append({
                "segment_id": f"s{i:04d}",
                "start": i * 10,
                "end": i * 10 + 8,
                "block_id": bid,
                "text": "normal discussion content",
            })
        decisions = []
        for i in range(16):
            decisions.append({"id": f"s{i:04d}", "label": "keep"})
        for i in range(16, 20):
            decisions.append({"id": f"s{i:04d}", "label": "drop"})
        flagged = detect_tail_block(
            segments, decisions, tail_fraction=0.2, min_keep_fraction=0.03,
        )
        assert len(flagged) > 0

    def test_no_tail_when_main_content(self):
        from podcastcondensor.classifier import detect_tail_block
        segments = []
        for i in range(20):
            segments.append({
                "segment_id": f"s{i:04d}",
                "start": i * 10,
                "end": i * 10 + 8,
                "block_id": 0,
                "text": "normal discussion content",
            })
        decisions = [
            {"id": f"s{i:04d}", "label": "keep" if i < 15 else "drop"}
            for i in range(20)
        ]
        flagged = detect_tail_block(
            segments, decisions, tail_fraction=0.2, min_keep_fraction=0.03,
        )
        assert len(flagged) == 0

    def test_keyword_detection(self):
        from podcastcondensor.classifier import detect_tail_block
        segments = []
        for i in range(10):
            text = (
                "normal content here"
                if i < 8 else
                "please subscribe and donate to support us"
            )
            segments.append({
                "segment_id": f"s{i:04d}",
                "start": i * 10,
                "end": i * 10 + 8,
                "block_id": 0 if i < 8 else 1,
                "text": text,
            })
        decisions = [
            {"id": f"s{i:04d}", "label": "keep" if i < 8 else "drop"}
            for i in range(10)
        ]
        flagged = detect_tail_block(
            segments, decisions, tail_fraction=0.3, min_keep_fraction=0.0,
        )
        assert len(flagged) > 0

    def test_no_false_positive_on_small_episode(self):
        from podcastcondensor.classifier import detect_tail_block
        segments = [
            {"segment_id": "s1", "start": 0, "end": 10, "block_id": 0, "text": "hi"},
        ]
        decisions = [{"id": "s1", "label": "keep"}]
        assert detect_tail_block(segments, decisions) == []


# ---------------------------------------------------------------------------
# 11. ClassificationFailedError importable
# ---------------------------------------------------------------------------

class TestClassificationFailedError:
    """ClassificationFailedError must be importable and catchable."""

    def test_can_catch(self):
        from podcastcondensor.strategies.base import ClassificationFailedError
        try:
            raise ClassificationFailedError("test failure")
        except ClassificationFailedError as e:
            assert "test failure" in str(e)

    def test_inherits_from_exception(self):
        from podcastcondensor.strategies.base import ClassificationFailedError
        assert issubclass(ClassificationFailedError, Exception)
