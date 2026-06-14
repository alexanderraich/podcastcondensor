"""Tests for universe_state — context filtering, dedup, entity schema."""

import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from podcastcondensor.universe_state import UniverseState, _content_dedup_key


# ---------------------------------------------------------------------------
# Helper: build a minimal UniverseState without a real file
# ---------------------------------------------------------------------------

def _make_state(episodes_built_from: list = None, concepts: list = None,
                entities: list = None, claims: list = None,
                gloss: list = None, scriptural: list = None,
                historical: list = None, threads: list = None,
                reps: list = None):
    data = {
        "metadata": {
            "source_playlist": "",
            "episodes_built_from": episodes_built_from or [1],
            "last_built_episode": max(episodes_built_from or [1]),
            "updated_at": "",
        },
        "entities": entities or [],
        "concepts": concepts or [],
        "claims": claims or [],
        "scriptural_links": scriptural or [],
        "historical_links": historical or [],
        "glossary": gloss or [],
        "open_threads": threads or [],
        "canonical_repetitions": reps or [],
    }
    # Create a temp file so UniverseState can load it
    tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False)
    json.dump(data, tmp)
    tmp.close()
    state = UniverseState(tmp.name)
    os.unlink(tmp.name)
    return state


# ---------------------------------------------------------------------------
# 1. exclude_episode_gte filter
# ---------------------------------------------------------------------------

class TestExcludeEpisodeGte:
    """get_context(exclude_episode_gte=N) must exclude items from episodes >= N."""

    def test_excludes_concepts_from_current_episode(self):
        state = _make_state(
            episodes_built_from=[1, 2, 3],
            concepts=[
                {"id": "pre_existing", "title": "From Ep 2",
                 "summary": "Known concept", "episode_numbers": [2]},
                {"id": "current_ep", "title": "From Ep 3",
                 "summary": "Should be excluded", "episode_numbers": [3]},
            ],
        )
        ctx = state.get_context(exclude_episode_gte=3)
        assert "From Ep 3" not in ctx, "current-episode concept leaked through"
        assert "From Ep 2" in ctx, "prior-episode concept missing"

    def test_includes_items_with_no_episode_numbers(self):
        """Items lacking episode_numbers must be excluded (conservative)."""
        state = _make_state(
            episodes_built_from=[1],
            concepts=[
                {"id": "legacy", "title": "Legacy Item",
                 "summary": "No episode_numbers field"},
            ],
        )
        # This item has no episode_numbers key at all
        state.data["concepts"][0].pop("episode_numbers", None)
        ctx = state.get_context(exclude_episode_gte=2)
        assert "Legacy Item" not in ctx, (
            "item without episode_numbers leaked through"
        )

    def test_no_filter_when_exclude_none(self):
        state = _make_state(
            episodes_built_from=[1, 2],
            concepts=[
                {"id": "a", "title": "First", "summary": "x",
                 "episode_numbers": [1]},
                {"id": "b", "title": "Second", "summary": "y",
                 "episode_numbers": [2]},
            ],
        )
        ctx = state.get_context(exclude_episode_gte=None)
        assert "First" in ctx and "Second" in ctx

    def test_exclude_only_episodes_gte(self):
        """Only items whose min episode is >= threshold are excluded."""
        state = _make_state(
            episodes_built_from=[1, 2, 3],
            concepts=[
                {"id": "multi_ep", "title": "Spans Eps 1-2",
                 "summary": "Appeared in 1 and 2",
                 "episode_numbers": [1, 2]},
            ],
        )
        ctx = state.get_context(exclude_episode_gte=3)
        assert "Spans Eps 1-2" in ctx, (
            "item spanning earlier episodes should be included"
        )

    def test_exclude_multi_ep_item_with_current(self):
        """Item that includes current episode must be excluded."""
        state = _make_state(
            episodes_built_from=[1, 2, 3],
            concepts=[
                {"id": "multi_ep", "title": "Spans 2-3",
                 "summary": "Also appears in current ep",
                 "episode_numbers": [2, 3]},
            ],
        )
        ctx = state.get_context(exclude_episode_gte=3)
        assert "Spans 2-3" not in ctx, (
            "item spanning current episode leaked through"
        )


# ---------------------------------------------------------------------------
# 2. Content-based dedup key
# ---------------------------------------------------------------------------

class TestContentDedupKey:
    """_content_dedup_key must produce stable keys for same content."""

    def test_entities_key(self):
        e1 = {"title": "  Adam  ", "summary": "First man."}
        e2 = {"title": "Adam", "summary": "First man."}
        assert _content_dedup_key(e1, "entities") == _content_dedup_key(e2, "entities")

    def test_different_entities_different_keys(self):
        e1 = {"title": "Adam", "summary": "First man."}
        e2 = {"title": "Eve", "summary": "First woman."}
        assert _content_dedup_key(e1, "entities") != _content_dedup_key(e2, "entities")

    def test_glossary_uses_term(self):
        g = {"term": "Theotokos", "definition": "God-bearer."}
        key = _content_dedup_key(g, "glossary")
        assert "theotokos" in key
        assert "god-bearer" in key

    def test_scriptural_uses_reference(self):
        s = {"reference": "Gen 1:1", "summary": "Creation."}
        key = _content_dedup_key(s, "scriptural_links")
        assert "gen 1:1" in key
        assert "creation" in key

    def test_fallback_empty_fields(self):
        e = {}
        key = _content_dedup_key(e, "entities")
        assert key == "::"  # empty name + empty summary


# ---------------------------------------------------------------------------
# 3. Dedup in add_episode_knowledge
# ---------------------------------------------------------------------------

class TestAddEpisodeKnowledgeDedup:
    """Repeated calls with same knowledge must not duplicate."""

    def test_id_based_dedup(self):
        state = _make_state()
        ep_num = 21
        knowledge = {
            "concepts": [
                {"id": "deification", "title": "Deification",
                 "summary": "Becoming like God", "episode_numbers": [ep_num]},
            ]
        }
        state.add_episode_knowledge(ep_num, knowledge)
        assert len(state.data["concepts"]) == 1
        state.add_episode_knowledge(ep_num, knowledge)
        assert len(state.data["concepts"]) == 1, (
            "Same ID should not duplicate"
        )

    def test_content_key_fallback(self):
        """Items with different IDs but same content must not duplicate."""
        state = _make_state()
        ep_num = 21

        knowledge_1 = {
            "concepts": [
                {"id": "def_123", "title": "Deification",
                 "summary": "Becoming like God", "episode_numbers": [ep_num]},
            ]
        }
        knowledge_2 = {
            "concepts": [
                {"id": "def_456", "title": "Deification",
                 "summary": "Becoming like God", "episode_numbers": [ep_num]},
            ]
        }

        state.add_episode_knowledge(ep_num, knowledge_1)
        assert len(state.data["concepts"]) == 1

        state.add_episode_knowledge(ep_num, knowledge_2)
        assert len(state.data["concepts"]) == 1, (
            "Content-based dedup should catch unstable LLM IDs"
        )

    def test_content_key_still_allows_different_content(self):
        """Items with different content should NOT be deduped."""
        state = _make_state()
        ep_num = 21

        k1 = {"concepts": [{"id": "a", "title": "Deification",
                            "summary": "Becoming like God",
                            "episode_numbers": [ep_num]}]}
        k2 = {"concepts": [{"id": "b", "title": "Theosis",
                            "summary": "Same concept, Greek term",
                            "episode_numbers": [ep_num]}]}
        state.add_episode_knowledge(ep_num, k1)
        state.add_episode_knowledge(ep_num, k2)
        assert len(state.data["concepts"]) == 2, (
            "Different content should NOT be deduped"
        )

    def test_string_item_dedup(self):
        """String items (auto-wrapped) must dedup deterministically."""
        state = _make_state()
        ep_num = 21
        knowledge = {"concepts": ["Deification", "Divine Simplicity"]}

        state.add_episode_knowledge(ep_num, knowledge)
        assert len(state.data["concepts"]) == 2

        state.add_episode_knowledge(ep_num, knowledge)
        assert len(state.data["concepts"]) == 2, (
            "String items should dedup on deterministic ID"
        )


# ---------------------------------------------------------------------------
# 4. Entity schema completeness
# ---------------------------------------------------------------------------

class TestEntitySchemaCompleteness:
    """Verify the prompts/extract_knowledge_fast.txt has explicit entity fields."""

    def test_prompt_has_entity_fields(self):
        prompt_path = os.path.join(
            os.path.dirname(__file__), "..", "prompts", "extract_knowledge_fast.txt"
        )
        with open(prompt_path) as f:
            text = f.read()
        assert "entities — each entity object must have:" in text, (
            "Entity schema section missing from prompt"
        )
        assert '"id"' in text, "Entity id field missing from prompt schema"
        assert '"title"' in text, "Entity title field missing"
        assert '"category"' in text, "Entity category field missing"
        assert '"summary"' in text, "Entity summary field missing"
        assert '"episode_numbers"' in text, (
            "Entity episode_numbers field missing"
        )

    def test_prompt_entity_categories_defined(self):
        prompt_path = os.path.join(
            os.path.dirname(__file__), "..", "prompts", "extract_knowledge_fast.txt"
        )
        with open(prompt_path) as f:
            text = f.read()
        # At least some valid categories must be listed
        assert '"person"' in text
        assert '"place"' in text

    def test_prompt_requires_entities_key(self):
        prompt_path = os.path.join(
            os.path.dirname(__file__), "..", "prompts", "extract_knowledge_fast.txt"
        )
        with open(prompt_path) as f:
            text = f.read()
        assert "entities" in text
        assert "return an empty array []" in text or "entities: []" in text
