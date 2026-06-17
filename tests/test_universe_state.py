"""Tests for universe_state — structured knowledge, extraction, context."""
import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from podcastcondensor.universe_state import UniverseState


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_state(episodes_built_from: list = None,
                concepts: list = None, entities: list = None,
                claims: list = None, gloss: list = None,
                scriptural: list = None,
                summaries: list = None) -> UniverseState:
    """Create a temporary UniverseState with given data.

    Cleans up the temp file on return.
    """
    data = {
        "metadata": {
            "source_playlist": "",
            "episodes_built_from": episodes_built_from or [1],
            "last_built_episode": max(episodes_built_from or [1]),
            "updated_at": "",
        },
        "episode_summaries": summaries or [],
        "entities": entities or [],
        "concepts": concepts or [],
        "claims": claims or [],
        "scriptural_links": scriptural or [],
        "glossary": gloss or [],
    }
    tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".json",
                                      delete=False)
    json.dump(data, tmp)
    tmp.close()
    state = UniverseState(tmp.name)
    return state


def _cleanup(state: UniverseState):
    try:
        os.unlink(state.path)
    except OSError:
        pass


# ---------------------------------------------------------------------------
# 1. I/O
# ---------------------------------------------------------------------------

class TestLoadSave:
    def test_load_nonexistent_starts_fresh(self):
        path = "/tmp/_test_us_nonexistent.json"
        if os.path.exists(path):
            os.unlink(path)
        try:
            state = UniverseState(path)
            assert state.data["metadata"]["last_built_episode"] == 0
            assert state.data["concepts"] == []
            assert state.data["entities"] == []
        finally:
            try:
                os.unlink(path)
            except OSError:
                pass

    def test_save_and_reload(self):
        path = "/tmp/_test_us_save.json"
        try:
            state = UniverseState(path)
            state.data["metadata"]["last_built_episode"] = 5
            state.data["concepts"].append({
                "id": "deification", "title": "Deification",
                "summary": "Becoming like God",
            })
            state.save()

            state2 = UniverseState(path)
            assert state2.data["metadata"]["last_built_episode"] == 5
            assert len(state2.data["concepts"]) == 1
            assert state2.data["concepts"][0]["id"] == "deification"
        finally:
            try:
                os.unlink(path)
            except OSError:
                pass


# ---------------------------------------------------------------------------
# 2. add_episode_knowledge — merging and dedup
# ---------------------------------------------------------------------------

class TestAddEpisodeKnowledge:
    def test_adds_summary(self):
        state = _make_state()
        try:
            state.add_episode_knowledge(5, {
                "summary": "Episode about deification.",
            })
            assert len(state.data["episode_summaries"]) == 1
            assert state.data["episode_summaries"][0]["summary"] == "Episode about deification."
        finally:
            _cleanup(state)

    def test_dedup_by_id(self):
        state = _make_state()
        try:
            k = {
                "concepts": [
                    {"id": "deification", "title": "Deification",
                     "summary": "Becoming like God"},
                ]
            }
            state.add_episode_knowledge(21, k)
            assert len(state.data["concepts"]) == 1
            state.add_episode_knowledge(22, k)
            # Same id should not duplicate
            assert len(state.data["concepts"]) == 1
        finally:
            _cleanup(state)

    def test_different_content_allowed(self):
        state = _make_state()
        try:
            k1 = {"concepts": [{"id": "a", "title": "Deification",
                                "summary": "Becoming like God"}]}
            k2 = {"concepts": [{"id": "b", "title": "Theosis",
                                "summary": "Same concept, Greek term"}]}
            state.add_episode_knowledge(21, k1)
            state.add_episode_knowledge(22, k2)
            assert len(state.data["concepts"]) == 2
        finally:
            _cleanup(state)

    def test_episode_numbers_added(self):
        state = _make_state()
        try:
            k = {"concepts": [{"id": "theosis", "title": "Theosis",
                               "summary": "Greek term"}]}
            state.add_episode_knowledge(7, k)
            assert state.data["concepts"][0]["episode_numbers"] == [7]
        finally:
            _cleanup(state)

    def test_metadata_updated(self):
        state = _make_state()
        try:
            state.add_episode_knowledge(9, {"summary": "Something"})
            assert 9 in state.data["metadata"]["episodes_built_from"]
            assert state.data["metadata"]["last_built_episode"] == 9
        finally:
            _cleanup(state)


# ---------------------------------------------------------------------------
# 3. get_context
# ---------------------------------------------------------------------------

class TestGetContext:
    def test_includes_summaries(self):
        state = _make_state(
            episodes_built_from=[1, 2],
            summaries=[
                {"episode_number": 1, "summary": "Angels and demons."},
                {"episode_number": 2, "summary": "Divine council."},
            ],
        )
        try:
            ctx = state.get_context()
            assert "Angels and demons" in ctx
            assert "Divine council" in ctx
        finally:
            _cleanup(state)

    def test_includes_concepts(self):
        state = _make_state(
            concepts=[
                {"id": "d", "title": "Deification",
                 "summary": "Becoming like God",
                 "episode_numbers": [1]},
            ]
        )
        try:
            ctx = state.get_context()
            assert "Deification" in ctx
        finally:
            _cleanup(state)

    def test_exclude_episode_gte(self):
        state = _make_state(
            concepts=[
                {"id": "old", "title": "Old Concept",
                 "summary": "From ep 5", "episode_numbers": [5]},
                {"id": "new", "title": "New Concept",
                 "summary": "From ep 8", "episode_numbers": [8]},
            ]
        )
        try:
            ctx = state.get_context(exclude_episode_gte=8)
            assert "Old Concept" in ctx
            assert "New Concept" not in ctx
        finally:
            _cleanup(state)

    def test_empty_when_nothing(self):
        state = _make_state()
        try:
            ctx = state.get_context()
            assert "no prior episodes" in ctx
        finally:
            _cleanup(state)

    def test_truncates_long_output(self):
        state = _make_state(
            summaries=[
                {"episode_number": 1, "summary": "word " * 2000},
            ]
        )
        try:
            ctx = state.get_context(max_chars=500)
            assert len(ctx) <= 800
        finally:
            _cleanup(state)


# ---------------------------------------------------------------------------
# 4. extract_knowledge_from_transcript — mocked LLM
# ---------------------------------------------------------------------------

class TestExtractKnowledge:
    def test_parses_valid_json_response(self):
        class MockClient:
            def generate(self, **kwargs):
                return json.dumps({
                    "summary": "Test episode.",
                    "concepts": [{"id": "test-concept", "title": "Test",
                                  "summary": "A test"}],
                    "entities": [],
                    "claims": [{"id": "test-claim", "text": "Claim here",
                                "topic": "Other"}],
                    "scriptural_links": [],
                    "glossary": [],
                })

        knowledge = UniverseState.extract_knowledge_from_transcript(
            transcript_text="This is a transcript.",
            episode_title="Test",
            episode_number=1,
            client=MockClient(),
        )
        assert knowledge.get("summary") == "Test episode."
        assert len(knowledge["concepts"]) == 1
        assert knowledge["concepts"][0]["id"] == "test-concept"
        assert len(knowledge["claims"]) == 1

    def test_handles_empty_response(self):
        class MockClient:
            def generate(self, **kwargs):
                return ""

        knowledge = UniverseState.extract_knowledge_from_transcript(
            transcript_text="Some text.",
            client=MockClient(),
        )
        assert knowledge == {}

    def test_handles_malformed_json(self):
        class MockClient:
            def generate(self, **kwargs):
                return '{"summary": "OK", "concepts": ['

        knowledge = UniverseState.extract_knowledge_from_transcript(
            transcript_text="Some text.",
            client=MockClient(),
        )
        # Should not crash, may get partial or empty
        assert isinstance(knowledge, dict)

    def test_handles_fenced_json(self):
        class MockClient:
            def generate(self, **kwargs):
                return "```json\n" + json.dumps({
                    "summary": "Fenced.",
                    "concepts": [],
                    "entities": [],
                    "claims": [],
                    "scriptural_links": [],
                    "glossary": [],
                }) + "\n```"

        knowledge = UniverseState.extract_knowledge_from_transcript(
            transcript_text="X.",
            client=MockClient(),
        )
        assert knowledge.get("summary") == "Fenced."

    def test_handles_client_error(self):
        class MockClient:
            def generate(self, **kwargs):
                raise RuntimeError("API down")

        knowledge = UniverseState.extract_knowledge_from_transcript(
            transcript_text="X.",
            client=MockClient(),
        )
        assert knowledge == {}


# ---------------------------------------------------------------------------
# 5. Reset
# ---------------------------------------------------------------------------

class TestReset:
    def test_clears_everything(self):
        state = _make_state(
            episodes_built_from=[1, 2, 3],
            concepts=[{"id": "x", "title": "X", "summary": "",
                       "episode_numbers": [1]}],
        )
        try:
            state.reset()
            assert state.data["concepts"] == []
            assert state.data["episode_summaries"] == []
            assert state.data["metadata"]["last_built_episode"] == 0
        finally:
            _cleanup(state)
