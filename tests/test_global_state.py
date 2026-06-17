"""Tests for global_state — merged outline + knowledge extraction, block mapping."""

import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pytest
from podcastcondensor.global_state import (
    build_global_state,
    map_blocks_to_segments,
    _parse_json_response,
)


# ---------------------------------------------------------------------------
# build_global_state — mocked DeepSeek
# ---------------------------------------------------------------------------

def _mock_client(return_text: str):
    class MockClient:
        def generate(self, **kwargs):
            assert kwargs["model"] == "deepseek-chat"
            assert kwargs["temperature"] == 0.1
            return return_text
    return MockClient()


class TestBuildGlobalState:
    def test_returns_all_keys_on_valid_response(self):
        client = _mock_client(json.dumps({
            "topic_segments": [
                {"segment_id": 1, "title": "Intro", "summary": "Opening.",
                 "start_word_index": 0, "end_word_index": 100},
                {"segment_id": 2, "title": "Body", "summary": "Main.",
                 "start_word_index": 100, "end_word_index": 500},
            ],
            "global_outline": "- Point 1\n- Point 2",
            "summary": "A narrative summary.",
            "concepts": [{"id": "deification", "title": "Deification",
                          "summary": "Becoming like God"}],
            "entities": [{"id": "adam", "title": "Adam",
                          "category": "person", "summary": "First man"}],
            "claims": [{"id": "claim-1", "text": "A claim.",
                        "topic": "Theology"}],
            "scriptural_links": [{"id": "gen-1-1", "reference": "Gen 1:1",
                                  "summary": "Creation"}],
            "glossary": [{"id": "theosis", "term": "Theosis",
                          "definition": "Deification"}],
        }))

        result = build_global_state(
            transcript_text="This is a test transcript. " * 50,
            episode_title="Test Episode",
            episode_number=5,
            client=client,
        )

        assert len(result["blocks"]) == 2
        assert len(result["block_summaries"]) == 2
        assert result["global_outline"] == "- Point 1\n- Point 2"
        assert result["summary"] == "A narrative summary."
        assert len(result["concepts"]) == 1
        assert result["concepts"][0]["id"] == "deification"
        assert len(result["entities"]) == 1
        assert result["entities"][0]["id"] == "adam"
        assert len(result["claims"]) == 1
        assert len(result["scriptural_links"]) == 1
        assert len(result["glossary"]) == 1
        assert result["chunk_to_block"] == {}

    def test_handles_fenced_json(self):
        client = _mock_client(
            "```json\n" + json.dumps({
                "topic_segments": [
                    {"segment_id": 1, "title": "A", "summary": "S",
                     "start_word_index": 0, "end_word_index": 50},
                ],
                "global_outline": "- X",
                "summary": "",
                "concepts": [],
                "entities": [],
                "claims": [],
                "scriptural_links": [],
                "glossary": [],
            }) + "\n```"
        )

        result = build_global_state(
            transcript_text="Test.",
            client=client,
        )
        assert len(result["blocks"]) == 1

    def test_raises_on_empty_response(self):
        class EmptyClient:
            def generate(self, **kwargs):
                return ""

        with pytest.raises(RuntimeError, match="empty|unparseable"):
            build_global_state(
                transcript_text="Test.",
                client=EmptyClient(),
            )

    def test_raises_on_missing_topic_segments(self):
        client = _mock_client(json.dumps({
            "global_outline": "- X",
            "summary": "",
            "concepts": [],
            "entities": [],
            "claims": [],
            "scriptural_links": [],
            "glossary": [],
        }))

        with pytest.raises(RuntimeError, match="no topic_segments"):
            build_global_state(
                transcript_text="Test.",
                client=client,
            )

    def test_normalises_list_outline(self):
        client = _mock_client(json.dumps({
            "topic_segments": [
                {"segment_id": 1, "title": "A", "summary": "S",
                 "start_word_index": 0, "end_word_index": 50},
            ],
            "global_outline": ["Point A", "Point B"],
            "summary": "",
            "concepts": [],
            "entities": [],
            "claims": [],
            "scriptural_links": [],
            "glossary": [],
        }))

        result = build_global_state(
            transcript_text="Test.",
            client=client,
        )
        assert "- Point A" in result["global_outline"]
        assert "- Point B" in result["global_outline"]

    def test_handles_custom_prompt(self):
        """Should load prompt from file if path given."""
        tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".txt",
                                          delete=False)
        tmp.write("Custom prompt.")
        tmp.close()

        class CallCheckClient:
            def generate(self, **kwargs):
                prompt = kwargs["prompt"]
                assert "Custom prompt." in prompt
                return json.dumps({
                    "topic_segments": [
                        {"segment_id": 1, "title": "A", "summary": "S",
                         "start_word_index": 0, "end_word_index": 50},
                    ],
                    "global_outline": "- X",
                    "summary": "",
                    "concepts": [],
                    "entities": [],
                    "claims": [],
                    "scriptural_links": [],
                    "glossary": [],
                })

        result = build_global_state(
            transcript_text="Test.",
            client=CallCheckClient(),
            prompt_path=tmp.name,
        )
        assert len(result["blocks"]) == 1
        os.unlink(tmp.name)


# ---------------------------------------------------------------------------
# map_blocks_to_segments
# ---------------------------------------------------------------------------

class TestMapBlocksToSegments:
    def test_maps_by_word_index_overlap(self):
        block_summaries = [
            {"block_id": 1, "start_word_index": 0, "end_word_index": 10},
            {"block_id": 2, "start_word_index": 10, "end_word_index": 20},
        ]
        segments = [
            {"segment_id": "seg-001", "text": "one two three four"},
            {"segment_id": "seg-002", "text": "five six seven eight nine ten"},
            {"segment_id": "seg-003", "text": "eleven twelve thirteen"},
        ]
        result = map_blocks_to_segments(segments, block_summaries, "")
        # seg-001: 4 words → [0,4) → overlaps block 1 [0,10) → block 1
        assert result["seg-001"] == 1
        # seg-002: 6 words → [4,10) → overlaps block 1 [0,10) → block 1
        assert result["seg-002"] == 1
        # seg-003: 3 words → [10,13) → overlaps block 2 [10,20) → block 2
        assert result["seg-003"] == 2

    def test_empty_when_no_segments(self):
        assert map_blocks_to_segments([], [{"block_id": 1}], "") == {}

    def test_handles_uid_fallback(self):
        block_summaries = [
            {"block_id": 1, "start_word_index": 0, "end_word_index": 5},
        ]
        segments = [
            {"segment_id": "s1", "uid": "u1", "text": "a b c"},
        ]
        result = map_blocks_to_segments(segments, block_summaries, "")
        assert result["u1"] == 1
        assert result.get("s1") is None  # only uid key

    def test_no_overlap_returns_empty(self):
        block_summaries = [
            {"block_id": 1, "start_word_index": 100, "end_word_index": 200},
        ]
        segments = [
            {"segment_id": "s1", "text": "short"},
        ]
        result = map_blocks_to_segments(segments, block_summaries, "")
        assert result == {}


# ---------------------------------------------------------------------------
# _parse_json_response
# ---------------------------------------------------------------------------

class TestParseJsonResponse:
    def test_plain_json(self):
        result = _parse_json_response('{"a": 1}')
        assert result == {"a": 1}

    def test_fenced_json(self):
        result = _parse_json_response('```json\n{"a": 1}\n```')
        assert result == {"a": 1}

    def test_fenced_no_lang(self):
        result = _parse_json_response('```\n{"a": 1}\n```')
        assert result == {"a": 1}

    def test_trailing_comma_repair(self):
        result = _parse_json_response('{"a": 1,}')
        assert result == {"a": 1}

    def test_empty_returns_none(self):
        assert _parse_json_response("") is None
        assert _parse_json_response("   ") is None

    def test_no_json_returns_none(self):
        assert _parse_json_response("just text") is None
