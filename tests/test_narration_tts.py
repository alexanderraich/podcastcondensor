"""Tests for narration + tts — digest-without-segments, TTS chunking.

Keeps real LLM/network calls out of the unit tests: ``narrate_episode``
accepts an injected client stub, and the TTS chunking is pure string logic.
"""
import json
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from podcastcondensor.narration import (
    _existing_narration,
    _mp3_path,
    _parse_chunk_range,
    build_corpus_narrations,
    narrate_episode,
)
from podcastcondensor.tts import _chunk_sentences, _split_sentences


def _make_episode_with_state(tmp_path, ep_num, summary, **categories):
    ep_dir = os.path.join(tmp_path, f"ep-{ep_num:03d}")
    os.makedirs(ep_dir, exist_ok=True)
    with open(os.path.join(ep_dir, "source_subtitles.srt"), "w", encoding="utf-8") as f:
        f.write("1\n00:00:00,000 --> 00:00:02,000\nA sentence.\n")
    data = {"summary": summary, **categories}
    with open(os.path.join(ep_dir, "global_state.json"), "w", encoding="utf-8") as f:
        json.dump(data, f)


class StubClient:
    """Minimal DeepSeekClient stand-in returning canned text."""
    def __init__(self, reply="Canned narration."):
        self.reply = reply
        self.prompts = []

    def generate(self, prompt, **kwargs):
        self.prompts.append(prompt)
        return self.reply


def test_narrate_episode_writes_text_without_segments(tmp_path):
    _make_episode_with_state(
        tmp_path, 1, "A summary.",
        claims=[{
            "text": "A claim.",
            "segments": [{"episode": 1, "start": 1704.0, "end": 1795.0}],
        }],
    )
    stub = StubClient("This is the narration prose.")

    narration = narrate_episode(output_root=str(tmp_path), ep_num=1, client=stub)

    assert narration == "This is the narration prose."
    # The digest in the prompt must NOT contain trace-back timestamps.
    assert "(ep 1, 28:24–29:55)" not in stub.prompts[0]


def test_narrate_episode_writes_file(tmp_path):
    _make_episode_with_state(tmp_path, 2, "Second summary.")
    stub = StubClient("Narration two.")

    narrate_episode(output_root=str(tmp_path), ep_num=2, client=stub)

    text = open(os.path.join(tmp_path, "ep-002", "narration.txt"), encoding="utf-8").read()
    assert text.strip() == "Narration two."


def test_split_sentences_basic():
    assert _split_sentences("One. Two! Three? Four.") == ["One.", "Two!", "Three?", "Four."]


def test_chunk_sentences_sentence_bounded():
    sentences = ["Alpha.", "Beta.", "Gamma."]
    chunks = _chunk_sentences(sentences, max_chars=12)
    # "Alpha. Beta." is 12 chars, fits; "Gamma." alone.
    assert chunks == ["Alpha. Beta.", "Gamma."]


def test_chunk_sentences_never_splits_sentence():
    sentences = ["Alpha.", "Beta."]
    chunks = _chunk_sentences(sentences, max_chars=6)
    # No chunk exceeds a single sentence, and no sentence is split.
    assert chunks == ["Alpha.", "Beta."]


def test_chunk_sentences_single_long_sentence_stays_whole():
    long_sentence = "A" * 100 + "."
    chunks = _chunk_sentences([long_sentence], max_chars=50)
    assert chunks == [long_sentence]


def test_existing_narration_reuses_written_text(tmp_path):
    ep_dir = os.path.join(tmp_path, "ep-005")
    os.makedirs(ep_dir, exist_ok=True)
    with open(os.path.join(ep_dir, "narration.txt"), "w", encoding="utf-8") as f:
        f.write("Existing narration.\n")
    assert _existing_narration(str(tmp_path), 5) == "Existing narration."


def test_existing_narration_empty_when_absent(tmp_path):
    assert _existing_narration(str(tmp_path), 5) == ""


def test_mp3_path_location(tmp_path):
    assert _mp3_path(str(tmp_path), 7) == os.path.join(tmp_path, "ep-007", "narration.mp3")


def test_parse_chunk_range():
    assert _parse_chunk_range("24-40") == (24, 40)
    assert _parse_chunk_range("24") == (24, 24)
    assert _parse_chunk_range(" 24 - 40 ") == (24, 40)


def test_parse_chunk_range_invalid():
    with pytest.raises(ValueError):
        _parse_chunk_range("40-24")
    with pytest.raises(ValueError):
        _parse_chunk_range("24-40-50")
    with pytest.raises(ValueError):
        _parse_chunk_range("abc")


def test_chunk_range_must_fall_within_narrate_range(tmp_path):
    # eps 25,30 in-range; 31,40 on disk but outside the narrate range 24-30.
    for n in (25, 30, 31, 40):
        _make_episode_with_state(tmp_path, n, "S.")
    # Guard fires before any LLM/TTS work — chunk range must be within start/end.
    with pytest.raises(RuntimeError, match="not fully covered"):
        build_corpus_narrations(
            output_root=str(tmp_path), start=24, end=30, chunk_range="25-40"
        )
