"""Tests for deep_narration — chunking, arc caching, prompt assembly, resumability."""
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from podcastcondensor.deep_narration import (
    _ARC_MAX_WORDS,
    _MAX_CHUNK_WORDS,
    _build_arc_prompt,
    _carry_forward_text,
    _load_arc_cache,
    _render_arc,
    _save_arc_cache,
    build_arc,
    build_deep_narration,
    chunk_transcript,
    identify_arc_chunk,
    narrate_deep,
    paragraphs,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _StubClient:
    """Records prompts; returns queued responses (an Exception is raised)."""

    def __init__(self, responses):
        self.responses = list(responses)
        self.prompts = []

    def generate(self, prompt, **kwargs):
        self.prompts.append(prompt)
        if not self.responses:
            raise AssertionError("stub client ran out of responses")
        r = self.responses.pop(0)
        if isinstance(r, Exception):
            raise r
        return r


def _entries(n_cues, words_per_cue=10):
    return [
        {"index": i + 1, "start": float(i * 5), "end": float(i * 5 + 4),
         "text": " ".join(["word"] * words_per_cue)}
        for i in range(n_cues)
    ]


def _write_srt(path, n_cues=20, words_per_cue=10):
    """Write an SRT whose cues are all distinct.

    The text must vary per cue: ``clean_entries`` drops whisper echoes and
    repeats, so identical consecutive cues would collapse to a single entry
    and every chunk-count assertion would be measuring the dedup, not the
    chunker.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for i in range(n_cues):
            h, rem = divmod(i * 5, 3600)
            m, s = divmod(rem, 60)
            f.write(f"{i + 1}\n")
            f.write(f"{h:02d}:{m:02d}:{s:02d},000 --> "
                    f"{h:02d}:{m:02d}:{s:02d},900\n")
            f.write(f"cue{i} " + " ".join(["word"] * (words_per_cue - 1)) + "\n\n")


def _episode(tmp_path, ep=145, n_cues=20, words_per_cue=10):
    _write_srt(os.path.join(str(tmp_path), f"ep-{ep:03d}",
                            "source_subtitles.srt"), n_cues, words_per_cue)
    return str(tmp_path)


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------


def test_chunk_transcript_respects_budget_and_loses_nothing():
    entries = _entries(50, words_per_cue=10)  # 500 words
    chunks = chunk_transcript(entries, max_words=120)
    assert len(chunks) == 5
    assert [len(c) for c in chunks] == [12, 12, 12, 12, 2]
    assert sum(len(c) for c in chunks) == 50
    for c in chunks[:-1]:
        assert sum(len(e["text"].split()) for e in c) >= 120


def test_chunk_transcript_single_chunk_when_under_budget():
    assert len(chunk_transcript(_entries(5), max_words=10000)) == 1


def test_chunk_transcript_empty():
    assert chunk_transcript([]) == []


def test_chunk_transcript_skips_empty_text():
    entries = _entries(3) + [{"index": 9, "start": 1.0, "end": 2.0, "text": "   "}]
    chunks = chunk_transcript(entries, max_words=10000)
    assert sum(len(c) for c in chunks) == 3


def test_default_chunk_budget_is_the_documented_one():
    assert _MAX_CHUNK_WORDS == 10000
    assert _ARC_MAX_WORDS == 2000


# ---------------------------------------------------------------------------
# Paragraphs
# ---------------------------------------------------------------------------


def test_paragraphs_are_word_bounded_and_timestamped():
    paras = paragraphs(_entries(40, words_per_cue=10), words_per_para=150)
    assert len(paras) > 1
    assert all(p.startswith("[") for p in paras)
    # first paragraph covers 150 words -> 15 cues -> starts at cue 0
    assert paras[0].startswith("[0:00]")
    assert len(paras[0].split()) == 151  # 150 words + the timestamp token


def test_paragraphs_timestamp_is_the_first_cue_start():
    paras = paragraphs(_entries(30, words_per_cue=10), words_per_para=100)
    # second paragraph starts at cue 10 -> 50s -> "0:50"
    assert paras[1].startswith("[0:50]")


def test_paragraphs_empty():
    assert paragraphs([]) == []


# ---------------------------------------------------------------------------
# Carry-forward bounding
# ---------------------------------------------------------------------------


def test_carry_forward_is_bounded_to_last_chunks():
    chunks = [{"carry_forward": f"note{i}"} for i in range(10)]
    text = _carry_forward_text(chunks)
    assert text.splitlines() == ["note6", "note7", "note8", "note9"]


def test_carry_forward_skips_empty_notes():
    assert _carry_forward_text([{"carry_forward": ""}, {"carry_forward": "a"}]) == "a"
    assert _carry_forward_text([]) == ""


# ---------------------------------------------------------------------------
# Arc rendering
# ---------------------------------------------------------------------------


def _mv(**kw):
    base = {"label": "L", "substance": "high", "develops": "D",
            "restates": "", "backrefs": "", "qa": False, "qa_merit": "",
            "key_points": [], "scripture": []}
    base.update(kw)
    return base


def test_render_arc_includes_all_populated_fields():
    text = _render_arc({"movements": [_mv(
        restates="already said", backrefs="ep 12", qa=True, qa_merit="novel",
        key_points=["k1", "k2"], scripture=["Psalm 82"],
    )]}, 145, "Title")
    assert "# Episode 145 — Title" in text
    assert "Restates earlier in this episode: already said" in text
    assert "Connects back to earlier episodes: ep 12" in text
    assert "Q&A — worth keeping: novel" in text
    assert "- k1" in text and "- k2" in text
    assert "Scripture: Psalm 82" in text


def test_render_arc_omits_empty_optional_fields():
    text = _render_arc({"movements": [_mv()]}, 1, "T")
    assert "Restates" not in text
    assert "Connects back" not in text
    assert "Q&A" not in text
    assert "Key points" not in text


def test_render_arc_marks_redundant_qa():
    text = _render_arc({"movements": [_mv(qa=True, qa_merit="")]}, 1, "T")
    assert "Q&A — redundant" in text


def test_render_arc_raises_without_movements():
    import pytest
    with pytest.raises(RuntimeError, match="no movements"):
        _render_arc({"movements": []}, 1, "T")


# ---------------------------------------------------------------------------
# Prompt assembly
# ---------------------------------------------------------------------------


def test_arc_prompt_includes_carry_forward_only_when_present():
    # NB: the prompt template itself mentions "Previously established…" in its
    # instructions, so assert on the injected note, not on the phrase.
    with_cf = _build_arc_prompt("TEXT", 145, "Title", 1, 3, "MARKER-prior")
    assert "Previously established in this episode:\nMARKER-prior" in with_cf
    assert "Part 2 of 3." in with_cf
    without = _build_arc_prompt("TEXT", 145, "Title", 0, 3, "")
    assert "MARKER-prior" not in without
    assert "Part 1 of 3." in without
    assert with_cf.endswith("TEXT")


# ---------------------------------------------------------------------------
# Arc cache
# ---------------------------------------------------------------------------


def test_arc_cache_round_trip(tmp_path):
    path = os.path.join(str(tmp_path), "arc_deep.json")
    chunks = [{"index": 0, "movements": [_mv()], "carry_forward": "x"}]
    _save_arc_cache(path, 145, "Title", 10000, chunks)

    loaded = _load_arc_cache(path, 10000)
    assert loaded == chunks
    on_disk = json.load(open(path, encoding="utf-8"))
    assert on_disk["episode"] == 145
    assert on_disk["chunk_words"] == 10000
    assert len(on_disk["movements"]) == 1  # flattened for convenience


def test_arc_cache_missing_file_is_empty(tmp_path):
    assert _load_arc_cache(os.path.join(str(tmp_path), "nope.json"), 10000) == []


def test_arc_cache_discarded_when_chunk_size_changed(tmp_path):
    path = os.path.join(str(tmp_path), "arc_deep.json")
    _save_arc_cache(path, 145, "T", 10000, [{"index": 0, "movements": []}])
    assert _load_arc_cache(path, 5000) == []


def test_arc_cache_unreadable_json_is_empty(tmp_path):
    path = os.path.join(str(tmp_path), "arc_deep.json")
    open(path, "w").write("{not json")
    assert _load_arc_cache(path, 10000) == []


# ---------------------------------------------------------------------------
# Stage 1 — calls, retry, caching
# ---------------------------------------------------------------------------


def test_identify_arc_chunk_retries_then_succeeds():
    client = _StubClient([RuntimeError("boom"), '{"movements": []}'])
    assert identify_arc_chunk("t", 1, "T", 0, 1, "", client) == {"movements": []}
    assert len(client.prompts) == 2


def test_identify_arc_chunk_raises_after_two_failures():
    import pytest
    client = _StubClient([RuntimeError("a"), RuntimeError("b")])
    with pytest.raises(RuntimeError, match="Arc identification failed"):
        identify_arc_chunk("t", 7, "T", 0, 1, "", client)


def test_identify_arc_chunk_raises_on_unparseable_json():
    import pytest
    client = _StubClient(["not json at all", "still not json"])
    with pytest.raises(RuntimeError, match="Arc identification failed"):
        identify_arc_chunk("t", 7, "T", 0, 1, "", client)


def test_build_arc_missing_transcript_raises(tmp_path):
    import pytest
    with pytest.raises(FileNotFoundError):
        build_arc(str(tmp_path), 999, "T", client=_StubClient([]))


def test_build_arc_calls_once_per_chunk_and_threads_carry_forward(tmp_path):
    root = _episode(tmp_path, ep=145, n_cues=30, words_per_cue=10)  # 300 words
    client = _StubClient([
        '{"movements": [{"label": "A"}], "carry_forward": "first note"}',
        '{"movements": [{"label": "B"}], "carry_forward": "second note"}',
        '{"movements": [{"label": "C"}], "carry_forward": "third note"}',
    ])
    arc = build_arc(root, 145, "Title", client=client, chunk_words=120)

    assert len(arc["chunks"]) == 3
    assert [m["label"] for m in arc["movements"]] == ["A", "B", "C"]
    # chunk 0 gets no carry-forward; later chunks see the earlier notes.
    # The prompt *template* always mentions the phrase, so the negative
    # assertion targets the injected section marker.
    assert "Previously established in this episode:\n" not in client.prompts[0]
    assert "first note" in client.prompts[1]
    assert "first note" in client.prompts[2]
    assert "second note" in client.prompts[2]
    # cache written after every chunk
    assert len(_load_arc_cache(
        os.path.join(root, "ep-145", "ep-145_arc_deep.json"), 120)) == 3


def test_build_arc_reuses_cached_chunks(tmp_path):
    root = _episode(tmp_path, ep=145, n_cues=30, words_per_cue=10)
    client1 = _StubClient([
        '{"movements": [], "carry_forward": "n1"}',
        '{"movements": [], "carry_forward": "n2"}',
        '{"movements": [], "carry_forward": "n3"}',
    ])
    build_arc(root, 145, "T", client=client1, chunk_words=120)
    assert len(client1.prompts) == 3

    # Second run: nothing left to do, 0 calls.
    client2 = _StubClient([])
    arc = build_arc(root, 145, "T", client=client2, chunk_words=120)
    assert client2.prompts == []
    assert len(arc["chunks"]) == 3

    # force re-runs all three.
    client3 = _StubClient([
        '{"movements": [], "carry_forward": "n1"}',
        '{"movements": [], "carry_forward": "n2"}',
        '{"movements": [], "carry_forward": "n3"}',
    ])
    build_arc(root, 145, "T", client=client3, chunk_words=120, force=True)
    assert len(client3.prompts) == 3


def test_build_arc_resumes_only_the_failed_chunk(tmp_path):
    root = _episode(tmp_path, ep=145, n_cues=30, words_per_cue=10)
    # chunk 1 fails twice -> whole first run raises, chunks 0 is cached
    client1 = _StubClient([
        '{"movements": [], "carry_forward": "n1"}',
        RuntimeError("boom"), RuntimeError("boom"),
    ])
    import pytest
    with pytest.raises(RuntimeError):
        build_arc(root, 145, "T", client=client1, chunk_words=120)

    cached = _load_arc_cache(
        os.path.join(root, "ep-145", "ep-145_arc_deep.json"), 120)
    assert [c["index"] for c in cached] == [0]

    # second run only spends the two missing chunks
    client2 = _StubClient([
        '{"movements": [], "carry_forward": "n2"}',
        '{"movements": [], "carry_forward": "n3"}',
    ])
    arc = build_arc(root, 145, "T", client=client2, chunk_words=120)
    assert len(client2.prompts) == 2
    assert len(arc["chunks"]) == 3


# ---------------------------------------------------------------------------
# Stage 2
# ---------------------------------------------------------------------------


def test_narrate_deep_retries_then_returns():
    client = _StubClient([RuntimeError("x"), "The narration."])
    out = narrate_deep({"movements": [_mv()]}, 1, "T", client)
    assert out == "The narration."
    assert len(client.prompts) == 2


def test_narrate_deep_raises_on_empty_response():
    import pytest
    client = _StubClient(["", ""])
    with pytest.raises(RuntimeError, match="Narration failed"):
        narrate_deep({"movements": [_mv()]}, 1, "T", client)


def test_narrate_deep_never_sends_the_transcript():
    client = _StubClient(["prose"])
    narrate_deep({"movements": [_mv(key_points=["unique-marker-xyz"])]},
                 1, "T", client)
    assert "unique-marker-xyz" in client.prompts[0]
    assert "TRANSCRIPT" not in client.prompts[0]


# ---------------------------------------------------------------------------
# Orchestration / resumability
# ---------------------------------------------------------------------------


def test_build_deep_narration_skips_when_mp3_exists(tmp_path):
    root = _episode(tmp_path, ep=145)
    mp3 = os.path.join(root, "ep-145", "ep-145_narration_deep.mp3")
    open(mp3, "wb").close()
    result = build_deep_narration(root, 145, client=_StubClient([]))
    assert result["skipped"] is True
    assert result["mp3"] == mp3


def test_build_deep_narration_missing_transcript_raises(tmp_path):
    import pytest
    with pytest.raises(FileNotFoundError):
        build_deep_narration(str(tmp_path), 999, client=_StubClient([]))


def test_build_deep_narration_end_to_end_without_tts(tmp_path):
    root = _episode(tmp_path, ep=145, n_cues=30, words_per_cue=10)
    client = _StubClient([
        '{"movements": [{"label": "A", "substance": "high", "develops": "D"}],'
        ' "carry_forward": "n1"}',
        "The full narration prose.",
    ])
    result = build_deep_narration(root, 145, client=client, skip_tts=True,
                                  chunk_words=10000)

    assert result["words"] == 4
    assert result["mp3"] == ""
    assert result["chunks"] == 1
    assert result["movements"] == 1
    text = open(os.path.join(root, "ep-145", "ep-145_narration_deep.txt"),
                encoding="utf-8").read()
    assert text.strip() == "The full narration prose."


def test_build_deep_narration_reuses_existing_text_and_arc(tmp_path):
    root = _episode(tmp_path, ep=145, n_cues=30, words_per_cue=10)
    build_deep_narration(
        root, 145, skip_tts=True, chunk_words=10000,
        client=_StubClient([
            '{"movements": [{"label": "A"}], "carry_forward": ""}',
            "First prose.",
        ]),
    )
    # Second run: arc cached, narration text on disk -> 0 calls.
    client = _StubClient([])
    result = build_deep_narration(root, 145, skip_tts=True,
                                  chunk_words=10000, client=client)
    assert client.prompts == []
    assert result["words"] == 2
