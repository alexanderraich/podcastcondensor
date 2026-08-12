"""Tests for summary_doc — episode scanning, Q&A exclusion, document build.

Covers the ``build-summary-doc`` utility: the 125 non-Q&A episode scan
(ep-18 excluded), header format with the ID3-title fallback, canonical
ordering, and summary assembly from per-episode ``global_state.json``.
"""
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from podcastcondensor.summary_doc import (
    _episode_dir_numbers,
    build_summary_doc,
)

SRT_TEMPLATE = (
    "1\n00:00:00,000 --> 00:00:02,000\nFirst sentence here.\n"
    "2\n00:00:02,000 --> 00:00:04,000\nSecond sentence, still one thought "
    "because this continues.\n"
)


def _make_episode(root, ep_num, srt_text=SRT_TEMPLATE):
    ep_dir = os.path.join(root, f"ep-{ep_num:03d}")
    os.makedirs(ep_dir, exist_ok=True)
    srt_path = os.path.join(ep_dir, "source_subtitles.srt")
    with open(srt_path, "w", encoding="utf-8") as f:
        f.write(srt_text)
    return srt_path


def _make_global_state(root, ep_num, summary, **categories):
    """Write a global_state with summary + category items.

    ``categories`` maps a category key (concepts/claims/entities/
    scriptural_links/glossary) to a list of dicts (the raw items).
    """
    ep_dir = os.path.join(root, f"ep-{ep_num:03d}")
    os.makedirs(ep_dir, exist_ok=True)
    data = {"summary": summary, **categories}
    path = os.path.join(ep_dir, "global_state.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f)
    return path


def test_episode_dir_numbers_excludes_qa_and_sorts(tmp_path):
    _make_episode(tmp_path, 18)  # Q&A special — must be excluded
    _make_episode(tmp_path, 2)
    _make_episode(tmp_path, 100)
    _make_episode(tmp_path, 1)
    assert _episode_dir_numbers(str(tmp_path)) == [1, 2, 100]


def test_episode_dir_numbers_no_srts(tmp_path):
    assert _episode_dir_numbers(str(tmp_path)) == []


def test_episode_title_falls_back_without_mp3(tmp_path):
    _make_episode(tmp_path, 3)
    _make_global_state(tmp_path, 3, "A summary.")
    text = open(build_summary_doc(output_root=str(tmp_path)), encoding="utf-8").read()
    assert "# Episode 3 — Episode 3" in text  # no MP3 ID3 tag → fallback title


def test_build_summary_doc_uses_summaries_and_excludes_qa(tmp_path):
    _make_episode(tmp_path, 18)  # Q&A — excluded
    _make_episode(tmp_path, 2)
    _make_episode(tmp_path, 1)
    _make_global_state(tmp_path, 1, "First episode's summary.")
    _make_global_state(tmp_path, 2, "Second episode's summary.")

    text = open(build_summary_doc(output_root=str(tmp_path)), encoding="utf-8").read()

    assert "First episode's summary." in text
    assert "Second episode's summary." in text
    assert "Episode 18" not in text
    headers = [l for l in text.splitlines() if l.startswith("# Episode ")]
    assert [h.split()[2] for h in headers] == ["1", "2"]  # ordered


def test_build_summary_doc_missing_global_state_raises(tmp_path):
    _make_episode(tmp_path, 5)  # SRT present, but no global_state.json
    try:
        build_summary_doc(output_root=str(tmp_path))
        assert False, "expected FileNotFoundError"
    except FileNotFoundError:
        pass


def test_build_summary_doc_empty_summary_raises(tmp_path):
    _make_episode(tmp_path, 5)
    _make_global_state(tmp_path, 5, "")
    try:
        build_summary_doc(output_root=str(tmp_path))
        assert False, "expected RuntimeError"
    except RuntimeError:
        pass


def test_build_summary_doc_renders_full_digest(tmp_path):
    _make_episode(tmp_path, 9)
    _make_global_state(
        tmp_path, 9, "A summary here.",
        concepts=[{"title": "Divine Council", "summary": "The divine assembly."}],
        claims=[{
            "text": "The gods of the nations are members of the council.",
            "segments": [
                {"episode": 1, "start": 1704.0, "end": 1795.0},
                {"episode": 1, "start": 2132.34, "end": 2222.7},
            ],
        }],
        entities=[{"title": "Yahweh", "category": "deity", "summary": "The Lord."}],
        scriptural_links=[{"reference": "Ps 82", "summary": "The council scene."}],
        glossary=[{"term": "Monolatry", "definition": "One-god worship."}],
    )

    text = open(build_summary_doc(output_root=str(tmp_path)), encoding="utf-8").read()

    assert "A summary here." in text
    assert "## Concepts" in text and "- Divine Council — The divine assembly." in text
    # Items WITHOUT segments get no suffix; items with them get timestamps.
    assert "- The gods of the nations are members of the council. (ep 1, 28:24–29:55; ep 1, 35:32–37:03)" in text
    assert "## Entities" in text and "- Yahweh (deity) — The Lord." in text
    assert "## Scripture" in text and "- Ps 82 — The council scene." in text
    assert "## Glossary" in text and "- Monolatry — One-god worship." in text


def test_build_summary_doc_omits_empty_categories(tmp_path):
    _make_episode(tmp_path, 10)
    _make_global_state(tmp_path, 10, "Only a summary here.")

    text = open(build_summary_doc(output_root=str(tmp_path)), encoding="utf-8").read()

    assert "## Concepts" not in text
    assert "## Claims" not in text
    assert "## Glossary" not in text
