"""Tests for super_cut — chunking, candidate capping, coalesce parsing."""
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from podcastcondensor.master_cut import ThemeSegment, Selection
from podcastcondensor.super_cut import (
    cap_candidate_segments,
    build_chunks,
    build_universe_from_global_states,
    _build_coalesce_prompt,
    _parse_coalesce_response,
    _dedupe_and_cap_global_themes,
    _save_candidates_cache,
    _load_candidates_cache,
    _merge_selections_cache,
    _load_selections_cache,
    _selections_from_dicts,
    analyze_brackets,
    _parse_episode_range,
    ChunkThemeRecord,
    GlobalTheme,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _seg(ep, start, end, relevance=1.0, is_intro=False, audio="a.mp3"):
    return ThemeSegment(
        theme_id="t", episode_number=ep, audio_path=audio,
        start=start, end=end, relevance_score=relevance, is_intro=is_intro,
    )


def _write_global_state(tmp_path, ep, items=None):
    d = tmp_path / f"ep-{ep:03d}"
    d.mkdir(exist_ok=True)
    data = {
        "summary": f"Summary {ep}",
        "concepts": items or [],
        "entities": [],
        "claims": [],
        "scriptural_links": [],
        "glossary": [],
    }
    (d / "global_state.json").write_text(json.dumps(data))


# ---------------------------------------------------------------------------
# cap_candidate_segments
# ---------------------------------------------------------------------------


class TestCapCandidateSegments:
    def test_under_cap_returns_all(self):
        segs = [_seg(i, 0, 10) for i in range(1, 6)]
        assert cap_candidate_segments(segs) == segs

    def test_empty(self):
        assert cap_candidate_segments([]) == []

    def test_respects_total_cap(self):
        # 60 segments spread across 15 episodes, 4 each
        segs = [_seg((i // 4) + 1, (i % 4) * 10, (i % 4) * 10 + 10)
                for i in range(60)]
        out = cap_candidate_segments(segs, max_total=20, max_per_episode=6,
                                     reserve_top=4)
        assert len(out) <= 20

    def test_promotes_episode_diversity(self):
        # 12 high-relevance segments all from ep 1 (would dominate without the
        # diversity cap), plus 1 lower-relevance segment from each of eps 2-13.
        segs = [_seg(1, i * 10, i * 10 + 10, relevance=5.0) for i in range(12)]
        segs += [_seg(ep, 0, 5, relevance=1.0) for ep in range(2, 14)]
        out = cap_candidate_segments(segs, max_total=24, max_per_episode=3,
                                     reserve_top=2)
        eps = {s.episode_number for s in out}
        assert len(eps) >= 10  # many distinct episodes represented
        assert eps <= set(range(1, 14))

    def test_preserves_input_order(self):
        segs = [_seg((i % 5) + 1, i, i + 10) for i in range(30)]
        out = cap_candidate_segments(segs, max_total=20)
        idxs = [segs.index(s) for s in out]
        assert idxs == sorted(idxs)

    def test_is_intro_survives(self):
        segs = [_seg((i % 6) + 1, i * 10, i * 10 + 10, relevance=1.0)
                for i in range(40)]
        segs[0].is_intro = True
        out = cap_candidate_segments(segs, max_total=10, reserve_top=2)
        assert any(s.is_intro for s in out)

    def test_per_episode_cap_limits_anchor_episode(self):
        # One episode dominates the pool; the cap must bound its contribution.
        segs = [_seg(1, i * 10, i * 10 + 10, relevance=5.0) for i in range(30)]
        out = cap_candidate_segments(segs, max_total=20, max_per_episode=6,
                                     reserve_top=6)
        ep1 = sum(1 for s in out if s.episode_number == 1)
        assert ep1 <= 6


# ---------------------------------------------------------------------------
# build_chunks
# ---------------------------------------------------------------------------


class TestBuildChunks:
    def test_bins_episodes_with_global_state(self, tmp_path):
        for ep in [1, 2, 3, 5, 6, 7, 8]:
            _write_global_state(tmp_path, ep)
        chunks = build_chunks(str(tmp_path), 1, 10, chunk_size=3)
        assert chunks == [[1, 2, 3], [5, 6, 7], [8]]

    def test_skips_missing_global_state(self, tmp_path):
        chunks = build_chunks(str(tmp_path), 1, 10, chunk_size=3)
        assert chunks == []

    def test_episode_without_global_state_excluded(self, tmp_path):
        _write_global_state(tmp_path, 2)
        _write_global_state(tmp_path, 4)
        chunks = build_chunks(str(tmp_path), 1, 6, chunk_size=2)
        assert chunks == [[2, 4]]


# ---------------------------------------------------------------------------
# build_universe_from_global_states
# ---------------------------------------------------------------------------


class TestMergeUniverse:
    def test_merges_and_dedups_by_id(self, tmp_path):
        _write_global_state(tmp_path, 1, items=[
            {"id": "divine-council", "title": "Divine Council",
             "summary": "heavenly court", "segments": [{"episode": 1, "start": 0, "end": 10}]},
        ])
        _write_global_state(tmp_path, 2, items=[
            {"id": "divine-council", "title": "Divine Council",
             "summary": "heavenly court", "segments": [{"episode": 2, "start": 5, "end": 15}]},
            {"id": "sheol", "title": "Sheol", "summary": "underworld",
             "segments": [{"episode": 2, "start": 20, "end": 30}]},
        ])
        data = build_universe_from_global_states(str(tmp_path), 1, 3)
        assert len(data["episode_summaries"]) == 2
        concepts = data["concepts"]
        by_id = {c["id"]: c for c in concepts}
        assert by_id["divine-council"]["episode_numbers"] == [1, 2]
        assert len(by_id["divine-council"]["segments"]) == 2
        assert by_id["sheol"]["episode_numbers"] == [2]
        assert data["metadata"]["last_built_episode"] == 2

    def test_missing_episodes_skipped(self, tmp_path):
        _write_global_state(tmp_path, 5)
        data = build_universe_from_global_states(str(tmp_path), 1, 6)
        assert [s["episode_number"] for s in data["episode_summaries"]] == [5]


# ---------------------------------------------------------------------------
# _parse_coalesce_response
# ---------------------------------------------------------------------------


class TestParseCoalesce:
    def test_plain_json(self):
        raw = json.dumps({"themes": [
            {"id": "afterlife", "title": "Afterlife", "description": "d",
             "importance": 0.9, "source_theme_ids": ["c00-afterlife", "c04-sheol"],
             "natural_intro_theme_ids": ["c00-afterlife"]},
        ], "unassigned_theme_ids": []})
        out = _parse_coalesce_response(raw)
        assert len(out) == 1
        assert out[0].id == "afterlife"
        assert out[0].source_theme_ids == ["c00-afterlife", "c04-sheol"]

    def test_fenced_json(self):
        raw = "```json\n" + json.dumps({"themes": [
            {"id": "x", "title": "X", "description": "d", "importance": 0.5,
             "source_theme_ids": []},
        ]}) + "\n```"
        assert len(_parse_coalesce_response(raw)) == 1

    def test_trailing_comma_repair(self):
        raw = ('{"themes": [{"id": "x", "title": "X", "description": "d",'
               ' "importance": 0.5, "source_theme_ids": [],},],}')
        assert len(_parse_coalesce_response(raw)) == 1

    def test_truncated_json_repair(self):
        raw = '{"themes": [{"id": "x", "title": "X", "description": "d", "importance": 0.5, "source_theme_ids": ["c00-'
        assert len(_parse_coalesce_response(raw)) == 1

    def test_malformed_entries_skipped(self):
        raw = json.dumps({"themes": [
            {"id": "ok", "title": "Ok", "description": "d", "importance": 0.5,
             "source_theme_ids": []},
            {"id": "bad", "title": "Bad", "description": "d",
             "importance": "not-a-number", "source_theme_ids": []},
        ]})
        out = _parse_coalesce_response(raw)
        assert [t.id for t in out] == ["ok"]

    def test_empty_or_garbage(self):
        assert _parse_coalesce_response("") == []
        assert _parse_coalesce_response("no json here") == []


# ---------------------------------------------------------------------------
# _dedupe_and_cap_global_themes
# ---------------------------------------------------------------------------


class TestDedupeCapGlobalThemes:
    def _gt(self, tid, title, imp, sources):
        return GlobalTheme(id=tid, title=title, description="d",
                           importance=imp, source_theme_ids=sources)

    def test_caps_to_max_themes(self):
        gts = [self._gt(f"t{i}", f"Theme {i}", 0.5 + i * 0.01,
                        [f"c00-{i}"]) for i in range(40)]
        out = _dedupe_and_cap_global_themes(gts, max_themes=25)
        assert len(out) == 25
        # Highest-importance themes survive the cap
        assert out[0].id == "t39"

    def test_merges_identical_source_duplicates(self):
        a = self._gt("a", "Afterlife", 0.9, ["c00-x", "c02-x"])
        b = self._gt("b", "Afterlife duplicate", 0.8, ["c00-x", "c02-x"])
        out = _dedupe_and_cap_global_themes([a, b], max_themes=25)
        assert len(out) == 1
        assert out[0].importance == 0.9  # kept the higher-importance one

    def test_merges_identical_titles(self):
        a = self._gt("a", "Theology of Angels", 0.7, ["c00-a"])
        b = self._gt("b", "Theology of Angels", 0.6, ["c03-b"])
        out = _dedupe_and_cap_global_themes([a, b], max_themes=25)
        assert len(out) == 1
        assert out[0].source_theme_ids == ["c00-a", "c03-b"]

    def test_drops_empty_sources(self):
        gts = [self._gt("a", "Empty", 0.9, []), self._gt("b", "Real", 0.5, ["c00-b"])]
        out = _dedupe_and_cap_global_themes(gts, max_themes=25)
        assert len(out) == 1
        assert out[0].id == "b"


# ---------------------------------------------------------------------------
# Candidates + selections caches
# ---------------------------------------------------------------------------


class TestCaches:
    def test_candidates_cache_roundtrip(self, tmp_path):
        from podcastcondensor.master_cut import ThemeWithSegments
        from podcastcondensor.theme_extraction import Theme
        gt = GlobalTheme(id="afterlife", title="Afterlife", description="d",
                         importance=0.9, source_theme_ids=["c00-x"])
        gt.episode_numbers = [42, 55]
        gt.candidate_count = 2
        tws = ThemeWithSegments(theme=Theme(id="afterlife", title="Afterlife",
                                            description="d"),
                                segments=[
            ThemeSegment(theme_id="afterlife", episode_number=42,
                         audio_path="a.mp3", start=100, end=200),
            ThemeSegment(theme_id="afterlife", episode_number=55,
                         audio_path="a.mp3", start=300, end=400),
        ])
        path = str(tmp_path / "cands.json")
        _save_candidates_cache(path, [gt], [tws])
        loaded = _load_candidates_cache(path)
        assert "afterlife" in loaded
        assert loaded["afterlife"]["episode_numbers"] == [42, 55]
        assert loaded["afterlife"]["candidate_count"] == 2

    def test_candidates_cache_missing_returns_empty(self, tmp_path):
        assert _load_candidates_cache(str(tmp_path / "nope.json")) == {}

    def test_selections_cache_merge_and_reload(self, tmp_path):
        path = str(tmp_path / "sels.json")
        sel = Selection(
            segment=ThemeSegment(theme_id="christology", episode_number=5,
                                 audio_path="a.mp3", start=100, end=200),
            theme_title="Christology", theme_id="christology", beep_before="none",
        )
        _merge_selections_cache(path, {"christology": [sel]})
        loaded = _load_selections_cache(path)
        assert "christology" in loaded
        assert loaded["christology"][0]["episode"] == 5
        # A second theme appends without losing the first
        sel2 = Selection(
            segment=ThemeSegment(theme_id="angelology", episode_number=8,
                                 audio_path="b.mp3", start=10, end=50),
            theme_title="Angelology", theme_id="angelology", beep_before="none",
        )
        _merge_selections_cache(path, {"angelology": [sel2]})
        loaded = _load_selections_cache(path)
        assert set(loaded.keys()) == {"christology", "angelology"}

    def test_selections_from_dicts_reconstructs(self):
        sels = _selections_from_dicts([
            {"theme_id": "christology", "theme_title": "Christology",
             "episode": 5, "audio_path": "a.mp3", "start": 100, "end": 200,
             "text_preview": "", "beep_before": "none"},
        ])
        assert len(sels) == 1
        assert sels[0].segment.episode_number == 5
        assert sels[0].theme_id == "christology"
        assert sels[0].segment.duration == 100.0


# ---------------------------------------------------------------------------
# Bracket analysis
# ---------------------------------------------------------------------------


class TestBrackets:
    def test_parse_episode_range(self):
        assert _parse_episode_range("40-80") == set(range(40, 81))
        assert _parse_episode_range("42") == {42}

    def test_analyze_brackets_ranks_by_in_bracket_candidates(self):
        candidates = {
            "christology": {
                "candidates": [
                    {"episode": 42, "duration": 100},
                    {"episode": 55, "duration": 100},
                ]
            },
            "nicene": {
                "candidates": [
                    {"episode": 42, "duration": 300},
                    {"episode": 43, "duration": 300},
                ]
            },
        }
        themes_info = [
            {"id": "christology", "title": "Christology", "importance": 0.9},
            {"id": "nicene", "title": "Nicaea", "importance": 0.2},
        ]
        results = analyze_brackets(candidates, themes_info, [("40-80", set(range(40, 81)))], top_n=5)
        bracket = results[0]
        assert bracket["bracket"] == "40-80"
        assert bracket["top"][0]["theme_id"] == "nicene"  # 2 candidates in bracket
        assert bracket["top"][1]["theme_id"] == "christology"

    def test_analyze_brackets_excludes_out_of_range(self):
        candidates = {
            "t": {"candidates": [{"episode": 90, "duration": 100}]},
        }
        themes_info = [{"id": "t", "title": "T", "importance": 0.5}]
        results = analyze_brackets(candidates, themes_info, [("40-80", set(range(40, 81)))])
        assert results[0]["top"] == []  # no candidates in 40-80


class TestBuildCoalescePrompt:
    def test_payload_contains_namespaced_ids_and_episodes(self):
        records = [
            ChunkThemeRecord(chunk_index=0, theme_id="c00-angelology",
                             title="Angelology", description="d", importance=0.8,
                             related_item_ids=["a1"], episode_span=(1, 42)),
            ChunkThemeRecord(chunk_index=3, theme_id="c03-celestial",
                             title="Celestial Beings", description="d", importance=0.6,
                             related_item_ids=["b1"], episode_span=(50, 88)),
        ]
        chunk_episodes = {0: [1, 2, 3, 5, 6], 3: [37, 38, 40, 42, 44]}
        prompt = _build_coalesce_prompt(records, chunk_episodes)
        payload_text = prompt.rsplit("\n\n", 1)[1]
        payload = json.loads(payload_text)
        assert len(payload["chunks"]) == 2
        c0 = payload["chunks"][0]
        assert c0["episodes"] == [1, 2, 3, 5, 6]
        assert c0["themes"][0]["id"] == "c00-angelology"
        assert c0["themes"][0]["episode_span"] == [1, 42]
