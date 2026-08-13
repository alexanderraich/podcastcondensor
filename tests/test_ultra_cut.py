"""Tests for ultra_cut — chunking, insight mining parse, coalesce dedup (no cap)."""
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pytest

from podcastcondensor.ultra_cut import (
    build_chunks,
    build_ultra_cut,
    collect_insights,
    coalesce_insights,
    dedupe_insights,
    _build_mine_prompt,
    _load_insights_cache,
    _parse_coalesce_response,
    _parse_mine_response,
    _save_insights_cache,
    mine_all_chunks,
    narrate_ultra_cut,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class FakeClient:
    """DeepSeekClient stub: returns the next canned response per call."""

    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def generate(self, prompt, **kwargs):
        self.calls.append({"prompt": prompt, **kwargs})
        if not self.responses:
            raise AssertionError("no canned response left for ultra-cut call")
        return self.responses.pop(0)


def _write_global_state(tmp_path, ep):
    d = tmp_path / f"ep-{ep:03d}"
    d.mkdir(exist_ok=True)
    data = {
        "summary": f"Summary {ep}: the divine council and the gods of the nations.",
        "concepts": [{"title": "Concept A", "summary": "A summary", "segments": []}],
        "claims": [],
        "entities": [],
        "scriptural_links": [],
        "glossary": [],
    }
    (d / "global_state.json").write_text(json.dumps(data))
    (d / "source_subtitles.srt").write_text("1\n00:00:00,000 --> 00:00:01,000\ntest\n")


def _insight(**kw):
    base = {
        "id": "x", "title": "T", "claim": "C", "why_novel": "W",
        "detail": "D", "episodes": [1],
    }
    base.update(kw)
    return base


# ---------------------------------------------------------------------------
# build_chunks
# ---------------------------------------------------------------------------


class TestBuildChunks:
    def test_splits_into_fixed_chunks(self):
        assert build_chunks([1, 2, 3, 4, 5], 2) == [[1, 2], [3, 4], [5]]

    def test_empty(self):
        assert build_chunks([], 3) == []

    def test_rejects_bad_chunk_size(self):
        with pytest.raises(ValueError):
            build_chunks([1], 0)


# ---------------------------------------------------------------------------
# _build_mine_prompt
# ---------------------------------------------------------------------------


class TestBuildMinePrompt:
    def test_no_prior_section_when_empty(self):
        prompt = _build_mine_prompt([(1, "digest 1")], [])
        assert "Previously found insights:" not in prompt
        assert "Episode digests:" in prompt
        assert "digest 1" in prompt

    def test_prior_section_present_when_prior(self):
        prompt = _build_mine_prompt(
            [(1, "digest 1")], [{"id": "c00-yahweh", "title": "Yahweh", "claim": "C"}]
        )
        assert "Previously found insights:" in prompt
        assert "c00-yahweh" in prompt


# ---------------------------------------------------------------------------
# _parse_mine_response
# ---------------------------------------------------------------------------


class TestParseMineResponse:
    def test_plain_json(self):
        raw = json.dumps({
            "insights": [_insight(id="many-gods", title="Many gods", episodes=[1, 40])],
            "covered_candidates": ["old idea"],
        })
        insights, covered = _parse_mine_response(raw)
        assert len(insights) == 1
        assert insights[0]["id"] == "many-gods"
        assert insights[0]["episodes"] == [1, 40]
        assert covered == ["old idea"]

    def test_fenced_json(self):
        raw = "```json\n" + json.dumps({"insights": [_insight()]}) + "\n```"
        insights, _ = _parse_mine_response(raw)
        assert len(insights) == 1

    def test_trailing_comma_repair(self):
        raw = '{"insights": [{"id": "a", "title": "T", "claim": "C", "why_novel": "W", "detail": "D", "episodes": [1],},],}'
        insights, _ = _parse_mine_response(raw)
        assert len(insights) == 1

    def test_truncated_json_repair(self):
        raw = '{"insights": [{"id": "a", "title": "T", "claim": "C", "why_novel": "W", "detail": "D", "episodes": [1]'
        insights, _ = _parse_mine_response(raw)
        assert len(insights) == 1

    def test_malformed_entries_skipped(self):
        raw = json.dumps({"insights": [_insight(), {"id": "bad", "episodes": "nope"}]})
        insights, _ = _parse_mine_response(raw)
        assert len(insights) == 1

    def test_empty_or_garbage(self):
        assert _parse_mine_response("") == ([], [])
        assert _parse_mine_response("no json here") == ([], [])


# ---------------------------------------------------------------------------
# collect_insights — namespacing
# ---------------------------------------------------------------------------


class TestCollectInsights:
    def test_namespaced_ids(self):
        cache = {
            "chunks": {
                "0": {"insights": [_insight(id="yahweh")]},
                "1": {"insights": [_insight(id="theosis")]},
            }
        }
        insights = collect_insights(cache)
        assert [i["id"] for i in insights] == ["c00-yahweh", "c01-theosis"]


# ---------------------------------------------------------------------------
# _parse_coalesce_response
# ---------------------------------------------------------------------------


class TestParseCoalesceResponse:
    def test_plain_json(self):
        raw = json.dumps({"themes": [{
            "id": "g-yahweh", "title": "Yahweh", "claim": "C", "why_novel": "W",
            "detail": "D", "importance": 0.9, "episodes": [1, 40],
            "insight_ids": ["c00-yahweh"],
        }]})
        themes = _parse_coalesce_response(raw)
        assert len(themes) == 1
        assert themes[0]["importance"] == 0.9
        assert themes[0]["episodes"] == [1, 40]

    def test_garbage(self):
        assert _parse_coalesce_response("nothing") == []


# ---------------------------------------------------------------------------
# dedupe_insights — deterministic dedup, NO cap
# ---------------------------------------------------------------------------


class TestDedupeInsights:
    def test_drops_empty_sources_and_claims(self):
        themes = [
            _insight(insight_ids=["c00-a"], claim=""),
            _insight(insight_ids=[]),
            _insight(insight_ids=["c00-b"], claim="real"),
        ]
        out = dedupe_insights(themes)
        assert len(out) == 1
        assert out[0]["claim"] == "real"

    def test_merges_identical_titles_union_episodes_and_sources(self):
        themes = [
            _insight(title="Many Gods", claim="low", episodes=[1], insight_ids=["c00-a"], importance=0.5),
            _insight(title="many gods!", claim="HIGH", episodes=[40, 77], insight_ids=["c03-b"], importance=0.9),
        ]
        out = dedupe_insights(themes)
        assert len(out) == 1
        assert out[0]["claim"] == "HIGH"          # higher importance wins
        assert out[0]["episodes"] == [1, 40, 77]  # union
        assert out[0]["insight_ids"] == ["c00-a", "c03-b"]

    def test_ranks_by_importance(self):
        themes = [
            _insight(title="B", importance=0.4, insight_ids=["c00-b"]),
            _insight(title="A", importance=0.8, insight_ids=["c00-a"]),
        ]
        out = dedupe_insights(themes)
        assert [t["title"] for t in out] == ["A", "B"]

    def test_no_cap_keeps_all_distinct(self):
        themes = [
            _insight(title=f"Distinct {i}", importance=0.5, insight_ids=[f"c00-{i}"])
            for i in range(40)
        ]
        out = dedupe_insights(themes)
        assert len(out) == 40  # distinct themes survive — no arbitrary cap


# ---------------------------------------------------------------------------
# coalesce_insights — known-source filter + dedup
# ---------------------------------------------------------------------------


class TestCoalesceInsights:
    def test_filters_unknown_source_ids(self):
        client = FakeClient([json.dumps({"themes": [
            _insight(insight_ids=["c00-real", "c99-ghost"]),
        ]})])
        insights = [_insight(id="c00-real")]
        themes = coalesce_insights(insights, client)
        assert themes[0]["insight_ids"] == ["c00-real"]
        assert client.calls[0]["force_json"] is True


# ---------------------------------------------------------------------------
# narrate_ultra_cut
# ---------------------------------------------------------------------------


class TestNarrateUltraCut:
    def test_writes_file_and_returns_text(self, tmp_path):
        client = FakeClient(["This is the final narration prose."])
        out = str(tmp_path / "narration.txt")
        text = narrate_ultra_cut([_insight()], client, out)
        assert text == "This is the final narration prose."
        assert "This is the final narration prose" in open(out).read()

    def test_retries_once_then_succeeds(self, tmp_path):
        class Flaky:
            def __init__(self):
                self.n = 0

            def generate(self, prompt, **kw):
                self.n += 1
                if self.n == 1:
                    raise RuntimeError("boom")
                return "recovered prose"

        out = str(tmp_path / "narration.txt")
        text = narrate_ultra_cut([_insight()], Flaky(), out)
        assert text == "recovered prose"

    def test_raises_after_two_failures(self, tmp_path):
        class Dead:
            def generate(self, prompt, **kw):
                raise RuntimeError("dead")

        with pytest.raises(RuntimeError):
            narrate_ultra_cut([_insight()], Dead(), str(tmp_path / "n.txt"))


# ---------------------------------------------------------------------------
# cache roundtrip
# ---------------------------------------------------------------------------


class TestInsightsCache:
    def test_roundtrip(self, tmp_path):
        path = str(tmp_path / "cache.json")
        _save_insights_cache(path, {"chunks": {"0": {"insights": [_insight()]}}})
        loaded = _load_insights_cache(path)
        assert loaded["chunks"]["0"]["insights"][0]["title"] == "T"

    def test_missing_returns_empty(self, tmp_path):
        assert _load_insights_cache(str(tmp_path / "nope.json")) == {"chunks": {}}

    def test_corrupt_starts_fresh(self, tmp_path):
        path = tmp_path / "cache.json"
        path.write_text("not json")
        assert _load_insights_cache(str(path)) == {"chunks": {}}


# ---------------------------------------------------------------------------
# Known-state seeding — the 1-40 baseline woven into novelty mining + narration
# ---------------------------------------------------------------------------


class TestKnownStateSeeding:
    def test_mine_all_chunks_seeds_chunk_zero(self, tmp_path):
        _write_global_state(tmp_path, 41)
        _write_global_state(tmp_path, 42)
        mine_json = json.dumps({"insights": [_insight(id="fresh", title="Brand new")],
                                "covered_candidates": []})
        client = FakeClient([mine_json])
        seed = [_insight(id="c00-known", title="Yahweh is one", claim="Known claim")]
        mine_all_chunks(
            str(tmp_path), [41, 42], chunk_size=12, client=client, seed=seed,
            cache_path=str(tmp_path / "ultra_insights_041_042.json"),
        )
        prompt = client.calls[0]["prompt"]
        assert "Yahweh is one" in prompt      # baseline woven into chunk 0
        assert "Known claim" in prompt

    def test_build_ultra_cut_with_prior_baseline(self, tmp_path):
        # Prior (known) range 1-3, target range 41-43.
        for ep in (1, 2, 3, 41, 42, 43):
            _write_global_state(tmp_path, ep)
        prior_mine = json.dumps({"insights": [
            _insight(id="yahweh", title="Yahweh is one", claim="Known", episodes=[1, 2]),
        ], "covered_candidates": []})
        target_mine = json.dumps({"insights": [
            _insight(id="new-thing", title="New ground", claim="Fresh", episodes=[42]),
        ], "covered_candidates": []})
        coalesce_json = json.dumps({"themes": [
            _insight(id="g-new", title="New ground", importance=0.9,
                     episodes=[42], insight_ids=["c00-new-thing"]),
        ]})
        client = FakeClient([prior_mine, target_mine, coalesce_json,
                             "The new material narration."])

        result = build_ultra_cut(
            output_root=str(tmp_path), start=41, end=43,
            chunk_size=12, client=client, skip_tts=True,
        )
        phases = [p["phase"] for p in result["phases"]]
        assert phases == ["known_state", "mine_insights", "coalesce", "narrate"]
        assert result["phases"][0]["known_insight_count"] == 1
        # The baseline insight seeded the target-range mining prompt...
        assert "Yahweh is one" in client.calls[1]["prompt"]
        # ...and the narration prompt carries it as known state (connect, don't re-explain).
        assert "Known from earlier episodes" in client.calls[3]["prompt"]
        assert "Yahweh is one" in client.calls[3]["prompt"]
        # Known baseline cache persisted permanently.
        assert os.path.exists(result["known_cache"])

    def test_narrate_with_known_titles(self, tmp_path):
        client = FakeClient(["Narration text."])
        out = str(tmp_path / "n.txt")
        narrate_ultra_cut([_insight()], client, out, known_titles=["Yahweh is one"])
        assert "Known from earlier episodes" in client.calls[0]["prompt"]
        assert "Yahweh is one" in client.calls[0]["prompt"]


# ---------------------------------------------------------------------------
# build_ultra_cut — full pipeline (skip_tts, fake client)
# ---------------------------------------------------------------------------


class TestBuildUltraCut:
    def test_end_to_end_with_cached_pipeline(self, tmp_path):
        for ep in (40, 41, 42):
            _write_global_state(tmp_path, ep)
        mine_json = json.dumps({"insights": [
            _insight(id="many-gods", title="Many gods, one Yahweh", episodes=[40, 42]),
        ], "covered_candidates": []})
        coalesce_json = json.dumps({"themes": [
            _insight(id="g-many-gods", title="Many gods, one Yahweh", importance=0.9,
                     episodes=[40, 42], insight_ids=["c00-many-gods"]),
        ]})
        client = FakeClient([mine_json, coalesce_json, "Final narration prose here."])

        result = build_ultra_cut(
            output_root=str(tmp_path), start=40, end=42,
            chunk_size=3, client=client, skip_tts=True,
            prior_start=1, prior_end=39,  # disjoint from target → no known-state phase
        )
        assert [p["phase"] for p in result["phases"]] == ["mine_insights", "coalesce", "narrate"]
        assert result["phases"][0]["insight_count"] == 1
        assert result["phases"][1]["theme_count"] == 1
        assert result["phases"][2]["words"] == 4
        assert os.path.exists(result["insights_cache"])
        assert os.path.exists(result["outline"])
        assert os.path.exists(result["narration"])
        # outline persisted, deduped, no cap
        outline = json.load(open(result["outline"]))
        assert outline[0]["id"] == "g-many-gods"

    def test_missing_global_state_fails_loud(self, tmp_path):
        _write_global_state(tmp_path, 40)
        # ep 41 has an SRT on disk (so it's in the manifest) but NO global_state.json
        d = tmp_path / "ep-041"
        d.mkdir(exist_ok=True)
        (d / "source_subtitles.srt").write_text("1\n00:00:00,000 --> 00:00:01,000\ntest\n")
        with pytest.raises(RuntimeError, match="missing global_state.json"):
            build_ultra_cut(output_root=str(tmp_path), start=40, end=41, client=FakeClient([]))

    def test_empty_range_raises(self, tmp_path):
        with pytest.raises(RuntimeError, match="No non-Q&A episodes"):
            build_ultra_cut(output_root=str(tmp_path), start=40, end=41, client=FakeClient([]))
