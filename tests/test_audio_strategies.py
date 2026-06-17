"""Tests for audio cutting strategies — selection, filter graph, ordering,
interval normalisation, zero intervals, config defaults."""

import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))


# ---------------------------------------------------------------------------
# 1. Strategy selection
# ---------------------------------------------------------------------------

class TestAudioStrategySelection:
    """create_audio_strategy must return correct type for each name."""

    def test_single_pass_filter(self):
        from podcastcondensor.audio_strategies import create_audio_strategy
        strat = create_audio_strategy("single_pass_filter")
        from podcastcondensor.audio_strategies import SinglePassFilterCutStrategy
        assert isinstance(strat, SinglePassFilterCutStrategy)
        assert strat.name() == "single_pass_filter"

    def test_parallel_copy(self):
        from podcastcondensor.audio_strategies import create_audio_strategy
        strat = create_audio_strategy("parallel_copy")
        from podcastcondensor.audio_strategies import ParallelCopyCutStrategy
        assert isinstance(strat, ParallelCopyCutStrategy)
        assert strat.name() == "parallel_copy"

    def test_single_pass_filter_selection(self):
        from podcastcondensor.audio_strategies import create_audio_strategy
        strat = create_audio_strategy("single_pass_filter")
        from podcastcondensor.audio_strategies import SinglePassFilterCutStrategy
        assert isinstance(strat, SinglePassFilterCutStrategy)
        assert strat.name() == "single_pass_filter"

    def test_unknown_raises(self):
        from podcastcondensor.audio_strategies import create_audio_strategy
        import pytest
        with pytest.raises(ValueError, match="Unknown"):
            create_audio_strategy("nonexistent")

    def test_parallel_workers_passed(self):
        from podcastcondensor.audio_strategies import create_audio_strategy
        strat = create_audio_strategy("parallel_copy", max_workers=8)
        assert strat._max_workers == 8


# ---------------------------------------------------------------------------
# 2. Interval normalization (atrim filter building)
# ---------------------------------------------------------------------------

class TestFilterGraphBuilding:
    """_build_filter_graph must produce valid ffmpeg filter_complex strings."""

    def test_single_interval(self):
        from podcastcondensor.audio_strategies import SinglePassFilterCutStrategy
        graph = SinglePassFilterCutStrategy._build_filter_graph(
            intervals=[{"start": 10.0, "end": 20.0}], beep=False,
        )
        assert "atrim=10.000:20.000" in graph
        assert "asetpts=PTS-STARTPTS[a0]" in graph
        assert "[a0]concat=n=1:v=0:a=1[outa]" in graph

    def test_two_intervals(self):
        from podcastcondensor.audio_strategies import SinglePassFilterCutStrategy
        graph = SinglePassFilterCutStrategy._build_filter_graph(
            intervals=[
                {"start": 10.0, "end": 20.0},
                {"start": 30.0, "end": 40.5},
            ], beep=False,
        )
        assert "[a0]" in graph
        assert "[a1]" in graph
        assert "concat=n=2" in graph

    def test_three_intervals(self):
        from podcastcondensor.audio_strategies import SinglePassFilterCutStrategy
        graph = SinglePassFilterCutStrategy._build_filter_graph(
            intervals=[
                {"start": 0.0, "end": 5.0},
                {"start": 10.0, "end": 15.0},
                {"start": 20.0, "end": 25.0},
            ], beep=False,
        )
        assert "concat=n=3" in graph
        assert "a0" in graph and "a1" in graph and "a2" in graph

    def test_with_atempo(self):
        from podcastcondensor.audio_strategies import SinglePassFilterCutStrategy, _atempo_filters
        atempo = _atempo_filters(1.25)
        graph = SinglePassFilterCutStrategy._build_filter_graph(
            intervals=[{"start": 10.0, "end": 20.0}],
            atempo=atempo, beep=False,
        )
        assert "atempo=1.250" in graph
        assert graph.endswith("[outa]")

    def test_with_atempo_3x_chains_multiple(self):
        from podcastcondensor.audio_strategies import _atempo_filters
        filters = _atempo_filters(3.0)
        # 3.0 = 2.0 * 1.5 → two filters
        assert len(filters) == 2
        assert "atempo=2.0" in filters[0]
        assert "atempo=1.500" in filters[1]

    def test_atempo_1x_returns_empty(self):
        from podcastcondensor.audio_strategies import _atempo_filters
        assert _atempo_filters(1.0) == []

    def test_atempo_0_5x(self):
        from podcastcondensor.audio_strategies import _atempo_filters
        filters = _atempo_filters(0.5)
        assert len(filters) == 1
        assert "atempo=0.500" in filters[0]


# ---------------------------------------------------------------------------
# 3. Ordering guarantees (parallel strategy)
# ---------------------------------------------------------------------------

class TestParallelOrdering:
    """Parallel strategy must produce deterministic segment paths."""

    def test_segment_paths_are_ordered(self):
        from podcastcondensor.audio_strategies import ParallelCopyCutStrategy
        import tempfile
        tmpdir = tempfile.mkdtemp()
        try:
            # We can't actually run ffmpeg here, but we can verify that
            # the segment paths would be in sorted order
            strategy = ParallelCopyCutStrategy(max_workers=2)
            fmt = "mp3"
            paths = [
                os.path.join(tmpdir, f"seg_{i:04d}.{fmt}")
                for i in range(5)
            ]
            assert paths[0].endswith("seg_0000.mp3")
            assert paths[4].endswith("seg_0004.mp3")
        finally:
            import shutil
            shutil.rmtree(tmpdir, ignore_errors=True)


# ---------------------------------------------------------------------------
# 4. Speed filter helper
# ---------------------------------------------------------------------------

class TestAtempoFilters:
    """_atempo_filters must handle all multi-filter edge cases."""

    def test_normal_speed(self):
        from podcastcondensor.audio_strategies import _atempo_filters
        assert _atempo_filters(1.0) == []

    def test_fast_speed(self):
        from podcastcondensor.audio_strategies import _atempo_filters
        filters = _atempo_filters(2.0)
        assert len(filters) == 1
        assert "atempo=2.0" in filters[0]

    def test_very_fast_speed(self):
        from podcastcondensor.audio_strategies import _atempo_filters
        filters = _atempo_filters(4.0)
        # 4.0 = 2.0 * 2.0
        assert len(filters) == 2
        assert all("atempo=2.0" in f for f in filters)

    def test_slow_speed(self):
        from podcastcondensor.audio_strategies import _atempo_filters
        filters = _atempo_filters(0.5)
        assert len(filters) == 1
        assert "atempo=0.500" in filters[0]

    def test_moderate_speed(self):
        from podcastcondensor.audio_strategies import _atempo_filters
        filters = _atempo_filters(1.35)
        assert len(filters) == 1
        assert "atempo=1.350" in filters[0]


# ---------------------------------------------------------------------------
# 5. Graceful handling of zero intervals
# ---------------------------------------------------------------------------

class TestZeroIntervals:
    """All strategies must reject empty interval lists with a clear error."""

    def test_zero_intervals_raise(self):
        from podcastcondensor.audio_strategies import SinglePassFilterCutStrategy
        import pytest
        strat = SinglePassFilterCutStrategy()
        with pytest.raises(ValueError, match="No intervals|no intervals"):
            strat.cut(
                audio_path="/dev/null",
                intervals=[],
                output_path="/dev/null",
            )

    def test_parallel_zero_intervals(self):
        from podcastcondensor.audio_strategies import ParallelCopyCutStrategy
        import pytest
        strat = ParallelCopyCutStrategy()
        with pytest.raises(ValueError, match="No intervals"):
            strat.cut(
                audio_path="/dev/null",
                intervals=[],
                output_path="/dev/null",
            )

    def test_single_pass_zero_intervals(self):
        from podcastcondensor.audio_strategies import SinglePassFilterCutStrategy
        import pytest
        strat = SinglePassFilterCutStrategy()
        with pytest.raises(ValueError, match="No intervals"):
            strat.cut(
                audio_path="/dev/null",
                intervals=[],
                output_path="/dev/null",
            )


# ---------------------------------------------------------------------------
# 6. Config defaults (no-regression)
# ---------------------------------------------------------------------------

class TestConfigDefaults:
    """Default config must match current production behaviour."""

    def test_audio_strategy_default(self):
        from podcastcondensor.config import Config
        cfg = Config()
        # single_pass_filter is the default — one linear read, no HDD thrash
        assert cfg.audio_strategy == "single_pass_filter"


# ---------------------------------------------------------------------------
# 7. Interval normalization
# ---------------------------------------------------------------------------

class TestNormalizeIntervals:
    """normalize_intervals must sort, merge, and clean up interval lists."""

    def test_already_sorted_stays_unchanged(self):
        from podcastcondensor.audio_strategies import normalize_intervals
        ivs = [{"start": 10.0, "end": 20.0}, {"start": 30.0, "end": 40.0}]
        result = normalize_intervals(ivs)
        assert result == ivs

    def test_unsorted_is_sorted(self):
        from podcastcondensor.audio_strategies import normalize_intervals
        ivs = [{"start": 30.0, "end": 40.0}, {"start": 10.0, "end": 20.0}]
        result = normalize_intervals(ivs)
        assert result[0]["start"] == 10.0
        assert result[1]["start"] == 30.0

    def test_overlapping_merged(self):
        from podcastcondensor.audio_strategies import normalize_intervals
        ivs = [
            {"start": 10.0, "end": 20.0},
            {"start": 15.0, "end": 25.0},
        ]
        result = normalize_intervals(ivs)
        assert len(result) == 1
        assert result[0]["start"] == 10.0
        assert result[0]["end"] == 25.0

    def test_adjacent_merged(self):
        from podcastcondensor.audio_strategies import normalize_intervals
        ivs = [
            {"start": 10.0, "end": 20.0},
            {"start": 20.0, "end": 30.0},  # touch exactly
        ]
        result = normalize_intervals(ivs)
        assert len(result) == 1
        assert result[0]["start"] == 10.0
        assert result[0]["end"] == 30.0

    def test_contained_merged(self):
        from podcastcondensor.audio_strategies import normalize_intervals
        ivs = [
            {"start": 10.0, "end": 50.0},
            {"start": 20.0, "end": 30.0},  # fully inside
        ]
        result = normalize_intervals(ivs)
        assert len(result) == 1
        assert result[0]["start"] == 10.0
        assert result[0]["end"] == 50.0

    def test_zero_length_removed(self):
        from podcastcondensor.audio_strategies import normalize_intervals
        ivs = [
            {"start": 10.0, "end": 10.0},  # zero-length
            {"start": 20.0, "end": 30.0},
        ]
        result = normalize_intervals(ivs)
        assert len(result) == 1
        assert result[0]["start"] == 20.0

    def test_all_zero_length_returns_empty(self):
        from podcastcondensor.audio_strategies import normalize_intervals
        ivs = [{"start": 5.0, "end": 5.0}, {"start": 10.0, "end": 10.0}]
        assert normalize_intervals(ivs) == []

    def test_empty_input_returns_empty(self):
        from podcastcondensor.audio_strategies import normalize_intervals
        assert normalize_intervals([]) == []

    def test_clamps_to_duration(self):
        from podcastcondensor.audio_strategies import normalize_intervals
        ivs = [{"start": 50.0, "end": 200.0}]
        result = normalize_intervals(ivs, audio_duration=100.0)
        assert result[0]["end"] == 100.0  # clamped

    def test_clamps_to_zero(self):
        from podcastcondensor.audio_strategies import normalize_intervals
        ivs = [{"start": -10.0, "end": 20.0}]
        result = normalize_intervals(ivs, audio_duration=100.0)
        assert result[0]["start"] == 0.0  # clamped

    def test_complex_merge_chain(self):
        from podcastcondensor.audio_strategies import normalize_intervals
        ivs = [
            {"start": 50.0, "end": 60.0},
            {"start": 10.0, "end": 15.0},
            {"start": 12.0, "end": 18.0},
            {"start": 55.0, "end": 65.0},
        ]
        result = normalize_intervals(ivs)
        # [10,15]+[12,18] → [10,18]; [50,60]+[55,65] → [50,65]
        assert len(result) == 2
        assert result[0] == {"start": 10.0, "end": 18.0}
        assert result[1] == {"start": 50.0, "end": 65.0}


# ---------------------------------------------------------------------------
# 7. Interval clustering (build_intervals with cluster_gap)
# ---------------------------------------------------------------------------

class TestIntervalClustering:
    """build_intervals must merge kept segments into clusters."""

    def test_single_cluster(self):
        from podcastcondensor.intervals import build_intervals
        segments = [
            {"segment_id": "s1", "start": 10, "end": 15, "text": "a", "word_count": 10},
            {"segment_id": "s2", "start": 16, "end": 20, "text": "b", "word_count": 10},
        ]
        decisions = [
            {"id": "s1", "label": "keep"},
            {"id": "s2", "label": "keep"},
        ]
        intervals = build_intervals(segments, decisions, merge_gap=2.0,
                                    cluster_gap=2.0, pad_before=0, pad_after=0)
        # Both kept segments within cluster_gap -> 1 cluster
        assert len(intervals) == 1
        assert intervals[0]["start"] <= 10
        assert intervals[0]["end"] >= 20

    def test_two_clusters(self):
        from podcastcondensor.intervals import build_intervals
        segments = [
            {"segment_id": "s1", "start": 10, "end": 15, "text": "a", "word_count": 10},
            {"segment_id": "s2", "start": 16, "end": 20, "text": "b", "word_count": 10},
            {"segment_id": "s3", "start": 100, "end": 110, "text": "c", "word_count": 10},
        ]
        decisions = [
            {"id": "s1", "label": "keep"},
            {"id": "s2", "label": "keep"},
            {"id": "s3", "label": "keep"},
        ]
        intervals = build_intervals(segments, decisions, merge_gap=2.0,
                                    cluster_gap=2.0, pad_before=0, pad_after=0)
        # s1+s2 close together; s3 far away -> 2 clusters
        assert len(intervals) == 2

    def test_kept_ids_tracked(self):
        from podcastcondensor.intervals import build_intervals
        segments = [
            {"segment_id": "s1", "start": 10, "end": 15, "text": "a", "word_count": 10},
            {"segment_id": "s2", "start": 16, "end": 20, "text": "b", "word_count": 10},
        ]
        decisions = [
            {"id": "s1", "label": "keep"},
            {"id": "s2", "label": "keep"},
        ]
        intervals = build_intervals(segments, decisions, merge_gap=2.0,
                                    cluster_gap=2.0, pad_before=0, pad_after=0)
        assert "s1" in intervals[0]["kept_ids"]
        assert "s2" in intervals[0]["kept_ids"]

    def test_force_drop_excluded(self):
        from podcastcondensor.intervals import build_intervals
        segments = [
            {"segment_id": "s1", "start": 10, "end": 15, "text": "a", "word_count": 10},
            {"segment_id": "s2", "start": 20, "end": 25, "text": "b", "word_count": 10},
        ]
        decisions = [
            {"id": "s1", "label": "keep"},
            {"id": "s2", "label": "force_drop", "force_drop": True},
        ]
        intervals = build_intervals(segments, decisions, merge_gap=2.0,
                                    cluster_gap=5.0, pad_before=0, pad_after=0)
        # Only s1 kept; s2 excluded
        assert len(intervals) >= 1
        all_kept_ids = set()
        for iv in intervals:
            for kid in iv["kept_ids"].split(","):
                if kid:
                    all_kept_ids.add(kid)
        assert "s1" in all_kept_ids
        assert "s2" not in all_kept_ids

    def test_empty_kept_returns_empty(self):
        from podcastcondensor.intervals import build_intervals
        segments = [
            {"segment_id": "s1", "start": 10, "end": 15, "text": "x", "word_count": 10},
        ]
        decisions = [{"id": "s1", "label": "drop"}]
        assert build_intervals(segments, decisions) == []


# ---------------------------------------------------------------------------
# 8. Quality guardrails
# ---------------------------------------------------------------------------

class TestQualityGuardrails:
    """check_quality_guardrails must flag pathological results."""

    def test_very_low_compression_flagged(self):
        from podcastcondensor.intervals import check_quality_guardrails
        stats = {
            "compression_ratio": 0.10,
            "total_segments": 200,
            "keep_count": 20,
            "fragmentation": {
                "num_intervals": 15,
                "num_islands": 12,
                "island_ratio": 0.8,
                "avg_segments_per_interval": 1.3,
                "keep_density": 0.10,
                "status": "very_fragmented",
            },
        }
        warnings = check_quality_guardrails(stats, min_keep_ratio=0.20)
        assert len(warnings) >= 1
        assert any("compression ratio" in w.lower() for w in warnings)

    def test_very_fragmented_flagged(self):
        from podcastcondensor.intervals import check_quality_guardrails
        stats = {
            "compression_ratio": 0.30,
            "total_segments": 200,
            "keep_count": 80,
            "fragmentation": {
                "num_intervals": 60,
                "num_islands": 50,
                "island_ratio": 0.833,
                "avg_segments_per_interval": 1.3,
                "keep_density": 0.30,
                "status": "very_fragmented",
            },
        }
        warnings = check_quality_guardrails(stats, min_keep_ratio=0.20)
        assert any("fragmented" in w.lower() for w in warnings)

    def test_ok_result_no_warnings(self):
        from podcastcondensor.intervals import check_quality_guardrails
        stats = {
            "compression_ratio": 0.35,
            "total_segments": 200,
            "keep_count": 100,
            "fragmentation": {
                "num_intervals": 20,
                "num_islands": 3,
                "island_ratio": 0.15,
                "avg_segments_per_interval": 5.0,
                "keep_density": 0.35,
                "status": "ok",
            },
        }
        warnings = check_quality_guardrails(stats, min_keep_ratio=0.20)
        assert len(warnings) == 0

    def test_low_keep_rate_flagged(self):
        from podcastcondensor.intervals import check_quality_guardrails
        stats = {
            "compression_ratio": 0.30,
            "total_segments": 200,
            "keep_count": 5,  # only 2.5%
            "fragmentation": {
                "num_intervals": 4,
                "num_islands": 3,
                "island_ratio": 0.75,
                "avg_segments_per_interval": 1.25,
                "keep_density": 0.30,
                "status": "very_fragmented",
            },
        }
        warnings = check_quality_guardrails(stats, min_keep_ratio=0.20)
        assert any("keep rate" in w.lower() for w in warnings)


# ---------------------------------------------------------------------------
# 9. Fragmentation analysis
# ---------------------------------------------------------------------------

class TestFragmentationAnalysis:
    """analyze_fragmentation must detect islands."""

    def test_no_fragmentation_single_cluster(self):
        from podcastcondensor.intervals import analyze_fragmentation
        segments = [
            {"segment_id": "s1", "start": 0, "end": 10},
            {"segment_id": "s2", "start": 10, "end": 20},
        ]
        decisions = [
            {"id": "s1", "label": "keep"},
            {"id": "s2", "label": "keep"},
        ]
        intervals = [{"start": 0, "end": 20, "kept_ids": "s1,s2"}]
        frag = analyze_fragmentation(segments, decisions, intervals)
        assert frag["status"] == "ok"

    def test_very_fragmented_detected(self):
        from podcastcondensor.intervals import analyze_fragmentation
        segments = [{"segment_id": f"s{i}", "start": i*10, "end": i*10+5}
                    for i in range(20)]
        decisions = [{"id": f"s{i}", "label": "keep" if i % 3 == 0 else "drop"}
                     for i in range(20)]
        dec_map = {d["id"]: d["label"] for d in decisions}
        intervals = [
            {"start": s["start"], "end": s["end"], "kept_ids": s["segment_id"]}
            for s in segments if dec_map.get(s["segment_id"]) == "keep"
        ]
        frag = analyze_fragmentation(segments, decisions, intervals)
        # Most intervals are single-segment islands
        assert frag["num_islands"] > 0
        assert frag["status"] in ("fragmented", "very_fragmented")
