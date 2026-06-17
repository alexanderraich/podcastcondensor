"""Tests for DeepSeekSegmentation — timing, monotonicity, error handling."""

import json


class TestDeepSeekSegmentationMocked:
    """DeepSeekSegmentation with mocked API calls."""

    def test_punctuated_path_produces_monotonic_timing(self):
        """Fast path (entries have punctuation) produces monotonic segments."""
        from podcastcondensor.segmentation.deepseek import DeepSeekSegmentation

        class MockClient:
            model = "deepseek-chat"
            def generate(self, **kwargs):
                return json.dumps({
                    "schema_version": 1,
                    "segments": [
                        {"segment_id": "seg-0001", "start_unit_id": 1, "end_unit_id": 3, "boundary_reason": "intro"},
                        {"segment_id": "seg-0002", "start_unit_id": 4, "end_unit_id": 5, "boundary_reason": "body"},
                    ]
                })

        seg = DeepSeekSegmentation(client=MockClient())
        entries = [
            {"index": i, "start": (i-1)*10, "end": i*10, "text": f"Entry {i}.", "type": "speech"}
            for i in range(1, 6)
        ]
        segs = seg.segment(entries, "Entry 1. Entry 2. Entry 3. Entry 4. Entry 5.")
        assert len(segs) >= 1
        for i in range(1, len(segs)):
            assert segs[i]["start"] >= segs[i-1]["end"], f"Non-monotonic at {i}"
        assert all(s["end"] >= s["start"] for s in segs)

    def test_no_punctuation_path_produces_monotonic_timing(self):
        """No-punctuation path produces monotonic segments with proportional timing."""
        from podcastcondensor.segmentation.deepseek import DeepSeekSegmentation

        class MockClient:
            model = "deepseek-chat"
            call_count = 0
            def generate(self, **kwargs):
                MockClient.call_count += 1
                p = kwargs.get("prompt", "")
                if "cleanup" in p.lower():
                    return json.dumps({"schema_version": 1, "text": "First. Second. Third. Fourth. Fifth."})
                return json.dumps({
                    "schema_version": 1,
                    "segments": [
                        {"segment_id": "seg-0001", "start_unit_id": 1, "end_unit_id": 2, "boundary_reason": "intro"},
                        {"segment_id": "seg-0002", "start_unit_id": 3, "end_unit_id": 5, "boundary_reason": "body"},
                    ]
                })

        seg = DeepSeekSegmentation(client=MockClient())
        entries = [
            {"index": i, "start": (i-1)*10, "end": i*10, "text": f"entry {i}", "type": "speech"}
            for i in range(1, 11)
        ]
        segs = seg.segment(entries, "first second third")
        assert MockClient.call_count == 2, f"Expected 2 calls, got {MockClient.call_count}"
        assert len(segs) >= 1
        for i in range(1, len(segs)):
            assert segs[i]["start"] >= segs[i-1]["end"], f"Non-monotonic at seg {i}"
        assert all(s["end"] >= s["start"] for s in segs)

    def test_empty_entries_raises(self):
        from podcastcondensor.segmentation.deepseek import DeepSeekSegmentation
        import pytest
        seg = DeepSeekSegmentation(client=None)
        with pytest.raises(RuntimeError, match="No entries"):
            seg.segment([], "")

    def test_raises_on_invalid_plan(self):
        """A plan with gaps must raise."""
        from podcastcondensor.segmentation.deepseek import DeepSeekSegmentation
        import pytest

        class MockClient:
            model = "deepseek-chat"
            def generate(self, **kwargs):
                return json.dumps({
                    "schema_version": 1,
                    "segments": [
                        {"segment_id": "seg-0001", "start_unit_id": 1, "end_unit_id": 1},
                        {"segment_id": "seg-0002", "start_unit_id": 3, "end_unit_id": 3},
                    ]
                })

        seg = DeepSeekSegmentation(client=MockClient())
        entries = [
            {"index": i, "start": (i-1)*10, "end": i*10, "text": f"Entry {i}.", "type": "speech"}
            for i in range(1, 4)
        ]
        with pytest.raises((RuntimeError, ValueError)):
            seg.segment(entries, "Entry 1. Entry 2. Entry 3.")


class TestSegmentReconstruction:
    """Test the internal reconstruction methods."""

    def test_reconstruct_entry_plan(self):
        from podcastcondensor.segmentation.deepseek import DeepSeekSegmentation
        from podcastcondensor.segmentation.schemas import SegmentationPlan, SegmentationPlanItem

        plan = SegmentationPlan(schema_version=1, segments=[
            SegmentationPlanItem("s1", 1, 2, "intro"),
            SegmentationPlanItem("s2", 3, 3, "body"),
        ])
        entries = [
            {"index": 1, "start": 0.0, "end": 5.0, "text": "Hello.", "type": "speech"},
            {"index": 2, "start": 5.0, "end": 10.0, "text": "World.", "type": "speech"},
            {"index": 3, "start": 10.0, "end": 15.0, "text": "Third.", "type": "speech"},
        ]
        segs = DeepSeekSegmentation._reconstruct_entry_plan(plan, entries)
        assert len(segs) == 2
        assert segs[0]["start"] == 0.0
        assert segs[1]["end"] == 15.0
        assert segs[0]["source_indices"] == [1, 2]
        assert segs[1]["source_indices"] == [3]
