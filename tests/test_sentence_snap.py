"""Tests for sentence-block continuation detection and boundary re-snap.

Covers two 2026-08-02 fixes:
1. ``build_sentence_blocks`` no longer closes a block on a whisper period that
   sits mid-thought (e.g. "...and yet at the same time.", "...And so.").
2. ``snap_to_sentence_blocks`` / super-cut ``_resnap_selections`` re-align
   stale-cache boundaries to sentence blocks.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from podcastcondensor.subtitles import build_sentence_blocks
from podcastcondensor.minimal_theme_cut import snap_to_sentence_blocks, RefinedSelection
from podcastcondensor.master_cut import Selection, ThemeSegment


def _entries(pairs):
    """[(start, end, text)] -> SRT entry dicts."""
    return [
        {"index": i + 1, "start": s, "end": e, "text": t}
        for i, (s, e, t) in enumerate(pairs)
    ]


# ---------------------------------------------------------------------------
# build_sentence_blocks continuation detection
# ---------------------------------------------------------------------------


def test_whisper_period_on_continuation_does_not_close_block():
    entries = _entries([
        (0.0, 3.0, "Yahweh cannot be approached by sinful humanity, and yet at the same time."),
        (3.0, 6.0, "people see Yahweh, talk to Yahweh, eat with Yahweh."),
    ])
    blocks = build_sentence_blocks(entries)
    # The whisper period on "at the same time." must NOT close a block there —
    # the block absorbs the continuation and closes at the real sentence end.
    assert len(blocks) == 1, f"continuation should stay in one block, got {len(blocks)}"
    assert blocks[0]["end"] == 6.0
    assert "at the same time" in blocks[0]["text"]
    assert blocks[0]["text"].endswith("eat with Yahweh.")


def test_continuation_detection_extends_past_and_so():
    entries = _entries([
        (0.0, 3.0, "So you have this one at the peak. Yeah. And so."),
        (3.0, 6.0, "the grace that Saint Dionysius talks about."),
    ])
    blocks = build_sentence_blocks(entries)
    assert len(blocks) == 1, f"'And so.' should not close the block, got {len(blocks)}"
    assert blocks[0]["end"] == 6.0


def test_dangling_preposition_does_not_close_block():
    entries = _entries([
        (0.0, 3.0, "This is the grace that Saint Dionysius talks about."),
        (3.0, 6.0, "It transforms the hierarchy."),
    ])
    blocks = build_sentence_blocks(entries)
    assert len(blocks) == 1
    assert blocks[0]["end"] == 6.0


def test_normal_period_still_closes_block():
    entries = _entries([
        (0.0, 3.0, "This is a complete sentence."),
        (3.0, 6.0, "Here is another one."),
    ])
    blocks = build_sentence_blocks(entries)
    assert len(blocks) == 2
    assert blocks[0]["end"] == 3.0
    assert blocks[1]["end"] == 6.0


def test_leading_fragment_block_merges_into_previous_and_absorbs():
    # Whisper ends the first sentence with a spurious period, then starts the
    # continuation phrase as a new block. The fragment ("of Western European
    # origin...") must merge into the previous block AND the block must keep
    # absorbing to the next real sentence end — a cut may never land on it.
    entries = _entries([
        (0.0, 3.0, "This is something other than Christianity."),
        (3.0, 6.0, "of Western European origin or connection or cultural tradition."),
        (6.0, 9.0, "May see this too, how it reshaped everything, right?"),
    ])
    blocks = build_sentence_blocks(entries)
    # The fragment is absorbed — 6.0 (its end) must NOT be a block end.
    assert all(abs(b["end"] - 6.0) > 0.01 for b in blocks)
    # The fragment + following sentence close as one block at the real end.
    assert blocks[-1]["end"] == 9.0
    assert "cultural tradition" in blocks[-1]["text"]


def test_legitimate_preposition_sentence_keeps_absorbing_not_ending():
    # "Of course..." starts with a dangling preposition (in the leading set).
    # The leading-continuation rule prevents CLOSING on such an entry, so the
    # block absorbs to the next real sentence end — a lengthening, never a
    # break. ("Of course..." is a complete sentence but the cost of absorbing
    # it is just a longer block; the costly failure would be a mid-sentence cut
    # landing on a true fragment, per the project's lean-aggressive rule.)
    entries = _entries([
        (0.0, 3.0, "Of course, that is the heart of the matter."),
        (3.0, 6.0, "And the wise act accordingly."),
    ])
    blocks = build_sentence_blocks(entries)
    assert len(blocks) == 1
    assert blocks[0]["end"] == 6.0
    assert "matter" in blocks[0]["text"]


# ---------------------------------------------------------------------------
# snap_to_sentence_blocks / _resnap_selections (stale-cache re-alignment)
# ---------------------------------------------------------------------------


def _fmt(sec):
    ms = int(round((sec - int(sec)) * 1000))
    h, m, s = int(sec // 3600), int((sec % 3600) // 60), int(sec % 60)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def _write_srt(path, pairs):
    lines = []
    for i, (s, e, t) in enumerate(pairs):
        lines.append(str(i + 1))
        lines.append(f"{_fmt(s)} --> {_fmt(e)}")
        lines.append(t)
        lines.append("")
    with open(path, "w") as f:
        f.write("\n".join(lines))


def _make_output_root(tmp_path):
    ep_dir = tmp_path / "output" / "ep-023"
    ep_dir.mkdir(parents=True)
    _write_srt(ep_dir / "source_subtitles.srt", [
        (30.0, 33.0, "We basically interact with everything else."),
        (33.0, 36.0, "And yet at the same time."),
        (36.0, 39.0, "people see Yahweh, talk to Yahweh, eat with Yahweh."),
        (39.0, 42.0, "Right."),
    ])
    return str(tmp_path / "output")


def test_snap_to_sentence_blocks_extends_mid_sentence_end(tmp_path):
    output_root = _make_output_root(tmp_path)
    sel = RefinedSelection(episode_number=23, audio_path="x", start=31.0, end=36.5)
    snapped = snap_to_sentence_blocks([sel], output_root)
    assert snapped[0].end == 39.0, f"expected snap to extend to 39.0, got {snapped[0].end}"


def test_resnap_selections_corrects_stale_cache(tmp_path):
    from podcastcondensor.super_cut import _resnap_selections

    output_root = _make_output_root(tmp_path)
    seg = ThemeSegment(
        theme_id="scripture-and-hermeneutics",
        episode_number=23,
        audio_path="x",
        start=31.0,          # mid-block
        end=36.5,            # mid-block (whisper period on "at the same time.")
        text_preview="old preview",
    )
    sel = Selection(segment=seg, theme_title="Scripture and Hermeneutics",
                    theme_id="scripture-and-hermeneutics", beep_before="triple")
    out = _resnap_selections([sel], output_root)
    assert len(out) == 1
    assert out[0].segment.start == 30.0
    assert out[0].segment.end == 39.0
    # editorial metadata preserved
    assert out[0].segment.text_preview == "old preview"
    assert out[0].beep_before == "triple"


# ---------------------------------------------------------------------------
# apply_decisions playback-order preservation (2026-08-03)
# ---------------------------------------------------------------------------


def _tws_with_segments():
    """Four candidates: ep5, ep3, ep5, ep3 (in candidate/seg_id order)."""
    from podcastcondensor.master_cut import ThemeWithSegments
    from podcastcondensor.theme_extraction import Theme

    segs = [
        ThemeSegment(theme_id="t", episode_number=5, audio_path="a5.mp3",
                     start=100, end=110),
        ThemeSegment(theme_id="t", episode_number=3, audio_path="a3.mp3",
                     start=50, end=60),
        ThemeSegment(theme_id="t", episode_number=5, audio_path="a5.mp3",
                     start=112, end=120),
        ThemeSegment(theme_id="t", episode_number=3, audio_path="a3.mp3",
                     start=400, end=410),
    ]
    return ThemeWithSegments(
        theme=Theme(id="t", title="T", description="d"),
        segments=segs,
    )


def _manifests():
    from podcastcondensor.download_pool import EpisodeManifest

    return [
        EpisodeManifest(episode_number=3, video_id="v", title="T",
                        audio_path="a3.mp3", srt_path=""),
        EpisodeManifest(episode_number=5, video_id="v", title="T",
                        audio_path="a5.mp3", srt_path=""),
    ]


def test_apply_decisions_preserves_llm_playback_order():
    from podcastcondensor.minimal_theme_cut import apply_decisions, SegmentDecision

    tws = _tws_with_segments()
    # LLM returns kept segments in PLAYBACK order: ep3 (open) → ep5 (develop)
    # → ep3 (close). Old code re-sorted by episode ([3,3,5]); new code keeps
    # the array order.
    decisions = [
        SegmentDecision(seg_id="seg_1", keep=True, refined_start=50, refined_end=60,
                        reason="open"),
        SegmentDecision(seg_id="seg_0", keep=True, refined_start=100, refined_end=110,
                        reason="develop"),
        SegmentDecision(seg_id="seg_3", keep=True, refined_start=400, refined_end=410,
                        reason="close"),
        SegmentDecision(seg_id="seg_2", keep=False, reason="drop"),
    ]
    out = apply_decisions(decisions, tws, _manifests(), output_root="")
    assert [o.episode_number for o in out] == [3, 5, 3]


def test_apply_decisions_merges_only_consecutive_same_episode():
    from podcastcondensor.minimal_theme_cut import apply_decisions, SegmentDecision

    tws = _tws_with_segments()
    # seg_0 (ep5 100-110) and seg_2 (ep5 112-120) are consecutive in the
    # LLM's returned order → merge into 100-120. seg_1 (ep3) stays in place.
    decisions = [
        SegmentDecision(seg_id="seg_0", keep=True, refined_start=100, refined_end=110,
                        reason="first part"),
        SegmentDecision(seg_id="seg_2", keep=True, refined_start=112, refined_end=120,
                        reason="second part"),
        SegmentDecision(seg_id="seg_1", keep=True, refined_start=50, refined_end=60,
                        reason="other ep"),
        SegmentDecision(seg_id="seg_3", keep=False, reason="drop"),
    ]
    out = apply_decisions(decisions, tws, _manifests(), output_root="")
    assert len(out) == 2, f"expected 2 (merged ep5 + ep3), got {len(out)}"
    assert out[0].episode_number == 5 and out[0].start == 100 and out[0].end == 120
    assert out[1].episode_number == 3


def test_apply_decisions_does_not_merge_across_episode_boundary():
    from podcastcondensor.minimal_theme_cut import apply_decisions, SegmentDecision

    tws = _tws_with_segments()
    # ep5 A → ep3 → ep5 B: the two ep5 segments are NOT consecutive in the
    # playback order (ep3 plays between them) and must stay apart.
    decisions = [
        SegmentDecision(seg_id="seg_0", keep=True, refined_start=100, refined_end=110,
                        reason="A"),
        SegmentDecision(seg_id="seg_1", keep=True, refined_start=50, refined_end=60,
                        reason="B"),
        SegmentDecision(seg_id="seg_2", keep=True, refined_start=112, refined_end=120,
                        reason="C"),
        SegmentDecision(seg_id="seg_3", keep=False, reason="drop"),
    ]
    out = apply_decisions(decisions, tws, _manifests(), output_root="")
    assert [o.episode_number for o in out] == [5, 3, 5]


def test_verify_cut_boundaries_accepts_touching_block_boundary(tmp_path):
    # Regression: blocks are contiguous, so a selection end that equals the
    # NEXT block's start must not be flagged as mid-sentence. The contains-
    # check (start <= x <= end) matches both neighbors and false-FAILs.
    from podcastcondensor.super_cut import verify_cut_boundaries

    output_root = _make_output_root(tmp_path)
    seg = ThemeSegment(
        theme_id="t", episode_number=23, audio_path="x",
        start=30.0,   # block start of "We basically interact..."
        end=39.0,     # block end of "...eat with Yahweh." (== next block start "Right.")
    )
    sel = Selection(segment=seg, theme_title="T", theme_id="t", beep_before="none")
    report = verify_cut_boundaries([sel], output_root)
    assert report[0]["start_ok"] is True
    assert report[0]["end_ok"] is True
    assert "eat with Yahweh" in report[0]["last_line"]


def test_apply_decisions_episode_diversity_cap_limits_single_episode():
    from podcastcondensor.minimal_theme_cut import apply_decisions, SegmentDecision

    # Kept order: ep5(r1), ep3(r4), ep5(r5), ep5(r3). max_per_episode=2 must
    # drop the lowest-relevance ep5 segment (r1) and keep ep3 + both other
    # ep5s, PRESERVING the original playback order (no re-sort by episode).
    segs = [
        ThemeSegment(theme_id="t", episode_number=5, audio_path="a5.mp3",
                     start=100, end=110, relevance_score=1.0),
        ThemeSegment(theme_id="t", episode_number=3, audio_path="a3.mp3",
                     start=50, end=60, relevance_score=4.0),
        ThemeSegment(theme_id="t", episode_number=5, audio_path="a5.mp3",
                     start=112, end=120, relevance_score=5.0),
        ThemeSegment(theme_id="t", episode_number=5, audio_path="a5.mp3",
                     start=400, end=410, relevance_score=3.0),
    ]
    from podcastcondensor.master_cut import ThemeWithSegments
    from podcastcondensor.theme_extraction import Theme
    tws = ThemeWithSegments(theme=Theme(id="t", title="T", description="d"),
                            segments=segs)
    decisions = [
        SegmentDecision(seg_id="seg_0", keep=True, reason="ep5 r1"),
        SegmentDecision(seg_id="seg_1", keep=True, reason="ep3 r4"),
        SegmentDecision(seg_id="seg_2", keep=True, reason="ep5 r5"),
        SegmentDecision(seg_id="seg_3", keep=True, reason="ep5 r3"),
    ]
    out = apply_decisions(decisions, tws, _manifests(), output_root="")
    # 3 kept after cap: ep3 (r4) + ep5 (r5, r3) — r1 dropped
    assert len(out) == 3, f"expected 3 after cap, got {len(out)}"
    assert [o.episode_number for o in out] == [3, 5, 5]
    assert abs(out[1].start - 112) < 0.01  # the r5 ep5 segment survived


def test_apply_decisions_cap_prefers_highest_relevance_per_episode():
    from podcastcondensor.minimal_theme_cut import apply_decisions, SegmentDecision

    # Three ep5 segments with relevance 1, 5, 3 → cap=1 keeps only r5.
    segs = [
        ThemeSegment(theme_id="t", episode_number=5, audio_path="a5.mp3",
                     start=100, end=110, relevance_score=1.0),
        ThemeSegment(theme_id="t", episode_number=5, audio_path="a5.mp3",
                     start=130, end=140, relevance_score=5.0),
        ThemeSegment(theme_id="t", episode_number=5, audio_path="a5.mp3",
                     start=160, end=170, relevance_score=3.0),
    ]
    from podcastcondensor.master_cut import ThemeWithSegments
    from podcastcondensor.theme_extraction import Theme
    tws = ThemeWithSegments(theme=Theme(id="t", title="T", description="d"),
                            segments=segs)
    decisions = [
        SegmentDecision(seg_id=f"seg_{i}", keep=True, reason=f"r{i}") for i in range(3)
    ]
    out = apply_decisions(decisions, tws, _manifests(), output_root="",
                          max_per_episode=1)
    assert len(out) == 1
    assert abs(out[0].start - 130) < 0.01  # the r5 segment survived


def test_apply_decisions_places_conclusion_segment_last():
    from podcastcondensor.minimal_theme_cut import apply_decisions, SegmentDecision
    from podcastcondensor.master_cut import ThemeWithSegments
    from podcastcondensor.theme_extraction import Theme

    # The LLM returns a "Concludes that..." segment mid-array. apply_decisions
    # must move it to the END of playback order, preserving everything else's
    # relative order. First word must be a conclusion verb — mid-arc reasons
    # like "Contrasts ... concluding that ..." must NOT move. Four distinct
    # episodes so the per-episode cap and merge don't interfere.
    segs = [
        ThemeSegment(theme_id="t", episode_number=1, audio_path="a1.mp3",
                     start=100, end=110, relevance_score=5.0),
        ThemeSegment(theme_id="t", episode_number=2, audio_path="a2.mp3",
                     start=100, end=110, relevance_score=5.0),
        ThemeSegment(theme_id="t", episode_number=3, audio_path="a3.mp3",
                     start=100, end=110, relevance_score=5.0),
        ThemeSegment(theme_id="t", episode_number=4, audio_path="a4.mp3",
                     start=100, end=110, relevance_score=5.0),
    ]
    tws = ThemeWithSegments(theme=Theme(id="t", title="T", description="d"),
                            segments=segs)
    decisions = [
        SegmentDecision(seg_id="seg_0", keep=True, reason="Introduces the topic and its roots"),
        SegmentDecision(seg_id="seg_1", keep=True, reason="Concludes that the true worship endures"),
        SegmentDecision(seg_id="seg_2", keep=True, reason="Explains the development in the middle"),
        SegmentDecision(seg_id="seg_3", keep=True, reason="Contrasts two views, concluding that X"),
    ]
    out = apply_decisions(decisions, tws, _manifests(), output_root="")
    assert len(out) == 4
    assert [o.reason.split()[0] for o in out] == [
        "Introduces", "Explains", "Contrasts", "Concludes",
    ]


def test_place_conclusion_selections_no_conclusion_unchanged():
    from podcastcondensor.minimal_theme_cut import _place_conclusion_segments, RefinedSelection

    sels = [
        RefinedSelection(episode_number=1, audio_path="a", start=0, end=10,
                         reason="Introduces the topic"),
        RefinedSelection(episode_number=2, audio_path="a", start=0, end=10,
                         reason="Explains the middle"),
    ]
    out = _place_conclusion_segments(sels)
    assert [s.reason for s in out] == [
        "Introduces the topic", "Explains the middle",
    ]


def test_order_combined_selections_assigns_triple_beeps_between_themes():
    from podcastcondensor.super_cut import order_combined_selections

    def _sel(tid, ep, start, end):
        return Selection(
            segment=ThemeSegment(theme_id=tid, episode_number=ep, audio_path="a.mp3",
                                 start=start, end=end),
            theme_title=tid, theme_id=tid, beep_before="none",
        )

    sels = [
        _sel("modernity", 99, 100, 200),
        _sel("election", 93, 50, 100),
        _sel("modernity", 102, 300, 400),
        _sel("election", 130, 10, 60),
    ]
    out = order_combined_selections(sels, ["modernity", "election"])
    assert [s.theme_id for s in out] == [
        "modernity", "modernity", "election", "election",
    ]
    assert out[0].beep_before == "none"
    assert out[1].beep_before == "single"   # within modernity
    assert out[2].beep_before == "triple"   # theme change
    assert out[3].beep_before == "single"   # within election
