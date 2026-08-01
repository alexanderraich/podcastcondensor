"""Per-theme LLM selection engine, shared by master-cut and super-cut.

The ``build-minimal-theme`` command was removed on 2026-08-01 — superseded by
``build-super-cut --theme <id>``. This module keeps the shared selection logic
that ``build-master-cut`` (Phase 5) and ``build-super-cut`` (Phase 5) both
reuse: building the per-theme selection prompt with transcript context,
parsing the LLM keep/drop decisions, and snapping refined boundaries to
sentence blocks.
"""

from __future__ import annotations

import json
import logging
import os
import re
import shutil
import subprocess
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from podcastcondensor.audio_strategies import _ionice_cmd, _atempo_filters, _concat_batch_files
from podcastcondensor.config import Config
from podcastcondensor.download_pool import EpisodeManifest
from podcastcondensor.llm.deepseek import DeepSeekClient
from podcastcondensor.universe_state import UniverseState
from podcastcondensor.master_cut import (
    resolve_theme_segments_from_state,
    ThemeSegment,
    ThemeWithSegments,
    _generate_beep_file,
    _extract_segment,
)
from podcastcondensor.subtitles import load_subtitles
from podcastcondensor.theme_extraction import Theme

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Phase 2: Load transcript context for segments
# ---------------------------------------------------------------------------


def _load_episode_entries(output_root: str, ep_num: int,
                          cache: Dict[int, List[dict]]) -> List[dict]:
    """Load cleaned SRT entries for an episode, using a cache."""
    if ep_num not in cache:
        ep_dir = os.path.join(output_root, f"ep-{ep_num:03d}")
        srt_path = os.path.join(ep_dir, "source_subtitles.srt")
        if os.path.exists(srt_path):
            cache[ep_num] = load_subtitles(srt_path, reindex=False)
            logger.debug("Loaded %d entries for ep %d", len(cache[ep_num]), ep_num)
        else:
            logger.warning("SRT not found for ep %d at %s", ep_num, srt_path)
            cache[ep_num] = []
    return cache[ep_num]


def _snap_to_sentence_blocks(
    selections: List[RefinedSelection],
    srt_entries: List[dict],
) -> List[RefinedSelection]:
    """Snap segment boundaries to sentence-block boundaries.

    Loads sentence blocks from the SRT entries and snaps each segment's
    start to the nearest block start, end to the nearest block end.

    This is a HARD CONSTRAINT — the snapped boundary is guaranteed to
    be a complete sentence boundary.
    """
    from podcastcondensor.subtitles import build_sentence_blocks

    blocks = build_sentence_blocks(srt_entries)
    if not blocks:
        return selections

    snapped = []
    for sel in selections:
        # Find containing block for start
        s_block = None
        for b in blocks:
            if b["start"] <= sel.start <= b["end"]:
                s_block = b
                break
        if s_block is None:
            # Start falls between blocks — snap to nearest
            for b in blocks:
                if b["start"] >= sel.start or abs(b["end"] - sel.start) < 5.0:
                    s_block = b
                    break

        # Find containing block for end
        e_block = None
        for b in reversed(blocks):
            if b["start"] <= sel.end <= b["end"]:
                e_block = b
                break
        if e_block is None:
            for b in reversed(blocks):
                if abs(b["start"] - sel.end) < 5.0 or b["end"] >= sel.end:
                    e_block = b
                    break

        if s_block and e_block:
            snapped.append(RefinedSelection(
                episode_number=sel.episode_number,
                audio_path=sel.audio_path,
                start=s_block["start"],
                end=e_block["end"],
                reason=sel.reason,
            ))
        else:
            # Can't snap — keep original (shouldn't happen with valid SRT)
            snapped.append(sel)

    return snapped


def _format_segment_with_context(
    seg: ThemeSegment,
    entries: List[dict],
    seg_index: int,
    episode_title: str,
    context_buffer: float = 30.0,
    *,
    next_seg_start: Optional[float] = None,
    prev_seg_end: Optional[float] = None,
) -> Tuple[str, bool]:
    """Format one segment's transcript text with context for the prompt.

    The candidate window is snapped to sentence-block boundaries so the
    LLM only sees complete-sentence candidates.

    The context window extends dynamically:
    - **End boundary:** extends to the next candidate segment's start
      in the same episode (+ small overlap), capped at snapped_end + 120s.
      Without a next segment, uses the fixed *context_buffer*.
    - **Start boundary:** extends backward to the previous segment's end
      (+ small overlap), or uses the fixed backward buffer.

    This ensures the LLM can see enough context to determine whether a
    thought continues into an adjacent segment — without arbitrary tuning.

    Returns (formatted_text, has_data) where has_data is False if
    the transcript couldn't be loaded.
    """
    if not entries:
        return (
            f"--- Segment {seg_index + 1} (seg_id: seg_{seg_index}) ---\n"
            f"Episode: {seg.episode_number} — {episode_title}\n"
            f"Candidate: {seg.start:.1f}s - {seg.end:.1f}s ({seg.duration:.0f}s)\n"
            f"(transcript not available)\n",
            False,
        )

    # Snap candidate window to sentence-block boundaries
    from podcastcondensor.subtitles import build_sentence_blocks
    blocks = build_sentence_blocks(entries)
    snapped_start = seg.start
    snapped_end = seg.end
    for b in blocks:
        if b["start"] <= seg.start <= b["end"]:
            snapped_start = b["start"]
        if b["start"] <= seg.end <= b["end"]:
            snapped_end = b["end"]

    # ── Dynamic context window ────────────────────────────────────────
    # End: extend to next segment's start (+ small overlap), capped at 120s
    MAX_CONTEXT = 120.0
    end_win = snapped_end + context_buffer
    if next_seg_start is not None:
        # Peek a few seconds into the next segment so the LLM can see
        # whether the current thought continues past the boundary
        end_win = min(next_seg_start + 10, snapped_end + MAX_CONTEXT)
    else:
        end_win = snapped_end + context_buffer

    # Start: extend backward to previous segment's end (+ small overlap)
    start_win = max(0, snapped_start - context_buffer)
    if prev_seg_end is not None:
        start_win = max(0, min(start_win, prev_seg_end + 5))

    text_lines = []
    in_segment = False
    for e in entries:
        if e["start"] >= end_win:
            break
        if e["end"] <= start_win:
            continue
        # Is this entry within the snapped candidate window?
        is_candidate = e["start"] >= snapped_start and e["end"] <= snapped_end
        marker = "  >>>  " if is_candidate else "       "
        if is_candidate and not in_segment:
            text_lines.append("       ── candidate window ──")
            in_segment = True
        text_lines.append(f"{marker}{e['start']:7.1f}-{e['end']:7.1f}: {e['text']}")

    return (
        f"--- Segment {seg_index + 1} (seg_id: seg_{seg_index}) ---\n"
        f"Episode: {seg.episode_number} — {episode_title}\n"
        f"Candidate: {snapped_start:.1f}s - {snapped_end:.1f}s ({snapped_end - snapped_start:.0f}s)\n"
        f"\nTranscript (>>> = candidate window, no marker = context):\n"
        + "\n".join(text_lines),
        True,
    )


def _volume_guide(time_budget: Optional[float]) -> List[str]:
    """Volume guidance lines for the selection prompt.

    Without a budget: generic 4-8 segments / 8-20 minutes guidance (single-theme
    dev tool). With a budget: the theme's allotted share of the master cut, as a
    HARD CAP. The LLM keeps only its best segments under the cap — volume is
    enforced in-prompt (DeepSeek ignores Python-side truncation if told to keep
    "everything relevant", so the cap must be explicit and non-negotiable).
    """
    if time_budget is not None:
        mins = time_budget / 60.0
        return [
            f"  HARD CAP: your kept segments must total AT MOST {mins:.1f} minutes",
            f"  ({time_budget:.0f} seconds) of audio. This is non-negotiable —",
            "  the editor will cut anything past the cap, so do not keep",
            "  segments you would not want to lose.",
            "  Choose the 2-6 BEST segments that explain the theme from scratch.",
            "  Strongly prefer quality over coverage: it is better to keep 3",
            "  excellent segments that fit the cap than 8 mediocre ones that",
            "  overflow it.",
        ]
    return [
        "  A thorough treatment of this theme probably needs 4-8 segments",
        "  totalling 8-20 minutes. That gives room for: a definition segment,",
        "  a development/argument segment, a couple examples, and a",
        "  connection-to-broader-theology segment. Fewer than 4 segments",
        "  is unlikely to be self-contained; more than 10 is probably",
        "  too repetitive.",
    ]


def _budget_block(time_budget: Optional[float]) -> List[str]:
    """Prominent, non-negotiable budget statement placed near the top of the
    selection prompt when a master-cut time budget applies.

    DeepSeek ignores soft volume guidance (keeps ~all candidates regardless),
    so the cap must be stated as an explicit hard constraint up front, with a
    concrete segment-count target the model can reason about.
    """
    if time_budget is None:
        return []
    mins = time_budget / 60.0
    # ~3 min per segment is a reasonable planning granularity for these long
    # podcast segments; gives a concrete "keep about N" target.
    target_count = max(2, int(round(time_budget / 180)))
    return [
        "HARD BUDGET:",
        f"  Your kept segments must total AT MOST {mins:.0f} minutes"
        f" ({time_budget:.0f}s) of audio — a firm cap, not a target to aim at.",
        f"  Keep roughly {target_count} segments (2-{target_count + 2}),"
        " prioritizing the very best explanations of the theme.",
        "  If a segment is good but the theme is already well covered without it,",
        "  DROP it. Do not keep segments merely because they are relevant — keep",
        "  only what a complete beginner needs to understand this theme.",
    ]


def build_selection_prompt(
    theme: Theme,
    tws: ThemeWithSegments,
    output_root: str,
    manifests: List[EpisodeManifest],
    context_buffer: float = 30.0,
    time_budget: Optional[float] = None,
) -> str:
    """Build the prompt asking the LLM to select minimal viable segments.

    Each candidate segment is shown with ~30s of surrounding transcript
    context so the LLM can determine complete-thought boundaries.

    If ``time_budget`` is given (seconds), it is stated in the prompt as the
    allotted share of the master cut for this theme. The LLM owns the volume
    — this is guidance, not a hard truncation.
    """
    ep_titles = {m.episode_number: m.title for m in manifests}
    entries_cache: Dict[int, List[dict]] = {}

    # Pre-load all needed SRT files
    for seg in tws.segments:
        _load_episode_entries(output_root, seg.episode_number, entries_cache)

    # Build per-episode segment lists for dynamic context windows
    # Grouped by episode so we can find the next/previous candidate
    # segment for each candidate within the same episode.
    ep_segments: Dict[int, List[Tuple[int, ThemeSegment]]] = {}
    for i, seg in enumerate(tws.segments):
        ep_segments.setdefault(seg.episode_number, []).append((i, seg))

    # Sort each episode's segments chronologically
    for ep in ep_segments:
        ep_segments[ep].sort(key=lambda x: x[1].start)

    parts = [
        f"You are editing a podcast anthology. The theme is \"{theme.title}\".",
        "",
        "THEME DESCRIPTION:",
        theme.description,
        "",
        "YOUR AUDIENCE: A smart friend who has NEVER listened to this podcast.",
        "They need to walk away actually understanding this theme — what it is,",
        "why it matters, and how it fits together.",
        "",
        "YOUR JOB:",
        "Select audio segments that together form a COMPLETE, SELF-CONTAINED",
        "explanation of this theme from scratch. Think of it like a mini-",
        "documentary segment on this one topic — it needs enough context to",
        "stand on its own.",
        *_budget_block(time_budget),
        "",
        "WHAT TO KEEP:",
        "  - Definitions of key terms (start from zero, build up)",
        "  - Core arguments and WHY this matters theologically",
        "  - Concrete examples that illustrate the concept",
        "  - Connections to scripture, church fathers, the liturgy — things",
        "    that ground the concept and show its importance",
        "  - Development over time (how understanding deepens across episodes)",
        "",
        "WHAT TO DROP:",
        "  - Host banter, teasers for later topics, meta-talk",
        "  - Tangents that don't directly build understanding of THIS theme",
        "  - Repetitive content (if two segments say the same thing, keep",
        "    the better one and drop the rest)",
        "  - Incomplete previews that set up an idea but don't deliver it",
        "    within the segment",
        "",
        "BOUNDARY REFINEMENT:",
        "  - The '>>>' markers show the candidate window, but the actual",
        "    thought likely starts before it or continues after it.",
        "    WIDEN the window to capture the complete thought.",
        "  - A segment should feel like a complete 'scene': it introduces",
        "    an idea, develops it, and either concludes or clearly hands",
        "    off to the next one.",
        "  - If two kept segments from the same episode overlap or are",
        "    within 5 seconds of each other, MERGE them (wider window).",
        "  - Above all: the END of each kept segment must land on a",
        "    COMPLETE SENTENCE. Check that the last SRT entry within your",
        "    segment ends with sentence punctuation (. ! ?). If the thought",
        "    carries into the next entry, extend your end boundary to include",
        "    it. No mid-thought cuts.",
        "",
        "GUIDE ON VOLUME:",
        *_volume_guide(time_budget),
        "OUTPUT FORMAT — valid JSON only, no extra text:",
        """{{"segments": [
  {{
    "seg_id": "seg_0",
    "keep": true,
    "refined_start": 460.0,
    "refined_end": 548.0,
    "reason": "Defines theosis as participation in divine nature"
  }},
  {{
    "seg_id": "seg_1",
    "keep": false,
    "reason": "Host banter, does not explain theosis"
  }}
]}}""",
        "",
        "For dropped segments, set keep: false (refined_start/end can be 0).",
        "For kept segments, refined_start/end ARE REQUIRED.",
        "",
        "CANDIDATE SEGMENTS (in chronological order):",
        "",
    ]

    for i, seg in enumerate(tws.segments):
        entries = entries_cache.get(seg.episode_number, [])
        title = ep_titles.get(seg.episode_number, f"Episode {seg.episode_number}")

        # Find next/prev segment in the same episode for dynamic context
        prev_seg_end = None
        next_seg_start = None
        ep_list = ep_segments.get(seg.episode_number, [])
        for j, (idx, s) in enumerate(ep_list):
            if idx == i:
                if j > 0:
                    prev_seg_end = ep_list[j - 1][1].end
                if j + 1 < len(ep_list):
                    next_seg_start = ep_list[j + 1][1].start
                break

        formatted, _ = _format_segment_with_context(
            seg, entries, i, title, context_buffer,
            prev_seg_end=prev_seg_end,
            next_seg_start=next_seg_start,
        )
        parts.append(formatted)
        parts.append("")

    # Build seg_id lookup mapping for response validation
    # (informational — embedded in the prompt implicitly)
    parts.append(
        "---\n"
        "Now decide for each seg_id above: keep or drop?\n"
        "Return ONLY valid JSON in the format specified above."
    )

    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Phase 3: Parse LLM selection response
# ---------------------------------------------------------------------------


@dataclass
class SegmentDecision:
    """LLM's decision on a single candidate segment."""
    seg_id: str
    keep: bool
    refined_start: float = 0.0
    refined_end: float = 0.0
    reason: str = ""


def _try_parse_json(text: str) -> Optional[dict]:
    """Attempt to parse JSON, trying several repair strategies."""
    text = text.strip()

    # Strip markdown code fences
    if "```" in text:
        lines = text.split("\n")
        clean = []
        in_block = False
        for line in lines:
            stripped = line.strip()
            if stripped.startswith("```"):
                in_block = not in_block
                continue
            if in_block:
                clean.append(line)
        if clean:
            text = "\n".join(clean).strip()

    # Find JSON object
    start = text.find("{")
    if start < 0:
        return None
    end = text.rfind("}")
    if end <= start:
        return None
    candidate = text[start:end + 1]

    # Strategy 1: strict
    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        pass

    # Strategy 2: trailing commas
    repaired = re.sub(r",\s*([}\]])", r"\1", candidate)
    try:
        return json.loads(repaired)
    except json.JSONDecodeError:
        pass

    return None


def parse_selection_response(raw: str) -> List[SegmentDecision]:
    """Parse the LLM's JSON response into a list of SegmentDecisions."""
    if not raw:
        logger.warning("Empty LLM response for segment selection")
        return []

    data = _try_parse_json(raw)
    if not data:
        logger.warning("Failed to parse selection response (first 200): %s", raw[:200])
        return []

    raw_segs = data.get("segments", [])
    if not raw_segs:
        logger.warning("No 'segments' array in selection response")
        return []

    decisions = []
    for rs in raw_segs:
        try:
            d = SegmentDecision(
                seg_id=str(rs.get("seg_id", "")),
                keep=bool(rs.get("keep", False)),
                refined_start=float(rs.get("refined_start", 0) or 0),
                refined_end=float(rs.get("refined_end", 0) or 0),
                reason=str(rs.get("reason", "")),
            )
            decisions.append(d)
        except (ValueError, TypeError, KeyError) as e:
            logger.warning("Skipping malformed decision: %s", e)
            continue

    kept = sum(1 for d in decisions if d.keep)
    logger.info("Selection response: %d/%d segments kept", kept, len(decisions))
    return decisions


# ---------------------------------------------------------------------------
# Phase 4: Build refined selections from decisions
# ---------------------------------------------------------------------------


@dataclass
class RefinedSelection:
    """A segment with LLM-refined boundaries, ready for audio cutting."""
    episode_number: int
    audio_path: str
    start: float
    end: float
    reason: str = ""

    @property
    def duration(self) -> float:
        return max(self.end - self.start, 0.0)


def apply_decisions(
    decisions: List[SegmentDecision],
    tws: ThemeWithSegments,
    manifests: List[EpisodeManifest],
    output_root: str = "",
) -> List[RefinedSelection]:
    """Apply LLM decisions to build the final selection list.

    Merges overlapping/adjacent kept segments from the same episode,
    then snaps boundaries to sentence-block completion points.

    When *output_root* is provided, performs a final snap to sentence-
    block boundaries — the returned segments are guaranteed to start
    and end at complete-sentence boundaries.
    """
    # Map seg_id -> original ThemeSegment
    seg_map: Dict[str, ThemeSegment] = {}
    for i, seg in enumerate(tws.segments):
        seg_map[f"seg_{i}"] = seg

    # Build audio path lookup
    ep_to_audio: Dict[int, str] = {}
    for m in manifests:
        ep_to_audio[m.episode_number] = m.audio_path

    # Collect kept segments with refined boundaries
    kept: List[RefinedSelection] = []
    for d in decisions:
        if not d.keep:
            continue
        orig = seg_map.get(d.seg_id)
        if not orig:
            logger.warning("Decision references unknown seg_id '%s'", d.seg_id)
            continue

        start = d.refined_start if d.refined_start > 0 else orig.start
        end = d.refined_end if d.refined_end > 0 else orig.end

        # Sanity: ensure end > start
        if end <= start:
            logger.warning("Invalid boundaries for %s: %.1f-%.1f, using originals %.1f-%.1f",
                          d.seg_id, start, end, orig.start, orig.end)
            start = orig.start
            end = orig.end

        # Ensure within audio duration
        start = max(0, start)

        kept.append(RefinedSelection(
            episode_number=orig.episode_number,
            audio_path=ep_to_audio.get(orig.episode_number, orig.audio_path),
            start=start,
            end=end,
            reason=d.reason,
        ))

    # Merge overlapping/adjacent segments from the same episode
    # Group by episode, sort by start, merge if they touch or overlap
    by_ep: Dict[int, List[RefinedSelection]] = {}
    for k in kept:
        by_ep.setdefault(k.episode_number, []).append(k)

    merged: List[RefinedSelection] = []
    for ep in sorted(by_ep.keys()):
        ep_segs = sorted(by_ep[ep], key=lambda s: s.start)
        current = ep_segs[0]
        for next_seg in ep_segs[1:]:
            gap = next_seg.start - current.end
            if gap <= 5.0:  # merge if within 5 seconds
                reasons = [r for r in (current.reason, next_seg.reason) if r]
                current = RefinedSelection(
                    episode_number=current.episode_number,
                    audio_path=current.audio_path,
                    start=current.start,
                    end=max(current.end, next_seg.end),
                    reason="; ".join(reasons),
                )
            else:
                merged.append(current)
                current = next_seg
        merged.append(current)

    logger.info(
        "Refined selections: %d kept after merging → %d segments",
        len(kept), len(merged),
    )
    for mseg in merged:
        logger.info(
            "  Ep %d: %.1f-%.1f (%.0fs) — %s",
            mseg.episode_number, mseg.start, mseg.end, mseg.duration,
            mseg.reason[:80],
        )

    # ── Snap to sentence-block boundaries (hard constraint) ────────────
    if output_root and merged:
        from podcastcondensor.subtitles import load_subtitles, build_sentence_blocks
        loaded_eps: Dict[int, List[dict]] = {}
        sentence_snapped: List[RefinedSelection] = []
        for mseg in merged:
            ep = mseg.episode_number
            # Load SRT entries for this episode (cache)
            if ep not in loaded_eps:
                srt_path = os.path.join(
                    output_root, f"ep-{ep:03d}", "source_subtitles.srt"
                )
                try:
                    loaded_eps[ep] = load_subtitles(srt_path, reindex=False)
                except FileNotFoundError:
                    loaded_eps[ep] = []
            entries = loaded_eps.get(ep, [])
            if not entries:
                continue

            blocks = build_sentence_blocks(entries)
            if not blocks:
                continue

            # Snap start to containing block start
            s_block = None
            for b in blocks:
                if b["start"] <= mseg.start <= b["end"]:
                    s_block = b
                    break
            if s_block is None:
                s_block = blocks[0]
                for b in blocks:
                    if b["end"] >= mseg.start:
                        s_block = b
                        break

            # Snap end to containing block end
            e_block = None
            for b in reversed(blocks):
                if b["start"] <= mseg.end <= b["end"]:
                    e_block = b
                    break
            if e_block is None:
                e_block = blocks[-1]
                for b in blocks:
                    if b["start"] >= mseg.end:
                        e_block = b
                        break

            snapped = RefinedSelection(
                episode_number=mseg.episode_number,
                audio_path=mseg.audio_path,
                start=s_block["start"] if s_block else mseg.start,
                end=e_block["end"] if e_block else mseg.end,
                reason=mseg.reason,
            )
            sentence_snapped.append(snapped)

        n_before = len(merged)
        n_after = len(sentence_snapped)
        logger.info(
            "Sentence-block snap: %d/%d segments aligned to sentence boundaries",
            n_after, n_before,
        )
        merged = sentence_snapped

    return merged
