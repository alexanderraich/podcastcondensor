"""Master cut — select segments from themes and assemble audio.

Phases (within the master-cut pipeline):
  1. Download all episode artefacts (delegated to download_pool)
  2. Build / ensure universe state with word_ranges → timestamp segments
  3. Extract themes from universe state (delegated to theme_extraction)
  4. Resolve segments from universe state (Phase 2 word_ranges → timestamps)
  5. Select segments within time budget (knapsack, this module)
  6. Assemble master cut audio with dual beeps (this module)

The top-level ``build_master_cut()`` orchestrates all phases.
"""

import json
import logging
import os
import shutil
import subprocess
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Optional

from podcastcondensor.audio_strategies import (
    _ionice_cmd,
    _atempo_filters,
    _concat_batch_files,
)
from podcastcondensor.config import Config
from podcastcondensor.download_pool import (
    EpisodeManifest,
    ensure_all_episode_artifacts,
)
from podcastcondensor.llm.deepseek import resolve_api_key, DeepSeekClient
from podcastcondensor.global_state import build_global_state as run_global_state
from podcastcondensor.segmentation.sentence_units import build_transcript_from_entries
from podcastcondensor.subtitles import load_subtitles
from podcastcondensor.theme_extraction import Theme, extract_themes
from podcastcondensor.universe_state import UniverseState

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data classes for theme segments (replaces deleted theme_mapping module)
# ---------------------------------------------------------------------------


@dataclass
class ThemeSegment:
    """A continuous audio segment related to a theme, stored in universe state."""
    theme_id: str = ""
    episode_number: int = 0
    audio_path: str = ""
    start: float = 0.0
    end: float = 0.0
    text_preview: str = ""
    is_intro: bool = False
    relevance_score: float = 0.0
    match_count: int = 0

    @property
    def duration(self) -> float:
        return max(self.end - self.start, 0.0)


@dataclass
class ThemeWithSegments:
    """A theme with all its discovered audio segments (from universe state)."""
    theme: Theme
    segments: List[ThemeSegment] = field(default_factory=list)

    @property
    def total_duration(self) -> float:
        return sum(s.duration for s in self.segments)


# ---------------------------------------------------------------------------
# Selection data classes
# ---------------------------------------------------------------------------


@dataclass
class Selection:
    """A segment selected for the master cut with assembly metadata."""
    segment: ThemeSegment
    theme_title: str
    theme_id: str
    beep_before: str = "single"  # "single" | "triple" | "none"


@dataclass
class MasterCutPlan:
    """Complete plan for the master cut assembly."""
    selections: List[Selection]
    total_duration: float
    theme_allocations: Dict[str, float]  # theme_id -> allocated seconds
    coverage: Dict[str, float]  # theme_id -> fraction of its available content used


# ---------------------------------------------------------------------------
# Phase 4: Resolve theme segments from universe state
# ---------------------------------------------------------------------------


def resolve_theme_segments_from_state(
    themes: List[Theme],
    universe_data: dict,
    manifests: List[EpisodeManifest],
    output_root: str = "",
) -> List[ThemeWithSegments]:
    """Resolve audio segments for each theme from universe state.

    Each theme has ``related_item_ids`` (IDs of concepts, entities, claims).
    Each universe-state item with one of those IDs carries a ``segments``
    array (pre-computed from Phase 2 word_ranges → timestamp conversion).

    This function simply reads those pre-computed segments and wraps them
    in ThemeSegment objects for the selection phase. No keyword grep, no
    text search — the segments are already correct by construction.

    Returns a ``ThemeWithSegments`` per theme (empty segments list if
    the theme's items have no segment data).
    """
    # Build lookup: item_id -> item from all universe state categories.
    # DeepSeek sometimes classifies the same content under different
    # categories (e.g. a theological argument as a "concept" in one
    # episode and as a "claim" in another, or the same segment appearing
    # in both). Including all categories guarantees we never miss a
    # candidate segment. The ep:start:end dedup below handles overlap.
    items_by_id: Dict[str, dict] = {}
    for category in ("concepts", "claims", "entities",
                     "scriptural_links", "glossary"):
        for item in universe_data.get(category, []):
            iid = item.get("id")
            if iid:
                items_by_id[iid] = item

    # Build lookup: episode_number -> audio_path from manifests
    ep_to_audio: Dict[int, str] = {}
    for m in manifests:
        ep_to_audio[m.episode_number] = m.audio_path

    results: List[ThemeWithSegments] = []

    for theme in themes:
        segments_map: Dict[str, ThemeSegment] = {}  # dedup by "ep:start:end"
        for item_id in theme.related_item_ids:
            item = items_by_id.get(item_id)
            if not item:
                continue
            for seg in item.get("segments", []):
                if not isinstance(seg, dict):
                    continue
                ep = seg.get("episode", 0)
                start = seg.get("start", 0)
                end = seg.get("end", 0)
                if end <= start:
                    continue
                key = f"{ep}:{start:.1f}:{end:.1f}"
                if key in segments_map:
                    continue
                ts = ThemeSegment(
                    theme_id=theme.id,
                    episode_number=ep,
                    audio_path=ep_to_audio.get(ep, ""),
                    start=start,
                    end=end,
                    text_preview="",
                    relevance_score=1.0,
                    match_count=1,
                )
                segments_map[key] = ts

        theme_segments = list(segments_map.values())

        # Mark intro: earliest chronological segment
        if theme_segments:
            theme_segments.sort(key=lambda s: (s.episode_number, s.start, s.end))
            theme_segments[0].is_intro = True

            # Compute relevance: longer segments get higher score
            # (they represent more substantive LLM-identified content)
            for ts in theme_segments:
                ts.relevance_score = round(min(ts.duration / 30.0, 5.0), 2)

        results.append(ThemeWithSegments(theme=theme, segments=theme_segments))

        logger.info(
            "  %s: %d segments from %d items (total %.0fs)",
            theme.id, len(theme_segments),
            sum(1 for iid in theme.related_item_ids if iid in items_by_id),
            sum(s.duration for s in theme_segments),
        )

    return results


# ---------------------------------------------------------------------------
# Phase 5 quality warnings
# ---------------------------------------------------------------------------


def _compute_selection_warnings(
    selections: List[Selection],
    themes: list,
) -> List[str]:
    """Compute quality warnings for a master cut selection plan.

    Flags:
      - Segments within first 3 minutes of episode → likely intro/banter
      - Segments shorter than 15 seconds → too short to convey meaning
      - Segments longer than 600 seconds (10 min) → too broad, may include
        unrelated content

    Returns list of warning strings (empty = all clean).
    """
    warnings: List[str] = []
    theme_lookup = {t.id: t for t in themes}

    for i, sel in enumerate(selections):
        ep = sel.segment.episode_number
        start = sel.segment.start
        dur = sel.segment.duration

        if start < 180:
            warnings.append(
                f"Selection {i+1}: '{sel.theme_title}' ep {ep} "
                f"starts at {start:.0f}s (<3min) — likely intro/banter"
            )

        if dur < 15:
            warnings.append(
                f"Selection {i+1}: '{sel.theme_title}' ep {ep} "
                f"is {dur:.0f}s long — too short to convey meaning"
            )

        if dur > 600:
            theme = theme_lookup.get(sel.theme_id)
            tname = theme.title if theme else sel.theme_id
            warnings.append(
                f"Selection {i+1}: '{tname}' ep {ep} "
                f"is {dur:.0f}s (>10 min) — likely too broad"
            )

    return warnings


# ---------------------------------------------------------------------------
# Phase 5: Per-theme LLM segment selection
# ---------------------------------------------------------------------------


def _select_segments_knapsack(
    themes_with_segments: List[ThemeWithSegments],
    target_duration: float = 6750,
    min_segment: float = 15.0,
) -> MasterCutPlan:
    """Fallback: select segments via knapsack (budget-filling).

    Used when LLM selection is unavailable or fails. Sorts by importance,
    fills proportional time budget per theme. No narrative coherence or
    boundary refinement.
    """
    if not themes_with_segments:
        return MasterCutPlan(
            selections=[], total_duration=0,
            theme_allocations={}, coverage={},
        )

    sorted_tws = sorted(
        themes_with_segments,
        key=lambda t: (-t.theme.importance, -len(t.segments)),
    )
    for tws in sorted_tws:
        tws.segments.sort(key=lambda s: (s.episode_number, s.start))

    all_durations = [s.duration for tws in sorted_tws for s in tws.segments if s.duration > 0]
    avg_seg = sum(all_durations) / len(all_durations) if all_durations else 60.0
    min_segs = max(3, min(10, int(target_duration / 500)))
    max_themes = max(4, min(12, int(target_duration / 360)))
    total_importance = sum(t.theme.importance for t in sorted_tws)
    if total_importance <= 0:
        total_importance = len(sorted_tws)

    selections: List[Selection] = []
    total_used = 0.0
    themes_included = 0

    for tws in sorted_tws:
        if not tws.segments:
            continue
        if themes_included >= max_themes:
            break
        if total_used >= target_duration:
            break

        budget = target_duration * (tws.theme.importance / total_importance)
        theme_segs: List[Selection] = []
        theme_used = 0.0
        for seg in tws.segments:
            if seg.duration < min_segment:
                continue
            if total_used + seg.duration > target_duration:
                break
            if theme_used + seg.duration > budget and len(theme_segs) >= min_segs:
                break
            beep = "none" if not selections and not theme_segs else "single"
            theme_segs.append(Selection(
                segment=seg, theme_title=tws.theme.title,
                theme_id=seg.theme_id, beep_before=beep,
            ))
            theme_used += seg.duration
            total_used += seg.duration

        if len(theme_segs) < min_segs:
            total_used -= theme_used
            continue

        theme_segs[0].beep_before = "none" if not selections else "triple"
        selections.extend(theme_segs)
        themes_included += 1

    total = sum(s.segment.duration for s in selections)
    included = len(set(s.theme_id for s in selections))
    logger.info(
        "Knapsack fallback: %d selections from %d themes "
        "(min %d/theme, avg seg %.0fs), total %.0fs (target %.0fs)",
        len(selections), included, min_segs, avg_seg, total, target_duration,
    )
    return MasterCutPlan(
        selections=selections,
        total_duration=round(total, 1),
        theme_allocations={}, coverage={},
    )


def select_segments_for_master_cut(
    themes_with_segments: List[ThemeWithSegments],
    manifests: List[EpisodeManifest],
    output_root: str,
    client=None,
    model: str = "deepseek-chat",
    timeout: int = 600,
    target_duration: float = 6750,
) -> MasterCutPlan:
    """Select segments via per-theme LLM selection with transcript context.

    For each theme (in importance order), sends a prompt showing each
    candidate segment with ~30s of surrounding SRT transcript context.
    The LLM decides keep/drop and refines boundaries (widening to capture
    complete thoughts). Kept segments are merged across adjacent gaps.

    Each theme's proportional share of the target duration is stated in the
    prompt (time_budget) so the LLM self-regulates volume. All kept segments
    across all themes are concatenated — there is no Python truncation loop.

    Falls back to knapsack for any theme where the LLM call fails or
    returns no valid decisions.

    Returns a MasterCutPlan ready for audio assembly.
    """
    from podcastcondensor.minimal_theme_cut import (
        build_selection_prompt,
        parse_selection_response,
        apply_decisions,
    )

    if not themes_with_segments:
        return MasterCutPlan(
            selections=[], total_duration=0,
            theme_allocations={}, coverage={},
        )

    sorted_tws = sorted(
        themes_with_segments,
        key=lambda t: (-t.theme.importance, -len(t.segments)),
    )

    selections: List[Selection] = []

    # Per-theme proportional time budget, enforced by the LLM (in the prompt)
    # rather than by a Python truncation loop. Each theme's share of the master
    # cut is target × importance/total_importance; the prompt states it and the
    # LLM decides volume. Every theme's kept segments are concatenated — no
    # greedy global cap that starves later themes.
    active_tws = [t for t in sorted_tws if t.segments]
    total_importance = sum(t.theme.importance for t in active_tws) or len(active_tws)
    budgets = {
        t.theme.id: target_duration * t.theme.importance / total_importance
        for t in active_tws
    }

    for tws in sorted_tws:
        if not tws.segments:
            continue

        budget = budgets.get(tws.theme.id, 0.0)
        logger.info(
            "LLM selection for theme '%s' (%d candidates, budget %.0fs)...",
            tws.theme.id, len(tws.segments), budget,
        )

        # Build prompt with transcript context + proportional time budget
        prompt = build_selection_prompt(
            theme=tws.theme,
            tws=tws,
            output_root=output_root,
            manifests=manifests,
            context_buffer=30.0,
            time_budget=budget,
        )

        # Call LLM
        try:
            raw = client.generate(
                prompt=prompt,
                model=model,
                timeout=timeout,
                temperature=0.3,
                max_tokens=8192,
                force_json=True,
            )
        except Exception as e:
            logger.warning(
                "LLM selection failed for '%s': %s — falling back to knapsack",
                tws.theme.id, e,
            )
            continue

        decisions = parse_selection_response(raw)
        if not decisions:
            logger.warning(
                "No valid decisions for '%s' — falling back to knapsack",
                tws.theme.id,
            )
            continue

        refined = apply_decisions(decisions, tws, manifests, output_root=output_root)

        if not refined:
            logger.info("LLM kept 0 segments for '%s' — skipping theme", tws.theme.id)
            continue

        # Convert RefinedSelection -> Selection. No global truncation — the
        # per-theme budget was given to the LLM in the prompt; keep whatever
        # the LLM deemed worthy for this theme.
        theme_selections: List[Selection] = []
        theme_used = 0.0
        for rs in refined:
            ts = ThemeSegment(
                theme_id=tws.theme.id,
                episode_number=rs.episode_number,
                audio_path=rs.audio_path,
                start=rs.start,
                end=rs.end,
                text_preview=rs.reason[:120],
                relevance_score=min(rs.duration / 30.0, 5.0),
            )
            theme_selections.append(Selection(
                segment=ts,
                theme_title=tws.theme.title,
                theme_id=tws.theme.id,
                beep_before="none",
            ))
            theme_used += rs.duration

        # Assign beeps: triple before first segment of each new theme
        if theme_selections:
            if selections:
                theme_selections[0].beep_before = "triple"
            for sel in theme_selections[1:]:
                sel.beep_before = "single"

        selections.extend(theme_selections)
        logger.info(
            "  → %d segments, %.0fs (%.1f min)",
            len(theme_selections), theme_used, theme_used / 60,
        )

    total = sum(s.segment.duration for s in selections)
    included = len(set(s.theme_id for s in selections))

    # Fallback: if LLM produced no selections, use knapsack
    if not selections and themes_with_segments:
        logger.warning("LLM selection produced no results — falling back to knapsack")
        return _select_segments_knapsack(
            themes_with_segments,
            target_duration=target_duration,
        )

    logger.info(
        "Master cut plan: %d segments from %d themes, "
        "total %.0fs (target %.0fs)",
        len(selections), included, total, target_duration,
    )

    return MasterCutPlan(
        selections=selections,
        total_duration=round(total, 1),
        theme_allocations={}, coverage={},
    )


# ---------------------------------------------------------------------------
# Phase 6: Audio assembly (multi-source with varied beeps)
# ---------------------------------------------------------------------------


def _extract_segment(
    source_audio: str,
    start: float,
    end: float,
    output_path: str,
    sample_rate: int = 22050,
    bitrate: str = "64k",
) -> None:
    """Extract one audio segment with consistent encoding.

    Re-encodes to ensure consistent format for concat demuxer.
    """
    duration = max(end - start, 0.01)
    cmd = [
        "ffmpeg", "-y",
        "-ss", f"{start:.3f}",
        "-i", source_audio,
        "-t", f"{duration:.3f}",
        "-ar", str(sample_rate),
        "-b:a", bitrate,
        "-ac", "1",
        output_path,
    ]
    result = subprocess.run(
        _ionice_cmd(cmd), capture_output=True, timeout=120,
    )
    if result.returncode != 0:
        # ffmpeg may output non-UTF-8 bytes to stderr; decode safely
        err_text = result.stderr.decode("utf-8", errors="replace")[:300]
        raise RuntimeError(
            f"Segment extraction failed ({source_audio}, {start:.1f}-{end:.1f}): "
            f"{err_text}"
        )


def _generate_beep_file(
    output_dir: str,
    filename: str,
    freq: float = 1000,
    tone_duration: float = 0.25,
    count: int = 1,
    gap: float = 0.25,
    sample_rate: int = 22050,
    bitrate: str = "64k",
) -> str:
    """Generate a beep audio file (single or multi-pulse).

    For single beep (count=1): a 250ms sine tone.
    For triple beep (count=3): three 250ms tones separated by 250ms
    silence, concatenated via the concat demuxer.

    Returns path to the generated file.
    """
    output_path = os.path.join(output_dir, filename)

    if count == 1:
        cmd = [
            "ffmpeg", "-y",
            "-f", "lavfi",
            "-i", f"sine=f={freq}:d={tone_duration}",
            "-ar", str(sample_rate),
            "-b:a", bitrate,
            "-ac", "1",
            output_path,
        ]
        result = subprocess.run(
            _ionice_cmd(cmd), capture_output=True, text=True, timeout=60,
        )
        if result.returncode != 0:
            raise RuntimeError(f"Beep generation failed: {result.stderr[:200]}")
        return output_path

    # Multi-pulse beep: generate individual beeps + silences, concat
    # Generate silence segment
    silence_path = os.path.join(output_dir, "_silence.mp3")
    silence_cmd = [
        "ffmpeg", "-y",
        "-f", "lavfi",
        "-i", f"anullsrc=r={sample_rate}:cl=mono:d={gap}",
        "-b:a", bitrate,
        "-ac", "1",
        silence_path,
    ]
    r = subprocess.run(_ionice_cmd(silence_cmd), capture_output=True, text=True, timeout=30)
    if r.returncode != 0:
        # Fallback: use tiny sine at very low volume
        silence_cmd = [
            "ffmpeg", "-y",
            "-f", "lavfi",
            "-i", f"sine=f=100:d={gap}:volume=0.01",
            "-ar", str(sample_rate),
            "-b:a", bitrate,
            "-ac", "1",
            silence_path,
        ]
        subprocess.run(_ionice_cmd(silence_cmd), capture_output=True, text=True, timeout=30)

    # Generate individual beeps
    beep_paths = []
    for i in range(count):
        bp = os.path.join(output_dir, f"_beep_{i}.mp3")
        beep_cmd = [
            "ffmpeg", "-y",
            "-f", "lavfi",
            "-i", f"sine=f={freq}:d={tone_duration}",
            "-ar", str(sample_rate),
            "-b:a", bitrate,
            "-ac", "1",
            bp,
        ]
        r = subprocess.run(_ionice_cmd(beep_cmd), capture_output=True, text=True, timeout=30)
        if r.returncode != 0:
            raise RuntimeError(f"Pulse beep {i} failed: {r.stderr[:200]}")
        beep_paths.append(bp)

    # Build concat list: beep, silence, beep, silence, ..., beep
    concat_file = os.path.join(output_dir, "_concat.txt")
    with open(concat_file, "w") as f:
        for i, bp in enumerate(beep_paths):
            if i > 0:
                f.write(f"file '{os.path.abspath(silence_path)}'\n")
            f.write(f"file '{os.path.abspath(bp)}'\n")

    # Concat pass
    concat_cmd = [
        "ffmpeg", "-y",
        "-f", "concat",
        "-safe", "0",
        "-i", concat_file,
        "-ar", str(sample_rate),
        "-b:a", bitrate,
        "-ac", "1",
        output_path,
    ]
    r = subprocess.run(_ionice_cmd(concat_cmd), capture_output=True, text=True, timeout=60)
    if r.returncode != 0:
        raise RuntimeError(f"Multi-beep concat failed: {r.stderr[:200]}")

    return output_path


def assemble_master_cut(
    selections: List[Selection],
    output_path: str,
    *,
    sample_rate: int = 22050,
    bitrate: str = "64k",
    speed: float = 1.25,
    parallel_workers: int = 4,
    keep_temp: bool = False,
) -> str:
    """Assemble master cut audio from selected segments.

    Extracts each segment from its source audio file, then concatenates
    with appropriate beeps (single within-theme, triple between-themes).

    Returns output_path on success.
    """
    if not selections:
        raise ValueError("No selections to assemble")

    t0 = time.time()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    tmpdir = tempfile.mkdtemp(prefix="master_cut_")
    atempo = _atempo_filters(speed)

    try:
        # ── Generate beep files ────────────────────────────────────────
        logger.info("Generating beep files...")
        single_beep = _generate_beep_file(
            tmpdir, "beep_single.mp3",
            freq=1000, tone_duration=0.25, count=1,
            sample_rate=sample_rate, bitrate=bitrate,
        )
        triple_beep = _generate_beep_file(
            tmpdir, "beep_triple.mp3",
            freq=1000, tone_duration=0.25, count=3, gap=0.25,
            sample_rate=sample_rate, bitrate=bitrate,
        )

        # ── Extract all segments (sequential, memory-safe) ─────────────
        seg_paths: List[str] = []
        total = len(selections)
        logger.info("Extracting %d segments...", total)

        # First pass: build segment path list
        for i, sel in enumerate(selections):
            seg_path = os.path.join(tmpdir, f"seg_{i:04d}.mp3")
            seg_paths.append(seg_path)

        # Extract in parallel with staggered start
        completed = 0

        def _extract_one(idx: int) -> str:
            sel = selections[idx]
            seg_path = seg_paths[idx]
            try:
                _extract_segment(
                    sel.segment.audio_path,
                    sel.segment.start,
                    sel.segment.end,
                    seg_path,
                    sample_rate=sample_rate,
                    bitrate=bitrate,
                )
                return idx, None
            except Exception as e:
                return idx, str(e)

        with ThreadPoolExecutor(max_workers=parallel_workers) as pool:
            futures = {pool.submit(_extract_one, i): i for i in range(total)}
            for future in as_completed(futures):
                idx, error = future.result()
                if error:
                    logger.error("  Segment %d/%d failed: %s", idx + 1, total, error)
                    raise RuntimeError(f"Segment {idx} extraction: {error}")
                completed += 1
                if completed % 20 == 0 or completed == total:
                    logger.info("  Extracted %d/%d segments", completed, total)

        # ── Build interleaved concat list with beeps ───────────────────
        logger.info("Building concat list with beeps...")
        interleaved: List[str] = []
        for i, sel in enumerate(selections):
            if i > 0:
                if sel.beep_before == "triple":
                    interleaved.append(triple_beep)
                else:
                    interleaved.append(single_beep)
            interleaved.append(seg_paths[i])

        # Count beep types for logging
        triple_count = sum(
            1 for s in selections if s.beep_before == "triple"
        )
        single_count = sum(
            1 for s in selections if s.beep_before == "single"
        )
        logger.info(
            "Concat: %d segments + %d single beeps + %d triple beeps",
            len(selections), single_count, triple_count,
        )

        # ── Concat pass ────────────────────────────────────────────────
        logger.info("Running final concat pass...")
        _concat_batch_files(
            batch_paths=interleaved,
            output_path=output_path,
            sample_rate=sample_rate,
            bitrate=bitrate,
            atempo=atempo,
            beep=False,  # beeps are already in the list
        )

        elapsed = time.time() - t0
        logger.info(
            "Master cut assembled: %.1fs real time → %.0fs output (%s)",
            elapsed, sum(s.segment.duration for s in selections) / max(speed, 1),
            output_path,
        )
        return output_path

    finally:
        if not keep_temp:
            shutil.rmtree(tmpdir, ignore_errors=True)
        else:
            logger.info("Keeping temp dir: %s", tmpdir)


# ---------------------------------------------------------------------------
# Full pipeline orchestration
# ---------------------------------------------------------------------------


def build_master_cut(
    playlist_url: str,
    cfg: Config,
    state_file: str = "",
    output_path: str = "master_cut.mp3",
    target_duration: int = 6750,
    start_episode: int = 1,
    end_episode: int = 140,
    *,
    parallel_downloads: int = 4,
) -> dict:
    """Build a master cut across all episodes.

    Always uses whisper transcription — YouTube subtitles are unreliable.

    Args:
        playlist_url: YouTube playlist URL.
        cfg: Pipeline configuration.
        state_file: Path to existing/desired universe state file. If empty,
                    uses output/universe_state_{START}_{END}.json (range-scoped).
        output_path: Output master cut audio path.
        target_duration: Target duration in seconds (default 6750 = 90min at 1.25x).
        start_episode: First episode to include.
        end_episode: Last episode to include (0 = auto).
        parallel_downloads: Parallel download workers.

    Returns:
        Dict with keys: phases (list), plan, output_path, errors.
    """
    overall_t0 = time.time()
    result = {
        "phases": [],
        "errors": [],
        "output_path": None,
    }

    output_root = cfg.output_root or os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "output",
    )

    # Ensure output root exists
    Path(output_root).mkdir(parents=True, exist_ok=True)

    # ── Phase 1: Download pool ──────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("PHASE 1: Download audio + subtitles (parallel=%d)", parallel_downloads)
    logger.info("=" * 60)
    t1 = time.time()

    try:
        manifests = ensure_all_episode_artifacts(
            playlist_url=playlist_url,
            output_root=output_root,
            start_episode=start_episode,
            end_episode=end_episode,
            parallel=parallel_downloads,
            audio_format=cfg.audio_format,
            audio_bitrate=cfg.audio_bitrate,
            whisper_model=cfg.whisper_model,
        )
    except RuntimeError as e:
        result["errors"].append(str(e))
        result["phases"].append({
            "phase": "download",
            "elapsed_sec": round(time.time() - t1, 1),
            "episodes_downloaded": 0,
            "error": str(e),
        })
        logger.error("Download phase aborted: %s", e)
        return result
    result["phases"].append({
        "phase": "download",
        "elapsed_sec": round(time.time() - t1, 1),
        "episodes_downloaded": len(manifests),
    })

    if not manifests:
        result["errors"].append("No episodes downloaded — cannot continue")
        return result

    # ── Phase 2: Build / ensure universe state ─────────────────────────
    logger.info("=" * 60)
    logger.info("PHASE 2: Build universe state (ephemeral, range-scoped)")
    logger.info("=" * 60)
    t2 = time.time()

    # Ephemeral universe state: delete stale cumulative file and rebuild
    # fresh from existing global_state.json on disk for the episode range
    # only. This guarantees theme extraction and segment resolution only
    # see content from the target window — no leakage from older episodes.
    if not state_file:
        state_file = os.path.join(
            output_root,
            f"universe_state_{start_episode:03d}_{end_episode:03d}.json",
        )
    if os.path.exists(state_file):
        os.remove(state_file)
        logger.info("Removed stale cumulative universe state (ephemeral per range)")
    state = UniverseState(state_file)  # fresh empty state

    api_key = resolve_api_key()
    if not api_key:
        result["errors"].append("DeepSeek API key not set")
        return result
    ds_client = DeepSeekClient(api_key=api_key)

    loaded_from_disk = 0
    deepseek_calls = 0

    for m in manifests:
        ep_dir = os.path.join(output_root, f"ep-{m.episode_number:03d}")
        gs_path = os.path.join(ep_dir, "global_state.json")

        if os.path.exists(gs_path):
            # Already has global state from a previous run — re-read from disk
            with open(gs_path) as f:
                global_data = json.load(f)
        else:
            # Run Phase 2 (DeepSeek call)
            logger.info(
                "Ep %d: extracting global state (DeepSeek)...",
                m.episode_number,
            )
            try:
                cleaned = load_subtitles(m.srt_path, reindex=True)
                transcript_text = build_transcript_from_entries(cleaned)

                global_data = run_global_state(
                    transcript_text=transcript_text,
                    episode_title=m.title,
                    episode_number=m.episode_number,
                    client=ds_client,
                    model=cfg.deepseek_model,
                    timeout=cfg.deepseek_timeout,
                    srt_entries=cleaned,
                )

                # Write checkpoint
                with open(gs_path, "w") as f:
                    json.dump(global_data, f, ensure_ascii=False, indent=2)
                deepseek_calls += 1
            except Exception as e:
                logger.error("Phase 2 failed for ep %d: %s", m.episode_number, e)
                global_data = None

        if global_data:
            knowledge = {
                "summary": global_data.get("summary", ""),
                "entities": global_data.get("entities", []),
                "concepts": global_data.get("concepts", []),
                "claims": global_data.get("claims", []),
                "scriptural_links": global_data.get("scriptural_links", []),
                "glossary": global_data.get("glossary", []),
            }
            state.add_episode_knowledge(m.episode_number, knowledge)
            loaded_from_disk += 1

    # Force-save state (should already be saved by add_episode_knowledge)
    state.save()

    # Refresh state data from file (to get the latest merged state)
    state.load()

    result["phases"].append({
        "phase": "build_universe",
        "elapsed_sec": round(time.time() - t2, 1),
        "loaded_from_disk": loaded_from_disk,
        "deepseek_calls": deepseek_calls,
        "total_episodes_in_state": len(state.data.get("episode_summaries", [])),
    })
    logger.info(
        "Universe state: %d concepts, %d entities, %d claims",
        len(state.data.get("concepts", [])),
        len(state.data.get("entities", [])),
        len(state.data.get("claims", [])),
    )

    # ── Phase 3: Extract themes ─────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("PHASE 3: Extract core themes from universe state")
    logger.info("=" * 60)
    t3 = time.time()

    themes = extract_themes(
        universe_data=state.data,
        client=ds_client,
        model=cfg.deepseek_model,
        timeout=cfg.deepseek_timeout,
    )
    result["phases"].append({
        "phase": "extract_themes",
        "elapsed_sec": round(time.time() - t3, 1),
        "theme_count": len(themes),
        "theme_ids": [t.id for t in themes],
    })

    if not themes:
        result["errors"].append("No themes extracted — cannot continue")
        return result

    for t in themes:
        logger.info("  Theme: %s (%.2f, %d items)", t.title, t.importance, len(t.related_item_ids))

    # ── Phase 4: Resolve segments from universe state ──────────────────
    logger.info("=" * 60)
    logger.info("PHASE 4: Resolve segments from universe state")
    logger.info("=" * 60)
    t4 = time.time()

    themes_with_segments = resolve_theme_segments_from_state(
        themes=themes,
        universe_data=state.data,
        manifests=manifests,
        output_root=output_root,
    )
    total_segments = sum(len(tws.segments) for tws in themes_with_segments)
    total_available = sum(s.duration for tws in themes_with_segments for s in tws.segments)
    result["phases"].append({
        "phase": "resolve_segments",
        "elapsed_sec": round(time.time() - t4, 1),
        "total_segments": total_segments,
        "total_available_sec": round(total_available, 1),
    })
    logger.info(
        "Segment resolution: %d total segments, %.0fs available content",
        total_segments, total_available,
    )

    # ── Phase 5: Select segments via per-theme LLM ────────────────────────
    logger.info("=" * 60)
    logger.info("PHASE 5: Per-theme LLM segment selection (target=%.0fs = %.1fh)",
                target_duration, target_duration / 3600)
    logger.info("=" * 60)
    t5 = time.time()

    plan = select_segments_for_master_cut(
        themes_with_segments,
        manifests=manifests,
        output_root=output_root,
        client=ds_client,
        model=cfg.deepseek_model,
        timeout=cfg.deepseek_timeout,
        target_duration=float(target_duration),
    )
    result["phases"].append({
        "phase": "select_segments",
        "elapsed_sec": round(time.time() - t5, 1),
        "selected_count": len(plan.selections),
        "total_duration_sec": plan.total_duration,
        "target_duration_sec": target_duration,
    })
    logger.info(
        "Selection: %d segments, %.0fs (target %.0fs)",
        len(plan.selections), plan.total_duration, target_duration,
    )

    if not plan.selections:
        result["errors"].append("No segments selected — cannot continue")
        return result

    # ── Quality warnings ──────────────────────────────────────────────────
    warnings = _compute_selection_warnings(plan.selections, themes)
    result["warnings"] = warnings
    if warnings:
        logger.info("Selection warnings (%d):", len(warnings))
        for w in warnings:
            logger.info("  ⚠ %s", w)

    # Log final plan
    for s in plan.selections[:5]:
        logger.info(
            "  %s (ep %d, %.1fs-%.1fs) [%s]",
            s.theme_title[:50],
            s.segment.episode_number,
            s.segment.start, s.segment.end,
            s.beep_before,
        )
    if len(plan.selections) > 5:
        logger.info("  ... (%d more)", len(plan.selections) - 5)

    # ── Phase 6: Audio assembly ─────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("PHASE 6: Assemble master cut audio")
    logger.info("=" * 60)

    # Resolve absolute output path
    if not os.path.isabs(output_path):
        output_path = os.path.join(output_root, output_path)

    t6 = time.time()
    try:
        assemble_master_cut(
            selections=plan.selections,
            output_path=output_path,
            sample_rate=cfg.audio_sample_rate,
            bitrate=cfg.audio_bitrate,
            speed=cfg.audio_speed,
            parallel_workers=parallel_downloads,
            keep_temp=cfg.keep_temp,
        )
        result["output_path"] = output_path
    except Exception as e:
        logger.exception("Master cut assembly failed: %s", e)
        result["errors"].append(f"Audio assembly failed: {e}")

    result["phases"].append({
        "phase": "assemble_audio",
        "elapsed_sec": round(time.time() - t6, 1),
        "output_path": output_path,
    })

    # Summary
    total_elapsed = time.time() - overall_t0
    logger.info("=" * 60)
    logger.info("MASTER CUT COMPLETE — total %.1fs (%.1fmin)",
                total_elapsed, total_elapsed / 60)
    logger.info("  Output: %s", output_path)
    logger.info("  Duration: %.0fs (%.1fh)", plan.total_duration, plan.total_duration / 3600)
    logger.info("  Themes: %d", len(themes))
    logger.info("  Segments: %d", len(plan.selections))

    warnings = result.get("warnings", [])
    if warnings:
        logger.info("  Warnings: %d", len(warnings))
        for w in warnings:
            logger.info("    ⚠ %s", w)
    else:
        logger.info("  Warnings: none (clean)")

    if result["errors"]:
        logger.info("  Errors: %d", len(result["errors"]))
    logger.info("=" * 60)

    # Write range-scoped master_cut_stats JSON alongside the audio output,
    # with the full selection trace so each batch's kept segments are recorded.
    if result["output_path"]:
        stats_path = os.path.join(
            os.path.dirname(os.path.abspath(result["output_path"])),
            f"master_cut_stats_{start_episode:03d}_{end_episode:03d}.json",
        )
        try:
            with open(stats_path, "w") as f:
                json.dump({
                    "total_duration_sec": plan.total_duration,
                    "target_duration_sec": target_duration,
                    "themes": len(themes),
                    "segments": len(plan.selections),
                    "warnings": warnings,
                    "errors": result.get("errors", []),
                    "phases": result.get("phases", []),
                    "selections": [
                        {
                            "theme_title": s.theme_title,
                            "theme_id": s.theme_id,
                            "episode": s.segment.episode_number,
                            "start": round(s.segment.start, 1),
                            "end": round(s.segment.end, 1),
                            "duration": round(s.segment.duration, 1),
                            "beep_before": s.beep_before,
                        }
                        for s in plan.selections
                    ],
                }, f, indent=2)
            logger.info("Stats written: %s", stats_path)
        except OSError as e:
            logger.warning("Failed to write stats: %s", e)

    return result
