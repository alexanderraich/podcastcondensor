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
    # Build lookup: item_id -> item from universe state categories.
    # Only use concepts for segment resolution — entities include the hosts
    # themselves (0-84s intros), claims are too granular, scriptural_links
    # are reference citations. Concepts are the actual theological topics
    # with grounded audio position data.
    items_by_id: Dict[str, dict] = {}
    for item in universe_data.get("concepts", []):
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
# Phase 5: Duration-optimised segment selection
# ---------------------------------------------------------------------------


def _compute_min_segments(target_duration: float) -> tuple:
    """Compute (min_segments_per_theme, max_themes) for a listenable result.

    Target: each theme gets ~5-7 minutes of content (4-6 segments @~70s).
    Number of themes scales with target:
      - 42min (2520s): ~5-7 themes
      - 3.5h  (12600s): ~10-12 themes

    Returns (min_seg, max_themes) where:
      - min_seg: minimum segments per theme to include it
      - max_themes: maximum themes allowed
    """
    min_seg = max(3, min(10, int(target_duration / 500)))
    max_themes = max(4, min(12, int(target_duration / 360)))
    return min_seg, max_themes


def select_segments_for_master_cut(
    themes_with_segments: List[ThemeWithSegments],
    target_duration: float = 12600,  # 3.5 hours in seconds
    min_segment: float = 15.0,
) -> MasterCutPlan:
    """Select segments to fill target_duration with thematic depth.

    Algorithm:
      1. Sort themes by importance descending; sort segments chronologically
         within each theme so they form coherent mini-narratives.
      2. Round 1 — breadth pass: take MIN_SEGMENTS from each theme (most
         important first) until we run out of time. This ensures every
         included theme has enough depth to be listenable (≥2 segs).
      3. Round 2 — depth pass: fill remaining time budget with more
         segments from the already-included themes (chronological order).
      4. Beep assignment: single within a theme, triple between themes.

    MIN_SEGMENTS scales with target_duration:
      - 600s (10 min):  min=2  →  top 3-4 themes, 2 segs each
      - 12600s (3.5h):  min=8  →  all 15 themes, 8+ segs each

    Returns a MasterCutPlan ready for audio assembly.
    """
    if not themes_with_segments:
        return MasterCutPlan(
            selections=[], total_duration=0,
            theme_allocations={}, coverage={},
        )

    # Sort themes by importance descending
    sorted_tws = sorted(
        themes_with_segments,
        key=lambda t: (-t.theme.importance, -len(t.segments)),
    )
    # Sort each theme's segments chronologically
    for tws in sorted_tws:
        tws.segments.sort(key=lambda s: (s.episode_number, s.start))

    all_durations = [s.duration for tws in sorted_tws for s in tws.segments if s.duration > 0]
    avg_seg = sum(all_durations) / len(all_durations) if all_durations else 60.0
    min_segs, max_themes = _compute_min_segments(target_duration)

    total_importance = sum(t.theme.importance for t in sorted_tws)
    if total_importance <= 0:
        total_importance = len(sorted_tws)

    selections: List[Selection] = []
    total_used = 0.0
    prev_theme_id: Optional[str] = None
    themes_included = 0

    # Single pass: each theme takes min_segs at minimum, then as many more
    # as fit within its proportional budget, all in one contiguous block.
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

        # Commit the theme's contiguous block
        theme_segs[0].beep_before = "none" if not selections else "triple"
        selections.extend(theme_segs)
        prev_theme_id = tws.theme.id
        themes_included += 1

    total = sum(s.segment.duration for s in selections)
    included = len(set(s.theme_id for s in selections))

    logger.info(
        "Master cut plan: %d selections from %d themes "
        "(min %d/theme, avg seg %.0fs), total %.0fs (target %.0fs)",
        len(selections), included, min_segs, avg_seg, total, target_duration,
    )
    by_theme: Dict[str, List[Selection]] = {}
    for s in selections:
        by_theme.setdefault(s.theme_id, []).append(s)
    for tid, segs in by_theme.items():
        used = sum(s.segment.duration for s in segs)
        tws = next(t for t in sorted_tws if t.theme.id == tid)
        available = sum(s.duration for s in tws.segments)
        cov = (used / available * 100) if available > 0 else 0
        logger.debug("  %s: %d segs, %.0fs (%.0f%% of available)",
                     tws.theme.title, len(segs), used, cov)

    return MasterCutPlan(
        selections=selections,
        total_duration=round(total, 1),
        theme_allocations={},
        coverage={},
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
        _ionice_cmd(cmd), capture_output=True, text=True, timeout=120,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"Segment extraction failed ({source_audio}, {start:.1f}-{end:.1f}): "
            f"{result.stderr[:300]}"
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
    target_duration: int = 12600,
    start_episode: int = 1,
    end_episode: int = 140,
    *,
    parallel_downloads: int = 4,
    prefer_yt_subs: bool = True,
    force_whisper: bool = False,
) -> dict:
    """Build a master cut across all episodes.

    This is the top-level entry point, orchestrating all 6 phases.

    Args:
        playlist_url: YouTube playlist URL.
        cfg: Pipeline configuration.
        state_file: Path to existing/desired universe state file. If empty,
                    uses output/universe_state.json.
        output_path: Output master cut audio path.
        target_duration: Target duration in seconds (default 3.5h = 12600).
        start_episode: First episode to include.
        end_episode: Last episode to include (0 = auto).
        parallel_downloads: Parallel download workers.
        prefer_yt_subs: Use YouTube subtitles when available.
        force_whisper: Skip YT subs, always use whisper.

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

    manifests = ensure_all_episode_artifacts(
        playlist_url=playlist_url,
        output_root=output_root,
        start_episode=start_episode,
        end_episode=end_episode,
        parallel=parallel_downloads,
        prefer_yt_subs=not force_whisper and prefer_yt_subs,
        audio_format=cfg.audio_format,
        audio_bitrate=cfg.audio_bitrate,
        whisper_model=cfg.whisper_model,
    )
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
    logger.info("PHASE 2: Build complete universe state")
    logger.info("=" * 60)
    t2 = time.time()

    # Load or create universe state
    if not state_file:
        state_file = os.path.join(output_root, "universe_state.json")
    state = UniverseState(state_file)
    existing_eps = set(state.data.get("metadata", {}).get("episodes_built_from", []))

    # Run Phase 2 for episodes that aren't in the state yet
    api_key = resolve_api_key()
    if not api_key:
        result["errors"].append("DeepSeek API key not set")
        return result
    ds_client = DeepSeekClient(api_key=api_key)

    new_global_states = 0
    skipped_existing = 0

    for m in manifests:
        if m.episode_number in existing_eps:
            skipped_existing += 1
            continue

        ep_dir = os.path.join(output_root, f"ep-{m.episode_number:03d}")
        gs_path = os.path.join(ep_dir, "global_state.json")

        if os.path.exists(gs_path):
            # Already has global state from a previous run
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
            new_global_states += 1

    # Force-save state (should already be saved by add_episode_knowledge)
    state.save()

    # Refresh state data from file (to get the latest merged state)
    state.load()

    result["phases"].append({
        "phase": "build_universe",
        "elapsed_sec": round(time.time() - t2, 1),
        "new_episodes": new_global_states,
        "existing_skipped": skipped_existing,
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

    # ── Phase 5: Select segments ────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("PHASE 5: Select segments (target=%.0fs = %.1fh)",
                target_duration, target_duration / 3600)
    logger.info("=" * 60)
    t5 = time.time()

    plan = select_segments_for_master_cut(
        themes_with_segments,
        target_duration=float(target_duration),
        min_segment=cfg.master_cut_min_segment,
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
    if result["errors"]:
        logger.info("  Errors: %d", len(result["errors"]))
    logger.info("=" * 60)

    return result
