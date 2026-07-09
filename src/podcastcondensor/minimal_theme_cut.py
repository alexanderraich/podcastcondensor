"""Minimal theme cut — one theme, no arbitrary length enforcements.

The LLM decides what the minimum viable set of segments is for
understanding a single theme. No target_duration, no min_segments,
no budget — just the content that matters.

Phases:
  1. Resolve candidate segments for one theme from universe state
  2. Load transcript text with context for each segment
  3. One DeepSeek call: select minimal set + refine thought boundaries
  4. Assemble audio with beep separators
"""

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
# Phase 1: Load theme from cache or re-extract
# ---------------------------------------------------------------------------


def load_theme_by_id(
    theme_id: str,
    themes_file: str = "",
    universe_data: Optional[dict] = None,
    client: Optional[DeepSeekClient] = None,
    model: str = "deepseek-chat",
    timeout: int = 600,
) -> Optional[Theme]:
    """Load a single theme by its ID.

    Checks the cached themes file first; if not found and universe_data +
    client are provided, re-runs theme extraction and searches the result.
    """
    # Try themes cache first
    if themes_file and os.path.exists(themes_file):
        with open(themes_file, "r", encoding="utf-8") as f:
            themes_data = json.load(f)
        for td in themes_data:
            if td.get("id") == theme_id:
                logger.info("Loaded theme '%s' from cache (%s)", theme_id, themes_file)
                # Handle both cache formats (save-time key vs load-time key)
                related = td.get("related_item_ids") or td.get("items") or []
                intro = td.get("natural_intro_items") or td.get("intro_items") or []
                desc = td.get("description") or td.get("desc") or ""
                return Theme(
                    id=td["id"],
                    title=td.get("title", ""),
                    description=desc,
                    importance=td.get("importance", 0.5),
                    related_item_ids=related,
                    natural_intro_items=intro,
                )
        logger.warning("Theme '%s' not found in cache file %s", theme_id, themes_file)
        if universe_data and client:
            logger.info("Re-running theme extraction to find '%s'...", theme_id)
    elif universe_data and client:
        logger.info("No themes cache — re-running extraction...")
    else:
        logger.error("Cannot find theme '%s' — no cache and no extraction available", theme_id)
        return None

    # Re-run extraction (expensive but recovers from missing cache)
    if universe_data and client:
        from podcastcondensor.theme_extraction import extract_themes
        all_themes = extract_themes(
            universe_data=universe_data,
            client=client,
            model=model,
            timeout=timeout,
        )
        for t in all_themes:
            if t.id == theme_id:
                logger.info("Found theme '%s' in fresh extraction", theme_id)
                return t
        logger.error("Theme '%s' not found in freshly extracted themes either", theme_id)
        return None

    return None


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


def _format_segment_with_context(
    seg: ThemeSegment,
    entries: List[dict],
    seg_index: int,
    episode_title: str,
    context_buffer: float = 30.0,
) -> Tuple[str, bool]:
    """Format one segment's transcript text with context for the prompt.

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

    start_win = max(0, seg.start - context_buffer)
    end_win = seg.end + context_buffer

    text_lines = []
    in_segment = False
    for e in entries:
        if e["start"] >= end_win:
            break
        if e["end"] <= start_win:
            continue
        # Is this entry within the candidate window?
        is_candidate = e["start"] >= seg.start and e["end"] <= seg.end
        marker = "  >>>  " if is_candidate else "       "
        if is_candidate and not in_segment:
            text_lines.append("       ── candidate window ──")
            in_segment = True
        text_lines.append(f"{marker}{e['start']:7.1f}-{e['end']:7.1f}: {e['text']}")

    return (
        f"--- Segment {seg_index + 1} (seg_id: seg_{seg_index}) ---\n"
        f"Episode: {seg.episode_number} — {episode_title}\n"
        f"Candidate: {seg.start:.1f}s - {seg.end:.1f}s ({seg.duration:.0f}s)\n"
        f"\nTranscript (>>> = candidate window, no marker = context):\n"
        + "\n".join(text_lines),
        True,
    )


def build_selection_prompt(
    theme: Theme,
    tws: ThemeWithSegments,
    output_root: str,
    manifests: List[EpisodeManifest],
    context_buffer: float = 30.0,
) -> str:
    """Build the prompt asking the LLM to select minimal viable segments.

    Each candidate segment is shown with ~30s of surrounding transcript
    context so the LLM can determine complete-thought boundaries.
    """
    ep_titles = {m.episode_number: m.title for m in manifests}
    entries_cache: Dict[int, List[dict]] = {}

    # Pre-load all needed SRT files
    for seg in tws.segments:
        _load_episode_entries(output_root, seg.episode_number, entries_cache)

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
        "  A thorough treatment of this theme probably needs 4-8 segments",
        "  totalling 8-20 minutes. That gives room for: a definition segment,",
        "  a development/argument segment, a couple examples, and a",
        "  connection-to-broader-theology segment. Fewer than 4 segments",
        "  is unlikely to be self-contained; more than 10 is probably",
        "  too repetitive.",
        "",
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
        formatted, _ = _format_segment_with_context(
            seg, entries, i, title, context_buffer,
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
) -> List[RefinedSelection]:
    """Apply LLM decisions to build the final selection list.

    Merges overlapping/adjacent kept segments from the same episode.
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

    return merged


# ---------------------------------------------------------------------------
# Phase 5: Audio assembly (same beep pattern as master cut)
# ---------------------------------------------------------------------------


def assemble_minimal_cut(
    selections: List[RefinedSelection],
    output_path: str,
    *,
    sample_rate: int = 22050,
    bitrate: str = "64k",
    speed: float = 1.25,
    parallel_workers: int = 4,
    keep_temp: bool = False,
) -> str:
    """Assemble the minimal cut audio from refined selections.

    Single beeps between segments from different episodes for clean
    separation. No triple beeps (single theme = no theme transitions).
    """
    if not selections:
        raise ValueError("No selections to assemble")

    t0 = time.time()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    tmpdir = tempfile.mkdtemp(prefix="minimal_cut_")
    atempo = _atempo_filters(speed)

    try:
        # Generate single beep file
        logger.info("Generating beep file...")
        beep_file = _generate_beep_file(
            tmpdir, "beep_single.mp3",
            freq=1000, tone_duration=0.25, count=1,
            sample_rate=sample_rate, bitrate=bitrate,
        )

        # Extract all segments
        seg_paths: List[str] = []
        total = len(selections)
        logger.info("Extracting %d segments...", total)

        for i in range(total):
            seg_paths.append(os.path.join(tmpdir, f"seg_{i:04d}.mp3"))

        completed = 0

        def _extract_one(idx: int) -> Tuple[int, Optional[str]]:
            sel = selections[idx]
            try:
                _extract_segment(
                    sel.audio_path,
                    sel.start,
                    sel.end,
                    seg_paths[idx],
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
                    raise RuntimeError(f"Segment {idx} extraction: {error}")
                completed += 1
                if completed % 10 == 0 or completed == total:
                    logger.info("  Extracted %d/%d segments", completed, total)

        # Build interleaved concat list with beeps
        interleaved: List[str] = []
        for i in range(total):
            if i > 0:
                interleaved.append(beep_file)
            interleaved.append(seg_paths[i])

        logger.info(
            "Concat: %d segments + %d beeps",
            total, total - 1,
        )

        # Final concat pass
        logger.info("Running final concat pass...")
        _concat_batch_files(
            batch_paths=interleaved,
            output_path=output_path,
            sample_rate=sample_rate,
            bitrate=bitrate,
            atempo=atempo,
            beep=False,
        )

        elapsed = time.time() - t0
        total_duration = sum(s.duration for s in selections) / max(speed, 1)
        logger.info(
            "Minimal cut assembled: %.1fs real → %.0fs output (%s)",
            elapsed, total_duration, output_path,
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


def _scan_existing_episodes(output_root: str, start: int, end: int) -> List[EpisodeManifest]:
    """Scan output/ for existing episode artefacts without YouTube calls.

    Builds manifests from any ep-NNN directory that has both an MP3 and
    source_subtitles.srt. Fast — no network.
    """
    manifests = []
    for ep_num in range(start, end + 1):
        ep_dir = os.path.join(output_root, f"ep-{ep_num:03d}")
        if not os.path.isdir(ep_dir):
            continue
        srt_path = os.path.join(ep_dir, "source_subtitles.srt")
        if not os.path.exists(srt_path):
            continue
        # Find audio file (any mp3 except temp/checkpoint files)
        audio = None
        for f in os.listdir(ep_dir):
            if f.endswith(".mp3") and not f.startswith("_"):
                audio = os.path.join(ep_dir, f)
                break
        if not audio:
            continue
        manifests.append(EpisodeManifest(
            episode_number=ep_num,
            video_id="",
            audio_path=audio,
            srt_path=srt_path,
            title=f"Episode {ep_num}",
        ))
    logger.info("Scanned %d existing episodes from %s", len(manifests), output_root)
    return manifests


def build_minimal_theme_cut(
    theme_id: str,
    playlist_url: str = "",
    cfg: Optional[Config] = None,
    state_file: str = "",
    themes_file: str = "",
    output_path: str = "output/minimal_theme_cut.mp3",
    start_episode: int = 1,
    end_episode: int = 140,
    *,
    context_buffer: float = 30.0,
) -> dict:
    """Build a minimal audio cut for a single theme.

    The LLM decides what the minimum viable set of segments is —
    no target_duration, no segment count minimums, no budget.

    Args:
        theme_id: Kebab-case theme ID (e.g. "theosis-and-deification").
        playlist_url: YouTube playlist URL.
        cfg: Pipeline configuration.
        state_file: Path to universe_state.json.
        themes_file: Path to cached themes JSON (from a prior run).
        output_path: Output audio path.
        start_episode: First episode to consider.
        end_episode: Last episode to consider.
        parallel_downloads: Workers for audio extraction.
        prefer_yt_subs: Prefer YouTube subtitles.
        force_whisper: Force whisper transcription.
        context_buffer: Seconds of transcript context around each segment.

    Returns:
        Dict with keys: phases (list), output_path, errors, plan.
    """
    if cfg is None:
        cfg = Config()

    overall_t0 = time.time()
    result = {"phases": [], "errors": [], "output_path": None}

    output_root = cfg.output_root or os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "output",
    )
    Path(output_root).mkdir(parents=True, exist_ok=True)

    # ── Phase 1: Scan existing artefacts (no YouTube calls) ──────────────
    logger.info("=" * 60)
    logger.info("PHASE 1: Scan existing episode artefacts")
    logger.info("=" * 60)
    t1 = time.time()

    manifests = _scan_existing_episodes(
        output_root, start_episode, end_episode,
    )
    result["phases"].append({
        "phase": "scan_existing",
        "elapsed_sec": round(time.time() - t1, 1),
        "episodes": len(manifests),
    })

    if not manifests:
        result["errors"].append("No episodes found in %s — cannot continue" % output_root)
        return result

    # ── Phase 2: Load universe state and theme ───────────────────────────
    logger.info("=" * 60)
    logger.info("PHASE 2: Load universe state and theme '%s'", theme_id)
    logger.info("=" * 60)
    t2 = time.time()

    if not state_file:
        state_file = os.path.join(output_root, "universe_state.json")

    if not os.path.exists(state_file):
        result["errors"].append(f"Universe state not found: {state_file}")
        return result

    with open(state_file, "r", encoding="utf-8") as f:
        universe_data = json.load(f)

    api_key = None
    client = None
    theme = load_theme_by_id(theme_id, themes_file)

    if not theme:
        # Try re-extracting themes (needs API)
        api_key = None
        try:
            from podcastcondensor.llm.deepseek import resolve_api_key
            api_key = resolve_api_key()
        except Exception:
            pass
        if not api_key:
            result["errors"].append(
                f"Theme '{theme_id}' not found in cache and no API key "
                "available for re-extraction"
            )
            return result
        client = DeepSeekClient(api_key=api_key)
        theme = load_theme_by_id(
            theme_id, themes_file,
            universe_data=universe_data,
            client=client,
            model=cfg.deepseek_model,
            timeout=cfg.deepseek_timeout,
        )

    if not theme:
        result["errors"].append(f"Theme '{theme_id}' not found")
        return result

    logger.info("Theme: %s (importance=%.2f, %d items)",
                theme.title, theme.importance, len(theme.related_item_ids))

    result["phases"].append({
        "phase": "load_theme",
        "elapsed_sec": round(time.time() - t2, 1),
        "theme_id": theme.id,
        "theme_title": theme.title,
        "related_items": len(theme.related_item_ids),
    })

    # ── Phase 3: Resolve segments from universe state ────────────────────
    logger.info("=" * 60)
    logger.info("PHASE 3: Resolve candidate segments from universe state")
    logger.info("=" * 60)
    t3 = time.time()

    themes_with_segments = resolve_theme_segments_from_state(
        themes=[theme],
        universe_data=universe_data,
        manifests=manifests,
        output_root=output_root,
    )

    if not themes_with_segments or not themes_with_segments[0].segments:
        result["errors"].append(f"No segments found for theme '{theme_id}'")
        return result

    tws = themes_with_segments[0]
    total_available = sum(s.duration for s in tws.segments)
    logger.info("Found %d candidate segments, %.0fs total available",
                len(tws.segments), total_available)

    result["phases"].append({
        "phase": "resolve_segments",
        "elapsed_sec": round(time.time() - t3, 1),
        "candidate_count": len(tws.segments),
        "total_available_sec": round(total_available, 1),
    })

    # ── Phase 4: LLM selects minimal set ─────────────────────────────────
    logger.info("=" * 60)
    logger.info("PHASE 4: LLM selects minimum viable segments")
    logger.info("=" * 60)
    t4 = time.time()

    # Ensure we have an API client
    if not client:
        from podcastcondensor.llm.deepseek import resolve_api_key
        api_key = resolve_api_key()
        if not api_key:
            result["errors"].append("DeepSeek API key not set")
            return result
        client = DeepSeekClient(api_key=api_key)

    prompt = build_selection_prompt(
        theme=theme,
        tws=tws,
        output_root=output_root,
        manifests=manifests,
        context_buffer=context_buffer,
    )

    logger.info("Selection prompt: %d chars (~%d tokens)",
                len(prompt), len(prompt) // 4)

    try:
        raw = client.generate(
            prompt=prompt,
            model=cfg.deepseek_model,
            timeout=cfg.deepseek_timeout,
            temperature=0.3,
            max_tokens=8192,
            force_json=True,
        )
    except Exception as e:
        result["errors"].append(f"LLM selection call failed: {e}")
        logger.exception("LLM selection call failed")
        return result

    decisions = parse_selection_response(raw)

    if not decisions:
        result["errors"].append("No valid decisions from LLM response")
        return result

    # Apply decisions: refine boundaries + merge adjacent
    selections = apply_decisions(decisions, tws, manifests)

    if not selections:
        result["errors"].append("LLM kept 0 segments — nothing to cut")
        return result

    total_selected = sum(s.duration for s in selections)
    logger.info("Selected %d segments, %.0fs total (%.1f min)",
                len(selections), total_selected, total_selected / 60)

    result["phases"].append({
        "phase": "llm_selection",
        "elapsed_sec": round(time.time() - t4, 1),
        "candidates": len(tws.segments),
        "selected": len(selections),
        "total_duration_sec": round(total_selected, 1),
    })

    # ── Phase 5: Assemble audio ──────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("PHASE 5: Assemble minimal cut audio")
    logger.info("=" * 60)

    if not os.path.isabs(output_path):
        output_path = os.path.join(output_root, output_path)

    t5 = time.time()
    try:
        assemble_minimal_cut(
            selections=selections,
            output_path=output_path,
            sample_rate=cfg.audio_sample_rate,
            bitrate=cfg.audio_bitrate,
            speed=cfg.audio_speed,
            parallel_workers=4,
            keep_temp=cfg.keep_temp,
        )
        result["output_path"] = output_path
    except Exception as e:
        logger.exception("Audio assembly failed: %s", e)
        result["errors"].append(f"Audio assembly failed: {e}")

    result["phases"].append({
        "phase": "assemble_audio",
        "elapsed_sec": round(time.time() - t5, 1),
        "output_path": output_path,
    })

    # Summary
    total_elapsed = time.time() - overall_t0
    logger.info("=" * 60)
    logger.info("MINIMAL THEME CUT COMPLETE — %.1fs (%.1fmin)",
                total_elapsed, total_elapsed / 60)
    logger.info("  Theme: %s", theme.title)
    logger.info("  Output: %s", output_path)
    logger.info("  Segments: %d (of %d candidates)", len(selections), len(tws.segments))
    logger.info("  Duration: %.0fs (%.1fmin)", total_selected, total_selected / 60)
    if result["errors"]:
        logger.info("  Errors: %d", len(result["errors"]))
    logger.info("=" * 60)

    return result
