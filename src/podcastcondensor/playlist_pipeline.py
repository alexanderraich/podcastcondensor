"""Playlist processing — builds universe state, runs full pipeline.

Two modes:
  build:   SRT → clean → Phase 2 (global state) → merge into state.
           No segmentation/audio/calls beyond the one-shot extraction.

  process: Full 6-phase pipeline per episode.  Phase 2 automatically
           extracts knowledge and merges into the universe state.
"""

import json
import logging
import os
import shutil
from pathlib import Path
from typing import List, Optional

from podcastcondensor.config import Config
from podcastcondensor.downloader import download_subtitles, resolve_episode_sources
from podcastcondensor.subtitles import load_subtitles
from podcastcondensor.universe_state import UniverseState
from podcastcondensor.pipeline import run_pipeline
from podcastcondensor.llm.deepseek import resolve_api_key, DeepSeekClient
from podcastcondensor.global_state import build_global_state

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Build universe state from episodes (Phase 2 only per episode)
# ---------------------------------------------------------------------------

def build_universe_state(
    playlist_url: str,
    cfg: Config,
    start_episode: int = 1,
    end_episode: int = 20,
    state_path: Optional[str] = None,
    dry_run: bool = False,
) -> UniverseState:
    """Build a UniverseState from a range of episodes.

    Per episode:
      1. Download SRT.
      2. Parse and clean programmatically.
      3. **Phase 2 call** — single DeepSeek: full transcript → outline +
         structured knowledge (entities, concepts, claims, etc.).
      4. Merge into universe state.

    Fully resumable: skips episodes whose ``global_state.json`` already exists.
    """
    # --- Initialise state ---------------------------------------------------
    if state_path:
        state = UniverseState(state_path)
        state.data["metadata"]["source_playlist"] = playlist_url
        state.save()
    else:
        sp = os.path.join(cfg.output_root, "universe_state.json")
        state = UniverseState(sp)
        state.data["metadata"]["source_playlist"] = playlist_url
        state.save()

    # --- Resolve episode sources --------------------------------------------
    sources = resolve_episode_sources(
        playlist_url=playlist_url,
        start_ep=start_episode,
        end_ep=end_episode,
    )
    logger.info("Resolved %d episode sources", len(sources))

    download_dir = "/tmp/podcastcondensor/downloads"

    for src in sources:
        episode_num = src["episode_number"]
        video_url = src["video_url"]
        title = src.get("title", f"Episode {episode_num}")

        logger.info("=" * 60)
        logger.info("Episode %d: %s", episode_num, title)
        logger.info("=" * 60)

        ep_dir = os.path.join(cfg.output_root, f"ep-{episode_num:03d}")
        Path(ep_dir).mkdir(parents=True, exist_ok=True)
        gs_path = os.path.join(ep_dir, "global_state.json")

        # Checkpoint: skip if global_state.json already exists
        if os.path.exists(gs_path):
            logger.info("Checkpoint HIT — global_state.json exists for episode %d", episode_num)
            with open(gs_path) as f:
                global_data = json.load(f)
        else:
            # Download SRT (cached by yt-dlp)
            subtitle_path = download_subtitles(
                url=video_url,
                output_dir=download_dir,
                video_id=src["id"],
                lang=cfg.lang,
                prefer_auto=cfg.prefer_auto_subs,
            )

            if not subtitle_path:
                logger.warning("No subtitles for episode %d, skipping", episode_num)
                continue

            # Copy raw SRT into ep dir
            target_srt = os.path.join(ep_dir, "source_subtitles.srt")
            if subtitle_path != target_srt:
                shutil.copy2(subtitle_path, target_srt)

            # Clean + build transcript
            cleaned = load_subtitles(subtitle_path)
            logger.info("Cleaned %d subtitle entries", len(cleaned))

            if dry_run:
                logger.info("Dry-run: skipping extraction")
                continue

            from podcastcondensor.segmentation.sentence_units import (
                build_transcript_from_entries,
            )
            transcript_text = build_transcript_from_entries(cleaned)

            # Phase 2 single-shot call
            api_key = resolve_api_key()
            if not api_key:
                logger.error("DeepSeek API key not set — skipping")
                continue

            ds_client = DeepSeekClient(api_key=api_key)
            global_data = build_global_state(
                transcript_text=transcript_text,
                episode_title=title,
                episode_number=episode_num,
                client=ds_client,
                model=cfg.deepseek_model,
                prompt_path=cfg.global_state_prompt_path,
                timeout=cfg.deepseek_timeout,
            )

            # Write checkpoint
            with open(gs_path, "w", encoding="utf-8") as f:
                json.dump(global_data, f, ensure_ascii=False, indent=2)
            logger.info("Wrote %s", gs_path)

            # Merge knowledge into state
            knowledge = {
                "summary": global_data.get("summary", ""),
                "entities": global_data.get("entities", []),
                "concepts": global_data.get("concepts", []),
                "claims": global_data.get("claims", []),
                "scriptural_links": global_data.get("scriptural_links", []),
                "glossary": global_data.get("glossary", []),
            }
            state.add_episode_knowledge(episode_num, knowledge)
            kc = len(knowledge.get("concepts", []))
            ke = len(knowledge.get("entities", []))
            logger.info("  → Added %d concepts, %d entities", kc, ke)

    meta = state.data["metadata"]
    logger.info("=" * 60)
    logger.info("BUILD COMPLETE")
    logger.info("  Episodes processed: %d", meta.get("last_built_episode", 0))
    logger.info("  Total concepts:     %d", len(state.data.get("concepts", [])))
    logger.info("  Total entities:     %d", len(state.data.get("entities", [])))
    logger.info("  Total claims:       %d", len(state.data.get("claims", [])))
    logger.info("  Glossary terms:     %d", len(state.data.get("glossary", [])))
    logger.info("=" * 60)

    return state


# ---------------------------------------------------------------------------
# Process episodes with an existing universe state (full pipeline)
# ---------------------------------------------------------------------------

def process_with_universe_state(
    playlist_url: str,
    cfg: Config,
    state: UniverseState,
    start_episode: int = 21,
    end_episode: Optional[int] = None,
    dry_run: bool = False,
    debug_max_intervals: int = 0,
) -> List[dict]:
    """Process episodes using an existing UniverseState.

    Runs the full 6-phase pipeline per episode.  Phase 2 inside the
    pipeline handles knowledge extraction and state update automatically.
    """
    effective_end = (
        end_episode if end_episode and end_episode >= start_episode
        else start_episode
    )
    sources = resolve_episode_sources(
        playlist_url=playlist_url,
        start_ep=start_episode,
        end_ep=effective_end,
    )
    logger.info(
        "Processing episodes with universe state (%d episodes in state)",
        state.data.get("metadata", {}).get("last_built_episode", 0),
    )

    results = []

    for src in sources:
        episode_num = src["episode_number"]
        video_url = src["video_url"]
        title = src.get("title", f"Episode {episode_num}")

        logger.info("=" * 60)
        logger.info("Processing Episode %d: %s", episode_num, title)
        logger.info("URL: %s", video_url)
        logger.info("=" * 60)

        try:
            result = run_pipeline(
                url=video_url,
                cfg=cfg,
                dry_run=dry_run,
                universe_state=state,
                episode_num=episode_num,
                debug_max_intervals=debug_max_intervals,
            )

            errors = result.get("errors", [])
            success = len(errors) == 0

            results.append({
                "episode": episode_num,
                "title": title,
                "url": video_url,
                "success": success,
                "errors": errors,
                "output_dir": result.get("output_dir"),
                "condensed_audio": (
                    result.get("phases", {})
                    .get("audio", {})
                    .get("condensed_path")
                ),
                "global_state": (
                    result.get("phases", {})
                    .get("global_state", {})
                ),
            })

            # Log stats
            stats_path = os.path.join(
                result.get("output_dir", ""), "stats.json",
            )
            if os.path.exists(stats_path):
                try:
                    with open(stats_path) as f:
                        stats_data = json.load(f)
                    logger.info(
                        "Episode %d: %d/%d kept (%.1f%% compression, "
                        "%.1fmin → %.1fmin)",
                        episode_num,
                        stats_data.get("keep_count", 0),
                        stats_data.get("total_segments", 0),
                        (1 - stats_data.get("compression_ratio", 1)) * 100,
                        stats_data.get("original_duration_sec", 0) / 60,
                        stats_data.get("condensed_duration_sec", 0) / 60,
                    )
                except (json.JSONDecodeError, OSError):
                    pass

        except Exception as e:
            logger.exception(
                "Failed to process episode %d: %s", episode_num, e,
            )
            results.append({
                "episode": episode_num,
                "title": title,
                "url": video_url,
                "success": False,
                "error": str(e),
            })

    successful = sum(1 for r in results if r.get("success"))
    logger.info("=" * 60)
    logger.info("PROCESSING COMPLETE")
    logger.info("  Episodes processed: %d/%d", successful, len(results))
    logger.info("  Universe state now has %d concepts",
                len(state.data.get("concepts", [])))
    logger.info("=" * 60)

    return results
