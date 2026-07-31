"""Playlist processing — builds universe state, runs full pipeline.

Two modes:
  build:   SRT → clean → Phase 2 (global state) → merge into state.
           No segmentation/audio/calls beyond the one-shot extraction.

  process: Full 6-phase pipeline per episode.  Phase 2 automatically
           extracts knowledge and merges into the universe state.

Two additional entry points for the master cut pipeline:
  ``build_master_cut()`` — orchestrates all phases for the master cut.
  (The actual implementation lives in ``master_cut.py`` — this module
   just re-exports it so the CLI can import from the same place.)
"""

import json
import logging
import os
import traceback
from datetime import datetime
from pathlib import Path
from typing import List, Optional

from podcastcondensor.config import Config
from podcastcondensor.downloader import download_audio, resolve_episode_sources
from podcastcondensor.transcribe import transcribe_audio
from podcastcondensor.subtitles import load_subtitles
from podcastcondensor.universe_state import UniverseState
from podcastcondensor.pipeline import run_pipeline
from podcastcondensor.llm.deepseek import resolve_api_key, DeepSeekClient
from podcastcondensor.global_state import build_global_state

logger = logging.getLogger(__name__)


def _write_crash_log(ep_dir: str, context: str, exc: Exception):
    """Crash-safe exception log (fsynced)."""
    path = os.path.join(ep_dir, "_crash.log")
    try:
        Path(ep_dir).mkdir(parents=True, exist_ok=True)
        with open(path, "a") as f:
            f.write(f"=== CRASH at {datetime.now().isoformat()} context=[{context}] ===\n")
            traceback.print_exception(type(exc), exc, exc.__traceback__, file=f)
            f.write("\n")
            f.flush()
            os.fsync(f.fileno())
    except Exception:
        pass


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
    skip_qa: bool = True,
    skip_reading: bool = True,
) -> UniverseState:
    """Build a UniverseState from a range of episodes.

    Always uses whisper transcription — YouTube subtitles are unreliable.

    Per episode:
      1. Download audio, transcribe with whisper.
      2. Parse and clean programmatically.
      3. **Phase 2 call** — single DeepSeek: full transcript → outline +
         structured knowledge (entities, concepts, claims, etc.).
      4. Merge into universe state.

    Fully resumable: skips episodes whose ``global_state.json`` already exists.

    When *skip_qa* is True (default), episodes with "q&a" in their YouTube
    title are skipped — they don't develop coherent cross-episode themes.
    When *skip_reading* is True (default), episodes with "read by" in their
    YouTube title are skipped — they are scriptural readings, not discussion.
    """
    if state_path:
        state = UniverseState(state_path)
        state.data["metadata"]["source_playlist"] = playlist_url
        state.save()
    else:
        sp = os.path.join(cfg.output_root, "universe_state.json")
        state = UniverseState(sp)
        state.data["metadata"]["source_playlist"] = playlist_url
        state.save()

    sources = resolve_episode_sources(
        playlist_url=playlist_url,
        start_ep=start_episode,
        end_ep=end_episode,
    )
    logger.info("Resolved %d episode sources", len(sources))

    # ── Filter out Q&A episodes ──────────────────────────────────────────
    if skip_qa:
        qa_skipped = [s for s in sources if "q&a" in s.get("title", "").lower()]
        if qa_skipped:
            logger.info(
                "Skipping %d Q&A episode(s): %s",
                len(qa_skipped),
                [(s["episode_number"], s.get("title", "")[:60]) for s in qa_skipped],
            )
            sources = [s for s in sources if "q&a" not in s.get("title", "").lower()]

    # ── Filter out Reading episodes ──────────────────────────────────────
    if skip_reading:
        reading_skipped = [s for s in sources if "read by" in s.get("title", "").lower()]
        if reading_skipped:
            logger.info(
                "Skipping %d Reading episode(s): %s",
                len(reading_skipped),
                [(s["episode_number"], s.get("title", "")[:60]) for s in reading_skipped],
            )
            sources = [s for s in sources if "read by" not in s.get("title", "").lower()]

    # Pipelined processing: downloads run AHEAD in a small thread pool (network
    # I/O, low memory), while whisper runs strictly ONE-AT-A-TIME in the main
    # thread (GPU/RAM — this machine has 8GB RAM / 6GB GPU; concurrent whisper
    # instances risk OOM). The overlap hides download time behind whisper time
    # (~30-40% faster). Global_state extraction stays sequential after each
    # episode's whisper. Episodes already on disk fast-skip both steps.
    def _download_audio_only(episode_num: int, src: dict) -> Optional[str]:
        """Download one episode's audio (I/O — safe to parallelize)."""
        ep_dir = os.path.join(cfg.output_root, f"ep-{episode_num:03d}")
        Path(ep_dir).mkdir(parents=True, exist_ok=True)
        target_srt = os.path.join(ep_dir, "source_subtitles.srt")
        if os.path.exists(target_srt):
            return None  # already transcribed — no download needed
        return download_audio(
            url=src["video_url"],
            output_dir=ep_dir,
            video_id=src["id"],
            audio_format=cfg.audio_format,
            audio_bitrate=cfg.audio_bitrate,
        )

    def _transcribe_serial(episode_num: int, src: dict, audio_path: Optional[str]) -> Optional[str]:
        """Whisper one episode (serial — GPU). Returns SRT path or None."""
        ep_dir = os.path.join(cfg.output_root, f"ep-{episode_num:03d}")
        target_srt = os.path.join(ep_dir, "source_subtitles.srt")
        if os.path.exists(target_srt):
            return target_srt
        if not audio_path:
            logger.warning("No audio for episode %d, skipping", episode_num)
            return None
        transcribe_audio(
            audio_path, ep_dir,
            model_size=cfg.whisper_model,
            beam_size=cfg.whisper_beam_size,
            vad_filter=cfg.whisper_vad_filter,
            condition_on_previous_text=cfg.whisper_condition_on_prev,
        )
        return target_srt if os.path.exists(target_srt) else None

    def _process_episode(episode_num: int, src: dict, srt_path: Optional[str], state: UniverseState) -> None:
        """Extract + merge global state for one episode (runs after its prep)."""
        title = src.get("title", f"Episode {episode_num}")
        ep_dir = os.path.join(cfg.output_root, f"ep-{episode_num:03d}")
        gs_path = os.path.join(ep_dir, "global_state.json")

        logger.info("=" * 60)
        logger.info("Episode %d: %s", episode_num, title)
        logger.info("=" * 60)

        if os.path.exists(gs_path):
            logger.info("Checkpoint HIT — global_state.json exists for episode %d", episode_num)
            with open(gs_path) as f:
                global_data = json.load(f)
        elif not srt_path:
            logger.warning("No SRT for episode %d — skipping", episode_num)
            return
        elif dry_run:
            logger.info("Dry-run: skipping extraction")
            return
        else:
            cleaned = load_subtitles(srt_path)
            logger.info("Cleaned %d subtitle entries", len(cleaned))
            from podcastcondensor.segmentation.sentence_units import build_transcript_from_entries
            transcript_text = build_transcript_from_entries(cleaned)

            api_key = resolve_api_key()
            if not api_key:
                logger.error("DeepSeek API key not set — skipping")
                return
            ds_client = DeepSeekClient(api_key=api_key)
            global_data = build_global_state(
                transcript_text=transcript_text,
                episode_title=title,
                episode_number=episode_num,
                client=ds_client,
                model=cfg.deepseek_model,
                prompt_path=cfg.global_state_prompt_path,
                timeout=cfg.deepseek_timeout,
                srt_entries=cleaned,
            )
            with open(gs_path, "w", encoding="utf-8") as f:
                json.dump(global_data, f, ensure_ascii=False, indent=2)
            logger.info("Wrote %s", gs_path)

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

    from concurrent.futures import ThreadPoolExecutor

    PREP_AHEAD = 3  # downloads that run ahead of whisper
    sources_list = list(sources)
    n = len(sources_list)

    with ThreadPoolExecutor(max_workers=PREP_AHEAD) as dl_pool:
        # Prime the download window with the first PREP_AHEAD episodes.
        dl_futures = {}
        for i in range(min(PREP_AHEAD, n)):
            s = sources_list[i]
            dl_futures[s["episode_number"]] = (
                dl_pool.submit(_download_audio_only, s["episode_number"], s), s
            )

        for i in range(n):
            s = sources_list[i]
            future, cur_src = dl_futures.pop(s["episode_number"])
            audio_path = future.result()

            # Keep the window full: submit download i + PREP_AHEAD.
            j = i + PREP_AHEAD
            if j < n:
                ns = sources_list[j]
                dl_futures[ns["episode_number"]] = (
                    dl_pool.submit(_download_audio_only, ns["episode_number"], ns), ns
                )

            # Whisper (serial) + extract global state.
            srt_path = _transcribe_serial(s["episode_number"], cur_src, audio_path)
            _process_episode(s["episode_number"], cur_src, srt_path, state)

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

def _episode_is_done(episode_dir: str, cfg: Config) -> bool:
    """Check if episode has all output artefacts and can be skipped."""
    if not os.path.isdir(episode_dir):
        return False
    if not os.path.exists(os.path.join(episode_dir, "source_subtitles.srt")):
        return False
    if cfg.skip_global_state:
        return os.path.exists(os.path.join(episode_dir, "compressed.json"))
    else:
        return os.path.exists(os.path.join(episode_dir, "decisions.json"))


def process_with_universe_state(
    playlist_url: str,
    cfg: Config,
    state: Optional[UniverseState] = None,
    start_episode: int = 21,
    end_episode: Optional[int] = None,
    dry_run: bool = False,
    debug_max_intervals: int = 0,
) -> List[dict]:
    """Process episodes using an existing UniverseState.

    When ``state`` is None and ``cfg.skip_global_state`` is True, runs the
    new one-shot compression pipeline (no universe state needed).

    Runs the full pipeline per episode. Phase 2 inside the pipeline handles
    the actual compression or knowledge extraction depending on mode.
    """
    effective_end = (
        end_episode if end_episode and end_episode >= start_episode
        else start_episode
    )

    # ── Quick skip: check output dirs before touching YouTube ────────────
    wanted = list(range(start_episode, effective_end + 1))
    skipped = []
    needed = []
    for ep in wanted:
        ep_dir = os.path.join(cfg.output_root, f"ep-{ep:03d}")
        if _episode_is_done(ep_dir, cfg):
            skipped.append(ep)
        else:
            needed.append(ep)

    if skipped:
        logger.info("Skipping %d already-done episode(s): %s", len(skipped), skipped)

    if not needed:
        logger.info("All episodes already processed — nothing to do.")
        return []

    # Only resolve YouTube for episodes that need processing
    sources = resolve_episode_sources(
        playlist_url=playlist_url,
        start_ep=min(needed),
        end_ep=max(needed),
    )
    if state is not None:
        logger.info(
            "Processing episodes with universe state (%d episodes in state)",
            state.data.get("metadata", {}).get("last_built_episode", 0),
        )
    else:
        logger.info("Processing episodes (compress mode, no universe state)")

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
            ep_dir = os.path.join(cfg.output_root, f"ep-{episode_num:03d}")
            _write_crash_log(ep_dir, f"process_episode_{episode_num}", e)
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
    if state is not None:
        logger.info("  Universe state now has %d concepts",
                    len(state.data.get("concepts", [])))
    logger.info("=" * 60)

    return results


# ---------------------------------------------------------------------------
# Master cut pipeline (re-exported from master_cut module)
# ---------------------------------------------------------------------------

from podcastcondensor.master_cut import build_master_cut  # noqa: E402,F811
