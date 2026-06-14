"""Playlist processing pipeline — batch orchestration with Universe State.

Two modes:
  build:  Process episodes to BUILD the universe state (Phase A + knowledge extraction only).
          No classification or audio cutting — just extract structured knowledge.

  process: Full pipeline using an existing universe state for aggressive classification.
           Downloads, segments, classifies with state context, cuts audio, then
           extracts new knowledge and updates the state.
"""

import json
import logging
import os
import shutil
import sys
from pathlib import Path
from datetime import datetime
from typing import List, Optional

from podcastcondensor.config import Config
from podcastcondensor.downloader import download_subtitles, resolve_episode_sources
from podcastcondensor.ollama_client import generate, check_ollama, find_best_model
from podcastcondensor.subtitles import load_subtitles
from podcastcondensor.rechunker import resegment, refine_segments
from podcastcondensor.global_map import build_global_map
from podcastcondensor.universe_state import UniverseState
from podcastcondensor.pipeline import run_pipeline

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Build universe state from episodes (Phase A + knowledge extraction only)
# ---------------------------------------------------------------------------

def build_universe_state(
    playlist_url: str,
    cfg: Config,
    start_episode: int = 1,
    end_episode: int = 20,
    state_path: Optional[str] = None,
    dry_run: bool = False,
) -> UniverseState:
    """Build a UniverseState from a range of playlist episodes.

    For each episode:
      1. Download subtitles (no audio needed)
      2. Parse, clean, resegment
      3. Phase A: Build global map (block summaries + outline)
      4. Extract structured knowledge from outline + summaries
      5. Merge into universe state

    Args:
        playlist_url: YouTube playlist URL
        cfg: Pipeline config
        start_episode: 1-based index of first episode (default 1)
        end_episode: 1-based index of last episode (default 20)
        state_path: Path to save/load universe state JSON
        dry_run: If True, skip LLM calls and just prepare data

    Returns:
        UniverseState populated with knowledge from the episode range.
    """
    # Initialize state
    if state_path:
        state = UniverseState(state_path)
        state.data["metadata"]["source_playlist"] = playlist_url
        state.data["metadata"]["updated_at"] = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
        state.save()
    else:
        state = UniverseState(
            os.path.join(cfg.output_root, "universe_state.json")
        )
        state.data["metadata"]["source_playlist"] = playlist_url
        state.save()

    # Check Ollama
    if not dry_run:
        if not check_ollama(cfg.ollama_host):
            logger.error("Ollama is not running. Start: ollama serve")
            sys.exit(1)

        model = find_best_model(cfg.default_model, cfg.fallback_model, cfg.ollama_host)
        if model is None:
            logger.error("No model found. Pull: ollama pull %s", cfg.default_model)
            sys.exit(1)
    else:
        model = cfg.default_model

    # Resolve episode sources (playlist + direct fallback)
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
        source_type = src.get("source_type", "unknown")

        logger.info("=" * 60)
        logger.info("Episode %d: %s [%s]", episode_num, title, source_type)
        logger.info("URL: %s", video_url)
        logger.info("=" * 60)

        run_dir = os.path.join(cfg.output_root, src["id"])
        Path(run_dir).mkdir(parents=True, exist_ok=True)

        knowledge_path = os.path.join(run_dir, "state_knowledge.json")

        # ----------------------------------------------------------
        # Artifact reuse hierarchy (cheapest first):
        #   1. state_knowledge.json (already extracted)
        #   2. global_map.json (block summaries + outline)
        #   3. segments.json (clean segmented transcript)
        #   4. raw subtitles
        # ----------------------------------------------------------

        # Level 1: Already has knowledge extracted
        if os.path.exists(knowledge_path):
            try:
                with open(knowledge_path) as f:
                    existing_knowledge = json.load(f)
                if existing_knowledge.get("concepts") or existing_knowledge.get("entities"):
                    logger.info("Cache HIT: reusing existing knowledge from %s", knowledge_path)
                    state.add_episode_knowledge(episode_num, existing_knowledge)
                    continue
            except (json.JSONDecodeError, KeyError):
                pass

        segments = None
        global_data = None

        # Level 2: Has global map (block summaries + outline) — the ideal reduced artifact
        global_map_path = os.path.join(run_dir, "global_map.json")
        segments_path = os.path.join(run_dir, "segments.json")

        if os.path.exists(global_map_path):
            logger.info("Cache HIT: reusing global map from %s", global_map_path)
            with open(global_map_path) as f:
                global_data = json.load(f)
            # Also load segments if available (may be needed for block assignment)
            if os.path.exists(segments_path):
                with open(segments_path) as f:
                    segments = json.load(f).get("segments", [])

        # Level 3: Only has segments (clean transcript)
        elif os.path.exists(segments_path):
            logger.info("Cache MISS for global map — reusing segments only")
            with open(segments_path) as f:
                segments = json.load(f).get("segments", [])

        # Level 4: Nothing cached — build from raw subtitles
        else:
            logger.info("Cache MISS — downloading and processing from scratch")
            # Resume fallback: check if we already have the SRT in the run dir
            local_srt = os.path.join(run_dir, "source_subtitles.srt")
            if os.path.exists(local_srt):
                logger.info("Reusing local SRT copy: %s", local_srt)
                subtitle_path = local_srt
            else:
                subtitle_path = download_subtitles(
                    url=video_url,
                    output_dir=download_dir,
                    video_id=src["id"],
                    lang=cfg.lang,
                    prefer_auto=cfg.prefer_auto_subs,
                )

            if not subtitle_path:
                logger.warning("No subtitles for episode %d, skipping", episode_num)
                _write_json(knowledge_path, {})
                continue

            shutil.copy2(subtitle_path, os.path.join(run_dir, "source_subtitles.srt"))

            logger.info("Parsing and segmenting...")
            cleaned = load_subtitles(subtitle_path)
            seg_objects = resegment(
                entries=cleaned,
                gap_threshold=cfg.segment_gap_threshold,
                gap_sentence_threshold=cfg.segment_gap_sentence_threshold,
                max_words=cfg.segment_max_words,
                min_words=cfg.segment_min_words,
                sentence_overflow_words=cfg.sentence_overflow_words,
            )

            # Pass 2: LLM-based per-boundary BREAK/CONTINUE refinement
            if cfg.refine_segments and len(seg_objects) > 1:
                seg_objects = refine_segments(
                    rough_segments=seg_objects,
                    entries=cleaned,
                    model=cfg.default_model,  # use 3b — fast, tiny output
                    host=cfg.ollama_host,
                    timeout=min(cfg.ollama_timeout, 60),
                    target_min_words=50,
                    target_max_words=200,
                )

            segments = [s.to_dict() for s in seg_objects]

            _write_json(
                segments_path,
                {"segments": segments, "pipeline": "podcastcondensor", "version": "0.3.0"},
            )
            logger.info("  → %d segments", len(segments))

        # If we have segments but no global_map, run Phase A to produce block summaries + outline
        if segments is not None and global_data is None and not dry_run:
            logger.info("Building global map (block summaries + outline)...")
            segments_for_map = []
            for s in segments:
                s_copy = dict(s)
                s_copy["uid"] = s["segment_id"]
                segments_for_map.append(s_copy)

            try:
                global_data = build_global_map(
                    chunks=segments_for_map,
                    model=model,
                    block_prompt_path=cfg.block_summary_prompt_path,
                    outline_prompt_path=cfg.outline_prompt_path,
                    block_size_words=cfg.block_size_words,
                    host=cfg.ollama_host,
                    timeout=cfg.ollama_timeout,
                )
                save_data = {
                    "block_summaries": global_data["block_summaries"],
                    "global_outline": global_data["global_outline"],
                    "segment_to_block": global_data.get("chunk_to_block", {}),
                    "num_blocks": len(global_data.get("blocks", [])),
                }
                _write_json(global_map_path, save_data)
                outline_path = os.path.join(run_dir, "global_outline.md")
                with open(outline_path, "w") as f:
                    f.write("# Global Episode Outline\n\n")
                    f.write(global_data["global_outline"])
            except Exception as e:
                logger.error("Phase A failed for episode %d: %s", episode_num, e)
                _write_json(knowledge_path, {})
                continue

        # Extract knowledge from the best available reduced artifact
        if dry_run:
            logger.info("Dry-run: skipping knowledge extraction")
            continue

        if global_data is not None:
            # Use block summaries + outline (compact — ~1 LLM call)
            logger.info("Extracting knowledge from block summaries + outline (1 call)")
            block_summaries = global_data.get("block_summaries", [])
            global_outline = global_data.get("global_outline", "")

            for attempt in range(2):
                EXTRACTION_TIMEOUT = 300
                try:
                    knowledge = UniverseState.extract_knowledge(
                        block_summaries=block_summaries,
                        global_outline=global_outline,
                        episode_title=title,
                        episode_number=episode_num,
                        model=model,
                        prompt_path=cfg.extract_concepts_prompt_path,
                        host=cfg.ollama_host,
                        timeout=EXTRACTION_TIMEOUT,
                    )
                    if knowledge and any(knowledge.values()):
                        break
                    logger.warning("Extraction empty for ep %d, retrying...", episode_num)
                    knowledge = {}
                except Exception as e:
                    logger.warning("Extraction failed for ep %d: %s%s",
                                   episode_num, e,
                                   "" if attempt == 0 else " — giving up")
                    knowledge = {}
        elif segments is not None:
            # Fallback: extract from raw segment text (multiple calls for long episodes)
            logger.info("Extracting knowledge directly from segment text (no global map available)")
            all_texts = [s.get("text", "") for s in segments]
            words = " ".join(all_texts).split()
            chunk_size = 2500
            chunks = [" ".join(words[i:i + chunk_size]) for i in range(0, max(1, len(words)), chunk_size)]
            logger.info("  %d words in %d chunks", len(words), len(chunks))

            with open(cfg.extract_concepts_prompt_path) as f:
                prompt_template = f.read()

            merged = {
                "entities": [], "concepts": [], "claims": [],
                "scriptural_links": [], "historical_links": [],
                "glossary": [], "open_threads": [], "canonical_repetitions": [],
            }
            for ci, chunk_text in enumerate(chunks):
                payload = json.dumps({
                    "episode_title": title,
                    "episode_number": episode_num,
                    "transcript_text": chunk_text,
                }, ensure_ascii=False)
                try:
                    raw = generate(
                        prompt=prompt_template.strip() + "\n\n" + payload,
                        model=model, host=cfg.ollama_host,
                        timeout=120, temperature=0.1, force_json=True,
                    )
                    ck = _parse_extract_response(raw)
                    if ck:
                        for key in merged:
                            merged[key].extend(ck.get(key, []))
                except Exception as e:
                    logger.warning("  Chunk %d failed: %s", ci + 1, e)
            knowledge = merged
        else:
            logger.warning("No transcript data for episode %d", episode_num)
            _write_json(knowledge_path, {})
            continue

        _write_json(knowledge_path, knowledge)

        any_items = any(len(v) for v in knowledge.values())
        if any_items:
            state.add_episode_knowledge(episode_num, knowledge)
            logger.info(
                "  → Added %d concepts, %d entities, %d claims to state",
                len(knowledge.get("concepts", [])),
                len(knowledge.get("entities", [])),
                len(knowledge.get("claims", [])),
            )
        else:
            logger.warning("  → Empty knowledge, nothing added")

    # Final save

    # Final save
    state.save()
    logger.info("=" * 60)
    logger.info("BUILD COMPLETE")
    logger.info("  Episodes processed: %d", state.data["metadata"].get("last_built_episode", 0))
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
) -> List[dict]:
    """Process episodes using an existing UniverseState.

    Runs the full pipeline (download → classify → cut audio) for each episode,
    using the UniverseState as context during classification. After each episode,
    new knowledge is extracted and the state is updated.

    Args:
        playlist_url: YouTube playlist URL
        cfg: Pipeline config
        state: Loaded UniverseState instance
        start_episode: 1-based index of first episode to process (default 21)
        end_episode: Optional 1-based index of last episode
        dry_run: If True, skip LLM calls

    Returns:
        List of result dicts, one per episode.
    """
    # Resolve episode sources
    # end_episode=0 means "process only the start episode"
    effective_end = end_episode if end_episode and end_episode >= start_episode else start_episode
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
        source_type = src.get("source_type", "unknown")

        logger.info("=" * 60)
        logger.info("Processing Episode %d: %s [%s]", episode_num, title, source_type)
        logger.info("URL: %s", video_url)
        logger.info("=" * 60)

        try:
            result = run_pipeline(
                url=video_url,
                cfg=cfg,
                dry_run=dry_run,
                universe_state=state,
                episode_num=episode_num,
            )

            errors = result.get("errors", [])

            results.append({
                "episode": episode_num,
                "title": title,
                "url": video_url,
                "success": len(errors) == 0,
                "errors": errors,
                "output_dir": result.get("output_dir"),
                "condensed_audio": result.get("phases", {}).get("audio", {}).get("condensed_path"),
                "universe_state_updated": result.get("phases", {}).get("universe_state", {}).get("knowledge_extracted", False),
            })

            # Log stats
            stats_data = {}
            stats_path = os.path.join(result.get("output_dir", ""), "stats.json")
            if os.path.exists(stats_path):
                try:
                    with open(stats_path) as f:
                        stats_data = json.load(f)
                    logger.info(
                        "Episode %d: %d/%d kept (%.1f%% compression, %.1fmin → %.1fmin)",
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
            logger.exception("Failed to process episode %d: %s", episode_num, e)
            results.append({
                "episode": episode_num,
                "title": title,
                "url": video_url,
                "success": False,
                "error": str(e),
            })

    # Summary
    successful = sum(1 for r in results if r.get("success"))
    logger.info("=" * 60)
    logger.info("PROCESSING COMPLETE")
    logger.info("  Episodes processed: %d/%d", successful, len(results))
    logger.info("  Universe state now has %d concepts",
                len(state.data.get("concepts", [])))
    logger.info("=" * 60)

    return results


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def _write_json(path: str, data: dict):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def _parse_extract_response(raw: str) -> Optional[dict]:
    """Parse the JSON response from knowledge extraction.

    Handles markdown code fences and surrounding text.
    """
    if not raw:
        return None
    text = raw.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        clean = []
        in_block = False
        for line in lines:
            if line.strip().startswith("```"):
                in_block = not in_block
                continue
            if in_block:
                clean.append(line)
        if clean:
            text = "\n".join(clean)
    start = text.find("{")
    if start >= 0:
        end = text.rfind("}")
        if end > start:
            candidate = text[start:end + 1]
            try:
                result = json.loads(candidate)
                if isinstance(result, dict):
                    return result
            except json.JSONDecodeError:
                pass
    return None


