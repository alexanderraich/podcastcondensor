"""Pipeline — orchestrate the full condensing workflow.

Three-phase architecture:
  Phase A: Global episode map (blocks → summaries → outline)
  Phase B: Segment classification with global context
  Phase C: Global cleanup / dedup
  Phase D: Universe state knowledge extraction (optional)
"""

import json
import logging
import os
import shutil
from pathlib import Path
from datetime import datetime
from typing import Optional

from podcastcondensor.config import Config
from podcastcondensor.downloader import download_all
from podcastcondensor.subtitles import load_subtitles
from podcastcondensor.rechunker import resegment, refine_segments, resolve_block_ids, Segment
from podcastcondensor.global_map import build_global_map
from podcastcondensor.ollama_client import check_ollama, find_best_model
from podcastcondensor.classifier import (
    classify_segments,
    global_cleanup,
    resolve_maybe,
)
from podcastcondensor.intervals import build_intervals, compute_stats
from podcastcondensor.audio import build_condensed_audio, get_audio_duration
from podcastcondensor.universe_state import UniverseState

logger = logging.getLogger(__name__)


def run_pipeline(
    url: str,
    cfg: Config,
    dry_run: bool = False,
    universe_state: Optional[UniverseState] = None,
    episode_num: Optional[int] = None,
) -> dict:
    """Run the full podcast condensing pipeline.

    Resumable: each phase checks for existing artifacts before running.
    """
    run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    artifacts = {
        "url": url,
        "timestamp": run_timestamp,
        "config": {
            "model": cfg.default_model,
            "classify_model": cfg.classify_model,
            "lang": cfg.lang,
            "merge_gap": cfg.output_merge_gap,
            "pad_before": cfg.pad_before,
            "pad_after": cfg.pad_after,
            "resolve_maybe": cfg.resolve_maybe,
            "max_segments_per_batch": cfg.max_segments_per_batch,
            "block_size_words": cfg.block_size_words,
            "segment_gap_threshold": cfg.segment_gap_threshold,
            "segment_max_words": cfg.segment_max_words,
            "speed": cfg.audio_speed,
            "segment_min_words": cfg.segment_min_words,
            "segment_gap_sentence_threshold": cfg.segment_gap_sentence_threshold,
            "sentence_overflow_words": cfg.sentence_overflow_words,
            "refine_segments": cfg.refine_segments,
            "refine_batch_size_words": cfg.refine_batch_size_words,
        },
        "phases": {},
        "errors": [],
    }

    def _ap(basename):
        return os.path.join(run_dir, basename)

    def _exists(basename):
        return os.path.exists(_ap(basename))

    def _write_json(path, data):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        return path

    # ----------------------------------------------------------------
    # Phase 1: Download
    # ----------------------------------------------------------------
    logger.info("=== Phase 1: Download ===")
    download_dir = "/tmp/podcastcondensor/downloads"
    meta = download_all(
        url,
        output_dir=download_dir,
        lang=cfg.lang,
        prefer_auto=cfg.prefer_auto_subs,
        audio_format=cfg.audio_format,
        audio_bitrate=cfg.audio_bitrate,
    )
    artifacts["phases"]["download"] = {
        "video_id": meta["video_id"],
        "title": meta["title"],
        "audio_path": meta["audio_path"],
        "subtitle_path": meta["subtitle_path"],
    }

    run_dir = os.path.join(cfg.output_root, meta["video_id"])
    Path(run_dir).mkdir(parents=True, exist_ok=True)

    if dry_run:
        logger.info("Dry run: stopping after download")
        return artifacts

    # ----------------------------------------------------------------
    # Ollama check
    # ----------------------------------------------------------------
    if not check_ollama(cfg.ollama_host):
        artifacts["errors"].append("Ollama is not running. Start: ollama serve")
        return artifacts

    # Resolve separate models for reduction vs. classification
    reduction_model = find_best_model(cfg.default_model, cfg.fallback_model, cfg.ollama_host)
    classify_model = find_best_model(cfg.classify_model, cfg.fallback_model, cfg.ollama_host)
    if reduction_model is None:
        artifacts["errors"].append(f"No model found. Pull: ollama pull {cfg.default_model}")
        return artifacts
    if classify_model is None:
        classify_model = reduction_model
        artifacts["config"]["classify_model_used"] = classify_model
    artifacts["config"]["reduction_model_used"] = reduction_model
    artifacts["config"]["classify_model_used"] = classify_model
    logger.info("Models: reduction=%s  classify=%s", reduction_model, classify_model)

    # ----------------------------------------------------------------
    # Phase 2: Parse subtitles → clean → resegment
    # ----------------------------------------------------------------
    logger.info("=== Phase 2: Parse, clean & resegment ===")
    if _exists("segments.json"):
        with open(_ap("segments.json")) as f:
            segments = json.load(f)["segments"]
        logger.info("Reusing %d segments from disk", len(segments))
    else:
        # Determine subtitle source: prefer download cache, fall back to
        # output-directory copy (so we can resume even if /tmp is cleaned).
        srt_source = meta.get("subtitle_path") or ""
        if not srt_source or not os.path.exists(srt_source):
            srt_source = _ap("source_subtitles.srt")
        if not srt_source or not os.path.exists(srt_source):
            artifacts["errors"].append("No subtitles available.")
            return artifacts

        # Persist a copy to the run directory for future resume
        if srt_source != _ap("source_subtitles.srt"):
            shutil.copy2(srt_source, _ap("source_subtitles.srt"))

        # Parse and clean (load_subtitles internally cleans)
        cleaned = load_subtitles(srt_source)

        # Resegment into editing units
        seg_objects = resegment(
            entries=cleaned,
            gap_threshold=cfg.segment_gap_threshold,
            gap_sentence_threshold=cfg.segment_gap_sentence_threshold,
            max_words=cfg.segment_max_words,
            min_words=cfg.segment_min_words,
            sentence_overflow_words=cfg.sentence_overflow_words,
        )

        # Pass 2: LLM-based semantic refinement (per-boundary BREAK/CONTINUE)
        # Each candidate boundary gets a binary decision from the 3b model.
        # Only called when rules (50-180 word range) don't force a decision.
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

        _write_json(_ap("segments.json"), {"segments": segments, "pipeline": "podcastcondensor", "version": "0.3.0"})

    artifacts["phases"]["segmentation"] = {"segment_count": len(segments)}

    # ----------------------------------------------------------------
    # Phase A: Global episode map
    # ----------------------------------------------------------------
    logger.info("=== Phase A: Global Episode Map ===")
    if _exists("global_map.json"):
        with open(_ap("global_map.json")) as f:
            global_data = json.load(f)
        logger.info(
            "Reusing global map: %d blocks",
            len(global_data.get("block_summaries", [])),
        )
    else:
        # Build global map from segments (they have start/end/text compatible with split_into_blocks)
        segments_for_map = []
        for s in segments:
            s_copy = dict(s)
            s_copy["uid"] = s["segment_id"]
            segments_for_map.append(s_copy)

        global_data = build_global_map(
            chunks=segments_for_map,
            model=reduction_model,
            block_prompt_path=cfg.block_summary_prompt_path,
            outline_prompt_path=cfg.outline_prompt_path,
            block_size_words=cfg.block_size_words,
            host=cfg.ollama_host,
            timeout=cfg.ollama_timeout,
        )
        save_data = {
            "block_summaries": global_data["block_summaries"],
            "global_outline": global_data["global_outline"],
            "segment_to_block": global_data["chunk_to_block"],
            "num_blocks": len(global_data["blocks"]),
        }
        _write_json(_ap("global_map.json"), save_data)
        with open(_ap("global_outline.md"), "w") as f:
            f.write("# Global Episode Outline\n\n")
            f.write(global_data["global_outline"])

    artifacts["phases"]["global_map"] = {
        "num_blocks": len(global_data.get("block_summaries", [])),
        "outline_length": len(global_data.get("global_outline", "")),
    }

    # Assign block_ids to segments (check both in-memory and on-disk key names)
    seg_to_block = (
        global_data.get("segment_to_block")
        or global_data.get("chunk_to_block")
        or {}
    )
    for s in segments:
        s["block_id"] = seg_to_block.get(s["segment_id"], 0)

    # Apply max_blocks limit: segments beyond the limit skip classification
    if cfg.max_blocks > 0 and len(global_data.get("block_summaries", [])) > cfg.max_blocks:
        classify_segs = [s for s in segments if s["block_id"] <= cfg.max_blocks]
        auto_drop_segs = [s for s in segments if s["block_id"] > cfg.max_blocks]
        logger.info(
            "max_blocks=%d: %d segments to classify, %d auto-drop",
            cfg.max_blocks, len(classify_segs), len(auto_drop_segs),
        )
    else:
        classify_segs = segments
        auto_drop_segs = []

    # ----------------------------------------------------------------
    # Phase B: Classify segments with global context
    # ----------------------------------------------------------------
    logger.info("=== Phase B: Classify segments ===")
    if _exists("decisions.json"):
        with open(_ap("decisions.json")) as f:
            decisions = json.load(f)
        logger.info("Reusing %d decisions from disk", len(decisions))
    else:
        universe_state_context = ""
        phase_b_kwargs = dict(
            segments=classify_segs,
            model=classify_model,
            prompt_path=cfg.classify_global_prompt_path,
            global_outline=global_data["global_outline"],
            block_summaries=global_data["block_summaries"],
            max_segments_per_batch=cfg.max_segments_per_batch,
            host=cfg.ollama_host,
            ollama_timeout=cfg.ollama_timeout,
            output_path=_ap("decisions.json"),
        )

        if universe_state is not None:
            universe_state_context = universe_state.get_context(
                max_items=8, max_chars=3000,
                exclude_episode_gte=episode_num,
            )
            phase_b_kwargs["universe_state_context"] = universe_state_context
            logger.info(
                "Universe state context: %d chars (excluding episodes >= %s)",
                len(universe_state_context), episode_num,
            )

        decisions = classify_segments(**phase_b_kwargs)
        if not _exists("decisions.json"):
            _write_json(_ap("decisions.json"), decisions)

        # Drop segments beyond max_blocks (we only want classified blocks)
        if auto_drop_segs:
            existing_ids = {d["id"] for d in decisions}
            for s in auto_drop_segs:
                if s["segment_id"] not in existing_ids:
                    decisions.append({
                        "id": s["segment_id"],
                        "label": "drop",
                        "reason": "beyond-max-blocks",
                    })
            _write_json(_ap("decisions.json"), decisions)
            logger.info("Merged %d auto-drop decisions", len(auto_drop_segs))

    artifacts["phases"]["classification"] = {"decision_count": len(decisions)}

    # ----------------------------------------------------------------
    # Phase C: Global cleanup
    # ----------------------------------------------------------------
    logger.info("=== Phase C: Global Cleanup ===")
    if _exists("decisions_clean.json"):
        with open(_ap("decisions_clean.json")) as f:
            decisions = json.load(f)
        logger.info("Reusing cleaned decisions from disk")
    else:
        decisions = global_cleanup(
            segments=segments,
            decisions=decisions,
        )
        _write_json(_ap("decisions_clean.json"), decisions)

    artifacts["phases"]["cleanup"] = {"decision_count": len(decisions)}

    # ----------------------------------------------------------------
    # Maybe resolution (within Phase C)
    # ----------------------------------------------------------------
    if cfg.resolve_maybe:
        logger.info("=== Resolve maybes ===")
        if _exists("decisions_resolved.json"):
            with open(_ap("decisions_resolved.json")) as f:
                decisions = json.load(f)
            logger.info("Reusing resolved decisions from disk")
        else:
            maybe_ids = [d["id"] for d in decisions if d.get("label") == "maybe"]
            maybe_segs = [s for s in segments if s["segment_id"] in maybe_ids]
            if maybe_segs:
                decisions = resolve_maybe(
                    maybe_segments=maybe_segs,
                    all_segments=segments,
                    all_decisions=decisions,
                    model=classify_model,
                    prompt_path=cfg.resolve_maybe_prompt_path,
                    host=cfg.ollama_host,
                    ollama_timeout=cfg.ollama_timeout,
                )
                _write_json(_ap("decisions_resolved.json"), decisions)
                artifacts["phases"]["maybe_resolution"] = {"resolved": len(maybe_segs)}
            else:
                _write_json(_ap("decisions_resolved.json"), decisions)
                artifacts["phases"]["maybe_resolution"] = {"resolved": 0}

    # ----------------------------------------------------------------
    # Phase D: Universe State knowledge extraction (optional)
    # ----------------------------------------------------------------
    if universe_state is not None and not dry_run and cfg.extract_concepts_prompt_path:
        logger.info("=== Phase D: Extract knowledge for universe state ===")

        # Use block summaries + global outline (already computed in Phase A)
        block_data = global_data.get("block_summaries", [])
        outline_text = global_data.get("global_outline", "")

        ep_title = meta.get("title", "")
        ep_number = episode_num

        # Check if knowledge was already extracted for this run
        if _exists("state_knowledge.json"):
            with open(_ap("state_knowledge.json")) as f:
                knowledge = json.load(f)
            logger.info("Reusing previously extracted knowledge from disk")
        else:
            knowledge = UniverseState.extract_knowledge(
                block_summaries=block_data,
                global_outline=outline_text,
                episode_title=ep_title,
                episode_number=ep_number,
                model=reduction_model,
                prompt_path=cfg.extract_concepts_prompt_path,
                host=cfg.ollama_host,
                timeout=cfg.ollama_timeout,
            )
            _write_json(_ap("state_knowledge.json"), knowledge)

        if knowledge:
            # Fall back to simple concept extraction if knowledge is partial
            if knowledge.get("concepts") or knowledge.get("entities"):
                universe_state.add_episode_knowledge(ep_number or 0, knowledge)
                artifacts["phases"]["universe_state"] = {
                    "knowledge_extracted": True,
                    "concepts_added": len(knowledge.get("concepts", [])),
                    "entities_added": len(knowledge.get("entities", [])),
                }
            else:
                logger.warning("Knowledge extraction returned empty, skipping state update")
                artifacts["phases"]["universe_state"] = {
                    "knowledge_extracted": False,
                    "reason": "empty_result",
                }

    # ----------------------------------------------------------------
    # Build intervals
    # ----------------------------------------------------------------
    logger.info("=== Build intervals ===")
    if _exists("keep_intervals.json"):
        with open(_ap("keep_intervals.json")) as f:
            intervals = json.load(f)
        logger.info("Reusing %d intervals from disk", len(intervals))
    else:
        audio_duration = get_audio_duration(meta["audio_path"])
        intervals = build_intervals(
            segments=segments,
            decisions=decisions,
            merge_gap=cfg.output_merge_gap,
            pad_before=cfg.pad_before,
            pad_after=cfg.pad_after,
            audio_duration=audio_duration,
        )
        _write_json(_ap("keep_intervals.json"), intervals)

        stats = compute_stats(segments, decisions, intervals)
        _write_json(_ap("stats.json"), stats)

    artifacts["phases"]["intervals"] = {"interval_count": len(intervals)}

    # ----------------------------------------------------------------
    # Audio cutting
    # ----------------------------------------------------------------
    logger.info("=== Audio cutting ===")
    condensed_path = _ap(f"condensed_{meta['video_id']}.{cfg.audio_format}")
    if os.path.exists(condensed_path):
        logger.info("Reusing condensed audio from disk")
        artifacts["phases"]["audio"] = {"condensed_path": condensed_path}
    elif intervals:
        build_condensed_audio(
            audio_path=meta["audio_path"],
            intervals=intervals,
            output_path=condensed_path,
            format_spec=cfg.audio_format,
            sample_rate=cfg.audio_sample_rate,
            bitrate=cfg.audio_bitrate,
            keep_temp=cfg.keep_temp,
            speed=cfg.audio_speed,
        )
        artifacts["phases"]["audio"] = {"condensed_path": condensed_path}
    else:
        artifacts["phases"]["audio"] = {"condensed_path": None}

    # ----------------------------------------------------------------
    # Review
    # ----------------------------------------------------------------
    logger.info("=== Review ===")
    if not _exists("stats.json"):
        stats = compute_stats(segments, decisions, intervals)
        _write_json(_ap("stats.json"), stats)
    else:
        with open(_ap("stats.json")) as f:
            stats = json.load(f)

    review_path = _write_review(_ap("review.md"), stats, artifacts)
    artifacts["phases"]["review"] = {"review_path": review_path}

    artifacts["output_dir"] = run_dir
    _write_json(_ap("pipeline_manifest.json"), artifacts)

    logger.info("=== Pipeline complete ===")
    logger.info(
        "Original: %.1fs -> Condensed: %.1fs (%.1f%%)",
        stats["original_duration_sec"],
        stats["condensed_duration_sec"],
        stats["compression_ratio"] * 100,
    )

    # Print stats to stdout so they appear in terminal output
    print("")
    print("=" * 50)
    print("PODCAST CONDENSOR — RESULTS")
    print("=" * 50)
    print(f"  Total segments:    {stats['total_segments']}")
    print(f"  Kept:              {stats['keep_count']}")
    print(f"  Dropped:           {stats['drop_count']}")
    print(f"  Maybe:             {stats['maybe_count']}")
    print(f"  Original duration: {_fmt_duration(stats['original_duration_sec'])}")
    print(f"  Condensed duration:{_fmt_duration(stats['condensed_duration_sec'])}")
    speed_label = f" (at {artifacts['config'].get('speed', 1.0)}x: {_fmt_duration(stats['condensed_duration_sec'] / artifacts['config'].get('speed', 1.0))})" if artifacts['config'].get('speed', 1.0) != 1.0 else ""
    print(f"  Compression ratio: {stats['compression_ratio']:.1%}{speed_label}")
    if artifacts.get('phases', {}).get('audio', {}).get('condensed_path'):
        print(f"  Output:            {artifacts['phases']['audio']['condensed_path']}")
    print("")

    return artifacts


def _write_review(path: str, stats: dict, artifacts: dict) -> str:
    lines = []
    lines.append("# Podcast Condensor - Review")
    lines.append("")
    lines.append(f"- URL: {artifacts['url']}")
    d = artifacts.get("phases", {}).get("download", {})
    lines.append(f"- Video ID: {d.get('video_id', '?')}")
    lines.append(f"- Title: {d.get('title', '?')}")
    lines.append(f"- Reduction model: {artifacts['config'].get('reduction_model_used', '?')}")
    lines.append(f"- Classify model:  {artifacts['config'].get('classify_model_used', '?')}")
    lines.append(f"- Run: {artifacts['timestamp']}")
    lines.append("")
    lines.append("## Statistics")
    lines.append("")
    lines.append(f"- Total segments: {stats['total_segments']}")
    lines.append(f"- Kept: {stats['keep_count']}")
    lines.append(f"- Dropped: {stats['drop_count']}")
    lines.append(f"- Maybe: {stats['maybe_count']}")
    lines.append(f"- Original content: {_fmt_duration(stats['original_duration_sec'])}")
    lines.append(f"- Condensed: {_fmt_duration(stats['condensed_duration_sec'])}")
    lines.append(f"- Compression ratio: {stats['compression_ratio']:.1%}")
    lines.append("")
    lines.append("## Uncertain Regions")
    lines.append("")
    if stats.get("maybe_segments"):
        for ms in stats["maybe_segments"]:
            lines.append(
                f"- [{ms['id']}] {_fmt_duration(ms['start'])}-"
                f"{_fmt_duration(ms['end'])}: {ms['text']}"
            )
    else:
        lines.append("(none)")
    lines.append("")
    lines.append("## Output")
    lines.append("")
    lines.append(f"- Condensed audio: {artifacts['phases'].get('audio', {}).get('condensed_path', 'N/A')}")
    lines.append(f"- Directory: {artifacts.get('output_dir', '?')}")

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    return path


def _fmt_duration(sec: float) -> str:
    m, s = divmod(int(sec), 60)
    h, m = divmod(m, 60)
    if h:
        return f"{h}h{m:02d}m{s:02d}s"
    return f"{m}m{s:02d}s"
