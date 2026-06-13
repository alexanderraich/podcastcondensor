"""Pipeline — orchestrate the full condensing workflow.

Three-phase architecture:
  Phase A: Global episode map (blocks → summaries → outline)
  Phase B: Segment classification with global context
  Phase C: Global cleanup / dedup
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
from podcastcondensor.rechunker import resegment, resolve_block_ids, Segment
from podcastcondensor.global_map import build_global_map
from podcastcondensor.ollama_client import check_ollama, find_best_model
from podcastcondensor.classifier import (
    classify_segments,
    global_cleanup,
    resolve_maybe,
)
from podcastcondensor.intervals import build_intervals, compute_stats
from podcastcondensor.audio import build_condensed_audio, get_audio_duration

logger = logging.getLogger(__name__)


def run_pipeline(
    url: str,
    cfg: Config,
    dry_run: bool = False,
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
    download_dir = os.path.join(cfg.output_root, "downloads")
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

    model = find_best_model(cfg.default_model, cfg.fallback_model, cfg.ollama_host)
    if model is None:
        artifacts["errors"].append(f"No model found. Pull: ollama pull {cfg.default_model}")
        return artifacts
    artifacts["config"]["model_used"] = model

    # ----------------------------------------------------------------
    # Phase 2: Parse subtitles → clean → resegment
    # ----------------------------------------------------------------
    logger.info("=== Phase 2: Parse, clean & resegment ===")
    if _exists("segments.json"):
        with open(_ap("segments.json")) as f:
            segments = json.load(f)["segments"]
        logger.info("Reusing %d segments from disk", len(segments))
    else:
        if not meta["subtitle_path"]:
            artifacts["errors"].append("No subtitles available.")
            return artifacts

        # Copy raw subtitles
        shutil.copy2(meta["subtitle_path"], _ap("source_subtitles.srt"))

        # Parse and clean (load_subtitles internally cleans)
        cleaned = load_subtitles(meta["subtitle_path"])

        # Resegment into editing units
        seg_objects = resegment(
            entries=cleaned,
            gap_threshold=cfg.segment_gap_threshold,
            max_words=cfg.segment_max_words,
            min_words=cfg.segment_min_words,
        )
        segments = [s.to_dict() for s in seg_objects]

        _write_json(_ap("segments.json"), {"segments": segments, "pipeline": "podcastcondensor", "version": "0.2.0"})

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
        decisions = classify_segments(
            segments=classify_segs,
            model=model,
            prompt_path=cfg.classify_global_prompt_path,
            global_outline=global_data["global_outline"],
            block_summaries=global_data["block_summaries"],
            max_segments_per_batch=cfg.max_segments_per_batch,
            host=cfg.ollama_host,
            ollama_timeout=cfg.ollama_timeout,
            output_path=_ap("decisions.json"),
        )
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
                    model=model,
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
    lines.append(f"- Model: {artifacts['config'].get('model_used', '?')}")
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
