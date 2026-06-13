"""Pipeline — orchestrate the full condensing workflow."""

import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Optional

from podcastcondensor.config import Config
from podcastcondensor.downloader import download_all
from podcastcondensor.subtitles import load_subtitles
from podcastcondensor.chunker import merge_chunks
from podcastcondensor.ollama_client import (
    check_ollama, find_best_model, ensure_model,
)
from podcastcondensor.classifier import classify_chunks, resolve_maybe_chunks
from podcastcondensor.intervals import build_intervals, compute_stats
from podcastcondensor.audio import build_condensed_audio, get_audio_duration

logger = logging.getLogger(__name__)


def run_pipeline(
    url: str,
    cfg: Config,
    dry_run: bool = False,
) -> dict:
    """Run the full podcast condensing pipeline.

    Args:
        url: YouTube URL
        cfg: Configuration
        dry_run: If True, skip LLM and audio steps

    Returns:
        Dict with paths to all output artifacts
    """
    # Phase 0: Setup
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
            "max_chunks_per_batch": cfg.max_chunks_per_batch,
        },
        "phases": {},
        "errors": [],
    }

    def _artifact_path(basename):
        return os.path.join(cfg.output_root, run_dir, basename)

    # Phase 1: Download
    logger.info("=== Phase 1: Download ===")
    meta = download_all(
        url,
        output_dir=os.path.join(cfg.output_root, f"{run_timestamp}_download"),
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
    run_dir = f"{run_timestamp}_{meta['video_id']}"
    Path(os.path.join(cfg.output_root, run_dir)).mkdir(parents=True, exist_ok=True)

    if dry_run:
        logger.info("Dry run: stopping after download")
        return artifacts

    # Verify Ollama is available
    if not check_ollama(cfg.ollama_host):
        error_msg = (
            "Ollama is not running. Start it with: ollama serve"
        )
        logger.error(error_msg)
        artifacts["errors"].append(error_msg)
        return artifacts

    # Select model
    model = find_best_model(
        preferred=cfg.default_model,
        fallback=cfg.fallback_model,
        host=cfg.ollama_host,
    )
    if model is None:
        error_msg = (
            "No suitable model found. Pull one with:\n"
            f"  ollama pull {cfg.default_model}\n"
            f"  ollama pull {cfg.fallback_model}"
        )
        artifacts["errors"].append(error_msg)
        return artifacts
    artifacts["config"]["model_used"] = model

    # Phase 2: Parse subtitles
    logger.info("=== Phase 2: Parse subtitles ===")
    if meta["subtitle_path"]:
        raw_chunks = load_subtitles(meta["subtitle_path"])
        # Save original subtitle file
        import shutil
        shutil.copy2(
            meta["subtitle_path"],
            _artifact_path("source_subtitles.srt"),
        )
        artifacts["phases"]["subtitles"] = {
            "source": meta["subtitle_path"],
            "entry_count": len(raw_chunks),
        }
    else:
        logger.warning("No subtitles found. Pipeline cannot proceed.")
        artifacts["errors"].append("No subtitles available for this video.")
        return artifacts

    # Phase 3: Chunking
    logger.info("=== Phase 3: Chunking ===")
    chunks = merge_chunks(
        raw_chunks,
        max_chars=cfg.max_chars_per_chunk,
        merge_gap=cfg.merge_gap_seconds,
    )
    _write_json(_artifact_path("normalized_chunks.json"), chunks)
    artifacts["phases"]["chunking"] = {"chunk_count": len(chunks)}

    # Phase 4: Classification
    logger.info("=== Phase 4: Classification ===")
    decisions = classify_chunks(
        chunks=chunks,
        model=model,
        prompt_path=cfg.classify_prompt_path,
        max_chunks_per_batch=cfg.max_chunks_per_batch,
        host=cfg.ollama_host,
        ollama_timeout=cfg.ollama_timeout,
        max_chars_per_chunk=cfg.max_chars_per_chunk,
    )
    _write_json(_artifact_path("first_pass_decisions.json"), decisions)
    artifacts["phases"]["classification"] = {
        "decision_count": len(decisions),
    }

    # Phase 5: Maybe resolution
    if cfg.resolve_maybe:
        logger.info("=== Phase 5: Resolve Maybe ===")
        maybe_uids = [
            d["id"] for d in decisions if d.get("label") == "maybe"
        ]
        maybe_chunks = [c for c in chunks if c["uid"] in maybe_uids]
        if maybe_chunks:
            decisions = resolve_maybe_chunks(
                maybe_chunks=maybe_chunks,
                all_chunks=chunks,
                all_decisions=decisions,
                model=model,
                prompt_path=cfg.resolve_maybe_prompt_path,
                host=cfg.ollama_host,
                ollama_timeout=cfg.ollama_timeout,
            )
            _write_json(_artifact_path("maybe_resolution.json"), decisions)
            artifacts["phases"]["maybe_resolution"] = {
                "resolved_count": len(maybe_chunks),
            }
        else:
            logger.info("No maybe chunks to resolve")
            artifacts["phases"]["maybe_resolution"] = {"resolved_count": 0}

    # Phase 6: Build intervals
    logger.info("=== Phase 6: Build intervals ===")
    audio_duration = get_audio_duration(meta["audio_path"])
    intervals = build_intervals(
        chunks=chunks,
        decisions=decisions,
        merge_gap=cfg.output_merge_gap,
        pad_before=cfg.pad_before,
        pad_after=cfg.pad_after,
        audio_duration=audio_duration,
    )
    _write_json(_artifact_path("keep_intervals.json"), intervals)
    artifacts["phases"]["intervals"] = {"interval_count": len(intervals)}

    # Compute stats
    stats = compute_stats(chunks, decisions, intervals)
    _write_json(_artifact_path("stats.json"), stats)

    # Phase 7: Audio cutting
    logger.info("=== Phase 7: Audio cutting ===")
    if intervals:
        condensed_path = _artifact_path(
            f"condensed_{meta['video_id']}.{cfg.audio_format}"
        )
        build_condensed_audio(
            audio_path=meta["audio_path"],
            intervals=intervals,
            output_path=condensed_path,
            format_spec=cfg.audio_format,
            sample_rate=cfg.audio_sample_rate,
            bitrate=cfg.audio_bitrate,
            keep_temp=cfg.keep_temp,
        )
        artifacts["phases"]["audio"] = {
            "condensed_path": condensed_path,
        }
    else:
        logger.warning("No intervals to cut — skipping audio phase")
        artifacts["phases"]["audio"] = {"condensed_path": None}

    # Phase 8: Write review
    logger.info("=== Phase 8: Write review ===")
    review_path = _write_review(_artifact_path("review.md"), stats, artifacts)
    artifacts["phases"]["review"] = {"review_path": review_path}

    # Save pipeline manifest
    _write_json(
        _artifact_path("pipeline_manifest.json"), artifacts
    )

    logger.info("=== Pipeline complete ===")
    logger.info(
        "Original: %.1fs → Condensed: %.1fs (%.1f%%)",
        stats["original_duration_sec"],
        stats["condensed_duration_sec"],
        stats["compression_ratio"] * 100,
    )

    return artifacts


def _write_json(path, data):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    return path


def _write_review(path: str, stats: dict, artifacts: dict) -> str:
    """Write human-readable review file."""
    lines = []
    lines.append("# Podcast Condensor — Review")
    lines.append("")
    lines.append(f"- URL: {artifacts['url']}")
    lines.append(f"- Video ID: {artifacts['phases'].get('download', {}).get('video_id', '?')}")
    lines.append(f"- Title: {artifacts['phases'].get('download', {}).get('title', '?')}")
    lines.append(f"- Model: {artifacts['config'].get('model_used', '?')}")
    lines.append(f"- Run: {artifacts['timestamp']}")
    lines.append("")
    lines.append("## Statistics")
    lines.append("")
    lines.append(f"- Total chunks: {stats['total_chunks']}")
    lines.append(f"- Kept: {stats['keep_count']}")
    lines.append(f"- Dropped: {stats['drop_count']}")
    lines.append(f"- Maybe: {stats['maybe_count']}")
    lines.append(f"- Original duration: {_fmt_duration(stats['original_duration_sec'])}")
    lines.append(f"- Condensed duration: {_fmt_duration(stats['condensed_duration_sec'])}")
    lines.append(f"- Compression ratio: {stats['compression_ratio']:.1%}")
    lines.append("")
    lines.append("## Uncertain Regions (Maybe)")
    lines.append("")
    if stats["maybe_chunks"]:
        for mc in stats["maybe_chunks"]:
            lines.append(
                f"- [{mc['uid']}] {_fmt_duration(mc['start'])}-"
                f"{_fmt_duration(mc['end'])}: {mc['text']}"
            )
    else:
        lines.append("(none — all resolved or none flagged)")
    lines.append("")
    lines.append("## Artifacts")
    lines.append("")
    lines.append(f"- Audio: {artifacts['phases'].get('audio', {}).get('condensed_path', 'N/A')}")
    lines.append("- JSON artifacts in output directory for detailed inspection")

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
