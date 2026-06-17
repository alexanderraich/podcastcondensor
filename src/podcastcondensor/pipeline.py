"""Pipeline — orchestrate condensing workflow (DeepSeek-only).

Phases (see CLAUDE.md):
  1: Download         — yt-dlp: raw MP3 + raw SRT
  2: Global state     — single DeepSeek call: outline + universe knowledge
  3: Segmentation     — DeepSeek punctuate + sentence-range segment
  4: Classification   — DeepSeek single-batch with universe context
  5: Finalize         — dedup, resolve maybes, continuity bias, tail detect
  6: Audio cutting    — build intervals, ffmpeg with beeps

Every phase writes its primary artefact and checks for it on re-run (resumable).
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
from podcastcondensor.global_state import build_global_state, map_blocks_to_segments
from podcastcondensor.classifier import (
    classify_segments,
    global_cleanup,
    resolve_maybe,
    apply_continuity_bias,
    detect_tail_block,
)
from podcastcondensor.intervals import build_intervals, compute_stats
from podcastcondensor.audio_strategies import _get_audio_duration as get_audio_duration
from podcastcondensor.audio_strategies import create_audio_strategy
from podcastcondensor.universe_state import UniverseState
from podcastcondensor.llm.deepseek import resolve_api_key, DeepSeekClient
from podcastcondensor.segmentation.deepseek import DeepSeekSegmentation

logger = logging.getLogger(__name__)


def run_pipeline(
    url: str,
    cfg: Config,
    dry_run: bool = False,
    universe_state: Optional[UniverseState] = None,
    episode_num: Optional[int] = None,
) -> dict:
    run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    artifacts = {
        "url": url,
        "timestamp": run_timestamp,
        "config": {
            "model": cfg.deepseek_model,
            "lang": cfg.lang,
            "merge_gap": cfg.output_merge_gap,
            "speed": cfg.audio_speed,
        },
        "phases": {},
        "errors": [],
    }

    def _ap(basename):
        return os.path.join(run_dir, basename)

    def _write_json(path, data):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        return path

    def _load_json(path):
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        return None

    # Resolve video ID
    from podcastcondensor.downloader import (
        extract_video_id,
        _find_existing_audio,
        _find_existing_subtitle,
    )
    local_video_id = extract_video_id(url)
    if not local_video_id:
        from podcastcondensor.downloader import download_metadata
        local_video_id = download_metadata(url)["id"]

    # Run dir
    if episode_num is not None:
        run_dir = os.path.join(cfg.output_root, f"ep-{episode_num:03d}")
    else:
        run_dir = os.path.join(cfg.output_root, local_video_id)
    Path(run_dir).mkdir(parents=True, exist_ok=True)

    # DeepSeek client
    api_key = resolve_api_key()
    if not api_key:
        artifacts["errors"].append(
            "DeepSeek API key not set (ANTHROPIC_AUTH_TOKEN or DEEPSEEK_API_KEY)"
        )
        return artifacts
    ds = DeepSeekClient(api_key=api_key)

    # ================================================================
    # Phase 1: Download  (artefact: source_subtitles.srt + audio)
    # ================================================================
    logger.info("=== Phase 1: Download ===")
    if os.path.exists(_ap("source_subtitles.srt")):
        logger.info("Checkpoint HIT — SRT exists, reusing")
        audio_path = _find_existing_audio(run_dir, local_video_id, cfg.audio_format)
        if not audio_path:
            from podcastcondensor.downloader import download_audio
            audio_path = download_audio(
                url, run_dir, local_video_id,
                audio_format=cfg.audio_format,
                audio_bitrate=cfg.audio_bitrate,
            )
        meta = {
            "video_id": local_video_id,
            "title": local_video_id,
            "audio_path": audio_path,
            "subtitle_path": _find_existing_subtitle(run_dir, local_video_id) or "",
        }
    else:
        meta = download_all(
            url, output_dir=run_dir, lang=cfg.lang,
            prefer_auto=cfg.prefer_auto_subs,
            audio_format=cfg.audio_format,
            audio_bitrate=cfg.audio_bitrate,
        )
        # Normalise SRT name
        srt_source = meta.get("subtitle_path") or ""
        if srt_source and srt_source != _ap("source_subtitles.srt"):
            shutil.copy2(srt_source, _ap("source_subtitles.srt"))
            if os.path.exists(srt_source):
                os.remove(srt_source)

    artifacts["phases"]["download"] = {
        "video_id": meta["video_id"],
        "title": meta["title"],
        "audio_path": meta["audio_path"],
    }

    if dry_run:
        logger.info("Dry run: stopping after download")
        return artifacts

    # Ensure source_subtitles.srt exists
    if not os.path.exists(_ap("source_subtitles.srt")):
        artifacts["errors"].append("No subtitles available after download.")
        return artifacts

    # ================================================================
    # Phase 2: Global state  (artefact: global_state.json)
    # ================================================================
    logger.info("=== Phase 2: Global state ===")
    global_data = _load_json(_ap("global_state.json"))
    if global_data is not None:
        logger.info("Checkpoint HIT — loaded global_state.json")
    else:
        cleaned = load_subtitles(_ap("source_subtitles.srt"))
        from podcastcondensor.segmentation.sentence_units import (
            build_transcript_from_entries,
        )
        transcript_text = build_transcript_from_entries(cleaned)

        global_data = build_global_state(
            transcript_text=transcript_text,
            episode_title=meta.get("title", ""),
            episode_number=episode_num,
            client=ds,
            model=cfg.deepseek_model,
            prompt_path=cfg.global_state_prompt_path,
            timeout=cfg.deepseek_timeout,
        )

        # Merge structured knowledge into universe state if available
        if universe_state is not None:
            knowledge = {
                "summary": global_data.get("summary", ""),
                "entities": global_data.get("entities", []),
                "concepts": global_data.get("concepts", []),
                "claims": global_data.get("claims", []),
                "scriptural_links": global_data.get("scriptural_links", []),
                "glossary": global_data.get("glossary", []),
            }
            universe_state.add_episode_knowledge(episode_num or 0, knowledge)

        # Persist
        _write_json(_ap("global_state.json"), global_data)

    artifacts["phases"]["global_state"] = {
        "num_blocks": len(global_data.get("block_summaries", [])),
        "entities": len(global_data.get("entities", [])),
        "concepts": len(global_data.get("concepts", [])),
    }

    # ================================================================
    # Phase 3: Segmentation  (artefact: segments.json)
    # ================================================================
    logger.info("=== Phase 3: Segmentation ===")
    segments = _load_json(_ap("segments.json"))
    if segments is not None:
        segments = segments.get("segments", segments)
        logger.info("Checkpoint HIT — loaded %d segments from segments.json", len(segments))
    else:
        cleaned = load_subtitles(_ap("source_subtitles.srt"))
        from podcastcondensor.segmentation.sentence_units import (
            build_transcript_from_entries,
        )
        transcript_text = build_transcript_from_entries(cleaned)

        seg_strategy = DeepSeekSegmentation(
            client=ds, timeout=cfg.deepseek_timeout,
            max_tokens=cfg.segmentation_max_tokens,
        )
        segments = seg_strategy.segment(
            entries=cleaned, transcript_text=transcript_text,
        )
        for s in segments:
            s["block_id"] = s.get("block_id", 0)

        _write_json(_ap("segments.json"), {
            "segments": segments,
            "pipeline": "podcastcondensor",
            "version": "0.3.0",
        })

    artifacts["phases"]["segmentation"] = {"segment_count": len(segments)}

    # Map segments to topic blocks by word-index overlap
    seg_to_block = global_data.get("chunk_to_block") or {}
    if not seg_to_block:
        # Rebuild transcript for word-offset calculation
        seg_text = " ".join(s["text"] for s in segments)
        seg_to_block = map_blocks_to_segments(
            segments,
            global_data.get("block_summaries", []),
            seg_text,
        )
        global_data["chunk_to_block"] = seg_to_block

    block_summaries = global_data.get("block_summaries", [])
    global_outline = global_data.get("global_outline", "")

    for s in segments:
        bid = seg_to_block.get(s.get("segment_id", s.get("uid")), 0)
        s["block_id"] = bid
        if bid == 0 and isinstance(s.get("uid"), str):
            bid = seg_to_block.get(s["uid"], 0)
            s["block_id"] = bid

    # max_blocks filter
    if cfg.max_blocks > 0 and len(block_summaries) > cfg.max_blocks:
        classify_segs = [s for s in segments if s["block_id"] <= cfg.max_blocks]
        auto_drop_segs = [s for s in segments if s["block_id"] > cfg.max_blocks]
    else:
        classify_segs = segments
        auto_drop_segs = []

    # ================================================================
    # Phase 4: Classification  (artefact: none — decisions flow to Phase 5)
    # ================================================================
    logger.info("=== Phase 4: Classification ===")

    universe_ctx = ""
    if universe_state is not None:
        universe_ctx = universe_state.get_context()

    decisions = classify_segments(
        segments=classify_segs, client=ds,
        global_outline=global_outline,
        block_summaries=block_summaries,
        model=cfg.deepseek_model,
        prompt_path=cfg.classify_global_prompt_path,
        timeout=cfg.deepseek_timeout,
        universe_state_context=universe_ctx,
    )

    if auto_drop_segs:
        existing_ids = {d["id"] for d in decisions}
        for s in auto_drop_segs:
            if s["segment_id"] not in existing_ids:
                decisions.append({
                    "id": s["segment_id"],
                    "label": "drop",
                    "reason": "beyond-max-blocks",
                })

    artifacts["phases"]["classification"] = {"decision_count": len(decisions)}

    _write_json(_ap("decisions.json"), decisions)

    # ================================================================
    # Phase 5: Finalize decisions  (artefact: decisions_final.json)
    # ================================================================
    logger.info("=== Phase 5: Finalize decisions ===")
    decisions = global_cleanup(segments, decisions)

    maybe_ids = [d["id"] for d in decisions if d.get("label") == "maybe"]
    if maybe_ids and cfg.resolve_maybe:
        maybe_segs = [s for s in segments if s["segment_id"] in maybe_ids]
        if maybe_segs:
            logger.info("Resolving %d maybe segments...", len(maybe_segs))
            decisions = resolve_maybe(
                maybe_segments=maybe_segs, all_segments=segments,
                all_decisions=decisions, client=ds,
                model=cfg.deepseek_model,
                prompt_path=cfg.classify_global_prompt_path,
                timeout=120,
            )

    if cfg.enable_continuity_bias:
        logger.info("Applying continuity bias...")
        decisions = apply_continuity_bias(
            segments, decisions, bridge_gap_sec=cfg.bridge_gap_sec,
        )

    if cfg.enable_tail_detection:
        tail_ids = detect_tail_block(
            segments, decisions,
            tail_fraction=cfg.tail_fraction,
            min_keep_fraction=cfg.tail_min_keep_fraction,
        )
        if tail_ids:
            logger.info("Tail detection: force-dropping %d segments", len(tail_ids))
            for d in decisions:
                if d["id"] in tail_ids:
                    d["label"] = "drop"
                    d["reason"] = "tail"

    _write_json(_ap("decisions_final.json"), decisions)
    artifacts["phases"]["finalize"] = {"decision_count": len(decisions)}

    if cfg.decisions_only:
        logger.info("decisions_only=True — skipping audio cutting")
        audio_duration = get_audio_duration(meta["audio_path"])
        intervals = build_intervals(
            segments, decisions,
            merge_gap=cfg.output_merge_gap,
            pad_before=cfg.pad_before,
            pad_after=cfg.pad_after,
            audio_duration=audio_duration,
            cluster_gap=0,
        )
        stats = compute_stats(segments, decisions, intervals)
        _write_json(_ap("stats.json"), stats)
        artifacts["phases"]["intervals"] = {"interval_count": len(intervals)}
        _print_results(stats, artifacts)
        return artifacts

    # ================================================================
    # Phase 6: Audio cutting  (artefact: condensed_*.mp3)
    # ================================================================
    logger.info("=== Phase 6: Audio cutting ===")
    audio_duration = get_audio_duration(meta["audio_path"])

    intervals = build_intervals(
        segments, decisions,
        merge_gap=cfg.output_merge_gap,
        pad_before=cfg.pad_before,
        pad_after=cfg.pad_after,
        audio_duration=audio_duration,
        cluster_gap=0,  # no clustering — decisions are final
    )
    artifacts["phases"]["intervals"] = {"interval_count": len(intervals)}

    condensed_path = _ap(f"condensed_{meta['video_id']}.{cfg.audio_format}")
    if intervals:
        strategy = create_audio_strategy(cfg.audio_strategy)
        strategy.cut(
            audio_path=meta["audio_path"], intervals=intervals,
            output_path=condensed_path, format_spec=cfg.audio_format,
            sample_rate=cfg.audio_sample_rate, bitrate=cfg.audio_bitrate,
            speed=cfg.audio_speed,
        )
        artifacts["phases"]["audio"] = {"condensed_path": condensed_path}
    else:
        artifacts["phases"]["audio"] = {"condensed_path": None}

    # ================================================================
    # Results
    # ================================================================
    stats = compute_stats(segments, decisions, intervals)
    _write_json(_ap("stats.json"), stats)
    artifacts["output_dir"] = run_dir
    _print_results(stats, artifacts)
    return artifacts


def _print_results(stats: dict, artifacts: dict):
    speed = artifacts["config"].get("speed", 1.0)
    print("")
    print("=" * 50)
    print("PODCAST CONDENSOR — RESULTS")
    print("=" * 50)
    print(f"  Total segments:    {stats['total_segments']}")
    print(f"  Kept:              {stats['keep_count']}")
    print(f"  Dropped:           {stats['drop_count']}")
    print(f"  Original duration: {_fmt_duration(stats['original_duration_sec'])}")
    cd = stats["condensed_duration_sec"]
    print(f"  Condensed duration:{_fmt_duration(cd)}")
    if speed != 1.0:
        print(f"  At {speed}x:         {_fmt_duration(cd / speed)}")
    print(f"  Compression ratio: {stats['compression_ratio']:.1%}")
    audio = artifacts.get("phases", {}).get("audio", {}).get("condensed_path")
    if audio:
        print(f"  Output:            {audio}")
    print("")


def _fmt_duration(sec: float) -> str:
    m, s = divmod(int(sec), 60)
    h, m = divmod(m, 60)
    if h:
        return f"{h}h{m:02d}m{s:02d}s"
    return f"{m}m{s:02d}s"
