"""Pipeline — orchestrate condensing workflow (DeepSeek-only).

Phases:
  1: Download          — yt-dlp: raw MP3 + raw SRT
  2: Global state      — single DeepSeek call: outline + universe knowledge
  3: Classify raw      — single DeepSeek call: tick keep/drop per SRT entry
  4: Audio cutting     — build intervals from kept entries' real timestamps

No segmentation, no punctuation fix, no sentence-to-entry mapping, no post-processing.
The LLM decides directly on raw SRT entries. Universe state is provided as context
so the LLM can avoid re-keeping content already covered in prior episodes.

Every phase writes its primary artefact and checks for it on re-run (resumable).
"""

import json
import logging
import os
from pathlib import Path
from datetime import datetime
from typing import Optional

from podcastcondensor.config import Config
from podcastcondensor.downloader import (
    download_audio,
    download_metadata,
    _find_existing_audio,
)
from podcastcondensor.transcribe import transcribe_audio
from podcastcondensor.subtitles import load_subtitles
from podcastcondensor.global_state import build_global_state
from podcastcondensor.classify_raw import classify_raw
from podcastcondensor.intervals import build_intervals, compute_stats
from podcastcondensor.audio_strategies import _get_audio_duration as get_audio_duration
from podcastcondensor.audio_strategies import create_audio_strategy
from podcastcondensor.universe_state import UniverseState
from podcastcondensor.llm.deepseek import resolve_api_key, DeepSeekClient
from podcastcondensor.segmentation.sentence_units import build_transcript_from_entries

logger = logging.getLogger(__name__)


def run_pipeline(
    url: str,
    cfg: Config,
    dry_run: bool = False,
    universe_state: Optional[UniverseState] = None,
    episode_num: Optional[int] = None,
    debug_max_intervals: int = 0,
) -> dict:
    run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    artifacts = {
        "url": url,
        "timestamp": run_timestamp,
        "config": {
            "model": cfg.deepseek_model,
            "lang": cfg.lang,
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
    from podcastcondensor.downloader import extract_video_id
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
    # Phase 1: Download audio + transcribe  (artefact: source_subtitles.srt)
    # ================================================================
    logger.info("=== Phase 1: Download + Transcribe ===")

    # ── Download audio (checkpoint: existing audio file) ─────────
    audio_path = _find_existing_audio(run_dir, local_video_id, cfg.audio_format)
    if not audio_path:
        logger.info("Downloading audio...")
        audio_path = download_audio(
            url, run_dir, local_video_id,
            audio_format=cfg.audio_format,
            audio_bitrate=cfg.audio_bitrate,
        )

    # ── Fetch metadata for title ─────────────────────────────────
    existing_gs = _load_json(_ap("global_state.json"))
    if existing_gs and existing_gs.get("episode_title"):
        title = existing_gs["episode_title"]
    else:
        try:
            meta_info = download_metadata(url)
            title = meta_info.get("title", local_video_id)
        except Exception:
            title = local_video_id

    # ── Transcribe audio to SRT (checkpoint: source_subtitles.srt) ──
    if not os.path.exists(_ap("source_subtitles.srt")):
        logger.info("Transcribing audio (this may be slow / crash-prone)...")
        transcribe_audio(
            audio_path, run_dir,
            model_size=cfg.whisper_model,
            beam_size=cfg.whisper_beam_size,
            vad_filter=cfg.whisper_vad_filter,
            condition_on_previous_text=cfg.whisper_condition_on_prev,
        )
    else:
        logger.info("Transcription checkpoint HIT — source_subtitles.srt exists, reusing")

    meta = {
        "video_id": local_video_id,
        "title": title,
        "audio_path": audio_path,
    }

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
        artifacts["errors"].append("No subtitles available after transcribing.")
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

    global_outline = global_data.get("global_outline", "")

    # ================================================================
    # Phase 3: Classify raw SRT entries  (artefact: decisions.json)
    # ================================================================
    logger.info("=== Phase 3: Classify raw SRT entries ===")
    # Load cleaned entries with original SRT indices
    # (clean_entries removes noise/echoes/dedups but preserves cue numbers)
    entries = load_subtitles(_ap("source_subtitles.srt"), reindex=False)
    logger.info("Phase 3: %d cleaned entries with original SRT indices", len(entries))

    decisions = _load_json(_ap("decisions.json"))
    if decisions is not None and isinstance(decisions, dict) and "ranges" in decisions:
        logger.info("Checkpoint HIT — loaded decisions.json")
    else:
        universe_ctx = ""
        if universe_state is not None:
            universe_ctx = universe_state.get_context()

        result = classify_raw(
            srt_path=_ap("source_subtitles.srt"),
            client=ds,
            global_outline=global_outline,
            universe_state_context=universe_ctx,
            model=cfg.deepseek_model,
            timeout=cfg.deepseek_timeout or 600,
            prompt_path=cfg.classify_raw_prompt_path,
        )

        kept_ranges = result.get("kept_ranges", [])
        dropped_ranges = result.get("dropped_ranges", [])

        # Build text lookup by entry index (raw SRT index)
        entry_texts = {e["index"]: e["text"] for e in entries}

        def _annotate_ranges(ranges, default_label):
            out = []
            for r in ranges:
                s, e = r.get("start", 0), r.get("end", 0)
                texts = []
                for ei in range(s, e + 1):
                    t = entry_texts.get(ei, "")
                    if t:
                        texts.append(t)
                out.append({
                    "start": s,
                    "end": e,
                    "label": default_label,
                    "reason": r.get("reason", ""),
                    "text": " ".join(texts),
                })
            return out

        annotated = (
            _annotate_ranges(kept_ranges, "keep") +
            _annotate_ranges(dropped_ranges, "drop")
        )
        annotated.sort(key=lambda r: r["start"])

        # Build per-entry decisions by raw entry index (matches LLM's cue numbers)
        reasons = {}
        for r in kept_ranges:
            for ei in range(r.get("start", 0), r.get("end", 0) + 1):
                reasons[ei] = ("keep", r.get("reason", ""))
        for r in dropped_ranges:
            for ei in range(r.get("start", 0), r.get("end", 0) + 1):
                existing = reasons.get(ei)
                if existing is None:
                    reasons[ei] = ("drop", r.get("reason", ""))

        # Only emit decisions for entries that exist in the raw SRT
        per_entry = []
        for e in entries:
            label, reason = reasons.get(e["index"], ("drop", ""))
            per_entry.append({"id": str(e["index"]), "label": label, "reason": reason})
        per_entry.sort(key=lambda d: int(d["id"]))

        decisions = {
            "version": "0.4.0",
            "episode": episode_num or 0,
            "universe_state_used": bool(universe_ctx),
            "total_entries": len(entries),
            "kept_count": sum(1 for d in per_entry if d["label"] == "keep"),
            "ranges": annotated,
        }

        _write_json(_ap("decisions.json"), decisions)

    # Expand ranges into per-entry decisions for the audio cutter (checkpoint path)
    if decisions.get("decisions") is None:
        sd = decisions.get("ranges", [])
        reasons = {}
        for r in sd:
            for ei in range(r.get("start", 0), r.get("end", 0) + 1):
                reasons[ei] = (r.get("label", "drop"), r.get("reason", ""))

        per_entry = []
        for e in entries:
            label, reason = reasons.get(e["index"], ("drop", ""))
            per_entry.append({"id": str(e["index"]), "label": label, "reason": reason})
        per_entry.sort(key=lambda d: int(d["id"]))
        decisions["decisions"] = per_entry
        # Fix stale metadata if checkpoint was from a buggy run
        decisions["total_entries"] = len(per_entry)
        decisions["kept_count"] = sum(1 for d in per_entry if d["label"] == "keep")

    artifacts["phases"]["classify"] = {
        "total_entries": decisions.get("total_entries", 0),
        "kept_count": decisions.get("kept_count", 0),
        "universe_state_used": decisions.get("universe_state_used", False),
    }

    # ================================================================
    # Phase 4: Audio cutting  (artefact: condensed_*.mp3)
    # ================================================================
    logger.info("=== Phase 4: Audio cutting ===")
    audio_path = meta["audio_path"]
    if not os.path.exists(audio_path):
        artifacts["errors"].append(f"Audio not found: {audio_path}")
        return artifacts

    audio_duration = get_audio_duration(audio_path)

    per_entry = decisions.get("decisions", [])

    # ── Print classifier stats before committing to audio cut ──────────
    kept_ids = {d["id"] for d in per_entry if d["label"] == "keep"}
    kept_dur = sum(e["end"] - e["start"] for e in entries if str(e["index"]) in kept_ids)
    total_dur = sum(e["end"] - e["start"] for e in entries) if entries else 1
    pct = 100 * kept_dur / total_dur if total_dur > 0 else 0
    kept_entries_frac = 100 * len(kept_ids) / len(entries) if entries else 0
    logger.info(
        "CLASSIFICATION STATS: %d/%d entries kept (%d%%) — "
        "estimate %.0fm / %.0fm (%.0f%% of duration)",
        len(kept_ids), len(entries), int(kept_entries_frac),
        kept_dur / 60, total_dur / 60, pct,
    )

    # Convert entries to segment-like dicts for intervals builder
    segments = [
        {"segment_id": str(e["index"]), "start": e["start"], "end": e["end"], "text": e["text"]}
        for e in entries
    ]
    intervals = build_intervals(
        segments=segments, decisions=per_entry,
        merge_gap=cfg.output_merge_gap,
        pad_before=cfg.pad_before,
        pad_after=cfg.pad_after,
        audio_duration=audio_duration,
        cluster_gap=cfg.cluster_gap,
    )
    artifacts["phases"]["intervals"] = {"interval_count": len(intervals)}

    # ── Snapshot full intervals for stats (before any debug cap) ────────
    full_intervals = list(intervals)

    # ── Compute + persist + print stats right away ──────────────────────
    stats = compute_stats(segments, per_entry, full_intervals)
    _write_json(_ap("stats.json"), stats)
    artifacts["output_dir"] = run_dir
    _print_results(stats, artifacts)

    # ── Early exit if --skip-audio ──────────────────────────────────────
    if cfg.skip_audio:
        logger.info("Skip-audio set — stopping after stats (no audio cutting)")
        return artifacts

    # ── DEBUG: cap intervals for quick test listen ──────────────────────
    if debug_max_intervals > 0 and intervals:
        original_count = len(intervals)
        intervals = intervals[:debug_max_intervals]
        logger.warning(
            "🔎 DEBUG CAP: intervals truncated from %d to first %d — "
            "output is NOT a full condensed episode!",
            original_count, debug_max_intervals,
        )

    condensed_path = _ap(f"condensed_{meta['video_id']}.{cfg.audio_format}")
    if debug_max_intervals > 0:
        # Mark debug output so it's not confused with a real run
        name, ext = os.path.splitext(condensed_path)
        condensed_path = f"{name}_DEBUG_{debug_max_intervals}intervals{ext}"
    if intervals:
        strategy = create_audio_strategy(
            cfg.audio_strategy,
            batch_size=cfg.audio_safe_batch_size,
        )
        strategy.cut(
            audio_path=audio_path, intervals=intervals,
            output_path=condensed_path, format_spec=cfg.audio_format,
            sample_rate=cfg.audio_sample_rate, bitrate=cfg.audio_bitrate,
            speed=cfg.audio_speed,
        )
        artifacts["phases"]["audio"] = {"condensed_path": condensed_path}
    else:
        artifacts["phases"]["audio"] = {"condensed_path": None}

    return artifacts


def _print_results(stats: dict, artifacts: dict):
    speed = artifacts["config"].get("speed", 1.0)
    print("")
    print("=" * 50)
    print("PODCAST CONDENSOR — RESULTS")
    print("=" * 50)
    print(f"  SRT entries:       {stats['total_segments']}")
    print(f"  Kept:              {stats['keep_count']}")
    print(f"  Dropped:           {stats['drop_count']}")
    print(f"  Original duration: {_fmt_duration(stats['original_duration_sec'])}")
    cd = stats["condensed_duration_sec"]
    print(f"  Condensed duration:{_fmt_duration(cd)}")
    if speed != 1.0:
        print(f"  At {speed}x:         {_fmt_duration(cd / speed)}")
    print(f"  Compression ratio: {stats['compression_ratio']:.1%}")
    print(f"  Universe considered:{'yes' if artifacts.get('phases',{}).get('classify',{}).get('universe_state_used') else 'no'}")
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
