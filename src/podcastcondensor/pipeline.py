"""Pipeline — orchestrate the full condensing workflow.

Three-phase architecture:
  Phase A: Global episode map (blocks → summaries → outline)
  Phase B: Segment classification with global context
  Phase C: Global cleanup / dedup
  Phase D: Universe state knowledge extraction (optional)

Strategy injection:
  - Classification uses ``ClassifierStrategy`` (Ollama or DeepSeek).
  - Knowledge extraction uses ``KnowledgeExtractionStrategy``.
  - Audio cutting uses ``AudioCuttingStrategy``.
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
from podcastcondensor.rechunker import resegment, refine_segments, Segment
from podcastcondensor.global_map import build_global_map
from podcastcondensor.ollama_client import check_ollama, find_best_model
from podcastcondensor.classifier import (
    classify_segments as _ollama_classify,
    global_cleanup,
    resolve_maybe as _ollama_resolve_maybe,
    apply_continuity_bias,
    detect_tail_block,
)
from podcastcondensor.intervals import build_intervals, compute_stats, check_quality_guardrails
from podcastcondensor.audio import build_condensed_audio as _ollama_audio
from podcastcondensor.audio import get_audio_duration
from podcastcondensor.universe_state import UniverseState

# Strategy imports
from podcastcondensor.strategies import (
    create_classifier,
    create_knowledge_extractor,
    ClassifierStrategy,
    KnowledgeExtractionStrategy,
)
from podcastcondensor.strategies.base import ClassificationFailedError
from podcastcondensor.strategies.classification import OllamaClassifierStrategy
from podcastcondensor.strategies.knowledge import OllamaKnowledgeExtractionStrategy
from podcastcondensor.audio_strategies import (
    create_audio_strategy,
    AudioCuttingStrategy,
)

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

    Provider selection is driven by ``cfg.classification_provider``,
    ``cfg.knowledge_provider``, and ``cfg.audio_strategy``.

    When ``cfg.decisions_only`` is True the pipeline stops after Phase C
    (final decisions) and interval building, skipping knowledge
    extraction and audio cutting.  Useful for classifier evaluation
    runs where only keep/drop stats are needed.
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
            # Provider selection
            "classification_provider": cfg.classification_provider,
            "classification_model": cfg.classification_model,
            "knowledge_provider": cfg.knowledge_provider,
            "knowledge_model": cfg.knowledge_model,
            "audio_strategy": cfg.audio_strategy,
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
    # Resolve video ID early (needed for run_dir, no network needed)
    # ----------------------------------------------------------------
    from podcastcondensor.downloader import extract_video_id, _find_existing_audio, _find_existing_subtitle
    local_video_id = extract_video_id(url)
    if not local_video_id:
        # Unrecognisable URL pattern — must hit API
        from podcastcondensor.downloader import download_metadata
        local_video_id = download_metadata(url)["id"]

    run_dir = os.path.join(cfg.output_root, local_video_id)
    Path(run_dir).mkdir(parents=True, exist_ok=True)

    # ----------------------------------------------------------------
    # Phase 1: Download — skip if we already have processed segments
    # ----------------------------------------------------------------
    if _exists("segments.json"):
        logger.info("=== Phase 1: Download (skipped — segments already exist) ===")
        download_dir = "/tmp/podcastcondensor/downloads"
        audio_path = _find_existing_audio(download_dir, local_video_id, cfg.audio_format)
        if not audio_path:
            logger.warning("Existing audio not found — re-downloading")
            from podcastcondensor.downloader import download_audio
            audio_path = download_audio(url, download_dir, local_video_id,
                                        audio_format=cfg.audio_format,
                                        audio_bitrate=cfg.audio_bitrate)
        meta = {
            "video_id": local_video_id,
            "title": local_video_id,
            "audio_path": audio_path,
            "subtitle_path": _find_existing_subtitle(download_dir, local_video_id) or "",
        }
    else:
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

    if dry_run:
        logger.info("Dry run: stopping after download")
        return artifacts

    # ----------------------------------------------------------------
    # Ollama check (still required for Phase A and segmentation)
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
        srt_source = meta.get("subtitle_path") or ""
        if not srt_source or not os.path.exists(srt_source):
            srt_source = _ap("source_subtitles.srt")
        if not srt_source or not os.path.exists(srt_source):
            artifacts["errors"].append("No subtitles available.")
            return artifacts

        if srt_source != _ap("source_subtitles.srt"):
            shutil.copy2(srt_source, _ap("source_subtitles.srt"))

        cleaned = load_subtitles(srt_source)
        seg_objects = resegment(
            entries=cleaned,
            gap_threshold=cfg.segment_gap_threshold,
            gap_sentence_threshold=cfg.segment_gap_sentence_threshold,
            max_words=cfg.segment_max_words,
            min_words=cfg.segment_min_words,
            sentence_overflow_words=cfg.sentence_overflow_words,
        )

        if cfg.refine_segments and len(seg_objects) > 1:
            seg_objects = refine_segments(
                rough_segments=seg_objects,
                entries=cleaned,
                model=cfg.default_model,
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

    seg_to_block = (
        global_data.get("segment_to_block")
        or global_data.get("chunk_to_block")
        or {}
    )
    for s in segments:
        s["block_id"] = seg_to_block.get(s["segment_id"], 0)

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
    # Phase B: Classify segments — strategy pattern
    # ----------------------------------------------------------------
    logger.info("=== Phase B: Classify segments ===")
    logger.info(
        "Classification provider: %s (model: %s)",
        cfg.classification_provider, cfg.classification_model,
    )

    decisions = None

    if _exists("decisions.json"):
        with open(_ap("decisions.json")) as f:
            decisions = json.load(f)
        # Validate decisions cache: check for degraded content
        if _decisions_are_degraded(decisions):
            logger.warning("Cached decisions are degraded (from prior failure) — re-classifying")
            decisions = None
            os.remove(_ap("decisions.json"))
        else:
            logger.info("Reusing %d decisions from disk", len(decisions))

    if decisions is None:
        universe_state_context = ""

        # Build the appropriate classifier strategy with fallback support
        classifier, classifier_label = _build_classifier_with_fallback(
            cfg, classify_model, artifacts,
        )

        if universe_state is not None:
            universe_state_context = universe_state.get_context(
                max_items=8, max_chars=3000,
                exclude_episode_gte=episode_num,
            )
            logger.info(
                "Universe state context: %d chars (excluding episodes >= %s)",
                len(universe_state_context), episode_num,
            )

        decisions = _run_classification_with_fallback(
            classifier=classifier,
            classifier_label=classifier_label,
            segments=classify_segs,
            global_outline=global_data["global_outline"],
            block_summaries=global_data["block_summaries"],
            cfg=cfg,
            artifacts=artifacts,
            universe_state_context=universe_state_context,
            ap_func=_ap,
            exists_func=_exists,
        )

        if decisions is None:
            artifacts["errors"].append("All classification providers failed — aborting")
            return artifacts

        if not _exists("decisions.json"):
            _write_json(_ap("decisions.json"), decisions)

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
    # Phase C: Finalize decisions (consolidated)
    # ----------------------------------------------------------------
    logger.info("=== Phase C: Finalize decisions ===")
    if _exists("decisions_final.json"):
        with open(_ap("decisions_final.json")) as f:
            decisions = json.load(f)
        logger.info("Reusing %d final decisions from disk", len(decisions))
    else:
        resolve_classifier = _build_resolve_classifier(cfg, classify_model, artifacts)
        decisions = finalize_decisions(
            segments=segments,
            decisions=decisions,
            cfg=cfg,
            classifier=resolve_classifier,
        )
        _write_json(_ap("decisions_final.json"), decisions)

    artifacts["phases"]["finalize_decisions"] = {"decision_count": len(decisions)}

    if cfg.decisions_only:
        logger.info("decisions_only=True — skipping Phase D (knowledge extraction)")

    # ----------------------------------------------------------------
    # Phase D: Universe State knowledge extraction — strategy pattern
    # ----------------------------------------------------------------
    if not cfg.decisions_only and universe_state is not None and not dry_run and cfg.extract_concepts_prompt_path:
        logger.info("=== Phase D: Extract knowledge for universe state ===")

        block_data = global_data.get("block_summaries", [])
        outline_text = global_data.get("global_outline", "")
        ep_title = meta.get("title", "")
        ep_number = episode_num

        if _exists("state_knowledge.json"):
            with open(_ap("state_knowledge.json")) as f:
                cached = json.load(f)

            # Check cache fingerprint
            if _cache_fingerprint_matches(cached, cfg, outline_text):
                knowledge = cached
                logger.info("Cache HIT: reusing previously extracted knowledge")
            else:
                logger.info("Cache MISS: fingerprint changed, re-extracting")
                knowledge = _extract_knowledge(
                    cfg, block_data, outline_text, ep_title, ep_number, artifacts,
                )
                _write_json(_ap("state_knowledge.json"), knowledge)
        else:
            knowledge = _extract_knowledge(
                cfg, block_data, outline_text, ep_title, ep_number, artifacts,
            )
            _write_json(_ap("state_knowledge.json"), knowledge)

        if knowledge:
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
            cluster_gap=cfg.cluster_gap,
        )
        _write_json(_ap("keep_intervals.json"), intervals)

        stats = compute_stats(segments, decisions, intervals)
        _write_json(_ap("stats.json"), stats)

        # Quality guardrails
        guardrail_warnings = check_quality_guardrails(stats, min_keep_ratio=cfg.min_keep_ratio)
        if guardrail_warnings:
            logger.warning("=== Quality guardrail warnings ===")
            for w in guardrail_warnings:
                logger.warning("  ⚠  %s", w)
            artifacts["phases"]["guardrails"] = {"warnings": guardrail_warnings}
        else:
            artifacts["phases"]["guardrails"] = {"warnings": []}

    artifacts["phases"]["intervals"] = {"interval_count": len(intervals)}

    if cfg.decisions_only:
        logger.info("decisions_only=True — skipping audio cutting")
    else:
        # ----------------------------------------------------------------
        # Audio cutting — strategy pattern
        # ----------------------------------------------------------------
        logger.info("=== Audio cutting ===")
        logger.info("Audio strategy: %s", cfg.audio_strategy)

        condensed_path = _ap(f"condensed_{meta['video_id']}.{cfg.audio_format}")
        if os.path.exists(condensed_path):
            logger.info("Reusing condensed audio from disk")
            artifacts["phases"]["audio"] = {"condensed_path": condensed_path}
        elif intervals:
            strategy_kwargs = {}
            if cfg.audio_strategy == "parallel_copy":
                strategy_kwargs["max_workers"] = cfg.audio_parallel_workers
            elif cfg.audio_strategy == "safe_batched":
                strategy_kwargs["batch_size"] = cfg.audio_safe_batch_size
            strategy = create_audio_strategy(cfg.audio_strategy, **strategy_kwargs)
            strategy.cut(
                audio_path=meta["audio_path"],
                intervals=intervals,
                output_path=condensed_path,
                format_spec=cfg.audio_format,
                sample_rate=cfg.audio_sample_rate,
                bitrate=cfg.audio_bitrate,
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


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def finalize_decisions(
    segments: list,
    decisions: list,
    cfg: Config,
    classifier: ClassifierStrategy,
) -> list:
    """Consolidated Phase C: run all decision post-processing in one pass.

    Order:
      1. Deterministic cleanup  (dedup + opening protection)
      2. LLM resolve_maybe       (only if *maybe* labels exist)
      3. Continuity bias         (bridge + context + short neighbour)
      4. Tail detection — force-drop off-topic trailing content **LAST**

    No intermediate files are written between sub-steps.  The caller is
    responsible for persisting the result to ``decisions_final.json``.
    """
    # 1. Deterministic cleanup
    decisions = global_cleanup(segments, decisions)

    # 2. Resolve maybes via LLM
    if cfg.resolve_maybe:
        maybe_ids = [d["id"] for d in decisions if d.get("label") == "maybe"]
        maybe_segs = [s for s in segments if s["segment_id"] in maybe_ids]
        if maybe_segs:
            logger.info("Resolving %d maybe segments...", len(maybe_segs))
            decisions = classifier.resolve_maybe(
                maybe_segments=maybe_segs,
                all_segments=segments,
                all_decisions=decisions,
            )

    # 3. Continuity bias (disabled for experiment matrix runs)
    if cfg.enable_continuity_bias:
        logger.info("Applying continuity bias...")
        decisions = apply_continuity_bias(
            segments=segments,
            decisions=decisions,
            bridge_gap_sec=cfg.bridge_gap_sec,
        )
    else:
        logger.info("Continuity bias disabled — classifier decisions unchanged")

    # 4. Tail detection — MUST be the final mutating step
    if cfg.enable_tail_detection:
        tail_ids = detect_tail_block(
            segments=segments,
            decisions=decisions,
            tail_fraction=cfg.tail_fraction,
            min_keep_fraction=cfg.tail_min_keep_fraction,
        )
        if tail_ids:
            logger.info("Tail detection: force-dropping %d segments", len(tail_ids))
            for d in decisions:
                if d["id"] in tail_ids:
                    d["label"] = "force_drop"
                    d["force_drop"] = True

    return decisions


def _build_resolve_classifier(
    cfg: Config, classify_model: str, artifacts: dict,
) -> ClassifierStrategy:
    """Build a classifier strategy for maybe-resolution.

    For DeepSeek primary with Ollama fallback, resolve_maybe uses
    the same provider as classification.
    """
    from podcastcondensor.llm.deepseek import resolve_api_key

    if cfg.classification_provider == "deepseek":
        api_key = resolve_api_key()
        if not api_key and cfg.classification_fallback_provider == "ollama":
            return OllamaClassifierStrategy(
                model=classify_model,
                prompt_path=cfg.classify_global_prompt_path,
                host=cfg.ollama_host,
                ollama_timeout=cfg.ollama_timeout,
                resolve_maybe_prompt_path=cfg.resolve_maybe_prompt_path,
            )
        return create_classifier(
            provider="deepseek",
            prompt_path=cfg.classify_global_prompt_path,
            resolve_maybe_prompt_path=cfg.resolve_maybe_prompt_path,
            model=cfg.classification_model,
            timeout=cfg.ollama_timeout,
            deepseek_base_url=cfg.classification_base_url or None,
            deepseek_api_key=api_key or "",
        )

    return OllamaClassifierStrategy(
        model=classify_model,
        prompt_path=cfg.classify_global_prompt_path,
        host=cfg.ollama_host,
        ollama_timeout=cfg.ollama_timeout,
        resolve_maybe_prompt_path=cfg.resolve_maybe_prompt_path,
    )


def _extract_knowledge(
    cfg: Config,
    block_data: list,
    outline_text: str,
    ep_title: str,
    ep_number: Optional[int],
    artifacts: dict,
) -> dict:
    """Extract knowledge using the configured provider strategy."""
    from podcastcondensor.llm.deepseek import resolve_api_key, ENV_API_KEY_VARS

    logger.info("Knowledge provider: %s (model: %s)", cfg.knowledge_provider, cfg.knowledge_model)

    if cfg.knowledge_provider == "deepseek":
        api_key = resolve_api_key()
        if not api_key:
            vars_help = " or ".join(f"${v}" for v in ENV_API_KEY_VARS)
            logger.warning(
                "%s not set for knowledge extraction — falling back to Ollama",
                vars_help,
            )
            extractor = OllamaKnowledgeExtractionStrategy(
                model=cfg.default_model,
                prompt_path=cfg.extract_concepts_prompt_path,
                host=cfg.ollama_host,
                timeout=cfg.ollama_timeout,
            )
        else:
            extractor = create_knowledge_extractor(
                provider="deepseek",
                prompt_path=cfg.extract_concepts_prompt_path,
                model=cfg.knowledge_model,
                timeout=cfg.ollama_timeout,
                deepseek_base_url=cfg.knowledge_base_url or None,
                deepseek_api_key=api_key,
            )
    else:
        extractor = OllamaKnowledgeExtractionStrategy(
            model=cfg.default_model,
            prompt_path=cfg.extract_concepts_prompt_path,
            host=cfg.ollama_host,
            timeout=cfg.ollama_timeout,
        )

    knowledge = extractor.extract(
        block_summaries=block_data,
        global_outline=outline_text,
        episode_title=ep_title,
        episode_number=ep_number,
    )

    # Attach cache fingerprint
    if knowledge:
        knowledge["_fingerprint"] = _compute_fingerprint(cfg, outline_text)

    artifacts["config"]["knowledge_extractor_used"] = extractor.name()

    return knowledge


def _compute_fingerprint(cfg: Config, outline_text: str) -> dict:
    """Compute a cache fingerprint for the knowledge extraction output.

    When any of these values changes, the cache is invalidated.
    """
    import hashlib

    prompt_hash = hashlib.sha256(
        open(cfg.extract_concepts_prompt_path, "rb").read()
    ).hexdigest()[:16]

    return {
        "provider": cfg.knowledge_provider,
        "model": cfg.knowledge_model,
        "prompt_hash": prompt_hash,
        "schema_version": cfg.knowledge_cache_schema_version,
        "outline_hash": hashlib.sha256(
            outline_text.encode("utf-8")
        ).hexdigest()[:16],
    }


def _cache_fingerprint_matches(cached: dict, cfg: Config, outline_text: str) -> bool:
    """Check whether the cached knowledge's fingerprint matches current config."""
    stored = cached.get("_fingerprint")
    if not stored:
        return False  # no fingerprint — legacy cache, treat as miss

    current = _compute_fingerprint(cfg, outline_text)

    for key in ("provider", "model", "prompt_hash", "schema_version"):
        if stored.get(key) != current.get(key):
            logger.info(
                "Fingerprint mismatch on '%s': stored=%s current=%s",
                key, stored.get(key), current.get(key),
            )
            return False

    return True


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
    lines.append(f"- Classifier:      {artifacts['config'].get('classifier_used', '?')}")
    lines.append(f"- Knowledge ext:   {artifacts['config'].get('knowledge_extractor_used', '?')}")
    lines.append(f"- Audio strategy:  {artifacts['config'].get('audio_strategy', '?')}")
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


# ---------------------------------------------------------------------------
# Classification fallback helpers
# ---------------------------------------------------------------------------


def _decisions_are_degraded(decisions: list) -> bool:
    """Check whether a cached decisions list is degraded (from prior failure).

    A degraded result is one where ALL segments are ``"maybe"`` with the
    reason ``"cloud-classification-failed"`` — this indicates the previous
    run's classifier failed but the bad output was persisted.
    """
    if not decisions:
        return False
    n_total = len(decisions)
    n_cloud_fail = sum(
        1 for d in decisions
        if d.get("label") == "maybe"
        and "cloud-classification-failed" in d.get("reason", "")
    )
    # If >50% of decisions are cloud-failure maybes, the cache is degraded
    return n_cloud_fail / n_total > 0.5


def _build_classifier_with_fallback(
    cfg: Config,
    classify_model: str,
    artifacts: dict,
) -> tuple:
    """Build a classifier strategy, with an ordered fallback chain.

    Returns ``(classifier, label_string)`` where *classifier* is the
    primary strategy (may raise ``ClassificationFailedError`` at call
    time).  The pipeline is responsible for catching and falling back.
    """
    from podcastcondensor.llm.deepseek import resolve_api_key, ENV_API_KEY_VARS

    if cfg.classification_provider == "deepseek":
        api_key = resolve_api_key()
        if not api_key:
            vars_help = " or ".join(f"${v}" for v in ENV_API_KEY_VARS)
            err_msg = f"DeepSeek selected for classification but {vars_help} is not set"
            if cfg.classification_fallback_provider == "ollama":
                logger.warning("%s — falling back to Ollama", err_msg)
                return (
                    OllamaClassifierStrategy(
                        model=classify_model,
                        prompt_path=cfg.classify_global_prompt_path,
                        host=cfg.ollama_host,
                        ollama_timeout=cfg.ollama_timeout,
                        resolve_maybe_prompt_path=cfg.resolve_maybe_prompt_path,
                    ),
                    "ollama(fallback-missing-key)",
                )
            else:
                raise ValueError(err_msg)

        return (
            create_classifier(
                provider="deepseek",
                prompt_path=cfg.classify_global_prompt_path,
                resolve_maybe_prompt_path=cfg.resolve_maybe_prompt_path,
                model=cfg.classification_model,
                timeout=cfg.ollama_timeout,
                max_segments_per_batch=cfg.max_segments_per_batch,
                deepseek_base_url=cfg.classification_base_url or None,
                deepseek_api_key=api_key,
            ),
            f"deepseek({cfg.classification_model})",
        )

    # Default: Ollama
    return (
        OllamaClassifierStrategy(
            model=classify_model,
            prompt_path=cfg.classify_global_prompt_path,
            host=cfg.ollama_host,
            ollama_timeout=cfg.ollama_timeout,
            resolve_maybe_prompt_path=cfg.resolve_maybe_prompt_path,
        ),
        f"ollama({classify_model})",
    )


def _run_classification_with_fallback(
    classifier: ClassifierStrategy,
    classifier_label: str,
    segments: list,
    global_outline: str,
    block_summaries: list,
    cfg: Config,
    artifacts: dict,
    universe_state_context: str,
    ap_func,
    exists_func,
) -> list:
    """Run classification with an ordered fallback chain.

    Tries:
      1. Primary classifier (e.g. DeepSeek).
      2. If ``ClassificationFailedError`` and Ollama fallback configured, retry
         with ``OllamaClassifierStrategy``.
      3. If that also fails, return ``None`` (caller must abort).

    Returns a decisions list, or ``None`` if ALL classifiers fail.
    """
    classifiers_to_try = [
        (classifier_label, classifier, False),
    ]

    # If primary is DeepSeek and fallback is Ollama, add the fallback
    if cfg.classification_provider == "deepseek" and cfg.classification_fallback_provider == "ollama":
        from podcastcondensor.strategies.classification import OllamaClassifierStrategy
        fallback_classifier = OllamaClassifierStrategy(
            model=cfg.classify_model or cfg.default_model,
            prompt_path=cfg.classify_global_prompt_path,
            host=cfg.ollama_host,
            ollama_timeout=cfg.ollama_timeout,
            resolve_maybe_prompt_path=cfg.resolve_maybe_prompt_path,
        )
        classifiers_to_try.append(("ollama(fallback)", fallback_classifier, True))

    last_error = None
    for label, clf, is_fallback in classifiers_to_try:
        try:
            logger.info("Attempting classification with %s", label)
            result = clf.classify_segments(
                segments=segments,
                global_outline=global_outline,
                block_summaries=block_summaries,
                max_segments_per_batch=cfg.max_segments_per_batch,
                output_path=ap_func("decisions.json") if not is_fallback else None,
                universe_state_context=universe_state_context,
            )
            artifacts["config"]["classifier_used"] = label
            return result
        except ClassificationFailedError as e:
            logger.warning("Classifier %s failed: %s", label, e)
            last_error = e
            continue
        except Exception as e:
            logger.warning("Classifier %s raised unexpected error: %s", label, e)
            last_error = e
            continue

    logger.error(
        "All classification providers failed.  Last error: %s", last_error,
    )
    return None
