"""CLI entry point for podcastcondensor."""

import argparse
import logging
import os
import sys

from podcastcondensor.config import Config
from podcastcondensor.ollama_client import check_ollama, list_models
from podcastcondensor.pipeline import run_pipeline
from podcastcondensor.playlist_pipeline import build_universe_state, process_with_universe_state
from podcastcondensor.universe_state import UniverseState


def setup_logging(verbose: bool = False):
    level = logging.DEBUG if verbose else logging.INFO
    fmt = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    logging.basicConfig(
        level=level,
        format=fmt,
        datefmt="%H:%M:%S",
        stream=sys.stderr,
    )


def cmd_run(args):
    cfg = Config(
        default_model=args.model,
        classify_model=args.classify_model,
        lang=args.lang,
        output_root=os.path.abspath(args.output_dir) if args.output_dir else "",
        output_merge_gap=args.merge_gap,
        pad_before=args.pad_before,
        pad_after=args.pad_after,
        max_segments_per_batch=args.batch_size,
        resolve_maybe=args.resolve_maybe,
        keep_temp=args.keep_temp,
        prefer_auto_subs=args.prefer_auto_subs,
        ollama_host=args.ollama_host,
        ollama_timeout=args.ollama_timeout,
        block_size_words=args.block_size,
        max_blocks=args.max_blocks,
        audio_speed=args.speed,
        segment_gap_threshold=args.segment_gap,
        segment_max_words=args.segment_max_words,
        segment_min_words=args.segment_min_words,
    )

    result = run_pipeline(
        url=args.url,
        cfg=cfg,
        dry_run=args.dry_run,
    )

    if result.get("errors"):
        print("\n⚠️  Pipeline completed with errors:", file=sys.stderr)
        for e in result["errors"]:
            print(f"  - {e}", file=sys.stderr)
        sys.exit(1)

    audio = result.get("phases", {}).get("audio", {}).get("condensed_path")
    if audio:
        print(f"\n✅ Condensed audio: {audio}")
    stats_path = os.path.join(
        os.path.dirname(os.path.dirname(audio)) if audio else "output",
        "review.md",
    )
    print(f"📋 Review: {stats_path}")
    print(f"🎯 Done.")


def cmd_status(args):
    host = args.ollama_host
    ok = check_ollama(host)
    if ok:
        print(f"✅ Ollama is running at {host}")
        models = list_models(host)
        if models:
            print(f"📦 Available models ({len(models)}):")
            for m in models:
                print(f"   - {m}")
        else:
            print("⚠️  No models found. Pull one with: ollama pull qwen3:8b")
    else:
        print(f"❌ Ollama is NOT running at {host}")
        print("   Start it with: ollama serve &")
        print("   Or install: curl -fsSL https://ollama.com/install.sh | sh")
        sys.exit(1)


def cmd_build_universe(args):
    """Build universe state from initial episodes (1-20)."""
    cfg = Config(
        default_model=args.model,
        lang=args.lang,
        output_root=os.path.abspath(args.output_dir) if args.output_dir else "",
        prefer_auto_subs=args.prefer_auto_subs,
        ollama_host=args.ollama_host,
        ollama_timeout=args.ollama_timeout,
        block_size_words=args.block_size,
        segment_gap_threshold=args.segment_gap,
        segment_max_words=args.segment_max_words,
        segment_min_words=args.segment_min_words,
    )

    state_path = os.path.abspath(args.state_file) if args.state_file else ""

    state = build_universe_state(
        playlist_url=args.playlist_url,
        cfg=cfg,
        start_episode=args.start,
        end_episode=args.end,
        state_path=state_path or None,
        dry_run=args.dry_run,
    )

    print("\n" + "=" * 50)
    print("UNIVERSE STATE BUILD COMPLETE")
    print("=" * 50)
    print(f"  State file:          {state.path}")
    print(f"  Episodes processed:  {state.data['metadata'].get('last_built_episode', 0)}")
    print(f"  Core concepts:       {len(state.data.get('concepts', []))}")
    print(f"  Entities:            {len(state.data.get('entities', []))}")
    print(f"  Claims:              {len(state.data.get('claims', []))}")
    print(f"  Glossary terms:      {len(state.data.get('glossary', []))}")
    print(f"  Scriptural links:    {len(state.data.get('scriptural_links', []))}")
    print(f"  Historical links:    {len(state.data.get('historical_links', []))}")
    print(f"  Canonical reps:      {len(state.data.get('canonical_repetitions', []))}")
    print(f"  Open threads:        {len(state.data.get('open_threads', []))}")
    print("")


def cmd_process_playlist(args):
    """Process playlist episodes using a pre-built universe state."""
    cfg = Config(
        default_model=args.model,
        classify_model=getattr(args, 'classify_model', 'qwen2.5:7b'),
        lang=args.lang,
        output_root=os.path.abspath(args.output_dir) if args.output_dir else "",
        output_merge_gap=args.merge_gap,
        pad_before=args.pad_before,
        pad_after=args.pad_after,
        max_segments_per_batch=args.batch_size,
        resolve_maybe=args.resolve_maybe,
        keep_temp=args.keep_temp,
        prefer_auto_subs=args.prefer_auto_subs,
        ollama_host=args.ollama_host,
        ollama_timeout=args.ollama_timeout,
        block_size_words=args.block_size,
        max_blocks=args.max_blocks,
        audio_speed=args.speed,
        segment_gap_threshold=args.segment_gap,
        segment_max_words=args.segment_max_words,
        segment_min_words=args.segment_min_words,
    )

    # Load universe state
    state_path = os.path.abspath(args.state_file)
    if not os.path.exists(state_path):
        print(f"\n❌ Universe state not found: {state_path}")
        print("   Build it first with: podcastcondensor build-universe <playlist_url>")
        sys.exit(1)
    state = UniverseState(state_path)
    print(f"\n📚 Loaded universe state: {state.data['metadata'].get('last_built_episode', 0)} episodes")

    results = process_with_universe_state(
        playlist_url=args.playlist_url,
        cfg=cfg,
        state=state,
        start_episode=args.start,
        end_episode=args.end,
        dry_run=args.dry_run,
    )

    print("\n" + "=" * 50)
    print("PLAYLIST PROCESSING COMPLETE")
    print("=" * 50)
    successful = sum(1 for r in results if r.get("success"))
    print(f"  Episodes: {successful}/{len(results)} successful")
    print(f"  State file: {state_path}")
    print(f"  Concepts now: {len(state.data.get('concepts', []))}")
    for r in results:
        status = "✅" if r.get("success") else "❌"
        audio = r.get("condensed_audio", "")
        if audio:
            print(f"  {status} Ep {r['episode']}: {os.path.basename(audio)}")
        else:
            errs = r.get("errors", [])
            err_str = f" — {errs[0]}" if errs else ""
            print(f"  {status} Ep {r['episode']}{err_str}")
    print("")


def main():
    parser = argparse.ArgumentParser(
        description="podcastcondensor — local-first podcast condensing pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python -m podcastcondensor run URL\n"
            "  python -m podcastcondensor run URL --model qwen2.5:7b\n"
            "  python -m podcastcondensor status\n"
        ),
    )
    parser.add_argument("--verbose", "-v", action="store_true", help="Debug logging")
    parser.add_argument(
        "--ollama-host",
        default="http://localhost:11434",
        help="Ollama API host (default: http://localhost:11434)",
    )

    sub = parser.add_subparsers(dest="command", required=True)

    # run
    run_p = sub.add_parser("run", help="Run condensation pipeline")
    run_p.add_argument("url", help="YouTube URL")
    run_p.add_argument("--model", default="qwen2.5:3b",
                        help="Ollama model for extraction/summarization (default: qwen2.5:3b)")
    run_p.add_argument("--classify-model", default="qwen2.5:7b",
                        help="Ollama model for classification (default: qwen2.5:7b)")
    run_p.add_argument("--lang", default="en")
    run_p.add_argument("--output-dir", default="")

    # Segmentation
    run_p.add_argument("--segment-gap", type=float, default=0.5,
                        help="Min silence (sec) to split segments (default: 0.5)")
    run_p.add_argument("--segment-max-words", type=int, default=400,
                        help="Max words per segment (default: 400)")
    run_p.add_argument("--segment-min-words", type=int, default=20,
                        help="Min words before merge orphan (default: 20)")

    # Classification
    run_p.add_argument("--batch-size", type=int, default=5,
                        help="Segments per classification batch (default: 5)")
    run_p.add_argument("--resolve-maybe", action=argparse.BooleanOptionalAction,
                        default=True)
    run_p.add_argument("--ollama-timeout", type=int, default=600)

    # Audio
    run_p.add_argument("--merge-gap", type=float, default=2.0,
                        help="Max gap (sec) to merge kept intervals (default: 2.0)")
    run_p.add_argument("--pad-before", type=float, default=0.35)
    run_p.add_argument("--pad-after", type=float, default=0.5)
    run_p.add_argument("--speed", type=float, default=1.25,
                        help="Playback speed (default: 1.25)")

    # Other
    run_p.add_argument("--dry-run", action="store_true")
    run_p.add_argument("--keep-temp", action="store_true")
    run_p.add_argument("--prefer-auto-subs", action="store_true")
    run_p.add_argument("--block-size", type=int, default=1200,
                        help="Target words per thematic block (default: 1200)")
    run_p.add_argument("--max-blocks", type=int, default=0,
                        help="Only process first N blocks, rest auto-kept (0=all, default: 0)")

    # ------------------------------------------------------------------
    # build-universe: Build cross-episode knowledge base from episodes 1-20
    # ------------------------------------------------------------------
    build_p = sub.add_parser(
        "build-universe",
        help=(
            "Build universe state from a playlist (Phase A + knowledge extraction only, "
            "no audio cutting). Processes episodes and extracts entities, concepts, "
            "claims, glossary terms, etc. into a cross-episode knowledge base."
        ),
    )
    build_p.add_argument("playlist_url", help="YouTube playlist URL")
    build_p.add_argument("--start", type=int, default=1,
                         help="First episode to process (1-based, default: 1)")
    build_p.add_argument("--end", type=int, default=20,
                         help="Last episode to process (1-based, default: 20)")
    build_p.add_argument("--state-file", default="",
                         help="Path to universe state JSON file (default: output/universe_state.json)")
    build_p.add_argument("--model", default="qwen2.5:3b",
                         help="Ollama model (default: qwen2.5:7b)")
    build_p.add_argument("--lang", default="en")
    build_p.add_argument("--output-dir", default="")
    build_p.add_argument("--block-size", type=int, default=1200)
    build_p.add_argument("--segment-gap", type=float, default=0.5)
    build_p.add_argument("--segment-max-words", type=int, default=400)
    build_p.add_argument("--segment-min-words", type=int, default=20)
    build_p.add_argument("--prefer-auto-subs", action="store_true")
    build_p.add_argument("--ollama-timeout", type=int, default=600)
    build_p.add_argument("--dry-run", action="store_true",
                         help="Skip LLM calls and just prepare data")

    # ------------------------------------------------------------------
    # process-playlist: Full pipeline with universe state context
    # ------------------------------------------------------------------
    proc_p = sub.add_parser(
        "process-playlist",
        help=(
            "Process playlist episodes using a pre-built universe state. "
            "Runs the full pipeline (download → classify with state context → cut audio) "
            "for each episode starting from a given index. Updates the universe state "
            "with new knowledge after each episode."
        ),
    )
    proc_p.add_argument("playlist_url", help="YouTube playlist URL")
    proc_p.add_argument("--state-file", required=True,
                        help="Path to universe state JSON file (required)")
    proc_p.add_argument("--start", type=int, default=21,
                        help="First episode to process (1-based, default: 21)")
    proc_p.add_argument("--end", type=int, default=0,
                        help="Last episode to process (0=until end, default: 0)")
    proc_p.add_argument("--model", default="qwen2.5:3b",
                         help="Ollama model for extraction/summarization (default: qwen2.5:3b)")
    proc_p.add_argument("--classify-model", default="qwen2.5:7b",
                         help="Ollama model for classification (default: qwen2.5:7b)")
    proc_p.add_argument("--lang", default="en")
    proc_p.add_argument("--output-dir", default="")
    proc_p.add_argument("--batch-size", type=int, default=5)
    proc_p.add_argument("--resolve-maybe", action=argparse.BooleanOptionalAction,
                        default=True)
    proc_p.add_argument("--merge-gap", type=float, default=2.0)
    proc_p.add_argument("--pad-before", type=float, default=0.35)
    proc_p.add_argument("--pad-after", type=float, default=0.5)
    proc_p.add_argument("--speed", type=float, default=1.25)
    proc_p.add_argument("--block-size", type=int, default=1200)
    proc_p.add_argument("--max-blocks", type=int, default=0)
    proc_p.add_argument("--segment-gap", type=float, default=0.5)
    proc_p.add_argument("--segment-max-words", type=int, default=400)
    proc_p.add_argument("--segment-min-words", type=int, default=20)
    proc_p.add_argument("--prefer-auto-subs", action="store_true")
    proc_p.add_argument("--keep-temp", action="store_true")
    proc_p.add_argument("--ollama-timeout", type=int, default=600)
    proc_p.add_argument("--dry-run", action="store_true")

    # status
    sub.add_parser("status", help="Check Ollama status")

    args = parser.parse_args()
    setup_logging(args.verbose)

    if args.command == "run":
        cmd_run(args)
    elif args.command == "status":
        cmd_status(args)
    elif args.command == "build-universe":
        cmd_build_universe(args)
    elif args.command == "process-playlist":
        cmd_process_playlist(args)


if __name__ == "__main__":
    main()
