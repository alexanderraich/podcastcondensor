"""CLI entry point for podcastcondensor."""

import argparse
import logging
import os
import sys

from podcastcondensor.config import Config
from podcastcondensor.ollama_client import check_ollama, list_models
from podcastcondensor.pipeline import run_pipeline


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
    run_p.add_argument("--model", default="qwen2.5:7b",
                        help="Ollama model (default: qwen2.5:7b)")
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

    # status
    sub.add_parser("status", help="Check Ollama status")

    args = parser.parse_args()
    setup_logging(args.verbose)

    if args.command == "run":
        cmd_run(args)
    elif args.command == "status":
        cmd_status(args)


if __name__ == "__main__":
    main()
