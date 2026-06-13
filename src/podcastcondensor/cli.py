"""CLI entry point for podcastcondensor."""

import argparse
import logging
import os
import sys

from podcastcondensor.config import Config
from podcastcondensor.ollama_client import check_ollama, list_models
from podcastcondensor.pipeline import run_pipeline


def setup_logging(verbose: bool = False):
    """Configure logging."""
    level = logging.DEBUG if verbose else logging.INFO
    fmt = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    logging.basicConfig(
        level=level,
        format=fmt,
        datefmt="%H:%M:%S",
        stream=sys.stderr,
    )


def cmd_run(args):
    """Run the condensation pipeline."""
    cfg = Config(
        default_model=args.model,
        lang=args.lang,
        output_root=os.path.abspath(args.output_dir) if args.output_dir else "",
        output_merge_gap=args.merge_gap,
        pad_before=args.pad_before,
        pad_after=args.pad_after,
        max_chunks_per_batch=args.max_chunks_per_batch,
        max_chars_per_chunk=args.max_chars_per_chunk,
        resolve_maybe=args.resolve_maybe,
        keep_temp=args.keep_temp,
        prefer_auto_subs=args.prefer_auto_subs,
        ollama_host=args.ollama_host,
        ollama_timeout=args.ollama_timeout,
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

    # Print summary
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
    """Check Ollama status and available models."""
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
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="podcastcondensor — local-first podcast condensing pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python -m podcastcondensor run URL\n"
            "  python -m podcastcondensor run URL --model qwen3:8b --lang en\n"
            "  python -m podcastcondensor status\n"
            "  python -m podcastcondensor run URL --dry-run\n"
        ),
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable debug logging",
    )
    parser.add_argument(
        "--ollama-host",
        default="http://localhost:11434",
        help="Ollama API host (default: http://localhost:11434)",
    )

    sub = parser.add_subparsers(dest="command", required=True)

    # run
    run_p = sub.add_parser("run", help="Run condensation pipeline")
    run_p.add_argument("url", help="YouTube URL")
    run_p.add_argument("--model", default="qwen3:8b",
                        help="Ollama model (default: qwen3:8b)")
    run_p.add_argument("--lang", default="en",
                        help="Subtitle language (default: en)")
    run_p.add_argument("--output-dir", default="",
                        help="Output directory (default: output/)")
    run_p.add_argument("--merge-gap", type=float, default=2.0,
                        help="Max gap in seconds to merge keep intervals (default: 2.0)")
    run_p.add_argument("--pad-before", type=float, default=0.35,
                        help="Seconds to pad before each interval (default: 0.35)")
    run_p.add_argument("--pad-after", type=float, default=0.5,
                        help="Seconds to pad after each interval (default: 0.5)")
    run_p.add_argument("--resolve-maybe", action=argparse.BooleanOptionalAction,
                        default=True,
                        help="Resolve maybe chunks with second pass (default: True)")
    run_p.add_argument("--dry-run", action="store_true",
                        help="Download only, skip LLM and audio")
    run_p.add_argument("--keep-temp", action="store_true",
                        help="Keep temporary segment files")
    run_p.add_argument("--prefer-auto-subs", action="store_true",
                        help="Prefer auto-generated subtitles")
    run_p.add_argument("--max-chunks-per-batch", type=int, default=20,
                        help="Chunks per classification batch (default: 20)")
    run_p.add_argument("--max-chars-per-chunk", type=int, default=600,
                        help="Max chars per chunk before truncation (default: 600)")
    run_p.add_argument("--ollama-timeout", type=int, default=120,
                        help="Ollama request timeout in seconds (default: 120)")

    # status
    status_p = sub.add_parser("status", help="Check Ollama status")

    args = parser.parse_args()
    setup_logging(args.verbose)

    if args.command == "run":
        cmd_run(args)
    elif args.command == "status":
        cmd_status(args)


if __name__ == "__main__":
    main()
