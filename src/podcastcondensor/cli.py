"""CLI entry point for podcastcondensor — DeepSeek-only."""

import argparse
import logging
import os
import sys

from podcastcondensor.config import Config
from podcastcondensor.llm.deepseek import resolve_api_key
from podcastcondensor.pipeline import run_pipeline
from podcastcondensor.playlist_pipeline import build_universe_state, process_with_universe_state
from podcastcondensor.universe_state import UniverseState


def setup_logging(verbose: bool = False):
    level = logging.DEBUG if verbose else logging.INFO
    fmt = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    logging.basicConfig(level=level, format=fmt, datefmt="%H:%M:%S", stream=sys.stderr)


def cmd_doctor(args):
    """Check DeepSeek API connectivity."""
    print("=" * 50)
    print("podcastcondensor doctor")
    print("=" * 50)
    api_key = resolve_api_key()
    print(f"\nDeepSeek API key: {'✅ set' if api_key else '❌ not set'}")
    if api_key and args.check:
        from podcastcondensor.llm.deepseek import DeepSeekClient
        client = DeepSeekClient(api_key=api_key)
        try:
            resp = client.generate("Reply OK", model="deepseek-chat", timeout=30, max_tokens=10)
            print(f"API connectivity: ✅ {resp[:50]}")
        except Exception as e:
            print(f"API connectivity: ❌ {e}")
    if not api_key:
        print("  Set ANTHROPIC_AUTH_TOKEN or DEEPSEEK_API_KEY env var")
    import subprocess
    try:
        r = subprocess.run(["ffmpeg", "-version"], capture_output=True, text=True, timeout=10)
        print(f"ffmpeg: {'✅' if r.returncode == 0 else '❌'}")
    except FileNotFoundError:
        print("ffmpeg: ❌ not found")


def cmd_build_universe(args):
    """Build universe state from episodes 1-21."""
    cfg = Config(
        lang=args.lang,
        output_root=os.path.abspath(args.output_dir) if args.output_dir else "",
        deepseek_timeout=600,
        prefer_auto_subs=args.prefer_auto_subs,
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
    summary = state.data.get("summary", "")
    print(f"\nUniverse state: {state.data['metadata'].get('last_built_episode', 0)} episodes")
    print(f"  Concepts: {len(state.data.get('concepts', []))}")
    print(f"  Entities: {len(state.data.get('entities', []))}")
    print(f"  Claims:   {len(state.data.get('claims', []))}")
    print(f"  Glossary: {len(state.data.get('glossary', []))}")


def cmd_process_playlist(args):
    """Process playlist episodes with universe state."""
    cfg = Config(
        lang=args.lang,
        output_root=os.path.abspath(args.output_dir) if args.output_dir else "",
        prefer_auto_subs=args.prefer_auto_subs,
        deepseek_timeout=600,
    )
    state_path = os.path.abspath(args.state_file)
    if not os.path.exists(state_path):
        print(f"Universe state not found: {state_path}")
        sys.exit(1)
    state = UniverseState(state_path)
    results = process_with_universe_state(
        playlist_url=args.playlist_url,
        cfg=cfg,
        state=state,
        start_episode=args.start,
        end_episode=args.end,
        dry_run=args.dry_run,
    )
    successful = sum(1 for r in results if r.get("success"))
    print(f"\nEpisodes: {successful}/{len(results)} successful")


def main():
    parser = argparse.ArgumentParser(description="podcastcondensor — DeepSeek-powered podcast condensing")
    parser.add_argument("--verbose", "-v", action="store_true")
    sub = parser.add_subparsers(dest="command", required=True)

    # doctor
    doc = sub.add_parser("doctor", help="Check DeepSeek API connectivity")
    doc.add_argument("--check", action="store_true", help="Test API (costs ~$0.0001)")
    doc.set_defaults(func=cmd_doctor)

    # build-universe
    build = sub.add_parser("build-universe", help="Build universe state from episodes 1-21")
    build.add_argument("playlist_url", help="YouTube playlist URL")
    build.add_argument("--start", type=int, default=1)
    build.add_argument("--end", type=int, default=21)
    build.add_argument("--state-file", default="", help="Output path for universe state JSON")
    build.add_argument("--output-dir", default="")
    build.add_argument("--dry-run", action="store_true")
    build.add_argument("--prefer-auto-subs", action="store_true")
    build.add_argument("--lang", default="en")
    build.set_defaults(func=cmd_build_universe)

    # process-playlist
    proc = sub.add_parser("process-playlist", help="Process episodes with universe state")
    proc.add_argument("playlist_url", help="YouTube playlist URL")
    proc.add_argument("--state-file", required=True, help="Path to universe state JSON")
    proc.add_argument("--start", type=int, default=22)
    proc.add_argument("--end", type=int, default=0, help="0 = until end")
    proc.add_argument("--output-dir", default="")
    proc.add_argument("--dry-run", action="store_true")
    proc.add_argument("--prefer-auto-subs", action="store_true")
    proc.add_argument("--lang", default="en")
    proc.set_defaults(func=cmd_process_playlist)

    args = parser.parse_args()
    setup_logging(args.verbose)
    args.func(args)


if __name__ == "__main__":
    main()
