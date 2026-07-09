"""CLI entry point for podcastcondensor — DeepSeek-only."""

import argparse
import logging
import os
import sys

from podcastcondensor.config import Config
from podcastcondensor.llm.deepseek import resolve_api_key
from podcastcondensor.pipeline import run_pipeline
from podcastcondensor.playlist_pipeline import (
    build_universe_state,
    process_with_universe_state,
    build_master_cut,
)
from podcastcondensor.minimal_theme_cut import build_minimal_theme_cut
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
        skip_audio=getattr(args, 'skip_audio', False),
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
        debug_max_intervals=args.debug_max_intervals,
    )
    successful = sum(1 for r in results if r.get("success"))
    print(f"\nEpisodes: {successful}/{len(results)} successful")


def cmd_build_master_cut(args):
    """Build a master cut across all episodes.

    6 phases:
      1. Parallel download audio + subs (YT subs first)
      2. Build complete universe state (Phase 2 DeepSeek for new episodes)
      3. Extract core themes (one DeepSeek call over universe state)
      4. Map themes to SRT segments (keyword search)
      5. Select segments within time budget
      6. Assemble audio with dual beeps (single=within-theme, triple=between-themes)
    """
    cfg = Config(
        lang=args.lang,
        output_root=os.path.abspath(args.output_dir) if args.output_dir else "",
        deepseek_timeout=600,
        prefer_auto_subs=args.prefer_auto_subs,
        master_cut_target_duration=args.target_duration,
        master_cut_output=args.output,
        master_cut_parallel_downloads=args.parallel_downloads,
        master_cut_prefer_yt_subs=not args.force_whisper,
        keep_temp=args.keep_temp,
        whisper_model=args.whisper_model,
    )

    end_ep = args.end if args.end > 0 else 140

    # Resolve output path — make absolute so build_master_cut doesn't
    # double-nest it under output_root
    out_path = os.path.abspath(args.output) if not os.path.isabs(args.output) else args.output

    result = build_master_cut(
        playlist_url=args.playlist_url,
        cfg=cfg,
        state_file=os.path.abspath(args.state_file) if args.state_file else "",
        output_path=out_path,
        target_duration=args.target_duration,
        start_episode=args.start,
        end_episode=end_ep,
        parallel_downloads=args.parallel_downloads,
        prefer_yt_subs=not args.force_whisper,
        force_whisper=args.force_whisper,
    )

    # Print results
    print("=" * 60)
    print("MASTER CUT RESULTS")
    print("=" * 60)
    for phase in result.get("phases", []):
        name = phase.get("phase", "?")
        elapsed = phase.get("elapsed_sec", 0)
        extra = ""
        if name == "download":
            extra = f", {phase.get('episodes_downloaded', 0)} episodes"
        elif name == "build_universe":
            extra = f", {phase.get('new_episodes', 0)} new + {phase.get('existing_skipped', 0)} existing"
        elif name == "extract_themes":
            extra = f", {phase.get('theme_count', 0)} themes"
        elif name == "map_themes":
            extra = f", {phase.get('total_segments', 0)} segments, {phase.get('total_available_sec', 0):.0f}s available"
        elif name == "select_segments":
            extra = f", {phase.get('selected_count', 0)} segments, {phase.get('total_duration_sec', 0):.0f}s"
        elif name == "assemble_audio":
            extra = f", → {phase.get('output_path', '?')}"
        print(f"  {name:20s} {elapsed:.0f}s{extra}")

    print(f"\n  Output:   {result.get('output_path', 'N/A')}")
    errors = result.get("errors", [])
    if errors:
        print(f"  Errors:   {len(errors)}")
        for e in errors[:3]:
            print(f"    - {e}")
    else:
        print(f"  Errors:   0 (success)")
    print("")


def cmd_build_minimal_theme(args):
    """Build a minimal cut of one theme — LLM decides the length."""
    cfg = Config(
        output_root=os.path.abspath(args.output_dir) if args.output_dir else "",
        deepseek_timeout=600,
        keep_temp=args.keep_temp,
    )

    end_ep = args.end if args.end > 0 else 140
    out_path = os.path.abspath(args.output) if not os.path.isabs(args.output) else args.output

    result = build_minimal_theme_cut(
        theme_id=args.theme_id,
        playlist_url=args.playlist_url,
        cfg=cfg,
        state_file=os.path.abspath(args.state_file) if args.state_file else "",
        themes_file=os.path.abspath(args.themes_file) if args.themes_file else "",
        output_path=out_path,
        start_episode=args.start,
        end_episode=end_ep,
    )

    # Print results
    print("=" * 60)
    print("MINIMAL THEME CUT RESULTS")
    print("=" * 60)
    for phase in result.get("phases", []):
        name = phase.get("phase", "?")
        elapsed = phase.get("elapsed_sec", 0)
        extra = ""
        if name == "download":
            extra = f", {phase.get('episodes', 0)} episodes"
        elif name == "load_theme":
            extra = f", {phase.get('theme_title', '?')} ({phase.get('related_items', 0)} items)"
        elif name == "resolve_segments":
            extra = f", {phase.get('candidate_count', 0)} candidates, {phase.get('total_available_sec', 0):.0f}s"
        elif name == "llm_selection":
            extra = f", {phase.get('selected', 0)}/{phase.get('candidates', 0)} kept, {phase.get('total_duration_sec', 0):.0f}s"
        elif name == "assemble_audio":
            extra = f", → {phase.get('output_path', '?')}"
        print(f"  {name:20s} {elapsed:.0f}s{extra}")

    print(f"\n  Output:   {result.get('output_path', 'N/A')}")
    errors = result.get("errors", [])
    if errors:
        print(f"  Errors:   {len(errors)}")
        for e in errors[:3]:
            print(f"    - {e}")
    else:
        print(f"  Errors:   0 (success)")
    print("")


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
    proc.add_argument("--debug-max-intervals", type=int, default=0,
                      help="DEBUG: cap at N intervals for quick test listen")
    proc.add_argument("--skip-audio", action="store_true",
                      help="Skip audio cutting phase (stats only)")
    proc.set_defaults(func=cmd_process_playlist)

    # build-master-cut
    mc = sub.add_parser(
        "build-master-cut",
        help="Build a master cut across all episodes (~3.5h thematic anthology)",
    )
    mc.add_argument("playlist_url", help="YouTube playlist URL")
    mc.add_argument("--state-file", default="",
                    help="Path to universe state JSON (default: output/universe_state.json)")
    mc.add_argument("--output", default="master_cut.mp3",
                    help="Output master cut audio path (default: master_cut.mp3)")
    mc.add_argument("--target-duration", type=int, default=12600,
                    help="Target duration in seconds (default: 12600 = 3.5h)")
    mc.add_argument("--start", type=int, default=1,
                    help="First episode to include (default: 1)")
    mc.add_argument("--end", type=int, default=0,
                    help="Last episode to include (default: 0 = 140)")
    mc.add_argument("--parallel-downloads", type=int, default=4,
                    help="Parallel download workers (default: 4)")
    mc.add_argument("--force-whisper", action="store_true",
                    help="Skip YT subs, always transcribe with whisper")
    mc.add_argument("--keep-temp", action="store_true",
                    help="Keep temporary files (debug)")
    mc.add_argument("--whisper-model", default="base",
                    help="Whisper model size when YT subs unavailable (default: base)")
    mc.add_argument("--output-dir", default="")
    mc.add_argument("--lang", default="en")
    mc.add_argument("--prefer-auto-subs", action="store_true")
    mc.set_defaults(func=cmd_build_master_cut)

    # build-minimal-theme
    mt = sub.add_parser(
        "build-minimal-theme",
        help="Build a minimal audio cut for one theme — LLM decides the length",
    )
    mt.add_argument("theme_id", help="Kebab-case theme ID (e.g. 'theosis-and-salvation')")
    mt.add_argument("playlist_url",
                    help="YouTube playlist URL (unused; kept for backwards compatibility)")
    mt.add_argument("--state-file", default="",
                    help="Path to universe state JSON (default: output/universe_state.json)")
    mt.add_argument("--themes-file", default="output/_themes.json",
                    help="Path to cached themes JSON (default: output/_themes.json)")
    mt.add_argument("--output", default="output/minimal_theme_cut.mp3",
                    help="Output audio path (default: output/minimal_theme_cut.mp3)")
    mt.add_argument("--start", type=int, default=1,
                    help="First episode to include (default: 1)")
    mt.add_argument("--end", type=int, default=0,
                    help="Last episode to include (default: 0 = 140)")
    mt.add_argument("--keep-temp", action="store_true",
                    help="Keep temporary files (debug)")
    mt.add_argument("--output-dir", default="")
    mt.set_defaults(func=cmd_build_minimal_theme)

    args = parser.parse_args()
    setup_logging(args.verbose)
    args.func(args)


if __name__ == "__main__":
    main()
