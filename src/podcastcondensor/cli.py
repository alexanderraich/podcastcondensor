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
from podcastcondensor.super_cut import build_super_cut, run_super_cut_brackets
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
    )
    state_path = os.path.abspath(args.state_file) if args.state_file else ""
    state = build_universe_state(
        playlist_url=args.playlist_url,
        cfg=cfg,
        start_episode=args.start,
        end_episode=args.end,
        state_path=state_path or None,
        dry_run=args.dry_run,
        skip_qa=not args.include_qa,
        skip_reading=not args.include_reading,
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
        deepseek_timeout=600,
        skip_audio=getattr(args, 'skip_audio', False),
        skip_global_state=not args.use_global_state,
    )

    if cfg.skip_global_state:
        # New compression path: one LLM call per episode, no universe state.
        # run_pipeline still handles Phase 1 (download) + new Phase 2 (compress).
        state = None
    else:
        # Legacy path: requires existing universe state for two-call pipeline.
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
      1. Parallel download audio + whisper transcription
      2. Build complete universe state (Phase 2 DeepSeek for new episodes)
      3. Extract core themes (one DeepSeek call over universe state)
      4. Resolve segments from universe state
      5. Select segments via per-theme LLM selection (with transcript context)
      6. Assemble audio with beeps (single=within-theme, triple=between-themes)
    """
    cfg = Config(
        lang=args.lang,
        output_root=os.path.abspath(args.output_dir) if args.output_dir else "",
        deepseek_timeout=600,
        master_cut_target_duration=args.target_duration,
        master_cut_output=args.output,
        master_cut_parallel_downloads=args.parallel_downloads,
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
            extra = f", {phase.get('deepseek_calls', 0)} new + {phase.get('loaded_from_disk', 0)} existing"
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

    warnings = result.get("warnings", [])
    if warnings:
        print(f"  Warnings: {len(warnings)}")
        for w in warnings[:10]:
            print(f"    ⚠ {w}")
    else:
        print(f"  Warnings: 0 (clean)")
    print("")


def cmd_build_super_cut(args):
    """Build per-theme MP3s for the top topics over the full episode range.

    Offline from disk (no download/whisper): merge per-episode global_state
    → chunked theme extraction → coalesce → episode-diverse resolution →
    per-theme LLM selection (LLM-owned volume, same as the range cuts) →
    one MP3 per theme. ``--dry-run`` stops after resolution and prints the
    theme table with per-theme episode spread.
    """
    cfg = Config(
        lang=args.lang,
        output_root=os.path.abspath(args.output_dir) if args.output_dir else "",
        deepseek_timeout=600,
        keep_temp=args.keep_temp,
        master_cut_target_duration=args.target_duration,
    )

    end_ep = args.end if args.end > 0 else 144

    result = build_super_cut(
        playlist_url=args.playlist_url,
        cfg=cfg,
        start_episode=args.start,
        end_episode=end_ep,
        chunk_size=args.chunk_size,
        max_themes=args.max_themes,
        theme_ids=args.theme,
        retry_empty_chunks=args.retry_empty_chunks,
        output_dir=os.path.abspath(args.output_dir) if args.output_dir else "",
        dry_run=args.dry_run,
        combine_name=args.combine,
        min_duration_floor=args.min_duration_floor,
    )

    print("=" * 60)
    print("SUPER CUT RESULTS" + (" (DRY RUN)" if args.dry_run else ""))
    print("=" * 60)
    for phase in result.get("phases", []):
        name = phase.get("phase", "?")
        elapsed = phase.get("elapsed_sec", 0)
        extra = ""
        if name == "merge_universe":
            extra = f", {phase.get('episodes', 0)} episodes, {phase.get('items', 0)} items"
        elif name == "chunk_themes":
            extra = f", {phase.get('chunks', 0)} chunks, {phase.get('chunk_theme_count', 0)} themes"
        elif name == "coalesce":
            extra = f", {phase.get('global_theme_count', 0)} global themes"
        elif name == "resolve_segments":
            extra = f", {phase.get('total_candidates', 0)} candidates"
        elif name == "select_segments":
            extra = f", {phase.get('selected_count', 0)} segments, {phase.get('total_duration_sec', 0):.0f}s"
        elif name == "assemble_audio":
            extra = f", → {phase.get('output_dir', '?')}"
        print(f"  {name:20s} {elapsed:.0f}s{extra}")

    if args.dry_run:
        print(f"\n  Themes report: {result.get('themes_path', 'N/A')}")
    else:
        outputs = result.get("outputs", {})
        print(f"\n  Per-theme MP3s: {len(outputs)}")
        for tid, path in list(outputs.items())[:10]:
            print(f"    - {tid}: {path}")
        if len(outputs) > 10:
            print(f"    ... ({len(outputs) - 10} more)")
        print(f"  Stats: {result.get('stats_path', 'N/A')}")

    errors = result.get("errors", [])
    if errors:
        print(f"  Errors:   {len(errors)}")
        for e in errors[:3]:
            print(f"    - {e}")
    else:
        print(f"  Errors:   0 (success)")

    verify_path = result.get("verify_path", "")
    if verify_path:
        print(f"  Boundary verification: {verify_path}")
    print("")


def cmd_super_cut_brackets(args):
    """Top theme per episode bracket — read-only from on-disk artefacts (0 LLM calls).

    Requires the super-cut discovery + candidates caches (built by one
    build-super-cut run or --dry-run). Brackets are episode-range strings
    like "40-80"; "rest" = all episodes not covered by any specified bracket.
    """
    output_root = os.path.abspath(args.output_dir) if args.output_dir else ""
    end_ep = args.end if args.end > 0 else 144

    analysis = run_super_cut_brackets(
        output_root=output_root,
        start_episode=args.start,
        end_episode=end_ep,
        bracket_specs=args.bracket,
        top_n=args.top,
    )

    print("=" * 60)
    print("TOP THEMES PER EPISODE BRACKET")
    print("=" * 60)
    for result in analysis["results"]:
        name = result["bracket"]
        print(f"\n── Bracket {name} ──")
        if not result["top"]:
            print("  (no candidates in this bracket)")
            continue
        for i, s in enumerate(result["top"], 1):
            span = _compact_span(s["episodes"])
            print(
                f"  {i}. {s['title']}  (imp={s['importance']:.2f}, "
                f"{s['candidates']} candidates, {s['duration_sec']:.0f}s, "
                f"eps {span})"
            )
    print("")


def _compact_span(eps):
    """Compact '1-5, 8' span for a sorted episode list (CLI-local helper)."""
    if not eps:
        return "?"
    parts = []
    start = prev = eps[0]
    for ep in eps[1:]:
        if ep == prev + 1:
            prev = ep
            continue
        parts.append(str(start) if start == prev else f"{start}-{prev}")
        start = prev = ep
    parts.append(str(start) if start == prev else f"{start}-{prev}")
    return ", ".join(parts)


def main():
    parser = argparse.ArgumentParser(description="podcastcondensor — DeepSeek-powered podcast condensing")
    parser.add_argument("--verbose", "-v", action="store_true")
    sub = parser.add_subparsers(dest="command", required=True)

    # doctor
    doc = sub.add_parser("doctor", help="Check DeepSeek API connectivity")
    doc.add_argument("--check", action="store_true", help="Test API (costs ~$0.0001)")
    doc.set_defaults(func=cmd_doctor)

    # build-universe
    build = sub.add_parser("build-universe", help="Build universe state from episodes")
    build.add_argument("playlist_url", help="YouTube playlist URL")
    build.add_argument("--start", type=int, default=1)
    build.add_argument("--end", type=int, default=21)
    build.add_argument("--state-file", default="", help="Output path for universe state JSON")
    build.add_argument("--output-dir", default="")
    build.add_argument("--dry-run", action="store_true")
    build.add_argument("--include-qa", action="store_true",
                       help="Include Q&A episodes (default: skip them — they don't develop coherent themes)")
    build.add_argument("--include-reading", action="store_true",
                       help="Include reading episodes (default: skip them — they're scriptural readings, not discussion)")
    build.add_argument("--lang", default="en")
    build.set_defaults(func=cmd_build_universe)

    # process-playlist
    proc = sub.add_parser("process-playlist", help="Main pipeline: download → compress → audio cut")
    proc.add_argument("playlist_url", help="YouTube playlist URL")
    proc.add_argument("--state-file", default="output/universe_state.json",
                      help="Path to universe state JSON (required with --use-global-state)")
    proc.add_argument("--start", type=int, default=22)
    proc.add_argument("--end", type=int, default=0, help="0 = until end")
    proc.add_argument("--output-dir", default="")
    proc.add_argument("--dry-run", action="store_true")
    proc.add_argument("--lang", default="en")
    proc.add_argument("--debug-max-intervals", type=int, default=0,
                      help="DEBUG: cap at N intervals for quick test listen")
    proc.add_argument("--skip-audio", action="store_true",
                      help="Skip audio cutting phase (stats only)")
    proc.add_argument("--use-global-state", action="store_true",
                      help="Enable legacy global-state pipeline (two LLM calls per episode instead of one-shot compress)")
    proc.set_defaults(func=cmd_process_playlist)

    # build-master-cut
    mc = sub.add_parser(
        "build-master-cut",
        help="Build a master cut across all episodes (~3.5h thematic anthology)",
    )
    mc.add_argument("playlist_url", help="YouTube playlist URL")
    mc.add_argument("--state-file", default="",
                    help="Path to universe state JSON (default: auto, range-scoped output/universe_state_{START}_{END}.json)")
    mc.add_argument("--output", default="master_cut.mp3",
                    help="Output master cut audio path (default: master_cut.mp3)")
    mc.add_argument("--target-duration", type=int, default=6750,
                    help="Target duration in seconds (default: 6750 = 90min at 1.25x)")
    mc.add_argument("--start", type=int, default=1,
                    help="First episode to include (default: 1)")
    mc.add_argument("--end", type=int, default=0,
                    help="Last episode to include (default: 0 = 140)")
    mc.add_argument("--parallel-downloads", type=int, default=4,
                    help="Parallel download workers (default: 4)")
    mc.add_argument("--keep-temp", action="store_true",
                    help="Keep temporary files (debug)")
    mc.add_argument("--whisper-model", default="base",
                    help="Whisper model size (default: base)")
    mc.add_argument("--output-dir", default="")
    mc.add_argument("--lang", default="en")
    mc.set_defaults(func=cmd_build_master_cut)

    # build-super-cut
    sc = sub.add_parser(
        "build-super-cut",
        help="Full-corpus thematic anthology: per-theme MP3s over all episodes (offline from disk)",
    )
    sc.add_argument("playlist_url",
                    help="YouTube playlist URL (unused; super cut is offline from disk)")
    sc.add_argument("--start", type=int, default=1,
                    help="First episode to include (default: 1)")
    sc.add_argument("--end", type=int, default=0,
                    help="Last episode to include (default: 0 = 144)")
    sc.add_argument("--chunk-size", type=int, default=12,
                    help="Episodes per theme-extraction chunk (default: 12)")
    sc.add_argument("--max-themes", type=int, default=25,
                    help="Cap on global themes after coalesce dedup (default: 25)")
    sc.add_argument("--theme", action="append", default=[],
                    help="Limit to matching theme(s) by id/title substring (repeatable; default: all)")
    sc.add_argument("--retry-empty-chunks", action="store_true",
                    help="From a discovery cache, re-attempt ONLY chunks that produced 0 themes (targeted recovery, not a full re-run)")
    sc.add_argument("--output-dir", default="")
    sc.add_argument("--target-duration", type=int, default=6750,
                    help="Informational only (knapsack fallback); volume is LLM-owned")
    sc.add_argument("--combine", default="",
                    help="Combine the --theme cuts into ONE MP3 named <combine>.mp3, "
                         "played in --theme order with triple beeps between themes")
    sc.add_argument("--min-duration-floor", type=float, default=0.0,
                    help="Deterministically top up each theme's selection to at "
                         "least this many seconds (@1x) using its best un-kept "
                         "candidates (per-episode capped, later episodes preferred). "
                         "0 = pure LLM-owned volume (archive default)")
    sc.add_argument("--dry-run", action="store_true",
                    help="Merge → chunk → coalesce → resolve only; print theme table")
    sc.add_argument("--keep-temp", action="store_true",
                    help="Keep temporary files (debug)")
    sc.add_argument("--lang", default="en")
    sc.set_defaults(func=cmd_build_super_cut)

    # super-cut-brackets
    br = sub.add_parser(
        "super-cut-brackets",
        help="Top theme per episode bracket — read-only from on-disk artefacts (0 LLM calls)",
    )
    br.add_argument("--start", type=int, default=1,
                    help="First episode to include (default: 1)")
    br.add_argument("--end", type=int, default=0,
                    help="Last episode to include (default: 0 = 144)")
    br.add_argument("--bracket", action="append", default=[],
                    help="Episode range like '40-80' or '81-120' (repeatable; rest = complement)")
    br.add_argument("--top", type=int, default=5,
                    help="How many top themes to show per bracket (default: 5)")
    br.add_argument("--output-dir", default="")
    br.set_defaults(func=cmd_super_cut_brackets)

    args = parser.parse_args()
    setup_logging(args.verbose)
    args.func(args)


if __name__ == "__main__":
    main()
