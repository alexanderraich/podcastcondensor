"""Cut concepts + claims segments from universe state into one audio file.

Usage:
    python3 scripts/cut_universe_segments.py [--state output/universe_state.json]
        [--output condensed_claims_concepts.mp3]
        [--episodes 31-40]
        [--speed 1.25]
        [--dry-run]
"""

import argparse
import json
import logging
import os
import shutil
import subprocess
import tempfile
import time
from collections import defaultdict
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def load_state(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_episode_titles(output_root: str) -> dict:
    """Try to load episode titles from manifest."""
    titles = {}
    # Check for episode manifests or fallback to summary
    state_path = os.path.join(output_root, "universe_state.json")
    if os.path.exists(state_path):
        with open(state_path) as f:
            data = json.load(f)
        for es in data.get("episode_summaries", []):
            titles[es["episode_number"]] = es.get("summary", "")[:60]
    return titles


def collect_segments(
    state: dict,
    categories: list,
    episode_range: set,
) -> list:
    """Collect all unique segments from specified categories for given episodes.

    Returns list of dicts: {episode, start, end, category, item_id, item_title}
    Deduplicated by (episode, start, end).
    """
    seen = set()
    segments = []

    for cat in categories:
        for item in state.get(cat, []):
            item_id = item.get("id", "?")
            item_title = item.get("title") or item.get("term") or item.get("text", "")[:60]
            for seg in item.get("segments", []):
                ep = seg.get("episode", 0)
                if ep not in episode_range:
                    continue
                start = seg.get("start", 0)
                end = seg.get("end", 0)
                if end <= start:
                    continue
                key = (ep, round(start, 1), round(end, 1))
                if key in seen:
                    continue
                seen.add(key)
                segments.append({
                    "episode": ep,
                    "start": start,
                    "end": end,
                    "duration": end - start,
                    "category": cat,
                    "item_id": item_id,
                    "item_title": item_title,
                })

    segments.sort(key=lambda s: (s["episode"], s["start"]))
    return segments


def find_audio_path(output_root: str, episode: int) -> str:
    """Find the mp3 audio file for an episode."""
    ep_dir = os.path.join(output_root, f"ep-{episode:03d}")
    if not os.path.isdir(ep_dir):
        return None
    for f in sorted(os.listdir(ep_dir)):
        if f.endswith(".mp3") and not f.startswith("condensed"):
            return os.path.join(ep_dir, f)
    return None


def _ionice_cmd(cmd: list) -> list:
    """Prepend ionice if available."""
    import shutil as _shutil
    if _shutil.which("ionice"):
        return ["ionice", "-c", "2", "-n", "7"] + cmd
    return cmd


def _atempo_filters(speed: float) -> list:
    filters = []
    remaining = speed
    while remaining > 2.0:
        filters.append("atempo=2.0")
        remaining /= 2.0
    filters.append(f"atempo={remaining:.2f}")
    return filters


def extract_and_assemble(
    segments: list,
    output_root: str,
    output_path: str,
    *,
    sample_rate: int = 22050,
    bitrate: str = "64k",
    speed: float = 1.25,
) -> str:
    """Extract each segment, add beep separators, concat into one file."""
    tmpdir = tempfile.mkdtemp(prefix="universe_cut_")
    segments_dir = os.path.join(tmpdir, "segments")
    Path(segments_dir).mkdir(exist_ok=True)

    try:
        # Generate beep
        beep_path = os.path.join(tmpdir, "beep.mp3")
        subprocess.run(_ionice_cmd([
            "ffmpeg", "-y",
            "-f", "lavfi",
            "-i", "sine=f=1000:d=0.25",
            "-ar", str(sample_rate),
            "-b:a", bitrate,
            "-ac", "1",
            beep_path,
        ]), capture_output=True, text=True, timeout=30, check=True)

        # Extract segments
        seg_paths = []
        total = len(segments)
        logger.info("Extracting %d segments...", total)

        by_ep = defaultdict(list)
        for i, s in enumerate(segments):
            by_ep[s["episode"]].append((i, s))

        audio_cache = {}
        failures = 0

        for ep in sorted(by_ep.keys()):
            audio_path = find_audio_path(output_root, ep)
            if not audio_path:
                logger.warning("  No audio found for ep %d, skipping %d segments", ep, len(by_ep[ep]))
                failures += len(by_ep[ep])
                continue
            audio_cache[ep] = audio_path

        for idx, seg in enumerate(segments):
            audio_path = audio_cache.get(seg["episode"])
            if not audio_path:
                seg_paths.append(None)
                continue

            out_path = os.path.join(segments_dir, f"seg_{idx:04d}.mp3")
            dur = seg["end"] - seg["start"]
            cmd = _ionice_cmd([
                "ffmpeg", "-y",
                "-ss", f"{seg['start']:.3f}",
                "-i", audio_path,
                "-t", f"{dur:.3f}",
                "-ar", str(sample_rate),
                "-b:a", bitrate,
                "-ac", "1",
                out_path,
            ])
            try:
                subprocess.run(cmd, capture_output=True, text=True, timeout=120, check=True)
                seg_paths.append(out_path)
            except Exception as e:
                logger.error("  Segment %d/%d failed: %s", idx + 1, total, e)
                seg_paths.append(None)
                failures += 1

            if (idx + 1) % 10 == 0 or idx + 1 == total:
                logger.info("  Extracted %d/%d segments", idx + 1, total)

        # Build concat list with beeps
        concat_file = os.path.join(tmpdir, "concat.txt")
        with open(concat_file, "w") as f:
            need_beep = False
            for sp in seg_paths:
                if sp is None:
                    continue
                if need_beep:
                    f.write(f"file '{os.path.abspath(beep_path)}'\n")
                f.write(f"file '{os.path.abspath(sp)}'\n")
                need_beep = True

        valid_count = sum(1 for p in seg_paths if p is not None)
        logger.info("Concat list built: %d segments + %d beeps", valid_count, valid_count - 1)

        # Final concat + speed
        atempo = ",".join(_atempo_filters(speed))
        cmd = _ionice_cmd([
            "ffmpeg", "-y",
            "-f", "concat",
            "-safe", "0",
            "-i", concat_file,
            "-af", atempo,
            "-ar", str(sample_rate),
            "-b:a", bitrate,
            "-ac", "1",
            output_path,
        ])
        subprocess.run(cmd, capture_output=True, text=True, timeout=3600, check=True)

        # Duration
        dur_cmd = ["ffprobe", "-v", "error", "-show_entries", "format=duration",
                    "-of", "csv=p=0", output_path]
        result = subprocess.run(dur_cmd, capture_output=True, text=True, timeout=30)
        final_dur = float(result.stdout.strip()) if result.stdout else 0

        logger.info("Done! Output: %s", output_path)
        logger.info("Duration: %.0fs (%.1f min)", final_dur, final_dur / 60)
        if failures:
            logger.warning("Failures: %d/%d segments", failures, total)

        return output_path

    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def main():
    parser = argparse.ArgumentParser(description="Cut concept+claim segments from universe state")
    parser.add_argument("--state", default="output/universe_state.json")
    parser.add_argument("--output-root", default="output")
    parser.add_argument("--output", default="output/condensed_claims_concepts_31-40.mp3")
    parser.add_argument("--episodes", default="31-40", help="Episode range (e.g. 31-40)")
    parser.add_argument("--speed", type=float, default=1.25)
    parser.add_argument("--dry-run", action="store_true", help="Just print stats, don't cut")
    args = parser.parse_args()

    # Parse episode range
    parts = args.episodes.split("-")
    if len(parts) == 1:
        ep_range = {int(parts[0])}
    else:
        ep_range = set(range(int(parts[0]), int(parts[1]) + 1))

    state = load_state(args.state)

    # Collect segments from concepts and claims
    segments = collect_segments(
        state=state,
        categories=["concepts", "claims"],
        episode_range=ep_range,
    )

    total_dur = sum(s["duration"] for s in segments)
    by_ep = defaultdict(list)
    for s in segments:
        by_ep[s["episode"]].append(s)

    print("=" * 60)
    print(f"CONCEPTS + CLAIMS SEGMENTS FOR EPS {args.episodes}")
    print("=" * 60)
    for ep in sorted(by_ep):
        dur = sum(s["duration"] for s in by_ep[ep])
        print(f"  Ep {ep:3d}: {len(by_ep[ep]):3d} clips, {dur/60:.0f}m ({dur:.0f}s)")

    print(f"\n  Total: {len(segments)} clips")
    print(f"  Duration: {total_dur:.0f}s = {total_dur/60:.0f}m = {total_dur/3600:.2f}h")
    print(f"  At {args.speed}×: {total_dur/args.speed:.0f}s = {total_dur/args.speed/60:.0f}m = {total_dur/args.speed/3600:.2f}h")
    print(f"  Categories: concepts + claims")
    print(f"  Episode range: {args.episodes}")

    if args.dry_run:
        print("\n  Dry run — no audio cut. Use --output to specify path.")
        return

    # Check audio availability
    print("\n--- Checking audio files ---")
    missing = []
    for ep in sorted(by_ep):
        audio = find_audio_path(args.output_root, ep)
        if audio:
            size_mb = os.path.getsize(audio) / 1024 / 1024
            print(f"  Ep {ep}: ✅ {os.path.basename(audio)} ({size_mb:.0f} MB)")
        else:
            print(f"  Ep {ep}: ❌ No audio found")
            missing.append(ep)

    if missing:
        logger.warning("Missing audio for episodes: %s", missing)
        answer = input("Continue without these episodes? (y/N): ")
        if answer.lower() != "y":
            logger.info("Aborted")
            return

    # Cut
    extract_and_assemble(
        segments=segments,
        output_root=args.output_root,
        output_path=args.output,
        speed=args.speed,
    )


if __name__ == "__main__":
    main()
