"""Audio cutting — ffmpeg-based interval extraction and concat."""

import logging
import os
import subprocess
import tempfile
from pathlib import Path
from typing import List

logger = logging.getLogger(__name__)


def get_audio_duration(audio_path: str) -> float:
    """Get audio duration in seconds via ffprobe."""
    cmd = [
        "ffprobe", "-v", "error",
        "-show_entries", "format=duration",
        "-of", "csv=p=0",
        audio_path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffprobe failed: {result.stderr}")
    return float(result.stdout.strip())


def extract_segment(
    audio_path: str,
    start: float,
    end: float,
    output_path: str,
    format_spec: str = "mp3",
    sample_rate: int = 22050,
    bitrate: str = "64k",
):
    """Extract a segment from the audio file using ffmpeg.

    Uses fast copy mode (-c copy) — no re-encoding during extraction.
    Format conversion and speed-up are done in the concat step.
    """
    duration = end - start
    if duration <= 0.01:
        logger.warning("Skipping zero-duration segment: %s-%s", start, end)
        return

    cmd = [
        "ffmpeg", "-y",
        "-i", audio_path,
        "-ss", str(start),
        "-t", str(duration),
        "-c", "copy",  # fast copy — no re-encoding
        output_path,
    ]
    logger.debug("Extracting segment: %.2f-%.2f -> %s", start, end, output_path)
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"ffmpeg segment extraction failed: {result.stderr}"
        )


def _atempo_filter(speed: float) -> list:
    """Return ffmpeg filter chain for speed adjustment."""
    if abs(speed - 1.0) < 0.01:
        return []
    # atempo supports 0.5-2.0 range per filter, chain if needed
    filters = []
    remaining = speed
    while remaining > 2.0:
        filters.append("atempo=2.0")
        remaining /= 2.0
    while remaining < 0.5:
        filters.append("atempo=0.5")
        remaining /= 0.5
    filters.append(f"atempo={remaining:.3f}")
    return filters


def concat_segments(
    segment_paths: List[str],
    output_path: str,
    format_spec: str = "mp3",
    sample_rate: int = 22050,
    bitrate: str = "64k",
    speed: float = 1.0,
) -> str:
    """Concat audio segments using ffmpeg concat demuxer.

    Args:
        speed: Playback speed multiplier (1.0 = normal, 1.25 = faster)

    Returns path to output file.
    """
    if not segment_paths:
        raise ValueError("No segments to concatenate")

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    atempo = _atempo_filter(speed)

    if len(segment_paths) == 1:
        cmd = [
            "ffmpeg", "-y",
            "-i", segment_paths[0],
            "-ar", str(sample_rate),
            "-b:a", bitrate,
            "-ac", "1",
        ]
        if atempo:
            cmd += ["-filter:a", ",".join(atempo)]
        cmd.append(output_path)
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"ffmpeg copy failed: {result.stderr}")
        return output_path

    tmpdir = tempfile.mkdtemp()
    concat_file = os.path.join(tmpdir, "concat.txt")
    with open(concat_file, "w") as f:
        for seg in segment_paths:
            abs_path = os.path.abspath(seg)
            f.write(f"file '{abs_path}'\n")

    cmd = [
        "ffmpeg", "-y",
        "-f", "concat",
        "-safe", "0",
        "-i", concat_file,
        "-ar", str(sample_rate),
        "-b:a", bitrate,
        "-ac", "1",
    ]
    if atempo:
        cmd += ["-filter:a", ",".join(atempo)]
    cmd.append(output_path)

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg concat failed: {result.stderr}")

    logger.info("Created condensed audio (%.2fx): %s", speed, output_path)
    return output_path


def build_condensed_audio(
    audio_path: str,
    intervals: List[dict],
    output_path: str,
    format_spec: str = "mp3",
    sample_rate: int = 22050,
    bitrate: str = "64k",
    keep_temp: bool = False,
    speed: float = 1.0,
) -> str:
    """Build condensed audio from intervals.

    Args:
        audio_path: Path to original audio file
        intervals: List of {start, end} dicts
        output_path: Where to write final file
        format_spec: Audio format
        sample_rate: Sample rate
        bitrate: Audio bitrate
        keep_temp: If True, keep temp segment files
        speed: Playback speed multiplier (1.0 = normal, 1.25 = faster)

    Returns:
        Path to final audio file
    """
    if not intervals:
        raise ValueError("No intervals to build audio from")

    tmpdir = tempfile.mkdtemp(prefix="pc_segments_")
    segment_paths = []

    try:
        total = len(intervals)
        total_sec = sum(iv["end"] - iv["start"] for iv in intervals) / speed if speed > 0 else 0
        logger.info("Cutting %d intervals (%.0fs at %.2fx)...", total, total_sec, speed)
        for i, interval in enumerate(intervals):
            seg_path = os.path.join(tmpdir, f"seg_{i:04d}.{format_spec}")
            logger.info(
                "  [%d/%d] extracting %.1fs-%.1fs (%.0fs)...",
                i + 1, total, interval["start"], interval["end"],
                interval["end"] - interval["start"],
            )
            extract_segment(
                audio_path=audio_path,
                start=interval["start"],
                end=interval["end"],
                output_path=seg_path,
                format_spec=format_spec,
                sample_rate=sample_rate,
                bitrate=bitrate,
            )
            segment_paths.append(seg_path)

        logger.info("Concatenating %d segments with %.2fx speed...", len(segment_paths), speed)
        result = concat_segments(
            segment_paths, output_path,
            format_spec=format_spec,
            sample_rate=sample_rate,
            bitrate=bitrate,
            speed=speed,
        )
        logger.info("Done: %s", result)
        return result
    finally:
        if not keep_temp:
            import shutil
            if os.path.exists(tmpdir):
                shutil.rmtree(tmpdir)
        else:
            logger.info("Temp segments preserved in: %s", tmpdir)
