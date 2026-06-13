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

    Fast cut using re-encode for reliable cutting.
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
        "-ar", str(sample_rate),
        "-b:a", bitrate,
        "-ac", "1",  # mono for speech
        output_path,
    ]
    logger.debug("Extracting segment: %.2f-%.2f -> %s", start, end, output_path)
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"ffmpeg segment extraction failed: {result.stderr}"
        )


def concat_segments(
    segment_paths: List[str],
    output_path: str,
    format_spec: str = "mp3",
    sample_rate: int = 22050,
    bitrate: str = "64k",
) -> str:
    """Concat audio segments using ffmpeg concat demuxer.

    Returns path to output file.
    """
    if not segment_paths:
        raise ValueError("No segments to concatenate")

    if len(segment_paths) == 1:
        # Single segment — just copy
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        cmd = [
            "ffmpeg", "-y",
            "-i", segment_paths[0],
            "-ar", str(sample_rate),
            "-b:a", bitrate,
            "-ac", "1",
            output_path,
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"ffmpeg copy failed: {result.stderr}")
        return output_path

    # Write concat file
    tmpdir = tempfile.mkdtemp()
    concat_file = os.path.join(tmpdir, "concat.txt")
    with open(concat_file, "w") as f:
        for seg in segment_paths:
            # Need absolute path for ffmpeg concat demuxer
            abs_path = os.path.abspath(seg)
            f.write(f"file '{abs_path}'\n")

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg", "-y",
        "-f", "concat",
        "-safe", "0",
        "-i", concat_file,
        "-ar", str(sample_rate),
        "-b:a", bitrate,
        "-ac", "1",
        output_path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg concat failed: {result.stderr}")

    logger.info("Created condensed audio: %s", output_path)
    return output_path


def build_condensed_audio(
    audio_path: str,
    intervals: List[dict],
    output_path: str,
    format_spec: str = "mp3",
    sample_rate: int = 22050,
    bitrate: str = "64k",
    keep_temp: bool = False,
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

    Returns:
        Path to final audio file
    """
    if not intervals:
        raise ValueError("No intervals to build audio from")

    tmpdir = tempfile.mkdtemp(prefix="pc_segments_")
    segment_paths = []

    try:
        for i, interval in enumerate(intervals):
            seg_path = os.path.join(tmpdir, f"seg_{i:04d}.{format_spec}")
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

        return concat_segments(
            segment_paths, output_path,
            format_spec=format_spec,
            sample_rate=sample_rate,
            bitrate=bitrate,
        )
    finally:
        if not keep_temp:
            import shutil
            if os.path.exists(tmpdir):
                shutil.rmtree(tmpdir)
        else:
            logger.info("Temp segments preserved in: %s", tmpdir)
