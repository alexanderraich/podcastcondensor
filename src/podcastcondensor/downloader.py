"""YouTube downloader — audio + subtitles via yt-dlp subprocess.

Resumable: checks for existing files before re-downloading.
"""

import json
import logging
import os
import re
import subprocess
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

VIDEO_ID_RE = re.compile(r"(?:v=|youtu\.be/|/shorts/)([\w-]{11})")


def extract_video_id(url: str) -> Optional[str]:
    """Extract YouTube video ID from various URL formats."""
    m = VIDEO_ID_RE.search(url)
    if m:
        return m.group(1)
    return None


def _run_ytdlp(args, **kwargs):
    """Run yt-dlp with given args, return (returncode, stdout, stderr)."""
    cmd = ["yt-dlp"] + args
    logger.debug("Running: %s", " ".join(cmd))
    result = subprocess.run(
        cmd, capture_output=True, text=True, **kwargs
    )
    return result.returncode, result.stdout, result.stderr


def download_metadata(url: str) -> dict:
    """Get video metadata without downloading."""
    code, out, err = _run_ytdlp([
        "--dump-json",
        "--no-playlist",
        url,
    ])
    if code != 0:
        raise RuntimeError(f"yt-dlp metadata failed: {err.strip()}")
    return json.loads(out)


def _find_existing_audio(output_dir: str, video_id: str, audio_format: str) -> Optional[str]:
    """Check if audio file already exists."""
    expected = os.path.join(output_dir, f"{video_id}.{audio_format}")
    if os.path.exists(expected):
        logger.info("Audio exists, reusing: %s", expected)
        return expected
    for f in os.listdir(output_dir):
        if f.startswith(video_id) and f.endswith(f".{audio_format}"):
            path = os.path.join(output_dir, f)
            if os.path.getsize(path) > 0:
                logger.info("Audio exists, reusing: %s", path)
                return path
    return None


def _find_existing_subtitle(output_dir: str, video_id: str) -> Optional[str]:
    """Check if subtitle file already exists (srt or vtt)."""
    for f in sorted(os.listdir(output_dir)):
        if f.startswith(video_id) and f.endswith((".srt", ".vtt")):
            path = os.path.join(output_dir, f)
            if os.path.getsize(path) > 0:
                logger.info("Subtitles exist, reusing: %s", path)
                return path
    return None


def download_audio(
    url: str,
    output_dir: str,
    video_id: str,
    format_spec: str = "bestaudio/best",
    audio_format: str = "mp3",
    audio_bitrate: str = "64k",
) -> str:
    """Download best available audio, return path to file.

    Skips download if audio file already exists in output_dir.
    """
    # Check for existing
    existing = _find_existing_audio(output_dir, video_id, audio_format)
    if existing:
        return existing

    outtmpl = os.path.join(output_dir, "%(id)s.%(ext)s")
    code, _, err = _run_ytdlp([
        "-x",
        "--audio-format", audio_format,
        "--audio-quality", audio_bitrate,
        "--no-playlist",
        "--embed-thumbnail",
        "--add-metadata",
        "-o", outtmpl,
        url,
    ])
    if code != 0:
        raise RuntimeError(f"Audio download failed: {err.strip()}")

    expected = os.path.join(output_dir, f"{video_id}.{audio_format}")
    if os.path.exists(expected):
        return expected

    for f in os.listdir(output_dir):
        if f.startswith(video_id) and f.endswith(f".{audio_format}"):
            return os.path.join(output_dir, f)
    raise FileNotFoundError(
        f"Could not find downloaded audio for {video_id} in {output_dir}"
    )


def download_subtitles(
    url: str,
    output_dir: str,
    video_id: str,
    lang: str = "en",
    prefer_auto: bool = False,
) -> Optional[str]:
    """Download subtitles.

    Returns path to the subtitle file actually used, or None if none found.
    Skips download if subtitle files already exist.

    Strategy:
    1. If prefer_auto: try auto subs first, then manual.
    2. Otherwise: try manual subs first, then auto subs.
    """
    # Check for existing
    existing = _find_existing_subtitle(output_dir, video_id)
    if existing:
        return existing

    outtmpl = os.path.join(output_dir, "%(id)s.%(ext)s")

    if prefer_auto:
        subtitle_types = ["auto", "manual"]
    else:
        subtitle_types = ["manual", "auto"]

    for stype in subtitle_types:
        if stype == "manual":
            subs = "--write-subs"
            langs = "--sub-langs", lang
        else:
            subs = "--write-auto-subs"
            langs = "--sub-langs", lang

        code, _, err = _run_ytdlp([
            "--skip-download",
            "--no-playlist",
            subs,
            *langs,
            "--convert-subs", "srt",
            "-o", outtmpl,
            url,
        ])
        if code == 0:
            for f in sorted(os.listdir(output_dir)):
                if f.startswith(video_id) and f.endswith((".srt", ".vtt")):
                    fpath = os.path.join(output_dir, f)
                    if os.path.getsize(fpath) > 0:
                        logger.info("Using subtitles: %s (%s)", f, stype)
                        return fpath
        logger.debug("No %s subtitles for %s: %s", stype, lang, err.strip())

    logger.warning("No subtitles found for %s (lang=%s)", video_id, lang)
    return None


def download_all(url: str, output_dir: str, lang: str = "en",
                 prefer_auto: bool = False,
                 audio_format: str = "mp3",
                 audio_bitrate: str = "64k") -> dict:
    """Download audio and subtitles for a URL.

    Skips existing files — safe to re-run.

    Returns dict with keys:
        video_id, title, audio_path, subtitle_path, duration
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    meta = download_metadata(url)
    video_id = meta["id"]
    title = meta.get("title", video_id)

    logger.info("Downloading: %s — %s", video_id, title)

    audio_path = download_audio(
        url, output_dir, video_id,
        audio_format=audio_format,
        audio_bitrate=audio_bitrate,
    )
    subtitle_path = download_subtitles(
        url, output_dir, video_id,
        lang=lang, prefer_auto=prefer_auto,
    )

    return {
        "video_id": video_id,
        "title": title,
        "audio_path": audio_path,
        "subtitle_path": subtitle_path,
        "duration": meta.get("duration", 0),
        "output_dir": output_dir,
    }
