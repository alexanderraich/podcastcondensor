"""YouTube downloader — audio + subtitles via yt-dlp subprocess."""

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
    # Try a more general fallback — yt-dlp can resolve it
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


def download_audio(
    url: str,
    output_dir: str,
    format_spec: str = "bestaudio/best",
    audio_format: str = "mp3",
    audio_bitrate: str = "64k",
) -> str:
    """Download best available audio, return path to file."""
    outtmpl = os.path.join(output_dir, "%(id)s.%(ext)s")
    # First download best audio in native format
    code, _, err = _run_ytdlp([
        "-x",  # extract audio
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

    # Find the produced file
    # Use yt-dlp to get the video ID
    meta = download_metadata(url)
    vid = meta["id"]
    expected = os.path.join(output_dir, f"{vid}.{audio_format}")
    if os.path.exists(expected):
        return expected

    # Fallback: scan directory for any matching file
    for f in os.listdir(output_dir):
        if f.startswith(vid) and f.endswith(f".{audio_format}"):
            return os.path.join(output_dir, f)
    raise FileNotFoundError(
        f"Could not find downloaded audio for {vid} in {output_dir}"
    )


def download_subtitles(
    url: str,
    output_dir: str,
    lang: str = "en",
    prefer_auto: bool = False,
) -> Optional[str]:
    """Download subtitles.

    Returns path to the subtitle file actually used, or None if none found.

    Strategy:
    1. If prefer_auto: try auto subs first, then manual.
    2. Otherwise: try manual subs first, then auto subs.
    3. Returns first successful download.
    """
    vid = extract_video_id(url)
    if not vid:
        meta = download_metadata(url)
        vid = meta["id"]

    outtmpl = os.path.join(output_dir, "%(id)s.%(ext)s")

    if prefer_auto:
        subtitle_types = ["auto", "manual"]
    else:
        subtitle_types = ["manual", "auto"]

    for stype in subtitle_types:
        if stype == "manual":
            subs = f"--write-subs"
            langs = f"--sub-langs", f"{lang}"
        else:
            subs = "--write-auto-subs"
            langs = "--sub-langs", f"{lang}"

        code, _, err = _run_ytdlp([
            "--skip-download",
            "--no-playlist",
            subs,
            *langs,
            "--convert-subs", "srt",  # easiest to parse
            "-o", outtmpl,
            url,
        ])
        if code == 0:
            # Find the actual subtitle file
            for f in sorted(os.listdir(output_dir)):
                if f.startswith(vid) and f.endswith((".srt", ".vtt")):
                    fpath = os.path.join(output_dir, f)
                    if os.path.getsize(fpath) > 0:
                        logger.info("Using subtitles: %s (%s)", f, stype)
                        return fpath
        # If manual subs requested but yt-dlp errored (no manual subs exist),
        # continue to try auto
        logger.debug(
            "No %s subtitles for %s: %s", stype, lang, err.strip()
        )

    logger.warning("No subtitles found for %s (lang=%s)", vid, lang)
    return None


def download_all(url: str, output_dir: str, lang: str = "en",
                 prefer_auto: bool = False,
                 audio_format: str = "mp3",
                 audio_bitrate: str = "64k") -> dict:
    """Download audio and subtitles for a URL.

    Returns dict with keys:
        video_id, title, audio_path, subtitle_path
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    meta = download_metadata(url)
    video_id = meta["id"]
    title = meta.get("title", video_id)

    logger.info("Downloading: %s — %s", video_id, title)

    audio_path = download_audio(
        url, output_dir, audio_format=audio_format,
        audio_bitrate=audio_bitrate
    )
    subtitle_path = download_subtitles(
        url, output_dir, lang=lang, prefer_auto=prefer_auto
    )

    return {
        "video_id": video_id,
        "title": title,
        "audio_path": audio_path,
        "subtitle_path": subtitle_path,
        "duration": meta.get("duration", 0),
        "output_dir": output_dir,
    }
