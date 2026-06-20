"""YouTube downloader — audio + subtitles via yt-dlp subprocess.

Resumable: checks for existing files before re-downloading.
"""

import json
import logging
import os
import re
import subprocess
from pathlib import Path
from typing import List, Optional

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

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    outtmpl = os.path.join(output_dir, "%(id)s.%(ext)s")
    code, _, err = _run_ytdlp([
        "-x",
        "--js-runtimes", "node",
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
    Path(output_dir).mkdir(parents=True, exist_ok=True)

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
            "--js-runtimes", "node",
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


def list_playlist(playlist_url: str, max_entries: int = 999) -> list:
    """Fetch all video entries from a YouTube playlist using yt-dlp.

    Args:
        playlist_url: YouTube playlist URL
        max_entries: Maximum number of entries to fetch (default 999).

    Returns list of dicts with keys: id, title, url, index (1-based).

    Raises RuntimeError if yt-dlp fails.
    """
    code, out, err = _run_ytdlp([
        "--flat-playlist",
        "--dump-json",
        "--playlist-end", str(max_entries),
        playlist_url,
    ], timeout=180)
    if code != 0:
        raise RuntimeError(
            f"yt-dlp failed to fetch playlist (exit {code}): {err.strip()}"
        )
    entries = []
    for line in out.strip().splitlines():
        if not line.strip():
            continue
        try:
            entry = json.loads(line)
            video_id = entry.get("id", "")
            entries.append({
                "id": video_id,
                "title": entry.get("title", ""),
                "url": f"https://www.youtube.com/watch?v={video_id}",
                "index": entry.get("playlist_index", len(entries) + 1),
            })
        except json.JSONDecodeError:
            logger.warning("Skipping unparseable playlist entry: %s", line[:80])
            continue

    logger.info("Fetched %d entries from playlist", len(entries))
    return entries


_EP_NUM_RE = re.compile(r"\[Ep\.?\s*(\d+)\]", re.IGNORECASE)


def extract_episode_number(title: str) -> Optional[int]:
    """Extract canonical episode number from a title like '[Ep. 121]'."""
    m = _EP_NUM_RE.search(title)
    return int(m.group(1)) if m else None


def resolve_video_by_search(episode_num: int, max_results: int = 5) -> Optional[dict]:
    """Search YouTube for a Lord of Spirits episode by number.

    Uses ytsearch with precise query. Returns first matching result
    whose title contains [Ep. N] or None if no match found.

    Result dict: {id, title, url, index}
    """
    query = f'"Lord of Spirits" "Episode {episode_num}" OR "[Ep. {episode_num}]"'
    code, out, err = _run_ytdlp([
        "--flat-playlist",
        "--dump-json",
        "--playlist-end", str(max_results),
        f"ytsearch{max_results}:{query}",
    ], timeout=60)

    if code != 0:
        logger.warning("Search failed for ep %d: %s", episode_num, err.strip()[:100])
        return None

    for line in out.strip().splitlines():
        if not line.strip():
            continue
        try:
            entry = json.loads(line)
            vid = entry.get("id", "")
            title = entry.get("title", "")
            ep = extract_episode_number(title)
            if ep == episode_num:
                logger.info("Search found Ep. %d: %s (%s)", episode_num, title[:70], vid)
                return {
                    "id": vid,
                    "title": title,
                    "url": f"https://www.youtube.com/watch?v={vid}",
                    "index": episode_num,
                }
        except json.JSONDecodeError:
            continue

    # Broader fallback: search without quotes
    query2 = f"Lord of Spirits {episode_num}"
    code2, out2, err2 = _run_ytdlp([
        "--flat-playlist",
        "--dump-json",
        "--playlist-end", str(max_results),
        f"ytsearch{max_results}:{query2}",
    ], timeout=60)
    if code2 != 0:
        return None
    for line in out2.strip().splitlines():
        if not line.strip():
            continue
        try:
            entry = json.loads(line)
            vid = entry.get("id", "")
            title = entry.get("title", "")
            ep = extract_episode_number(title)
            if ep == episode_num:
                return {"id": vid, "title": title, "url": f"https://www.youtube.com/watch?v={vid}", "index": episode_num}
        except json.JSONDecodeError:
            continue

    logger.warning("Could not find Ep. %d via search", episode_num)
    return None


def resolve_episode_sources(
    playlist_url: str,
    start_ep: int = 1,
    end_ep: int = 20,
) -> List[dict]:
    """Resolve episode video URLs for a range, combining playlist + direct search.

    1. Fetches full playlist and indexes entries by [Ep. N] in titles.
    2. For each requested episode, returns playlist entry if found.
    3. For missing episodes, falls back to direct YouTube search.

    Returns list of dicts: {episode_number, title, video_url, source_type}
    Sorted by episode_number ascending.
    """
    # Fetch playlist entries
    playlist_entries = list_playlist(playlist_url)
    logger.info("Playlist has %d total entries", len(playlist_entries))

    # Build lookup: episode_number -> entry from playlist
    playlist_by_ep = {}
    for e in playlist_entries:
        ep = extract_episode_number(e["title"])
        if ep is not None:
            if ep not in playlist_by_ep:
                playlist_by_ep[ep] = e

    logger.info(
        "Found %d unique episode numbers in playlist (range: %d-%d)",
        len(playlist_by_ep),
        min(playlist_by_ep.keys()) if playlist_by_ep else 0,
        max(playlist_by_ep.keys()) if playlist_by_ep else 0,
    )

    # Build source list for requested range
    sources = []
    for ep in range(start_ep, end_ep + 1):
        if ep in playlist_by_ep:
            e = playlist_by_ep[ep]
            sources.append({
                "episode_number": ep,
                "title": e["title"],
                "video_url": e["url"],
                "id": e["id"],
                "source_type": "playlist",
            })
            logger.info("  Ep. %d: from playlist (index %d)", ep, e.get("index", "?"))
        else:
            logger.info("  Ep. %d: not in playlist, searching directly...", ep)
            found = resolve_video_by_search(ep)
            if found:
                sources.append({
                    "episode_number": ep,
                    "title": found["title"],
                    "video_url": found["url"],
                    "id": found["id"],
                    "source_type": "direct_fallback",
                })
                logger.info("  Ep. %d: found via search: %s", ep, found["id"])
            else:
                logger.warning("  Ep. %d: could not resolve! Skipping.", ep)

    sources.sort(key=lambda s: s["episode_number"])
    playlist_count = sum(1 for s in sources if s["source_type"] == "playlist")
    direct_count = sum(1 for s in sources if s["source_type"] == "direct_fallback")
    logger.info(
        "Resolved %d/%d episodes (%d from playlist, %d from direct search)",
        len(sources), end_ep - start_ep + 1, playlist_count, direct_count,
    )
    return sources


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
