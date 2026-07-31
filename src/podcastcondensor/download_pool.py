"""Parallel episode download pool — audio + subtitles for a range of episodes.

Downloads audio and YouTube subtitles for multiple episodes concurrently,
with checkpoint support (skips already-downloaded files).
"""

import logging
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

from podcastcondensor.downloader import (
    download_audio,
    resolve_episode_sources,
)
from podcastcondensor.transcribe import transcribe_audio

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Manifest data class
# ---------------------------------------------------------------------------


@dataclass
class EpisodeManifest:
    """Audio + subtitles paths for one episode."""
    episode_number: int
    video_id: str
    title: str
    audio_path: str
    srt_path: str


# ---------------------------------------------------------------------------
# Single episode download
# ---------------------------------------------------------------------------


def _ensure_episode_artifacts(
    episode_num: int,
    video_url: str,
    title: str,
    video_id: str,
    output_root: str,
    *,
    audio_format: str = "mp3",
    audio_bitrate: str = "64k",
    whisper_model: str = "base",
) -> Optional[EpisodeManifest]:
    """Ensure one episode has audio + SRT ready.

    Always uses whisper transcription — YouTube subtitles are unreliable.

    Returns EpisodeManifest, or None if download/transcription fails.
    """
    ep_dir = os.path.join(output_root, f"ep-{episode_num:03d}")
    Path(ep_dir).mkdir(parents=True, exist_ok=True)

    # -- Audio --------------------------------------------------------------
    try:
        audio_path = download_audio(
            url=video_url,
            output_dir=ep_dir,
            video_id=video_id,
            audio_format=audio_format,
            audio_bitrate=audio_bitrate,
        )
    except Exception as e:
        logger.error("Audio download failed for ep %d: %s", episode_num, e)
        return None

    if not audio_path or not os.path.exists(audio_path):
        logger.error("No audio file for ep %d", episode_num)
        return None

    # -- Subtitles ----------------------------------------------------------
    srt_path = os.path.join(ep_dir, "source_subtitles.srt")

    # Check if SRT already exists
    if os.path.exists(srt_path):
        logger.info("SRT exists for ep %d, reusing", episode_num)
        return EpisodeManifest(
            episode_number=episode_num,
            video_id=video_id,
            title=title,
            audio_path=audio_path,
            srt_path=srt_path,
        )

    # Whisper transcription
    logger.info("Ep %d: transcribing via whisper...", episode_num)
    try:
        transcribe_audio(
            audio_path, ep_dir,
            model_size=whisper_model,
            beam_size=1,
            vad_filter=False,
            condition_on_previous_text=False,
        )
    except Exception as e:
        logger.error("Whisper transcription failed for ep %d: %s", episode_num, e)
        return None

    if not os.path.exists(srt_path):
        logger.error("Whisper produced no SRT for ep %d", episode_num)
        return None

    return EpisodeManifest(
        episode_number=episode_num,
        video_id=video_id,
        title=title,
        audio_path=audio_path,
        srt_path=srt_path,
    )


# ---------------------------------------------------------------------------
# Parallel download pool
# ---------------------------------------------------------------------------


def _scan_existing_episodes(
    output_root: str, start_ep: int, end_ep: int,
) -> List[EpisodeManifest]:
    """Scan output/ for existing episode artefacts without YouTube calls.

    Builds manifests from any ep-NNN directory that has both an MP3 and
    source_subtitles.srt. The MP3 filename is the video ID (set by
    download_audio). Fast — no network.
    """
    manifests = []
    for ep_num in range(start_ep, end_ep + 1):
        ep_dir = os.path.join(output_root, f"ep-{ep_num:03d}")
        if not os.path.isdir(ep_dir):
            continue
        srt_path = os.path.join(ep_dir, "source_subtitles.srt")
        if not os.path.exists(srt_path):
            continue
        # Find audio file (any mp3 except temp/checkpoint files)
        audio = None
        video_id = ""
        for f in os.listdir(ep_dir):
            if f.endswith(".mp3") and not f.startswith("_"):
                audio = os.path.join(ep_dir, f)
                video_id = os.path.splitext(f)[0]
                break
        if not audio:
            continue
        manifests.append(EpisodeManifest(
            episode_number=ep_num,
            video_id=video_id,
            title=f"Episode {ep_num}",
            audio_path=audio,
            srt_path=srt_path,
        ))
    return manifests


def ensure_all_episode_artifacts(
    playlist_url: str,
    output_root: str,
    start_episode: int = 1,
    end_episode: int = 140,
    *,
    parallel: int = 4,
    audio_format: str = "mp3",
    audio_bitrate: str = "64k",
    whisper_model: str = "base",
) -> List[EpisodeManifest]:
    """Download audio + SRT for a range of episodes in parallel.

    Always uses whisper transcription — YouTube subtitles are unreliable.

    Resolution strategy (in order):
      1. Scan output/ep-NNN/ for existing MP3+SRT — no YouTube call.
      2. For the remaining missing episodes, use playlist + YouTube search
         fallback via resolve_episode_sources().

    Returns list of EpisodeManifest (successful downloads only).
    """
    # ── Step 1: Scan existing artefacts on disk (fast, no network) ──────
    existing = _scan_existing_episodes(
        output_root, start_episode, end_episode,
    )
    existing_by_ep = {m.episode_number: m for m in existing}
    logger.info(
        "On-disk scan: %d/%d episodes already have artefacts",
        len(existing), end_episode - start_episode + 1,
    )

    missing = [ep for ep in range(start_episode, end_episode + 1)
               if ep not in existing_by_ep]
    if not missing:
        logger.info("All episodes already on disk — skipping YouTube entirely")
        existing.sort(key=lambda m: m.episode_number)
        return existing

    # ── Step 2: Resolve sources for missing episodes via YouTube ────────
    sources = resolve_episode_sources(
        playlist_url=playlist_url,
        start_ep=min(missing),
        end_ep=max(missing),
    )
    missing_set = set(missing)

    logger.info(
        "Download pool: %d existing + %d from YouTube = %d total",
        len(existing), len(sources), len(existing) + len(sources),
    )

    manifests: List[EpisodeManifest] = list(existing)
    failed: List[int] = []

    with ThreadPoolExecutor(max_workers=parallel) as pool:
        futures = {}
        for src in sources:
            ep_num = src["episode_number"]
            logger.info("Queuing ep %d: %s", ep_num, src.get("title", "")[:60])
            future = pool.submit(
                _ensure_episode_artifacts,
                episode_num=ep_num,
                video_url=src["video_url"],
                title=src.get("title", f"Episode {ep_num}"),
                video_id=src["id"],
                output_root=output_root,
                audio_format=audio_format,
                audio_bitrate=audio_bitrate,
                whisper_model=whisper_model,
            )
            futures[future] = ep_num

        for future in as_completed(futures):
            ep_num = futures[future]
            try:
                manifest = future.result()
                if manifest:
                    manifests.append(manifest)
                    logger.info(
                        "✓ Ep %d: audio=%s srt=%s",
                        ep_num,
                        os.path.basename(manifest.audio_path),
                        os.path.basename(manifest.srt_path),
                    )
                else:
                    logger.warning("✗ Ep %d: download failed", ep_num)
                    failed.append(ep_num)
            except Exception as e:
                logger.error("Ep %d download raised: %s", ep_num, e)
                failed.append(ep_num)

    # Sort by episode number for deterministic ordering
    manifests.sort(key=lambda m: m.episode_number)
    success_count = len(manifests)
    logger.info(
        "Download pool complete: %d/%d episodes ready",
        success_count, len(sources),
    )

    # Fail loudly on partial download — a master cut with a silently missing
    # episode must not be treated as complete.
    if failed:
        missing = ", ".join(f"ep-{n:03d}" for n in sorted(failed))
        raise RuntimeError(
            f"Download failed for {len(failed)} episode(s): {missing}. "
            "Aborting — no partial master cut."
        )
    return manifests
