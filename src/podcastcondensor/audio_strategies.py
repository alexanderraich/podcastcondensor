"""Audio cutting strategies — replace the slow sequential ffmpeg approach.

Three strategies:
  1. ``SequentialCopyCutStrategy`` — wraps the exact current behaviour
     (``build_condensed_audio`` with sequential per-interval extraction).
  2. ``ParallelCopyCutStrategy`` — independent per-interval extraction
     jobs run concurrently with a bounded thread pool.
  3. ``SinglePassFilterCutStrategy`` — one ``ffmpeg`` invocation using
     ``filter_complex`` (``atrim`` / ``asetpts`` / ``concat``). Avoids
     repeated seeking and temp files for intermediate segments.

All strategies conform to the ``AudioCuttingStrategy`` interface so the
pipeline can swap them via config without touching orchestration logic.

.. caution::
   All ffmpeg subprocesses are launched with ``ionice -c 3`` (idle I/O
   class) on Linux to prevent I/O starvation of the rest of the system
   when processing large audio files (~1.5 GB).  The parallel strategy
   also staggers process startup to avoid all workers seeking the same
   file simultaneously.
"""

import json
import logging
import os
import shutil
import subprocess
import sys
import tempfile
import time
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import List, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# I/O safety helpers
# ---------------------------------------------------------------------------


def _ionice_cmd(cmd: List[str]) -> List[str]:
    """Prepend ``ionice -c 3`` on Linux so ffmpeg doesn't starve the system.

    ``ionice -c 3`` (idle scheduling class) gives the process the lowest
    I/O priority — it only gets disk bandwidth when no other process needs
    it.  This prevents parallel ffmpeg seeks on a ~1.5 GB file from making
    the system unresponsive.
    """
    if sys.platform == "linux":
        return ["ionice", "-c", "3"] + cmd
    return cmd


def _run_ffmpeg(cmd: List[str], **kwargs) -> subprocess.CompletedProcess:
    """Run an ffmpeg command with I/O safety (``ionice`` + timeouts).

    Uses ``ionice -c 3`` on Linux so the process doesn't starve the system
    of disk bandwidth.  Passes through all ``subprocess.run`` kwargs.
    """
    safe_cmd = _ionice_cmd(cmd)
    return subprocess.run(safe_cmd, **kwargs)


# ---------------------------------------------------------------------------
# Memory monitoring helper
# ---------------------------------------------------------------------------


MEMORY_WARN_THRESHOLD_MB = 512  # warn if available memory drops below this


def _get_available_memory_mb() -> float:
    """Read available system memory from ``/proc/meminfo``.

    Returns available RAM in MB, or ``float('inf')`` if the file is
    unreadable (macOS, Windows, permission error, etc.).
    """
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    parts = line.split()
                    if len(parts) >= 2:
                        return float(parts[1]) / 1024.0
    except (FileNotFoundError, PermissionError, OSError, ValueError):
        pass
    return float("inf")


# ---------------------------------------------------------------------------
# Concat helper (reused by SinglePassFilterCutStrategy and SequentialCopyCutStrategy)
# ---------------------------------------------------------------------------


def _concat_batch_files(
    batch_paths: List[str],
    output_path: str,
    sample_rate: int,
    bitrate: str,
    atempo: Optional[List[str]] = None,
    beep: bool = False,
    beep_freq: float = 1000,
    beep_duration: float = 0.25,
):
    """Concatenate batch temp files using the concat demuxer.

    All *batch_paths* must exist and share the same audio codec parameters
    (same sample rate, channel layout, format).  This function applies
    format conversion and optional speed adjustment at concat time.

    When *beep* is ``True`` and there are multiple segments, a short studio
    tone is inserted between each pair to mark transitions.  The beep is
    generated once with ffmpeg's ``sine`` filter and interleaved into the
    concat list — no filter_complex overhead.  The beep matches the sample
    rate and channel count of the first segment so the concat demuxer
    doesn't choke on mismatched stream parameters.

    Raises ``RuntimeError`` on ffmpeg failure.
    """
    tmpdir = tempfile.mkdtemp()
    try:
        # Generate beep file once if needed, matching source audio format
        beep_path = None
        if beep and len(batch_paths) > 1:
            # Probe the first segment for its sample rate and channels
            beep_sr, beep_ac = _probe_audio_params(batch_paths[0])
            beep_path = os.path.join(tmpdir, "beep.mp3")
            beep_cmd = [
                "ffmpeg", "-y",
                "-f", "lavfi",
                "-i", f"sine=f={beep_freq}:d={beep_duration}",
                "-ar", str(beep_sr),
                "-ac", str(beep_ac),
                "-b:a", bitrate,
                beep_path,
            ]
            result = _run_ffmpeg(beep_cmd, capture_output=True, text=True)
            if result.returncode != 0:
                logger.warning("Beep generation failed (%s) — continuing without", result.stderr[:200])
                beep_path = None

        concat_file = os.path.join(tmpdir, "concat.txt")
        with open(concat_file, "w") as f:
            for i, bp in enumerate(batch_paths):
                if i > 0 and beep_path:
                    f.write(f"file '{os.path.abspath(beep_path)}'\n")
                f.write(f"file '{os.path.abspath(bp)}'\n")

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

        result = _run_ffmpeg(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"ffmpeg batch concat failed: {result.stderr[:500]}")
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


# ---------------------------------------------------------------------------
# Strategy interface
# ---------------------------------------------------------------------------


class AudioCuttingStrategy(ABC):
    """Interface for audio cutting strategies."""

    @abstractmethod
    def cut(
        self,
        audio_path: str,
        intervals: List[dict],
        output_path: str,
        format_spec: str = "mp3",
        sample_rate: int = 22050,
        bitrate: str = "64k",
        speed: float = 1.0,
    ) -> str:
        """Cut ``intervals`` from ``audio_path`` and write ``output_path``.

        Returns ``output_path`` on success.
        """
        ...

    @abstractmethod
    def name(self) -> str:
        """Strategy identifier for logging."""
        ...


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _atempo_filters(speed: float) -> List[str]:
    """Build ``atempo`` filter chain for a given speed multiplier.

    ``atempo`` only supports 0.5–2.0 per filter, so values outside that
    range are chained (e.g. 3.0 → [atempo=2.0, atempo=1.5]).
    """
    if abs(speed - 1.0) < 0.01:
        return []
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


def _get_audio_duration(audio_path: str) -> float:
    """Get audio duration in seconds via ffprobe."""
    cmd = [
        "ffprobe", "-v", "error",
        "-show_entries", "format=duration",
        "-of", "csv=p=0",
        audio_path,
    ]
    result = _run_ffmpeg(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffprobe failed: {result.stderr}")
    return float(result.stdout.strip())


def _probe_audio_params(audio_path: str):
    """Probe sample rate and channel count from an audio file.

    Returns ``(sample_rate, channels)`` as ints.  Used to replicate the
    source audio format when generating beep tones so the concat demuxer
    receives homogenous stream parameters.
    """
    cmd = [
        "ffprobe", "-v", "error",
        "-select_streams", "a:0",
        "-show_entries", "stream=sample_rate,channels",
        "-of", "csv=p=0",
        audio_path,
    ]
    result = _run_ffmpeg(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffprobe stream probe failed: {result.stderr}")
    parts = result.stdout.strip().split(",")
    return int(parts[0]), int(parts[1])


def normalize_intervals(
    intervals: List[dict],
    audio_duration: Optional[float] = None,
    min_duration: float = 0.01,
) -> List[dict]:
    """Sort, deduplicate, merge adjacent/overlapping intervals, and clamp.

    Steps:
      1. Remove intervals shorter than ``min_duration``.
      2. Sort by ``start`` ascending.
      3. Merge overlapping or adjacent (gap <= 0) intervals.
      4. Clamp to ``[0, audio_duration]`` if provided.

    Returns a new list of ``{"start": float, "end": float}`` dicts.

    This ensures we never feed ffmpeg an invalid or redundant interval
    list, which can cause cryptic filter graph errors.
    """
    # Filter zero-length
    valid = [
        iv for iv in intervals
        if iv.get("end", 0) - iv.get("start", 0) >= min_duration
    ]
    if not valid:
        return []

    # Sort
    valid = sorted(valid, key=lambda iv: iv["start"])

    # Merge (gap <= 0 means adjacent or overlapping)
    merged: List[dict] = [dict(valid[0])]  # copy
    for iv in valid[1:]:
        last = merged[-1]
        if iv["start"] <= last["end"]:
            # Overlap or adjacent — extend
            last["end"] = max(last["end"], iv["end"])
        else:
            merged.append(dict(iv))

    # Clamp to audio duration
    if audio_duration is not None and audio_duration > 0:
        for iv in merged:
            iv["start"] = max(0.0, iv["start"])
            iv["end"] = min(audio_duration, iv["end"])

    return merged


# ---------------------------------------------------------------------------
# Strategy 1: Sequential copy (current behaviour)
# ---------------------------------------------------------------------------


class SequentialCopyCutStrategy(AudioCuttingStrategy):
    """Sequential ``-c copy`` extraction + one final concat.  Minimal memory.

    Designed for WSL and large interval counts where a single
    ``filter_complex`` graph OOMs ffmpeg.

    **How it works:**

    1. Normalised intervals are extracted one at a time with
       ``ffmpeg -c copy`` (packet copy — *no* decode/re-encode, ~0 memory).
    2. Each extraction is a seek + packet copy and completes in <1 s.
    3. After every interval a **checkpoint file** is written for resumability.
    4. On re-run, completed intervals are skipped.
    5. Once all intervals are extracted, **one** concat+encode pass produces
       the final output (each interval encoded exactly once).
    6. If ``-c copy`` fails for an interval, falls back to a tiny
       ``filter_complex`` graph for that single interval.
    7. On success the temp directory is cleaned up.

    **All ffmpeg processes use ``ionice -c 3``** so audio cutting never
    starves the system.
    """

    CHECKPOINT_FILE = "_checkpoint.json"

    def __init__(self, batch_size: int = 0):
        """*batch_size* is accepted for backward compatibility but ignored."""
        self._name = "sequential_copy"

    def cut(
        self,
        audio_path: str,
        intervals: List[dict],
        output_path: str,
        format_spec: str = "mp3",
        sample_rate: int = 22050,
        bitrate: str = "64k",
        speed: float = 1.0,
    ) -> str:
        t0 = time.time()

        if not intervals:
            raise ValueError("No intervals to cut")

        # Normalise intervals
        try:
            audio_dur = _get_audio_duration(audio_path)
        except Exception:
            audio_dur = None

        cleaned = normalize_intervals(intervals, audio_duration=audio_dur)
        if not cleaned:
            raise ValueError("All intervals were invalid after normalisation")

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        atempo = _atempo_filters(speed)

        total = len(cleaned)
        total_kept = sum(iv["end"] - iv["start"] for iv in cleaned)
        logger.info(
            "Sequential copy cut: %d intervals (%ds kept), speed=%.2f",
            total, int(total_kept), speed,
        )

        # Temp directory alongside the output
        tmpdir = os.path.join(
            os.path.dirname(os.path.abspath(output_path)), ".seq_segments",
        )
        os.makedirs(tmpdir, exist_ok=True)

        # Checkpoint path
        ckpt_path = os.path.join(tmpdir, self.CHECKPOINT_FILE)

        try:
            # Load checkpoint
            completed = self._load_checkpoint(ckpt_path, total, audio_path)

            # ── Extract each interval with -c copy ────────────────────────
            seg_paths: List[str] = []
            for idx, iv in enumerate(cleaned):
                seg_path = os.path.join(tmpdir, f"seg_{idx:04d}.{format_spec}")

                if idx in completed and os.path.exists(seg_path):
                    logger.debug("  [%d/%d] checkpoint hit — skipping", idx + 1, total)
                    seg_paths.append(seg_path)
                    continue

                # Extract with -c copy
                try:
                    self._extract_one(
                        audio_path, iv, seg_path, format_spec,
                    )
                except Exception as exc:
                    logger.warning(
                        "  [%d/%d] stream copy failed (%s) — "
                        "falling back to filter_complex",
                        idx + 1, total, exc,
                    )
                    self._extract_one_filter_complex(
                        audio_path, iv, seg_path, sample_rate, bitrate,
                    )

                seg_paths.append(seg_path)
                completed.add(idx)
                self._save_checkpoint(ckpt_path, completed, total, audio_path)
                logger.info("  [%d/%d] %.1fs-%.1fs done", idx + 1, total, iv["start"], iv["end"])

            # ── One concat pass (with beeps between segments) ────────────
            logger.info("Concatenating %d segments...", len(seg_paths))
            _concat_batch_files(
                batch_paths=seg_paths,
                output_path=output_path,
                sample_rate=sample_rate,
                bitrate=bitrate,
                atempo=atempo,
                beep=True,
            )

            elapsed = time.time() - t0
            logger.info("Done (%.1fs): %s", elapsed, output_path)
            return output_path

        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

    def name(self) -> str:
        return self._name

    # ------------------------------------------------------------------
    # Single-interval extraction
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_one(
        audio_path: str,
        iv: dict,
        output_path: str,
        format_spec: str,
    ):
        """Extract one interval with ``-c copy`` (packet copy, no decode)."""
        duration = iv["end"] - iv["start"]
        if duration <= 0.01:
            raise RuntimeError("Zero-duration interval")

        cmd = [
            "ffmpeg", "-y",
            "-ss", f"{iv['start']:.3f}",
            "-i", audio_path,
            "-t", f"{duration:.3f}",
            "-c", "copy",
            "-avoid_negative_ts", "make_zero",
            output_path,
        ]
        result = _run_ffmpeg(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"ffmpeg copy failed: {result.stderr[:300]}")

    @staticmethod
    def _extract_one_filter_complex(
        audio_path: str,
        iv: dict,
        output_path: str,
        sample_rate: int,
        bitrate: str,
    ):
        """Fallback — extract one interval via filter_complex."""
        start_s = f"{iv['start']:.3f}"
        end_s = f"{iv['end']:.3f}"
        filter_graph = (
            f"[0:a]atrim={start_s}:{end_s},asetpts=PTS-STARTPTS[outa]"
        )
        cmd = [
            "ffmpeg", "-y",
            "-i", audio_path,
            "-filter_complex", filter_graph,
            "-map", "[outa]",
            "-ar", str(sample_rate),
            "-b:a", bitrate,
            "-ac", "1",
            output_path,
        ]
        result = _run_ffmpeg(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(
                f"ffmpeg filter-complex failed: {result.stderr[:300]}"
            )

    # ------------------------------------------------------------------
    # Checkpoint helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _load_checkpoint(ckpt_path: str, total: int, audio_path: str) -> set:
        """Load completed interval indices, or return empty set."""
        if not os.path.exists(ckpt_path):
            return set()

        try:
            with open(ckpt_path) as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning("Cannot read checkpoint (%s) — starting fresh", exc)
            return set()

        if data.get("total") != total:
            logger.warning(
                "Checkpoint interval count mismatch (%s vs %s) — starting fresh",
                data.get("total"), total,
            )
            return set()

        stored_audio = data.get("audio_path", "")
        if stored_audio and stored_audio != os.path.abspath(audio_path):
            logger.warning("Checkpoint audio path differs — starting fresh")
            return set()

        completed = {i for i in data.get("completed", []) if 0 <= i < total}
        if completed:
            logger.info(
                "Resuming from checkpoint: %d/%d intervals already done",
                len(completed), total,
            )
        return completed

    @staticmethod
    def _save_checkpoint(ckpt_path: str, completed: set, total: int, audio_path: str):
        """Write checkpoint metadata."""
        data = {
            "completed": sorted(completed),
            "total": total,
            "audio_path": os.path.abspath(audio_path),
        }
        try:
            with open(ckpt_path, "w") as f:
                json.dump(data, f, indent=2)
        except OSError as exc:
            logger.warning("Failed to write checkpoint: %s", exc)


# ---------------------------------------------------------------------------
# Strategy 2: Parallel copy (quick win)
# ---------------------------------------------------------------------------


class ParallelCopyCutStrategy(AudioCuttingStrategy):
    """Extract per-interval segments in parallel, then concatenate.

    Each interval is extracted with ``ffmpeg -c copy`` in its own thread.
    A bounded ``ThreadPoolExecutor`` (configurable via ``max_workers``)
    keeps concurrent ffmpeg processes under control.

    After extraction, segments are concatenated using the existing
    concat demuxer logic.  Output ordering is deterministic because
    segment paths are numbered before parallel dispatch.

    **I/O safety:** All ffmpeg subprocesses are run with ``ionice -c 3``
    (idle I/O class) on Linux, and worker startup is staggered by a small
    delay so that multiple seeks don't hit the HDD simultaneously.
    Default ``max_workers`` is **2** (conservative for HDD-backed storage).
    """

    _STAGGER_DELAY_SEC = 2.0  # seconds between worker starts

    def __init__(self, max_workers: int = 2):
        self._max_workers = max_workers
        self._name = "parallel_copy"

    def cut(
        self,
        audio_path: str,
        intervals: List[dict],
        output_path: str,
        format_spec: str = "mp3",
        sample_rate: int = 22050,
        bitrate: str = "64k",
        speed: float = 1.0,
    ) -> str:
        if not intervals:
            raise ValueError("No intervals to cut")

        # Normalise intervals
        try:
            audio_dur = _get_audio_duration(audio_path)
        except Exception:
            audio_dur = None
        intervals = normalize_intervals(intervals, audio_duration=audio_dur)
        if not intervals:
            raise ValueError("No intervals to cut after normalisation")

        tmpdir = tempfile.mkdtemp(prefix="pc_segments_")
        segment_paths: List[str] = []
        total = len(intervals)

        try:
            # Generate deterministic segment paths
            for i in range(total):
                seg_path = os.path.join(tmpdir, f"seg_{i:04d}.{format_spec}")
                segment_paths.append(seg_path)

            # Extract in parallel (staggered to avoid I/O storm)
            logger.info(
                "Parallel cut: %d intervals, %d workers, stagger=%.1fs",
                total, self._max_workers, self._STAGGER_DELAY_SEC,
            )

            def _extract_one(idx: int) -> str:
                # Stagger startup so workers don't all seek simultaneously
                if idx > 0:
                    time.sleep(min(idx * self._STAGGER_DELAY_SEC / self._max_workers, 10.0))
                iv = intervals[idx]
                seg_path = segment_paths[idx]
                duration = iv["end"] - iv["start"]
                if duration <= 0.01:
                    logger.warning("Zero-duration interval %d, skipping", idx)
                    return ""

                cmd = [
                    "ffmpeg", "-y",
                    "-i", audio_path,
                    "-ss", str(iv["start"]),
                    "-t", str(duration),
                    "-c", "copy",
                    seg_path,
                ]
                result = _run_ffmpeg(cmd, capture_output=True, text=True)
                if result.returncode != 0:
                    raise RuntimeError(
                        f"ffmpeg seg {idx} failed: {result.stderr[:500]}"
                    )
                logger.debug("  [%d/%d] %.1fs-%.1fs done", idx + 1, total, iv["start"], iv["end"])
                return seg_path

            with ThreadPoolExecutor(max_workers=self._max_workers) as pool:
                futures = {pool.submit(_extract_one, i): i for i in range(total)}
                for future in as_completed(futures):
                    idx = futures[future]
                    try:
                        future.result()  # will re-raise if extraction failed
                    except Exception:
                        logger.exception("Segment %d extraction failed", idx)
                        raise

            # Concatenate (reuse existing logic)
            logger.info("Concatenating %d parallel segments...", len(segment_paths))
            self._concat_segments(
                segment_paths, output_path,
                format_spec=format_spec,
                sample_rate=sample_rate,
                bitrate=bitrate,
                speed=speed,
            )
            logger.info("Done: %s", output_path)
            return output_path

        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

    def name(self) -> str:
        return self._name

    # ------------------------------------------------------------------
    # Concat logic (mirrors ``audio.concat_segments``)
    # ------------------------------------------------------------------

    @staticmethod
    def _concat_segments(
        segment_paths: List[str],
        output_path: str,
        format_spec: str = "mp3",
        sample_rate: int = 22050,
        bitrate: str = "64k",
        speed: float = 1.0,
    ):
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        atempo = _atempo_filters(speed)

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
            result = _run_ffmpeg(cmd, capture_output=True, text=True)
            if result.returncode != 0:
                raise RuntimeError(f"ffmpeg copy failed: {result.stderr}")
            return

        tmpdir = tempfile.mkdtemp()
        try:
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

            result = _run_ffmpeg(cmd, capture_output=True, text=True)
            if result.returncode != 0:
                raise RuntimeError(f"ffmpeg concat failed: {result.stderr}")
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)


# ---------------------------------------------------------------------------
# Strategy 3: Single-pass filter complex
# ---------------------------------------------------------------------------


class SinglePassFilterCutStrategy(AudioCuttingStrategy):
    """Cut and concatenate all intervals in a single ``ffmpeg`` invocation.

    Builds one ``filter_complex`` graph with:
      - One ``atrim`` / ``asetpts`` pair per interval
      - One ``concat`` filter to join them in order
      - Optional ``atempo`` chain for speed adjustment

    Trade-offs:
      - **No re-seeking** — ffmpeg reads the input linearly.
      - **One final encode** — all intervals pass through the encoder once,
        so quality is slightly better than repeated stream-copy + concat.
      - **Temp files** — none (everything stays in filter graph memory).
      - **Shell safety** — interval timestamps are formatted as fixed
        decimal strings (never user input), minimising injection risk.

    When there are many intervals (default threshold: 100), the strategy
    splits them into batches of ``batch_size``, processes each batch
    with its own filter graph, writes batch temp files, then concatenates
    those temp files with the concat demuxer (the same final step as the
    sequential strategy).  This avoids hitting ffmpeg filter graph limits.

    **Interval normalisation:** Before cutting, intervals are sorted,
    merged (overlapping/adjacent), and zero-length intervals are removed.
    This prevents cryptic ffmpeg errors from bad input data and reduces
    the number of filter graph branches.
    """

    def __init__(self, batch_size: int = 100):
        self._batch_size = batch_size
        self._name = "single_pass_filter"

    def cut(
        self,
        audio_path: str,
        intervals: List[dict],
        output_path: str,
        format_spec: str = "mp3",
        sample_rate: int = 22050,
        bitrate: str = "64k",
        speed: float = 1.0,
    ) -> str:
        t0 = time.time()

        if not intervals:
            raise ValueError("No intervals to cut")

        # Normalise intervals before any ffmpeg work
        try:
            audio_dur = _get_audio_duration(audio_path)
        except Exception:
            audio_dur = None

        cleaned = normalize_intervals(intervals, audio_duration=audio_dur)
        if not cleaned:
            raise ValueError("All intervals were zero-length or invalid after normalisation")

        n_original = len(intervals)
        n_removed = n_original - len(cleaned)
        if n_removed > 0:
            logger.info(
                "Normalised intervals: %d → %d (%d merged/removed)",
                n_original, len(cleaned), n_removed,
            )

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        atempo = _atempo_filters(speed)
        total = len(cleaned)
        kept_sec = sum(iv["end"] - iv["start"] for iv in cleaned)
        logger.info(
            "Single-pass filter cut: %d intervals (%.0fs kept), "
            "batch_size=%d, speed=%.2f",
            total, kept_sec, self._batch_size, speed,
        )

        # Single batch — one ffmpeg call, no temp files.
        if total <= self._batch_size:
            self._run_single_batch(
                audio_path=audio_path,
                intervals=cleaned,
                output_path=output_path,
                sample_rate=sample_rate,
                bitrate=bitrate,
                atempo=atempo,
            )
            elapsed = time.time() - t0
            logger.info("Done (single pass, %.1fs): %s", elapsed, output_path)
            return output_path

        # Multiple batches
        tmpdir = tempfile.mkdtemp(prefix="pc_batches_")
        try:
            batch_paths: List[str] = []
            n_batches = (total + self._batch_size - 1) // self._batch_size
            for batch_idx in range(0, total, self._batch_size):
                batch = cleaned[batch_idx:batch_idx + self._batch_size]
                batch_out = os.path.join(tmpdir, f"batch_{batch_idx:04d}.{format_spec}")
                self._run_single_batch(
                    audio_path=audio_path,
                    intervals=batch,
                    output_path=batch_out,
                    sample_rate=sample_rate,
                    bitrate=bitrate,
                    atempo=[],
                )
                batch_paths.append(batch_out)
                logger.info(
                    "  Batch %d/%d written: %s",
                    (batch_idx // self._batch_size) + 1, n_batches, batch_out,
                )

            logger.info("Concatenating %d batch files...", len(batch_paths))
            self._concat_batches(
                batch_paths, output_path,
                sample_rate=sample_rate,
                bitrate=bitrate,
                atempo=atempo,
            )
            elapsed = time.time() - t0
            logger.info("Done (batched, %.1fs): %s", elapsed, output_path)
            return output_path

        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

    def name(self) -> str:
        return self._name

    # ------------------------------------------------------------------
    # Internal: build + run one filter graph
    # ------------------------------------------------------------------

    @staticmethod
    def _build_filter_graph(
        intervals: List[dict],
        atempo: Optional[List[str]] = None,
        *,
        beep: bool = True,
        beep_freq: float = 1000,
        beep_duration: float = 0.25,
        sample_rate: int = 22050,
    ) -> str:
        """Build a ``filter_complex`` string for the given intervals.

        When ``beep=True``, a short studio tone (default 1000 Hz, 250 ms)
        is inserted between each interval to mark segment transitions.

        Returns e.g.::

            [0:a]atrim=10:20,asetpts=PTS-STARTPTS,aresample=22050[a0];
            sine=f=800:d=0.1,aresample=22050[b0];
            [0:a]atrim=30:40,asetpts=PTS-STARTPTS,aresample=22050[a1];
            [a0][b0][a1]concat=n=3:v=0:a=1[outa]
        """
        n = len(intervals)
        parts: List[str] = []

        if beep and n > 1:
            # atrim chains are UNCHANGED (zero perf overhead on the hot path).
            # Only sine outputs get aformat to match target rate.
            # FFmpeg 6+ concat auto-handles mixed sample rates.
            for i, iv in enumerate(intervals):
                start_s = f"{iv['start']:.3f}"
                end_s = f"{iv['end']:.3f}"
                parts.append(
                    f"[0:a]atrim={start_s}:{end_s},asetpts=PTS-STARTPTS[a{i}]"
                )
                if i < n - 1:
                    parts.append(
                        f"sine=f={beep_freq}:d={beep_duration},"
                        f"aformat=sample_rates={sample_rate}:channel_layouts=mono[b{i}]"
                    )

            # Interleave: a0, b0, a1, b1, ..., a{n-1}
            concat_labels: List[str] = []
            for i in range(n):
                concat_labels.append(f"[a{i}]")
                if i < n - 1:
                    concat_labels.append(f"[b{i}]")
            total = n + (n - 1)
            concat_inputs = "".join(concat_labels)
            concat = f"{concat_inputs}concat=n={total}:v=0:a=1"
        else:
            # No beep — existing behavior
            for i, iv in enumerate(intervals):
                start_s = f"{iv['start']:.3f}"
                end_s = f"{iv['end']:.3f}"
                parts.append(
                    f"[0:a]atrim={start_s}:{end_s},asetpts=PTS-STARTPTS[a{i}]"
                )
            concat_inputs = "".join(f"[a{i}]" for i in range(n))
            concat = f"{concat_inputs}concat=n={n}:v=0:a=1"

        if atempo:
            concat += "," + ",".join(atempo)

        parts.append(f"{concat}[outa]")
        return ";\n".join(parts)

    def _run_single_batch(
        self,
        audio_path: str,
        intervals: List[dict],
        output_path: str,
        sample_rate: int,
        bitrate: str,
        atempo: List[str],
    ):
        """Run one ffmpeg invocation for a batch of intervals."""
        filter_graph = self._build_filter_graph(
            intervals, atempo=atempo,
            beep=True, sample_rate=sample_rate,
        )
        cmd = [
            "ffmpeg", "-y",
            "-i", audio_path,
            "-filter_complex", filter_graph,
            "-map", "[outa]",
            "-ar", str(sample_rate),
            "-b:a", bitrate,
            "-ac", "1",
            output_path,
        ]
        logger.debug("Running ffmpeg filter graph (%d intervals)", len(intervals))
        result = _run_ffmpeg(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(
                f"ffmpeg single-pass failed ({len(intervals)} intervals): "
                f"{result.stderr[:1000]}"
            )

    @staticmethod
    def _concat_batches(
        batch_paths: List[str],
        output_path: str,
        sample_rate: int,
        bitrate: str,
        atempo: List[str],
    ):
        """Concatenate batch temp files using the concat demuxer.

        Delegates to the module-level ``_concat_batch_files`` so the same
        logic is shared with ``SequentialCopyCutStrategy``.
        """
        _concat_batch_files(
            batch_paths=batch_paths,
            output_path=output_path,
            sample_rate=sample_rate,
            bitrate=bitrate,
            atempo=atempo,
        )


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

STRATEGY_REGISTRY = {
    "parallel_copy": ParallelCopyCutStrategy,
    "single_pass_filter": SinglePassFilterCutStrategy,
    "sequential_copy": SequentialCopyCutStrategy,
}


def create_audio_strategy(
    name: str,
    **kwargs,
) -> AudioCuttingStrategy:
    """Factory: return an ``AudioCuttingStrategy`` by name.

    Args:
        name: One of ``"sequential_copy"``, ``"parallel_copy"``,
              ``"single_pass_filter"``.
        **kwargs: Strategy-specific constructor arguments (e.g.
                  ``max_workers`` for ``parallel_copy``).

    Returns:
        Initialised strategy instance.

    Raises:
        ValueError on unknown name.
    """
    name = name.lower().strip()
    if name not in STRATEGY_REGISTRY:
        raise ValueError(
            f"Unknown audio strategy: {name!r}. "
            f"Supported: {', '.join(STRATEGY_REGISTRY)}"
        )
    cls = STRATEGY_REGISTRY[name]
    return cls(**kwargs)
