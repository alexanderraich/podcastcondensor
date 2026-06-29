# Plan: Fix Pipeline Transcription Crashes (Ep 28+)

## Context of this document

You (the user) said the pipeline "starts to crash from ep28 onwards" and
runtimes are "going up from 2:30".  Before implementing any fix, I analysed
what is known and what is not.  This file captures the analysis + proposed
fix so you can resume from here next time without re-doing the investigation.

---

## 0. Approach: Instrument first, fix second

Instead of guessing the root cause, the plan is:

1. **Add crash-safe diagnostics** (fsynced `_transcribe_diag.log`) that survive an OOM/segfault
2. **Run ep-028** and read the diagnostics to determine the actual failure mode
3. **Implement the targeted fix** based on real data, not guesswork

The instrumented code logs: GPU VRAM before/after, audio size/duration, when `model.transcribe()` is called, when the generator returns, and progress every 100 segments — all fsynced to disk on every write.

## 1. What We Actually Know

### System state (at investigation time, 2026-06-29)

| Measurement | Value |
|---|---|
| GPU VRAM | 6144 MiB total, **5471 MiB free** |
| System RAM | 7.7 GiB total, **6.0 GiB free** |
| OOM killer logs in dmesg | **None** |
| Ep-028 audio file validity | Valid MP3 (48 kHz stereo, no ffprobe errors) |
| Ep-028 actual duration (from ffprobe) | 9954 s = **2h45m** |
| Ep-028 YouTube-reported duration | 188m42s = **3h08m** |
| Current working tree diff in transcribe.py | `beam_size=5` → `beam_size=3` only |

### Episode processing history (relevant range: ep-023 to ep-028)

All processed with the same faster-whisper code (beam_size=5).

| Ep | Duration | SRT produced? | Notes |
|----|----------|--------------|-------|
| 023 | — | ✅ Jun 24 | |
| 024 | — | ✅ Jun 24 (42 min later) | |
| 025 | — | ✅ Jun 24 (17 min later) | |
| 026 | 2h09m | ✅ Jun 24 (10 min later) | |
| 027 | **3h46m** | ✅ Jun 29 (~9 min later) | **Longest episode — succeeded** |
| 028 | 2h45m / 3h08m | **❌ Failed** | Audio downloaded, no SRT |

### What the ep-028 failure looks like

```
output/ep-028/
  I3SQ0KEXYhE.mp3        (80 MB — valid MP3)
  _step.txt               (contains "PHASE1_AUDIO_DOWNLOADED")
  [no source_subtitles.srt]
  [no _crash.log]
```

The `_step.txt` shows transcription was started but never completed.
There is **no Python traceback** (no `_crash.log`), which means either:

- The process was killed by the OS (OOM killer, segfault in ctranslate2,
  WSL crash) before Python could catch the exception
- The user killed it manually (e.g. from the terminal)
- faster-whisper hung indefinitely (no timeout in the transcribe call)

### The `_step.txt` telemetry system

The `_step.txt` / `_crash.log` system was added in the **working tree**
(pipeline.py diff, not yet committed).  Ep-023 through ep-027 were processed
with the committed code (no telemetry).  Ep-028 is the first episode run
with the telemetry-enabled code.  The telemetry changes are purely
additive log writes — they cannot cause the transcription to fail.

---

## 2. What We DON'T Know (and why we can't easily determine it)

| Question | Why we can't answer |
|---|---|
| Did the transcription crash, hang, or get killed? | No traceback, no crash log — just the absence of the success marker |
| Is it specific to ep-028's audio content? | Can't safely re-run transcription (might crash the system again) |
| Is it a WSL GPU driver state issue after 27 episodes? | Would need to reproduce from a fresh WSL session |
| Does faster-whisper segfault on this particular MP3? | Would need to run it with GDB or a segfault handler |

**Key puzzle:** Ep-027 (3h46m, the longest of all) succeeded.  Ep-028
(2h45m actual ffprobe duration) failed.  This rules out a simple
"audio too long → GPU OOM" explanation.  The cause could be:

- An audio-specific bug in faster-whisper / ctranslate2 (codec quirk,
  unusual waveform, VAD edge case)
- WSL GPU driver state degraded after the long ep-027 transcription
  (known WSL2 issue — GPU state can become unreliable after heavy use)
- An intermittent system issue
- User action (manual kill)
- A combination of factors

---

## 3. Proposed Fix: Chunked Transcription

Regardless of the exact cause, **chunked transcription** makes the
pipeline more robust against ALL plausible failure modes:

| Failure mode | How chunking helps |
|---|---|
| GPU memory pressure (even if not primary cause) | Each chunk uses less peak VRAM |
| Audio-specific bug (corrupt section) | Only that chunk fails (not the whole episode) — or we skip the bad chunk |
| WSL GPU state degradation | Shorter GPU workloads per chunk, lower cumulative stress |
| User impatience / no progress signal | Progress logging per chunk (user can see it's working) |
| faster-whisper hang | Timeout per chunk (can be added) |
| Process gets killed mid-way | Resumable at chunk granularity (not all-or-nothing) |

### Design

**Chunk method:** Split the MP3 into segments using ffmpeg's `-ss` / `-t`
(seek-and-cut) rather than re-encoding.  Each chunk is extracted as a temp
file before being fed to faster-whisper.

**Chunk size:** 20 minutes (1200 seconds).  Chosen because:
- Small enough to guarantee non-problematic GPU load
- Large enough that overhead of chunking + model reload is negligible
- A 3h episode produces 9 chunks — manageable

**Offset adjustment:** Each chunk's segment timestamps get the chunk
start offset added so the final SRT has correct timestamps.

**Resumability:**
- Check `source_subtitles.srt` before starting (already works — checkpoint)
- Temp chunk files go in `output_dir` with a `_chunk_` prefix, cleaned up
  on success or failure
- If interrupted mid-chunk, re-running starts from scratch (no SRT = no
  checkpoint hit)

### Implementation: changes to `src/podcastcondensor/transcribe.py`

Two new helper functions + modified `transcribe_audio()`:

```python
def _get_audio_duration(audio_path: str) -> Optional[float]:
    """Use ffprobe to get audio duration in seconds. Returns None on failure."""

def _transcribe_chunked(
    audio_path: str,
    output_srt: str,
    model: WhisperModel,
    chunk_duration: float = 1200.0,
) -> int:
    """Transcribe long audio in chunks, merge results, write SRT."""
```

The `transcribe_audio()` function signature stays the same (backward
compatible).  It gains an internal branch:

```python
def transcribe_audio(audio_path, output_dir, model_size="base"):
    if os.path.exists(output_srt):
        return output_srt  # checkpoint hit

    duration = _get_audio_duration(audio_path)
    if duration and duration > CHUNK_THRESHOLD:
        count = _transcribe_chunked(audio_path, output_srt, model)
    else:
        count = _transcribe_monolithic(audio_path, output_srt, model)

    logger.info("Transcription complete: %d segments -> %s", count, output_srt)
    return output_srt
```

No changes to `pipeline.py`, `config.py`, or `playlist_pipeline.py`
required — the function signature doesn't change.

### Fallback: YouTube SRT download (optional additive fix)

The `downloader.py` already has a working `download_subtitles()` function
that can pull YouTube's auto-generated captions.  It's just never called
from the pipeline.  Could be wired in as:

1. A config flag `--prefer-youtube-subs` that bypasses whisper entirely
2. A fallback: try whisper first, if it fails, download YouTube SRT

This is lower priority since auto-captions are worse for the specialized
theological vocabulary, but it gives a zero-GPU path for very long episodes.

---

## 4. Recovery / Resume Instructions

Once the fix is applied:

```bash
# Re-run from the failing episode — audio is already downloaded,
# the pipeline will skip to transcription with the new chunked code:
python3 -m podcastcondensor process-playlist \
  https://www.youtube.com/playlist?list=PLZxCUWw2kdo1vAsOOOa3RwzwYvHbybjHR \
  --state-file output/universe_state.json \
  --start 28

# The pipeline is resumable: phase 2+ will use existing artifacts
# if they were completed from the partial run.
```

### All remaining episodes at a glance (28+)

| Ep | YouTube duration | Risk note |
|----|-----------------|-----------|
| 28 | 3h08m | ❌ Already failed — first to test |
| 29 | 2h33m | — |
| 30 | 3h06m | Long |
| 31 | 2h54m | — |
| 32 | 2h58m | — |
| 33 | 1h57m | — |
| 34 | 3h03m | Long |
| 35 | 2h25m | — |
| 36 | 2h09m | — |
| 37 | 2h35m | — |
| 38 | 2h36m | — |
| 39 | 2h45m | — |
| 40 | 3h27m | Long |
| 41 | 2h50m | — |
| 42 | 3h31m | Long |
| 43 | 3h53m | Longest |
| 44 | 2h58m | — |
| 45 | 2h33m | — |

With chunking, none of these should be problematic.

---

## 5. Verification (safe to run)

```bash
# 1. Doctor check (no GPU needed)
python3 -m podcastcondensor doctor --check

# 2. Try ep-028 only (the known failure)
python3 -m podcastcondensor process-playlist \
  [PLAYLIST_URL] --state-file output/universe_state.json \
  --start 28 --end 28

# 3. If success, resume the batch
python3 -m podcastcondensor process-playlist \
  [PLAYLIST_URL] --state-file output/universe_state.json \
  --start 28
```

---

## 6. Quick alternative: just use a smaller whisper model

Before implementing chunking, you could test whether simply switching to
`"tiny"` model fixes ep-028:

```bash
# Run with tiny model (config needs --whisper-model or env var)
python3 -m podcastcondensor process-playlist \
  ... --start 28 --end 28
```

The `"tiny"` model uses ~1/3 the resources of `"base"` with slightly
worse accuracy — but for podcast transcription going through an LLM
classifier, the accuracy difference is usually negligible.

This is a 2-minute config change vs. a coding change.  Worth trying first
if you want to narrow down the root cause.
