# podcastcondensor

Condensing "Lord of Spirits" podcast episodes using DeepSeek LLM.

**Playlist:** https://www.youtube.com/playlist?list=PLZxCUWw2kdo1vAsOOOa3RwzwYvHbybjHR

## Pipeline (4 phases)

Phase 1 downloads the raw SRT. The **global state** phase runs on the full
transcript and produces both the episode outline AND the structured universe
knowledge in a single DeepSeek call. The **raw classifier** receives the raw
SRT file verbatim, the episode outline, and the universe state as context —
one DeepSeek call returns keep/drop per SRT cue entry.

| # | Phase | Artefact | Key file |
|---|-------|----------|----------|
| 1 | **Download** | `source_subtitles.srt` + `.mp3` | `downloader.py` |
| 2 | **Global state** | `global_state.json` (outline + knowledge) | `global_state.py` + `universe_state.py` |
| 3 | **Classify raw** | `decisions.json` (per-entry decisions) | `classify_raw.py` + `prompts/classify_raw.txt` |
| 4 | **Audio cutting** | `condensed_*.mp3` | `audio_strategies.py` + `intervals.py` |

**Resumable:** every phase checks for its artefact before running. If the
artefact exists, the phase is skipped. Interrupted runs pick up where they
left off.

**No timestamp bugs:** Audio cuts use real SRT entry timestamps directly.
The LLM decides keep/drop on raw SRT entries; kept entries go straight
to the interval builder which clusters by time gap and pads.

Phase 2 merges knowledge into the universe state automatically (both in
build-universe and process-playlist modes). No separate extraction phase.

## Universe State (rolling structured knowledge)

Each episode's **global state** phase produces a structured extraction via one
DeepSeek call over the full cleaned transcript. The state accumulates across
episodes:

- `episode_summaries` — 2-3 paragraph narrative per episode
- `entities` — people, places, theological categories (with episode provenance)
- `concepts` — theological concepts and terms
- `claims` — specific doctrinal positions or arguments
- `scriptural_links` — biblical references cited and how they're used
- `glossary` — episode-defined terms (**episode-local**, not universe-wide)

Merged across episodes with dedup by stable ID. The classifier gets the
accumulated universe state as context so it can avoid keeping content
already covered by prior episodes.

**Two modes:**
- `build-universe` — SRT download → clean → **phase 2** (global state) → merge.
  No classification/audio.
- `process-playlist` — full 4-phase pipeline. Phase 2 keeps the state current
  automatically.

## Required

- DeepSeek API key in `ANTHROPIC_AUTH_TOKEN` or `DEEPSEEK_API_KEY`
- ffmpeg

## Commands

```bash
# Build universe state from episodes 1-21 (SRT + one DeepSeek call per episode)
python3 -m podcastcondensor build-universe [PLAYLIST_URL] --start 1 --end 21

# Process episodes 22+ with universe state (full pipeline)
python3 -m podcastcondensor process-playlist [PLAYLIST_URL] \
  --state-file output/universe_state.json --start 22

# Doctor check
python3 -m podcastcondensor doctor --check
```

## Output

```
output/
  universe_state.json       # accumulated universe knowledge across episodes
  ep-NNN/
    source_subtitles.srt    # phase 1 — raw downloaded SRT
    global_state.json       # phase 2 — outline + structured knowledge
    decisions.json          # phase 3 — per-entry keep/drop decisions
    condensed_*.mp3         # phase 4 — final audio with beeps
    stats.json              # post-run stats
```

## Architecture

```
                        build-universe              process-playlist
                        ─────────────              ────────────────
SRT → clean ──→ Global state ──→ merge into state
                  (phase 2)         │
                        (universe state used as context during classification)
                                    │
SRT → clean → Global state → classify raw → cut audio
               (phase 2)     (3)          (4)
```

## Master Cut — Cross-Episode Thematic Anthology

The `build-master-cut` subcommand produces a single audio file from all ~140
episodes containing only the core thematic content, at configurable duration
(default 3.5h).

### Pipeline (6 phases)

| # | Phase | Artefact | Key file |
|---|-------|----------|----------|
| 1 | **Download** | audio + SRT per episode | `download_pool.py` |
| 2 | **Global state** | `global_state.json` with `word_ranges` per item | `global_state.py` |
| 3 | **Universe merge** | `universe_state.json` with timestamp segments | `universe_state.py` |
| 4 | **Theme extraction** | `themes.json` (one DeepSeek call) | `theme_extraction.py` |
| 5 | **Selection** | ordered playlist within time budget | `master_cut.py` |
| 6 | **Audio assembly** | `master_cut.mp3` with dual beeps | `master_cut.py` |

### Audio Position Architecture (critical design decision)

**No keyword-based back-mapping from concepts to audio.**

Phase 2 produces `word_ranges` on each concept/entity/claim alongside its
definition. A post-process converts word indices to timestamps via the SRT
entries. The universe state stores:

```json
{
  "id": "divine-council",
  "title": "Divine Council",
  "segments": [
    {"episode": 1, "start": 600, "end": 669, "word_start": 1700, "word_end": 1900}
  ]
}
```

**Known limitation — Phase 2 over-tagging:** The LLM's word_ranges include
*tangential mentions* of a concept (e.g. "we're not teaching angelology"
gets tagged under `divine-council`). Solutions:
1. When resolving theme segments, **use concepts only** — skip entities,
   claims, scriptural_links, and glossary (which contain host intros and
   generic references). See `resolve_theme_segments_from_state()`.
2. For tighter audio, **run Phase 3 classifier** with a stricter prompt
   (target 80-90% drop) and use its keep/drop decisions as a filter on
   the concept word_ranges. Without classifier, expect 30-50% content
   to be intros, previews, and meta-talk.

### Phase 3 Classifier

The classifier is required to produce listenable output. It receives the
raw SRT, the episode outline, and the universe state as context. The
default prompt targets 50% compression; for master cut purposes the
prompt should target 80-90% drop — only keep substantive exegesis,
historical context, and doctrinal claims.

### Master Cut Selection Algorithm

The `select_segments_for_master_cut()` function:
1. Sorts themes by importance descending.
2. Allocates a time budget proportional to each theme's importance weight.
3. Takes segments chronologically within each theme as a **contiguous block**
   (not interleaved across themes).
4. Ensures at least `min_segs` per theme (scales with target duration:
   2 for 10min, 5 for 42min, 10 for 3.5h).
5. Skips themes that can't fill `min_segs` — their budget redistributes.
6. Single beep within a theme, triple beep between themes.

### Data sizes (determines everything)

| Metric | Per episode | 29 eps | 140 eps |
|--------|-------------|--------|---------|
| Cleaned transcript | ~113K chars / ~28K tokens | 3.3M chars | 15.9 MB |
| SRT raw | ~480K chars | 13.9 MB | 67 MB |

**Batching is impossible:** even 2 episodes fill 56K tokens — beyond the
64K DeepSeek context with any prompt overhead. Phase 2 must be one
DeepSeek call per episode. This is the binding constraint.

### Why Phase 2 must be per-episode (and no keyword grep)

The naive approach would be: universe state stores only concept names →
keyword-grep all SRTs to find where each gets mentioned → map to
timestamps. This fails because YT auto-captions transcribe vocabulary
unreliably ("divine council" → "the divine counsel", "theosis" →
absent). Even whisper has recognition variation across episodes.

The correct flow: whisper transcribe → DeepSeek reads the full cleaned
transcript → **DeepSeek itself** identifies where each concept is
discussed (via word indices) → post-process converts to exact audio
timestamps via SRT entries. No text search, no vocabulary guesswork.

## `build-master-cut` command

```bash
python3 -m podcastcondensor build-master-cut \
  [PLAYLIST_URL] \
  --state-file output/universe_state.json \
  --output master_cut.mp3 \
  --target-duration 12600 \
  [--force-whisper] \
  [--parallel-downloads 4]
```

### Dual beep system

- **Single beep** (250ms, 1000Hz): between segments within the same theme
- **Triple beep** (3×250ms, 250ms gaps): between different themes
Generated via ffmpeg sine + concat demuxer. No filter_complex.

## Transcription (faster-whisper) — OOM prevention

On the 8 GB RAM / 6 GB VRAM WSL2 machine, transcription is the most crash-prone phase. Defaults are set for memory-conservative operation:

| Setting | Default | Why |
|---------|---------|-----|
| `whisper_beam_size` | `1` | Beam 3 keeps 3× decoder state |
| `whisper_vad_filter` | `False` | VAD pre-scan doubles peak GPU memory on 2.75h audio |
| `whisper_condition_on_prev` | `False` | Text cache grows unbounded on long audio — disabling prevents memory leak |

Environment: `OMP_NUM_THREADS=2` and `MKL_NUM_THREADS=2` are set before any C library import to prevent OpenMP thread explosion.

All three are configurable via `config.py` under the `# Transcription` section. To run on a machine with 16+ GB RAM / 8+ GB VRAM:
```python
whisper_beam_size = 5
whisper_vad_filter = True
whisper_condition_on_prev = True
```

**Diagnostics:** Every `logger.info/warning/error` from `transcribe.py` is automatically tee'd (via `_DiagLogHandler`) to `output/ep-NNN/_transcribe_diag.log` with immediate fsync. Additionally a watchdog daemon thread heartbeats every 30s during the blocking `model.transcribe()` call, with GPU memory snapshots every 60s and system memory every 120s. If the Python process is OOM-killed, the diag log survives up to the last written line.

**DON'T** use `os.dup2` / fd redirection for diagnostics — it can corrupt the terminal state of the parent Claude session on crash.
