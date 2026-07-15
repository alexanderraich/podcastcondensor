# podcastcondensor

Condensing "Lord of Spirits" podcast episodes using DeepSeek LLM.

**Playlist:** https://www.youtube.com/playlist?list=PLZxCUWw2kdo1vAsOOOa3RwzwYvHbybjHR

## Architecture — per-episode aggressive compression

The project tried a "universe state" approach (cross-episode knowledge →
thematic cuts). It failed — mid-thought cuts, unreliable timestamps, brittle
pipeline. **Current approach:** Condense each episode independently with a
single strict LLM call.

### Phase 1: Download + Transcribe

Downloads MP3 + SRT per episode. Prefers YouTube auto-subs; falls back to
faster-whisper. Same as before — see "Transcription OOM prevention" below.

**Artefact:** `source_subtitles.srt` + `.mp3` per episode dir.

### Phase 2: LLM Compression (single DeepSeek call per episode)

Receives the **full timestamped SRT transcript** — each entry shown as
`[INDEX] START-END: TEXT` — and returns the episode's core idea plus
direct timestamp segments for the most relevant passages.

```json
{
  "core_idea": "One sentence describing the episode's main argument",
  "segments": [
    {"start": 620.0, "end": 850.0, "reason": "Core theological argument: the divine council as a template for Israel's governance"},
    {"start": 1800.0, "end": 2100.0, "reason": "Historical context on Ugaritic texts and their relation to Psalm 82"}
  ]
}
```

**Strategy:** The show has 3 roughly-equal halves (thirds). The LLM
identifies the core idea, then picks the 1-2 most important continuous
passages from each half that best convey that idea. The prompt guides
this — no hard caps in code, the LLM decides how many segments are
appropriate.

**Cardinal rules (enforced via prompt, not code):**
- **Default to DROP.** The LLM must justify keeping something.
- **Complete thoughts only.** Start where the speaker begins explaining,
  end where the idea wraps up. Never mid-sentence or mid-argument.
- **3-halves structure.** Pick 1-2 passages per half that build the core
  idea. If a half is thin (banter, repetition), skip it — don't pad.
- **No trailing fragments.** Widen to full sentence start/end if a
  boundary would cut mid-thought.

**Validation (code level):** Returned timestamps are snapped to SRT entry
boundaries. Segments <3s dropped (hallucination guard). Mid-sentence
boundaries are logged as warnings (not rejected). No maximum segment
count — the LLM decides.

**Cost:** ~$0.03/episode. One call, no state, no post-processing.

### Phase 3: Audio Cut

Extracts the selected segments from the source MP3, concatenates with beep
separators, applies 1.25× speed. Same audio cutting as before — see
`audio_strategies.py`.

## Why this approach works

| Problem | Old approach | New approach |
|---------|-------------|--------------|
| Mid-thought cuts | LLM estimated word indices or theme boundaries; always wrong | LLM sees full timestamps, told to find complete thought boundaries |
| Too much content kept | Prompt said "move through full arc" | Prompt says "3 halves, 1-2 substantive passages each" |
| Complex state | Two LLM calls + universe state + JSON merging | One LLM call, no cross-episode state |
| Brittle | global_state → decisions.json → intervals | Direct segment ranges → intervals |
| No hard caps | Code overrode LLM with arbitrary limits | Code only validates (snap to SRT, hallucination guard); LLM decides what to keep |

## Current status (July 2026)

**Working:** Per-episode compression pipeline. Tested on Ep. 29 (Monster
Manual, 2h50m → ~20m, 7 segments across all 3 halves).

- `source_subtitles.srt` + MP3 exist for eps 1-29 (YouTube subs or whisper)
- Phase 2 compression tested and working with `compress_episode.txt` prompt
- Audio cutting works (sequential copy strategy, beep separators, 1.25× speed)
- Artefact skipping implemented: checks `output/ep-NNN/compressed.json` before
  hitting YouTube or the LLM

**Not working / unmaintained:** The universe-state approach (`build-universe`,
`build-minimal-theme`, `build-master-cut`) is abandoned. Code still in repo
but not the main workflow.

**Next:** Run compression across remaining episodes, iterate on the prompt
as needed based on output quality.

## Data sizes

| Metric | Per episode | 29 eps | 140 eps |
|--------|-------------|--------|---------|
| Cleaned transcript | ~113K chars / ~28K tokens | 3.3M chars | 15.9 MB |
| SRT entries | ~1500-2200 | ~50K entries | ~250K entries |
| Timestamped format | ~170K chars / ~42K tokens | 4.9M chars | 23.8 MB |

## `process-playlist` command (main workflow)

```bash
python3 -m podcastcondensor process-playlist [PLAYLIST_URL] --start 1 --end 29
```

Runs all three phases per episode. Fully resumable: skips episodes whose
`condensed_*.mp3` already exists (or whose SRT exists if only Phase 1+2 done).

**Cost:** ~$0.03/episode (one DeepSeek call). No universe state needed.

## `doctor` command

```bash
python3 -m podcastcondensor doctor --check
```

## Legacy commands (unmaintained)

These were part of the abandoned universe-state approach. They still exist
in the repo but are not the main workflow:

| Command | Description |
|---------|-------------|
| `build-universe` | Builds cross-episode universe state (Phase 2 only, no audio) |
| `build-minimal-theme` | Single-theme audio cut from universe state |
| `build-master-cut` | Cross-episode thematic anthology |

## Output

```
output/
  ep-NNN/
    source_subtitles.srt    # raw downloaded SRT
    compressed.json         # Phase 2 output: core idea + 1-2 timestamp segments
    condensed_epNNN.mp3     # final audio (1-2 segments, beep-separated, 1.25x)
    stats.json              # compression statistics (legacy path only)
    global_state.json       # (legacy) from universe-state approach
    decisions.json          # (legacy) from universe-state approach
```

## Required

- DeepSeek API key in `ANTHROPIC_AUTH_TOKEN` or `DEEPSEEK_API_KEY`
- ffmpeg

## Prompts

| File | Used by | Description |
|------|---------|-------------|
| `prompts/compress_episode.txt` | Phase 2 (compression) | Full transcript → core idea + 3-half timestamped segments |
| `prompts/classify_raw.txt` | *(legacy)* | Archived old classifier prompt |
| `prompts/global_state.txt` | *(legacy)* | Archived universe-state prompt |
| `prompts/extract_themes.txt` | *(legacy)* | Archived theme extraction prompt |

## Transcription (faster-whisper) — OOM prevention

On the 8 GB RAM / 6 GB VRAM WSL2 machine, transcription is the most crash-prone
phase. Defaults are set for memory-conservative operation:

| Setting | Default | Why |
|---------|---------|-----|
| `whisper_beam_size` | `1` | Beam 3 keeps 3× decoder state |
| `whisper_vad_filter` | `False` | VAD pre-scan doubles peak GPU memory on 2.75h audio |
| `whisper_condition_on_prev` | `False` | Text cache grows unbounded on long audio |

Environment: `OMP_NUM_THREADS=2` and `MKL_NUM_THREADS=2` are set before any
C library import to prevent OpenMP thread explosion.

All three are configurable via `config.py` under the `# Transcription` section.

**Diagnostics:** Every `logger.info/warning/error` from `transcribe.py` is
automatically tee'd to `output/ep-NNN/_transcribe_diag.log` with immediate
fsync. A watchdog daemon heartbeats every 30s during `model.transcribe()`,
with GPU memory snapshots every 60s and system memory every 120s.

**DON'T** use `os.dup2` / fd redirection for diagnostics — it can corrupt
the terminal state of the parent Claude session on crash.
