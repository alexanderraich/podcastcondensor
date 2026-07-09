# podcastcondensor

Condensing "Lord of Spirits" podcast episodes using DeepSeek LLM.

**Playlist:** https://www.youtube.com/playlist?list=PLZxCUWw2kdo1vAsOOOa3RwzwYvHbybjHR

## Architecture (3 phases, two modes)

### Phase 1: Scan/Load
For `build-universe`: downloads MP3 + SRT per episode. Prefers YouTube auto-subs;
falls back to faster-whisper.

For `build-minimal-theme`: scans `output/ep-NNN/` for existing artefacts —
no YouTube calls, no downloads, no whisper. Takes ~0s.

**Artefact:** `source_subtitles.srt` + `.mp3` per episode dir.

### Phase 2: Global State (one DeepSeek call per episode)
Receives the **timestamped SRT transcript** — each entry shown as
`[INDEX] START-END: TEXT` — and returns structured knowledge with **direct
timestamp segments** (no word indices, no post-conversion).

```json
{
  "summary": "2-3 paragraph narrative",
  "concepts": [
    {"id": "divine-council", "title": "Divine Council", "summary": "...",
     "segments": [{"start": 600.0, "end": 669.0}]}
  ],
  "entities": [...],
  "claims": [...],
  "scriptural_links": [...],
  "glossary": [...]
}
```

**Validation:** returned timestamps are snapped to SRT entry boundaries.
Segments <3s are dropped (hallucination guard). Mid-sentence boundaries
are logged as warnings.

The output is **merged** into the universe state, accumulating across episodes.

### Phase 3: Thematic Cut (for producing audio)
Takes the universe state for one theme, resolves candidate segments, then
sends them to DeepSeek with surrounding transcript context for selection
and boundary refinement. No target duration, no budget — the LLM decides
what's needed.

**Artefact:** `.mp3` file with beep-separated segments.

### Two modes

| Mode | Description |
|------|-------------|
| `build-universe` | Runs Phase 1 + 2 on a range of episodes. Builds/extends `universe_state.json`. No audio produced. |
| `build-minimal-theme` | Single-theme audio cut. Resolves segments from existing universe state, runs Phase 3 selection, assembles audio. |

## Data sizes (determines everything)

| Metric | Per episode | 29 eps | 140 eps |
|--------|-------------|--------|---------|
| Cleaned transcript | ~113K chars / ~28K tokens | 3.3M chars | 15.9 MB |
| SRT entries | ~1500-2200 | ~50K entries | ~250K entries |
| Timestamped format | ~170K chars / ~42K tokens | 4.9M chars | 23.8 MB |

**Batching is impossible:** even 2 episodes exceed 64K context with prompt
overhead. Phase 2 must be one DeepSeek call per episode.

## Direct Timestamps (why no word_ranges)

Phase 2 previously used `word_ranges` (word indices) which the LLM had to
estimate, then a Python post-process converted them to timestamps via word
counting. The LLM was poor at word-level indexing, producing inaccurate
segments that cut mid-sentence or covered wrong content.

**Current approach:** The LLM sees actual SRT timestamps in the input and
returns timestamps directly. No indirection. The only post-processing is
`_validate_segments()` which snaps to entry boundaries and drops garbage.

**Known limitation — Phase 2 over-tagging:** The LLM still includes
*tangential mentions* of a concept (e.g. "we're not teaching angelology"
gets tagged under `divine-council`). The Phase 3 selection call filters
this out — it shows each candidate segment with surrounding transcript
context and asks the LLM to classify as DEFINITION / APPLICATION / OFF-TOPIC.

## `build-universe` command

```bash
python3 -m podcastcondensor build-universe [PLAYLIST_URL] --start 1 --end N
```

Rebuilds the universe state from scratch. Requires existing SRT files in
`output/ep-NNN/source_subtitles.srt`. One DeepSeek call per episode.
Cost: ~$0.03/episode.

## `build-minimal-theme` command

```bash
python3 -m podcastcondensor build-minimal-theme THEME_ID "unused" \
  --output output/my_cut.mp3
```

Single theme, no target duration. The LLM decides what to keep.
Cost: ~$0.03 (one DeepSeek call for selection).
Scans `output/ep-NNN/` for existing downloads — no YouTube or whisper needed.

List available themes:
```bash
python3 -c "import json; [print(t['id']) for t in json.load(open('output/_themes.json'))]"
```

### Full workflow

```bash
# 1. Build/rebuild universe state (Phase 2, one DeepSeek call per episode)
python3 -m podcastcondensor build-universe [URL] --start 1 --end 29

# 2. Extract themes (one call over universe state, cached to output/_themes.json)
#    Already done if you run build-universe — cached by build-minimal-theme

# 3. List available themes
python3 -c "import json; [print(t['id']) for t in json.load(open('output/_themes.json'))]"

# 4. Build a single-theme audio cut
python3 -m podcastcondensor build-minimal-theme body-and-materiality "unused"

# 5. Doctor check
python3 -m podcastcondensor doctor --check
```

## Output

```
output/
  universe_state.json       # accumulated knowledge across episodes
  _themes.json              # cached theme extraction (reusable)
  ep-NNN/
    source_subtitles.srt    # raw downloaded SRT
    global_state.json       # Phase 2 structured knowledge + segments
    decisions.json          # old Phase 3 classifier output (not used by master cut)
    condensed_*.mp3         # old per-episode pipeline output (not used by master cut)
```

## Required

- DeepSeek API key in `ANTHROPIC_AUTH_TOKEN` or `DEEPSEEK_API_KEY`
- ffmpeg

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
