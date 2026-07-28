# podcastcondensor

Condensing "Lord of Spirits" podcast episodes using DeepSeek LLM.

**Playlist:** https://www.youtube.com/playlist?list=PLZxCUWw2kdo1vAsOOOa3RwzwYvHbybjHR

## Architecture — cross-episode universe state → concept/claim audio cuts

Two-phase approach:

1. **Build universe state** — download SRTs, extract structured knowledge per episode via one DeepSeek call, merge into cross-episode knowledge base.
2. **Audio cutting** — pick concepts/claims from the universe state, collect their timestamp segments across all episodes, assemble into a single audio file.

### Phase 1: Download SRTs

Two sub-modes via `--yt-subs` flag:

| Mode | Flag | Speed | SRT Quality | Use case |
|------|------|-------|-------------|----------|
| YouTube subs | `--yt-subs` | ~30s/ep | Fragmented, overlapping chunks | Quick universe knowledge building |
| Whisper | *(default)* | ~30-45min/ep | Clean sentence-level chunks | Clean audio cutting boundaries |

YT subs are fine for LLM knowledge extraction (the model handles fragmentary text). Whisper is needed when cutting audio so segments land on clean boundaries.

### Phase 2: LLM Extraction (one DeepSeek call per episode)

Receives the **full timestamped SRT transcript** — each entry shown as
`[INDEX] START-END: TEXT` — and returns structured knowledge with
direct timestamp segments:

```json
{
  "summary": "2-3 paragraph narrative summary…",
  "concepts": [
    {"id": "divine-council", "title": "Divine Council", "summary": "…",
     "segments": [{"episode": 5, "start": 620.0, "end": 850.0}]}
  ],
  "entities": [
    {"id": "melchizedek", "title": "Melchizedek", "category": "person",
     "summary": "…", "segments": […]}
  ],
  "claims": [ … ],
  "scriptural_links": [ … ],
  "glossary": [ … ]
}
```

**Cost:** ~$0.03/episode. One call per episode.

### Phase 3: Merge into universe state

Each episode's extracted knowledge is merged into `output/universe_state.json`.
Items with the same `id` across episodes accumulate `segments` and
`episode_numbers` arrays — building up a cross-episode audio position map.

### Phase 4: Audio cutting

Extract segments for selected concepts/claims from the universe state, download
audio for those episodes, cut+concatenate with beep separators at 1.25× speed.

## Q&A episodes

Q&A episodes (detected by "Q&A" in the YouTube title) are **skipped by default**
— they jump between unrelated caller questions rather than developing coherent
themes. Use `--include-qa` to override.

Episodes known as Q&A: 18, 34, 38 (and any others with "Q&A" in title).

## Current status (July 2026)

**Universe state:** built for eps 1-40 (eps 1-28 with whisper, eps 31-40 whisper rebuild in progress).

**Audio cuts:** `build-minimal-theme` cuts one concept's clips. `build-master-cut`
does multi-theme anthology. Custom scripts used for category-specific cuts.

**Commands are the main workflow:**

| Command | Purpose | Status |
|---------|---------|--------|
| `build-universe --yt-subs` | Fast universe state (YT subs, ~30s/ep) | ✅ Working |
| `build-universe` | Full universe state (whisper, ~30min/ep) | ✅ Working |
| `build-master-cut` | Multi-theme audio anthology | ✅ Working |
| `build-minimal-theme` | Single-concept audio cut | ✅ Working |

**Legacy (abandoned):** The per-episode compression pipeline
(`process-playlist` compress mode, `prompts/compress_episode.txt`) is
unmaintained. The old two-call classifier pipeline
(`global_state.json` → `decisions.json`) is also legacy.

## Output

```
output/
  universe_state.json     # cross-episode knowledge base
  ep-NNN/
    source_subtitles.srt  # SRT (YT or whisper)
    global_state.json     # per-episode DeepSeek extraction
```

## Commands

### `build-universe`

```bash
# Fast path (YT subs) — for knowledge building only
python3 -m podcastcondensor build-universe [PLAYLIST_URL] \
  --start N --end N --yt-subs

# Full path (whisper) — for clean audio cutting
python3 -m podcastcondensor build-universe [PLAYLIST_URL] \
  --start N --end N
```

Resumable: skips episodes with existing `global_state.json`. Runs Phase 1
(download) + Phase 2 (DeepSeek extraction) per episode.

### `build-master-cut`

```bash
python3 -m podcastcondensor build-master-cut [PLAYLIST_URL] \
  --start N --end N \
  --target-duration 3600 \
  --output my_cut.mp3
```

Does: download → universe state → theme extraction → segment selection → audio cut.

### `build-minimal-theme`

```bash
python3 -m podcastcondensor build-minimal-theme [THEME_ID] \
  [PLAYLIST_URL]
```

Cuts all segments for one concept across all episodes.

## Data sizes

| Metric | Per episode | 10 eps | 140 eps |
|--------|-------------|--------|---------|
| Cleaned transcript | ~113K chars / ~28K tokens | 1.1M chars | 15.9 MB |
| SRT entries | ~1500-2200 | 15-22K | ~250K entries |
| Universe state items | ~30-50 concepts+claims | ~300-500 | ~4000-7000 |

## Required

- DeepSeek API key in `ANTHROPIC_AUTH_TOKEN` or `DEEPSEEK_API_KEY`
- ffmpeg
- yt-dlp
- faster-whisper (for whisper mode)

## Prompts

| File | Used by | Description |
|------|---------|-------------|
| `prompts/global_state.txt` | Phase 2 (extraction) | Full transcript → structured knowledge with timestamp segments |
| `prompts/compress_episode.txt` | *(legacy)* | Archived old per-episode compression prompt |
| `prompts/classify_raw.txt` | *(legacy)* | Archived old classifier prompt |
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
