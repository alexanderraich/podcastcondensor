# podcastcondensor

Condensing "Lord of Spirits" podcast episodes using DeepSeek LLM.

## Core decisions (July 2026)

- **Only whisper SRTs in git.** YouTube subtitles are unreliable (fragmented,
  auto-generated artifacts). All SRT files in git are whisper-produced.
  Pipeline modes default to whisper always — no YT sub fallback.
- **Only `source_subtitles.srt` tracked per episode.** Everything else
  (`global_state.json`, `compressed.json`, `decisions.json`, `stats.json`,
  `universe_state.json`, `_themes.json`) is derivable from the SRT + LLM
  calls and is NOT version-controlled.
- **`universe_state.json` is disposable.** It can be reconstructed from
  scratch with `build-universe` using the version-controlled SRTs. Not
  tracked in git after July 2026 cleanup.
- **`_themes.json` is dead.** Was a stale cache from Jul 9 (eps 1-28 era).
  Removed from git, added to gitignore. All callers fall back to fresh
  DeepSeek extraction.

## Pipelines

Four entry points:

### `build-universe` — cross-episode knowledge extraction

Downloads audio, transcribes with whisper, runs one DeepSeek call per
episode for structured knowledge (concepts, entities, claims, etc.),
merges into cross-episode `universe_state.json`.

| Produces | Tracked? |
|---|---|
| `output/ep-NNN/source_subtitles.srt` | ✅ git |
| `output/ep-NNN/global_state.json` | ❌ gitignored |
| `output/universe_state.json` | ❌ gitignored |

```bash
python3 -m podcastcondensor build-universe <PLAYLIST_URL> --start N --end N
```

### `process-playlist` — single-episode compression

Two sub-modes:
- **Compress (default, `skip_global_state=True`):** one DeepSeek call →
  `compressed.json` with 1-2 direct timestamp segments. Fast, $0.01/ep.
- **Legacy (`--use-global-state`):** two-call pipeline (`global_state` +
  `classify_raw`). Produces `global_state.json` + `decisions.json` +
  `stats.json`. More expensive, richer structure.

| Produces | Tracked? | Mode |
|---|---|---|
| `output/ep-NNN/source_subtitles.srt` | ✅ git | Both |
| `output/ep-NNN/compressed.json` | ❌ gitignored | Compress |
| `output/ep-NNN/global_state.json` | ❌ gitignored | Legacy |
| `output/ep-NNN/decisions.json` | ❌ gitignored | Legacy |
| `output/ep-NNN/stats.json` | ❌ gitignored | Legacy |

```bash
python3 -m podcastcondensor process-playlist <PLAYLIST_URL> --start N --end N
```

### `build-master-cut` — cross-episode thematic anthology

6 phases: download → build/ensure universe state → extract themes →
resolve segments → select segments → assemble audio.

**Phase 5 uses per-theme LLM selection** (not a budget-filling algorithm).
Each theme's candidate segments are shown with transcript context, the
LLM decides keep/drop and refines boundaries. ~1 DeepSeek call per theme.

| Produces | Tracked? |
|---|---|
| `output/master_cut*.mp3` | ❌ gitignored |

```bash
python3 -m podcastcondensor build-master-cut <PLAYLIST_URL> \
  --start N --end N --target-duration 12600 --output output/master_cut.mp3
```

### `build-minimal-theme` — single-theme dev tool

Uses the same LLM selection approach as master cut Phase 5. Reads
universe state + optional themes cache (no longer cached — always
fresh extraction).

```bash
python3 -m podcastcondensor build-minimal-theme <THEME_ID>
```

## Source modules

| File | Purpose |
|------|---------|
| `src/podcastcondensor/cli.py` | CLI entry point, all 4 subcommands |
| `src/podcastcondensor/pipeline.py` | `process-playlist` single-ep orchestration |
| `src/podcastcondensor/playlist_pipeline.py` | `build-universe` + re-exports `build_master_cut` |
| `src/podcastcondensor/master_cut.py` | `build-master-cut` orchestration (6 phases) |
| `src/podcastcondensor/minimal_theme_cut.py` | `build-minimal-theme` + LLM selection logic |
| `src/podcastcondensor/universe_state.py` | Cross-episode knowledge base management |
| `src/podcastcondensor/global_state.py` | Per-episode DeepSeek: transcript → structured knowledge |
| `src/podcastcondensor/classify_raw.py` | Per-episode compression (one-shot) + legacy classification |
| `src/podcastcondensor/subtitles.py` | SRT/VTT parsing, cleaning, dedup, sentence-block building |
| `src/podcastcondensor/theme_extraction.py` | Theme extraction from universe state (one DeepSeek call) |
| `src/podcastcondensor/intervals.py` | Interval building + stats computation |
| `src/podcastcondensor/audio_strategies.py` | Audio cutting strategies (sequential, parallel, single-pass) |
| `src/podcastcondensor/download_pool.py` | Parallel episode download for master cut |
| `src/podcastcondensor/downloader.py` | Audio + subtitle download via yt-dlp |
| `src/podcastcondensor/transcribe.py` | Whisper transcription |
| `src/podcastcondensor/dedup.py` | Transcript dedup merge |
| `src/podcastcondensor/config.py` | Configuration dataclass |
| `src/podcastcondensor/segmentation/sentence_units.py` | Sentence-unit extraction (legacy) |
| `src/podcastcondensor/llm/deepseek.py` | DeepSeek API client |
| `prompts/global_state.txt` | Extraction prompt (global_state.py) |
| `prompts/compress_episode.txt` | Compression prompt (classify_raw.py) |
| `prompts/classify_raw.txt` | Legacy classification prompt |
| `prompts/extract_themes.txt` | Theme extraction prompt |

## Sentence-block preprocessing — highest priority fix

**Guarantee**: no mid-sentence or mid-thought cuts in the final audio.

SRT entries are timed silence/pause chunks — they routinely split sentences
mid-thought. Using them as-is guarantees choppy output.

**How it works:**
1. `subtitles.py::build_sentence_blocks()` merges consecutive SRT entries
   until the combined text ends with `.`, `!`, or `?`. Each block has
   `{start, end, text}` — guaranteed complete-sentence boundaries.
2. In `minimal_theme_cut.py::_format_segment_with_context()`, the candidate
   window displayed to the LLM is snapped to sentence-block boundaries so
   the LLM only sees complete-sentence candidates.
3. In `minimal_theme_cut.py::apply_decisions()`, after the LLM returns its
   refined boundaries, a HARD CONSTRAINT snap aligns them to sentence-block
   boundaries. No LLM prompt request — this is deterministic code.

**This is the highest-value item because:**
- It's a hard constraint, not a soft prompt request
- It applies to both master cut and minimal theme cut
- Zero LLM cost (purely algorithmic, runs in milliseconds)

### #1: Replace knapsack budget-filling with per-theme LLM selection

**Problem:** `master_cut.py::select_segments_for_master_cut()` is a pure
time-budget algorithm — sorts by importance, fills proportionally. No
narrative coherence, no context awareness, no boundary refinement.

**Solution:** For each theme, run the selection logic from
`minimal_theme_cut.py`:
1. Build a prompt showing each candidate segment with ~30s of surrounding
   SRT transcript context
2. One DeepSeek call per theme: decide keep/drop + refine boundaries
3. Merge overlapping/adjacent kept segments
4. Concatenate all kept theme segments with beep transitions

**Changes:** `master_cut.py` Phase 5 — replace
`select_segments_for_master_cut()` with per-theme LLM selection.
Factor `build_selection_prompt()` and `apply_decisions()` for import.

**Cost:** ~1 DeepSeek call per theme × ~12 themes ≈ $0.12

### #2: Port segment boundary widening to `_validate_segments()`

**Problem:** `global_state.py::_validate_segments()` snaps to SRT
boundaries and logs mid-sentence warnings — but never widens.
`classify_raw.py::_snap_segment()` already has working widening logic.

**Solution:** Port from `_snap_segment()`:
- Widen start backward by 1 SRT entry if first word is lowercase
- Widen end forward by 1 SRT entry if text continues mid-thought
- Extend to minimum duration by absorbing subsequent entries within gaps

**Changes:** `global_state.py::_validate_segments()` — add widening.

### #3: Quality warnings at master cut completion

**Problem:** Master cut produces no quality report. Bad segments
(intro-adjacent, too short, too long) are invisible.

**Solution:** After Phase 5 selection, compute warnings:
- Segments within first 3 minutes of episode → likely intro/banter
- Segments <15s → too short to convey meaning
- Segments >600s → likely too broad
- Write warnings to result dict, display at end

### #4: Remove YT sub path from all pipelines

**Problem:** Multiple code paths check `prefer_yt_subs` and fall back to
whisper. YouTube subs are unreliable — there's no reason to try them.

**Solution:**
- `download_pool.py::_ensure_episode_artifacts()` — remove YT sub download
- `master_cut.py::build_master_cut()` — remove `prefer_yt_subs` param,
  always transcribe
- `playlist_pipeline.py::build_universe_state()` — remove YT sub path
- `cli.py` — remove `--yt-subs` and `--force-whisper` flags

### #5: Remove non-SRT artifacts from git

- `git rm --cached` all `output/ep-*/global_state.json`,
  `output/ep-*/decisions.json`, `output/ep-*/stats.json`,
  `output/ep-*/compressed.json`
- `git rm output/_themes.json`
- `git rm output/universe_state.json`
- Update `.gitignore` to only allow `source_subtitles.srt`
- Remove local unversioned intermediates (keep downloaded MP3s)

## Universe state coverage (before cleanup)

| Episodes | SRT source | In universe state |
|----------|-----------|-------------------|
| 1-28 | Whisper | ✅ |
| 29-30 | YouTube subs + whisper | Partial |
| 31-40 | Whisper | ✅ |

## Required

- DeepSeek API key in `ANTHROPIC_AUTH_TOKEN` or `DEEPSEEK_API_KEY`
- ffmpeg, yt-dlp, faster-whisper
