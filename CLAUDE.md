# podcastcondensor

Condensing "Lord of Spirits" podcast episodes using DeepSeek LLM.

**🚫 Must ask before any code change, pipeline run, or config change.** Do not implement, modify, or execute anything without describing the plan and getting explicit approval. This is a standing rule — I do not need to be reminded of it per-session.

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
- **On-disk scan first for episode resolution.** Before any YouTube call,
  `ensure_all_episode_artifacts()` scans `output/ep-NNN/` for existing
  MP3+SRT and builds manifests directly from filenames (the MP3 filename
  contains the video ID). YouTube is only contacted for episodes missing
  from disk. This makes the pipeline fully offline for already-processed
  ranges.
- **Master cut target: 90min at 1.25x (6750s at 1x).** Default in both
  `config.py` and `build-master-cut` CLI. Pass `--target-duration` to
  override. 6750s = 90 × 60 × 1.25.
- **Playlist pagination via web client.** `list_playlist()` passes
  `--extractor-args youtube:player_client=web` to yt-dlp to force web
  client pagination. Without this, YouTube tab API caps at 100 entries
  (one page). With it, `--playlist-end 999` reliably fetches all pages.

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
  --start N --end N --target-duration 6750 --output output/master_cut.mp3
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

### Squelched: `build_sentence_blocks` info log → debug

`subtitles.py::build_sentence_blocks()` was `logger.info(...)` — fires for
every candidate segment in every theme selection prompt. In 10-episode
batch, that's hundreds of identical "Sentence blocks: N entries → M blocks"
lines. Changed to `logger.debug(...)`.

## Three segmentation fixes (July 2026)

Three bugs found in the ep31-40 master cut caused a setup-heavy segment
("Protestant friends, let's have a talk...") to be included without the
actual theological argument that followed. Fixed in order:

### Fixed #1: Comma defeats continuation-prefix check in `_validate_segments`

**Root cause:** `global_state.py` uses `end_text.lower().startswith(("so ",
...))` to detect mid-thought cuts at segment end. Text like `"So, and the
reason..."` has a comma after "So" → `startswith("so ")` is False → widening
never fires → segment ends mid-setup.

**Fix:** Instead of `startswith(prefix)` on full text, extract the first word,
strip punctuation, and check against a set of continuation words ("and",
"but", "so", "because", ...). This catches comma variants (e.g. `"So,` →
stripped to `"so"` → matches).

**File:** `global_state.py:237-244`

### Fixed #2: Claims (and all categories) included in theme segment resolution

**Root cause:** `master_cut.py::resolve_theme_segments_from_state()` only
iterates `universe_data["concepts"]`. DeepSeek classified the substantive
theological argument for `royal-priesthood-of-believers` as a **claim**
(`priesthood-not-abolished`) — making it invisible to the selection LLM.

**Fix:** Iterate all categories (concepts, claims, entities,
scriptural_links, glossary) into the `items_by_id` lookup. The existing
`ep:start:end` dedup in `segments_map` prevents duplicate segments across
categories.

**File:** `master_cut.py:130-136`

### Fixed #3: Dynamic context window replacing fixed 30s buffer

**Root cause:** `minimal_theme_cut.py` hardcodes `context_buffer=30.0`
for transcript context around each candidate segment. The snapped end was
8361s → context extends to 8391s. The `priesthood-not-abolished` claim
segment starts at 8395s — 4s beyond the LLM's view, so it can't see
the thought continues and widens.

**Fix:** Instead of fixed `end + 30s`, extend the end window to the start
of the *next* candidate segment in the same episode (+ small overlap),
capped at 120s. Similarly, extend the start backward to the *previous*
segment's end. This is naturally bounded and doesn't rely on arbitrary
tuning.

**File:** `minimal_theme_cut.py:226-228` (`_format_segment_with_context`)
and `build_selection_prompt` call site.

## Master cut status (2026-07-29)

| Batch | Status | Notes |
|-------|--------|-------|
| ep31-40 | ✅ Done (2nd pass) | Rebuilt with all fixes, 78 segments / 9 themes, 12418s |
| ep41-50 | ⏳ Pending | Ask before running any pipeline |

## Universe state coverage (before cleanup)

| Episodes | SRT source | In universe state |
|----------|-----------|-------------------|
| 1-28 | Whisper | ✅ |
| 29-30 | YouTube subs + whisper | Partial |
| 31-40 | Whisper | ✅ |

## Required

- DeepSeek API key in `ANTHROPIC_AUTH_TOKEN` or `DEEPSEEK_API_KEY`
- ffmpeg, yt-dlp, faster-whisper
