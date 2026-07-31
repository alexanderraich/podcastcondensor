# podcastcondensor

Condensing "Lord of Spirits" podcast episodes using DeepSeek LLM.

**🚫 Must ask before any code change, pipeline run, config change, or architectural decision.** Do not implement, modify, or execute anything without describing the plan and getting explicit approval. This is a standing rule — I do not need to be reminded of it per-session.

**Document architectural decisions in CLAUDE.md.** Any substantive architectural change (data flow, pipeline design, scoping strategy, state management) is recorded here with rationale. This documents the decision's context and reasoning for future reference.

## Core decisions (July 2026)

- **Only whisper SRTs in git.** YouTube subtitles are unreliable (fragmented,
  auto-generated artifacts). All SRT files in git are whisper-produced.
  Pipeline modes default to whisper always — no YT sub fallback.
- **Only `source_subtitles.srt` tracked per episode.** Everything else
  (`global_state.json`, `compressed.json`, `decisions.json`, `stats.json`,
  `universe_state.json`, `_themes.json`) is derivable from the SRT + LLM
  calls and is NOT version-controlled.
- **`universe_state.json` is disposable and ephemeral per range cut.** It can be
  reconstructed from scratch with `build-universe` using the version-controlled
  SRTs. Not tracked in git after July 2026 cleanup. `build-master-cut` and
  `build-minimal-theme` delete the stale file at the start of each run and
  rebuild it fresh from existing per-episode `global_state.json` on disk for
  the target episode range only. This guarantees theme extraction and segment
  resolution only see content from the target window — no leakage from older
  episodes. The cumulative state is only produced by the dedicated
  `build-universe` pipeline.
- **Range-scoped artifact filenames (July 2026).** Since the per-range state is
  ephemeral, `build-master-cut` and `build-minimal-theme` write
  `universe_state_{START:03d}_{END:03d}.json` (e.g. `universe_state_041_050.json`)
  instead of a fixed `universe_state.json`, and `build-master-cut` writes
  `master_cut_stats_{START:03d}_{END:03d}.json` containing a full `selections`
  array (theme, episode, start, end, duration, beep) so each batch keeps a
  trace of exactly what was kept. Fixed-name `universe_state.json` /
  `master_cut_stats.json` files left on disk are stale leftovers from before
  this convention.
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
  override. 6750s = 90 × 60 × 1.25. **NOTE (2026-07-31):** since the master
  cut is now an LLM-volume archive, this target is informational only — the
  LLM decides volume and the actual output is typically 5-6h.
- **Playlist pagination via web client.** `list_playlist()` passes
  `--extractor-args youtube:player_client=web` to yt-dlp to force web
  client pagination. Without this, YouTube tab API caps at 100 entries
  (one page). **Caveat (verified 2026-07-31):** even with the web client,
  this playlist caps at 100 entries (the newest 100, episodes 45-144).
  Older episodes are resolved via the direct-search fallback in
  `resolve_episode_sources()` — that's what covers eps < 45.
- **Download resilience.** `download_audio()` retries transient yt-dlp
  failures (403/429 rate limiting) with 3 attempts + backoff, plus
  `--retries 3 --retry-sleep linear=2:15` inside yt-dlp. In parallel
  downloads, YouTube can 403 individual workers; without retry a transient
  error silently dropped an episode from the batch.
- **Fail-loud on partial download.** `ensure_all_episode_artifacts()` raises
  `RuntimeError` listing any episodes whose download/transcription failed;
  `build-master-cut` aborts instead of assembling a master cut with missing
  episodes. A partial anthology must never be reported as a clean success.

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

### `build-master-cut` — cross-episode thematic archive

6 phases: download → build/ensure universe state → extract themes →
resolve segments → select segments → assemble audio.

**Phase 5 uses per-theme LLM selection** (not a budget-filling algorithm).
Each theme's candidate segments are shown with transcript context, the
LLM decides keep/drop and refines boundaries. ~1 DeepSeek call per theme.

**Volume is fully LLM-owned — the master cut is an archive, not a forced
digest.** The LLM keeps the segments that best explain each theme, and
everything it keeps across ALL themes is concatenated. There is NO time
budget in the prompt and NO Python truncation loop. The result is the LLM's
full editorial judgment: a ~6h archive from a ~30h batch (~20-30% of source),
covering every theme that survives extraction. The `--target-duration` flag
is informational only (used by the knapsack fallback).

**Why (history, 2026-07-31):** three budget approaches failed in the ep41-50
batch. A greedy global Python cap hit the 6750s target but starved themes
4-11 to zero (3/13 themes in the cut). A per-theme time budget *in the
prompt* was ignored by DeepSeek — it kept 2-7x every stated budget (6h total).
A hardened "HARD BUDGET / non-negotiable" prompt did the same. Conclusion:
DeepSeek does not honor volume constraints, and a 90-min cut would discard
3/4 of the content the LLM judged worth keeping. The product is therefore
the full LLM-volume archive.

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

### Fixed #4: Ephemeral universe state per range cut

**Root cause:** `build-master-cut --start N --end M` loaded the cumulative
`universe_state.json` on disk, which contained concepts/claims/entities from
*all* previously processed episodes (not just the target range). Theme
extraction and segment resolution saw out-of-range content, so the time
budget could be allocated to concepts from older episodes.

**Fix:** `universe_state.json` is now ephemeral per range cut. Both
`build-master-cut` and `build-minimal-theme` delete the stale file and
rebuild fresh from per-episode `global_state.json` on disk, scoped to
the manifest range. Zero extra API calls — already-processed episodes
have their `global_state.json` on disk.

**Files:** `master_cut.py:848-926`, `minimal_theme_cut.py:917-952`

## Master cut status

| Batch | Status | Notes |
|-------|--------|-------|
| ep31-40 | ✅ Done (2nd pass) | 10 themes / 35 segments, 6704.9s vs 6750 target, 3 warnings, 0 errors. 31-40 archive comparison run was killed mid-assembly (2026-07-31) — deprioritized in favor of the 51-140 transcription effort. |
| ep41-50 | ✅ Archive done | LLM-volume archive: 9.13h / 20 themes / 185 segments / 31% of 29.5h source, 0 errors, 3 "too broad" warnings. |

## Roadmap — 140-episode super master cut (2026-07-31, TO BE DONE)

**End goal:** a true cross-episode thematic anthology — "the most important
insights over all 140 episodes" — built from the full audio database.

**Why the current per-theme master cut is NOT the end goal:** it is effectively
per-episode curation. The show is organized episode-per-topic, so each theme's
content is concentrated in its dedicated episode (ep42 = afterlife, ep43 =
prophecy, ep44 = scripture). Per-theme selection therefore keeps ~one episode's
content per theme. Verified in the ep41-50 archive: 20/20 "Afterlife" segments
were all from ep 42. The result is a labeled episode-cores anthology, not
cross-episode synthesis.

**Planned pipeline (to be built):**
1. **Full data** — whisper SRT + `global_state.json` for all 140 episodes
   (eps 1-50 done; **51-140 transcription in progress**, `build-universe`).
2. **Universe state merge** — build a real cumulative `universe_state.json`
   across all 140 (currently range-scoped per batch; needs a merge step over
   the per-episode `global_state.json` files).
3. **Theme identification** across the full 140 (one DeepSeek call).
4. **Cross-episode insight synthesis (the missing core step)** — per theme,
   ONE DeepSeek call over all episodes' claims/segments producing the ~8-15
   most important insights ordered narratively, **required to span different
   episodes** (traces how understanding deepens across the arc). This is what
   makes it overarching instead of per-episode.
5. **Resolve each insight to audio** + assemble per-theme insight tracks
   (or a combined cut). Optional future enhancement: LLM-written narrator
   intros between insights (undecided).

**Decisions taken 2026-07-31:**
- Master cut volume is LLM-owned (archive, not forced 90-min). Both the greedy
  global Python cap and pure-prompt budgets failed (see history in the
  `build-master-cut` section); volume enforcement is deferred until the
  insight-synthesis design lands.
- Q&A and Reading episodes are excluded from `build-universe` (default) — they
  don't develop coherent themes.
- The 90-min target is informational for master cut; the archive is typically
  9h+ per 10-episode batch.

## Universe state coverage (before cleanup)

| Episodes | SRT source | In universe state |
|----------|-----------|-------------------|
| 1-28 | Whisper | ✅ |
| 29-30 | YouTube subs + whisper | Partial |
| 31-40 | Whisper | ✅ |
| 41-50 | Whisper | ✅ (per-episode `global_state.json` on disk; range-scoped state ephemeral per run) |

## Required

- DeepSeek API key in `ANTHROPIC_AUTH_TOKEN` or `DEEPSEEK_API_KEY`
- ffmpeg, yt-dlp, faster-whisper
