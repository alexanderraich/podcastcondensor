# podcastcondensor

Condensing "Lord of Spirits" podcast episodes using DeepSeek LLM.

**🚫 Must ask before any code change, pipeline run, config change, or architectural decision.** Do not implement, modify, or execute anything without describing the plan and getting explicit approval. This is a standing rule — I do not need to be reminded of it per-session.

**Document architectural decisions in CLAUDE.md.** Any substantive architectural change (data flow, pipeline design, scoping strategy, state management) is recorded here with rationale. This documents the decision's context and reasoning for future reference.

**Planning → CLAUDE.md first, then implement.** For any substantive change,
the standing workflow is: plan → hammer the plan down into CLAUDE.md
(reconcile it with existing content, resolve contradictions) → THEN implement.
A plan is not approved until it is documented in CLAUDE.md. This applies to
every architectural decision, not just the first one.

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

## Source modules

| File | Purpose |
|------|---------|
| `src/podcastcondensor/cli.py` | CLI entry point, all 4 subcommands |
| `src/podcastcondensor/super_cut.py` | `build-super-cut` orchestration (offline: merge → chunk → coalesce → resolve → select → assemble) |
| `src/podcastcondensor/pipeline.py` | `process-playlist` single-ep orchestration |
| `src/podcastcondensor/playlist_pipeline.py` | `build-universe` + re-exports `build_master_cut` |
| `src/podcastcondensor/master_cut.py` | `build-master-cut` orchestration (6 phases) |
| `src/podcastcondensor/minimal_theme_cut.py` | Per-theme LLM selection engine (shared by master-cut & super-cut) |
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
| `prompts/coalesce_themes.txt` | Coalesce prompt (super_cut.py) — fuses chunk themes into global themes |

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
| ep31-40 | ✅ Done (2nd pass) | 10 themes / 35 segments, 6704.9s vs 6750 target, 3 warnings, 0 errors. 31-40 archive comparison run was killed mid-assembly (2026-07-31) — deprioritized in favor of the transcription effort. |
| ep41-50 | ✅ Archive done | LLM-volume archive: 9.13h / 20 themes / 185 segments / 31% of 29.5h source, 0 errors, 3 "too broad" warnings. |

## Episode data (2026-08-01 — COMPLETE)

**All 125 non-Q&A episodes have whisper SRTs in git AND `global_state.json` on
disk** (eps 1-144 minus 19 Q&A). The 126th SRT is ep-18 — itself a Q&A episode
that was transcribed before Q&A skipping was standard, so it has an SRT but no
`global_state.json`. Per convention, `global_state.json` stays gitignored.

**Why 125, not 144:** the 19 Q&A / "Pantheon & Pandemonium Live Q&A" specials
(18, 66, 67, 74, 78, 80, 89, 90, 98, 104, 106, 111, 117, 121, 122, 126, 134,
135, 141) don't develop coherent themes, so `build-universe` skips them by
default (`--include-qa` to include). They are the ONLY episodes without
`global_state.json`; every non-Q&A episode has one. **Correction (2026-08-01):**
ep-18 is a Q&A episode (added to the list — previously mislabeled non-Q&A).

| Episodes | SRT | global_state (on disk) |
|----------|-----|------------------------|
| 1-30 | ✅ all (in git) | ✅ all (ep-18 = Q&A skipped; ep-29 extracted offline after search-fallback failure) |
| 31-50 | ✅ all | ✅ all |
| 51-98 | ✅ all (minus Q&A 66,67,74,78,80,89,90,98) | ✅ all |
| 99-144 | ✅ all (minus Q&A 104,106,111,117,121,122,126,134,135,141) | ✅ all |
| Q&A (19) | ❌ intentionally skipped (except ep-18, pre-existing SRT) | ❌ |

## Super master cut — 144-episode thematic anthology (2026-08-01, PLAN)

**End goal:** a true cross-episode thematic anthology over all ~144 episodes —
"the most important insights over all the universe" — where each top theme's
cut quotes MANY episodes (a topic may be concentrated in a few episodes or
diffuse across the run). This replaces the old 2026-07-31 "roadmap" (which
proposed a separate insight-synthesis call); see "Why not insight synthesis".

**Approved design (2026-08-01):**
1. **Output = per-theme MP3s** (one file per top theme), not one combined file.
2. **Volume = mirror the proven range-cut mechanism** (ep41-50: 9.13h / 20
   themes / 185 segments / 31% of 29.5h). Reuse `select_segments_for_master_cut`
   with NO time budget — the soft "4-8 segments / 8-20 min" guide; LLM-owned
   volume. Per-theme volume is unbounded by design (~8-20h total expected).
3. **Theme discovery is chunked + coalesced.** The full universe is ~3,900
   items — too big for one 64K-context call — and the existing
   `extract_themes` truncation (200 items, 20 related_item_ids/theme cap)
   fights the "many episodes quoted" goal:
   - Split 1-144 into ~11 chunks of ~12 eps; run existing `extract_themes`
     per chunk (re-merge the chunk's global_states so `episode_numbers`
     reflect chunk-local frequency, not corpus-wide).
   - One new `coalesce_themes` DeepSeek call fuses all chunk themes
     (~100-140) into the top 15-25 global themes, each mapped to its
     constituent chunk-theme ids.
   - A global theme's items = union of its chunk themes' related_item_ids,
     resolved against the cumulative universe → full cross-episode span.
4. **Episode-diversity cap** on candidate segments per theme (reserve top-N
   by relevance + round-robin across episodes, `max_total≈48`,
   `max_per_episode≈6`) so the selection prompt stays in context AND quotes
   many episodes — this is the mechanism that delivers "throughout the run".
5. **Pipeline runs offline from disk** (no YouTube, 0 download/whisper):
   merge cumulative `universe_state_001_144.json` from per-episode
   global_states → chunk → coalesce → resolve → select → assemble per-theme
   MP3s. **Output is FLAT** — MP3s go directly into `output_root` as
   `<theme_id>.mp3` (no `super_cut_001_144/` subfolder; decided 2026-08-01).

**Prerequisites (done 2026-08-01):** output dir cleaned of stale pipeline
artifacts — only the per-episode 4-file set (`source_subtitles.srt`,
`global_state.json`, `<video_id>.mp3`, `_transcribe_diag.log`) plus top-level
`.gitkeep` remain; the 210MB `master_cut_41_50.mp3` and ep41-50 state/stats
were removed. eps 1-30 global_state regenerated via the standard
`build-universe --start 1 --end 30` (ep-29 via one-off offline extraction;
ep-18 correctly skipped as Q&A) — **full 125/125 non-Q&A coverage achieved.**

**Why not insight synthesis (replaces 2026-07-31 roadmap step 4):** the user
explicitly chose to mirror the proven range-cut volume mechanism (per-theme
LLM selection, no budget) rather than a new "exactly N insights" scheme. The
cross-episode character comes from global themes discovered across chunks +
episode-diverse candidate pools — not a separate narrative-synthesis call.
LLM-written narrator intros remain out of scope (undecided).

**New code:** `src/podcastcondensor/super_cut.py` (orchestrator) +
`build-super-cut` CLI subcommand + `prompts/coalesce_themes.txt`; small
refactor of `universe_state.py::add_episode_knowledge` → pure
`merge_episode_knowledge`. Reuses unchanged: `extract_themes`,
`resolve_theme_segments_from_state`, `select_segments_for_master_cut`,
`assemble_master_cut`, audio helpers. Artifacts (all gitignored):
`universe_state_001_144.json`, `super_cut_themes_001_144.json` (the "top
topics" report, with per-theme `episode_numbers`), `super_cut_stats_001_144.json`,
and flat `<theme_id>.mp3` files in `output_root`.

**Coalesce dedup + cap (decision, 2026-08-01):** DeepSeek does NOT honor the
coalesce prompt's "15-25 themes" count — the first full dry-run produced
**101 global themes** from 198 chunk themes, with many near-duplicates
("Theology of the Cross and Atonement" ×2, "Theology of Penance and
Confession" ×3, ...). Fixed deterministically (not by trusting the prompt):
`_dedupe_and_cap_global_themes` drops empty-source themes, merges themes with
the exact same `source_theme_ids` set, merges identical normalized titles
(union of sources, max importance), then caps to `--max-themes` (default 25)
by importance. Result: 101 → 25 clean themes. Same lesson as the master-cut
budget saga: constraints must be enforced in code, not the prompt.

**Status (2026-08-01):** implemented + unit-tested (23 tests); eps 1-30
regenerated (full 125/125 non-Q&A coverage); 41-50 dry-run validated; full
1-144 dry-run validated (125 eps / 3,094 items → 198 chunk themes → 25
curated global themes → 2,831 candidate segments; top themes span 34-45
episodes, e.g. "Christology" 45 eps, "Theosis" 43). Christology single-theme
pilot cut ran (8 segments / 7 eps / 26 min) then was deleted on request.
**`build-minimal-theme` was removed on 2026-08-01** as a leftover —
superseded by `build-super-cut --theme <id>` for single-theme cuts; its
per-theme selection engine survives as the shared `minimal_theme_cut.py`.

### Intermediate artefacts on disk (plan, 2026-08-01 — TO BE IMPLEMENTED)

**Principle: the only expensive work is theme discovery (~12 DeepSeek calls),
and it runs ONCE and is persisted. Everything downstream is deterministic from
persisted artefacts, and each theme's selection is persisted the moment it's
cut.** Repeated theme cuts must never re-run discovery or re-spend calls on
themes already selected. Range-scoped to `_001_144`:

| # | Artefact | Contents | Produced by | Cost |
|---|----------|----------|-------------|------|
| 1 | `universe_state_001_144.json` | cumulative merge (125 eps, ~3k items) | merge | 0 calls, deterministic |
| 2 | `super_cut_discovery_001_144.json` | chunk themes + 25 global themes + chunk→episode map | chunk extraction + coalesce | **~12 calls, ONCE** |
| 3 | `super_cut_candidates_001_144.json` | per global theme: capped candidate segments (ep, start, end, audio_path, relevance, is_intro, episode_numbers) | resolve + cap | 0 calls, deterministic from #1+#2 |
| 4 | `super_cut_selections_001_144.json` | per theme: LLM-selected segments with refined boundaries | selection | **1 call per theme, appended** |
| 5 | `super_cut_themes_001_144.json` | all 25 themes report (importance, episode_numbers, candidate/selected counts) | derived from #2+#3+#4 | 0 calls |
| 6 | `super_cut_stats_001_144.json` | selection trace | assembly | 0 calls |

**Flows:**
- **"Cut next theme"** → load #2+#3; if theme already in #4 → assemble only
  (0 LLM calls). If not → **1 selection call**, append to #4, assemble.
- **Full archive** → iterate #3, select only the missing themes, assemble.
  0 discovery, 0 resolve — just N selection calls + ffmpeg.
- **Bracket analysis** (top theme per episode bracket, e.g. 40-80 / 81-120 /
  rest) → pure offline computation from #3 + #1 (rank by importance ×
  in-bracket candidate weight). A `super-cut-brackets` subcommand, 0 LLM calls.
- **Regenerating the full themes report** → recomputed from #2+#3+#4; a
  `--theme` filtered run must NOT overwrite the all-themes report (fixed: the
  report is always written from the full global-theme list).

**Implementation checklist (DONE 2026-08-01):** discovery cache (`_save/load`),
candidates cache (`_save/load`), selections cache (`_merge/load` +
`_selections_from_dicts`); resolve step always resolves ALL themes (0 calls) so
candidates/report/brackets are complete regardless of `--theme`; the full
themes report is always written from the full theme list (a `--theme` run no
longer overwrites it); selection skips already-cached themes (0 calls → pure
audio re-cut); `super-cut-brackets` subcommand (read-only). Unit-tested (30
tests). NOT yet seeded: caches need ONE `build-super-cut --dry-run` run
(~12 discovery calls) to populate `super_cut_discovery_001_144.json` +
`super_cut_candidates_001_144.json`.

## Universe state coverage (2026-08-01 — current)

| Episodes | SRT source | In universe state |
|----------|-----------|-------------------|
| 1-28 | Whisper | ✅ (per-episode `global_state.json`) |
| 29-30 | YouTube subs + whisper | ✅ (per-episode `global_state.json`; ep-29 extracted offline) |
| 31-50 | Whisper | ✅ (per-episode `global_state.json`) |
| 51-144 | Whisper | ✅ (per-episode `global_state.json`) |

All 125 non-Q&A episodes have per-episode `global_state.json` on disk
(ep-18 = Q&A, excluded by design). The **cumulative 1-144 universe state is
built by the `build-super-cut` merge phase** (offline, 0 API calls) — see the
super-cut section. Per convention all universe state is gitignored.

## Required

- DeepSeek API key in `ANTHROPIC_AUTH_TOKEN` or `DEEPSEEK_API_KEY`
- ffmpeg, yt-dlp, faster-whisper
