# podcastcondensor

Local-first pipeline for condensing "Lord of Spirits" podcast episodes using LLMs.

## Commands

```bash
# Build universe state (cross-episode knowledge base) from episodes
python3 -m podcastcondensor build-universe [PLAYLIST_URL] --start 1 --end 20

# Process episodes with universe state context (Ollama default)
python3 -m podcastcondensor process-playlist [PLAYLIST_URL] --state-file output/universe_state.json --start 21

# Process episodes with DeepSeek classification
python3 -m podcastcondensor process-playlist [PLAYLIST_URL] --state-file output/universe_state.json --start 21 \
  --classification-provider deepseek --classification-fallback ollama

# Process episodes with everything (single_pass_filter audio — new default)
python3 -m podcastcondensor process-playlist [PLAYLIST_URL] --state-file output/universe_state.json --start 21 \
  --classification-provider deepseek --knowledge-provider deepseek

# Process single episode (Ollama default)
python3 -m podcastcondensor run [URL]

# Process single episode with DeepSeek
python3 -m podcastcondensor run [URL] --classification-provider deepseek

# Test one block of an episode (Ollama default)
python3 -m podcastcondensor process-playlist [PLAYLIST_URL] --state-file output/universe_state.json --start 21 --end 21 --max-blocks 1

# Environment diagnostics
python3 -m podcastcondensor doctor

# Test API connectivity (costs a token)
python3 -m podcastcondensor doctor --check
```

## Model routing

### Local (Ollama) — default for all phases

- **Extraction / summarization**: `qwen2.5:3b` (`--model`) — fast, VRAM-light, handles short structured prompts
- **Classification**: `qwen2.5:7b` (`--classify-model`) — reliable on full-context JSON with universe state

### Cloud (DeepSeek — OpenAI-compatible)

- **Classification**: `deepseek-chat` via `--classification-provider deepseek`
- **Knowledge extraction**: `deepseek-chat` via `--knowledge-provider deepseek`
- API key: read from `ANTHROPIC_AUTH_TOKEN` or `DEEPSEEK_API_KEY` environment variable
- Fallback: `--classification-fallback ollama` keeps pipeline working when cloud is unavailable
- Defaults are set in `Config` class in `config.py`

### Audio cutting strategies

Set via `--audio-strategy`:
- `single_pass_filter` — one ffmpeg invocation using `filter_complex` (atrim/asetpts/concat). **Default.** Reads source once linearly — no HDD thrashing. No temp segment files. Batched at 100 intervals.
- `sequential_copy` — original per-interval ffmpeg extraction, sequential (slowest)
- `parallel_copy` — per-interval extraction in parallel with bounded workers (`--audio-parallel-workers N`, default 2). Staggered startup + ionice to reduce HDD contention.
- `safe_batched` — tiny filter_complex batches of 5, sequential, with checkpoint resume. Recommended for WSL / low-memory environments. Set via `--audio-safe-batch-size N` (default 5).

### Experiment flags

- `--no-enable-continuity-bias` — disables the 3-pass continuity/heuristic layer (bridge, context, short-neighbour). Use in evaluation runs to measure classifier-only performance.
- `--decisions-only` — stop after Phase C + intervals; skip knowledge extraction and audio cutting. Use for classifier evaluation (no audio cost).
- `--max-blocks N` — only classify first N blocks; rest auto-dropped. Use for fast block-level classifier comparison.

## Architecture

- Three-tier state: Universe (cross-episode) → Global (episode outline) → Local (block summary)
- Phase A: Build global episode map (block summaries + outline) — uses Ollama 3b (not yet strategy-migrated)
- Phase B: Classify segments keep/drop — strategy pattern (`ClassifierStrategy`)
- Phase C: Finalize decisions (single consolidated pass) — dedup → resolve_maybe → continuity bias (toggle) → tail detection (last). Output: `decisions_final.json` only. No intermediate files.
- Phase D: Extract knowledge into universe state — strategy pattern (`KnowledgeExtractionStrategy`)
- Extraction and merging are separate: LLM extracts per-episode knowledge, Python merges/deduplicates globally
- Audio cutting — strategy pattern (`AudioCuttingStrategy`)

### Strategy provider layer

```
┌──────────────────────────────────────────────────────────┐
│                    Pipeline (orchestrator)                │
├───────────┬────────────────┬─────────────────────────────┤
│ Phase B/C │   Phase D      │  Audio cutting              │
│ Classifier│  KnowledgeExt  │  AudioCuttingStrategy       │
│ Strategy  │  ractionStrat  │                             │
├───────────┴────────────────┴─────────────────────────────┤
│                    Strategy layer                         │
│   OllamaXStrategy / DeepSeekXStrategy / Seq/Par/Single   │
├──────────────────────────────────────────────────────────┤
│                    LLM Provider layer                     │
│   LLMClient (ABC) ← OllamaClient / DeepSeekClient        │
└──────────────────────────────────────────────────────────┘
```

### New files (added 2026-06-15)

- `src/podcastcondensor/llm/` — LLM provider package (base, ollama, deepseek clients)
- `src/podcastcondensor/strategies/` — Strategy package (base, classification, knowledge)
- `src/podcastcondensor/audio_strategies.py` — Audio cutting strategies + factory
- `tests/test_strategies.py` — 56 tests for provider selection, fallback, cache fingerprint, continuity bias, tail detection, degraded decisions
- `tests/test_audio_strategies.py` — 49 tests for filter graphs, ordering, zero intervals, interval clustering, guardrails, fragmentation

### Modified files (2026-06-15)

- `config.py` — Added `classification_provider`, `knowledge_provider`, `audio_strategy`, cache fingerprint fields, listenability config, `enable_continuity_bias`, `decisions_only`
- `cli.py` — Added `doctor` command, `_add_provider_args()` helper with provider selection args, `--enable-continuity-bias`, `--decisions-only`, `--audio-safe-batch-size`
- `pipeline.py` — Strategy injection for Phases B/D and audio cutting, cache fingerprint validation, Phase C consolidation (single `finalize_decisions()` call)
- `playlist_pipeline.py` — Strategy-based extraction in `build_universe_state`
- `classifier.py` — Added `apply_continuity_bias()`, `detect_tail_block()`; removed opening protection; prompts reverted to aggressive committed versions

### Phase C consolidation (2026-06-15)

The old 5-stage decision chain (classify → cleanup → continuity → tail → resolve maybe) with 4 intermediate checkpoint files has been replaced:

- One `finalize_decisions()` call in `pipeline.py` runs deterministic dedup → resolve_maybe → continuity bias (toggle) → tail detection LAST
- Single output file: `decisions_final.json`
- Intermediate files `decisions_clean.json`, `decisions_continuity.json`, `decisions_resolved.json` no longer written
- Opening protection (first 3 segments auto-kept) removed from `global_cleanup()`
- Continuity bias disableable via `--no-enable-continuity-bias`
- Phase D + audio skipable via `--decisions-only`

### Prompts reverted to aggressive versions (2026-06-15)

Both `classify_chunks_global.txt` and `resolve_maybe.txt` reverted to committed aggressive defaults:
- `classify_chunks_global.txt`: "Default to DROP unless clearly new", "If borderline DROP", "Precision matters more than recall". Listenability/bridging keep rules removed.
- `resolve_maybe.txt`: "When uncertain default to drop — the podcast is already too long". No continuity-bias duplication.

### Segmentation (two-pass architecture)

**Pass 1 — Deterministic rough cut** (`resegment` in `rechunker.py`):
Three signals, all sentence-boundary-aware:
- **Gap silence** — gaps >8s always split; gaps 0.5s–8s split only at sentence boundaries
- **Discourse markers** ("So ", "Now ", "But "...) — split only at sentence boundaries
- **Hard word cap** (400 words) — enters sentence-completion overflow mode: allows up to 150 extra words to find the next `.`, `!`, or `?` before cutting

**Pass 2 — LLM refinement** (`refine_segments` in `rechunker.py`):
Takes one rough segment at a time (~400-550 words), sends its merged text to qwen2.5:7b, and asks for verbatim substrings split at topic + sentence boundaries. Strict validation checks:
- Concatenating outputs reproduces the input exactly
- Every output ends with sentence punctuation
- No hallucinated or rephrased content

On any validation failure, falls back to deterministic sentence-boundary grouping. Configurable via `--refine`/`--no-refine`. Never cuts mid-sentence. Still local-only (not strategy-migrated).

## Classifier comparison findings (2026-06-15)

Experiment on Episode 21 (285 segments, 133 min) with aggressive (reverted) prompts, bias OFF:

| Provider | Primary keep | Final keep (Phase C) | Condensed audio |
|---|---|---|---|
| DeepSeek | 195 (68%) | 168 + 29 tail drops | 89.6 min |
| Ollama | ~80-100 (estimated) | TBD | ~50 min (target) |

Key findings:
- **DeepSeek is inherently more lenient** — keeps ~68% even with aggressive prompts. To match Ollama's aggression, DeepSeek needs a custom stricter prompt or not used for classification.
- **Continuity bias inflates output by +20-51 segments** (~20-40 min). Alway evaluate with `--no-enable-continuity-bias` first.
- **Tail detection removes 29 segments** (~30-40 min runtime) on this episode — always active for block-7 off-topic content.
- **Prompt softening was the primary cause of compression collapse** — softened prompt kept 252/285; reverted aggressive prompt kept 195/285 (with DeepSeek). Ollama's delta is larger.
- **Block-1 comparisons are not representative of full runs** — block 1 is intro material which always keeps more.

**Recommendation:** Classify with Ollama (`--classification-provider ollama`) for tighter compression. Keep DeepSeek for knowledge extraction (`--knowledge-provider deepseek`).

## Known issues

### Fixed

- **Phase C consolidation** — 4 intermediate decision files replaced with single `decisions_final.json`. Opening protection removed. Continuity bias toggleable. (2026-06-15)
- **Prompts reverted to aggressive** — classify and resolve_maybe prompts restored to committed "default DROP" versions. Listenability/bridge keep rules removed from classify prompt. (2026-06-15)
- **Download skip on rerun** — Phase 1 now skips yt-dlp metadata API call when `segments.json` already exists. (2026-06-15)
- **Cloud failure no longer masquerades as valid output** — `DeepSeekClassifierStrategy` raises `ClassificationFailedError`, falls back to Ollama. (Fixed 2026-06-15)
- **Continuity bias added (toggleable)** — bridge/context/neighbour passes to reduce fragmentation. Disable with `--no-enable-continuity-bias`. (Added 2026-06-15)
- **Off-topic tail detection** — force-drops trailing administrative content. (Added 2026-06-15)
- **Cluster-aware interval merging** — `cluster_gap` (1.5s) groups nearby kept segments. (Added 2026-06-15)
- **resolve_maybe defaults to "drop"** — error/fallback paths fixed. (Fixed 2026-06-14)
- **Segmentation no longer cuts mid-sentence** — hard cap enters overflow mode (150 extra words) to find sentence boundary. (Fixed 2026-06-14)
- **Universe state leak fixed** — `get_context()` excludes episodes >= current. (Fixed 2026-06-14)
- **Entity/claims schema in extraction prompt** — explicit required fields added. Claims extract properly. (Fixed 2026-06-14)
- **Phase D dedup hardened** — content-based dedup key fallback. (Fixed 2026-06-14)
- **Knowledge cache fingerprinting** — auto-invalidates on provider/model/prompt change. (Added 2026-06-15)

### Deferred — universe state persistence bugs

See above. Universe state includes current episode's prior knowledge, causing double-add on rerun. Must manually delete `state_knowledge.json` to force re-extraction.

### Known — DeepSeek rate limiting

`DeepSeekClient` raises on 429 without adaptive backoff. May need retry-with-backoff for large runs.

### Known — Phase A still Ollama-only

Global map building (Phase A) and segmentation refinement use Ollama directly. Ollama must be running even with DeepSeek for classification.

### Deferred — fallback should be automatic for DeepSeek

`--classification-provider deepseek` should auto-opt-in `--classification-fallback ollama`. Config default in `config.py`.

## Audio cutting performance

With the universe-state fix, keep rate jumped from 6.7% to 50.9%, producing ~100 intervals. Strategies sorted by speed:

| Strategy | I/O pattern | Wall clock (100 intervals) | Memory profile |
|---|---|---|---|
| `single_pass_filter` | One linear read via filter_complex | **~2–5 min** | High (single large graph) |
| `safe_batched` (batch=5) | 13 small filter graphs + concat | ~5–8 min | **Low** (tiny graphs) |
| `parallel_copy` | N concurrent seeks (ionice + staggered) | ~10–15 min | Medium (N ffmpeg procs) |
| `sequential_copy` | 100 sequential seeks via `-c copy` | ~50 min | Low (one proc) |

- `safe_batched` is the **recommended default for WSL/low-memory environments**. Splits intervals into batches of 5, writes checkpoint after each batch, resumes on crash. Configure via `--audio-safe-batch-size N` (default 5).
- All strategies use `ionice -c 3` and `nice -n 19` on Linux so ffmpeg never starves the system.

## Useful files

- `output/universe_state.json` — cross-episode knowledge base
- `prompts/classify_chunks_global.txt` — classification prompt (the main lever for compression aggressiveness)
- `prompts/extract_knowledge_fast.txt` — knowledge extraction prompt
- `AGENT.md` — engineering practices guide

## Shell / execution preferences

- **Always run commands with visible console output** — do NOT use background tasks (`&`, `run_in_background`), `tail -200` pipes, or anything that hides stdout/stderr. The user wants to observe the run on the console in real time.
- Use `2>&1` to combine stderr with stdout so nothing is lost.
- The `ANTHROPIC_AUTH_TOKEN` env var stores the DeepSeek API key (starts with `sk-`). The key value is stripped of whitespace/trailing `\r` in `resolve_api_key()` in `deepseek.py`.
