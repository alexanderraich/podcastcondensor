# podcastcondensor

Local-first pipeline for condensing "Lord of Spirits" podcast episodes using LLMs.

## Commands

```bash
# Build universe state (cross-episode knowledge base) from episodes
python3 -m podcastcondensor build-universe [PLAYLIST_URL] --start 1 --end 20

# Process episodes with universe state context
python3 -m podcastcondensor process-playlist [PLAYLIST_URL] --state-file output/universe_state.json --start 21

# Process single episode
python3 -m podcastcondensor run [URL]

# Test one block of an episode
python3 -m podcastcondensor process-playlist [PLAYLIST_URL] --state-file output/universe_state.json --start 21 --end 21 --max-blocks 1
```

## Model routing

- **Extraction / summarization**: `qwen2.5:3b` (--model) — fast, VRAM-light, handles short structured prompts
- **Classification**: `qwen2.5:7b` (--classify-model) — reliable on full-context JSON with universe state
- Defaults are set in `Config` class in `config.py`

## Architecture

- Three-tier state: Universe (cross-episode) → Global (episode outline) → Local (block summary)
- Phase A: Build global episode map (block summaries + outline) — uses 3b
- Phase B: Classify segments keep/drop — uses 7b
- Phase C: Cleanup (dedup, opening protection)
- Phase D: Extract knowledge into universe state — uses 3b
- Extraction and merging are separate: LLM extracts per-episode knowledge, Python merges/deduplicates globally

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

On any validation failure, falls back to deterministic sentence-boundary grouping. Configurable via `--refine`/`--no-refine`. Never cuts mid-sentence.

## Known issues

### Fixed

- **resolve_maybe now defaults to "drop"** — both error/fallback paths in `resolve_maybe()` were defaulting to "keep", contradicting the prompt's instruction. Changed to "drop". (Fixed 2026-06-14)
- **Segmentation no longer cuts mid-sentence** — Signal C (hard cap) now enters a sentence-completion overflow mode, accumulating up to `sentence_overflow_words` (default 150) additional words to find the next `.`, `!`, or `?` before cutting. Controlled by `sentence_overflow_words` in Config. (Fixed 2026-06-14)

### Deferred (need history rebuild)

- State knowledge for already-processed episodes gets duplicated on re-run (Phase D appends without dedup)
- 7 episodes have 0 entities extracted (the prompt's entity schema lost field specs in a cleanup edit)
- If restarting `process-playlist`, manually delete `state_knowledge.json` for that episode to force re-extraction

These three issues (#1, #2, #4 in project tracking) all need a history rebuild. Parked for now.

## Useful files

- `output/universe_state.json` — cross-episode knowledge base
- `prompts/classify_chunks_global.txt` — classification prompt (the main lever for compression aggressiveness)
- `prompts/extract_knowledge_fast.txt` — knowledge extraction prompt
- `AGENT.md` — engineering practices guide
