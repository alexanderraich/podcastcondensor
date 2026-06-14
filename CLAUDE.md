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

## Known issues

- State knowledge for already-processed episodes gets duplicated on re-run (Phase D appends without dedup)
- 7 episodes have 0 entities extracted (the prompt's entity schema lost field specs in a cleanup edit)
- If restarting `process-playlist`, manually delete `state_knowledge.json` for that episode to force re-extraction
- The `resolve_maybe` prompt exists but may default back to permissive behavior

## Useful files

- `output/universe_state.json` — cross-episode knowledge base
- `prompts/classify_chunks_global.txt` — classification prompt (the main lever for compression aggressiveness)
- `prompts/extract_knowledge_fast.txt` — knowledge extraction prompt
- `AGENT.md` — engineering practices guide
