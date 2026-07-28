# podcastcondensor (archived)

Condensing "Lord of Spirits" podcast episodes using DeepSeek LLM.

**Status: Archived.** The show's conversational style is fundamentally
uncuttable — the hosts meander, loop back, banter, and stretch ~5 min
of content across 2 hours. No amount of LLM prompt engineering, segment
filtering, or boundary snapping produces listenable audio cuts.

## What was built

A cross-episode knowledge base (universe state) for eps 1-40, with
whisper transcriptions, per-episode DeepSeek extractions, and a merged
knowledge graph of concepts/entities/claims with timestamp segments.

This is useful as a **searchable reference** — you can find where a
specific concept was discussed — but the audio cutting pipeline is a
dead end.

## Universe state coverage

| Episodes | SRT source | In universe state |
|----------|-----------|-------------------|
| 1-28 | Whisper | ✅ |
| 29-30 | YouTube subs + whisper | Partial (compressed.json only) |
| 31-40 | Whisper | ✅ |

Total: 64 concepts, 65 entities, 58 claims, 199 scriptural links, 48 glossary terms.

## Repository contents

- `src/podcastcondensor/` — Python package: downloader, transcriber, LLM extraction, universe state
- `prompts/global_state.txt` — The working extraction prompt
- `output/ep-NNN/` — Per-episode artifacts (SRT, global_state.json, decisions.json)
- `output/universe_state.json` — Cross-episode knowledge base

## Key files

| File | Purpose |
|------|---------|
| `src/podcastcondensor/cli.py` | CLI entry point |
| `src/podcastcondensor/playlist_pipeline.py` | `build-universe` orchestration |
| `src/podcastcondensor/universe_state.py` | Cross-episode knowledge base |
| `src/podcastcondensor/global_state.py` | Per-episode DeepSeek extraction |
| `src/podcastcondensor/subtitles.py` | SRT parsing + cleaning |
| `prompts/global_state.txt` | Extraction prompt |

## Legacy (not maintained)

- `process-playlist` — Per-episode one-shot compression
- `build-master-cut` — Multi-theme audio anthology
- `build-minimal-theme` — Single-concept audio cut
- `prompts/compress_episode.txt`, `classify_raw.txt`, `extract_themes.txt`

## Commands

```bash
# Build universe state with whisper (clean SRT, slow)
python3 -m podcastcondensor build-universe [PLAYLIST_URL] --start N --end N

# Build with YouTube subs (fragmented, fast, knowledge only)
python3 -m podcastcondensor build-universe [PLAYLIST_URL] --start N --end N --yt-subs

# Health check
python3 -m podcastcondensor doctor --check
```

## Required

- DeepSeek API key in `ANTHROPIC_AUTH_TOKEN` or `DEEPSEEK_API_KEY`
- ffmpeg, yt-dlp, faster-whisper

## Git-tracked artifacts per episode

- `source_subtitles.srt` — Whisper or YT transcription
- `global_state.json` — Per-episode DeepSeek extraction
- `decisions.json` — Legacy per-entry classifier decisions (eps 1-28)
- `stats.json` — Legacy compression stats (eps 23-29)
