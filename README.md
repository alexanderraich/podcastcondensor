# podcastcondensor

Condensing "Lord of Spirits" podcast episodes using DeepSeek LLM.

Downloads YouTube episodes, segments subtitles via DeepSeek, classifies transcript segments as keep/drop using a cross-episode universe state for context, and produces a condensed audio file free of filler, banter, and slow windups.

## Pipeline (8 phases)

1. **Download** — audio + subtitles via `yt-dlp`
2. **Punctuation** — DeepSeek adds `.` `!` `?` if auto-captions lack them (≥95% missing)
3. **Segmentation** — DeepSeek groups subtitle entries into topic segments at sentence boundaries
4. **Global map** — DeepSeek identifies natural topic blocks + produces an episode outline (single shot)
5. **Classification** — DeepSeek classifies each segment keep/drop/maybe with universe state context (single batch)
6. **Build intervals** — cluster kept segments, apply padding, merge overlaps
7. **Audio cutting** — ffmpeg with beeps (800Hz/100ms) between segments
8. **Knowledge extraction** — DeepSeek single-shot: full transcript → structured summary, entities, concepts, claims, glossary — merged into universe state

## Universe State

Rolling structured knowledge base. Per episode:
1. Programmatically clean the SRT (no LLM)
2. Build deduplicated transcript text
3. **One DeepSeek call** extracting entities, concepts, claims, scriptural links, glossary, and a narrative summary
4. Merge into accumulated state with dedup by stable ID

The state is passed as context to the classifier so it knows what's already been covered.

**`build-universe`** — SRT-only mode (no audio/segmentation) for building state from scratch.
**`process-playlist`** — full pipeline per episode; Phase 8 keeps the state up to date automatically.

## Requirements

- Python 3.10+
- DeepSeek API key in `ANTHROPIC_AUTH_TOKEN` or `DEEPSEEK_API_KEY`
- `ffmpeg`
- `yt-dlp`

```bash
sudo apt install -y ffmpeg yt-dlp python3 python3-pip
```

## Quick Start

```bash
# Build universe state from episodes 1-21
python3 -m podcastcondensor build-universe \
  "https://www.youtube.com/playlist?list=PLZxCUWw2kdo1vAsOOOa3RwzwYvHbybjHR" \
  --start 1 --end 21

# Process episodes 22+ with the built state
python3 -m podcastcondensor process-playlist \
  "https://www.youtube.com/playlist?list=PLZxCUWw2kdo1vAsOOOa3RwzwYvHbybjHR" \
  --state-file output/universe_state.json --start 22

# Check connectivity
python3 -m podcastcondensor doctor --check
```

## Output Structure

```
output/
  universe_state.json        # Accumulated knowledge across episodes
  universe.dump              # (removed — no longer used)
  ep-NNN/
    source_subtitles.srt     # Raw SRT from YouTube
    segments.json            # Phase 2 — segmented transcript
    decisions.json           # Phase 5 — keep/drop/bridge decisions
    condensed_<id>.mp3       # Phase 7 — final condensed audio
```

## CLI

| Command | Description |
|---|---|
| `doctor` | Check DeepSeek API + ffmpeg availability |
| `build-universe <url>` | Build universe state (SRT only, one DeepSeek call/episode) |
| `process-playlist <url>` | Full pipeline with existing universe state |

Global flags: `--verbose`
