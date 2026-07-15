# podcastcondensor

Condenses "Lord of Spirits" podcast episodes using DeepSeek — extracting
only the **1–2 most important discussion segments** per episode. No filler,
no banter, no ads, no mid-thought cuts.

## Pipeline (per episode)

```
Download MP3 + SRT  →  LLM picks 1–2 segments  →  Audio cut
```

Each episode is condensed independently in three phases:

1. **Download** — yt-dlp: MP3 + YouTube auto-subs (falls back to faster-whisper)
2. **LLM Compression** — single DeepSeek call: full timestamped SRT transcript →
   the 1–2 most substantive, self-contained discussion segments with timestamps
3. **Audio Cut** — extract those segments from the source audio and concatenate
   with beep separators

**The LLM default is DROP.** It sees the entire episode transcript and returns
only the segment ranges worth keeping. No world-building, no cross-episode
state, no fragile post-processing.

## Why this approach

Earlier versions tried a "universe state" — building cross-episode knowledge
then producing thematic cuts. It didn't work:

- The LLM's direct timestamp estimates were unreliable; segments cut mid-thought
- The multi-phase pipeline was complex and brittle
- The compression prompt aimed for ~50% but only caught ads and pleasantries

**Current approach:** Forget themes and cross-episode state. Condense each
episode independently with a much stricter prompt: default DROP, keep only the
1–2 most important discussions, always capture complete thoughts. The result is
shorter, cleaner, and predictable.

## Requirements

- Python 3.10+
- DeepSeek API key in `ANTHROPIC_AUTH_TOKEN` or `DEEPSEEK_API_KEY`
- `ffmpeg` + `yt-dlp`

```bash
sudo apt install -y ffmpeg yt-dlp python3 python3-pip
pip3 install openai
```

## Quick start

```bash
# Process episodes 1–29 (download → compress → audio cut)
python3 -m podcastcondensor process-playlist \
  "https://www.youtube.com/playlist?list=PLZxCUWw2kdo1vAsOOOa3RwzwYvHbybjHR" \
  --start 1 --end 29

# Quick API connectivity check
python3 -m podcastcondensor doctor --check
```

Each episode goes to `output/ep-NNN/` with stats and the condensed `.mp3`.

## Output

```
output/
  ep-NNN/
    source_subtitles.srt  # raw downloaded SRT or whisper output
    condensed_epNNN.mp3   # final audio (1–2 segments, beep-separated)
    stats.json            # compression stats
```

## CLI

| Command | Description |
|---|---|
| `doctor` | Check DeepSeek API + ffmpeg |
| `process-playlist` | Main pipeline over a range of episodes |
| `build-universe` | *(legacy)* Build cross-episode universe state |
| `build-minimal-theme` | *(legacy)* Thematic cut from universe state |
| `build-master-cut` | *(legacy)* Anthology across all episodes |

Add `--verbose` for detailed logging. See `--help` on any subcommand.

## Legacy commands (unmaintained)

`build-universe`, `build-minimal-theme`, and `build-master-cut` are from the
abandoned universe-state approach. They still exist in the repo but are no
longer the main workflow.
