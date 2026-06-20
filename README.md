# podcastcondensor

Condenses "Lord of Spirits" podcast episodes using DeepSeek, removing
filler, banter, ads, and repetition so you get the substance in ~70%
of the original time.

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
# Build cross-episode knowledge from episodes 1–21
python3 -m podcastcondensor build-universe \
  "https://www.youtube.com/playlist?list=PLZxCUWw2kdo1vAsOOOa3RwzwYvHbybjHR" \
  --start 1 --end 21

# Process episodes 22+ with the knowledge base
python3 -m podcastcondensor process-playlist \
  "https://www.youtube.com/playlist?list=PLZxCUWw2kdo1vAsOOOa3RwzwYvHbybjHR" \
  --state-file output/universe_state.json --start 22

# Quick API connectivity check
python3 -m podcastcondensor doctor --check
```

## Output

```
output/
  universe_state.json     # accumulated knowledge, fed back into classifier
  ep-NNN/
    source_subtitles.srt  # raw YouTube SRT
    global_state.json     # episode outline + structured extractions
    decisions.json        # per-entry keep/drop from classifier
    condensed_*.mp3       # final audio (beeps between clips, 1.25x)
    stats.json            # compression stats
```

## CLI

| Command | Description |
|---|---|
| `doctor` | Check DeepSeek API + ffmpeg |
| `build-universe` | SRT → knowledge only, no audio |
| `process-playlist` | Full pipeline over a range of episodes |

Add `--verbose` for detailed logging. See `--help` on any subcommand.
