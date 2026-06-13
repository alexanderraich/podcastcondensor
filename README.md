# podcastcondensor

Local-first podcast condensing pipeline. Takes a YouTube video, downloads audio and subtitles, uses a local LLM (via Ollama) to classify transcript chunks as keep/drop/maybe, and produces a condensed audio file free of filler, banter, and slow windups.

## How It Works

1. **Download** — audio + subtitles via `yt-dlp` (prefers manual subs, falls back to auto)
2. **Parse & chunk** — normalize `.vtt`/`.srt` into timestamped chunks, merge small entries
3. **Classify** — send chunks to a local Ollama Qwen model; each chunk gets `keep`, `drop`, or `maybe`
4. **Resolve maybes** — optional second pass that looks at contextual neighbors
5. **Build intervals** — merge adjacent kept regions, apply padding, remove overlaps
6. **Cut audio** — extract segments with ffmpeg, concat into final condensed file
7. **Review** — stats, compression ratio, uncertain regions listed in a review file

## WSL Dependency Setup

```bash
# System packages
sudo apt update
sudo apt install -y ffmpeg python3 python3-pip python3-venv jq curl git

# yt-dlp (latest, robust method)
sudo apt install -y yt-dlp 2>/dev/null || \
  sudo curl -L https://github.com/yt-dlp/yt-dlp/releases/latest/download/yt-dlp \
    -o /usr/local/bin/yt-dlp && sudo chmod a+rx /usr/local/bin/yt-dlp

# Verify
ffmpeg -version | head -1
yt-dlp --version
python3 --version
```

## Ollama Setup

```bash
# Install Ollama for Linux (works in WSL)
curl -fsSL https://ollama.com/install.sh | sh

# Start Ollama server
ollama serve &

# Pull the recommended local model (qwen3:8b — fits GTX 1060 6GB)
ollama pull qwen3:8b

# Fallback if needed (smaller)
ollama pull qwen2.5:7b

# Verify
python3 -m podcastcondensor status
```

### GPU Notes for GTX 1060 6GB

- **qwen3:8b** is the sweet spot — fits in 6GB VRAM with room for context
- **qwen2.5:7b** is a lighter fallback if qwen3:8b is unstable
- Do **not** try 14B-class models (qwen2.5:14b, etc.) — they will OOM
- If Ollama doesn't detect your GPU, check: `ollama run qwen3:8b` and watch VRAM
- WSL2 should expose the NVIDIA GPU via CUDA if you have the [NVIDIA CUDA on WSL driver](https://developer.nvidia.com/cuda/wsl) installed

## Quick Start

```bash
# Create and activate virtual environment
cd /mnt/c/Users/raich/Projects/podcastcondensor
python3 -m venv venv
source venv/bin/activate

# Install (only requests needed for Ollama HTTP API)
pip install requests

# Smoke test — one YouTube URL
python3 -m podcastcondensor run "https://www.youtube.com/watch?v=uo8G_NuOE0E" --lang en --model qwen3:8b
```

## CLI Usage

```bash
# Run pipeline
python3 -m podcastcondensor run <youtube-url> [options]

# Check Ollama health + available models
python3 -m podcastcondensor status

# Options for `run`:
#   --model qwen3:8b           Ollama model to use
#   --lang en                  Subtitle language
#   --output-dir ./output      Output directory
#   --merge-gap 2.0            Max gap (seconds) to merge kept intervals
#   --pad-before 0.35          Padding before each interval (seconds)
#   --pad-after 0.5            Padding after each interval (seconds)
#   --no-resolve-maybe         Skip the second-pass maybe resolution
#   --dry-run                  Download only, skip LLM/audio
#   --keep-temp                Keep temp segment files for debugging
#   --prefer-auto-subs         Prefer auto-generated captions
#   --max-chunks-per-batch 20  Chunks per LLM batch
#   --max-chars-per-chunk 600  Max chars per chunk before truncation
```

## Output Structure

Every run creates a timestamped directory under `output/`:

```
output/
  20251201_143012_<video-id>/
    source_subtitles.srt      # Original subtitle file used
    normalized_chunks.json    # After merging small entries
    first_pass_decisions.json # keep/drop/maybe per chunk
    maybe_resolution.json     # Resolution of maybe chunks (if enabled)
    keep_intervals.json       # Final cut intervals with padding
    stats.json                # Durations and compression stats
    review.md                 # Human-readable summary
    condensed_<id>.mp3        # The condensed audio
    pipeline_manifest.json    # Full run metadata
```

## Model Recommendations

| Use Case | Model | Hardware |
|---|---|---|
| Local classification (default) | `qwen3:8b` | GTX 1060 6GB |
| Local fallback (lighter) | `qwen2.5:7b` | Any GPU 4GB+ |
| Remote coding (cost-speed) | `deepseek-chat` / DeepSeek V3.2 | Cloud API |

Do not run 14B+ models locally on this hardware.

## Known Limitations

- Requires Ollama running locally with a Qwen model pulled
- No manual-caption videos will use auto-generated subs (quality varies)
- Very long videos (>3h) may take a while to classify (batching is slow on local GPU)
- The model may occasionally produce malformed JSON — the pipeline retries with smaller batches
- Video must have captions (auto or manual) — no-caption videos cannot be processed
- ffmpeg full re-encode per segment (slower but reliable — no sync issues)

## Troubleshooting

| Problem | Fix |
|---|---|
| `Ollama is not running` | Run `ollama serve &` in another terminal |
| `Model not found` | Run `ollama pull qwen3:8b` |
| CUDA out of memory | Use `ollama pull qwen2.5:7b` and pass `--model qwen2.5:7b` |
| `yt-dlp: command not found` | Install via `sudo curl` command above |
| `ffmpeg: command not found` | `sudo apt install ffmpeg` |
| Git push fails | See manual auth step below |

## Manual GitHub Auth

If `git push` fails with authentication required:

```bash
# Option 1: Use GitHub CLI
gh auth login
git push -u origin main

# Option 2: Use personal access token
git remote set-url origin https://<token>@github.com/alexanderraich/podcastcondensor.git
git push -u origin main
```

## Architecture

```
podcastcondensor/
├── README.md
├── requirements.txt
├── .gitignore
├── prompts/
│   ├── classify_chunks.txt     # LLM prompt for first-pass classification
│   └── resolve_maybe.txt       # LLM prompt for maybe resolution
├── output/                     # All run artifacts
├── src/
│   └── podcastcondensor/
│       ├── __init__.py
│       ├── __main__.py         # python -m entry point
│       ├── cli.py              # Argument parsing and dispatch
│       ├── config.py           # Configuration defaults
│       ├── downloader.py       # yt-dlp audio + subtitle download
│       ├── subtitles.py        # .srt/.vtt parsing and normalization
│       ├── chunker.py          # Merge small entries into semantic chunks
│       ├── ollama_client.py    # Ollama HTTP API interaction
│       ├── classifier.py       # LLM classification orchestration
│       ├── intervals.py        # Keep-interval generation and merging
│       ├── audio.py            # ffmpeg segment cutting and concat
│       └── pipeline.py         # Full pipeline orchestration
```

## Principles

- **Subtitles as source of truth** — the LLM classifies existing text, never rewrites or summarizes
- **Conservative dropping** — when in doubt, prefer keep or maybe over drop
- **Local-first** — everything runs on your machine, no cloud API costs
- **Inspectable** — every intermediate artifact is saved as JSON or text
- **No paraphrasing** — the LLM only labels chunks, it never modifies transcript content
