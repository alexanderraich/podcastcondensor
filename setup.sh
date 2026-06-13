#!/usr/bin/env bash
set -euo pipefail

# podcastcondensor — bootstrap script
# Run this with: bash setup.sh
# Parts that need sudo will prompt for password.

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

echo "=== podcastcondensor setup ==="
echo ""

# 1. System deps (needs sudo)
echo ">>> Installing system packages (sudo required)..."
sudo apt update -qq
sudo apt install -y ffmpeg python3 python3-pip python3-venv jq curl zstd

# 2. yt-dlp
echo ">>> Installing yt-dlp..."
if command -v yt-dlp &>/dev/null; then
    echo "yt-dlp already installed: $(yt-dlp --version)"
else
    sudo curl -L "https://github.com/yt-dlp/yt-dlp/releases/latest/download/yt-dlp" \
        -o /usr/local/bin/yt-dlp
    sudo chmod a+rx /usr/local/bin/yt-dlp
    echo "yt-dlp: $(yt-dlp --version)"
fi

# 3. Ollama
echo ">>> Installing Ollama..."
if command -v ollama &>/dev/null; then
    echo "Ollama already installed: $(ollama --version 2>/dev/null || echo '?')"
else
    curl -fsSL https://ollama.com/install.sh | sh
fi

# 4. Python venv
echo ">>> Setting up Python virtual environment..."
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install requests

# 5. Pull models
echo ">>> Pulling default classification model..."
ollama pull qwen3:8b
echo ""
echo ">>> Pulling fallback model..."
ollama pull qwen2.5:7b

# 6. Verify
echo ""
echo "=== Verification ==="
echo "ffmpeg:  $(ffmpeg -version 2>&1 | head -1)"
echo "yt-dlp:  $(yt-dlp --version)"
echo "ollama:  $(ollama --version 2>&1 || true)"
echo "python:  $(python3 --version)"
echo ""

echo "=== Setup complete ==="
echo ""
echo "Quick smoke test:"
echo "  source venv/bin/activate"
echo "  python3 -m podcastcondensor run \"https://www.youtube.com/watch?v=uo8G_NuOE0E\" --lang en --model qwen3:8b"
echo ""
echo "Or check status first:"
echo "  python3 -m podcastcondensor status"
