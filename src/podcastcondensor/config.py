"""Configuration defaults for podcastcondensor."""

import os
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Config:
    # Ollama
    ollama_host: str = "http://localhost:11434"
    default_model: str = "qwen3:8b"
    fallback_model: str = "qwen2.5:7b"  # smaller fallback if qwen3:8b unavailable
    ollama_timeout: int = 120  # seconds per request

    # Classification
    max_chars_per_chunk: int = 600
    max_chunks_per_batch: int = 20
    classify_prompt_path: str = ""
    resolve_maybe_prompt_path: str = ""

    # Chunking
    merge_gap_seconds: float = 0.5  # merge subtitles within this gap

    # Interval / padding
    output_merge_gap: float = 2.0  # seconds between kept intervals to merge
    pad_before: float = 0.35
    pad_after: float = 0.5

    # Audio
    audio_format: str = "mp3"
    audio_sample_rate: int = 22050  # adequate for speech
    audio_bitrate: str = "64k"  # speech doesn't need high bitrate

    # Output
    output_root: str = ""
    resolve_maybe: bool = True
    keep_temp: bool = False

    # Download
    prefer_auto_subs: bool = False
    lang: str = "en"

    def __post_init__(self):
        base = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "..")
        )
        if not self.output_root:
            self.output_root = os.path.join(base, "output")
        if not self.classify_prompt_path:
            self.classify_prompt_path = os.path.join(
                base, "prompts", "classify_chunks.txt"
            )
        if not self.resolve_maybe_prompt_path:
            self.resolve_maybe_prompt_path = os.path.join(
                base, "prompts", "resolve_maybe.txt"
            )
