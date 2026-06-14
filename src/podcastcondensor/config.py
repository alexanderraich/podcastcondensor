"""Configuration defaults for podcastcondensor."""

import os
from dataclasses import dataclass


@dataclass
class Config:
    # Ollama
    ollama_host: str = "http://localhost:11434"
    default_model: str = "qwen2.5:3b"
    classify_model: str = "qwen2.5:7b"
    fallback_model: str = "qwen2.5:3b"
    ollama_timeout: int = 600

    # Global map
    block_size_words: int = 600
    max_blocks: int = 0  # 0 = all blocks, N = only process first N blocks
    block_summary_prompt_path: str = ""
    outline_prompt_path: str = ""

    # Segmentation (replaces old chunking)
    segment_gap_threshold: float = 0.5
    segment_gap_sentence_threshold: float = 8.0
    segment_max_words: int = 400
    segment_min_words: int = 20
    sentence_overflow_words: int = 150
    refine_segments: bool = True
    refine_batch_size_words: int = 1000

    # Classification (segment-based)
    max_segments_per_batch: int = 5
    classify_global_prompt_path: str = ""
    classify_local_prompt_path: str = ""
    resolve_maybe_prompt_path: str = ""
    extract_concepts_prompt_path: str = ""

    # Interval / padding
    output_merge_gap: float = 2.0
    pad_before: float = 0.35
    pad_after: float = 0.5

    # Audio
    audio_format: str = "mp3"
    audio_sample_rate: int = 22050
    audio_bitrate: str = "64k"
    audio_speed: float = 1.25

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
        if not self.classify_local_prompt_path:
            self.classify_local_prompt_path = os.path.join(
                base, "prompts", "classify_chunks.txt"
            )
        if not self.classify_global_prompt_path:
            self.classify_global_prompt_path = os.path.join(
                base, "prompts", "classify_chunks_global.txt"
            )
        if not self.resolve_maybe_prompt_path:
            self.resolve_maybe_prompt_path = os.path.join(
                base, "prompts", "resolve_maybe.txt"
            )
        if not self.extract_concepts_prompt_path:
            self.extract_concepts_prompt_path = os.path.join(
                base, "prompts", "extract_knowledge_fast.txt"
            )
        if not self.block_summary_prompt_path:
            self.block_summary_prompt_path = os.path.join(
                base, "prompts", "summarize_block.txt"
            )
        if not self.outline_prompt_path:
            self.outline_prompt_path = os.path.join(
                base, "prompts", "synthesize_outline.txt"
            )
