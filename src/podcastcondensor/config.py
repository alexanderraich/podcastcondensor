"""Configuration defaults for podcastcondensor — DeepSeek-only."""

import os
from dataclasses import dataclass
from typing import Optional


@dataclass
class Config:
    # DeepSeek
    deepseek_model: str = "deepseek-chat"
    deepseek_timeout: int = 300
    deepseek_max_tokens: int = 12000

    # Segmentation (cloud only)
    segmentation_max_tokens: int = 12000

    # Prompt paths
    global_state_prompt_path: str = ""
    classify_global_prompt_path: str = ""

    # Block map
    block_size_words: int = 600
    max_blocks: int = 0

    # Classification
    max_segments_per_batch: int = 5

    # Interval / padding
    output_merge_gap: float = 2.0
    pad_before: float = 0.35
    pad_after: float = 0.5
    cluster_gap: float = 1.5

    # Listenability / continuity bridging (default OFF — opt-in)
    min_keep_ratio: float = 0.20
    enable_continuity_bias: bool = False
    bridge_gap_sec: float = 1.5
    enable_tail_detection: bool = True
    tail_fraction: float = 0.12
    tail_min_keep_fraction: float = 0.03

    # Audio
    audio_format: str = "mp3"
    audio_sample_rate: int = 22050
    audio_bitrate: str = "64k"
    audio_speed: float = 1.25
    audio_strategy: str = "single_pass_filter"
    audio_parallel_workers: int = 2
    audio_safe_batch_size: int = 5

    # Operation
    decisions_only: bool = False
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
        if not self.global_state_prompt_path:
            self.global_state_prompt_path = os.path.join(
                base, "prompts", "global_state.txt"
            )
        if not self.classify_global_prompt_path:
            self.classify_global_prompt_path = os.path.join(
                base, "prompts", "classify_chunks_global.txt"
            )
