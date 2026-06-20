"""Configuration defaults for podcastcondensor — DeepSeek-only, entry-classify pipeline."""

import os
from dataclasses import dataclass


@dataclass
class Config:
    # DeepSeek
    deepseek_model: str = "deepseek-chat"
    deepseek_timeout: int = 600
    deepseek_max_tokens: int = 16000

    # Prompt paths
    global_state_prompt_path: str = ""
    classify_raw_prompt_path: str = ""

    # Interval / padding
    output_merge_gap: float = 2.0
    pad_before: float = 0.35
    pad_after: float = 0.5
    cluster_gap: float = 3.0

    # Audio
    audio_format: str = "mp3"
    audio_sample_rate: int = 22050
    audio_bitrate: str = "64k"
    audio_speed: float = 1.25
    audio_strategy: str = "sequential_copy"
    audio_parallel_workers: int = 2
    audio_safe_batch_size: int = 6

    # Operation
    output_root: str = ""
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
        if not self.classify_raw_prompt_path:
            self.classify_raw_prompt_path = os.path.join(
                base, "prompts", "classify_raw.txt"
            )
