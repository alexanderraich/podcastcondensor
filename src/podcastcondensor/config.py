"""Configuration defaults for podcastcondensor."""

import os
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Config:
    # Ollama
    ollama_host: str = "http://localhost:11434"
    default_model: str = "qwen2.5:3b"
    classify_model: str = "qwen2.5:7b"
    fallback_model: str = "qwen2.5:3b"
    ollama_timeout: int = 600

    # ------------------------------------------------------------------
    # Provider selection (strategy pattern)
    # ------------------------------------------------------------------

    # Classification provider
    classification_provider: str = "ollama"  # "ollama" | "deepseek"
    classification_model: str = ""           # empty = provider default
    classification_base_url: str = ""        # for DeepSeek
    classification_fallback_provider: str = ""  # "ollama" | "" = no fallback

    # Knowledge extraction provider
    knowledge_provider: str = "ollama"        # "ollama" | "deepseek"
    knowledge_model: str = ""                 # empty = provider default
    knowledge_base_url: str = ""              # for DeepSeek

    # DeepSeek settings (shared across phases)
    deepseek_api_key_env_var: str = "ANTHROPIC_AUTH_TOKEN"  # also falls back to DEEPSEEK_API_KEY

    # Knowledge cache fingerprint
    knowledge_cache_schema_version: str = "1"

    # ------------------------------------------------------------------
    # Audio strategy
    # ------------------------------------------------------------------

    audio_strategy: str = "single_pass_filter"  # "sequential_copy" | "parallel_copy" | "single_pass_filter" | "safe_batched"
    audio_parallel_workers: int = 2  # conservative for HDD; increase for SSD/NVMe
    audio_single_pass_batch_size: int = 100  # intervals per filter-complex batch
    audio_safe_batch_size: int = 5  # intervals per batch in safe_batched mode

    # ------------------------------------------------------------------
    # Global map
    # ------------------------------------------------------------------

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
    cluster_gap: float = 1.5  # merge kept segments within this gap into clusters

    # Listenability / coherence guardrails
    min_keep_ratio: float = 0.20         # warn if compression ratio falls below this
    enable_continuity_bias: bool = True  # bridge/context/neighbour passes after classification
    bridge_gap_sec: float = 3.0           # max gap to auto-bridge dropped->keep
    context_padding_segments: bool = True # add context around isolated keeps
    enable_tail_detection: bool = True    # detect and remove off-topic trailing content
    tail_fraction: float = 0.12           # check last 12% for off-topic tail
    tail_min_keep_fraction: float = 0.03  # <3% of kept content in last block → tail

    # Audio
    audio_format: str = "mp3"
    audio_sample_rate: int = 22050
    audio_bitrate: str = "64k"
    audio_speed: float = 1.25

    # Operation mode
    decisions_only: bool = False       # stop after Phase C + intervals, skip Phase D + audio
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

        # Resolve empty model overrides to provider defaults
        if not self.classification_model:
            if self.classification_provider == "deepseek":
                self.classification_model = "deepseek-chat"
            else:
                self.classification_model = self.classify_model

        if not self.knowledge_model:
            if self.knowledge_provider == "deepseek":
                self.knowledge_model = "deepseek-chat"
            else:
                self.knowledge_model = self.default_model
