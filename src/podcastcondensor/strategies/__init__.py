"""Strategy abstractions and factory for LLM-dependent pipeline phases.

Usage::

    from podcastcondensor.strategies import create_classifier, create_knowledge_extractor

    classifier = create_classifier(
        provider="deepseek",
        model="deepseek-chat",
        prompt_path="prompts/classify_chunks_global.txt",
        base_url="https://api.deepseek.com/v1",
    )
    decisions = classifier.classify_segments(segments, ...)
"""

from typing import Optional

from podcastcondensor.llm import create_llm_client, LLMClient
from podcastcondensor.strategies.base import (
    ClassifierStrategy,
    KnowledgeExtractionStrategy,
)
from podcastcondensor.strategies.classification import (
    OllamaClassifierStrategy,
    DeepSeekClassifierStrategy,
)
from podcastcondensor.strategies.knowledge import (
    OllamaKnowledgeExtractionStrategy,
    DeepSeekKnowledgeExtractionStrategy,
)


def create_classifier(
    provider: str,
    *,
    # Shared
    prompt_path: str,
    resolve_maybe_prompt_path: str = "",
    model: Optional[str] = None,
    timeout: int = 600,
    max_segments_per_batch: int = 3,
    # Ollama-specific
    ollama_host: str = "http://localhost:11434",
    # DeepSeek-specific
    deepseek_base_url: Optional[str] = None,
    deepseek_api_key: Optional[str] = None,
) -> ClassifierStrategy:
    """Factory: build a ``ClassifierStrategy`` for the named provider.

    Args:
        provider: ``"ollama"`` or ``"deepseek"``.
        prompt_path: Path to classification prompt template.
        resolve_maybe_prompt_path: Path to resolve-maybe prompt template.
        model: Model identifier (provider-specific default if omitted).
        timeout: Request timeout in seconds.
        max_segments_per_batch: Segments per classification batch.
        ollama_host: Ollama API base URL.
        deepseek_base_url: DeepSeek API base URL.
        deepseek_api_key: DeepSeek API key (falls back to ``ANTHROPIC_AUTH_TOKEN`` or ``DEEPSEEK_API_KEY`` env vars).

    Returns:
        Initialized ``ClassifierStrategy``.
    """
    provider = provider.lower().strip()

    if provider == "ollama":
        effective_model = model or "qwen2.5:7b"
        return OllamaClassifierStrategy(
            model=effective_model,
            prompt_path=prompt_path,
            host=ollama_host,
            ollama_timeout=timeout,
            resolve_maybe_prompt_path=resolve_maybe_prompt_path,
        )

    if provider == "deepseek":
        from podcastcondensor.llm.deepseek import DEFAULT_MODEL
        client = _make_deepseek_client(deepseek_base_url, deepseek_api_key)
        effective_model = model or DEFAULT_MODEL
        return DeepSeekClassifierStrategy(
            client=client,
            prompt_path=prompt_path,
            resolve_maybe_prompt_path=resolve_maybe_prompt_path,
            model=effective_model,
            timeout=timeout,
            max_segments_per_batch=max_segments_per_batch,
        )

    raise ValueError(f"Unknown classifier provider: {provider!r}")


def create_knowledge_extractor(
    provider: str,
    *,
    # Shared
    prompt_path: str,
    model: Optional[str] = None,
    timeout: int = 300,
    # Ollama-specific
    ollama_host: str = "http://localhost:11434",
    # DeepSeek-specific
    deepseek_base_url: Optional[str] = None,
    deepseek_api_key: Optional[str] = None,
) -> KnowledgeExtractionStrategy:
    """Factory: build a ``KnowledgeExtractionStrategy`` for the named provider.

    Args:
        provider: ``"ollama"`` or ``"deepseek"``.
        prompt_path: Path to extraction prompt template.
        model: Model identifier.
        timeout: Request timeout in seconds.
        ollama_host: Ollama API base URL.
        deepseek_base_url: DeepSeek API base URL.
        deepseek_api_key: DeepSeek API key (falls back to ``ANTHROPIC_AUTH_TOKEN`` or ``DEEPSEEK_API_KEY`` env vars).

    Returns:
        Initialized ``KnowledgeExtractionStrategy``.
    """
    provider = provider.lower().strip()

    if provider == "ollama":
        effective_model = model or "qwen2.5:3b"
        return OllamaKnowledgeExtractionStrategy(
            model=effective_model,
            prompt_path=prompt_path,
            host=ollama_host,
            timeout=timeout,
        )

    if provider == "deepseek":
        from podcastcondensor.llm.deepseek import DEFAULT_MODEL
        client = _make_deepseek_client(deepseek_base_url, deepseek_api_key)
        effective_model = model or DEFAULT_MODEL
        return DeepSeekKnowledgeExtractionStrategy(
            client=client,
            prompt_path=prompt_path,
            model=effective_model,
            timeout=timeout,
        )

    raise ValueError(f"Unknown knowledge provider: {provider!r}")


# ------------------------------------------------------------------
# Internal helpers
# ------------------------------------------------------------------


def _make_deepseek_client(
    base_url: Optional[str] = None,
    api_key: Optional[str] = None,
) -> LLMClient:
    """Build a ``DeepSeekClient``, validating that the API key is present."""
    from podcastcondensor.llm.deepseek import DeepSeekClient, DEFAULT_BASE_URL, resolve_api_key, ENV_API_KEY_VARS

    effective_key = resolve_api_key(api_key)
    if not effective_key:
        vars_help = " or ".join(f"${v}" for v in ENV_API_KEY_VARS)
        raise ValueError(
            f"DeepSeek API key is not set.  Set the {vars_help} "
            f"environment variable or pass ``deepseek_api_key`` to the factory."
        )

    return DeepSeekClient(
        base_url=base_url or DEFAULT_BASE_URL,
        api_key=effective_key,
    )
