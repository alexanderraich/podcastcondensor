"""LLM provider abstraction layer.

Provides a common interface for different LLM backends (Ollama, DeepSeek, etc.)
and factory functions to select the appropriate provider by config.
"""

import logging
from typing import Optional

from podcastcondensor.llm.base import LLMClient
from podcastcondensor.llm.ollama import OllamaClient
from podcastcondensor.llm.deepseek import DeepSeekClient

logger = logging.getLogger(__name__)


def create_llm_client(
    provider: str,
    *,
    host: str = "http://localhost:11434",
    base_url: Optional[str] = None,
    api_key: Optional[str] = None,
    model: Optional[str] = None,
) -> LLMClient:
    """Factory: return an ``LLMClient`` for the given provider name.

    Args:
        provider: ``"ollama"`` or ``"deepseek"`` (case-insensitive).
        host: Ollama host URL (used when provider == "ollama").
        base_url: DeepSeek base URL (used when provider == "deepseek").
        api_key: DeepSeek API key (falls back to ``ANTHROPIC_AUTH_TOKEN`` or ``DEEPSEEK_API_KEY`` env vars).
        model: Model override (e.g. ``"deepseek-chat"``).

    Returns:
        An initialized ``LLMClient`` instance.

    Raises:
        ValueError: Unknown provider name.
    """
    provider = provider.lower().strip()

    if provider == "ollama":
        logger.debug("Creating Ollama client: host=%s", host)
        return OllamaClient(host=host)

    if provider == "deepseek":
        from podcastcondensor.llm.deepseek import DEFAULT_BASE_URL, DEFAULT_MODEL
        cls = DeepSeekClient(
            base_url=base_url or DEFAULT_BASE_URL,
            api_key=api_key,
            model=model or DEFAULT_MODEL,
        )
        logger.debug("Creating DeepSeek client: base_url=%s", cls._base_url)
        return cls

    raise ValueError(
        f"Unknown LLM provider: {provider!r}. "
        f"Supported: 'ollama', 'deepseek'"
    )


__all__ = [
    "LLMClient",
    "OllamaClient",
    "DeepSeekClient",
    "create_llm_client",
]
