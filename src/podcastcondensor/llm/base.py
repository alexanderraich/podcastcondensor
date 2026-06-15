"""Abstract LLM client interface.

All LLM providers (Ollama, DeepSeek, OpenAI-compatible, etc.) implement
this interface so strategy code can remain provider-agnostic.
"""

from abc import ABC, abstractmethod
from typing import Optional


class LLMClient(ABC):
    """Transport-level client for text generation via an LLM API.

    Subclasses handle the HTTP/SDK transport, authentication, and
    any provider-specific response parsing.  The return value is always
    raw response text — strategy code is responsible for JSON parsing,
    retry logic, and fallback.
    """

    @abstractmethod
    def generate(
        self,
        prompt: str,
        *,
        model: str,
        timeout: int = 120,
        temperature: float = 0.1,
        max_tokens: int = 2048,
        system: Optional[str] = None,
        force_json: bool = False,
    ) -> str:
        """Send a completion prompt and return the raw response text.

        Args:
            prompt: The full prompt text.
            model: Model identifier (provider-specific).
            timeout: Request timeout in seconds.
            temperature: Sampling temperature.
            max_tokens: Maximum tokens to generate.
            system: Optional system prompt.
            force_json: Hint to the provider to constrain output to valid JSON.

        Returns:
            Raw response text as a string.

        Raises:
            LLMConnectionError: Network / auth failure.
            LLMTimeoutError: Request timed out.
            LLMResponseError: API returned an error status.
        """
        ...

    @abstractmethod
    def name(self) -> str:
        """Human-readable provider name, used in logging and cache fingerprints."""
        ...


class LLMConnectionError(ConnectionError):
    """Network, DNS, or authentication failure talking to the LLM API."""


class LLMTimeoutError(TimeoutError):
    """Request exceeded the configured timeout."""


class LLMResponseError(RuntimeError):
    """API returned a non-success status or malformed response."""
