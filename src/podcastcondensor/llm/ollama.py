"""Ollama LLM client — wraps the existing ollama_client module.

This adapter delegates to the working functions in
``podcastcondensor.ollama_client`` so the existing Ollama path
is preserved exactly, while exposing the ``LLMClient`` interface.
"""

import logging
from typing import Optional

from podcastcondensor.llm.base import (
    LLMClient,
    LLMConnectionError,
    LLMResponseError,
    LLMTimeoutError,
)
from podcastcondensor.ollama_client import generate as _ollama_generate

logger = logging.getLogger(__name__)


class OllamaClient(LLMClient):
    """Adapter that wraps the existing ``ollama_client`` module functions.

    Every call delegates to the battle-tested ``generate()`` function
    from ``podcastcondensor.ollama_client``, including its error handling
    and retry-parse logic for batch calls.
    """

    def __init__(self, host: str = "http://localhost:11434"):
        self._host = host.rstrip("/")

    # ------------------------------------------------------------------
    # LLMClient interface
    # ------------------------------------------------------------------

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
        """Delegate to ``podcastcondensor.ollama_client.generate``."""
        try:
            return _ollama_generate(
                prompt=prompt,
                model=model,
                host=self._host,
                timeout=timeout,
                temperature=temperature,
                max_tokens=max_tokens,
                system=system,
                force_json=force_json,
            )
        except TimeoutError:
            raise LLMTimeoutError(f"Ollama request timed out after {timeout}s")
        except ConnectionError as exc:
            raise LLMConnectionError(f"Cannot reach Ollama at {self._host}: {exc}")
        except Exception as exc:
            raise LLMResponseError(f"Ollama request failed: {exc}")

    def name(self) -> str:
        return "ollama"
