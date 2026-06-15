"""DeepSeek LLM client — OpenAI-compatible HTTP transport.

Uses the ``requests`` library to call the DeepSeek Chat Completions API,
which follows the OpenAI message format.  No OpenAI SDK dependency.
"""

import json
import logging
import os
import time
from typing import Optional

import requests

from podcastcondensor.llm.base import (
    LLMClient,
    LLMConnectionError,
    LLMResponseError,
    LLMTimeoutError,
)

logger = logging.getLogger(__name__)

# Env var names checked (in order) for the API key.
# ANTHROPIC_AUTH_TOKEN is the primary; DEEPSEEK_API_KEY is the legacy fallback.
ENV_API_KEY_VARS = ("ANTHROPIC_AUTH_TOKEN", "DEEPSEEK_API_KEY")
# Used in error/log messages — refers to the preferred var name.
ENV_API_KEY = ENV_API_KEY_VARS[0]
DEFAULT_BASE_URL = "https://api.deepseek.com/v1"
DEFAULT_MODEL = "deepseek-chat"


def resolve_api_key(api_key: Optional[str] = None) -> str:
    """Resolve the DeepSeek API key from explicit arg or env vars.

    Checks env vars in order: ``ANTHROPIC_AUTH_TOKEN``, ``DEEPSEEK_API_KEY``.
    Returns empty string if none found.  Whitespace (including trailing
    carriage returns from .env files) is stripped.
    """
    if api_key:
        return api_key.strip()
    for var in ENV_API_KEY_VARS:
        value = os.environ.get(var, "")
        if value:
            return value.strip()
    return ""


class DeepSeekClient(LLMClient):
    """LLM client for DeepSeek via the OpenAI-compatible Chat Completions API.

    Configuration (in order of precedence — constructor arg > env vars):
        - ``base_url``:  API base URL (default ``https://api.deepseek.com/v1``)
        - ``api_key``:   API key; falls back to ``ANTHROPIC_AUTH_TOKEN``,
                         then ``DEEPSEEK_API_KEY`` env var
        - ``model``:     Model name (default ``deepseek-chat``)

    Raises ``LLMConnectionError`` on auth/network failures and
    ``LLMResponseError`` on API error status codes.
    """

    def __init__(
        self,
        base_url: str = DEFAULT_BASE_URL,
        api_key: Optional[str] = None,
        model: str = DEFAULT_MODEL,
        request_timeout: int = 300,
    ):
        self._base_url = base_url.rstrip("/")
        self._api_key = resolve_api_key(api_key)
        self._model = model
        self._request_timeout = request_timeout

        if not self._api_key:
            logger.warning(
                "DeepSeek API key not provided — set %s or %s env var, "
                "or pass api_key",
                *ENV_API_KEY_VARS,
            )

    # ------------------------------------------------------------------
    # Public helpers
    # ------------------------------------------------------------------

    @property
    def model(self) -> str:
        return self._model

    @property
    def configured(self) -> bool:
        """True if an API key is available (via constructor or env var)."""
        return bool(self._api_key)

    # ------------------------------------------------------------------
    # LLMClient interface
    # ------------------------------------------------------------------

    def generate(
        self,
        prompt: str,
        *,
        model: Optional[str] = None,
        timeout: Optional[int] = None,
        temperature: float = 0.1,
        max_tokens: int = 8192,
        system: Optional[str] = None,
        force_json: bool = False,
    ) -> str:
        """Send a chat-completion request and return the assistant's reply text.

        The prompt is sent as a ``user`` message.  If ``system`` is provided
        it becomes the ``system`` message.  JSON mode is enabled via
        ``response_format: {"type": "json_object"}`` when ``force_json=True``.

        Raises:
            LLMConnectionError: API key missing, network error, or auth failure.
            LLMTimeoutError: Request exceeded timeout.
            LLMResponseError: API returned a non-2xx status.
        """
        effective_model = model or self._model
        effective_timeout = timeout or self._request_timeout

        if not self._api_key:
            vars_help = " or ".join(f"${v}" for v in ENV_API_KEY_VARS)
            raise LLMConnectionError(
                f"DeepSeek API key is not set.  Set the {vars_help} "
                f"environment variable or pass ``api_key`` to the constructor."
            )

        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        body: dict = {
            "model": effective_model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": False,
        }
        if force_json:
            body["response_format"] = {"type": "json_object"}

        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }

        url = f"{self._base_url}/chat/completions"
        logger.debug(
            "DeepSeek request: model=%s  chars=%d  timeout=%d  json=%s",
            effective_model, len(prompt), effective_timeout, force_json,
        )

        t0 = time.time()
        try:
            resp = requests.post(
                url,
                headers=headers,
                json=body,
                timeout=effective_timeout,
            )
        except requests.Timeout:
            raise LLMTimeoutError(
                f"DeepSeek request timed out after {effective_timeout}s"
            )
        except requests.ConnectionError as exc:
            raise LLMConnectionError(
                f"Cannot reach DeepSeek API at {self._base_url}: {exc}"
            )

        elapsed = time.time() - t0

        if resp.status_code == 401:
            vars_help = " or ".join(f"${v}" for v in ENV_API_KEY_VARS)
            raise LLMConnectionError(
                f"DeepSeek authentication failed (401).  Check your {vars_help}."
            )
        if resp.status_code == 429:
            raise LLMResponseError(
                f"DeepSeek rate limited (429).  Retry after backoff."
            )
        if resp.status_code != 200:
            raise LLMResponseError(
                f"DeepSeek API error {resp.status_code}: {resp.text[:500]}"
            )

        try:
            data = resp.json()
        except json.JSONDecodeError as exc:
            raise LLMResponseError(f"DeepSeek returned non-JSON body: {exc}")

        choices = data.get("choices", [])
        if not choices:
            raise LLMResponseError("DeepSeek returned zero choices in response")

        content = (choices[0].get("message") or {}).get("content", "").strip()
        logger.debug(
            "DeepSeek response: %d chars in %.1fs  (finish_reason=%s)",
            len(content), elapsed,
            choices[0].get("finish_reason", "?"),
        )
        return content

    def name(self) -> str:
        return "deepseek"
