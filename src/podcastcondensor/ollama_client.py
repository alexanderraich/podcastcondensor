"""Ollama HTTP API client — local LLM inference."""

import json
import logging
import subprocess
import time
from typing import Any, Optional

import requests

logger = logging.getLogger(__name__)


def check_ollama(host: str = "http://localhost:11434") -> bool:
    """Check if Ollama server is running."""
    try:
        r = requests.get(f"{host}/api/tags", timeout=5)
        return r.status_code == 200
    except requests.ConnectionError:
        return False


def list_models(host: str = "http://localhost:11434") -> list:
    """List available models from Ollama."""
    try:
        r = requests.get(f"{host}/api/tags", timeout=10)
        if r.status_code == 200:
            models = r.json().get("models", [])
            return [m["name"] for m in models]
    except Exception:
        pass
    return []


def find_best_model(
    preferred: str = "qwen3:8b",
    fallback: str = "qwen2.5:7b",
    host: str = "http://localhost:11434",
) -> Optional[str]:
    """Find the best available model, trying preferred then fallback.

    Returns model name string or None if nothing suitable found.
    """
    available = list_models(host)
    if not available:
        logger.warning("No models found in Ollama")
        return None

    logger.info("Available models: %s", available)

    # Try exact match for preferred
    for m in available:
        if m == preferred:
            return preferred

    # Try fallback
    for m in available:
        if m == fallback:
            return fallback

    # Try any qwen instruct model
    for m in available:
        if "qwen" in m.lower() and ("instruct" in m.lower() or ":8b" in m.lower() or ":7b" in m.lower()):
            return m

    # Last resort: first available model
    logger.warning("Preferred models not found, using: %s", available[0])
    return available[0]


def _pull_model(model: str, host: str = "http://localhost:11434") -> bool:
    """Pull a model via Ollama CLI."""
    logger.info("Pulling model %s (this may take a while)...", model)
    try:
        result = subprocess.run(
            ["ollama", "pull", model],
            capture_output=True, text=True, timeout=600
        )
        if result.returncode == 0:
            logger.info("Model %s pulled successfully", model)
            return True
        else:
            logger.error("Failed to pull %s: %s", model, result.stderr)
            return False
    except FileNotFoundError:
        logger.error("ollama CLI not found. Install Ollama first.")
        return False
    except subprocess.TimeoutExpired:
        logger.error("Timed out pulling model %s", model)
        return False


def ensure_model(
    model: str,
    host: str = "http://localhost:11434",
    auto_pull: bool = True,
) -> bool:
    """Ensure a model is available, optionally pulling it."""
    available = list_models(host)
    if model in available:
        return True
    if auto_pull:
        logger.info("Model %s not found, attempting to pull...", model)
        return _pull_model(model, host)
    return False


def generate(
    prompt: str,
    model: str,
    host: str = "http://localhost:11434",
    timeout: int = 120,
    temperature: float = 0.1,
    max_tokens: int = 2048,
    system: Optional[str] = None,
    force_json: bool = False,
) -> str:
    """Send a generate request to Ollama, return raw text response.

    Args:
        prompt: The prompt text
        model: Model name
        host: Ollama host URL
        timeout: Request timeout in seconds
        temperature: Sampling temperature (low for deterministic)
        max_tokens: Max tokens to generate
        system: Optional system prompt
        force_json: If True, forces Ollama to output valid JSON

    Returns:
        Raw response text
    """
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": temperature,
            "num_predict": max_tokens,
        },
    }
    if system:
        payload["system"] = system
    if force_json:
        payload["format"] = "json"

    url = f"{host}/api/generate"
    logger.debug(
        "Sending %d chars to %s (timeout=%ds temp=%.1f)",
        len(prompt), model, timeout, temperature,
    )

    t0 = time.time()
    try:
        r = requests.post(url, json=payload, timeout=timeout)
        r.raise_for_status()
        data = r.json()
        elapsed = time.time() - t0
        response = data.get("response", "").strip()
        logger.debug(
            "Response: %d chars in %.1fs", len(response), elapsed
        )
        return response
    except requests.Timeout:
        logger.error("Request timed out after %ds", timeout)
        raise
    except requests.RequestException as e:
        logger.error("Ollama request failed: %s", e)
        raise


def generate_batch(
    prompt_template: str,
    chunks: list,
    model: str,
    host: str = "http://localhost:11434",
    timeout: int = 120,
    temperature: float = 0.1,
    retries: int = 1,
    payload_override: Optional[str] = None,
) -> list:
    """Classify a batch of chunks using the local model.

    Sends prompt + JSON chunks, parses JSON response.
    Retries on parse failure.

    Args:
        prompt_template: The classification prompt text
        chunks: List of chunk dicts
        model: Model name
        host: Ollama host
        timeout: Request timeout
        temperature: Sampling temperature
        retries: Number of retries on parse failure
        payload_override: If set, use this as the full JSON payload
                         instead of building from chunks only.
                         Used for global-context classification.

    Returns list of decision dicts or raises on failure.
    """
    import json

    if payload_override:
        full_prompt = prompt_template.strip() + "\n\n" + payload_override
    else:
        payload = json.dumps({"chunks": chunks}, ensure_ascii=False, indent=2)
        full_prompt = prompt_template.strip() + "\n\n" + payload

    for attempt in range(retries + 1):
        try:
            raw = generate(
                prompt=full_prompt,
                model=model,
                host=host,
                timeout=timeout,
                temperature=temperature,
                force_json=True,
            )
            decisions = _parse_json_response(raw)
            if decisions:
                return decisions
            logger.warning(
                "Attempt %d/%d: empty or invalid response (raw: %s...)",
                attempt + 1, retries + 1, raw[:200],
            )
        except Exception as e:
            logger.warning(
                "Attempt %d/%d failed: %s", attempt + 1, retries + 1, e
            )

    raise RuntimeError(
        f"Failed to get valid classification after {retries + 1} attempts"
    )


def _parse_json_response(raw: str) -> Optional[list]:
    """Extract JSON array from model response.

    Handles:
    - Pure JSON output
    - JSON wrapped in markdown code fences
    - JSON preceded/followed by explanatory text
    """
    if not raw:
        return None

    # Try parsing raw first
    text = raw.strip()

    # Remove markdown code fences
    if text.startswith("```"):
        # Find the JSON block
        lines = text.split("\n")
        clean = []
        in_block = False
        for line in lines:
            if line.strip().startswith("```"):
                in_block = not in_block
                continue
            if in_block:
                clean.append(line)
        if clean:
            text = "\n".join(clean)

    # Try to find JSON object or array in the text
    for start_char, end_char, parser_fn in [
        ("{", "}", lambda t: _classify_parse_dict(t)),
        ("[", "]", json.loads),
    ]:
        start = text.find(start_char)
        if start >= 0:
            end = text.rfind(end_char)
            if end > start:
                candidate = text[start:end + 1]
                try:
                    result = parser_fn(candidate)
                    if isinstance(result, list):
                        return result
                except json.JSONDecodeError:
                    continue

    return None


def _classify_parse_dict(text: str) -> Optional[list]:
    """Parse a dict response, accepting multiple formats.

    Accepts:
    - {"decisions": [{"id": "...", "label":"keep", "reason":"..."}]}
    - {"classification": "keep", "reason": "...", "id": "..."}
    - {"label": "keep", "reason": "..."}
    """
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return None
    if not isinstance(data, dict):
        return None

    # Format 1: {"decisions": [...]} — normalize each item
    decisions = data.get("decisions")
    if isinstance(decisions, list):
        normalized = []
        for item in decisions:
            if not isinstance(item, dict):
                continue
            item["id"] = item.get("id") or item.get("chunk_id") or item.get("segment_id", "?")
            item["label"] = item.get("label") or item.get("classification", "maybe")
            normalized.append(item)
        return normalized if normalized else None

    # Format 2: single classification object
    label = data.get("label") or data.get("classification")
    if label in ("keep", "drop", "maybe"):
        return [{
            "id": data.get("id", data.get("chunk_id", "?")),
            "label": label,
            "reason": data.get("reason", ""),
        }]

    return None
