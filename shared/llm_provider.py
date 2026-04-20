"""LLM provider abstraction for all designs.

Supports both Ollama (local) and API providers (Anthropic, OpenAI).
All designs should use this instead of direct API calls.

Configuration via environment variables:
    RECOMMEND_LLM_PROVIDER  - "ollama" | "anthropic" | "openai"  (default: "ollama")
    RECOMMEND_LLM_MODEL     - model name for text generation
                              (default depends on provider)
    RECOMMEND_EMBED_MODEL   - model name for embeddings
                              (default depends on provider)
    RECOMMEND_OLLAMA_URL    - Ollama base URL (default: "http://localhost:11434")
    ANTHROPIC_API_KEY       - required when provider is "anthropic"
    OPENAI_API_KEY          - required when provider is "openai"
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass

import httpx

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Default models per provider
# ---------------------------------------------------------------------------
_DEFAULT_MODELS: dict[str, dict[str, str]] = {
    "ollama": {
        "llm": "qwen2.5:3b",
        "embed": "nomic-embed-text",
    },
    "anthropic": {
        "llm": "claude-sonnet-4-20250514",
        "embed": "voyage-3",  # Anthropic recommends Voyage for embeddings
    },
    "openai": {
        "llm": "gpt-4o-mini",
        "embed": "text-embedding-3-small",
    },
}

# ---------------------------------------------------------------------------
# Timeout / retry settings (kept conservative for a POC)
# ---------------------------------------------------------------------------
_TIMEOUT = httpx.Timeout(600.0, connect=10.0)
_MAX_RETRIES = 2


# ---------------------------------------------------------------------------
# Provider implementation
# ---------------------------------------------------------------------------
@dataclass
class LLMProvider:
    """Unified LLM interface wrapping Ollama, Anthropic, or OpenAI.

    Usage::

        from shared import get_provider

        llm = get_provider()
        text = llm.generate("Summarize this review: ...")
        vec  = llm.embed("lightweight carbon ski")
    """

    provider: str
    llm_model: str
    embed_model: str
    ollama_url: str = "http://localhost:11434"
    api_key: str | None = None

    # ------------------------------------------------------------------
    # Text generation
    # ------------------------------------------------------------------
    def generate(self, prompt: str, *, json_mode: bool = False) -> str:
        """Generate text from *prompt*.

        Args:
            prompt: The full prompt string.
            json_mode: When True, instruct the model to respond with valid
                       JSON (provider support varies; best-effort).

        Returns:
            The generated text. When *json_mode* is True the string should
            be parseable JSON, but callers should still handle parse errors.

        Raises:
            RuntimeError: On unrecoverable provider errors after retries.
        """
        if self.provider == "ollama":
            return self._generate_ollama(prompt, json_mode)
        if self.provider == "anthropic":
            return self._generate_anthropic(prompt, json_mode)
        if self.provider == "openai":
            return self._generate_openai(prompt, json_mode)
        raise ValueError(f"Unknown provider: {self.provider}")

    # ------------------------------------------------------------------
    # Embeddings
    # ------------------------------------------------------------------
    def embed(self, text: str) -> list[float]:
        """Return an embedding vector for *text*.

        Args:
            text: Input text to embed.

        Returns:
            List of floats representing the embedding vector.

        Raises:
            RuntimeError: On unrecoverable provider errors after retries.
        """
        if self.provider == "ollama":
            return self._embed_ollama(text)
        if self.provider == "anthropic":
            return self._embed_voyage(text)
        if self.provider == "openai":
            return self._embed_openai(text)
        raise ValueError(f"Unknown provider: {self.provider}")

    # ------------------------------------------------------------------
    # Ollama implementations
    # ------------------------------------------------------------------
    def _generate_ollama(self, prompt: str, json_mode: bool) -> str:
        import re

        url = f"{self.ollama_url}/api/generate"
        payload: dict = {
            "model": self.llm_model,
            "prompt": prompt,
            "stream": False,
        }
        # Don't use format:json — it breaks thinking models (Qwen 3.5, etc.)
        # that return empty responses when constrained. Instead, rely on the
        # prompt to request JSON and strip thinking tags from the response.
        resp = self._post(url, payload)
        text = resp["response"]

        # Strip <think>...</think> blocks from thinking models
        if "<think>" in text:
            text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()

        # When json_mode requested, extract the JSON object/array from the
        # response — models often wrap JSON in markdown fences or extra text.
        if json_mode and text:
            # Strip markdown code fences
            text = re.sub(r"```(?:json)?\s*", "", text)
            text = re.sub(r"```\s*$", "", text)
            # Find the outermost JSON object or array
            start = -1
            brace = None
            for i, ch in enumerate(text):
                if ch in ('{', '['):
                    start = i
                    brace = ch
                    break
            if start >= 0:
                close = '}' if brace == '{' else ']'
                depth = 0
                end = start
                for i in range(start, len(text)):
                    if text[i] == brace:
                        depth += 1
                    elif text[i] == close:
                        depth -= 1
                        if depth == 0:
                            end = i
                            break
                text = text[start:end + 1]

        return text

    def _embed_ollama(self, text: str) -> list[float]:
        url = f"{self.ollama_url}/api/embed"
        payload = {"model": self.embed_model, "input": text}
        resp = self._post(url, payload)
        # Ollama returns {"embeddings": [[...]]}
        return resp["embeddings"][0]

    # ------------------------------------------------------------------
    # Anthropic implementation
    # ------------------------------------------------------------------
    def _generate_anthropic(self, prompt: str, json_mode: bool) -> str:
        if not self.api_key:
            raise RuntimeError(
                "ANTHROPIC_API_KEY must be set when provider is 'anthropic'"
            )
        url = "https://api.anthropic.com/v1/messages"
        headers = {
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        }
        system_parts = []
        if json_mode:
            system_parts.append(
                "Respond ONLY with valid JSON. No markdown fences, no commentary."
            )
        payload: dict = {
            "model": self.llm_model,
            "max_tokens": 4096,
            "messages": [{"role": "user", "content": prompt}],
        }
        if system_parts:
            payload["system"] = " ".join(system_parts)
        resp = self._post(url, payload, headers=headers)
        # Anthropic returns {"content": [{"type": "text", "text": "..."}]}
        return resp["content"][0]["text"]

    def _embed_voyage(self, text: str) -> list[float]:
        """Use Voyage AI for embeddings (recommended by Anthropic).

        Requires ANTHROPIC_API_KEY — Voyage accepts the same key when
        accessed via the Anthropic-partnered endpoint.  Falls back to
        OpenAI-compatible embedding if Voyage is unavailable.
        """
        if not self.api_key:
            raise RuntimeError(
                "ANTHROPIC_API_KEY must be set for Voyage embeddings"
            )
        url = "https://api.voyageai.com/v1/embeddings"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "content-type": "application/json",
        }
        payload = {
            "model": self.embed_model,
            "input": [text],
            "input_type": "document",
        }
        resp = self._post(url, payload, headers=headers)
        return resp["data"][0]["embedding"]

    # ------------------------------------------------------------------
    # OpenAI implementations
    # ------------------------------------------------------------------
    def _generate_openai(self, prompt: str, json_mode: bool) -> str:
        if not self.api_key:
            raise RuntimeError(
                "OPENAI_API_KEY must be set when provider is 'openai'"
            )
        url = "https://api.openai.com/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "content-type": "application/json",
        }
        payload: dict = {
            "model": self.llm_model,
            "messages": [{"role": "user", "content": prompt}],
        }
        if json_mode:
            payload["response_format"] = {"type": "json_object"}
        resp = self._post(url, payload, headers=headers)
        return resp["choices"][0]["message"]["content"]

    def _embed_openai(self, text: str) -> list[float]:
        if not self.api_key:
            raise RuntimeError(
                "OPENAI_API_KEY must be set for OpenAI embeddings"
            )
        url = "https://api.openai.com/v1/embeddings"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "content-type": "application/json",
        }
        payload = {
            "model": self.embed_model,
            "input": text,
        }
        resp = self._post(url, payload, headers=headers)
        return resp["data"][0]["embedding"]

    # ------------------------------------------------------------------
    # HTTP helper with retries
    # ------------------------------------------------------------------
    def _post(
        self,
        url: str,
        payload: dict,
        *,
        headers: dict[str, str] | None = None,
    ) -> dict:
        """POST JSON and return parsed response, with retries."""
        last_exc: Exception | None = None
        for attempt in range(_MAX_RETRIES + 1):
            try:
                with httpx.Client(timeout=_TIMEOUT) as client:
                    r = client.post(url, json=payload, headers=headers or {})
                    r.raise_for_status()
                    return r.json()
            except (httpx.HTTPStatusError, httpx.TransportError) as exc:
                last_exc = exc
                logger.warning(
                    "Request to %s failed (attempt %d/%d): %s",
                    url,
                    attempt + 1,
                    _MAX_RETRIES + 1,
                    exc,
                )
        raise RuntimeError(
            f"All {_MAX_RETRIES + 1} attempts to {url} failed"
        ) from last_exc


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------
def get_provider() -> LLMProvider:
    """Build an ``LLMProvider`` from environment variables.

    Environment variables (all optional — sensible defaults are used):

    * ``RECOMMEND_LLM_PROVIDER`` — ``"ollama"`` (default), ``"anthropic"``,
      or ``"openai"``.
    * ``RECOMMEND_LLM_MODEL`` — Override the text-generation model name.
    * ``RECOMMEND_EMBED_MODEL`` — Override the embedding model name.
    * ``RECOMMEND_OLLAMA_URL`` — Base URL for Ollama
      (default ``http://localhost:11434``).
    * ``ANTHROPIC_API_KEY`` / ``OPENAI_API_KEY`` — API keys for cloud
      providers.
    """
    provider = os.environ.get("RECOMMEND_LLM_PROVIDER", "ollama").lower()
    if provider not in _DEFAULT_MODELS:
        raise ValueError(
            f"RECOMMEND_LLM_PROVIDER must be one of {list(_DEFAULT_MODELS)}, "
            f"got {provider!r}"
        )

    defaults = _DEFAULT_MODELS[provider]
    llm_model = os.environ.get("RECOMMEND_LLM_MODEL", defaults["llm"])
    embed_model = os.environ.get("RECOMMEND_EMBED_MODEL", defaults["embed"])
    ollama_url = os.environ.get("RECOMMEND_OLLAMA_URL", "http://localhost:11434")

    # Resolve API key
    api_key: str | None = None
    if provider == "anthropic":
        api_key = os.environ.get("ANTHROPIC_API_KEY")
    elif provider == "openai":
        api_key = os.environ.get("OPENAI_API_KEY")

    return LLMProvider(
        provider=provider,
        llm_model=llm_model,
        embed_model=embed_model,
        ollama_url=ollama_url,
        api_key=api_key,
    )
