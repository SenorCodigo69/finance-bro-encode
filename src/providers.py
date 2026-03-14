"""LLM provider abstraction layer.

Supports multiple backends via a unified interface:
- Anthropic (Claude) — via official SDK
- OpenAI-compatible APIs — Google Gemini (and any OpenAI-compat endpoint)
- Ollama (local) — native API with thinking disabled for speed
"""

from __future__ import annotations

import asyncio
import json
from abc import ABC, abstractmethod

import aiohttp

from src.utils import log


class LLMProvider(ABC):
    """Base class for LLM API providers."""

    def __init__(self, name: str, model: str):
        self.name = name
        self.model = model
        self._total_tokens = 0

    @abstractmethod
    async def chat(
        self, system: str, user: str, max_tokens: int = 2000, timeout: float = 30.0
    ) -> str:
        """Send a chat completion request. Returns the response text."""
        ...

    @property
    def total_tokens(self) -> int:
        return self._total_tokens

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name!r}, model={self.model!r})"


class AnthropicProvider(LLMProvider):
    """Claude via the Anthropic SDK."""

    def __init__(self, api_key: str, model: str = "claude-haiku-4-5-20251001"):
        super().__init__("claude", model)
        import anthropic

        self._client = anthropic.Anthropic(api_key=api_key)

    async def chat(
        self, system: str, user: str, max_tokens: int = 2000, timeout: float = 30.0
    ) -> str:
        response = await asyncio.to_thread(
            self._client.messages.create,
            model=self.model,
            max_tokens=max_tokens,
            system=system,
            messages=[{"role": "user", "content": user}],
            timeout=timeout,
        )
        text = response.content[0].text
        self._total_tokens += response.usage.input_tokens + response.usage.output_tokens
        return _strip_markdown(text)


class OpenAICompatibleProvider(LLMProvider):
    """Generic provider for OpenAI-compatible APIs.

    Works with: Google Gemini (OpenAI compat mode) and any OpenAI-compatible endpoint.
    """

    # Known base URLs for convenience
    KNOWN_URLS = {
        "gemini": "https://generativelanguage.googleapis.com/v1beta/openai",
    }

    def __init__(self, name: str, api_key: str, model: str, base_url: str | None = None):
        super().__init__(name, model)
        self._api_key = api_key
        self._base_url = (base_url or self.KNOWN_URLS.get(name, "")).rstrip("/")
        if not self._base_url:
            raise ValueError(f"No base_url for provider {name!r}. Pass one or use a known name.")
        # [OPT-8] Reuse aiohttp session across calls for connection pooling
        self._session: aiohttp.ClientSession | None = None

    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession()
        return self._session

    async def close(self):
        if self._session and not self._session.closed:
            await self._session.close()

    async def chat(
        self, system: str, user: str, max_tokens: int = 2000, timeout: float = 30.0
    ) -> str:
        url = f"{self._base_url}/chat/completions"
        headers = {"Content-Type": "application/json"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"

        payload: dict = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "max_tokens": max_tokens,
            "temperature": 0.3,
        }

        # Retry with exponential backoff for 429 (quota/rate limit)
        max_retries = 3
        session = await self._get_session()
        for attempt in range(max_retries + 1):
            async with session.post(
                url,
                json=payload,
                headers=headers,
                timeout=aiohttp.ClientTimeout(total=timeout),
            ) as resp:
                if resp.status == 429 and attempt < max_retries:
                    wait = 2 ** attempt  # 1s, 2s, 4s
                    log.warning(f"[{self.name}] 429 rate limited, retry {attempt + 1}/{max_retries} in {wait}s")
                    await asyncio.sleep(wait)
                    continue
                if resp.status != 200:
                    body = await resp.text()
                    raise RuntimeError(f"[{self.name}] HTTP {resp.status}: {body[:300]}")
                data = await resp.json()
                break

        text = data["choices"][0]["message"]["content"]
        usage = data.get("usage", {})
        self._total_tokens += usage.get("total_tokens", 0)
        return _strip_markdown(text)


class OllamaProvider(LLMProvider):
    """Ollama via native API — uses think:false for fast CPU inference.

    Supports fallback: if primary URL fails (e.g. local not running),
    tries fallback URL (e.g. remote Hetzner server).
    """

    def __init__(
        self,
        model: str,
        base_url: str = "http://localhost:11434",
        fallback_url: str | None = None,
    ):
        super().__init__("ollama", model)
        self._base_url = base_url.rstrip("/").removesuffix("/v1")
        self._fallback_url = fallback_url.rstrip("/").removesuffix("/v1") if fallback_url else None
        self._active_url: str | None = None  # Cache which URL works
        self._session: aiohttp.ClientSession | None = None

    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession()
        return self._session

    async def close(self):
        if self._session and not self._session.closed:
            await self._session.close()

    async def _try_chat(self, base: str, payload: dict, timeout: float) -> dict:
        """Attempt a chat request against a specific Ollama endpoint."""
        url = f"{base}/api/chat"
        session = await self._get_session()
        async with session.post(
            url, json=payload, timeout=aiohttp.ClientTimeout(total=timeout),
        ) as resp:
            if resp.status != 200:
                body = await resp.text()
                raise RuntimeError(f"[ollama] HTTP {resp.status}: {body[:300]}")
            return await resp.json()

    async def chat(
        self, system: str, user: str, max_tokens: int = 2000, timeout: float = 60.0
    ) -> str:
        # [SEC] Enable JSON mode when the prompt asks for JSON output
        # This prevents Ollama from returning free-text that breaks JSON callers
        wants_json = "json" in user.lower()[-200:] or "respond only with json" in user.lower()
        payload = {
            "model": self.model,
            "stream": False,
            "think": False,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "options": {"num_predict": max_tokens},
        }
        if wants_json:
            payload["format"] = "json"

        # If we already know which URL works, use it
        if self._active_url:
            data = await self._try_chat(self._active_url, payload, timeout)
        else:
            # Try primary, fall back to secondary
            try:
                data = await self._try_chat(self._base_url, payload, timeout)
                self._active_url = self._base_url
                log.info(f"[ollama] connected to {self._base_url}")
            except (aiohttp.ClientError, OSError, RuntimeError) as exc:
                if not self._fallback_url:
                    raise
                log.warning(f"[ollama] primary {self._base_url} failed ({exc}), trying fallback")
                data = await self._try_chat(self._fallback_url, payload, timeout)
                self._active_url = self._fallback_url
                log.info(f"[ollama] connected to fallback {self._fallback_url}")

        text = data["message"]["content"]
        tokens = data.get("eval_count", 0) + data.get("prompt_eval_count", 0)
        self._total_tokens += tokens
        return _strip_markdown(text)


def _strip_markdown(text: str) -> str:
    """Strip markdown code fences from LLM responses."""
    text = text.strip()
    if text.startswith("```"):
        first_newline = text.index("\n") if "\n" in text else len(text)
        text = text[first_newline + 1 :]
    if text.endswith("```"):
        text = text[:-3]
    return text.strip()


def build_providers(config) -> list[LLMProvider]:
    """Build LLM providers from config. Only returns providers with valid API keys."""
    import os

    providers: list[LLMProvider] = []

    for prov_cfg in config.agent.providers:
        if not prov_cfg.get("enabled", True):
            continue

        name = prov_cfg["name"]
        model = prov_cfg["model"]
        base_url = prov_cfg.get("base_url")

        if name == "claude":
            api_key = os.getenv("ANTHROPIC_API_KEY", "")
            if api_key:
                providers.append(AnthropicProvider(api_key, model))
                log.info(f"  + {name} ({model})")
            else:
                log.debug(f"  - {name} skipped (no ANTHROPIC_API_KEY)")

        elif name == "ollama":
            # Ollama: try local first, fall back to remote server
            fallback_url = prov_cfg.get("fallback_url") or os.getenv("OLLAMA_FALLBACK_URL")
            providers.append(
                OllamaProvider(model, base_url or "http://localhost:11434", fallback_url)
            )
            log.info(f"  + {name} ({model}) — local" + (f" (fallback: {fallback_url})" if fallback_url else ""))

        else:
            # OpenAI-compatible: gemini or custom
            env_key = prov_cfg.get("env_key") or f"{name.upper()}_API_KEY"
            api_key = os.getenv(env_key, "")
            if api_key:
                providers.append(OpenAICompatibleProvider(name, api_key, model, base_url))
                log.info(f"  + {name} ({model})")
            else:
                log.debug(f"  - {name} skipped (no {env_key})")

    return providers
