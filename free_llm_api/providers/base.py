"""Provider base classes.

* ``BaseProvider`` is the minimal interface every provider implements.
* ``OpenAICompatibleProvider`` implements the very common
  ``POST /chat/completions`` request/response shape, so concrete providers
  (Groq, Together, OpenRouter, NVIDIA, ...) usually only declare a base URL.

Providers whose API differs (e.g. HuggingFace's serverless Inference API)
simply subclass ``BaseProvider`` directly — the plugin system does not care.
"""
from __future__ import annotations

import json
import logging
from typing import Any, Dict, Generator, Optional

import requests

from ..errors import (
    InvalidKeyError,
    ProviderError,
    ProviderTimeoutError,
    RateLimitError,
)

logger = logging.getLogger(__name__)


class BaseProvider:
    """The contract every provider must fulfil."""

    #: set automatically by ``@register("...")``
    provider_type: str = "base"
    #: whether an API key is mandatory (manager skips key-less providers)
    requires_key: bool = True

    def __init__(
        self,
        name: str,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        timeout: float = 20.0,
        weight: int = 1,
        extra: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.name = name
        self.api_key = api_key
        self.model = model
        self.timeout = timeout
        self.weight = weight
        self.extra = extra or {}

    def generate(self, prompt: str, **kwargs: Any) -> Dict[str, Any]:
        """Return a normalized response dict.

        Required keys in the returned dict: ``text``, ``provider``, ``model``.
        Must raise a :class:`~free_llm_api.errors.ProviderError` subclass on
        failure so the manager can fail over.
        """
        raise NotImplementedError

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return f"<{self.__class__.__name__} name={self.name} model={self.model}>"


class OpenAICompatibleProvider(BaseProvider):
    """Base for providers exposing an OpenAI ``/chat/completions`` endpoint."""

    #: e.g. "https://api.groq.com/openai/v1" — overridable via config ``extra.base_url``
    base_url: str = ""
    chat_path: str = "/chat/completions"

    def _endpoint(self) -> str:
        base = (self.extra.get("base_url") or self.base_url).rstrip("/")
        return base + self.chat_path

    def _headers(self) -> Dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.api_key:  # some free gateways (e.g. LLM7.io) need no key
            headers["Authorization"] = f"Bearer {self.api_key}"
        headers.update(self.extra.get("headers", {}))
        return headers

    def _build_payload(self, prompt: str, **kwargs: Any) -> Dict[str, Any]:
        messages = []
        system = kwargs.get("system")
        if system:
            messages.append({"role": "system", "content": system})
        if "messages" in kwargs:  # allow callers to pass full chat history
            messages.extend(kwargs["messages"])
        else:
            messages.append({"role": "user", "content": prompt})

        payload: Dict[str, Any] = {
            "model": kwargs.get("model", self.model),
            "messages": messages,
        }
        for key in ("max_tokens", "temperature", "top_p", "stop"):
            if key in kwargs:
                payload[key] = kwargs[key]
        payload.update(self.extra.get("params", {}))
        return payload

    def generate(self, prompt: str, **kwargs: Any) -> Dict[str, Any]:
        url = self._endpoint()
        payload = self._build_payload(prompt, **kwargs)
        try:
            resp = requests.post(
                url, headers=self._headers(), json=payload, timeout=self.timeout
            )
        except requests.Timeout as exc:
            raise ProviderTimeoutError(
                f"{self.name} timed out after {self.timeout}s", provider=self.name
            ) from exc
        except requests.RequestException as exc:
            raise ProviderError(
                f"{self.name} connection error: {exc}", provider=self.name
            ) from exc

        self._raise_for_status(resp)

        try:
            data = resp.json()
        except ValueError as exc:
            raise ProviderError(
                f"{self.name} returned non-JSON response", provider=self.name
            ) from exc
        return self._normalize(data)

    def stream_generate(
        self, prompt: str, **kwargs: Any
    ) -> Generator[Dict[str, Any], None, None]:
        url = self._endpoint()
        payload = self._build_payload(prompt, **kwargs)
        payload["stream"] = True
        try:
            resp = requests.post(
                url, headers=self._headers(), json=payload,
                timeout=self.timeout, stream=True,
            )
        except requests.Timeout as exc:
            raise ProviderTimeoutError(
                f"{self.name} timed out after {self.timeout}s", provider=self.name
            ) from exc
        except requests.RequestException as exc:
            raise ProviderError(
                f"{self.name} connection error: {exc}", provider=self.name
            ) from exc

        self._raise_for_status(resp)

        text_buffer = ""
        for line in resp.iter_lines():
            if not line:
                continue
            decoded = line.decode()
            if not decoded.startswith("data: "):
                continue
            data = decoded[6:]
            if data == "[DONE]":
                break
            try:
                chunk = json.loads(data)
            except json.JSONDecodeError:
                continue
            choices = chunk.get("choices", [])
            if not choices:
                continue
            delta = choices[0].get("delta", {})
            content = delta.get("content", "")
            finish_reason = choices[0].get("finish_reason")
            if content:
                text_buffer += content
            yield {
                "text": content,
                "provider": self.name,
                "model": chunk.get("model", self.model),
                "finish_reason": finish_reason,
                "raw": chunk,
            }

    def _raise_for_status(self, resp: requests.Response) -> None:
        if resp.status_code < 400:
            return
        body = resp.text[:500]
        if resp.status_code == 429:
            raise RateLimitError(
                f"{self.name} rate limited: {body}",
                provider=self.name,
                retry_after=_to_float(resp.headers.get("Retry-After")),
            )
        if resp.status_code in (401, 403):
            raise InvalidKeyError(
                f"{self.name} rejected credentials ({resp.status_code}): {body}",
                provider=self.name,
                status=resp.status_code,
            )
        # 5xx and everything else: retryable failure
        raise ProviderError(
            f"{self.name} HTTP {resp.status_code}: {body}",
            provider=self.name,
            status=resp.status_code,
        )

    def _normalize(self, data: Dict[str, Any]) -> Dict[str, Any]:
        # Some OpenAI-compatible gateways return errors with HTTP 200.
        if isinstance(data, dict) and data.get("error") and "choices" not in data:
            err = data["error"]
            msg = err.get("message") if isinstance(err, dict) else str(err)
            raise ProviderError(f"{self.name}: {msg}", provider=self.name)
        try:
            text = data["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise ProviderError(
                f"{self.name} unexpected response shape", provider=self.name
            ) from exc
        return {
            "text": text,
            "provider": self.name,
            "model": data.get("model", self.model),
            "usage": data.get("usage"),
            "raw": data,
        }


def _to_float(value: Any) -> Optional[float]:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
