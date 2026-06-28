"""HuggingFace serverless Inference API (free tier).

Unlike the others this is NOT OpenAI-shaped: it posts ``{"inputs": ...}`` and
returns ``[{"generated_text": ...}]``. Implementing it against the plain
``BaseProvider`` interface demonstrates that the plugin system handles
heterogeneous APIs without any core changes.

https://huggingface.co/docs/api-inference
"""
from __future__ import annotations

from typing import Any, Dict

import requests

from ..errors import (
    InvalidKeyError,
    ProviderError,
    ProviderTimeoutError,
    RateLimitError,
)
from .base import BaseProvider, _to_float
from .registry import register


@register("huggingface")
class HuggingFaceProvider(BaseProvider):
    base_url = "https://api-inference.huggingface.co/models"

    def _endpoint(self) -> str:
        base = (self.extra.get("base_url") or self.base_url).rstrip("/")
        return f"{base}/{self.model}"

    def generate(self, prompt: str, **kwargs: Any) -> Dict[str, Any]:
        parameters: Dict[str, Any] = {"return_full_text": False}
        if "max_tokens" in kwargs:
            parameters["max_new_tokens"] = kwargs["max_tokens"]
        if "temperature" in kwargs:
            parameters["temperature"] = kwargs["temperature"]
        parameters.update(self.extra.get("parameters", {}))

        payload = {
            "inputs": prompt,
            "parameters": parameters,
            # ask HF to block until a cold model is ready instead of 503-ing
            "options": {"wait_for_model": True},
        }
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        try:
            resp = requests.post(
                self._endpoint(), headers=headers, json=payload, timeout=self.timeout
            )
        except requests.Timeout as exc:
            raise ProviderTimeoutError(
                f"{self.name} timed out after {self.timeout}s", provider=self.name
            ) from exc
        except requests.RequestException as exc:
            raise ProviderError(
                f"{self.name} connection error: {exc}", provider=self.name
            ) from exc

        if resp.status_code == 429:
            raise RateLimitError(
                f"{self.name} rate limited",
                provider=self.name,
                retry_after=_to_float(resp.headers.get("Retry-After")),
            )
        if resp.status_code in (401, 403):
            raise InvalidKeyError(
                f"{self.name} rejected credentials ({resp.status_code})",
                provider=self.name,
                status=resp.status_code,
            )
        if resp.status_code == 503:
            # model is loading — retry shortly
            raise ProviderTimeoutError(
                f"{self.name} model loading (503)", provider=self.name
            )
        if resp.status_code >= 400:
            raise ProviderError(
                f"{self.name} HTTP {resp.status_code}: {resp.text[:500]}",
                provider=self.name,
                status=resp.status_code,
            )

        try:
            data = resp.json()
        except ValueError as exc:
            raise ProviderError(
                f"{self.name} returned non-JSON response", provider=self.name
            ) from exc
        return self._normalize(data)

    def _normalize(self, data: Any) -> Dict[str, Any]:
        text = None
        if isinstance(data, list) and data:
            first = data[0]
            if isinstance(first, dict):
                text = first.get("generated_text")
        elif isinstance(data, dict):
            if data.get("error"):
                raise ProviderError(f"{self.name}: {data['error']}", provider=self.name)
            text = data.get("generated_text")
        if text is None:
            raise ProviderError(
                f"{self.name} unexpected response shape", provider=self.name
            )
        return {
            "text": text,
            "provider": self.name,
            "model": self.model,
            "usage": None,
            "raw": data,
        }
