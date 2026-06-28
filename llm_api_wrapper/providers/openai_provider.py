"""Generic OpenAI-compatible provider — the one class behind most providers.

Groq, Together, OpenRouter, NVIDIA, Mistral, Cerebras, Z.AI, LLM7.io, Google
Gemini and friends all speak the OpenAI ``/chat/completions`` dialect. Rather
than a file per provider, register ONE reusable type and point it at any base
URL (and optional extra headers / params) straight from config::

    - name: mistral
      type: openai
      api_key: "${MISTRAL_API_KEY}"
      model: "mistral-small-latest"
      extra:
        base_url: "https://api.mistral.ai/v1"

Adding another OpenAI-compatible provider then needs no code at all — just a
config entry. Use ``requires_key: false`` for key-less gateways (e.g. LLM7.io),
and ``extra.headers`` for providers that want extra headers (e.g. OpenRouter).
"""
from __future__ import annotations

from .base import OpenAICompatibleProvider
from .registry import register


@register("openai")
class OpenAIProvider(OpenAICompatibleProvider):
    """OpenAI-compatible provider configured entirely via ``extra.base_url``."""
