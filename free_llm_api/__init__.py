"""free_llm_api — resilient routing across FREE LLM API providers.

Quick start::

    from free_llm_api import endpoints
    response = endpoints.generate("Explain quantum computing")
    print(response["text"])
"""
from __future__ import annotations

import logging

from . import endpoints
from .errors import (
    AllProvidersFailedError,
    ConfigError,
    InvalidKeyError,
    LLMWrapperError,
    ProviderError,
    ProviderTimeoutError,
    RateLimitError,
)
from .providers.base import BaseProvider, OpenAICompatibleProvider
from .providers.registry import register

# Library best-practice: a NullHandler keeps importing the package quiet;
# the host application decides how logs are emitted.
logging.getLogger(__name__).addHandler(logging.NullHandler())

#: Convenience aliases.
generate = endpoints.generate
stream_generate = endpoints.stream_generate

__version__ = "0.2.0"

__all__ = [
    "endpoints",
    "generate",
    "stream_generate",
    "register",
    "BaseProvider",
    "OpenAICompatibleProvider",
    "LLMWrapperError",
    "ConfigError",
    "ProviderError",
    "RateLimitError",
    "InvalidKeyError",
    "ProviderTimeoutError",
    "AllProvidersFailedError",
]
