"""Exception hierarchy for free_llm_api.

``ProviderError`` and its subclasses are *recoverable*: the manager catches
them and fails over to the next provider. Each carries failure-handling hints
(``cooldown`` seconds, whether the provider should be ``disables``-d) so the
manager can react without knowing provider internals.
"""
from __future__ import annotations

from typing import Dict, Optional


class LLMWrapperError(Exception):
    """Base class for every error raised by this package."""


class ConfigError(LLMWrapperError):
    """Configuration is missing or invalid."""


class ProviderError(LLMWrapperError):
    """A single provider failed a request (recoverable via failover)."""

    #: seconds to skip this provider after the failure
    cooldown: float = 10.0
    #: if True the provider is disabled until the next config reload
    disables: bool = False

    def __init__(
        self,
        message: str,
        *,
        provider: Optional[str] = None,
        status: Optional[int] = None,
    ) -> None:
        super().__init__(message)
        self.provider = provider
        self.status = status


class RateLimitError(ProviderError):
    """HTTP 429 / provider signalled rate limiting — back off for a while."""

    cooldown = 60.0

    def __init__(
        self,
        message: str,
        *,
        provider: Optional[str] = None,
        status: Optional[int] = 429,
        retry_after: Optional[float] = None,
    ) -> None:
        super().__init__(message, provider=provider, status=status)
        self.retry_after = retry_after


class InvalidKeyError(ProviderError):
    """HTTP 401/403 — the API key is bad; disable the provider."""

    cooldown = 0.0
    disables = True


class ProviderTimeoutError(ProviderError):
    """Timed out / temporarily unreachable / model still loading."""

    cooldown = 15.0


class AllProvidersFailedError(LLMWrapperError):
    """Every candidate provider failed for a single request."""

    def __init__(self, message: str, *, errors: Optional[Dict[str, str]] = None) -> None:
        super().__init__(message)
        self.errors = errors or {}
