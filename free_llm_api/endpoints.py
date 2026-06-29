"""Public API.

Usage::

    from free_llm_api import endpoints
    response = endpoints.generate("Explain quantum computing")
    print(response["text"])

A single process-wide :class:`~free_llm_api.manager.Manager` is created
lazily on first use.
"""
from __future__ import annotations

import threading
from typing import Any, Dict, Generator, Optional

from .manager import Manager

_manager: Optional[Manager] = None
_lock = threading.Lock()


def _get_manager() -> Manager:
    global _manager
    if _manager is None:
        with _lock:
            if _manager is None:
                _manager = Manager()
    return _manager


def generate(prompt: str, **kwargs: Any) -> Dict[str, Any]:
    """Route ``prompt`` to a free provider with automatic failover.

    Returns a normalized dict with at least ``text``, ``provider`` and
    ``model`` (plus ``latency``, ``usage`` and ``raw``).
    Raises :class:`~free_llm_api.errors.AllProvidersFailedError` if every
    provider fails.

    Common kwargs: ``model``, ``system``, ``max_tokens``, ``temperature``,
    ``top_p``, ``stop`` (passed through to the chosen provider).
    """
    return _get_manager().generate(prompt, **kwargs)


def reload() -> None:
    """Force an immediate config + provider reload."""
    _get_manager().reload()


def stats() -> Dict[str, Any]:
    """Return per-provider runtime health and latency statistics."""
    return _get_manager().stats()


def status() -> Dict[str, Any]:
    """Return a comprehensive config + runtime snapshot.

    Includes settings, config path, and per-provider details (model,
    weight, masked API key, type, availability, health).
    """
    return _get_manager().status()


def stream_generate(prompt: str, **kwargs: Any) -> Generator[Dict[str, Any], None, None]:
    """Stream a response from a free provider with automatic failover.

    Yields normalized chunk dicts (``text``, ``provider``, ``model``,
    ``finish_reason``).  The final chunk will have ``finish_reason="stop"``.
    Raises :class:`~free_llm_api.errors.AllProvidersFailedError` if every
    provider fails.
    """
    yield from _get_manager().stream_generate(prompt, **kwargs)


def configure(config_path: str) -> None:
    """(Re)initialise the global manager from a specific config file."""
    global _manager
    with _lock:
        _manager = Manager(config_path=config_path)
