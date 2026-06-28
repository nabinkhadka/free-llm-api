"""Plugin registry: providers self-register with ``@register("name")``.

The core (manager/scheduler) never imports concrete providers; it only looks
them up here by type. Adding a provider = drop a new ``*_provider.py`` file
that calls ``@register`` — no core changes required.
"""
from __future__ import annotations

from typing import Dict, List, Type

from .base import BaseProvider

_REGISTRY: Dict[str, Type[BaseProvider]] = {}


def register(name: str):
    """Class decorator registering a provider implementation under ``name``."""

    def decorator(cls: Type[BaseProvider]) -> Type[BaseProvider]:
        key = name.lower()
        if key in _REGISTRY and _REGISTRY[key] is not cls:
            raise ValueError(f"Provider type '{name}' is already registered")
        cls.provider_type = key
        _REGISTRY[key] = cls
        return cls

    return decorator


def get_provider_class(name: str) -> "Type[BaseProvider] | None":
    return _REGISTRY.get(name.lower())


def available_providers() -> List[str]:
    return sorted(_REGISTRY)
