"""Provider package.

Auto-discovers every ``*_provider.py`` module so their ``@register`` calls run
at import time. Dropping a new file into this directory is enough to make a new
provider available — nothing else needs editing.
"""
from __future__ import annotations

import importlib
import logging
import pkgutil

from .base import BaseProvider, OpenAICompatibleProvider
from .registry import available_providers, get_provider_class, register

logger = logging.getLogger(__name__)


def _autodiscover() -> None:
    for module in pkgutil.iter_modules(__path__):
        if module.name.endswith("_provider"):
            try:
                importlib.import_module(f"{__name__}.{module.name}")
            except Exception as exc:  # pragma: no cover - defensive
                logger.error("Failed to import provider module %s: %s", module.name, exc)


_autodiscover()

__all__ = [
    "BaseProvider",
    "OpenAICompatibleProvider",
    "register",
    "get_provider_class",
    "available_providers",
]
