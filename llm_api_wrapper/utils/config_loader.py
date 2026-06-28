"""Load and lightly validate ``config.yaml``.

Supports ``${ENV_VAR}`` (and ``${ENV_VAR:-default}``) expansion so secrets
live in the environment, never in the file. Also exposes the file's mtime so
the manager can hot-reload when it changes.
"""
from __future__ import annotations

import os
import re
from typing import Any, Dict, Optional

import yaml

from ..errors import ConfigError

_ENV_PATTERN = re.compile(r"\$\{([^}]+)\}")


def _expand(value: Any) -> Any:
    """Recursively expand ``${VAR}`` / ``${VAR:-default}`` in strings."""
    if isinstance(value, str):

        def repl(match: "re.Match[str]") -> str:
            expr = match.group(1)
            default = ""
            if ":-" in expr:
                expr, default = expr.split(":-", 1)
            return os.environ.get(expr.strip(), default)

        return _ENV_PATTERN.sub(repl, value)
    if isinstance(value, list):
        return [_expand(v) for v in value]
    if isinstance(value, dict):
        return {k: _expand(v) for k, v in value.items()}
    return value


def find_config(path: Optional[str] = None) -> str:
    """Resolve a config path: explicit arg > $LLM_WRAPPER_CONFIG > CWD > packaged."""
    candidates = []
    if path:
        candidates.append(path)
    env_path = os.environ.get("LLM_WRAPPER_CONFIG")
    if env_path:
        candidates.append(env_path)
    candidates.append(os.path.join(os.getcwd(), "config.yaml"))
    candidates.append(
        os.path.join(os.path.dirname(os.path.dirname(__file__)), "config.yaml")
    )
    for candidate in candidates:
        if candidate and os.path.isfile(candidate):
            return candidate
    raise ConfigError(
        "config.yaml not found. Set $LLM_WRAPPER_CONFIG or place config.yaml "
        "in the working directory."
    )


def load_config(path: Optional[str] = None) -> Dict[str, Any]:
    resolved = find_config(path)
    try:
        with open(resolved, "r", encoding="utf-8") as fh:
            raw = yaml.safe_load(fh) or {}
    except (OSError, yaml.YAMLError) as exc:
        raise ConfigError(f"Failed to read config '{resolved}': {exc}") from exc

    if not isinstance(raw, dict) or not isinstance(raw.get("providers"), list):
        raise ConfigError("config must contain a 'providers' list")

    config = _expand(raw)
    config["_path"] = resolved
    config["_mtime"] = os.path.getmtime(resolved)
    config.setdefault("settings", {})
    return config


def config_mtime(path: str) -> float:
    try:
        return os.path.getmtime(path)
    except OSError:
        return 0.0
