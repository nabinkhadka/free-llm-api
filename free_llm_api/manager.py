"""Manager: load config, build providers, schedule, fail over, track health.

Responsibilities:
* load + hot-reload ``config.yaml``
* instantiate providers from the plugin registry
* schedule via the configured strategy (weighted round-robin by default)
* fail over on ANY provider error, applying smart cooldowns
* track per-provider health and latency
"""
from __future__ import annotations

import logging
import threading
import time
from collections import deque
from typing import Any, Deque, Dict, Generator, List, Optional, Set

from .errors import AllProvidersFailedError, ConfigError, ProviderError
from .providers import get_provider_class
from .scheduler import build_scheduler
from .utils.config_loader import config_mtime, load_config

logger = logging.getLogger(__name__)


class _ProviderState:
    """Runtime health for one provider instance."""

    def __init__(self, instance: Any) -> None:
        self.instance = instance
        self.disabled = False          # invalid key -> until next reload
        self.cooldown_until = 0.0      # epoch seconds; skip provider until then
        self.failures = 0              # consecutive failures (drives backoff)
        self.successes = 0
        self.last_error: Optional[str] = None
        self.latencies: Deque[float] = deque(maxlen=20)

    @property
    def name(self) -> str:
        return self.instance.name

    @property
    def weight(self) -> int:
        return max(1, int(self.instance.weight))

    def available(self, now: float) -> bool:
        return not self.disabled and now >= self.cooldown_until

    def mean_latency(self) -> float:
        return sum(self.latencies) / len(self.latencies) if self.latencies else float("inf")

    def record_success(self, latency: float) -> None:
        self.successes += 1
        self.failures = 0
        self.last_error = None
        self.cooldown_until = 0.0
        self.latencies.append(latency)


class Manager:
    """Loads providers and routes a prompt to the first one that succeeds."""

    def __init__(self, config_path: Optional[str] = None, auto_reload: bool = True) -> None:
        self._config_path_arg = config_path
        self._auto_reload = auto_reload
        self._lock = threading.RLock()
        self._states: Dict[str, _ProviderState] = {}
        self._order: List[str] = []
        self._settings: Dict[str, Any] = {}
        self._config_file: Optional[str] = None
        self._config_mtime: float = 0.0
        self._scheduler = None
        self._load()

    # ---- config / provider construction -------------------------------

    def _load(self) -> None:
        config = load_config(self._config_path_arg)
        self._config_file = config["_path"]
        self._config_mtime = config["_mtime"]
        self._settings = config.get("settings", {}) or {}

        new_states: Dict[str, _ProviderState] = {}
        order: List[str] = []
        for entry in config["providers"]:
            name = entry.get("name")
            if not name:
                logger.warning("Skipping provider with no name: %r", entry)
                continue
            if not entry.get("enabled", True):
                logger.info("Provider '%s' disabled in config", name)
                continue
            ptype = entry.get("type", name)
            cls = get_provider_class(ptype)
            if cls is None:
                logger.warning("No implementation registered for provider type '%s'", ptype)
                continue
            api_key = entry.get("api_key") or None
            requires_key = entry.get("requires_key", cls.requires_key)
            if requires_key and not api_key:
                logger.warning("Provider '%s' has no api_key — skipping", name)
                continue
            try:
                instance = cls(
                    name=name,
                    api_key=api_key,
                    model=entry.get("model"),
                    timeout=float(entry.get("timeout", self._settings.get("timeout", 20))),
                    weight=int(entry.get("weight", 1)),
                    extra=entry.get("extra") or {},
                )
            except Exception as exc:  # pragma: no cover - defensive
                logger.error("Failed to init provider '%s': %s", name, exc)
                continue

            state = _ProviderState(instance)
            # Carry over latency/health across reloads, but NOT `disabled`:
            # a config edit may have fixed a bad key, so give it a fresh chance.
            prev = self._states.get(name)
            if prev is not None:
                state.cooldown_until = prev.cooldown_until
                state.failures = prev.failures
                state.successes = prev.successes
                state.latencies = prev.latencies
            new_states[name] = state
            order.append(name)

        if not new_states:
            raise ConfigError("No usable providers configured (check keys / enabled flags)")

        self._states = new_states
        self._order = order
        self._scheduler = build_scheduler(
            self._settings.get("strategy", "weighted_round_robin"),
            self._latency_of,
        )
        logger.info("Loaded %d provider(s): %s", len(order), ", ".join(order))

    def _latency_of(self, name: str) -> float:
        state = self._states.get(name)
        return state.mean_latency() if state else float("inf")

    def _maybe_reload(self) -> None:
        if not self._auto_reload or not self._config_file:
            return
        mtime = config_mtime(self._config_file)
        if mtime and mtime != self._config_mtime:
            logger.info("config.yaml changed — reloading providers")
            try:
                self._load()
            except ConfigError as exc:
                logger.error("Reload failed, keeping previous config: %s", exc)

    def reload(self) -> None:
        """Force a config + provider reload."""
        with self._lock:
            self._load()

    # ---- main entrypoint ---------------------------------------------

    def generate(self, prompt: str, **kwargs: Any) -> Dict[str, Any]:
        """Route ``prompt`` to a provider, failing over until one succeeds."""
        with self._lock:
            self._maybe_reload()
            retry = bool(self._settings.get("retry_on_failure", True))

        # Initial pick uses the scheduler (respects weights).
        with self._lock:
            name = self._pick(exclude=set())

        tried: Dict[str, str] = {}
        while name is not None:
            with self._lock:
                state = self._states.get(name)
            if state is None:
                break

            logger.info("-> trying '%s' (%s)", state.name, state.instance.model)
            start = time.monotonic()
            try:
                result = state.instance.generate(prompt, **kwargs)
            except ProviderError as exc:
                self._on_failure(state, exc)
                tried[state.name] = str(exc)
                logger.warning("x '%s' failed (%.2fs): %s", state.name,
                               time.monotonic() - start, exc)
                if not retry:
                    break
                # Don't call scheduler on retry — walk remaining providers in order
                # so failures don't corrupt the scheduler's weight distribution.
                with self._lock:
                    name = self._next_available(exclude=set(tried))
                continue
            except Exception as exc:  # unexpected: treat as a normal failure
                self._on_failure(state, ProviderError(str(exc), provider=state.name))
                tried[state.name] = str(exc)
                logger.exception("x '%s' raised unexpectedly", state.name)
                if not retry:
                    break
                with self._lock:
                    name = self._next_available(exclude=set(tried))
                continue

            latency = time.monotonic() - start
            with self._lock:
                state.record_success(latency)
                result.setdefault("latency", round(latency, 3))
            logger.info("v '%s' succeeded in %.2fs", state.name, latency)
            return result

        raise AllProvidersFailedError(
            "All providers failed (tried %d): %s"
            % (len(tried), "; ".join(f"{n}: {e}" for n, e in tried.items())),
            errors=tried,
        )

    def stream_generate(
        self, prompt: str, **kwargs: Any
    ) -> Generator[Dict[str, Any], None, None]:
        with self._lock:
            self._maybe_reload()
            retry = bool(self._settings.get("retry_on_failure", True))

        with self._lock:
            name = self._pick(exclude=set())

        tried: Dict[str, str] = {}
        while name is not None:
            with self._lock:
                state = self._states.get(name)
            if state is None:
                break

            logger.info("-> trying '%s' (%s)", state.name, state.instance.model)
            start = time.monotonic()
            try:
                yielded = False
                for chunk in state.instance.stream_generate(prompt, **kwargs):
                    yielded = True
                    yield chunk

                if not yielded:
                    raise ProviderError(
                        f"{state.name} returned empty stream",
                        provider=state.name,
                    )
            except ProviderError as exc:
                self._on_failure(state, exc)
                tried[state.name] = str(exc)
                logger.warning("x '%s' failed (%.2fs): %s", state.name,
                               time.monotonic() - start, exc)
                if not retry:
                    break
                with self._lock:
                    name = self._next_available(exclude=set(tried))
                continue
            except Exception as exc:
                self._on_failure(state, ProviderError(str(exc), provider=state.name))
                tried[state.name] = str(exc)
                logger.exception("x '%s' raised unexpectedly", state.name)
                if not retry:
                    break
                with self._lock:
                    name = self._next_available(exclude=set(tried))
                continue

            latency = time.monotonic() - start
            with self._lock:
                state.record_success(latency)
            logger.info("v '%s' succeeded in %.2fs", state.name, latency)
            return

        raise AllProvidersFailedError(
            "All providers failed (tried %d): %s"
            % (len(tried), "; ".join(f"{n}: {e}" for n, e in tried.items())),
            errors=tried,
        )

    def _pick(self, exclude: Set[str]) -> Optional[str]:
        now = time.time()
        candidates = [
            (s.name, s.weight)
            for s in self._states.values()
            if s.name not in exclude and s.available(now)
        ]
        if not candidates:
            return None
        return self._scheduler.select(candidates)

    def _next_available(self, exclude: Set[str]) -> Optional[str]:
        """Return the first available provider *not* in ``exclude``, in config
        order.  Used for retries after a failure so the scheduler's internal
        weight state is never corrupted by failed attempts."""
        now = time.time()
        for name in self._order:
            state = self._states.get(name)
            if state and name not in exclude and state.available(now):
                return name
        return None

    def _on_failure(self, state: _ProviderState, exc: ProviderError) -> None:
        with self._lock:
            state.failures += 1
            state.last_error = str(exc)
            if getattr(exc, "disables", False):
                state.disabled = True
                logger.error("Disabling provider '%s': %s", state.name, exc)
                return
            cooldown = float(getattr(exc, "cooldown", 10.0))
            retry_after = getattr(exc, "retry_after", None)
            if retry_after:
                cooldown = max(cooldown, float(retry_after))
            # gentle linear backoff for repeated failures, capped at 5x
            cooldown *= min(state.failures, 5)
            state.cooldown_until = time.time() + cooldown
            logger.info("Cooling down '%s' for %.0fs", state.name, cooldown)

    # ---- introspection ------------------------------------------------

    def stats(self) -> Dict[str, Any]:
        now = time.time()
        with self._lock:
            return {
                name: {
                    "model": s.instance.model,
                    "weight": s.weight,
                    "available": s.available(now),
                    "disabled": s.disabled,
                    "cooldown_remaining": round(max(0.0, s.cooldown_until - now), 1),
                    "successes": s.successes,
                    "failures": s.failures,
                    "mean_latency": round(s.mean_latency(), 3) if s.latencies else None,
                    "last_error": s.last_error,
                }
                for name, s in self._states.items()
            }

    def status(self) -> Dict[str, Any]:
        """Return a comprehensive snapshot: config settings + per-provider
        config and runtime state."""
        now = time.time()
        with self._lock:
            strategy = self._settings.get("strategy", "weighted_round_robin")
            providers = {}
            for name, s in self._states.items():
                key = s.instance.api_key or ""
                masked = (key[:4] + "****") if len(key) > 8 else bool(key)
                providers[name] = {
                    "type": getattr(s.instance, "provider_type", "unknown"),
                    "model": s.instance.model,
                    "weight": s.weight,
                    "api_key": masked,
                    "available": s.available(now),
                    "disabled": s.disabled,
                    "cooldown_remaining": round(max(0.0, s.cooldown_until - now), 1),
                    "successes": s.successes,
                    "failures": s.failures,
                    "mean_latency": round(s.mean_latency(), 3) if s.latencies else None,
                    "last_error": s.last_error,
                }
            return {
                "config_file": self._config_file,
                "settings": {
                    "strategy": strategy,
                    "timeout": self._settings.get("timeout", 20),
                    "retry_on_failure": self._settings.get("retry_on_failure", True),
                },
                "providers": providers,
            }
