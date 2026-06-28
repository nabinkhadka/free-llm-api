"""Scheduling strategies.

Default is *smooth* weighted round-robin (the nginx algorithm): it interleaves
providers proportionally to their weight instead of emitting bursts, and it
cycles correctly. A ``fastest`` strategy (bonus) picks the lowest-latency
provider dynamically.
"""
from __future__ import annotations

import threading
from typing import Callable, Dict, List, Optional, Sequence, Tuple

# A candidate is (name, weight).
Candidate = Tuple[str, int]


class WeightedRoundRobin:
    """Smooth weighted round-robin.

    State (``current weight`` per provider) is keyed by name, so it stays
    correct even as the set of *available* providers changes due to cooldowns.
    Over any whole number of cycles the distribution matches the weights
    exactly.
    """

    def __init__(self) -> None:
        self._current: Dict[str, float] = {}
        self._lock = threading.Lock()

    def select(self, candidates: Sequence[Candidate]) -> Optional[str]:
        usable: List[Candidate] = [(n, int(w)) for n, w in candidates if w > 0]
        if not usable:
            return None
        total = sum(w for _, w in usable)
        with self._lock:
            best_name: Optional[str] = None
            best_weight: Optional[float] = None
            for name, weight in usable:
                cw = self._current.get(name, 0.0) + weight
                self._current[name] = cw
                if best_weight is None or cw > best_weight:
                    best_weight = cw
                    best_name = name
            assert best_name is not None
            self._current[best_name] -= total
            return best_name


class FastestFirst:
    """Pick the available provider with the lowest mean latency (bonus).

    Falls back to the first candidate while latencies are unknown.
    """

    def __init__(self, latency_fn: Callable[[str], float]) -> None:
        self._latency_fn = latency_fn

    def select(self, candidates: Sequence[Candidate]) -> Optional[str]:
        names = [n for n, w in candidates if w > 0]
        if not names:
            return None
        return min(names, key=self._latency_fn)


def build_scheduler(strategy: str, latency_fn: Callable[[str], float]):
    strategy = (strategy or "weighted_round_robin").lower()
    if strategy in ("weighted_round_robin", "wrr"):
        return WeightedRoundRobin()
    if strategy in ("fastest", "fastest_first", "latency"):
        return FastestFirst(latency_fn)
    raise ValueError(f"Unknown scheduling strategy: {strategy!r}")
