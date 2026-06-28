"""Network-free tests for scheduling, failover, cooldown and hot-reload.

Runnable either with pytest (`pytest -q`) or directly (`python tests/test_wrapper.py`).
Fake providers are registered so nothing hits the network.
"""
from __future__ import annotations

import os
import tempfile
import time
from collections import Counter

from llm_api_wrapper.errors import AllProvidersFailedError
from llm_api_wrapper.manager import Manager
from llm_api_wrapper.providers.base import BaseProvider
from llm_api_wrapper.providers.registry import register
from llm_api_wrapper.scheduler import WeightedRoundRobin
from llm_api_wrapper import errors


# --- fake providers -------------------------------------------------------

@register("ok")
class OkProvider(BaseProvider):
    def generate(self, prompt, **kwargs):
        return {"text": f"{self.name}:{prompt}", "provider": self.name, "model": self.model}


@register("boom")
class BoomProvider(BaseProvider):
    def generate(self, prompt, **kwargs):
        raise errors.ProviderError("always fails", provider=self.name)


@register("badkey")
class BadKeyProvider(BaseProvider):
    def generate(self, prompt, **kwargs):
        raise errors.InvalidKeyError("nope", provider=self.name)


@register("limited")
class RateLimitedProvider(BaseProvider):
    def generate(self, prompt, **kwargs):
        raise errors.RateLimitError("slow down", provider=self.name, retry_after=30)


def _write_config(body: str) -> str:
    fh = tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False)
    fh.write(body)
    fh.close()
    return fh.name


# --- scheduler ------------------------------------------------------------

def test_weighted_round_robin_distribution():
    wrr = WeightedRoundRobin()
    counts = Counter(wrr.select([("a", 3), ("b", 1)]) for _ in range(100))
    assert counts["a"] == 75 and counts["b"] == 25, counts


def test_weighted_round_robin_smoothness():
    wrr = WeightedRoundRobin()
    seq = [wrr.select([("a", 3), ("b", 1)]) for _ in range(4)]
    # smooth (not bursty): b should not be last/clumped — exactly one b per cycle
    assert seq.count("a") == 3 and seq.count("b") == 1, seq


# --- manager behaviours ---------------------------------------------------

def test_failover_picks_a_working_provider():
    cfg = _write_config(
        """
providers:
  - {name: bad1, type: boom, api_key: x, model: m, weight: 5}
  - {name: good, type: ok,   api_key: x, model: m, weight: 1}
settings: {strategy: weighted_round_robin, retry_on_failure: true}
"""
    )
    m = Manager(config_path=cfg, auto_reload=False)
    res = m.generate("hi")
    assert res["provider"] == "good"
    assert res["text"] == "good:hi"
    assert res["latency"] >= 0
    os.unlink(cfg)


def test_all_fail_raises():
    cfg = _write_config(
        """
providers:
  - {name: b1, type: boom, api_key: x, model: m}
  - {name: b2, type: boom, api_key: x, model: m}
settings: {retry_on_failure: true}
"""
    )
    m = Manager(config_path=cfg, auto_reload=False)
    try:
        m.generate("hi")
        assert False, "expected AllProvidersFailedError"
    except AllProvidersFailedError as exc:
        assert set(exc.errors) == {"b1", "b2"}
    os.unlink(cfg)


def test_invalid_key_disables_provider():
    cfg = _write_config(
        """
providers:
  - {name: bk, type: badkey, api_key: x, model: m}
  - {name: good, type: ok, api_key: x, model: m}
settings: {retry_on_failure: true}
"""
    )
    m = Manager(config_path=cfg, auto_reload=False)
    m.generate("hi")
    assert m.stats()["bk"]["disabled"] is True
    # next call must not even consider the disabled provider
    res = m.generate("again")
    assert res["provider"] == "good"
    os.unlink(cfg)


def test_rate_limit_sets_cooldown():
    cfg = _write_config(
        """
providers:
  - {name: rl, type: limited, api_key: x, model: m}
  - {name: good, type: ok, api_key: x, model: m}
settings: {retry_on_failure: true}
"""
    )
    m = Manager(config_path=cfg, auto_reload=False)
    m.generate("hi")
    cd = m.stats()["rl"]["cooldown_remaining"]
    assert cd >= 30, cd  # honoured Retry-After
    os.unlink(cfg)


def test_hot_reload_adds_provider():
    cfg = _write_config(
        """
providers:
  - {name: good, type: ok, api_key: x, model: m1}
settings: {}
"""
    )
    m = Manager(config_path=cfg, auto_reload=True)
    assert set(m.stats()) == {"good"}
    time.sleep(0.01)
    with open(cfg, "w") as fh:
        fh.write(
            """
providers:
  - {name: good, type: ok, api_key: x, model: m1}
  - {name: extra, type: ok, api_key: x, model: m2}
settings: {}
"""
        )
    m.generate("trigger reload")  # generate() checks mtime and reloads
    assert set(m.stats()) == {"good", "extra"}
    os.unlink(cfg)


def _run_all():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"PASS {t.__name__}")
    print(f"\n{len(tests)} tests passed")


if __name__ == "__main__":
    _run_all()
