"""Probe which models actually WORK (and are free) for a provider, using your keys.

Run in the shell where your API keys are exported:

    python3 tests/try_models.py llm7                       # auto-tries LLM7's free-tier models
    python3 tests/try_models.py zai                         # tries built-in free-first candidates
    python3 tests/try_models.py zai glm-4.5-flash glm-4.6   # or pass your own list
    python3 tests/try_models.py cerebras
    python3 tests/try_models.py gemini

For each candidate it sends a tiny prompt and prints OK (+ snippet) or the exact
error — so you can tell free vs paid/locked vs unknown-model. Put a working one
in config.yaml (it hot-reloads).
"""
import sys

import requests

from free_llm_api.utils.config_loader import load_config

# Free-first candidates per provider (override by passing models on the CLI).
CANDIDATES = {
    "zai": ["glm-4.5-flash", "glm-4.6-flash", "glm-4.5-air", "glm-4-flash", "glm-4.6"],
    "cerebras": ["gpt-oss-120b", "llama-3.3-70b", "qwen-3-32b", "llama3.1-8b"],
    "gemini": ["gemini-2.5-flash", "gemini-2.5-flash-lite",
               "gemini-3.1-flash-lite", "gemini-3.5-flash"],
}


def free_llm7_models(base: str) -> list:
    """LLM7 marks pay-per-token models with usage_based_only=true; keep the rest."""
    data = requests.get(base + "/models", timeout=15).json()
    rows = data.get("data", data)
    return [m["id"] for m in rows if not m.get("usage_based_only", True)]


def candidates_for(name: str, entry: dict, base: str, cli_models: list) -> list:
    if cli_models:
        return cli_models
    if name == "llm7":
        try:
            return free_llm7_models(base)
        except Exception:
            pass
    return CANDIDATES.get(name) or CANDIDATES.get(entry.get("type", "")) or [entry.get("model")]


def main() -> None:
    if len(sys.argv) < 2:
        print("usage: python3 tests/try_models.py <provider-name> [model ...]")
        return
    target, cli_models = sys.argv[1], sys.argv[2:]

    entry = next((p for p in load_config()["providers"] if p.get("name") == target), None)
    if entry is None:
        print(f"No provider named '{target}' in config.yaml")
        return
    if entry.get("type", target) != "openai":
        print(f"'{target}' is type={entry.get('type')} — this tester only handles OpenAI-compatible providers")
        return

    extra = entry.get("extra") or {}
    base = extra.get("base_url", "").rstrip("/")
    key = entry.get("api_key") or None
    url = base + "/chat/completions"
    headers = {"Content-Type": "application/json", **extra.get("headers", {})}
    if key:
        headers["Authorization"] = f"Bearer {key}"

    models = candidates_for(target, entry, base, cli_models)
    print(f"\nTesting {len(models)} model(s) on '{target}'  ({base})\n")
    for model in models:
        payload = {"model": model,
                   "messages": [{"role": "user", "content": "Say OK"}],
                   "max_tokens": 50}
        payload.update(extra.get("params", {}))  # e.g. zai's thinking:disabled
        try:
            r = requests.post(url, headers=headers, json=payload, timeout=30)
            if r.status_code < 400:
                txt = r.json()["choices"][0]["message"]["content"].strip().replace("\n", " ")
                print(f"  OK    {model:24} -> {txt[:40]!r}")
            else:
                print(f"  FAIL  {model:24} [{r.status_code}] {r.text[:110]}")
        except Exception as exc:
            print(f"  ERR   {model:24} {exc}")


if __name__ == "__main__":
    main()
