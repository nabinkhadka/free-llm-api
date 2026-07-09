# Advanced

## Scheduling

Providers are picked via smooth weighted round-robin (the algorithm nginx uses), or set  strategy to `fastest` to always route to whichever provider currently has the lowest tracked latency. Per-provider latency is tracked automatically and exposed via `endpoints.stats()`.

## Override model & weight per provider

Every provider's `model` and `weight` can be overridden at runtime via environment variables — no config file edits needed:

```bash
# Override model only
GROQ_MODEL=llama-4 python3 tests/example.py

# Override model + weight
GROQ_MODEL=llama-4 GROQ_WEIGHT=20 python3 tests/example.py

# Override multiple providers
GROQ_WEIGHT=30 GEMINI_WEIGHT=1 python3 tests/example.py
```

The env var name follows the pattern `{NAME}_MODEL` / `{NAME}_WEIGHT`, where `NAME` is the uppercase provider name from `config.yaml`. If unset, the built-in default from `config.yaml` is used.

Config resolution order: explicit path → `$FREE_LLM_API_CONFIG` → `./config.yaml` → the packaged default.

## Add a new provider

All included providers are OpenAI-compatible, so they share one generic `type: openai` implementation and differ only by `base_url` in config.

**If it's OpenAI-compatible (most are): no code at all.** Just add a config block pointing the generic `openai` type at its base URL:

```yaml
  - name: deepinfra
    type: openai
    enabled: true
    api_key: "${DEEPINFRA_API_KEY}"
    model: "meta-llama/Meta-Llama-3.1-8B-Instruct"
    weight: 5
    extra:
      base_url: "https://api.deepinfra.com/v1/openai"
      # headers: { ... }     # optional extra headers
      # params:  { ... }     # optional extra body params
  # requires_key: false      # for key-less gateways
```

**If its API isn't OpenAI-shaped:** drop a `*_provider.py` file that subclasses `BaseProvider`, implements `generate()` to return `{"text", "provider", "model"}`, and raises a `ProviderError` subclass on failure. Register it with `@register("yourname")` and  reference it via `type: yourname`. It's auto-discovered on import — the manager and scheduler never need to know it exists.

## How it behaves on failure

| Situation | Reaction |
|---|---|
| Timeout / connection error | `ProviderTimeoutError` → cool down ~15s, try next |
| HTTP 429 | `RateLimitError` → cool down ≥60s (or `Retry-After`), try next |
| HTTP 401 / 403 | `InvalidKeyError` → disable provider until next reload |
| HTTP 5xx / bad body | `ProviderError` → cool down ~10s, try next |
| Repeated failures | cooldown scales linearly (×2, ×3 … capped ×5) |
| All providers exhausted | raise `AllProvidersFailedError` |

Cooldowns are cleared on the next successful call. Editing `config.yaml` (e.g. fixing a key) gives a previously disabled provider a fresh chance.

## Project layout

```
free_llm_api/
├── config.yaml            # all providers + settings
├── endpoints.py           # public API: generate / stats / reload / configure
├── manager.py             # load, schedule, fail over, health tracking
├── scheduler.py           # weighted round-robin (+ fastest)
├── errors.py              # exception hierarchy with failure-handling hints
├── providers/
│   ├── base.py            # BaseProvider + OpenAICompatibleProvider
│   ├── registry.py        # @register plugin registry
│   └── openai_provider.py # ONE generic type for all OpenAI-compatible APIs
└── utils/
    └── config_loader.py   # YAML + ${ENV} expansion + mtime for hot reload
```
