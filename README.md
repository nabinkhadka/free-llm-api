# free-llm-api / FreeLLMAPI

A lightweight, **OpenRouter-style router for FREE LLM APIs**. It rotates across
multiple free providers, fails over the instant one errors out, respects rate
limits with cooldowns, and is driven entirely by a `config.yaml` you can edit
while it runs.

```python
from free_llm_api import endpoints

response = endpoints.generate("Explain quantum computing")
print(response["text"])
```

Built for **resilience, simplicity, and easy extensibility** on top of
unreliable free tiers. Providers come from
[awesome-free-llm-apis](https://github.com/mnfst/awesome-free-llm-apis).

---

## Features

| Requirement | How it's met |
|---|---|
| Config-driven | Every provider lives in `config.yaml`; keys via `${ENV_VAR}` |
| Plugin providers | `@register("name")` + auto-discovery — **no core edits to add one** |
| Scheduling | Smooth **weighted round-robin** (nginx algorithm), or `fastest` |
| Failover | Any error (timeout / HTTP / 429) → immediately try next provider |
| Smart failures | 429 → cooldown (honours `Retry-After`); 401/403 → disable; timeout → retry later |
| Cooldown system | Per-provider "skip until" with linear backoff on repeated failures |
| Hot reload | Edits to `config.yaml` are picked up automatically — new keys work at once |
| Bonus | Per-provider latency tracking, `fastest` strategy |

### Included free providers

`groq` · `openrouter` (`:free`) · `nvidia` · `mistral` · `zai` (Zhipu GLM) ·
`cerebras` · `llm7` (key-less) · `gemini` (Google AI Studio)

All are OpenAI-compatible, so they share **one** generic `type: openai`
implementation and differ only by `base_url` in config — adding another such
provider needs zero code.

---

## Install

```bash
pip install -r requirements.txt        # requests + PyYAML
# or, as a package:
pip install -e .
```

Requires Python 3.9+.

## Configure

Edit `free_llm_api/config.yaml` (or point `$FREE_LLM_API_CONFIG` at your own),
then export keys for the providers you have. Get free keys at:

| Provider | Key env var | Sign-up |
|---|---|---|
| Groq ⭐ | `GROQ_API_KEY` | https://console.groq.com/keys |
| Gemini (AI Studio) | `GEMINI_API_KEY` | https://aistudio.google.com/apikey |
| Cerebras | `CEREBRAS_API_KEY` | https://cloud.cerebras.ai |
| OpenRouter | `OPENROUTER_API_KEY` | https://openrouter.ai/keys |
| Mistral AI | `MISTRAL_API_KEY` | https://console.mistral.ai/api-keys |
| Z.AI (Zhipu) | `ZAI_API_KEY` | https://z.ai (API keys in console) |
| NVIDIA | `NVIDIA_API_KEY` | https://build.nvidia.com |
| LLM7.io | _(none — key-less)_ | https://llm7.io |

```bash
export GROQ_API_KEY=gsk_...
export OPENROUTER_API_KEY=sk-or-...
# providers whose key is empty are skipped automatically — you don't need all of them
```

Config resolution order: explicit path → `$FREE_LLM_API_CONFIG` → `./config.yaml`
→ the packaged default.

## Run

```bash
python tests/example.py          # end-to-end demo + provider stats
python tests/test_wrapper.py   # offline tests (no keys / network needed)
```

---

## Usage

```python
from free_llm_api import endpoints

# basic
r = endpoints.generate("Write a haiku about the sea")
print(r["text"], "via", r["provider"])

# generation params are passed straight through to the chosen provider
r = endpoints.generate(
    "Summarise the French Revolution",
    system="You are a concise historian.",
    max_tokens=300,
    temperature=0.4,
)

# inspect live health / latency
import json; print(json.dumps(endpoints.stats(), indent=2))

# force a reload (also happens automatically when config.yaml changes)
endpoints.reload()
```

Every response is normalized:

```python
{
  "text": "...",          # the completion
  "provider": "groq",     # which provider answered
  "model": "llama-3.3-70b-versatile",
  "latency": 0.42,        # seconds
  "usage": {...},         # if the provider reports it
  "raw": {...},           # untouched provider payload
}
```

If **every** provider fails, `AllProvidersFailedError` is raised with a
per-provider error map.

---

## Add a new provider

**If it's OpenAI-compatible (most are): no code at all.** Just add a config
block pointing the generic `openai` type at its base URL:

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

**If its API isn't OpenAI-shaped:** drop a `*_provider.py` file that subclasses
`BaseProvider`, implements `generate()` to return `{"text", "provider",
"model"}`, and raises a `ProviderError` subclass on failure. Register it with
`@register("yourname")` and reference it via `type: yourname`. It's
auto-discovered on import — the manager and scheduler never need to know it
exists.

---

## How it behaves on failure

| Situation | Reaction |
|---|---|
| Timeout / connection error | `ProviderTimeoutError` → cool down ~15s, try next |
| HTTP 429 | `RateLimitError` → cool down ≥60s (or `Retry-After`), try next |
| HTTP 401 / 403 | `InvalidKeyError` → **disable** provider until next reload |
| HTTP 5xx / bad body | `ProviderError` → cool down ~10s, try next |
| Repeated failures | cooldown scales linearly (×2, ×3 … capped ×5) |
| All providers exhausted | raise `AllProvidersFailedError` |

Cooldowns are cleared on the next successful call. Editing `config.yaml` (e.g.
fixing a key) gives a previously disabled provider a fresh chance.

---

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

## Notes

* Only free tiers are configured. For OpenRouter, keep the `:free` model suffix.
* Model IDs drift on free tiers — if one 404s, swap it in `config.yaml` (no restart needed).
```
