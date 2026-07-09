# free-llm-api

Many LLM providers offer free API access, but each one has its own rate limits, and they run out or go down at different times. This project solves that problem. It lets you send one request and have it automatically routed to whichever free provider is available. If a provider fails or hits its rate limit, the request moves to the next one, so your app keeps working without extra code on your end.

You just call one simple function, and the library handles the rest: picking a provider, retrying on failure, and balancing load across all the providers you've configured.

```python
from free_llm_api import endpoints

response = endpoints.generate("Explain quantum computing")
print(response["text"])
```

## Features

- **Config-driven** — every provider lives in `config.yaml`, keys via `${ENV_VAR}`
- **Failover & load balancing** — any error (timeout, HTTP, 429) moves to the next provider, weighted round-robin by default
- **Hot reload** — edit `config.yaml` while it's running, changes pick up automatically
- **Plugin providers** — add a new one with `@register("name")`, no core edits needed

## Included free providers

`groq` · `openrouter` (`:free`) · `nvidia` · `mistral` · `zai` (Zhipu GLM) · `cerebras` · `llm7` (key-less) · `gemini` (Google AI Studio)

Provider list comes from
[awesome-free-llm-apis](https://github.com/mnfst/awesome-free-llm-apis).

## Install

```bash
pip install -r requirements.txt        # requests + PyYAML
# or, as a package:
pip install -e .
```

Requires Python 3.9+.

Check it works:

```bash
python tests/example.py          # end-to-end demo + provider stats
python tests/test_wrapper.py     # offline tests (no keys / network needed)
```

## Configure

Edit `free_llm_api/config.yaml` (or point `$FREE_LLM_API_CONFIG` at your own), then set keys for the providers you want to use. Use a `.env` file (easiest) or export them as environment variables. Get free keys at:

| Provider | Key env var | Sign-up |
|---|---|---|
| Groq | `GROQ_API_KEY` | https://console.groq.com/keys |
| Gemini (AI Studio) | `GEMINI_API_KEY` | https://aistudio.google.com/apikey |
| Cerebras | `CEREBRAS_API_KEY` | https://cloud.cerebras.ai |
| OpenRouter | `OPENROUTER_API_KEY` | https://openrouter.ai/keys |
| Mistral AI | `MISTRAL_API_KEY` | https://console.mistral.ai/api-keys |
| Z.AI (Zhipu) | `ZAI_API_KEY` | https://z.ai (API keys in console) |
| NVIDIA | `NVIDIA_API_KEY` | https://build.nvidia.com |
| LLM7.io | _(none — key-less)_ | https://llm7.io |

```bash
# Option A — .env file (recommended)
cp .env.example .env   # then edit .env with your keys

# Option B — export manually
export GROQ_API_KEY=gsk_...
export OPENROUTER_API_KEY=sk-or-...

# providers whose key is empty are skipped automatically — you don't need all of them
```

How keys are loaded (in order of priority): if a key is set as an OS environment variable,that's used first. Otherwise, the library checks your `.env` file. If neither is set, it falls back to the default in `config.yaml` (written as `${VAR:-default}`).

Free-tier models change often, and providers sometimes retire old model IDs. If you get a 404 error, it likely means the model ID has changed. Just open `config.yaml`, update the model name, and save — no restart needed.

## Usage

```python
from free_llm_api import endpoints, status

r = endpoints.generate(
    "Summarise the French Revolution",
    system="You are a concise historian.",
    max_tokens=300,
    temperature=0.4,
)
print(r["text"], "via", r["provider"])

# inspect config + live health
import json; print(json.dumps(endpoints.status(), indent=2))

# or just runtime stats (lighter)
print(json.dumps(endpoints.stats(), indent=2))

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

If every provider fails, `AllProvidersFailedError` is raised with a per-provider error map.

---

More on scheduling, adding providers, failure handling, and project layout: [docs/ADVANCED.md](docs/ADVANCED.md).
