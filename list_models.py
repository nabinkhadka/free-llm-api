"""List the models each configured provider currently offers.

Run it in the SAME shell where your API keys are exported:

    python3 list_models.py

For every OpenAI-compatible provider it calls GET {base_url}/models using the
key from your environment, so you see the exact, current model IDs to put in
config.yaml. Use this whenever a provider returns "unknown model" / 404.
"""
import requests

from llm_api_wrapper.utils.config_loader import load_config


def main() -> None:
    cfg = load_config()
    for p in cfg["providers"]:
        name = p.get("name", "?")
        ptype = p.get("type", name)
        if not p.get("enabled", True):
            print(f"\n[{name}] disabled — skipped")
            continue
        if ptype != "openai":
            print(f"\n[{name}] type={ptype} — no OpenAI /models endpoint, skipped")
            continue

        base = (p.get("extra") or {}).get("base_url", "").rstrip("/")
        key = p.get("api_key") or None
        url = base + "/models"
        headers = {"Authorization": f"Bearer {key}"} if key else {}

        try:
            r = requests.get(url, headers=headers, timeout=20)
        except requests.RequestException as exc:
            print(f"\n[{name}] {url}\n    REQUEST ERROR: {exc}")
            continue

        if r.status_code >= 400:
            print(f"\n[{name}] {url}\n    HTTP {r.status_code}: {r.text[:160]}")
            continue

        data = r.json()
        rows = data.get("data", data) if isinstance(data, dict) else data
        ids = sorted(m.get("id") for m in rows if isinstance(m, dict) and m.get("id"))
        current = p.get("model")
        ok = "OK" if current in ids else "NOT IN LIST"
        print(f"\n[{name}] {len(ids)} models  (configured: {current} -> {ok})")
        for mid in ids:
            print(f"    {mid}")


if __name__ == "__main__":
    main()
