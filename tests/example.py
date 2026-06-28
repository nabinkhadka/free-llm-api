"""Ask the same question several times and see which provider answered.

    python3 tests/example.py        # 10 calls
    python3 tests/example.py 20     # custom count

Each call is routed by the weighted round-robin scheduler, so across several
calls you'll see the providers rotate and can compare their answers.
"""
import logging
import sys
import textwrap

from free_llm_api import endpoints

# WARNING keeps output clean; failures still print. Set to INFO to trace routing.
logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")

QUESTION = "Generate a python script to generate first 10 numbers in the fibonacci series. Only generate code, do not generate anything else."


def main() -> None:
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 10

    print(f"\nAsking the same question {n} times:\nQ: {QUESTION}\n")
    for i in range(n):
        try:
            r = endpoints.generate(QUESTION, max_tokens=120, temperature=0.3)
            print(f"#{i + 1:2}  [{r['provider']}]  model={r['model']}  ({r.get('latency')}s)")
            print(textwrap.fill(r["text"].strip(), width=88,
                                initial_indent="     ", subsequent_indent="     "))
        except Exception as exc:
            print(f"#{i + 1:2}  ALL PROVIDERS FAILED: {exc}")
        print()


if __name__ == "__main__":
    main()
