"""OpenAI-compatible API server wrapping llm_api_wrapper.

Run::

    python server.py                    # http://localhost:8000
    python server.py --port 8080
    python server.py --host 0.0.0.0     # accessible on LAN

Then point Cursor / VS Code (Continue / Cline) at::

    http://localhost:8000/v1/chat/completions

No API key is required (the server uses the providers configured in config.yaml).
"""

from __future__ import annotations

import json
import logging
import time
from http.server import HTTPServer, BaseHTTPRequestHandler
from typing import Any

from llm_api_wrapper import endpoints

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("openai-server")

_CHAT_ID = 0


def _chat_id() -> str:
    global _CHAT_ID
    _CHAT_ID += 1
    return f"chatcmpl-{_CHAT_ID}"


def _build_openai_response(provider_result: dict, chat_id: str) -> dict:
    return {
        "id": chat_id,
        "object": "chat.completion",
        "created": int(time.time()),
        "model": provider_result.get("model", "llm-wrapper"),
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": provider_result["text"],
                },
                "finish_reason": "stop",
            }
        ],
        "usage": provider_result.get("usage"),
    }


def _to_gen_kwargs(body: dict) -> dict[str, Any]:
    """Extract generation kwargs from the OpenAI request body."""
    messages = body.get("messages", [])
    kwargs: dict[str, Any] = {}

    system = None
    user_messages = []
    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        if role == "system":
            system = (system + "\n" + content) if system else content
        else:
            user_messages.append(msg)

    if system:
        kwargs["system"] = system
    kwargs["messages"] = user_messages

    if body.get("max_tokens"):
        kwargs["max_tokens"] = body["max_tokens"]
    if body.get("temperature"):
        kwargs["temperature"] = body["temperature"]
    if body.get("top_p"):
        kwargs["top_p"] = body["top_p"]
    if body.get("stop"):
        kwargs["stop"] = body["stop"]
    return kwargs


class Handler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        log.info(format, *args)

    def _send_json(self, data: Any, status: int = 200):
        body = json.dumps(data, indent=2).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_error_json(self, message: str, status: int = 500):
        self._send_json({"error": {"message": message, "type": "server_error"}}, status)

    def do_GET(self):
        if self.path == "/v1/models":
            return self._handle_models()
        elif self.path == "/health":
            return self._send_json({"status": "ok"})
        self._send_error_json("Not found", 404)

    def do_POST(self):
        if self.path == "/v1/chat/completions":
            return self._handle_chat_completions()
        self._send_error_json("Not found", 404)

    def _read_body(self) -> dict:
        length = int(self.headers.get("Content-Length", 0))
        raw = self.rfile.read(length) if length else b"{}"
        return json.loads(raw)

    def _handle_models(self):
        models = []
        for name, info in endpoints.stats().items():
            models.append({
                "id": info.get("model", name),
                "object": "model",
                "owned_by": name,
                "provider": name,
            })
        if not models:
            models.append({"id": "llm-wrapper", "object": "model", "owned_by": "llm_api_wrapper"})
        self._send_json({"object": "list", "data": models})

    def _handle_chat_completions(self):
        try:
            body = self._read_body()
        except json.JSONDecodeError:
            return self._send_error_json("Invalid JSON body", 400)

        messages = body.get("messages", [])
        if not messages:
            return self._send_error_json("messages field is required", 400)

        stream = body.get("stream", False)
        gen_kwargs = _to_gen_kwargs(body)

        if stream:
            self._handle_streaming(gen_kwargs)
        else:
            try:
                result = endpoints.generate("", **gen_kwargs)
            except Exception as e:
                log.exception("All providers failed")
                return self._send_error_json(str(e), 503)
            cid = _chat_id()
            self._send_json(_build_openai_response(result, cid))

    def _handle_streaming(self, gen_kwargs: dict):
        cid = _chat_id()
        created = int(time.time())

        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()

        model = "llm-wrapper"
        try:
            for chunk in endpoints.stream_generate("", **gen_kwargs):
                content = chunk.get("text", "")
                if not content:
                    continue
                if chunk.get("model"):
                    model = chunk["model"]
                sse = {
                    "id": cid,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": model,
                    "choices": [
                        {
                            "index": 0,
                            "delta": {"content": content},
                            "finish_reason": chunk.get("finish_reason"),
                        }
                    ],
                }
                self.wfile.write(f"data: {json.dumps(sse)}\n\n".encode())
                self.wfile.flush()
        except Exception as e:
            log.exception("Streaming failed")
            # Send error as a chunk so the client sees it
            err_sse = {
                "id": cid,
                "object": "chat.completion.chunk",
                "created": created,
                "model": model,
                "choices": [
                    {
                        "index": 0,
                        "delta": {"content": f"\n\n[Error: {e}]"},
                        "finish_reason": "stop",
                    }
                ],
            }
            self.wfile.write(f"data: {json.dumps(err_sse)}\n\n".encode())

        final_chunk = {
            "id": cid,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model,
            "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
        }
        self.wfile.write(f"data: {json.dumps(final_chunk)}\n\n".encode())
        self.wfile.write("data: [DONE]\n\n".encode())
        self.wfile.flush()
        self.close_connection = True


def main():
    import argparse

    parser = argparse.ArgumentParser(description="OpenAI-compatible server for llm_api_wrapper")
    parser.add_argument("--host", default="127.0.0.1", help="Bind address (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8000, help="Port (default: 8000)")
    args = parser.parse_args()

    server = HTTPServer((args.host, args.port), Handler)
    log.info("OpenAI-compatible server listening on http://%s:%d", args.host, args.port)
    log.info("Endpoint: http://%s:%d/v1/chat/completions", args.host, args.port)
    log.info("Models:   http://%s:%d/v1/models", args.host, args.port)
    log.info("Health:   http://%s:%d/health", args.host, args.port)
    log.info("(no API key needed — providers are configured in config.yaml)")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        log.info("Shutting down...")
        server.shutdown()


if __name__ == "__main__":
    main()
