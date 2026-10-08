"""Proxy for a disaggregated (prefill/decode) deployment on one node.

Each chat completion is first sent to a prefill instance with max_tokens 1 and
``kv_transfer_params.do_remote_decode``; the prefill instance computes the
prompt's KV cache and returns where to fetch it. The request then goes to a
decode instance with those parameters, which pulls the KV (NIXL) and streams
the answer back. Instances are used round robin. Same protocol as vLLM's
tests/v1/kv_connector/nixl_integration/toy_proxy_server.py, without its
connection-pool cap (100), and with /v1/models and /metrics answered by the
first decode instance so the bench client works unchanged.

    python3 -m apertus_bench.pd_proxy --port 8200 \\
        --prefill http://127.0.0.1:8300 --decode http://127.0.0.1:8301

Runs inside the serving container (FastAPI and uvicorn come with vLLM).
"""

from __future__ import annotations

import argparse
import itertools
import uuid
from typing import Any

import httpx

REMOTE_DECODE = {
    "do_remote_decode": True,
    "do_remote_prefill": False,
    "remote_engine_id": None,
    "remote_block_ids": None,
    "remote_host": None,
    "remote_port": None,
}


def prefill_request(body: dict[str, Any]) -> dict[str, Any]:
    """The prefill leg: one token, no streaming, KV kept for a remote decode."""
    request = {key: value for key, value in body.items() if key not in {"stream_options"}}
    request["kv_transfer_params"] = dict(REMOTE_DECODE)
    request["stream"] = False
    request["max_tokens"] = 1
    if "max_completion_tokens" in request:
        request["max_completion_tokens"] = 1
    request.pop("min_tokens", None)
    request.pop("min_completion_tokens", None)
    return request


def decode_request(body: dict[str, Any], prefill_response: dict[str, Any]) -> dict[str, Any]:
    """The decode leg: the client's request plus where to fetch the prompt's KV."""
    request = dict(body)
    params = prefill_response.get("kv_transfer_params")
    if params:
        request["kv_transfer_params"] = params
    return request


def build_app(prefill: list[str], decode: list[str]):
    from fastapi import FastAPI, Request
    from fastapi.responses import JSONResponse, Response, StreamingResponse

    # No connection cap: the load generator sets the concurrency.
    client = httpx.AsyncClient(
        timeout=httpx.Timeout(None), limits=httpx.Limits(max_connections=None)
    )
    prefill_cycle = itertools.cycle(prefill)
    decode_cycle = itertools.cycle(decode)
    app = FastAPI()

    @app.post("/v1/chat/completions")
    async def chat(request: Request):
        body = await request.json()
        headers = {"X-Request-Id": str(uuid.uuid4())}
        first = await client.post(
            f"{next(prefill_cycle)}/v1/chat/completions",
            json=prefill_request(body),
            headers=headers,
        )
        if first.status_code != 200:
            return Response(first.content, status_code=first.status_code)
        target = f"{next(decode_cycle)}/v1/chat/completions"
        second = decode_request(body, first.json())
        if not body.get("stream"):
            response = await client.post(target, json=second, headers=headers)
            return Response(
                response.content,
                status_code=response.status_code,
                media_type="application/json",
            )

        upstream = await client.send(
            client.build_request("POST", target, json=second, headers=headers), stream=True
        )

        async def relay():
            try:
                async for chunk in upstream.aiter_raw():
                    yield chunk
            finally:
                await upstream.aclose()

        return StreamingResponse(
            relay(), status_code=upstream.status_code, media_type="text/event-stream"
        )

    @app.get("/v1/models")
    async def models():
        response = await client.get(f"{decode[0]}/v1/models")
        return JSONResponse(response.json(), status_code=response.status_code)

    @app.get("/metrics")
    async def metrics():
        response = await client.get(f"{decode[0]}/metrics")
        return Response(response.content, media_type="text/plain; version=0.0.4")

    @app.get("/healthcheck")
    async def health():
        return {"prefill": prefill, "decode": decode}

    return app


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="apertus_bench.pd_proxy")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--prefill", nargs="+", required=True, help="prefill base URLs")
    parser.add_argument("--decode", nargs="+", required=True, help="decode base URLs")
    args = parser.parse_args(argv)
    import uvicorn

    uvicorn.run(
        build_app(args.prefill, args.decode), host=args.host, port=args.port, log_level="warning"
    )


if __name__ == "__main__":
    main()
