from __future__ import annotations

import hashlib
import json
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from typing import Any

import httpx

from apertus_bench.workloads import Prompt


@dataclass
class RequestMeasurement:
    request_index: int
    prompt_id: str
    workload: str
    success: bool
    status_code: int | None
    e2e_ms: float
    ttft_ms: float | None
    tpot_ms: float | None
    stream_event_gaps_ms: list[float] = field(default_factory=list)
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    finish_reason: str | None = None
    output_sha256: str | None = None
    output: str | None = None
    error: str | None = None
    reasoning_output: str | None = None
    final_output: str | None = None
    reasoning_tokens: int | None = None
    final_tokens: int | None = None
    channel_token_count_source: str | None = None
    server_reasoning_tokens: int | None = None
    stream_events: list[dict[str, Any]] | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _reasoning_payload(delta: dict[str, Any]) -> str:
    # New vLLM uses reasoning; older releases use reasoning_content. These are
    # aliases, not two distinct channels to concatenate.
    for key in ("reasoning", "reasoning_content"):
        value = delta.get(key)
        if isinstance(value, str) and value:
            return value
    return ""


def _token_payload(delta: dict[str, Any]) -> str:
    pieces: list[str] = []
    for key in ("content",):
        value = delta.get(key)
        if isinstance(value, str) and value:
            pieces.append(value)
    pieces.append(_reasoning_payload(delta))
    for tool_call in delta.get("tool_calls") or []:
        function = tool_call.get("function") or {}
        arguments = function.get("arguments")
        if isinstance(arguments, str) and arguments:
            pieces.append(arguments)
    return "".join(pieces)


class StreamingChatClient:
    def __init__(
        self,
        base_url: str,
        model: str,
        api_key: str | None,
        *,
        timeout_seconds: float,
        token_counter: Callable[[str], int] | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.token_counter = token_counter
        headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        timeout = httpx.Timeout(timeout_seconds, connect=min(timeout_seconds, 30.0))
        # No pool cap: concurrency is set by the caller's workers. httpx's default of
        # 100 connections silently queued requests client-side above concurrency 100
        # (load test 2026-09-29: at C=256 the server never saw more than 99).
        # Drop idle connections before the server does: vLLM's server closes idle
        # keep-alive connections after 5 s, httpx's default expiry is also 5 s, and a
        # connection reused at that moment fails with ReadError (load test, C=256).
        limits = httpx.Limits(
            max_connections=None, max_keepalive_connections=None, keepalive_expiry=1.0
        )
        self.http = httpx.AsyncClient(headers=headers, timeout=timeout, limits=limits)

    async def __aenter__(self) -> StreamingChatClient:
        return self

    async def __aexit__(self, *args: object) -> None:
        await self.http.aclose()

    async def request(
        self,
        prompt: Prompt,
        request_index: int,
        *,
        temperature: float,
        top_p: float,
        seed: int,
        max_tokens: int | None,
        ignore_eos: bool,
        store_output: bool,
        top_k: int | None = None,
        min_p: float | None = None,
        reasoning_effort: str | None = None,
        store_stream: bool = False,
    ) -> RequestMeasurement:
        payload = {
            "model": self.model,
            "messages": prompt.messages,
            "max_tokens": max_tokens or prompt.max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "seed": seed,
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        if prompt.chat_template_kwargs:
            payload["chat_template_kwargs"] = prompt.chat_template_kwargs
        if top_k is not None:
            # vLLM extension of the OpenAI schema; omitted means the server default.
            payload["top_k"] = top_k
        if min_p is not None:
            payload["min_p"] = min_p
        if reasoning_effort is not None:
            if reasoning_effort not in {"low", "medium", "high"}:
                raise ValueError("reasoning_effort must be low, medium or high")
            if (prompt.chat_template_kwargs or {}).get("reasoning_effort") not in (
                None,
                reasoning_effort,
            ):
                raise ValueError("prompt chat template conflicts with requested reasoning_effort")
            payload["reasoning_effort"] = reasoning_effort
        if ignore_eos:
            payload["ignore_eos"] = True

        started = time.perf_counter()
        first_payload_at: float | None = None
        payload_times: list[float] = []
        output_parts: list[str] = []
        reasoning_parts: list[str] = []
        final_parts: list[str] = []
        raw_events: list[dict[str, Any]] = []
        usage: dict[str, Any] | None = None
        finish_reason: str | None = None
        status_code: int | None = None

        try:
            async with self.http.stream(
                "POST", f"{self.base_url}/v1/chat/completions", json=payload
            ) as response:
                status_code = response.status_code
                if response.status_code >= 400:
                    body = (await response.aread()).decode(errors="replace")
                    raise RuntimeError(f"HTTP {response.status_code}: {body[:500]}")
                async for line in response.aiter_lines():
                    if not line.startswith("data:"):
                        continue
                    raw_event = line[5:].strip()
                    if not raw_event or raw_event == "[DONE]":
                        continue
                    event = json.loads(raw_event)
                    if store_stream:
                        raw_events.append(
                            {
                                "elapsed_ms": (time.perf_counter() - started) * 1000,
                                "event": event,
                            }
                        )
                    if event.get("usage"):
                        usage = event["usage"]
                    choices = event.get("choices") or []
                    if not choices:
                        continue
                    choice = choices[0]
                    if choice.get("finish_reason"):
                        finish_reason = choice["finish_reason"]
                    delta = choice.get("delta") or {}
                    reasoning_parts.append(_reasoning_payload(delta))
                    if isinstance(delta.get("content"), str):
                        final_parts.append(delta["content"])
                    piece = _token_payload(delta)
                    if piece:
                        now = time.perf_counter()
                        if first_payload_at is None:
                            first_payload_at = now
                        payload_times.append(now)
                        output_parts.append(piece)
        except Exception as error:  # the failed request remains part of the load sample
            ended = time.perf_counter()
            return RequestMeasurement(
                request_index=request_index,
                prompt_id=prompt.id,
                workload=prompt.workload,
                success=False,
                status_code=status_code,
                e2e_ms=(ended - started) * 1000,
                ttft_ms=(first_payload_at - started) * 1000 if first_payload_at else None,
                tpot_ms=None,
                error=f"{type(error).__name__}: {error}",
                output="".join(output_parts) if store_output else None,
                reasoning_output="".join(reasoning_parts) if store_output else None,
                final_output="".join(final_parts) if store_output else None,
                stream_events=raw_events if store_stream else None,
            )

        ended = time.perf_counter()
        output = "".join(output_parts)
        completion_tokens = int(usage["completion_tokens"]) if usage else None
        prompt_tokens = int(usage["prompt_tokens"]) if usage else None
        ttft_ms = (first_payload_at - started) * 1000 if first_payload_at else None
        e2e_ms = (ended - started) * 1000
        tpot_ms = None
        if ttft_ms is not None and completion_tokens is not None and completion_tokens > 1:
            tpot_ms = (e2e_ms - ttft_ms) / (completion_tokens - 1)
        event_gaps = [
            (current - previous) * 1000
            for previous, current in zip(payload_times, payload_times[1:], strict=False)
        ]
        error = None if usage else "stream completed without usage; token throughput unavailable"
        reasoning = "".join(reasoning_parts)
        final = "".join(final_parts)
        details = (usage or {}).get("completion_tokens_details") or {}
        return RequestMeasurement(
            request_index=request_index,
            prompt_id=prompt.id,
            workload=prompt.workload,
            success=usage is not None,
            status_code=status_code,
            e2e_ms=e2e_ms,
            ttft_ms=ttft_ms,
            tpot_ms=tpot_ms,
            stream_event_gaps_ms=event_gaps,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            finish_reason=finish_reason,
            output_sha256=hashlib.sha256(output.encode()).hexdigest(),
            output=output if store_output else None,
            error=error,
            reasoning_output=reasoning if store_output else None,
            final_output=final if store_output else None,
            reasoning_tokens=self.token_counter(reasoning) if self.token_counter else None,
            final_tokens=self.token_counter(final) if self.token_counter else None,
            channel_token_count_source="tokenizer_reencoding" if self.token_counter else None,
            server_reasoning_tokens=details.get("reasoning_tokens"),
            stream_events=raw_events if store_stream else None,
        )
