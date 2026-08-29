from __future__ import annotations

import hashlib
import json
import time
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

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _token_payload(delta: dict[str, Any]) -> str:
    pieces: list[str] = []
    for key in ("content", "reasoning_content"):
        value = delta.get(key)
        if isinstance(value, str) and value:
            pieces.append(value)
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
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        timeout = httpx.Timeout(timeout_seconds, connect=min(timeout_seconds, 30.0))
        self.http = httpx.AsyncClient(headers=headers, timeout=timeout)

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
        if ignore_eos:
            payload["ignore_eos"] = True

        started = time.perf_counter()
        first_payload_at: float | None = None
        payload_times: list[float] = []
        output_parts: list[str] = []
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
                    if event.get("usage"):
                        usage = event["usage"]
                    choices = event.get("choices") or []
                    if not choices:
                        continue
                    choice = choices[0]
                    if choice.get("finish_reason"):
                        finish_reason = choice["finish_reason"]
                    piece = _token_payload(choice.get("delta") or {})
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
        )
