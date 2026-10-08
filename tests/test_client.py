import json

import httpx
import pytest

from apertus_bench.client import StreamingChatClient
from apertus_bench.workloads import Prompt


@pytest.mark.asyncio
async def test_streaming_client_parses_usage_and_output() -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        assert payload["stream_options"] == {"include_usage": True}
        events = [
            {"choices": [{"delta": {"content": "hello"}, "finish_reason": None}]},
            {"choices": [{"delta": {"content": " world"}, "finish_reason": "stop"}]},
            {
                "choices": [],
                "usage": {"prompt_tokens": 7, "completion_tokens": 2, "total_tokens": 9},
            },
        ]
        body = "".join(f"data: {json.dumps(event)}\n\n" for event in events) + "data: [DONE]\n\n"
        return httpx.Response(200, text=body)

    client = StreamingChatClient("https://example.test", "target", None, timeout_seconds=10)
    await client.http.aclose()
    client.http = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    prompt = Prompt(
        id="p1",
        workload="chat",
        messages=[{"role": "user", "content": "hi"}],
        max_tokens=10,
    )
    async with client:
        result = await client.request(
            prompt,
            0,
            temperature=0.0,
            top_p=1.0,
            seed=1,
            max_tokens=None,
            ignore_eos=False,
            store_output=True,
        )
    assert result.success is True
    assert result.output == "hello world"
    assert result.prompt_tokens == 7
    assert result.completion_tokens == 2
    assert result.ttft_ms is not None
    assert result.tpot_ms is not None


@pytest.mark.asyncio
@pytest.mark.parametrize("top_k", [None, 20])
async def test_streaming_client_sends_top_k_only_when_given(top_k: int | None) -> None:
    seen: dict[str, object] = {}

    async def handler(request: httpx.Request) -> httpx.Response:
        seen.update(json.loads(request.content))
        body = (
            'data: {"choices": [{"delta": {"content": "x"}, "finish_reason": "stop"}]}\n\n'
            "data: [DONE]\n\n"
        )
        return httpx.Response(200, text=body)

    client = StreamingChatClient("https://example.test", "target", None, timeout_seconds=10)
    await client.http.aclose()
    client.http = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    prompt = Prompt(
        id="p1", workload="chat", messages=[{"role": "user", "content": "hi"}], max_tokens=4
    )
    async with client:
        await client.request(
            prompt,
            0,
            temperature=0.7,
            top_p=0.8,
            seed=1,
            max_tokens=None,
            ignore_eos=False,
            store_output=False,
            top_k=top_k,
        )
    assert ("top_k" in seen) is (top_k is not None)
    if top_k is not None:
        assert seen["top_k"] == top_k
