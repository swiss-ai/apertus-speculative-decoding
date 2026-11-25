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


@pytest.mark.asyncio
async def test_streaming_client_sends_chat_template_kwargs_only_when_set() -> None:
    seen: list[dict[str, object]] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        seen.append(json.loads(request.content))
        body = (
            'data: {"choices": [{"delta": {"content": "x"}, "finish_reason": "stop"}]}\n\n'
            "data: [DONE]\n\n"
        )
        return httpx.Response(200, text=body)

    client = StreamingChatClient("https://example.test", "target", None, timeout_seconds=10)
    await client.http.aclose()
    client.http = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    messages = [{"role": "user", "content": "hi"}]
    plain = Prompt(id="p1", workload="chat", messages=messages, max_tokens=4)
    thinking = Prompt(
        id="p2",
        workload="chat",
        messages=messages,
        max_tokens=4,
        chat_template_kwargs={"enable_thinking": True},
    )
    async with client:
        for prompt in (plain, thinking):
            await client.request(
                prompt,
                0,
                temperature=0.0,
                top_p=1.0,
                seed=1,
                max_tokens=None,
                ignore_eos=False,
                store_output=False,
            )
    assert "chat_template_kwargs" not in seen[0]
    assert seen[1]["chat_template_kwargs"] == {"enable_thinking": True}


@pytest.mark.parametrize("field", ["reasoning", "reasoning_content"])
async def test_native_effort_preserves_channels_and_raw_stream(field: str) -> None:
    async def handler(request):
        payload = json.loads(request.content)
        assert payload["reasoning_effort"] == "high"
        assert payload["min_p"] == 0
        events = [
            {"choices": [{"delta": {field: "Let me think."}}]},
            {"choices": [{"delta": {"content": "42"}, "finish_reason": "stop"}]},
            {
                "choices": [],
                "usage": {
                    "prompt_tokens": 8,
                    "completion_tokens": 12,
                    "completion_tokens_details": {"reasoning_tokens": 7},
                },
            },
        ]
        return httpx.Response(
            200,
            text="".join(f"data: {json.dumps(event)}\n\n" for event in events) + "data: [DONE]\n\n",
        )

    client = StreamingChatClient(
        "http://test", "m", None, timeout_seconds=1, token_counter=lambda text: len(text.split())
    )
    await client.http.aclose()
    client.http = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    async with client:
        result = await client.request(
            Prompt("p", "math", [{"role": "user", "content": "q"}], 100),
            0,
            temperature=1,
            top_p=1,
            min_p=0,
            reasoning_effort="high",
            seed=0,
            max_tokens=100,
            ignore_eos=False,
            store_output=True,
            store_stream=True,
        )
    assert result.reasoning_output == "Let me think."
    assert result.final_output == "42"
    assert result.reasoning_tokens == 3
    assert result.final_tokens == 1
    assert result.server_reasoning_tokens == 7
    assert result.channel_token_count_source == "tokenizer_reencoding"
    assert len(result.stream_events) == 3
    assert result.stream_events[0]["event"]["choices"][0]["delta"][field] == "Let me think."


async def test_conflicting_effort_is_refused_before_sending_request() -> None:
    async with StreamingChatClient("http://unused", "m", None, timeout_seconds=1) as client:
        with pytest.raises(ValueError, match="conflicts"):
            await client.request(
                Prompt(
                    "p",
                    "math",
                    [{"role": "user", "content": "q"}],
                    10,
                    chat_template_kwargs={"reasoning_effort": "low"},
                ),
                0,
                temperature=1,
                top_p=1,
                seed=0,
                max_tokens=10,
                ignore_eos=False,
                store_output=True,
                reasoning_effort="high",
            )
