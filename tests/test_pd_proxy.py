from apertus_bench.pd_proxy import decode_request, prefill_request


def test_prefill_leg_asks_for_one_token_and_a_remote_decode() -> None:
    body = {
        "model": "m",
        "messages": [{"role": "user", "content": "hi"}],
        "stream": True,
        "stream_options": {"include_usage": True},
        "max_tokens": 256,
        "min_tokens": 4,
    }
    request = prefill_request(body)
    assert request["max_tokens"] == 1
    assert request["stream"] is False
    assert "stream_options" not in request
    assert "min_tokens" not in request
    assert request["kv_transfer_params"]["do_remote_decode"] is True
    assert body["max_tokens"] == 256  # the client's request is left alone


def test_decode_leg_carries_the_prefill_kv_parameters() -> None:
    body = {"model": "m", "messages": [], "stream": True, "max_tokens": 256}
    params = {"remote_engine_id": "e", "remote_block_ids": [1, 2]}
    request = decode_request(body, {"kv_transfer_params": params})
    assert request["kv_transfer_params"] == params
    assert request["max_tokens"] == 256 and request["stream"] is True
    assert "kv_transfer_params" not in decode_request(body, {})
