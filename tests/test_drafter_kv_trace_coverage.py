"""Exercise host-only trace scheduling without importing vLLM/Torch on the CPU host."""

import ast
from pathlib import Path
from types import SimpleNamespace

from draft_kv.tracing import first_completed_prompt_requests, trace_due


def integration_hooks():
    """Isolate only the GPU integration hooks; exercise host helpers by import."""
    path = Path(__file__).resolve().parents[1] / "serving/draft_kv/integration.py"
    tree = ast.parse(path.read_text())
    names = {"trace_native", "notify_request_removed"}
    functions = [node for node in tree.body if isinstance(node, ast.FunctionDef)
                 and node.name in names]
    namespace = {"HORIZON": 2048, "trace_record": lambda record: None,
                 "trace_due": trace_due,
                 "first_completed_prompt_requests": first_completed_prompt_requests}
    exec(compile(ast.Module(body=functions, type_ignores=[]), str(path), "exec"), namespace)
    return namespace


def batch(ids, computed, scheduled, lengths):
    return SimpleNamespace(req_ids=ids, num_computed_prefill_tokens_np=computed,
                           num_scheduled_tokens=scheduled, prefill_len_np=lengths)


def test_chunked_prefill_samples_only_first_completed_prompt_not_first_chunk(monkeypatch):
    monkeypatch.setenv("VLLM_DRAFT_KV_TRACE", "/trace")
    sample = first_completed_prompt_requests
    owner = SimpleNamespace()
    assert sample(owner, batch(["long"], [0], [1024], [3000])) == []
    assert sample(owner, batch(["long"], [1024], [1024], [3000])) == []
    assert sample(owner, batch(["long"], [2048], [952], [3000])) == ["long"]
    assert sample(owner, batch(["long"], [3000], [4], [3000])) == []
    integration_hooks()["notify_request_removed"](owner, "long")
    assert sample(owner, batch(["long"], [2048], [952], [3000])) == ["long"]


def test_batch_completion_is_independent_per_request_and_disabled_trace_has_no_state(monkeypatch):
    sample = first_completed_prompt_requests
    owner = SimpleNamespace()
    monkeypatch.delenv("VLLM_DRAFT_KV_TRACE", raising=False)
    assert sample(owner, batch(["a"], [0], [10], [10])) == []
    assert not hasattr(owner, "_apertus_trace_completed_requests")
    monkeypatch.setenv("VLLM_DRAFT_KV_TRACE", "/trace")
    assert sample(owner, batch(["a", "b"], [0, 0], [10, 10], [10, 20])) == ["a"]
    assert sample(owner, batch(["a", "b"], [10, 10], [4, 10], [10, 20])) == ["b"]


class HostTensor:
    def __init__(self, values):
        self.values = values

    def cpu(self):
        return self

    def tolist(self):
        return self.values


def test_native_nonperiodic_completed_prefill_is_forwarded_as_same_output_snapshot(monkeypatch):
    monkeypatch.setenv("VLLM_DRAFT_KV_TRACE", "/trace")
    functions = integration_hooks()
    records = []
    functions["trace_record"] = records.append
    owner = SimpleNamespace(_apertus_trace_step=60, query_cudagraph_manager=None,
                            num_query_per_req=8)
    current = batch(["summarization"], [2000], [100], [2100])
    current.positions = HostTensor([2099])
    current.query_start_loc_np = [0, 1]
    current.num_reqs = 1
    current.num_tokens = 100
    functions["trace_native"](owner, current, HostTensor([0]))
    assert len(records) == 1
    assert records[0]["step"] == 61
    assert records[0]["native_sequence_lengths"] == [2100]
    assert records[0]["first_completed_prompt_request_ids"] == ["summarization"]
    assert owner._apertus_worker_trace is records[0]
    functions["trace_native"](owner, current, HostTensor([0]))
    assert len(records) == 1
    assert owner._apertus_worker_trace is None
