from __future__ import annotations

from types import SimpleNamespace

import pytest

from apertus_bench import node_sweep


@pytest.mark.asyncio
async def test_failed_correctness_request_blocks_benchmark_and_keeps_capture(tmp_path, monkeypatch):
    monkeypatch.setattr(node_sweep, "load_prompts", lambda _: [object()])

    async def capture(client, prompts, path, **kwargs):
        path.parent.mkdir(parents=True)
        path.write_text('{"failed_request": true}')
        return {"cases": [{"measurement": {"success": False}}]}

    monkeypatch.setattr(node_sweep, "capture_greedy_outputs", capture)
    arm = SimpleNamespace(port=8100, served_model="native-control")
    with pytest.raises(RuntimeError, match="failed request"):
        await node_sweep.capture_drafter_kv_controls(
            arm, tmp_path, {"VLLM_DRAFT_KV_CORRECTNESS_WORKLOADS": "frozen-smoke.jsonl"},
        )
    assert (tmp_path / "correctness/capture-1.json").exists()
    assert not (tmp_path / "correctness/capture-2.json").exists()


@pytest.mark.asyncio
async def test_controls_capture_twice_and_never_overwrite(tmp_path, monkeypatch):
    monkeypatch.setattr(node_sweep, "load_prompts", lambda _: [object()])
    calls = []

    async def capture(client, prompts, path, **kwargs):
        calls.append(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('{"preserved": true}')
        return {"cases": [{"measurement": {"success": True}}]}

    monkeypatch.setattr(node_sweep, "capture_greedy_outputs", capture)
    arm = SimpleNamespace(port=8100, served_model="native-control")
    env = {"VLLM_DRAFT_KV_CORRECTNESS_WORKLOADS": "frozen-smoke.jsonl"}
    await node_sweep.capture_drafter_kv_controls(arm, tmp_path, env)
    assert len(calls) == 2
    with pytest.raises(FileExistsError):
        await node_sweep.capture_drafter_kv_controls(arm, tmp_path, env)
    assert len(calls) == 2


@pytest.mark.asyncio
async def test_other_sweeps_do_not_capture_or_read_inputs(tmp_path, monkeypatch):
    def unexpected(_):
        raise AssertionError("correctness inputs read for unrelated sweep")

    monkeypatch.setattr(node_sweep, "load_prompts", unexpected)
    await node_sweep.capture_drafter_kv_controls(None, tmp_path, {})
