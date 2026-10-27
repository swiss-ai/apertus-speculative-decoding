import json
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest

from apertus_bench.client import StreamingChatClient
from apertus_bench.effort import (
    SPEC_PATH,
    measure_cell,
    read_spec,
    render_check,
    schedule,
    serve_command,
    training_plan,
)
from apertus_bench.effort_report import IDENTITY, build_report
from apertus_bench.effort_server import pin_harmony_date
from apertus_bench.workloads import Prompt


def test_native_harmony_date_overlay_preserves_explicit_dates_and_effort():
    def original(reasoning_effort=None, start_date=None):
        return reasoning_effort, start_date

    module = SimpleNamespace(get_system_message=original)
    assert pin_harmony_date(module, "2026-10-27") is original
    assert module.get_system_message("high") == ("high", "2026-10-27")
    assert module.get_system_message("low", "2026-01-01") == ("low", "2026-01-01")


def test_fixed_schedule_and_server_arguments(tmp_path):
    spec = read_spec(SPEC_PATH)
    assert schedule(0) == (["plain", "dflash"], ["low", "medium", "high"])
    assert schedule(1) == (["dflash", "plain"], ["medium", "high", "low"])
    plain = serve_command(spec, None, provenance=tmp_path / "plain.json")
    draft = serve_command(spec, Path("/head"), provenance=tmp_path / "draft.json")
    assert "--speculative-config" not in plain
    assert "apertus_bench.effort_server" in plain
    assert "--no-async-scheduling" in plain and "--no-async-scheduling" in draft
    assert json.loads(draft[draft.index("--speculative-config") + 1]) == {
        "method": "dflash",
        "model": "/head",
        "num_speculative_tokens": 7,
    }


async def test_render_check_uses_actual_chat_render_route(tmp_path):
    async def handler(request):
        assert request.url.path == "/v1/chat/completions/render"
        assert json.loads(request.content)["reasoning_effort"] == "high"
        return httpx.Response(200, json={"token_ids": [1, 2]})

    client = StreamingChatClient("http://test", "m", None, timeout_seconds=1)
    await client.http.aclose()
    client.http = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    tokenizer = SimpleNamespace(
        decode=lambda *a, **k: "Current date: 2026-10-27\nReasoning: high\n"
    )
    async with client:
        await render_check(
            client,
            tokenizer,
            "high",
            Prompt(
                "p",
                "math",
                [{"role": "user", "content": "q"}],
                10,
            ),
            tmp_path,
            read_spec(SPEC_PATH),
        )
    assert json.loads((tmp_path / "high-p.json").read_text())["token_ids"] == [1, 2]


async def test_measure_cell_journals_streams_counters_and_rejects_overwrite(tmp_path):
    rounds = 0

    def snapshot():
        return (
            "\n".join(
                [
                    f'vllm:spec_decode_num_drafts_total{{model_name="m"}} {rounds}',
                    f'vllm:spec_decode_num_draft_tokens_total{{model_name="m"}} {rounds * 7}',
                    f'vllm:spec_decode_num_accepted_tokens_total{{model_name="m"}} {rounds * 4}',
                ]
            )
            + "\n"
        )

    async def handler(request):
        nonlocal rounds
        if request.url.path == "/metrics":
            return httpx.Response(200, text=snapshot())
        assert json.loads(request.content)["reasoning_effort"] == "low"
        rounds += 1
        return httpx.Response(
            200,
            text='data: {"choices":[{"delta":{"reasoning":"r"}}]}\n\n'
            'data: {"choices":[{"delta":{"content":"a"},'
            '"finish_reason":"stop"}]}\n\n'
            'data: {"choices":[],"usage":{"prompt_tokens":10,'
            '"completion_tokens":8}}\n\ndata: [DONE]\n\n',
        )

    client = StreamingChatClient("http://test", "m", None, timeout_seconds=1, token_counter=len)
    await client.http.aclose()
    client.http = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    spec = {**read_spec(SPEC_PATH), "warmup_requests": 1}
    prompts = [Prompt("p", "math", [{"role": "user", "content": "q"}], 32768)]
    async with client:
        summary = await measure_cell(
            client, prompts, tmp_path / "cell", spec, "low", 0, {}, speculative=True
        )
        with pytest.raises(FileExistsError):
            await measure_cell(
                client, prompts, tmp_path / "cell", spec, "low", 0, {}, speculative=True
            )
    row = json.loads((tmp_path / "cell/requests.jsonl").read_text())
    assert row["reasoning_output"] == "r" and row["final_output"] == "a"
    assert row["speculative_decoding"]["drafts"] == 1  # warmup excluded
    assert row["speculative_decoding"]["accepted_tokens"] == 4
    assert len(row["stream_events"]) == 3
    assert summary["speculative_decoding"]["drafts"] == 1
    assert (tmp_path / "cell/metrics/000-before.prom").exists()


def make_campaign(root: Path, *, low_acceptance=4.0):
    spec = read_spec(SPEC_PATH)
    spec["expected_prompts"] = 2
    inputs = root / "inputs"
    inputs.mkdir()
    ids = ["p1", "p2"]
    (inputs / "lock.json").write_text(json.dumps({"spec": spec, "workload": {"prompt_ids": ids}}))
    for block in range(3):
        block_dir = root / "phase1" / f"block-{block}"
        for arm in ("plain", "dflash"):
            for effort in ("low", "medium", "high"):
                cell = block_dir / arm / "cells" / effort
                cell.mkdir(parents=True)
                extra = dict.fromkeys(IDENTITY, "same")
                generation = {
                    key: spec[key]
                    for key in (
                        "temperature",
                        "top_p",
                        "top_k",
                        "min_p",
                        "max_tokens",
                    )
                }
                generation.update(reasoning_effort=effort, seed=block, ignore_eos=False)
                (cell / "metadata.json").write_text(
                    json.dumps(
                        {
                            "extra": extra,
                            "cell": {"generation": generation, "repeat": block, "concurrency": 1},
                            "workload_file_sha256": "corpus",
                            "prompt_ids": ids,
                        }
                    )
                )
                latency = 1000 if arm == "plain" else 500
                accepted = low_acceptance if effort == "low" else 4
                (cell / "summary.json").write_text(
                    json.dumps(
                        {
                            "validity": {"valid": True, "cap_fraction": 0},
                            "speculative_decoding": {
                                "drafts": 10,
                                "accepted_tokens": accepted * 10,
                                "acceptance_rate": accepted / 7,
                            },
                            "latency_ms": {"e2e": {"p50": latency}},
                            "tokens": {"output_tokens_per_second": 100},
                        }
                    )
                )
                (cell / "status.json").write_text('{"status":"complete"}')
                with (cell / "requests.jsonl").open("w") as sink:
                    for index, identifier in enumerate(ids):
                        sink.write(
                            json.dumps(
                                {
                                    "prompt_id": identifier,
                                    "workload": "math" if index == 0 else "code",
                                    "e2e_ms": latency,
                                    "ttft_ms": 100,
                                    "completion_tokens": 100,
                                }
                            )
                            + "\n"
                        )
        (block_dir / "status.json").write_text('{"status":"complete"}')
    return root


def test_gate_requires_consistent_effort_sensitivity_in_complete_paired_blocks(tmp_path):
    campaign = make_campaign(tmp_path, low_acceptance=5)
    report = build_report(campaign)
    assert report["decision"] == "proceed"
    assert report["sensitive_efforts"] == ["low"]
    assert len(report["rows"]) == 18
    assert all(
        row["latency_speedup_vs_plain"] == 2 for row in report["rows"] if row["variant"] == "dflash"
    )
    (campaign / "phase1/block-2/dflash/cells/high/status.json").unlink()
    assert build_report(campaign)["decision"] == "pending"


def test_gate_stops_with_stable_shared_head_and_refuses_unmatched_baseline(tmp_path):
    campaign = make_campaign(tmp_path)
    assert build_report(campaign)["decision"] == "stop"
    path = campaign / "phase1/block-0/dflash/cells/low/metadata.json"
    metadata = json.loads(path.read_text())
    metadata["extra"]["target_revision"] = "different-target"
    path.write_text(json.dumps(metadata))
    assert build_report(campaign)["decision"] == "invalid"


def test_training_cannot_start_from_an_inconclusive_or_stop_result(tmp_path):
    campaign = make_campaign(tmp_path)
    with pytest.raises(ValueError, match="disabled"):
        training_plan(
            campaign,
            tmp_path / "training",
            Path("missing-train"),
            Path("missing-dev"),
            100,
            Path("missing-optimizer"),
        )
