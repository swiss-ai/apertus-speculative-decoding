import hashlib
import importlib.util
import json
import tarfile
from pathlib import Path

import httpx
import pytest

from apertus_bench.client import StreamingChatClient
from apertus_bench.workloads import Prompt

MODULE = Path(__file__).resolve().parents[1] / "experiments/serving-pilot/run-usage-demo.py"
spec = importlib.util.spec_from_file_location("usage_demo", MODULE)
demo = importlib.util.module_from_spec(spec)
spec.loader.exec_module(demo)


@pytest.mark.asyncio
async def test_demo_retains_real_streamed_answers_and_never_sends_gateway_key_to_metrics(tmp_path):
    seen = []
    metrics_count = 0
    model = "owner/org/candidate"

    def respond(request):
        nonlocal metrics_count
        seen.append((request.url.path, request.headers.get("authorization")))
        if request.url.path == "/metrics":
            metrics_count += 1
            values = [(1, 3, 1), (3, 9, 5)][metrics_count - 1]
            names = ["drafts", "draft_tokens", "accepted_tokens"]
            text = (
                "\n".join(
                    f'vllm:spec_decode_num_{name}_total{{model_name="{model}"}} {value}'
                    for name, value in zip(names, values, strict=True)
                )
                + "\n"
            )
            return httpx.Response(200, text=text)
        body = json.loads(request.content)
        assert body["stream"] and body["chat_template_kwargs"]["enable_thinking"] is False
        events = [
            {"choices": [{"delta": {"content": '{"answer":42}'}}]},
            {
                "choices": [{"delta": {}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 12, "completion_tokens": 7},
            },
        ]
        stream = "".join("data: " + json.dumps(e) + "\n\n" for e in events) + "data: [DONE]\n\n"
        return httpx.Response(200, text=stream, headers={"content-type": "text/event-stream"})

    transport = httpx.MockTransport(respond)
    client = StreamingChatClient("https://gateway", model, "synthetic-test-key", timeout_seconds=30)
    await client.http.aclose()
    client.http = httpx.AsyncClient(
        transport=transport, headers={"Authorization": "Bearer synthetic-test-key"}
    )
    prompt = Prompt(
        "extract",
        "extraction",
        [{"role": "user", "content": "JSON please"}],
        64,
        {"enable_thinking": False},
    )
    case = {
        "id": "extract",
        "messages": prompt.messages,
        "review": "Exact extraction",
        "expected_json": {"answer": 42},
    }
    async with client, httpx.AsyncClient(transport=transport) as metrics:
        row = await demo.capture_cell(
            client,
            metrics,
            "http://worker/metrics",
            prompt,
            case,
            tmp_path,
            "dspark",
            {"corpus_sha256": "fixture"},
        )
    assert row["measurement"]["final_output"] == '{"answer":42}'
    assert all(row["checks"].values())
    assert row["speculative_decoding"]["acceptance_rate"] == pytest.approx(4 / 6)
    assert all(auth is None for path, auth in seen if path == "/metrics")
    assert all(auth == "Bearer synthetic-test-key" for path, auth in seen if path != "/metrics")
    cell = tmp_path / "cells/extract/dspark"
    saved = json.loads((cell / "requests.jsonl").read_text())
    assert saved["output"] == '{"answer":42}' and saved["stream_events"]
    assert (cell / "metrics_before.prom").exists() and (cell / "metrics_after.prom").exists()
    assert "synthetic-test-key" not in "".join(p.read_text() for p in cell.iterdir())
    report = demo.readable_report([case], [row], model, "owner/org/plain")
    assert '{"answer":42}' in report and "not a production sample" in report
    assert "Exact extraction" in report


def test_demo_flags_truncation_and_malformed_json():
    checks = demo.objective_checks(
        {"expected_json": {"answer": 42}},
        {
            "success": True,
            "finish_reason": "length",
            "final_output": "```json\n{}\n```",
            "reasoning_output": "unexpected reasoning",
        },
    )
    assert checks["transport_success"]
    assert not checks["natural_finish"]
    assert not checks["exact_json_extraction"]
    assert not checks["no_reasoning_output"]


def test_shipped_cases_have_bounded_caps_and_support_multiturn(tmp_path):
    cases = demo.load_cases(MODULE.with_name("usage-prompts.yaml"))
    assert len(cases) == 10
    assert any(len(c["messages"]) > 1 for c in cases)
    assert all(c["chat_template_kwargs"] == {"enable_thinking": False} for c in cases)


@pytest.mark.asyncio
async def test_demo_never_requests_a_completion_before_startup_validation_passes(
    tmp_path, monkeypatch
):
    run = tmp_path / "deployment"
    reports = run / "automation/controller"
    reports.mkdir(parents=True)
    (reports / "report.json").write_text(
        json.dumps({"passed": True, "gateway_checks": "saved-checks"})
    )
    snapshots = iter(
        [
            {"slurm": "123|PENDING|None"},
            {
                "slurm": "123|RUNNING|node",
                "base_url": "http://worker",
                "preflight": {"status": "passed"},
            },
        ]
    )
    monkeypatch.setattr(demo.h, "inspect_run", lambda _: next(snapshots))
    sleeps = []

    async def sleep(seconds):
        sleeps.append(seconds)

    monkeypatch.setattr(demo.asyncio, "sleep", sleep)
    info = await demo.wait_ready(run, 60)
    assert info["base_url"] == "http://worker" and len(sleeps) == 1


@pytest.mark.asyncio
async def test_explicit_direct_route_requires_passing_same_job_checks_and_live_health(
    tmp_path, monkeypatch
):
    run = tmp_path / "deployment"
    (run / "checks/direct").mkdir(parents=True)
    (run / "automation/failed").mkdir(parents=True)
    (run / "checks/direct/report.json").write_text(
        json.dumps({"passed": True, "gateway": False, "runtime_job_id": "123"})
    )
    (run / "automation/failed/report.json").write_text(
        json.dumps({"passed": False, "error": "gateway not discoverable"})
    )
    monkeypatch.setattr(
        demo.h,
        "inspect_run",
        lambda _: {
            "slurm": "123|RUNNING|node",
            "base_url": "http://worker",
            "preflight": {"status": "passed", "slurm_job_id": "123"},
        },
    )
    original = httpx.AsyncClient

    def handler(request):
        assert request.url.path == "/health"
        assert not request.headers.get("authorization")
        return httpx.Response(200)

    monkeypatch.setattr(
        demo.httpx,
        "AsyncClient",
        lambda **kwargs: original(transport=httpx.MockTransport(handler), **kwargs),
    )
    with pytest.raises(RuntimeError, match="startup validation failed"):
        await demo.wait_ready(run, 30)
    info = await demo.wait_ready(run, 30, direct=True)
    assert "direct worker" in info["demo_route"]
    assert info["startup_validation_report"].endswith("checks/direct/report.json")


def test_collector_verifies_results_and_preserves_existing_local_files(tmp_path):
    path = MODULE.with_name("collect-usage-demo.py")
    loader = importlib.util.spec_from_file_location("usage_collector", path)
    collector = importlib.util.module_from_spec(loader)
    loader.loader.exec_module(collector)
    source = tmp_path / "source/demo"
    source.mkdir(parents=True)
    report = source / "report.md"
    report.write_text("actual saved answer\n")
    (source / "artifact-manifest.json").write_text(
        json.dumps({"report.md": hashlib.sha256(report.read_bytes()).hexdigest()})
    )
    archive = tmp_path / "demo.tar.gz"
    with tarfile.open(archive, "w:gz") as sink:
        sink.add(source, arcname="demo")
    output = tmp_path / "local/demo"
    output.mkdir(parents=True)
    (output / "README.md").write_text("Waiting for startup\n")
    collector.install_archive(archive, output)
    assert (output / "report.md").read_text() == "actual saved answer\n"
    assert "completed" in (output / "README.md").read_text()
    with pytest.raises(FileExistsError):
        collector.install_archive(archive, output)
    report.write_text("altered\n")
    with tarfile.open(archive, "w:gz") as sink:
        sink.add(source, arcname="demo")
    with pytest.raises(ValueError, match="hash mismatch"):
        collector.install_archive(archive, tmp_path / "other/demo")
