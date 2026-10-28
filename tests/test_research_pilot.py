import argparse
import json
import shutil
import subprocess
from pathlib import Path

import httpx
import pytest
import yaml

from apertus_bench import pilot_validation as validation
from apertus_bench import research_pilot as pilot
from apertus_bench.client import StreamingChatClient
from apertus_bench.runner import CellSettings, Variant, run_cell
from apertus_bench.workloads import Prompt, file_sha256

REPO = Path(__file__).resolve().parents[1]


@pytest.fixture
def prepared(tmp_path, monkeypatch):
    """Run the real launch scripts against a renderer and small pinned artifacts."""
    repo = tmp_path / "repo"
    repo.mkdir()
    for name in (
        "src",
        "serving",
        "methods/dspark/launch",
        "targets/8b",
        "experiments/serving-pilot",
    ):
        shutil.copytree(REPO / name, repo / name, ignore=shutil.ignore_patterns("__pycache__"))
    shutil.copy2(REPO / "pyproject.toml", repo / "pyproject.toml")
    target = tmp_path / "target"
    target.mkdir()
    mapping = {
        "config.json": "config_sha256",
        "tokenizer.json": "tokenizer_json_sha256",
        "tokenizer_config.json": "tokenizer_config_sha256",
        "chat_template.jinja": "chat_template_sha256",
        "generation_config.json": "generation_config_sha256",
        "special_tokens_map.json": "special_tokens_map_sha256",
        "model.safetensors.index.json": "weight_index_sha256",
    }
    contract = {"source": {"hf_id": "swiss-ai/Apertus-v1.5-8B", "revision": "a" * 40}}
    for name, key in mapping.items():
        (target / name).write_text("original template" if name.endswith("jinja") else "{}")
        contract["source"][key] = file_sha256(target / name)
    (target / "model.safetensors").write_bytes(b"target weights")
    contract["source"]["weight_manifest"] = {
        "files": [
            {
                "name": "model.safetensors",
                "bytes": 14,
                "sha256": file_sha256(target / "model.safetensors"),
            }
        ]
    }
    pilot.write_json(repo / "targets/8b/test-contract.json", contract)
    head = tmp_path / "head"
    head.mkdir()
    pilot.write_json(
        head / "config.json",
        {
            "speculators_model_type": "dspark",
            "block_size": 8,
            "sample_from_anchor": True,
            "aux_hidden_state_layer_ids": [1, 8, 15, 22, 29],
            "speculators_config": {"verifier": {"name_or_path": "/snapshots/" + "a" * 40}},
        },
    )
    (head / "model.safetensors").write_bytes(b"draft weights")
    image = tmp_path / "image.sqsh"
    image.write_bytes(b"pinned image")
    launcher = tmp_path / "launcher"
    (launcher / "src/swiss_ai_model_launch").mkdir(parents=True)
    (launcher / "src/swiss_ai_model_launch/__init__.py").write_text("")
    sml = launcher / "sml"
    sml.write_text("""#!/usr/bin/env python3
import json, pathlib, sys
args=sys.argv[1:]
out=pathlib.Path(args[args.index('--output-script')+1])
out.mkdir()
(out/'master.sh').write_text('#!/bin/bash\\ntrue\\n')
(out/'head.sh').write_text('#!/bin/bash\\nvllm serve '+args[args.index('--framework-args')+1]+'\\n')
(out/'arguments.json').write_text(json.dumps(args))
""")
    sml.chmod(0o755)
    env = tmp_path / "env.toml"
    env.write_text(f'image = "{image}"\nmounts = [\n]\n[env]\n')
    otela = tmp_path / "otela"
    otela.write_text("#!/bin/sh\necho sai-v0.0.7\n")
    otela.chmod(0o755)
    spec = yaml.safe_load((REPO / "experiments/serving-pilot/dspark-e10.yaml").read_text())
    spec["target"].update(
        path=str(target), revision="a" * 40, contract="targets/8b/test-contract.json"
    )
    spec["drafter"]["path"] = str(head)
    spec["runtime"].update(
        image=str(image),
        image_sha256=file_sha256(image),
        model_launch_root=str(launcher),
        sml_bin=str(sml),
        environment=str(env),
        opentela_binary=str(otela),
    )
    for key in ("probe", "workloads"):
        path = repo / spec["validation"][key]
        path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(repo / spec["validation"]["smoke"], path)
    profile = repo / "profile.yaml"
    profile.write_text(yaml.safe_dump(spec))
    monkeypatch.setattr(pilot, "REPO", repo)
    original_command = pilot.command

    def external(argv, **kwargs):
        if str(argv[1]).endswith("patch-vllm.sh"):
            overlay = Path(kwargs["env"]["VLLM_PATCH_DIR"])
            overlay.mkdir()
            (overlay / "patched.py").write_text("# fixed runtime overlay\n")
            return subprocess.CompletedProcess(
                argv, 0, f"{overlay}/patched.py:/workspace/fixed.py\n"
            )
        return original_command(argv, **kwargs)

    monkeypatch.setattr(pilot, "command", external)
    monkeypatch.setenv("CSCS_SERVING_API", "test-secret-never-persist")
    output = tmp_path / "run"
    args = argparse.Namespace(
        profile=profile,
        platform=None,
        variant="dspark",
        model_suffix="",
        partition=None,
        time=None,
        output=output,
    )
    pilot.prepare(args)
    return output, profile, args


def test_prepare_uses_real_launchers_and_freezes_inputs(prepared):
    run, _, args = prepared
    dep = pilot.read_json(run / "deployment.json")
    assert dep["model"].count("/") == 2
    renderer = pilot.read_json(run / "rendered/arguments.json")
    framework = renderer[renderer.index("--framework-args") + 1]
    assert "--speculative-config.method dspark" in framework
    assert "--speculative-config.num_speculative_tokens 3" in framework
    assert "--disable-log-requests" in framework
    assert "--dtype bfloat16" in framework
    assert "--dev" in renderer
    assert "python3 -m vllm.entrypoints.openai.api_server" in (run / "rendered/head.sh").read_text()
    assert "vllm serve" in (run / "rendered-entrypoint.patch").read_text()
    assert (run / "chat_template.jinja").read_text().startswith("{% set enable_thinking = false %}")
    environment = (run / "environment.toml").read_text()
    assert f'"{run}:{run}"' in environment
    assert (run / "source/workloads/8b/eagle-8b-test.jsonl").exists()
    pilot.verify_manifest(run, "source-manifest.json")
    pilot.verify_manifest(run, "launch-manifest.json")
    for path in run.rglob("*"):
        if path.is_file() and path.suffix in {".json", ".toml", ".log", ".sh", ".patch"}:
            assert "test-secret-never-persist" not in path.read_text()
    with pytest.raises(FileExistsError):
        pilot.prepare(args)


def test_profile_rejects_shared_ids_and_incompatible_heads(prepared):
    run, profile, _ = prepared
    spec = yaml.safe_load(profile.read_text())
    spec["deployment"]["model_id"] = spec["deployment"]["fallback_model_id"]
    profile.write_text(yaml.safe_dump(spec))
    with pytest.raises(ValueError, match="separate model IDs"):
        pilot.load_profile(profile)
    resolved = pilot.read_json(run / "profile.json")
    head = Path(resolved["drafter"]["path"])
    cfg = pilot.read_json(head / "config.json")
    cfg["speculators_config"]["verifier"]["name_or_path"] = "/wrong-revision"
    (head / "config.json").write_text(json.dumps(cfg))
    with pytest.raises(ValueError, match="verifier revision"):
        pilot.validate_drafter(resolved, full=False)


def test_submission_is_single_attempt_and_detects_tampering(prepared, monkeypatch):
    run, _, _ = prepared
    calls = []

    def sbatch(argv, **kwargs):
        calls.append(argv)
        return subprocess.CompletedProcess(argv, 0, "12345\n", "")

    monkeypatch.setattr(pilot.subprocess, "run", sbatch)
    assert pilot.submit(run)["job_id"] == "12345"
    assert pilot.submit(run)["job_id"] == "12345"
    assert len(calls) == 1
    (run / "job.json").unlink()
    with pytest.raises(ValueError, match="already attempted"):
        pilot.submit(run)
    (run / "submission-attempt.json").unlink()
    (run / "rendered/master.sh").write_text("#!/bin/bash\nwrong-job\n")
    with pytest.raises(ValueError, match="immutable run artifact changed"):
        pilot.submit(run)
    assert len(calls) == 1


def test_preflight_needs_allocation_and_detects_changed_target(prepared, monkeypatch):
    run, _, _ = prepared
    monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    with pytest.raises(ValueError, match="Slurm allocation"):
        pilot.preflight(run)
    spec = pilot.read_json(run / "profile.json")
    contract = pilot.read_json(run / "source/targets/8b/test-contract.json")
    pilot.validate_target(spec, contract, full=True)
    (Path(spec["target"]["path"]) / "model.safetensors").write_bytes(b"wrong weights!")
    with pytest.raises(ValueError, match="weight hash mismatch"):
        pilot.validate_target(spec, contract, full=True)


def test_research_launch_requires_matching_unchanged_evidence(prepared, tmp_path):
    run, _, args = prepared
    args.platform = "research"
    args.output = tmp_path / "research"
    with pytest.raises(ValueError, match="requires --qualification"):
        pilot.prepare(args)
    assert not args.output.exists()
    spec = pilot.read_json(run / "profile.json")
    evidence = tmp_path / "evidence.json"
    evidence.write_text('{"passed":true}\n')
    report = tmp_path / "qualified.json"
    pilot.write_json(
        report,
        {
            "passed": True,
            "status": "ready_for_pilot",
            "configuration_signature": pilot.configuration_signature(spec),
            "evidence_sha256": {str(evidence): file_sha256(evidence)},
        },
    )
    pilot.verify_qualification(report, spec)
    spec["drafter"]["k"] = 7
    with pytest.raises(ValueError, match="different deployment configuration"):
        pilot.verify_qualification(report, spec)
    spec["drafter"]["k"] = 3
    evidence.write_text('{"passed":false}\n')
    with pytest.raises(ValueError, match="evidence changed"):
        pilot.verify_qualification(report, spec)


def test_gate_needs_independent_matched_control_and_rejects_divergence(prepared, tmp_path):
    original, _, _ = prepared
    spec = pilot.read_json(original / "profile.json")
    checks = []
    for index, variant in enumerate(("plain", "dspark", "plain"), 1):
        run = tmp_path / f"comparison-{index}"
        run.mkdir()
        pilot.write_json(run / "profile.json", spec)
        pilot.write_json(run / "deployment.json", {"variant": variant, "model": str(index)})
        pilot.write_json(
            run / "preflight.json",
            {
                "status": "passed",
                "slurm_job_id": str(index),
                "target_files": {"config": "target"},
                "image_sha256": "image",
                "vllm_revision": "vllm",
                "chat_template_sha256": "template",
                "drafter": {"checkpoint_sha256": "head"} if variant == "dspark" else None,
            },
        )
        folder = run / "checks"
        folder.mkdir()
        pilot.write_json(
            folder / "report.json",
            {"passed": True, "run": str(run), "corpus_sha256": "same-prompts"},
        )
        for repeat in (1, 2):
            pilot.write_json(
                folder / f"capture-{repeat}.json",
                {
                    "model": str(index),
                    "base_url": f"http://engine-{index}",
                    "sampling": {"temperature": 0, "top_p": 1, "seed": 1},
                    "cases": [
                        {
                            "prompt_id": "one",
                            "workload": "smoke",
                            "measurement": {
                                "success": True,
                                "output": "A" * 40,
                                "completion_tokens": 10,
                            },
                        }
                    ],
                },
            )
        checks.append(folder)
    args = argparse.Namespace(baseline=checks[0], candidate=checks[1], control=checks[2])
    output = validation.gate(args)
    assert pilot.read_json(output / "correctness.json")["passed"]
    for repeat in (1, 2):
        capture = checks[1] / f"capture-{repeat}.json"
        data = pilot.read_json(capture)
        data["cases"][0]["measurement"]["output"] = "B" * 40
        capture.write_text(json.dumps(data))
    with pytest.raises(RuntimeError, match="correctness gate failed"):
        validation.gate(args)
    args.control = checks[0]
    with pytest.raises(ValueError, match="independently launched"):
        validation.gate(args)


@pytest.fixture
def fake_engine(prepared, monkeypatch):
    run, _, _ = prepared
    dep = pilot.read_json(run / "deployment.json")
    pilot.write_json(run / "preflight.json", {"status": "passed", "slurm_job_id": "12345"})
    counts = {"requests": 0, "metrics": True, "wrong_model": False, "headers": []}

    def handle(request):
        counts["headers"].append((request.url.host, request.headers.get("authorization")))
        if request.url.path == "/v1/models":
            return httpx.Response(200, json={"data": [{"id": dep["model"]}]})
        if request.url.path == "/health":
            return httpx.Response(200)
        if request.url.path == "/metrics":
            n = counts["requests"]
            body = ""
            if counts["metrics"]:
                labels = f'{{model_name="{dep["model"]}"}}'
                body = (
                    f"vllm:spec_decode_num_drafts_total{labels} {n}\n"
                    f"vllm:spec_decode_num_draft_tokens_total{labels} {3 * n}\n"
                    f"vllm:spec_decode_num_accepted_tokens_total{labels} {n}\n"
                )
            return httpx.Response(200, text=body)
        data = json.loads(request.content)
        counts["requests"] += 1
        model = "wrong" if counts["wrong_model"] else data["model"]
        usage = {"prompt_tokens": 10, "completion_tokens": 1}
        if data.get("stream"):
            lines = [
                {
                    "model": model,
                    "choices": [{"delta": {"content": "Hello"}, "finish_reason": None}],
                },
                {"model": model, "choices": [{"delta": {}, "finish_reason": "length"}]},
                {"model": model, "choices": [], "usage": usage},
            ]
            body = (
                "".join("data: " + json.dumps(line) + "\n\n" for line in lines) + "data: [DONE]\n\n"
            )
            return httpx.Response(200, text=body, headers={"content-type": "text/event-stream"})
        return httpx.Response(
            200,
            json={
                "model": model,
                "usage": usage,
                "choices": [
                    {
                        "message": {"role": "assistant", "content": "Hello"},
                        "finish_reason": "length",
                    }
                ],
            },
        )

    original = httpx.AsyncClient
    monkeypatch.setattr(
        httpx, "AsyncClient", lambda **kw: original(transport=httpx.MockTransport(handle), **kw)
    )
    args = argparse.Namespace(
        run=run,
        base_url="http://worker",
        gateway=False,
        api_key_env="CSCS_SERVING_API",
        metrics_url=None,
        limits=False,
    )
    return run, args, counts


async def test_smoke_requires_real_counters_and_uses_separate_auth(fake_engine):
    run, args, engine = fake_engine
    output = await validation.smoke(args)
    assert pilot.read_json(output / "report.json")["passed"]
    assert (output / "capture-1.json").exists()
    args.gateway = True
    args.metrics_url = "http://worker/metrics"
    output = await validation.smoke(args)
    assert pilot.read_json(output / "report.json")["authenticated"]
    assert all(token is None for host, token in engine["headers"] if host == "worker")
    assert any(token == "Bearer test-secret-never-persist" for _, token in engine["headers"])
    engine["metrics"] = False
    with pytest.raises(RuntimeError, match="smoke checks failed"):
        await validation.smoke(args)
    reports = list((run / "checks").glob("*/report.json"))
    assert any(not pilot.read_json(path)["passed"] for path in reports)


async def test_smoke_rejects_wrong_response_model(fake_engine):
    _, args, engine = fake_engine
    engine["wrong_model"] = True
    with pytest.raises(RuntimeError, match="smoke checks failed"):
        await validation.smoke(args)


async def test_runner_does_not_send_gateway_token_to_metrics(fake_engine, tmp_path):
    _, _, engine = fake_engine
    prompt = Prompt("one", "smoke", [{"role": "user", "content": "Hello"}], 1)
    corpus = tmp_path / "corpus.jsonl"
    corpus.write_text("{}\n")
    async with (
        StreamingChatClient(
            "https://gateway", "model", "private-key", timeout_seconds=10
        ) as client,
        StreamingChatClient("http://worker", "model", None, timeout_seconds=10) as metrics,
    ):
        await run_cell(
            client,
            [prompt],
            corpus,
            tmp_path / "cell",
            CellSettings("smoke", 1, 1, 0, 1),
            Variant("plain", "none"),
            "http://worker/metrics",
            "model",
            metrics_client=metrics,
        )
    assert all(token is None for host, token in engine["headers"] if host == "worker")
    assert ("gateway", "Bearer private-key") in engine["headers"]


def test_open_loop_keeps_credentials_out_and_rejects_dropped_iterations(fake_engine, monkeypatch):
    run, _, _ = fake_engine
    spec = pilot.read_json(run / "profile.json")
    corpus = run / "source" / spec["validation"]["workloads"]
    row = json.loads(corpus.read_text().splitlines()[0])
    row["workload"] = "chat"
    corpus.write_text(json.dumps(row) + "\n")
    script = run / "model-launch-source/loadtest/k6/script.js"
    script.parent.mkdir(parents=True)
    script.write_text("// snapshotted SML k6 script\n")
    monkeypatch.setattr(pilot, "inspect_run", lambda run: {"base_url": "http://worker:8080"})
    monkeypatch.setattr(
        pilot, "command", lambda *a, **kw: subprocess.CompletedProcess(a, 0, "abc123 file\n")
    )
    drops = {"count": 0}

    def k6(argv, **kwargs):
        assert kwargs["env"].get("CSCS_SERVING_API") is None
        config = json.loads(kwargs["env"]["RUN_CONFIG_JSON"])
        assert config["api_key"] == ""
        folder = Path(argv[argv.index("--summary-export") + 1]).parent
        pilot.write_json(
            folder / "summary.json",
            {
                "throughput": {
                    "total_requests": 360,
                    "success_rate": "100.0%",
                    "error_count": 0,
                }
            },
        )
        pilot.write_json(
            folder / "raw-summary.json",
            {
                "metrics": {
                    "dropped_iterations": {"count": drops["count"]},
                }
            },
        )
        assert "--jobid=12345" in argv
        return subprocess.CompletedProcess(argv, 0)

    monkeypatch.setattr(validation.subprocess, "run", k6)
    args = argparse.Namespace(run=run, base_url=None)
    folder = validation.open_loop(args)
    assert pilot.read_json(folder / "status.json")["passed"]
    drops["count"] = 1
    with pytest.raises(RuntimeError, match="arrival-rate check failed"):
        validation.open_loop(args)
