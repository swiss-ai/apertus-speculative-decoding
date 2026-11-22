"""Behavior checks, bounded measurements and release evidence for the pilot."""

from __future__ import annotations

import argparse
import asyncio
import math
import os
import shutil
import subprocess
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

import httpx

from apertus_bench.client import StreamingChatClient
from apertus_bench.correctness import (
    GateTolerance,
    capture_greedy_outputs,
    run_correctness_gate,
)
from apertus_bench.load import parse_load_sample
from apertus_bench.prometheus import parse_speculative_snapshot, speculative_delta
from apertus_bench.runner import CellSettings, GenerationSettings, Variant, run_cell
from apertus_bench.workloads import file_sha256, load_prompts


def helpers():
    from apertus_bench import research_pilot

    return research_pilot


def state(run: Path) -> tuple[dict, dict, dict]:
    h = helpers()
    spec, dep = h.read_json(run / "profile.json"), h.read_json(run / "deployment.json")
    preflight = h.read_json(run / "preflight.json")
    if preflight["status"] != "passed":
        raise ValueError("deployment has no passing runtime preflight")
    return spec, dep, preflight


def source_path(run: Path, spec: dict, key: str) -> Path:
    return run / "source" / spec["validation"][key]


def gateway_key(run: Path, env_name: str, platform: str) -> str:
    """Read credentials in memory using the SML environment's keyring backend."""
    explicit = os.environ.get(env_name)
    if explicit:
        return explicit
    if platform not in {"dev", "research"}:
        raise ValueError("unknown gateway credential platform")
    h = helpers()
    spec = h.read_json(run / "profile.json")
    python = Path(spec["runtime"]["sml_bin"]).parent / "python"
    getter = """import sys
try:
    if sys.argv[1] == "dev":
        import keyring
        value = keyring.get_password("apertus-speculative-decoding", "dev-api-key")
    else:
        from swiss_ai_model_launch.cli.configuration import InitConfig
        value = InitConfig.load().get_non_none_value("swissai_research_api_key")
except Exception:
    raise SystemExit(1)
if value:
    sys.stdout.write(value)
"""
    try:
        result = h.command(
            [str(python), "-c", getter, platform],
            capture_output=True,
            stdin=subprocess.DEVNULL,
            timeout=15,
        )
    except (OSError, subprocess.SubprocessError):
        raise ValueError(
            f"could not read the stored {platform} gateway credential; set {env_name}"
        ) from None
    if not result.stdout:
        location = (
            "keyring service apertus-speculative-decoding, username dev-api-key"
            if platform == "dev"
            else "SML swissai_research_api_key"
        )
        raise ValueError(f"no {platform} gateway key available; use {location} or set {env_name}")
    return result.stdout


def endpoint(args, dep: dict) -> tuple[str, str | None]:
    if args.gateway:
        base = dep["gateway"]
        key = gateway_key(args.run, args.api_key_env, dep["platform"])
    else:
        base = args.base_url
        key = None
        if not base:
            base = helpers().inspect_run(args.run).get("base_url")
        if not base:
            raise ValueError("replica address unavailable; supply --base-url")
    return base.rstrip("/").removesuffix("/v1"), key


def valid_message(data: dict, model: str) -> bool:
    choices, usage = data.get("choices") or [], data.get("usage") or {}
    message = choices[0].get("message", {}) if choices else {}
    text = message.get("content")
    return bool(
        data.get("model") == model
        and choices
        and isinstance(text, str)
        and text
        and choices[0].get("finish_reason") in {"stop", "length"}
        and all(
            isinstance(usage.get(k), int) and usage[k] > 0
            for k in ("prompt_tokens", "completion_tokens")
        )
        and not any(message.get(k) for k in ("reasoning", "reasoning_content"))
        and "<|inner_prefix|>" not in text
    )


async def limit_checks(
    client: httpx.AsyncClient,
    base: str,
    model: str,
    context: int,
    *,
    tokenize_path: str = "/tokenize",
) -> dict:
    """Use the engine tokenizer to avoid estimating a near-limit request by chars."""

    async def length(count: int) -> tuple[int, list[dict]]:
        messages = [{"role": "user", "content": "word " * count}]
        response = await client.post(
            base + tokenize_path,
            json={
                "model": model,
                "messages": messages,
                "chat_template_kwargs": {"enable_thinking": False},
                "add_generation_prompt": True,
            },
        )
        response.raise_for_status()
        return response.json()["count"], messages

    low, high, chosen = 1, context, None
    while low <= high:
        middle = (low + high) // 2
        count, messages = await length(middle)
        if count <= context - 64:
            chosen = (count, messages)
            low = middle + 1
        else:
            high = middle - 1
    if chosen is None or chosen[0] < context - 128:
        raise ValueError("could not construct a request close to the context limit")
    response = await client.post(
        base + "/v1/chat/completions",
        json={
            "model": model,
            "messages": chosen[1],
            "max_tokens": 1,
            "temperature": 0,
        },
    )
    near = response.status_code == 200 and valid_message(response.json(), model)
    too_long = await client.post(
        base + "/v1/chat/completions",
        json={
            "model": model,
            "messages": chosen[1],
            "max_tokens": context,
            "temperature": 0,
        },
    )
    return {
        "passed": near and too_long.status_code == 400,
        "tokenize_path": tokenize_path,
        "near_limit_prompt_tokens": chosen[0],
        "over_limit_status": too_long.status_code,
    }


async def smoke(args) -> Path:
    h = helpers()
    spec, dep, preflight = state(args.run)
    base, key = endpoint(args, dep)
    output = args.run / "checks" / h.stamp()
    output.mkdir(parents=True)
    report = {
        "at": h.now(),
        "run": str(args.run),
        "deployment_id": dep["deployment_id"],
        "model": dep["model"],
        "base_url": base,
        "gateway": args.gateway,
        "authenticated": bool(key),
        "checks": {},
        "limits_tested": args.limits,
        "corpus_sha256": file_sha256(source_path(args.run, spec, "smoke")),
    }
    prompts = load_prompts(source_path(args.run, spec, "smoke"))
    metrics_url = args.metrics_url or (None if args.gateway else base + "/metrics")
    try:
        # Do not send a public gateway bearer token to the separate metrics endpoint.
        async with httpx.AsyncClient(timeout=120) as metrics_client:
            before_raw = ""
            if metrics_url:
                response = await metrics_client.get(metrics_url)
                response.raise_for_status()
                before_raw = response.text
                (output / "metrics_before.prom").write_text(before_raw)
            async with StreamingChatClient(base, dep["model"], key, timeout_seconds=180) as client:
                models = await client.http.get(base + "/v1/models")
                models.raise_for_status()
                report["checks"]["discovery"] = any(
                    row.get("id") == dep["model"] for row in models.json().get("data", [])
                )
                if not args.gateway:
                    report["checks"]["health"] = (
                        await client.http.get(base + "/health")
                    ).status_code == 200
                for repeat in (1, 2):
                    capture = await capture_greedy_outputs(
                        client,
                        prompts,
                        output / f"capture-{repeat}.json",
                        max_tokens=spec["validation"]["correctness_max_tokens"],
                        seed=1,
                    )
                    report["checks"][f"streaming-{repeat}"] = all(
                        case["measurement"]["success"]
                        and case["measurement"].get("completion_tokens", 0)
                        and case["measurement"].get("prompt_tokens", 0)
                        and case["measurement"].get("finish_reason") in {"stop", "length"}
                        and "<|inner_prefix|>" not in (case["measurement"].get("output") or "")
                        for case in capture["cases"]
                    )
                # Include sampled multi-turn and multilingual behavior.
                for index, prompt in enumerate((prompts[1], prompts[-1])):
                    for temp in (0, spec["validation"]["temperature"]):
                        response = await client.http.post(
                            base + "/v1/chat/completions",
                            json={
                                "model": dep["model"],
                                "messages": prompt.messages,
                                "max_tokens": prompt.max_tokens,
                                "temperature": temp,
                                "top_p": spec["validation"]["top_p"] if temp else 1,
                                "chat_template_kwargs": {"enable_thinking": True},
                            },
                        )
                        report["checks"][f"nonstream-{index}-t{temp}"] = (
                            response.status_code == 200
                            and valid_message(response.json(), dep["model"])
                        )
                response = await client.http.post(
                    base + "/v1/chat/completions",
                    json={
                        "model": dep["model"],
                        "messages": prompts[0].messages,
                        "max_tokens": 1,
                        "temperature": 0,
                    },
                )
                report["checks"]["token-cap"] = (
                    response.status_code == 200
                    and valid_message(response.json(), dep["model"])
                    and response.json()["usage"]["completion_tokens"] <= 1
                )
                payload = {
                    "model": dep["model"],
                    "messages": prompts[2].messages,
                    "max_tokens": 128,
                    "stream": True,
                }
                cancelled = False
                async with client.http.stream(
                    "POST", base + "/v1/chat/completions", json=payload
                ) as res:
                    if res.status_code == 200:
                        async for line in res.aiter_lines():
                            if line.startswith("data: ") and line != "data: [DONE]":
                                cancelled = True
                                break
                response = await client.http.post(
                    base + "/v1/chat/completions",
                    json={
                        "model": dep["model"],
                        "messages": prompts[0].messages,
                        "max_tokens": 32,
                    },
                )
                report["checks"]["disconnect-and-recover"] = (
                    cancelled
                    and response.status_code == 200
                    and valid_message(response.json(), dep["model"])
                )
                if args.limits:
                    report["limits"] = await limit_checks(
                        client.http,
                        base,
                        dep["model"],
                        spec["engine"]["max_model_len"],
                        tokenize_path="/v1/tokenize" if args.gateway else "/tokenize",
                    )
                    report["checks"]["context-limits"] = report["limits"]["passed"]
            if metrics_url:
                response = await metrics_client.get(metrics_url)
                response.raise_for_status()
                (output / "metrics_after.prom").write_text(response.text)
                delta = speculative_delta(
                    parse_speculative_snapshot(before_raw, dep["model"]),
                    parse_speculative_snapshot(response.text, dep["model"]),
                )
                report["speculative_decoding"] = delta
                if dep["variant"] == "dspark":
                    report["checks"]["acceptance-counters"] = bool(
                        delta.get("enabled")
                        and delta.get("drafts", 0) > 0
                        and 0 <= delta.get("acceptance_rate", -1) <= 1
                    )
            elif dep["variant"] == "dspark":
                raise ValueError("DSpark smoke checks require a separate --metrics-url")
        report["passed"] = all(report["checks"].values())
    except Exception as exc:
        report["passed"] = False
        report["error"] = str(exc)
    report["runtime_job_id"] = preflight["slurm_job_id"]
    h.write_json(output / "report.json", report)
    print(output)
    if not report["passed"]:
        raise RuntimeError(f"pilot smoke checks failed; see {output / 'report.json'}")
    return output


async def measure(args) -> Path:
    if not os.environ.get("SLURM_JOB_ID"):
        raise ValueError("load generation must run inside a Slurm allocation")
    h = helpers()
    spec, dep, preflight = state(args.run)
    base, key = endpoint(args, dep)
    if args.gateway and not args.metrics_url:
        raise ValueError("gateway measurements need a separate --metrics-url")
    output = args.run / "measurements" / h.stamp()
    output.mkdir(parents=True)
    metadata = {
        "deployment_id": dep["deployment_id"],
        "target_model": spec["target"]["model"],
        "target_revision": spec["target"]["revision"],
        "target_tensor_parallel_size": "1",
        "slurm_job_id": preflight["slurm_job_id"],
        "slurm_node_id": preflight["node"],
        "image_sha256": preflight["image_sha256"],
        "path": "gateway" if args.gateway else "direct",
    }
    if dep["variant"] == "dspark":
        metadata["checkpoint_sha256"] = preflight["drafter"]["checkpoint_sha256"]
    variant = Variant(
        dep["variant"],
        "dspark" if dep["variant"] == "dspark" else "none",
        spec["drafter"]["k"] if dep["variant"] == "dspark" else None,
    )
    try:
        async with (
            StreamingChatClient(base, dep["model"], key, timeout_seconds=600) as client,
            StreamingChatClient(base, dep["model"], None, timeout_seconds=30) as metrics_client,
        ):
            for workload in ["probe", *spec["validation"]["workload_names"]]:
                path = source_path(args.run, spec, "probe" if workload == "probe" else "workloads")
                prompts = load_prompts(path, workload)
                prompts = [
                    replace(p, chat_template_kwargs={"enable_thinking": False}) for p in prompts
                ]
                concurrencies = [8] if workload == "probe" else spec["validation"]["concurrencies"]
                for c in concurrencies:
                    settings = CellSettings(
                        workload,
                        c,
                        max(len(prompts), math.ceil(4 * c / len(prompts)) * len(prompts)),
                        c,
                        1,
                        GenerationSettings(max_tokens=spec["validation"]["max_tokens"]),
                    )
                    summary = await run_cell(
                        client,
                        prompts,
                        path,
                        output / workload / f"c{c}",
                        settings,
                        variant,
                        args.metrics_url or base + "/metrics",
                        dep["model"],
                        metadata,
                        metrics_interval=1,
                        metrics_client=metrics_client,
                    )
                    if summary["requests"]["failed"]:
                        raise RuntimeError(f"failed requests in {workload} C={c}")
                    if dep["variant"] == "dspark" and not summary["speculative_decoding"].get(
                        "enabled"
                    ):
                        raise RuntimeError("speculative counters missing from measurement")
        h.write_json(
            output / "status.json",
            {
                "passed": True,
                "at": h.now(),
                "run": str(args.run),
                "gateway": args.gateway,
            },
        )
    except Exception as exc:
        h.write_json(output / "status.json", {"passed": False, "at": h.now(), "error": str(exc)})
        raise
    print(output)
    return output


def evidence_signature(run: Path) -> str:
    h = helpers()
    spec, dep, runtime = state(run)
    return h.digest(
        {
            "target": runtime["target_files"],
            "image": runtime["image_sha256"],
            "vllm": runtime["vllm_revision"],
            "template": runtime["chat_template_sha256"],
            "overlays": h.overlay_hashes(run),
            "variant": dep["variant"],
            "engine": spec["engine"],
            "head": runtime["drafter"]["checkpoint_sha256"] if runtime["drafter"] else None,
            "k": spec["drafter"]["k"] if dep["variant"] == "dspark" else None,
        }
    )


def open_loop(args) -> Path:
    """Run SML's existing k6 script inside the server's own allocation."""
    h = helpers()
    spec, dep, runtime = state(args.run)
    info = h.inspect_run(args.run)
    base = args.base_url or info.get("base_url")
    if not base or not base.startswith("http://"):
        raise ValueError("open-loop check needs the direct worker HTTP address")
    cfg = spec["validation"]["open_loop"]
    output = args.run / "open-loop" / h.stamp()
    output.mkdir(parents=True)
    corpus = [
        dict(messages=p.messages, max_tokens=p.max_tokens, label=p.workload)
        for p in load_prompts(source_path(args.run, spec, "workloads"))
        if p.workload in spec["validation"]["workload_names"]
    ]
    if not corpus:
        raise ValueError("no prompts for the arrival-rate check")
    h.write_json(output / "prompts.json", corpus)
    config = {
        "server_url": base.rstrip("/"),
        "model": dep["model"],
        "api_key": "",
        "scenario": "pilot-open-loop",
        "prompt_seed": 1,
        "ignore_eos": False,
        "max_tokens": cfg["max_tokens"],
        "think_time": 0,
        "request_timeout": "120s",
        "scenario_definition": {
            "executor": "constant-arrival-rate",
            "rate": cfg["rate"],
            "timeUnit": "1s",
            "duration": f"{cfg['duration_seconds']}s",
            "preAllocatedVUs": cfg["preallocated_vus"],
            "maxVUs": cfg["max_vus"],
        },
    }
    h.write_json(output / "run_config.json", config)
    # No API credentials enter the k6 configuration or its environment.
    env = h.job_environment()
    env.update(RUN_CONFIG_JSON=h.json.dumps(config), PROMPTS_FILE=str(output / "prompts.json"))
    script = args.run / "model-launch-source/loadtest/k6/script.js"
    prefix = ["srun", "--overlap", f"--jobid={runtime['slurm_job_id']}", "--nodes=1", "--ntasks=1"]
    image_hash = h.command(
        prefix + ["sha256sum", cfg["image"]], capture_output=True
    ).stdout.split()[0]
    argv = prefix + [
        f"--container-image={cfg['image']}",
        f"--container-mounts={args.run}:{args.run}",
        f"--container-workdir={output}",
        "--container-env=RUN_CONFIG_JSON,PROMPTS_FILE",
        "k6",
        "run",
        "--summary-export",
        str(output / "raw-summary.json"),
        "--out",
        f"json={output / 'raw.jsonl'}",
        "--tag",
        f"model={dep['model']}",
        "--tag",
        f"run_label={output.name}",
        str(script),
    ]
    h.write_json(
        output / "provenance.json",
        {
            "argv": argv,
            "run": str(args.run),
            "k6_image_sha256": image_hash,
            "script_sha256": file_sha256(script),
            "corpus_sha256": file_sha256(output / "prompts.json"),
            "sampling": {"temperature": 1.0, "stream": False},
            "config": config,
        },
    )
    with (output / "k6.log").open("x") as log:
        result = subprocess.run(argv, env=env, stdout=log, stderr=subprocess.STDOUT, check=False)
    summary = h.read_json(output / "summary.json") if (output / "summary.json").exists() else {}
    raw = h.read_json(output / "raw-summary.json") if (output / "raw-summary.json").exists() else {}
    dropped_metric = raw.get("metrics", {}).get("dropped_iterations", {})
    dropped = dropped_metric.get("values", dropped_metric).get("count", 0)
    throughput = summary.get("throughput", {})
    passed = (
        result.returncode == 0
        and throughput.get("success_rate") == "100.0%"
        and throughput.get("error_count", -1) == 0
        and dropped == 0
        and throughput.get("total_requests", 0) >= cfg["rate"] * cfg["duration_seconds"]
    )
    h.write_json(
        output / "status.json",
        {
            "passed": passed,
            "run": str(args.run),
            "returncode": result.returncode,
            "dropped_iterations": dropped,
            "at": h.now(),
        },
    )
    print(output)
    if not passed:
        raise RuntimeError("arrival-rate check failed or dropped iterations; inspect k6 artifacts")
    return output


def gate(args) -> Path:
    h = helpers()
    dirs = [args.baseline.resolve(), args.candidate.resolve(), args.control.resolve()]
    reports = [h.read_json(p / "report.json") for p in dirs]
    if not all(r["passed"] for r in reports):
        raise ValueError("all smoke checks must pass before comparing correctness")
    runs = [Path(r["run"]) for r in reports]
    if len({state(r)[2]["slurm_job_id"] for r in runs}) != 3:
        raise ValueError(
            "baseline, candidate and control must be independently launched deployments"
        )
    baseline, candidate, control = runs
    if state(baseline)[1]["variant"] != "plain" or state(candidate)[1]["variant"] != "dspark":
        raise ValueError("treatment must compare plain against DSpark")
    if evidence_signature(baseline) != evidence_signature(control):
        raise ValueError("calibration control is not the same configuration as the plain baseline")
    if state(baseline)[2]["target_files"] != state(candidate)[2]["target_files"]:
        raise ValueError("treatment target assets differ")
    for key in ("image_sha256", "vllm_revision", "chat_template_sha256"):
        if state(baseline)[2][key] != state(candidate)[2][key]:
            raise ValueError(f"treatment runtime differs in {key}")
    if state(baseline)[0]["engine"] != state(candidate)[0]["engine"] or h.overlay_hashes(
        baseline
    ) != h.overlay_hashes(candidate):
        raise ValueError("treatment engine settings or source overlays differ")
    if len({r["corpus_sha256"] for r in reports}) != 1:
        raise ValueError("correctness corpora differ")
    spec = state(candidate)[0]
    for run in runs:
        if (
            state(run)[0]["validation"]["correctness_max_tokens"]
            != spec["validation"]["correctness_max_tokens"]
        ):
            raise ValueError("correctness output caps differ")
    output = candidate / "gates" / h.stamp()
    output.mkdir(parents=True)
    result = run_correctness_gate(
        (dirs[0] / "capture-1.json", dirs[1] / "capture-1.json"),
        [(p / "capture-1.json", p / "capture-2.json") for p in dirs]
        + [(dirs[0] / "capture-1.json", dirs[2] / "capture-1.json")],
        output / "correctness.json",
        GateTolerance(**spec["validation"]["tolerance"]),
    )
    h.write_json(
        output / "inputs.json",
        {"checks": [str(p) for p in dirs], "signatures": [evidence_signature(r) for r in runs]},
    )
    print(output)
    if not result["passed"]:
        raise RuntimeError("calibrated correctness gate failed")
    return output


async def soak(args) -> Path:
    h = helpers()
    spec, dep, _ = state(args.run)
    seconds = spec["validation"]["soak_seconds"] if args.seconds is None else args.seconds
    if seconds <= 0:
        raise ValueError("observation duration must be positive")
    base, key = endpoint(args, dep)
    if not args.metrics_url:
        raise ValueError("soak requires the engine's --metrics-url")
    output = args.run / "soaks" / h.stamp()
    output.mkdir(parents=True)
    prompt = load_prompts(source_path(args.run, spec, "smoke"))[0]
    started, samples, before = time.monotonic(), [], None
    try:
        async with (
            StreamingChatClient(base, dep["model"], key, timeout_seconds=120) as client,
            httpx.AsyncClient(timeout=15) as metrics,
        ):
            while time.monotonic() - started < seconds:
                result = await client.request(
                    prompt,
                    len(samples),
                    temperature=0,
                    top_p=1,
                    seed=1,
                    max_tokens=32,
                    ignore_eos=False,
                    store_output=False,
                )
                response = await metrics.get(args.metrics_url)
                response.raise_for_status()
                raw = response.text
                current = parse_speculative_snapshot(raw, dep["model"])
                if before is None:
                    before = current
                delta = speculative_delta(before, current)
                load = parse_load_sample(raw, dep["model"])
                samples.append(
                    {
                        "at": h.now(),
                        "success": result.success,
                        "e2e_ms": result.e2e_ms,
                        "load": load,
                    }
                )
                (output / f"metrics-{len(samples):04d}.prom").write_text(raw)
                if not result.success or (dep["variant"] == "dspark" and not delta.get("enabled")):
                    raise RuntimeError("request or speculative counter failure during soak")
                remaining = seconds - (time.monotonic() - started)
                if remaining > 0:
                    await asyncio.sleep(min(30, remaining))
        queues = [s["load"].get("waiting") for s in samples[-4:]]
        growing = (
            len(queues) == 4
            and all(q is not None for q in queues)
            and all(right > left for left, right in zip(queues, queues[1:], strict=False))
        )
        report = {
            "passed": not growing,
            "seconds": time.monotonic() - started,
            "samples": samples,
            "run": str(args.run),
            "at": h.now(),
            "gateway": args.gateway,
        }
    except Exception as exc:
        report = {
            "passed": False,
            "seconds": time.monotonic() - started,
            "samples": samples,
            "run": str(args.run),
            "error": str(exc),
            "at": h.now(),
        }
    h.write_json(output / "report.json", report)
    print(output)
    if not report["passed"]:
        raise RuntimeError("pilot soak failed")
    return output


def qualify(args) -> Path:
    """Require saved behavior, correctness, performance and fallback evidence."""
    h = helpers()
    draft_check = h.read_json(args.candidate_checks / "report.json")
    plain_check = h.read_json(args.fallback_checks / "report.json")
    candidate, baseline = Path(draft_check["run"]), Path(plain_check["run"])
    spec, dep, runtime = state(candidate)
    correctness = h.read_json(args.gate / "correctness.json")
    gate_inputs = h.read_json(args.gate / "inputs.json")
    inputs = [
        args.candidate_checks / "report.json",
        args.fallback_checks / "report.json",
        args.gate / "correctness.json",
        args.gate / "inputs.json",
    ]
    checks = {
        "candidate_gateway": bool(
            draft_check["passed"] and draft_check["gateway"] and draft_check["authenticated"]
        ),
        "context_limits": bool(draft_check.get("limits", {}).get("passed")),
        "fallback_gateway": bool(
            plain_check["passed"] and plain_check["gateway"] and plain_check["authenticated"]
        ),
        "fallback_identity": plain_check["model"] == dep["fallback_model"],
        "matched_target": state(baseline)[2]["target_files"] == runtime["target_files"],
        "matched_runtime": all(
            state(baseline)[2][key] == runtime[key]
            for key in ("image_sha256", "vllm_revision", "chat_template_sha256")
        )
        and h.overlay_hashes(baseline) == h.overlay_hashes(candidate)
        and state(baseline)[0]["engine"] == spec["engine"],
        "correctness": bool(
            correctness["passed"] and gate_inputs["signatures"][1] == evidence_signature(candidate)
        ),
    }
    arrival = []
    for folder, run in ((args.baseline_open_loop, baseline), (args.candidate_open_loop, candidate)):
        record = h.read_json(folder / "status.json")
        if not record["passed"] or Path(record["run"]) != run:
            raise ValueError("arrival-rate test failed or belongs to another deployment")
        arrival.append(h.read_json(folder / "provenance.json"))
        inputs.extend(
            folder / name
            for name in (
                "status.json",
                "provenance.json",
                "summary.json",
                "raw-summary.json",
                "raw.jsonl",
                "prompts.json",
            )
        )
    checks["matched_arrival_test"] = all(
        arrival[0][key] == arrival[1][key]
        for key in ("k6_image_sha256", "script_sha256", "corpus_sha256", "sampling")
    ) and all(
        arrival[0]["config"][key] == arrival[1]["config"][key]
        for key in (
            "scenario_definition",
            "max_tokens",
            "ignore_eos",
            "request_timeout",
            "prompt_seed",
            "think_time",
        )
    )
    roots = [(args.baseline_measurements, baseline), (args.candidate_measurements, candidate)]
    for root, run in roots:
        status_path = root / "status.json"
        status = h.read_json(status_path)
        if not status["passed"] or Path(status["run"]) != run:
            raise ValueError("measurement status does not match the checked deployment")
        inputs.append(status_path)
    rows = []
    expected = [("probe", 8)] + [
        (w, c)
        for w in spec["validation"]["workload_names"]
        for c in spec["validation"]["concurrencies"]
    ]
    for workload, concurrency in expected:
        cells = []
        for root, run in roots:
            cell = root / workload / f"c{concurrency}"
            metadata, summary = (
                h.read_json(cell / "metadata.json"),
                h.read_json(cell / "summary.json"),
            )
            if metadata["extra"]["deployment_id"] != state(run)[1]["deployment_id"]:
                raise ValueError("measurement provenance differs from the checked deployment")
            if summary["requests"]["failed"]:
                raise ValueError("failed measurement requests block qualification")
            cells.append((metadata, summary))
            inputs.extend(
                cell / name
                for name in (
                    "metadata.json",
                    "summary.json",
                    "requests.jsonl",
                    "metrics_before.prom",
                    "metrics_after.prom",
                )
            )
        left, right = cells
        for key in ("workload_file_sha256", "prompt_ids", "cell"):
            if left[0][key] != right[0][key]:
                raise ValueError(f"measurement cells differ in {key}")
        if left[0]["extra"]["path"] != right[0]["extra"]["path"]:
            raise ValueError("cannot compare a gateway cell with a direct cell")
        acceptance = right[1]["speculative_decoding"]
        checks[f"acceptance-{workload}-{concurrency}"] = bool(
            acceptance.get("enabled") and acceptance.get("drafts", 0) > 0
        )
        plain_p95, draft_p95 = (
            left[1]["latency_ms"]["e2e"]["p95"],
            right[1]["latency_ms"]["e2e"]["p95"],
        )
        ratio = draft_p95 / plain_p95 if plain_p95 else None
        checks[f"p95-{workload}-{concurrency}"] = bool(
            ratio is not None and ratio <= 1 + spec["validation"]["max_p95_regression"]
        )
        speedup = (
            right[1]["tokens"]["output_tokens_per_second"]
            / left[1]["tokens"]["output_tokens_per_second"]
        )
        rows.append(
            {
                "workload": workload,
                "concurrency": concurrency,
                "throughput_speedup": speedup,
                "p95_ratio": ratio,
            }
        )
    checks["useful_speedup"] = any(row["throughput_speedup"] > 1.05 for row in rows)
    stable = False
    if args.soak:
        observation = h.read_json(args.soak / "report.json")
        stable = bool(
            observation["passed"]
            and Path(observation["run"]) == candidate
            and observation["seconds"] >= spec["validation"]["soak_seconds"]
        )
        checks["observation"] = stable
        inputs.append(args.soak / "report.json")
    for folder in gate_inputs["checks"]:
        inputs.extend(
            Path(folder) / name for name in ("report.json", "capture-1.json", "capture-2.json")
        )
    result = {
        "at": h.now(),
        "passed": all(checks.values()),
        "status": ("stable" if stable else "ready_for_pilot") if all(checks.values()) else "failed",
        "run": str(candidate),
        "checks": checks,
        "measurements": rows,
        "configuration_signature": h.configuration_signature(spec),
        "target_files": runtime["target_files"],
        "checkpoint_sha256": runtime["drafter"]["checkpoint_sha256"],
        "overlay_sha256": h.overlay_hashes(candidate),
        "evidence_sha256": {str(p.resolve()): file_sha256(p) for p in inputs},
    }
    output = candidate / "qualifications" / h.stamp()
    output.mkdir(parents=True)
    h.write_json(output / "report.json", result)
    print(output)
    if not result["passed"]:
        raise RuntimeError("pilot qualification failed; inspect the saved checks")
    return output


async def validate_when_ready(args) -> Path:
    if args.wait_seconds <= 0:
        raise ValueError("validation wait must be positive")
    h = helpers()
    if not (args.run / "job.json").exists():
        raise ValueError("submit the deployment before starting validation")
    with (args.run / ".validate.lock").open("a") as lock:
        try:
            h.fcntl.flock(lock, h.fcntl.LOCK_EX | h.fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise ValueError("another validation controller already owns this deployment") from exc
        return await _validate_when_ready(args)


async def _validate_when_ready(args) -> Path:
    """Lightweight login-node controller; load generation stays in the allocation."""
    h = helpers()
    output = args.run / "automation" / h.stamp()
    output.mkdir(parents=True)
    for name in ("research_pilot.py", "pilot_validation.py"):
        shutil.copy2(Path(__file__).parent / name, output / name)
    h.write_json(
        output / "controller.json",
        {
            "created_at": h.now(),
            "repository": h.git_info(h.REPO),
            "wait_seconds": args.wait_seconds,
            "smoke_only": args.smoke_only,
            "stop_after": args.stop_after,
            "gateway": not args.no_gateway,
            "source_sha256": {
                name: file_sha256(output / name)
                for name in ("research_pilot.py", "pilot_validation.py")
            },
        },
    )
    report = {"run": str(args.run), "started_at": h.now(), "passed": False}
    try:
        deadline = time.monotonic() + args.wait_seconds
        base = None
        async with httpx.AsyncClient(timeout=5) as client:
            while time.monotonic() < deadline:
                info = h.inspect_run(args.run)
                for failure in ("preflight-failed", "preparation-failed"):
                    if failure in info:
                        raise RuntimeError(f"deployment {failure}: {info[failure]}")
                row = info.get("slurm", "").split("|")
                if len(row) > 1 and row[1].split()[0] in {
                    "FAILED",
                    "CANCELLED",
                    "TIMEOUT",
                    "OUT_OF_MEMORY",
                    "NODE_FAIL",
                    "COMPLETED",
                }:
                    raise RuntimeError(f"deployment ended before validation: {row[1]}")
                base = info.get("base_url")
                if base and info.get("preflight", {}).get("status") == "passed":
                    try:
                        if (await client.get(base + "/health")).status_code == 200:
                            break
                    except httpx.HTTPError:
                        pass
                base = None
                await asyncio.sleep(min(30, max(0, deadline - time.monotonic())))
        if not base:
            raise TimeoutError("pilot did not become ready within the validation wait")
        options = argparse.Namespace(
            run=args.run,
            base_url=base,
            gateway=False,
            api_key_env="CSCS_SERVING_API",
            metrics_url=base + "/metrics",
            limits=True,
        )
        report["direct_checks"] = str(await smoke(options))
        if not args.no_gateway:
            options.gateway = True
            report["gateway_checks"] = str(await smoke(options))
        if not args.smoke_only:
            _, _, runtime = state(args.run)
            # The original environment retains the vLLM overlay path; the client
            # only needs the frozen benchmark source in addition to that path.
            inside = (
                'export PYTHONPATH="$1/source/src:${PYTHONPATH:-}"; '
                'exec python3 -m apertus_bench.research_pilot measure "$1" --base-url "$2"'
            )
            argv = [
                "srun",
                "--overlap",
                f"--jobid={runtime['slurm_job_id']}",
                "--nodes=1",
                "--ntasks=1",
                f"--environment={args.run / 'environment.toml'}",
                "bash",
                "-c",
                inside,
                "pilot-client",
                str(args.run),
                base,
            ]
            h.write_json(output / "measurement-command.json", {"argv": argv})
            with (output / "measure.log").open("x") as log:
                h.command(argv, stdout=log, stderr=subprocess.STDOUT, env=h.job_environment())
            report["measurements"] = str(sorted((args.run / "measurements").iterdir())[-1])
            report["open_loop"] = str(open_loop(argparse.Namespace(run=args.run, base_url=base)))
        report["passed"] = True
    except Exception as exc:
        report["error"] = str(exc)
    finally:
        report["ended_at"] = h.now()
        h.write_json(output / "report.json", report)
        if args.stop_after:
            h.stop(args.run)
        else:
            h.collect(args.run)
    print(output)
    if not report["passed"]:
        raise RuntimeError(f"automated validation failed; see {output / 'report.json'}")
    return output


def add_parsers(sub) -> None:
    for name, help_text in (
        ("smoke", "check requests and retain correctness captures"),
        ("measure", "run the bounded probe and concurrency cells"),
        ("soak", "observe a deployment without retaining live content"),
    ):
        cmd = sub.add_parser(name, help=help_text)
        cmd.add_argument("run", type=Path)
        cmd.add_argument("--base-url")
        cmd.add_argument("--gateway", action="store_true")
        cmd.add_argument("--api-key-env", default="CSCS_SERVING_API")
        cmd.add_argument("--metrics-url")
        if name == "smoke":
            cmd.add_argument("--limits", action="store_true")
        elif name == "soak":
            cmd.add_argument("--seconds", type=int)
    cmd = sub.add_parser(
        "gate", help="calibrate greedy divergence against a separate plain control"
    )
    cmd.add_argument("--baseline", type=Path, required=True, help="baseline checks directory")
    cmd.add_argument("--candidate", type=Path, required=True, help="DSpark checks directory")
    cmd.add_argument(
        "--control", type=Path, required=True, help="independent plain checks directory"
    )
    cmd = sub.add_parser("qualify", help="check saved behavior, performance and fallback evidence")
    for name in (
        "candidate-checks",
        "fallback-checks",
        "baseline-measurements",
        "candidate-measurements",
        "gate",
        "baseline-open-loop",
        "candidate-open-loop",
    ):
        cmd.add_argument("--" + name, type=Path, required=True)
    cmd.add_argument("--soak", type=Path)
    cmd = sub.add_parser("open-loop", help="run SML k6 at a matched fixed arrival rate")
    cmd.add_argument("run", type=Path)
    cmd.add_argument("--base-url")
    cmd = sub.add_parser("validate", help="wait for readiness, then run smoke, cells and k6")
    cmd.add_argument("run", type=Path)
    cmd.add_argument("--wait-seconds", type=int, default=10800)
    cmd.add_argument("--smoke-only", action="store_true")
    cmd.add_argument("--no-gateway", action="store_true")
    cmd.add_argument("--stop-after", action="store_true")


def dispatch(args) -> Any:
    return {
        "smoke": smoke,
        "measure": measure,
        "gate": gate,
        "soak": soak,
        "qualify": qualify,
        "open-loop": open_loop,
        "validate": validate_when_ready,
    }[args.action](args)
