"""Capture a small interactive usage demo on an existing, validated deployment.

This is a bounded HTTP client: no model launches, benchmark load, or cancellations.
It waits for the existing smoke controller, then sends one streamed request at a
time. Each raw cell is immutable. Credentials are read in memory via the pilot's
existing resolver and never sent to the separate metrics connection.
"""

from __future__ import annotations

import argparse
import asyncio
import fcntl
import json
import os
import shutil
import statistics
import sys
import tarfile
import time
from pathlib import Path

import httpx
import yaml

from apertus_bench import research_pilot as h
from apertus_bench.client import StreamingChatClient
from apertus_bench.pilot_validation import gateway_key
from apertus_bench.prometheus import parse_speculative_snapshot, speculative_delta
from apertus_bench.runner import summarize
from apertus_bench.workloads import Prompt, file_sha256, load_prompts

ROOT = Path(__file__).resolve().parents[2]
TERMINAL = {"FAILED", "CANCELLED", "TIMEOUT", "COMPLETED", "NODE_FAIL", "OUT_OF_MEMORY"}
LIMITATION = (
    "Handwritten synthetic usage examples, not a production sample or correctness gate. "
    "One sequential request per prompt/model, normal EOS, thinking requested off. "
    "The existing vanilla endpoint has different hardware/runtime/template/context settings "
    "and uncontrolled concurrent traffic. Timings show the observed user experience; "
    "they do not isolate the drafter's speedup. Draft acceptance uses worker-wide counter "
    "deltas and can include other traffic; it is not a token-level trace of this request."
)


def progress(out: Path, phase: str, **values) -> None:
    temporary = out / "progress.tmp"
    temporary.write_text(json.dumps({"at": h.now(), "phase": phase, **values}, indent=2) + "\n")
    os.replace(temporary, out / "progress.json")


def load_cases(path: Path) -> list[dict]:
    cases = yaml.safe_load(path.read_text())["prompts"]
    if not cases or len({c["id"] for c in cases}) != len(cases):
        raise ValueError("usage cases must be nonempty and have unique IDs")
    for case in cases:
        if not case["id"].replace("-", "").isalnum() or not 1 <= case["max_tokens"] <= 1024:
            raise ValueError("unsafe case ID or excessive output cap")
        case["chat_template_kwargs"] = {"enable_thinking": False}
    return cases


def objective_checks(case: dict, measurement: dict) -> dict:
    checks = {
        "transport_success": measurement["success"],
        "natural_finish": measurement.get("finish_reason") == "stop",
        "nonempty_final_answer": bool(measurement.get("final_output")),
        "no_reasoning_output": not measurement.get("reasoning_output"),
    }
    if "expected_json" in case:
        try:
            checks["exact_json_extraction"] = (
                json.loads(measurement.get("final_output") or "") == case["expected_json"]
            )
        except (ValueError, TypeError):
            checks["exact_json_extraction"] = False
    return checks


async def wait_ready(run: Path, seconds: float, direct: bool = False) -> dict:
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        info = h.inspect_run(run)
        fields = info.get("slurm", "").split("|")
        if len(fields) > 1 and fields[1].split()[0] in TERMINAL:
            raise RuntimeError(f"serving deployment ended before demo: {fields[1]}")
        if direct and info.get("base_url") and info.get("preflight", {}).get("status") == "passed":
            job = info["preflight"]["slurm_job_id"]
            for path in sorted((run / "checks").glob("*/report.json"), reverse=True):
                report = h.read_json(path)
                if (
                    report.get("passed")
                    and not report.get("gateway")
                    and report.get("runtime_job_id") == job
                ):
                    async with httpx.AsyncClient(timeout=5) as health:
                        response = await health.get(info["base_url"] + "/health")
                        response.raise_for_status()
                    info["startup_validation_report"] = str(path)
                    info["demo_route"] = "direct worker; dev gateway remains unavailable"
                    return info
        for path in sorted((run / "automation").glob("*/report.json"), reverse=True):
            report = h.read_json(path)
            if report.get("passed") and report.get("gateway_checks"):
                info["startup_validation_report"] = str(path)
                if info.get("base_url") and info.get("preflight", {}).get("status") == "passed":
                    return info
            if report.get("error"):
                raise RuntimeError("startup validation failed; inspect its report before demo")
        await asyncio.sleep(min(30, max(0, deadline - time.monotonic())))
    raise TimeoutError("deployment did not finish startup checks within demo wait")


async def capture_cell(client, metrics, metrics_url, prompt, case, out, arm, metadata):
    cell = out / "cells" / prompt.id / arm
    cell.mkdir(parents=True, exist_ok=False)
    h.write_json(
        cell / "metadata.json",
        {
            **metadata,
            "arm": arm,
            "model": client.model,
            "base_url": client.base_url,
            "started_at": h.now(),
            "prompt_id": prompt.id,
            "prompt_sha256": h.digest(case),
            "generation": {
                "temperature": 0,
                "top_p": 1,
                "seed": 1,
                "max_tokens": prompt.max_tokens,
                "ignore_eos": False,
                "chat_template_kwargs": prompt.chat_template_kwargs,
            },
            "concurrency": 1,
            "request_count": 1,
            "metrics_scope": "shared worker",
        },
    )
    before = ""
    metrics_error = None
    if metrics_url:
        try:
            response = await metrics.get(metrics_url)
            response.raise_for_status()
            before = response.text
            (cell / "metrics_before.prom").write_text(before)
        except httpx.HTTPError as exc:
            metrics_error = type(exc).__name__
    started = time.perf_counter()
    result = await client.request(
        prompt,
        0,
        temperature=0,
        top_p=1,
        seed=1,
        max_tokens=None,
        ignore_eos=False,
        store_output=True,
        store_stream=True,
    )
    elapsed = time.perf_counter() - started
    row = result.to_dict()
    # Persist the response even if a later metrics read or parser fails.
    (cell / "requests.jsonl").write_text(json.dumps(row, ensure_ascii=False) + "\n")
    delta = {"enabled": False}
    if metrics_url:
        try:
            response = await metrics.get(metrics_url)
            response.raise_for_status()
            (cell / "metrics_after.prom").write_text(response.text)
            if before:
                delta = speculative_delta(
                    parse_speculative_snapshot(before, client.model),
                    parse_speculative_snapshot(response.text, client.model),
                )
        except (httpx.HTTPError, ValueError) as exc:
            metrics_error = type(exc).__name__
    summary = summarize([result], elapsed, delta)
    summary["objective_checks"] = objective_checks(case, row)
    summary["metrics_error"] = metrics_error
    h.write_json(cell / "summary.json", summary)
    return {
        "case_id": prompt.id,
        "arm": arm,
        "measurement": row,
        "speculative_decoding": delta,
        "checks": summary["objective_checks"],
        "cell": str(cell.relative_to(out)),
    }


def readable_report(cases: list[dict], rows: list[dict], model: str, baseline: str) -> str:
    lines = [
        "# DSpark e10: representative usage examples",
        "",
        LIMITATION,
        "",
        f"DSpark: `{model}`. Existing vanilla: `{baseline}`.",
        "",
        "| Prompt | Endpoint | First token (s) | Total (s) | Output tokens "
        "| Finish | Draft acceptance* |",
        "|---|---|---:|---:|---:|---|---:|",
    ]
    for row in rows:
        m, spec = row["measurement"], row["speculative_decoding"]
        ttft = f"{m['ttft_ms'] / 1000:.3f}" if m.get("ttft_ms") is not None else "unavailable"
        rate = spec.get("acceptance_rate")
        acceptance = f"{100 * rate:.1f}%" if rate is not None else "unavailable"
        lines.append(
            f"| {row['case_id']} | {row['arm']} | {ttft} | {m['e2e_ms'] / 1000:.3f} "
            f"| {m.get('completion_tokens')} | {m.get('finish_reason')} | {acceptance} |"
        )
    lines += [
        "",
        "*Worker-wide metric delta during the DSpark request; other traffic can contribute.",
        "",
        "Finish `length` means the answer reached its output cap. "
        "Timings include gateway/network overhead. "
        "Objective checks below are limited checks, not a quality or losslessness verdict.",
    ]
    for case in cases:
        lines += ["", f"## {case['id']}", "", "### Prompt", ""]
        for message in case["messages"]:
            lines += [f"**{message['role']}**", "", message["content"], ""]
        lines += ["Review points: " + case["review"], ""]
        for row in rows:
            if row["case_id"] != case["id"]:
                continue
            m = row["measurement"]
            lines += [
                f"### {row['arm']} answer",
                "",
                m.get("final_output") or m.get("output") or "(no answer)",
                "",
                "Objective checks: "
                + ", ".join(f"{k}={'pass' if v else 'fail'}" for k, v in row["checks"].items()),
                "",
                f"Raw response: [{row['cell']}/requests.jsonl]({row['cell']}/requests.jsonl)",
            ]
            if m.get("error"):
                lines += ["", "Request error: " + m["error"]]
    return "\n".join(lines) + "\n"


async def execute(args, out: Path, cases: list[dict], info: dict) -> None:
    run = args.run.resolve()
    dep = h.read_json(run / "deployment.json")
    runtime = info["preflight"]
    profile = h.read_json(run / "profile.json")
    direct = getattr(args, "direct", False)
    baseline_platform = getattr(args, "baseline_platform", None) or dep["platform"]
    base = profile["deployment"]["gateway"][baseline_platform]
    key = gateway_key(run, "CSCS_SERVING_API", baseline_platform)
    candidate_base = info["base_url"] if direct else dep["gateway"]
    candidate_key = None if direct else gateway_key(run, "CSCS_SERVING_API", dep["platform"])
    metadata = {
        "deployment_id": dep["deployment_id"],
        "checkpoint_sha256": runtime["drafter"]["checkpoint_sha256"],
        "target_model": profile["target"]["model"],
        "target_revision": profile["target"]["revision"],
        "target_tensor_parallel_size": profile["engine"]["target_tensor_parallel_size"],
        "image_sha256": runtime["image_sha256"],
        "slurm_job_id": runtime["slurm_job_id"],
        "node": runtime["node"],
        "harness_repository": h.git_info(ROOT),
        "corpus_sha256": file_sha256(out / "prompts.jsonl"),
        "gateway": base,
        "candidate_route": candidate_base,
        "baseline_route": base,
        "purpose": "bounded interactive demonstration",
        "limitations": LIMITATION,
    }
    metadata["harness_repository"].pop("diff", None)
    async with httpx.AsyncClient(timeout=30) as metrics:
        listing = await metrics.get(base + "/v1/models")
        listing.raise_for_status()
        matched = [
            m
            for m in listing.json().get("data", [])
            if m.get("id") in (dep["model"], args.baseline)
        ]
        h.write_json(out / "endpoint-listing.json", {"at": h.now(), "models": matched})
        baseline = next((m for m in matched if m["id"] == args.baseline), None)
        if not baseline or baseline.get("status") not in (None, "ready"):
            raise RuntimeError("configured existing vanilla endpoint is not ready")
        labels = baseline.get("labels", {})
        revision = labels.get("model_revision")
        if revision and revision != profile["target"]["revision"]:
            raise ValueError("existing baseline advertises a different target revision")
        for name in (
            "deployment.json",
            "profile.json",
            "preflight.json",
            "launch.json",
            "source-manifest.json",
            "launch-manifest.json",
            "environment.toml",
            "repository.json",
            "rendered-entrypoint.patch",
            "submission-attempt.json",
            "submission-result.json",
            "job.json",
        ):
            shutil.copy2(run / name, out / "serving-provenance" / name)
        shutil.copytree(run / "rendered", out / "serving-provenance/rendered")
        shutil.copy2(
            info["startup_validation_report"], out / "serving-provenance/startup-validation.json"
        )
        h.write_json(out / "metadata.json", metadata)
        h.write_json(out / "runtime-status.json", info)
        vanilla_metadata = {
            **metadata,
            "comparison_deployment_id": dep["deployment_id"],
            "deployment_id": baseline.get("worker_group_id"),
            "checkpoint_sha256": None,
            "image_sha256": None,
            "slurm_job_id": baseline.get("slurm_job_id") or None,
            "node": None,
            "pod_hostname": baseline.get("hostname"),
            "target_revision": revision,
            "target_tensor_parallel_size": None,
            "identity_source": "gateway model listing; complete launch provenance unavailable",
            "advertised_runtime_labels": labels,
        }
        rows = []
        async with (
            StreamingChatClient(
                candidate_base, dep["model"], candidate_key, timeout_seconds=180
            ) as candidate,
            StreamingChatClient(base, args.baseline, key, timeout_seconds=180) as vanilla,
        ):
            clients = {"dspark": candidate, "vanilla": vanilla}
            warmup = Prompt(
                "warmup",
                "warmup",
                [{"role": "user", "content": "Reply with one short greeting."}],
                16,
                {"enable_thinking": False},
            )
            with (out / "warmup.jsonl").open("x") as log:
                for arm, client in clients.items():
                    response = await client.request(
                        warmup,
                        0,
                        temperature=0,
                        top_p=1,
                        seed=1,
                        max_tokens=None,
                        ignore_eos=False,
                        store_output=True,
                    )
                    log.write(
                        json.dumps({"arm": arm, **response.to_dict()}, ensure_ascii=False) + "\n"
                    )
            prompts = load_prompts(out / "prompts.jsonl")
            for index, (prompt, case) in enumerate(zip(prompts, cases, strict=True)):
                order = ("dspark", "vanilla") if index % 2 == 0 else ("vanilla", "dspark")
                for arm in order:
                    progress(out, "running", prompt=prompt.id, endpoint=arm, completed=len(rows))
                    rows.append(
                        await capture_cell(
                            clients[arm],
                            metrics,
                            info["base_url"] + "/metrics" if arm == "dspark" else None,
                            prompt,
                            case,
                            out,
                            arm,
                            metadata if arm == "dspark" else vanilla_metadata,
                        )
                    )
        text = readable_report(cases, rows, dep["model"], args.baseline)
        text += f"\nRequest routes: DSpark `{candidate_base}`, vanilla `{base}`.\n"
        if direct:
            text += (
                "DSpark was queried directly because it was not discoverable on the dev gateway.\n"
            )
        (out / "report.md").write_text(text)
        summary = {
            "at": h.now(),
            "requests": len(rows),
            "limitations": LIMITATION,
            "all_requests_succeeded": all(r["measurement"]["success"] for r in rows),
            "quality_verdict": "manual review required; see full answers and review points",
        }
        for arm in ("dspark", "vanilla"):
            samples = [
                r["measurement"] for r in rows if r["arm"] == arm and r["measurement"]["success"]
            ]
            summary[arm] = {
                "successful": len(samples),
                "median_e2e_ms": statistics.median(m["e2e_ms"] for m in samples)
                if samples
                else None,
                "median_ttft_ms": statistics.median(
                    m["ttft_ms"] for m in samples if m["ttft_ms"] is not None
                )
                if any(m["ttft_ms"] is not None for m in samples)
                else None,
            }
        observed = [r["speculative_decoding"] for r in rows if r["arm"] == "dspark"]
        draft_tokens = sum(s.get("draft_tokens", 0) for s in observed)
        summary["dspark"]["draft_token_acceptance"] = (
            sum(s.get("accepted_tokens", 0) for s in observed) / draft_tokens
            if draft_tokens
            else None
        )
        summary["dspark"]["advancing_speculative_counters_observed"] = all(
            s.get("enabled") and s.get("drafts", 0) > 0 for s in observed
        )
        h.write_json(out / "summary.json", summary)
        progress(out, "complete", report=str(out / "report.md"), requests=len(rows))


async def run(args) -> None:
    if args.wait_seconds <= 0:
        raise ValueError("wait duration must be positive")
    out = args.output.resolve()
    cases = load_cases(args.prompts)
    out.mkdir(parents=True, exist_ok=False)
    (out / "serving-provenance").mkdir()
    shutil.copy2(args.prompts, out / "usage-prompts.yaml")
    shutil.copy2(__file__, out / "run-usage-demo.py")
    with (out / "prompts.jsonl").open("x") as sink:
        for case in cases:
            sink.write(json.dumps(case, ensure_ascii=False) + "\n")
    h.write_json(
        out / "controller.json",
        {
            "created_at": h.now(),
            "argv": sys.argv,
            "run": str(args.run.resolve()),
            "source_sha256": file_sha256(Path(__file__)),
            "prompt_sha256": file_sha256(args.prompts),
            "repository": h.git_info(ROOT),
        },
    )
    with (out / ".lock").open("x") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            progress(
                out,
                "waiting_for_startup_checks",
                job_id=h.read_json(args.run / "job.json")["job_id"],
            )
            info = await wait_ready(args.run, args.wait_seconds, getattr(args, "direct", False))
            await execute(args, out, cases, info)
        except Exception as exc:
            h.write_json(out / "failure.json", {"at": h.now(), "error": str(exc)})
            progress(out, "failed", error=str(exc))
            raise
        finally:
            h.write_json(
                out / "artifact-manifest.json",
                {
                    str(p.relative_to(out)): file_sha256(p)
                    for p in sorted(out.rglob("*"))
                    if p.is_file() and p.name not in {".lock", "progress.tmp"}
                },
            )
            # Includes raw requests, full answers and launch provenance; no API key.
            with tarfile.open(out.with_suffix(".tar.gz"), "w:gz") as archive:
                archive.add(out, arcname=out.name)
    print(out / "report.md")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--prompts", type=Path, default=Path(__file__).with_name("usage-prompts.yaml")
    )
    parser.add_argument("--baseline", default="SwissAI-Research/swiss-ai/Apertus-v1.5-8B-fixed")
    parser.add_argument("--wait-seconds", type=float, default=129600)
    parser.add_argument("--direct", action="store_true", help="use validated direct worker route")
    parser.add_argument("--baseline-platform", choices=("dev", "research"))
    asyncio.run(run(parser.parse_args()))
