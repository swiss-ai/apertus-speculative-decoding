"""Resume only the unfinished final-checkpoint layout campaign.

The controller runs on the login host and submits one bounded existing node
sweep at a time. It performs only lightweight status/JSON checks there. Servers,
benchmarks, native imports and image/checkpoint hashing run in Slurm allocations.
"""

from __future__ import annotations

import argparse
import copy
import gzip
import hashlib
import json
import os
import re
import subprocess
import time
from pathlib import Path
from typing import Any

import yaml

from apertus_bench.prometheus import parse_speculative_snapshot, speculative_delta

REPO = Path(__file__).resolve().parents[2]
SPECS = REPO / "experiments/high-concurrency-8b"
# This image exports CUDA graph buffers via cudaIpcGetMemHandle, which does
# not accept expandable-segment allocations. Retain the standard allocator.
ALLOCATOR = {"PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:False"}
STAGES = [
    ("preflight", "01:30:00"),
    ("tp4-dspark-off-k7", "02:30:00"),
    ("tp4-eagle-k7", "02:30:00"),
    ("tp4-dspark-on-k7", "03:00:00"),
    ("think-pd", "05:00:00"),
    ("tp2dp2", "05:00:00"),
    ("pd-off", "05:00:00"),
]
TERMINAL = {
    "COMPLETED",
    "FAILED",
    "CANCELLED",
    "TIMEOUT",
    "NODE_FAIL",
    "OUT_OF_MEMORY",
    "PREEMPTED",
    "BOOT_FAIL",
    "DEADLINE",
}


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def write_state(path: Path, state: dict[str, Any]) -> None:
    """Expose a bounded top-level registry for the parent's read-only monitor."""
    state["workstream"] = "layouts"
    state["state"] = state["campaign_state"]
    state["job_ids"] = [row["job_id"] for row in state["stages"]]
    state["updated_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    state["scientific_completion"] = "pending_scientific_review"
    if state["campaign_state"] == "active":
        current = state["stages"][-1] if state["stages"] else {}
        state["current_job"] = current.get("job_id")
        state["next_action"] = (
            "Await active stage, validate gates, then submit the next bounded stage"
        )
    else:
        state["next_action"] = (
            "Inspect gates/artifacts and finish scientific review before any recommendation"
        )
    write_json(path, state)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(1024 * 1024):
            h.update(chunk)
    return h.hexdigest()


def memory_safe(spec: dict[str, Any]) -> dict[str, Any]:
    """Reserve 9.5 GiB extra headroom with IPC-compatible graph allocations."""
    result = copy.deepcopy(spec)
    result.setdefault("engine", {})["gpu_memory_utilization"] = 0.7
    for arm in result["arms"]:
        arm.setdefault("env", {}).update(ALLOCATOR)
    return result


def dependency_paths(spec: dict[str, Any], site: Path) -> dict[str, Any]:
    result = copy.deepcopy(spec)
    for arm in result["arms"]:
        if arm.get("method") == "pd":
            arm["pythonpath"] = [str(site)]
    return result


def load_spec(filename: str) -> dict[str, Any]:
    return yaml.safe_load((SPECS / filename).read_text())


def recovery_specs(site: Path) -> dict[str, dict[str, Any]]:
    off = memory_safe(load_spec("tp4-b1.yaml"))
    eagle = memory_safe(load_spec("tp4-c2.yaml"))
    thinking = memory_safe(load_spec("think-layouts.yaml"))
    thinking["arms"] = [
        a for a in thinking["arms"] if a["name"] in {"plain-tp4", "dspark-on-k7-tp4"}
    ]
    thinking["engine"]["max_num_seqs"] = 1024
    preflight = copy.deepcopy(off)
    preflight["loadtests"] = [
        {
            **off["loadtests"][0],
            "label": "opb-memory-stress",
            "concurrencies": [1024],
            "requests_per_slot": 1,
            "min_requests": 1024,
            "max_tokens": 128,
        },
    ]
    preflight["arms"] = [
        {
            "name": "nixl-native-import",
            "method": "command",
            "gpus": 1,
            "command": [
                "python3",
                "{repo}/experiments/high-concurrency-8b/recovery/check_dependencies.py",
                "--site",
                str(site),
                "--output",
                "{arm_dir}/dependency-report.json",
            ],
        },
        copy.deepcopy(off["arms"][0]),
        copy.deepcopy(eagle["arms"][1]),
        copy.deepcopy(off["arms"][1]),
    ]
    # Keep the preflight matched thinking-off; the thinking-on paired retry
    # validates its own original prompts instead of mixing mode/head controls.
    # k=3/5 were the original bounded finalists, selected from the single-GPU
    # grid. Retain both in TP2xDP2 because changing TP can shift the crossover.
    # P/D uses the original k=3 high-load setting, with no new k/model sweep.
    specs = {
        "preflight": preflight,
        "tp4-dspark-off-k7": off,
        "tp4-eagle-k7": eagle,
        "tp4-dspark-on-k7": thinking,
        "think-pd": dependency_paths(load_spec("think-pd.yaml"), site),
        "tp2dp2": load_spec("tp2dp2.yaml"),
        "pd-off": dependency_paths(load_spec("pd.yaml"), site),
    }
    for name, spec in specs.items():
        spec["name"] = "layout-recovery-" + name
    return specs


def source_identity() -> dict[str, Any]:
    tracked = (
        subprocess.check_output(
            [
                "git",
                "ls-files",
                "-z",
                "src",
                "serving",
                "methods/eagle/configs",
                "experiments/high-concurrency-8b",
            ],
            cwd=REPO,
        )
        .decode()
        .split("\0")
    )
    return {
        "commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO, text=True
        ).strip(),
        "files": {name: sha256(REPO / name) for name in tracked if name},
    }


def cell_quality(path: Path, method: str) -> dict[str, Any]:
    """Check retained provenance and observable counter continuity."""
    errors = []
    metadata_path = path.with_name("metadata.json")
    meta = json.loads(metadata_path.read_text()) if metadata_path.exists() else {}
    extra = meta.get("extra", {})
    if (
        not meta.get("workload_file_sha256")
        or not extra.get("deployment_id")
        or not extra.get("engine")
    ):
        errors.append("Missing corpus/deployment/engine provenance")
    if method == "eagle" and any(
        not extra.get(k)
        for k in (
            "checkpoint_sha256",
            "target_model",
            "target_revision",
            "target_tensor_parallel_size",
        )
    ):
        errors.append("Missing EAGLE target/checkpoint contract")
    snapshots = [path.with_name("metrics_before.prom"), path.with_name("metrics_after.prom")]
    counter_result = None
    if not all(p.exists() for p in snapshots):
        errors.append("Missing raw metric snapshots")
    else:
        try:
            counter_result = speculative_delta(
                *(parse_speculative_snapshot(p.read_text()) for p in snapshots)
            )
            if method in {"dspark", "eagle"} and not counter_result.get("enabled"):
                errors.append("Speculative counters unavailable")
        except ValueError as exc:
            errors.append(str(exc))
    series = path.with_name("metrics_timeseries.jsonl")
    opener = Path.open
    if not series.exists() and series.with_suffix(".jsonl.gz").exists():
        series = series.with_suffix(".jsonl.gz")
        opener = gzip.open
    if not series.exists():
        errors.append("Missing metric timeseries")
    else:
        previous = {}
        with opener(series, "rt") as stream:
            for line in stream:
                row = json.loads(line)
                for key in ("prompt_tokens", "generation_tokens", "preemptions"):
                    value = row.get(key)
                    if value is not None and key in previous and value < previous[key]:
                        errors.append(f"Cumulative {key} decreased")
                    if value is not None:
                        previous[key] = value
    return {
        "passed": not errors,
        "errors": sorted(set(errors)),
        "counter_delta": counter_result,
        "metadata_sha256": sha256(metadata_path) if metadata_path.exists() else None,
        "observability_limit": (
            "Snapshots plus sampled generic counters cannot exclude every unsampled restart."
        ),
    }


def cell_gate(run: Path, spec: dict[str, Any], stage: str = "") -> dict[str, Any]:
    """Slurm COMPLETED is insufficient: every intended arm and cell must pass."""
    checks: list[dict[str, Any]] = []
    status_file = run / "sweep-status.json"
    statuses = json.loads(status_file.read_text()) if status_file.exists() else {}
    for arm in spec["arms"]:
        name = arm["name"]
        checks.append(
            {
                "kind": "arm_status",
                "arm": name,
                "passed": statuses.get(name) == "ok",
                "observed": statuses.get(name),
            }
        )
        if arm["method"] == "command":
            p = run / name / "dependency-report.json"
            passed = p.exists() and json.loads(p.read_text()).get("passed") is True
            checks.append({"kind": "native_dependency_imports", "arm": name, "passed": passed})
            continue
        for test in spec["loadtests"]:
            label = test.get("label", test["workload"])
            for c in test["concurrencies"]:
                path = run / name / label / test["workload"] / f"c{c}" / "summary.json"
                passed = False
                requests: dict[str, Any] = {}
                if path.exists():
                    data = json.loads(path.read_text())
                    requests = data.get("requests", {})
                    passed = (
                        requests.get("attempted", 0) > 0
                        and requests.get("success_rate", 0) >= 0.99
                        and data.get("tokens", {}).get("output_tokens_per_second", 0) > 0
                    )
                checks.append(
                    {
                        "kind": "cell",
                        "arm": name,
                        "label": label,
                        "concurrency": c,
                        "path": str(path),
                        "passed": passed,
                        "requests": requests,
                    }
                )
                if path.exists():
                    quality = cell_quality(path, arm.get("serve_method", arm["method"]))
                    checks.append(
                        {
                            "kind": "provenance_counter_continuity",
                            "arm": name,
                            "label": label,
                            "concurrency": c,
                            **quality,
                        }
                    )
                # Catch the original gross slowdown (c128=2,017; c1024=839
                # tok/s) before accepting ratios. These are deliberately loose
                # floors relative to healthy old TP4 baselines. The 0.7 budget
                # can change capacity: a failed floor requires manual review,
                # not an automatic claim that the treatment has no benefit.
                floors = {
                    "opb": {1: 240, 128: 9500, 1024: 8000},
                    "opb-think": {4: 900, 128: 11500, 1024: 9500},
                }
                floor = floors.get(label, {}).get(c)
                if stage.startswith("tp4-") and name == "plain-tp4" and floor is not None:
                    observed = (
                        data.get("tokens", {}).get("output_tokens_per_second", 0)
                        if path.exists()
                        else 0
                    )
                    checks.append(
                        {
                            "kind": "baseline_health",
                            "arm": name,
                            "label": label,
                            "concurrency": c,
                            "minimum_tok_s": floor,
                            "observed_tok_s": observed,
                            "passed": observed >= floor,
                        }
                    )
    return {
        "passed": all(row["passed"] for row in checks),
        "checks": checks,
        "claim_limit": (
            "Infrastructure/completeness gate; no proof of output equivalence "
            "or repeatable deployment benefit."
        ),
    }


def collect_rows(run: Path, stage: str) -> list[dict[str, Any]]:
    rows = []
    for p in sorted(run.glob("*/**/c*/summary.json")):
        d = json.loads(p.read_text())
        rel = p.relative_to(run).parts
        s = d.get("speculative_decoding", {}) or {}
        metadata_path = p.with_name("metadata.json")
        meta = json.loads(metadata_path.read_text()) if metadata_path.exists() else {}
        method = meta.get("variant", {}).get("method", "baseline")
        method = "eagle" if method == "eagle3" else method
        rows.append(
            {
                "stage": stage,
                "run": str(run),
                "arm": rel[0],
                "label": rel[1],
                "concurrency": int(p.parent.name[1:]),
                "summary": str(p),
                "summary_sha256": sha256(p),
                "metadata": {k: meta.get(k) for k in ("workload_file_sha256", "cell", "extra")},
                "quality": cell_quality(p, method),
                "throughput_tok_s": d["tokens"]["output_tokens_per_second"],
                "successful": d["requests"]["successful"],
                "failed": d["requests"]["failed"],
                "acceptance_length": s.get("mean_acceptance_length"),
                "latency_ms": d.get("latency_ms"),
                "load": d.get("load"),
                "completion_tokens": d["tokens"]["completion"],
                "wall_seconds": d["wall_seconds"],
            }
        )
    plain = {
        (r["arm"], r["label"], r["concurrency"]): r for r in rows if r["arm"].startswith("plain")
    }
    for r in rows:
        suffix = (
            "tp4"
            if r["arm"].endswith("tp4")
            else "tp2dp2"
            if r["arm"].endswith("tp2dp2")
            else "dp4"
        )
        base = plain.get(("plain-" + suffix, r["label"], r["concurrency"]))
        r["speedup_vs_matched_plain"] = (
            r["throughput_tok_s"] / base["throughput_tok_s"] if base else None
        )
        r["plain_summary"] = base["summary"] if base else None
        r["matched_generation_and_corpus"] = bool(
            base
            and r["metadata"].get("workload_file_sha256")
            == base["metadata"].get("workload_file_sha256")
            and (r["metadata"].get("cell") or {}).get("generation")
            == (base["metadata"].get("cell") or {}).get("generation")
        )
    return rows


def slurm_state(job: str) -> dict[str, Any]:
    cmd = ["sacct", "-X", "-n", "-P", "-j", job, "--format=JobIDRaw,State,ExitCode,ElapsedRaw"]
    result = subprocess.run(cmd, text=True, capture_output=True, check=True, timeout=30)
    for line in result.stdout.splitlines():
        fields = line.split("|")
        if fields[0] == job:
            return {
                "job_id": job,
                "state": fields[1].split()[0].rstrip("+"),
                "exit_code": fields[2],
                "elapsed_seconds": fields[3],
            }
    return {"job_id": job, "state": "UNKNOWN"}


def controller(root: Path, site: Path, partition: str, poll: int) -> None:
    root.mkdir(parents=True, exist_ok=True)
    state_path = root / "state.json"
    state = (
        json.loads(state_path.read_text())
        if state_path.exists()
        else {
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "stages": [],
            "campaign_state": "active",
            "scope": "unfinished layout recovery only",
            "source_commit": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], text=True, cwd=REPO
            ).strip(),
            "dependency_site": str(site),
            "plain_treatment_memory_matching": True,
            "source_identity": source_identity(),
            "job_id_schema": (
                "stages[*].job_id (decimal Slurm allocation ID; "
                "stage/output/spec/gate are adjacent)"
            ),
            "completion_rule": (
                "All bounded stages measured plus scientific review; "
                "terminal Slurm states alone never mean campaign complete"
            ),
            "memory_recipe": (
                "TP4 k7 retries/preflight: utilization0.7 and expandable_segments=False "
                "for both paired arms; other original layouts0.8"
            ),
        }
    )
    if state.get("pending_submission"):
        state.update(
            campaign_state="needs_attention",
            error=(
                "Previous submission outcome uncertain; inspect pending_submission output/job.txt "
                "and Slurm before recording a job ID or retrying"
            ),
        )
        write_state(state_path, state)
        return
    specs = recovery_specs(site)
    spec_root = root / "specs"
    spec_root.mkdir(exist_ok=True)
    env = {
        **os.environ,
        "PATH": str(Path.home() / "venvs/sml-apertus/bin") + ":" + os.environ["PATH"],
        "MODEL_LAUNCH_ROOT": str(Path.home() / "model-launch-apertus"),
    }
    for stage, limit in STAGES:
        records = [r for r in state["stages"] if r["stage"] == stage]
        if (
            records
            and records[-1].get("state") == "measured"
            and records[-1].get("gate", {}).get("passed")
        ):
            continue
        record = records[-1] if records else None
        spec = specs[stage]
        path = spec_root / f"{stage}.yaml"
        expected = yaml.safe_dump(spec, sort_keys=False)
        if path.exists() and path.read_text() != expected:
            raise RuntimeError(f"Refusing to change a submitted stage configuration: {path}")
        path.write_text(expected)
        if record is None:
            if source_identity() != state["source_identity"]:
                state.update(
                    campaign_state="needs_attention",
                    error="Source changed after controller registration; refusing new deployments",
                )
                write_state(state_path, state)
                return
            stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
            out = root / "runs" / f"{stage}-{stamp}"
            if out.exists():
                raise RuntimeError(f"Fresh run directory already exists: {out}")
            run_env = {
                **env,
                "SWEEP_PARTITION": partition,
                "SWEEP_TIME": limit,
                "SWEEP_OUTPUT": str(out),
                "VLLM_PATCH_DIR": str(Path.home() / f".sml/lr-{stage}-{stamp}"),
            }
            command = [str(REPO / "serving/node-sweep.sh"), str(path)]
            state["pending_submission"] = {
                "stage": stage,
                "output": str(out),
                "spec": str(path),
                "command": command,
                "started_at": stamp,
            }
            write_state(state_path, state)
            try:
                submitted = subprocess.run(
                    command, text=True, capture_output=True, cwd=REPO, env=run_env, timeout=600
                )
            except (subprocess.SubprocessError, OSError) as exc:
                state.update(
                    campaign_state="needs_attention", error=f"Submission outcome uncertain: {exc}"
                )
                write_state(state_path, state)
                return
            submit = {
                "command": command,
                "returncode": submitted.returncode,
                "stdout": submitted.stdout,
                "stderr": submitted.stderr,
                "environment": {
                    k: run_env[k]
                    for k in [
                        "SWEEP_PARTITION",
                        "SWEEP_TIME",
                        "SWEEP_OUTPUT",
                        "VLLM_PATCH_DIR",
                        "MODEL_LAUNCH_ROOT",
                    ]
                },
            }
            write_json(root / f"submit-{stage}-{stamp}.json", submit)
            match = re.search(r"job=(\d+)", submitted.stdout)
            if submitted.returncode or not match:
                state.update(campaign_state="needs_attention", error=f"Submission failed: {stage}")
                write_state(state_path, state)
                raise RuntimeError(state["error"])
            record = {
                "stage": stage,
                "job_id": match[1],
                "output": str(out),
                "spec": str(path),
                "spec_sha256": sha256(path),
                "state": "submitted",
            }
            state["stages"].append(record)
            state.pop("pending_submission", None)
            write_state(state_path, state)
        while True:
            try:
                observed = slurm_state(record["job_id"])
            except (subprocess.SubprocessError, OSError) as exc:
                record["status_poll_error"] = str(exc)
                record["consecutive_poll_errors"] = record.get("consecutive_poll_errors", 0) + 1
                write_state(state_path, state)
                if record["consecutive_poll_errors"] >= 10:
                    state.update(
                        campaign_state="needs_attention",
                        error=(
                            "Ten consecutive Slurm status errors; "
                            "inspect existing job before resuming"
                        ),
                    )
                    write_state(state_path, state)
                    return
                time.sleep(poll)
                continue
            record["consecutive_poll_errors"] = 0
            record["slurm"] = observed
            state["observed_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
            write_state(state_path, state)
            if observed["state"] in TERMINAL:
                try:
                    record["gate"] = cell_gate(Path(record["output"]), spec, stage)
                except (OSError, ValueError, KeyError) as exc:
                    record["gate"] = {
                        "passed": False,
                        "error": f"Malformed/incomplete terminal artifacts: {exc}",
                    }
                record["state"] = (
                    "measured"
                    if observed["state"] == "COMPLETED" and record["gate"]["passed"]
                    else "needs_attention"
                )
                write_json(root / f"gate-{stage}-{record['job_id']}.json", record["gate"])
                if record["state"] != "measured":
                    state.update(
                        campaign_state="needs_attention",
                        error=(
                            f"{stage} ended {observed['state']} "
                            "or failed completeness/request gates"
                        ),
                    )
                    write_state(state_path, state)
                    return
                write_json(
                    root / f"analysis-{stage}-{record['job_id']}.json",
                    {
                        "rows": collect_rows(Path(record["output"]), stage),
                        "claim_limit": (
                            "One paired deployment per configuration; "
                            "descriptive screening, not statistical confirmation."
                        ),
                    },
                )
                write_state(state_path, state)
                break
            time.sleep(poll)
    state["campaign_state"] = "measurements_complete_pending_scientific_review"
    state["implementation_recommendation"] = (
        "pending review; only substantial reproducible benefit warrants implementation"
    )
    write_state(state_path, state)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="command", required=True)
    p = sub.add_parser("generate")
    p.add_argument("--site", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p = sub.add_parser("controller")
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--site", type=Path, required=True)
    p.add_argument("--partition", default="normal")
    p.add_argument("--poll", type=int, default=60)
    a = ap.parse_args(argv)
    if a.command == "generate":
        a.output.mkdir(parents=True, exist_ok=False)
        for name, spec in recovery_specs(a.site).items():
            (a.output / f"{name}.yaml").write_text(yaml.safe_dump(spec, sort_keys=False))
    else:
        controller(a.root, a.site, a.partition, max(30, a.poll))


if __name__ == "__main__":
    main()
