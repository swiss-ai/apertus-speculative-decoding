"""Original T5 confirmation: two paired rounds on the same physical GPUs.

The existing node-sweep launcher runs this command on four allocated GPUs.
Round A serves base/adapt at k3 on GPUs0/1 and k7 on GPUs2/3. Round B
reverses each pair on those same GPUs. Captures calibrate cross-deployment
correctness; full original held-out passes measure serving performance.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import platform
import shutil
import subprocess
import sys
import tomllib
from pathlib import Path
from typing import Any

from apertus_eagle.adaptation_decision import (
    BASE_HEAD,
    digest,
    immutable_json,
    paired_metrics,
    read,
)

from apertus_bench.correctness import compare_captures, run_correctness_gate
from apertus_bench.eagle import validate_eagle_head
from apertus_bench.node_sweep import REPO_ROOT, build_plan, run_plan
from apertus_bench.workloads import load_prompts


def round_spec(selection: dict[str, Any], policy: dict[str, Any], repeat: str) -> dict[str, Any]:
    order = policy["confirmation"][f"gpus_round_{repeat}"]
    workloads = selection["workloads"]
    return {
        "name": selection["name"] + "-" + repeat,
        "stage": "8b",
        "node_gpus": 4,
        "engine": {"max_num_batched_tokens": 16384, "max_num_seqs": 256},
        "loadtests": [
            {
                **{k: w[k] for k in ("label", "workload", "workloads_file")},
                "concurrencies": policy["confirmation"]["concurrencies"],
                "requests_per_slot": policy["confirmation"]["requests_per_slot"],
                "min_requests": policy["confirmation"]["minimum_requests"],
                "generation": {"temperature": 0.7, "top_p": 0.8, "top_k": 20, "seed": 1},
            }
            for w in workloads
        ],
        "captures": [{k: w[k] for k in ("label", "workload", "workloads_file")} for w in workloads],
        "arms": [
            {
                "name": name,
                "method": "eagle",
                "k": int(name[-1]),
                "drafter": BASE_HEAD if name.startswith("base") else selection["head_dir"],
            }
            for name in sorted(order, key=order.get)
        ],
    }


def collect_report(
    output: Path, selection: dict[str, Any], policy: dict[str, Any]
) -> dict[str, Any]:
    gates, cells = [], []
    for repeat in ("a", "b"):
        status = read(output / repeat / "sweep-status.json")
        if len(status) != 4 or any(v != "ok" for v in status.values()):
            raise ValueError(f"Unsuccessful serving arms in round {repeat}: {status}")
        for arm, gpu in policy["confirmation"][f"gpus_round_{repeat}"].items():
            arm_status = read(output / repeat / arm / "status.json")
            if arm_status["gpus"] != [gpu]:
                raise ValueError(f"GPU assignment was not the preregistered swap: {repeat}/{arm}")
    for k in policy["confirmation"]["k"]:
        for workload in selection["workloads"]:
            label = workload["label"]

            def capture(repeat: str, arm: str, k: int = k, label: str = label) -> Path:
                return output / repeat / f"{arm}-k{k}" / f"capture-{label}.json"

            controls = [(capture("a", arm), capture("b", arm)) for arm in ("base", "adapt")]
            for left, right in controls:
                c = compare_captures(read(left), read(right))
                if c["total_cases"] == 0 or c["comparable_cases"] != c["total_cases"]:
                    raise ValueError("Incomplete same-configuration correctness calibration")
            for repeat in ("a", "b"):
                gate = run_correctness_gate(
                    (capture(repeat, "base"), capture(repeat, "adapt")),
                    controls,
                    output / "correctness" / f"k{k}-{label}-{repeat}.json",
                )
                gates.append({"k": k, "label": label, "repeat": repeat, **gate})
            for concurrency in policy["confirmation"]["concurrencies"]:
                pairs = []
                for repeat in ("a", "b"):
                    rows = []
                    for arm in ("base", "adapt"):
                        report = read(
                            output / repeat / f"{arm}-k{k}" / label / "loadtest-summary.json"
                        )
                        levels = {r["concurrency"]: r for r in report["levels"]}
                        if set(levels) != set(policy["confirmation"]["concurrencies"]):
                            raise ValueError("Missing or unexpected concurrency levels")
                        rows.append(levels[concurrency])
                        cell = (
                            output
                            / repeat
                            / f"{arm}-k{k}"
                            / label
                            / workload["workload"]
                            / f"c{concurrency}"
                        )
                        if not all(
                            (cell / name).is_file()
                            for name in (
                                "metadata.json",
                                "requests.jsonl",
                                "summary.json",
                                "metrics_before.prom",
                                "metrics_after.prom",
                            )
                        ):
                            raise ValueError(f"Incomplete raw measurement/provenance: {cell}")
                    pairs.append({"repeat": repeat, **paired_metrics(*rows)})
                cells.append(
                    {
                        "label": label,
                        "role": workload["role"],
                        "k": k,
                        "concurrency": concurrency,
                        "pairs": pairs,
                    }
                )
    return {
        "name": selection["name"],
        "correctness_passed": all(g["passed"] for g in gates),
        "correctness": gates,
        "cells": cells,
        "claim": (
            "Within-job serving comparisons and calibrated greedy controls; simulated"
            " acceptance is not serving performance."
        ),
    }


def provenance(output: Path, selection_path: Path, policy_path: Path, environment: Path) -> None:
    dest = output / "provenance"
    dest.mkdir(parents=True, exist_ok=False)
    resolved = tomllib.loads(environment.read_text())
    image = Path(resolved["image"])
    shutil.copy2(environment, dest / "environment.toml")
    shutil.copy2(selection_path, dest / "selection.json")
    shutil.copy2(policy_path, dest / "preregistration.json")
    shutil.copy2(REPO_ROOT / "targets/8b/contract.json", dest / "target-contract.json")
    frozen = REPO_ROOT / "source-manifest.json"
    source = read(frozen)
    for name, sha in source["files"].items():
        if digest(REPO_ROOT / name) != sha:
            raise ValueError(f"Frozen source changed: {name}")
    shutil.copy2(frozen, dest / "source-manifest.json")
    revision = subprocess.check_output(
        ["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"], text=True
    ).strip()
    if revision != source["revision"]:
        raise ValueError("Frozen source Git HEAD changed")
    (dest / "checkout.patch").write_bytes(
        subprocess.check_output(["git", "-C", str(REPO_ROOT), "diff", "HEAD", "--binary"])
    )
    patches = []
    for path in (REPO_ROOT / "serving/patches").glob("*.patch"):
        # The submitter's exact applied subset is retained in its patches.txt.
        patches.append({"path": str(path), "sha256": digest(path)})
    selection = read(selection_path)
    for label, head in (("base", BASE_HEAD), ("adapted", selection["head_dir"])):
        immutable_json(
            dest / f"{label}-head-report.json",
            validate_eagle_head(
                Path(head),
                expected_algorithm="eagle31",
                target_contract_path=REPO_ROOT / "targets/8b/contract.json",
            ),
        )
    corpora = {
        w["workloads_file"]: digest(Path(w["workloads_file"])) for w in selection["workloads"]
    }
    if corpora != selection["corpus_sha256"]:
        raise ValueError("Held-out corpus changed after confirmation selection")
    for w in selection["workloads"]:
        load_prompts(Path(w["workloads_file"]), w["workload"])
    immutable_json(
        dest / "runtime.json",
        {
            "argv": sys.argv,
            "git_revision": revision,
            "image": str(image),
            "image_sha256": digest(image),
            "python": sys.version,
            "machine": platform.machine(),
            "patch_files": patches,
            "overlay_mounts": resolved.get("mounts", []),
            "mount_entries": len(resolved.get("mounts", [])),
            "mount_specs_comma_joined_utf8_bytes": len(
                ",".join(resolved.get("mounts", [])).encode()
            ),
            "corpus_sha256": corpora,
            "environment": {
                key: os.environ[key]
                for key in (
                    "SLURM_JOB_ID",
                    "SLURM_JOB_NODELIST",
                    "SLURMD_NODENAME",
                    "CUDA_VISIBLE_DEVICES",
                    "PYTHONPATH",
                )
                if key in os.environ
            },
        },
    )


async def run(output: Path, selection: dict[str, Any], policy: dict[str, Any]) -> None:
    for repeat in ("a", "b"):
        spec = round_spec(selection, policy, repeat)
        round_out = output / repeat
        round_out.mkdir(exist_ok=False)
        os.environ["SWEEP_CACHE_ROOT"] = str(round_out / "vllm-cache")
        immutable_json(round_out / "spec.json", spec)
        plan = build_plan(spec)
        for arm in plan.arms:
            arm.metadata.update(
                {
                    "recovery_head": selection["name"],
                    "paired_repeat": repeat,
                    "slurm_job_id": os.environ["SLURM_JOB_ID"],
                    "slurm_node_id": os.environ.get("SLURMD_NODENAME", "unknown"),
                    "compile_cache": "fresh-per-deployment",
                    "compile_cache_root": str(round_out / "vllm-cache" / arm.name),
                }
            )
        results = await run_plan(plan, round_out)
        if any(r["status"] != "ok" for r in results):
            raise RuntimeError(f"Round {repeat} infrastructure failed; stop before further tests")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--environment", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    selection, policy = read(args.selection), read(args.policy)
    try:
        provenance(args.output, args.selection, args.policy, args.environment)
        asyncio.run(run(args.output, selection, policy))
        report = collect_report(args.output, selection, policy)
        immutable_json(args.output / "confirmation-summary.json", report)
        if not report["correctness_passed"]:
            raise RuntimeError("Calibrated correctness gate failed")
    except Exception as error:
        immutable_json(
            args.output / "failure.json",
            {"state": "needs_repair", "error": f"{type(error).__name__}: {error}"},
        )
        raise


if __name__ == "__main__":
    main()
