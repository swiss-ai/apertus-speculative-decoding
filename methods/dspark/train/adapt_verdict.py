"""Verdict of the DSpark adaptation repeat under the unchanged EAGLE rule.

Login node (stdlib + methods/eagle on PYTHONPATH). Reads the paired
confirmation summary, applies apertus_eagle.adaptation_decision.final_verdict
with the EAGLE preregistration, and writes verdict.json plus a CSV of the
absolute per-cell metrics. Compute is charged as for EAGLE: the reused trace
allocation (EAGLE trace jobs) plus every DSpark adaptation and confirmation
allocation, failed attempts included.

    PYTHONPATH=methods/eagle python3 methods/dspark/train/adapt_verdict.py --root ROOT \
        --attempt debug-20261105 --jobs 3856449,3861814,... --training-jobs ...
"""

from __future__ import annotations

import argparse
import csv
import subprocess
from pathlib import Path

from apertus_eagle.adaptation_decision import classify_cell, final_verdict, immutable_json, read

EAGLE_TRACE_JOBS = ("3769227", "3790165", "3802584", "3804324")
EAGLE_TRACE_GPU_SECONDS = 11380  # EAGLE campaign receipt: all four trace jobs.


def allocations(jobs: list[str]) -> dict:
    out = subprocess.check_output(
        ["sacct", "-X", "-n", "-P", "-j", ",".join(jobs),
         "--format=JobIDRaw,ElapsedRaw,AllocNodes,State,ExitCode,Partition"], text=True,
    )  # fmt: skip
    result = {}
    for line in out.splitlines():
        job, elapsed, nodes, state, code, partition = line.split("|")
        if job in jobs:
            result[job] = {"elapsed_seconds": int(elapsed), "nodes": int(nodes),
                           "gpu_seconds": 4 * int(elapsed) * int(nodes), "state": state,
                           "exit_code": code, "partition": partition}  # fmt: skip
    if set(result) != set(jobs):
        raise RuntimeError(f"missing accounting for {set(jobs) - set(result)}")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--attempt", required=True)
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--training-jobs", required=True, help="all adaptation-stage jobs")
    parser.add_argument("--confirmation-jobs", required=True)
    args = parser.parse_args()
    root = args.root
    training = args.training_jobs.split(",")
    confirmation = args.confirmation_jobs.split(",")
    jobs = allocations(training + confirmation)
    selection_path = next((root / "confirmation-configs" / args.attempt).glob("*-selection.json"))
    selected = read(selection_path)
    selected["training_wall_seconds"] = sum(
        jobs[j]["elapsed_seconds"] for j in training if jobs[j]["state"] == "COMPLETED"
    )
    report = read(root / "confirmations" / args.attempt / "paired" / "confirmation-summary.json")
    compute = {
        "trace_allocated_gpu_seconds": EAGLE_TRACE_GPU_SECONDS,
        "domain_training_allocated_gpu_seconds": {
            "mult": sum(jobs[j]["gpu_seconds"] for j in training)
        },
        "whole_campaign_allocated_gpu_seconds": EAGLE_TRACE_GPU_SECONDS
        + sum(v["gpu_seconds"] for v in jobs.values()),
        "basis": (
            "4 GPUs per exclusive node. Traces are the EAGLE campaign's (jobs "
            + ", ".join(EAGLE_TRACE_JOBS)
            + ") reused unchanged; DSpark adaptation and confirmation jobs include failed"
            " attempts. EAGLE campaign training is not charged here."
        ),
        "jobs": jobs,
    }
    policy = read(args.policy)
    if report["correctness_passed"]:
        verdict = final_verdict({"selected": [selected]}, [report], policy, compute)
    else:
        # The frozen policy makes a correctness failure terminal (needs_repair) and
        # final_verdict refuses it. Classify the cells with the same rule anyway, so
        # the record shows whether a passing gate could have changed the outcome;
        # nothing is recommended from a run whose correctness gate failed.
        cells = [cell | classify_cell(cell["pairs"], policy) for cell in report["cells"]]
        benefits = [
            c for c in cells if c["role"] != "forgetting" and c["substantial_repeatable_benefit"]
        ]
        verdict = {
            "state": "needs_repair",
            "verdict": "correctness_gate_failed"
            + ("" if benefits else "_and_no_substantial_repeatable_benefit"),
            "failed_gates": [g for g in report["correctness"] if not g["passed"]],
            "correctness_gates_passed": sum(g["passed"] for g in report["correctness"]),
            "correctness_gates_total": len(report["correctness"]),
            "selection": {"selected": [selected]},
            "heads": [
                {
                    "name": selected["name"],
                    "development": selected,
                    "cells": cells,
                    "recommendation": "do_not_implement_from_this_experiment",
                }
            ],  # fmt: skip
            "compute": compute,
            "claim_limits": [
                "Calibrated correctness failed in one gate; no serving claim is made.",
                "Cell classification is reported only to show the rule's outcome.",
            ],
        }
    verdict["scope_completed"] = (
        "Conditional DSpark e10 repeat of the EAGLE best setting (adapt-mult-n8k-r0), "
        "including the unchanged paired serving confirmation."
    )
    verdict.pop("conditional_dspark_followup", None)
    verdict["development"] = read(root / "development-report.json")
    immutable_json(root / "verdict.json", verdict)
    with (root / "serving-absolute-cells.csv").open("x", newline="") as handle:
        writer = csv.writer(handle)
        metrics = ["output_tokens_per_second", "tpot_p50_ms", "tpot_p95_ms", "ttft_p50_ms",
                   "ttft_p95_ms", "e2e_p95_ms", "mean_acceptance_length"]  # fmt: skip
        head = ["label", "role", "k", "concurrency", "repeat", "substantial", "verdict"]
        writer.writerow(head + [f"base_{m}" for m in metrics] + [f"adapted_{m}" for m in metrics])
        for cell in verdict["heads"][0]["cells"]:
            for pair in cell["pairs"]:
                writer.writerow(
                    [cell["label"], cell["role"], cell["k"], cell["concurrency"], pair["repeat"],
                     cell["substantial_repeatable_benefit"], cell["verdict"]]
                    + [pair["baseline"][m] for m in metrics]
                    + [pair["adapted"][m] for m in metrics]
                )  # fmt: skip
    print(verdict["verdict"], verdict["heads"][0]["recommendation"])
    if report["correctness_passed"] is False:
        print(
            "correctness gate failed:",
            [(g["k"], g["label"], g["repeat"]) for g in verdict["failed_gates"]],
        )


if __name__ == "__main__":
    main()
