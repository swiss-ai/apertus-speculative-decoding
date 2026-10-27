#!/usr/bin/env python3
"""Analyze recovered historical cells without editing original measurements."""

from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path

from apertus_bench.layout_recovery import REPO, collect_rows

RUNS = [
    "dp4-a-20261023T084033Z",
    "dp4-b-20261023T122249Z",
    "dp4-c-20261023T182609Z",
    "tp4-a-20261023T093209Z",
    "tp4-a2-20261023T141405Z",
    "tp4-b1-20261024T132121Z",
    "tp4-b2-20261024T132206Z",
    "tp4-c1-20261024T145347Z",
    "tp4-c2-20261024T160931Z",
    "think-layouts-20261022T122305Z",
    "think-pd-20261022T122307Z",
]


def verify_usage(row):
    p = Path(row["summary"]).with_name("requests.jsonl")
    if not p.exists():
        p = p.with_suffix(".jsonl.gz")
    if not p.exists():
        return {"passed": False, "error": "Missing raw requests"}
    opener = gzip.open if p.suffix == ".gz" else Path.open
    attempted = successful = missing_usage = completion = 0
    with opener(p, "rt") as stream:
        for line in stream:
            r = json.loads(line)
            attempted += 1
            if r.get("success"):
                successful += 1
                if r.get("completion_tokens") is None or r.get("prompt_tokens") is None:
                    missing_usage += 1
                completion += r.get("completion_tokens") or 0
    return {
        "passed": missing_usage == 0
        and completion == row["completion_tokens"]
        and successful == row["successful"]
        and attempted == row["successful"] + row["failed"],
        "attempted": attempted,
        "successful": successful,
        "missing_usage": missing_usage,
        "completion_tokens": completion,
        "source": str(p),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--verify-requests", action="store_true")
    a = ap.parse_args()
    rows = []
    runs = []
    for name in RUNS:
        root = REPO / "results/8b/sweeps" / name
        statuses = root / "sweep-status.json"
        runs.append(
            {
                "run": str(root),
                "sweep_status": json.loads(statuses.read_text()) if statuses.exists() else None,
                "partial_run": not statuses.exists(),
            }
        )
        rows.extend(collect_rows(root, "historical-" + name))
    for row in rows:
        row["request_gate_passed"] = row["successful"] / (row["successful"] + row["failed"]) >= 0.99
        if a.verify_requests:
            row["usage_validation"] = verify_usage(row)
        row["screening_cell_eligible"] = bool(
            row["quality"]["passed"]
            and row["request_gate_passed"]
            and row["matched_generation_and_corpus"]
            and row.get("usage_validation", {}).get("passed", True)
        )
    report = {
        "scope": "Original completed/partial layout runs only; immutable source cells",
        "runs": runs,
        "rows": rows,
        "cells": len(rows),
        "quality_passes": sum(r["quality"]["passed"] for r in rows),
        "screening_eligible": sum(r["screening_cell_eligible"] for r in rows),
        "deployment_repeats": "Generally one per setting; no deployment-level confidence interval",
        "recommendation_gate": (
            "Only substantial repeatable gain; no promotion of minute improvements"
        ),
        "claim_limit": (
            "Screening only; no per-layout calibrated correctness proof. "
            "Failed or partial arm cells do not complete missing strata."
        ),
    }
    a.output.parent.mkdir(parents=True, exist_ok=True)
    with a.output.open("x") as f:
        json.dump(report, f, indent=2)
        f.write("\n")
    print(json.dumps({k: report[k] for k in ["cells", "quality_passes", "screening_eligible"]}))


if __name__ == "__main__":
    main()
