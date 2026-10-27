#!/usr/bin/env python3
"""Validate and pair the completed, unprofiled waste-sum a/b measurements."""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import gzip
import hashlib
import json
import math
import platform
import subprocess
import sys
from pathlib import Path

from apertus_bench.prometheus import parse_speculative_snapshot, speculative_delta
from apertus_bench.waste_report import cell_costs

RUNS = ["waste-sum-a-20261024T133232Z", "waste-sum-b-20261024T133235Z"]
ARMS = ["plain", "dspark-e10-k1", "dspark-e10-k3", "dspark-e10-k7-paper-cut-kv"]
LEVELS = [1, 8, 32, 64, 128, 256]


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[4])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    repo = args.repo.resolve()
    recovery = repo / "results/8b/waste/recovery-20261027"
    manifest = json.loads((recovery / "source-recovery/collection-manifest.json").read_text())
    compressed = json.loads((recovery / "collection-verification.json").read_text())[
        "compressed_source_files"
    ]
    compressed = {r["source_path"]: r["stored_path"] for r in compressed}
    for record in manifest["source_files"] + manifest["recovery_files"]:
        path = repo / record["path"]
        raw = (
            path.read_bytes()
            if path.exists()
            else gzip.decompress((repo / compressed[record["path"]]).read_bytes())
        )
        assert digest(raw) == record["sha256"], path

    rows, validations, source_hashes = [], [], {}
    cells = {}
    for name in RUNS:
        sweep = repo / "results/8b/sweeps" / name
        status = json.loads((sweep / "sweep-status.json").read_text())
        assert status == dict.fromkeys(ARMS, "ok"), status
        costs = {(r["arm"], r["concurrency"]): r for r in cell_costs(sweep, "summarization")}
        assert len(costs) == len(ARMS) * len(LEVELS)
        for arm in ARMS:
            arm_status = json.loads((sweep / arm / "status.json").read_text())
            assert arm_status["status"] == "ok" and all(
                t["exit_code"] == 0 for t in arm_status["tests"].values()
            )
            for concurrency in LEVELS:
                cell = sweep / arm / "summarization/summarization" / f"c{concurrency}"
                summary = json.loads((cell / "summary.json").read_text())
                metadata = json.loads((cell / "metadata.json").read_text())
                requests = [
                    json.loads(line)
                    for line in (cell / "requests.jsonl").read_text().splitlines()
                    if line
                ]
                assert (
                    summary["requests"]["failed"] == 0 and summary["requests"]["success_rate"] == 1
                )
                assert len(requests) == summary["requests"]["successful"]
                assert all(r["success"] and r["status_code"] == 200 for r in requests)
                assert (
                    sum(r["completion_tokens"] for r in requests) == summary["tokens"]["completion"]
                )
                assert summary["warmup"]["failed"] == 0
                before = parse_speculative_snapshot(
                    (cell / "metrics_before.prom").read_text(), metadata["model"]
                )
                after = parse_speculative_snapshot(
                    (cell / "metrics_after.prom").read_text(), metadata["model"]
                )
                delta = speculative_delta(before, after)
                if arm != "plain":
                    assert delta["enabled"] and delta["drafts"] > 0 and delta["draft_tokens"] > 0
                    assert math.isclose(
                        delta["mean_acceptance_length"],
                        summary["speculative_decoding"]["mean_acceptance_length"],
                    )
                assert not costs[arm, concurrency]["profiled"]
                row = dict(costs[arm, concurrency])
                row.update(
                    requests=len(requests),
                    completion_tokens=summary["tokens"]["completion"],
                    mean_completion_tokens=summary["tokens"]["completion"] / len(requests),
                    ttft_p95_ms=summary["latency_ms"]["ttft"]["p95"],
                    tpot_p95_ms=summary["latency_ms"]["tpot"]["p95"],
                    e2e_p95_ms=summary["latency_ms"]["e2e"]["p95"],
                    running_requests_mean=summary["load"]["running"]["mean"],
                    preemptions=summary["load"]["preemptions"],
                )
                rows.append(row)
                cells[name, arm, concurrency] = (row, metadata, [r["prompt_id"] for r in requests])
                validations.append(
                    {
                        "cell": str(cell.relative_to(repo)),
                        "requests": len(requests),
                        "success": True,
                        "counter_delta_valid": True,
                        "recorded_harness_revision": metadata["harness_git_revision"],
                    }
                )
                for path in cell.iterdir():
                    if path.is_file():
                        source_hashes[str(path.relative_to(repo))] = digest(path.read_bytes())
    paired = []
    for arm in ARMS[1:]:
        for c in LEVELS:
            rr = []
            for name in RUNS:
                row, meta, ids = cells[name, arm, c]
                plain, baseline_meta, baseline_ids = cells[name, "plain", c]
                for key in ["workload_file_sha256", "prompt_ids"]:
                    assert meta[key] == baseline_meta[key]
                assert meta["cell"]["generation"] == baseline_meta["cell"]["generation"]
                assert ids == baseline_ids
                rr.append(
                    {
                        "run": name,
                        "throughput_ratio": row["output_tokens_per_second"]
                        / plain["output_tokens_per_second"],
                        "cost_ratio": row["gpu_seconds_per_million_tokens"]
                        / plain["gpu_seconds_per_million_tokens"],
                        "ttft_p95_ratio": row["ttft_p95_ms"] / plain["ttft_p95_ms"],
                        "tpot_p95_ratio": row["tpot_p95_ms"] / plain["tpot_p95_ms"],
                        "mean_completion_tokens_ratio": row["mean_completion_tokens"]
                        / plain["mean_completion_tokens"],
                    }
                )
            ratios = [r["throughput_ratio"] for r in rr]
            paired.append(
                {
                    "arm": arm,
                    "concurrency": c,
                    "throughput_ratio_geomean": math.prod(ratios) ** 0.5,
                    "throughput_ratio_min": min(ratios),
                    "throughput_ratio_max": max(ratios),
                    "replicates": rr,
                }
            )
    report = {
        "runs": RUNS,
        "successful_requests": sum(r["requests"] for r in rows),
        "cells": len(rows),
        "raw_source_files_verified": len(manifest["source_files"])
        + len(manifest["recovery_files"]),
        "absolute": rows,
        "paired": paired,
        "claim_limit": (
            "Two independent node allocations; descriptive paired ratios/ranges, "
            "no confidence interval. Legacy provenance limitations are recorded separately."
        ),
    }
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    (args.output / "validation.json").write_text(json.dumps(validations, indent=2) + "\n")
    (args.output / "input-sha256.json").write_text(json.dumps(source_hashes, indent=2) + "\n")
    for name, records in [("absolute", rows), ("paired", paired)]:
        simple = [{k: v for k, v in r.items() if not isinstance(v, (dict, list))} for r in records]
        with (args.output / f"{name}.csv").open("w") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(simple[0]))
            writer.writeheader()
            writer.writerows(simple)
    provenance = {
        "created_at": dt.datetime.now(dt.UTC).isoformat(),
        "argv": sys.argv,
        "python": platform.python_version(),
        "analysis_git_revision": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=repo, text=True
        ).strip(),
        "analysis_source_sha256": digest(Path(__file__).read_bytes()),
        "cost_module_source_sha256": digest(
            (repo / "src/apertus_bench/waste_report.py").read_bytes()
        ),
        "source_dirty": bool(
            subprocess.check_output(["git", "status", "--porcelain"], cwd=repo, text=True)
        ),
        "provenance_limit": manifest["claim_limit"],
    }
    (args.output / "analysis-provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    print(
        json.dumps(
            {
                "cells": report["cells"],
                "successful_requests": report["successful_requests"],
                "output": str(args.output),
            }
        )
    )


if __name__ == "__main__":
    main()
