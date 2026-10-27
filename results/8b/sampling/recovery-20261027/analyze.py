#!/usr/bin/env python3
"""Validate completed sampling cells and compare only matched plain baselines."""

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
from collections import Counter
from pathlib import Path

from apertus_bench.prometheus import parse_speculative_snapshot, speculative_delta
from apertus_bench.stats import percentile

RUNS = {
    "topk-off-20261022T122308Z": [
        "plain",
        "dspark-off-k3",
        "dspark-off-k7",
        "eagle-k3",
        "eagle-k7",
    ],
    "topk-think-20261022T122311Z": ["plain", "dspark-on-k3", "dspark-on-k5", "dspark-on-k7"],
}


def sha(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def raw_path(path: Path) -> Path:
    return path if path.exists() else path.with_name(path.name + ".gz")


def read_bytes(path: Path) -> bytes:
    path = raw_path(path)
    return gzip.decompress(path.read_bytes()) if path.suffix == ".gz" else path.read_bytes()


def read_requests(path: Path):
    path = raw_path(path)
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def assert_close(a: float, b: float, name: str) -> None:
    if not math.isclose(a, b, rel_tol=1e-9, abs_tol=1e-7):
        raise AssertionError(f"{name}: {a} != {b}")


def setting(label: str) -> str:
    for suffix in ["greedy", "default", "t1-kall", "t1-k50", "t1-k20", "t1-k5"]:
        if label.endswith(suffix):
            return suffix
    raise ValueError(label)


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("x", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[4])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    repo, output = args.repo.resolve(), args.output.resolve()
    if output.exists():
        raise FileExistsError(output)
    recovery = repo / "results/8b/sampling/recovery-20261027"
    collection = json.loads((recovery / "source-recovery/collection-manifest.json").read_text())
    source_verification = []
    for record in (
        collection["source_files"]
        + collection["recovery_files"]
        + collection["derived_source_files"]
    ):
        path = repo / record["path"]
        data = read_bytes(path)
        assert len(data) == record["bytes"]
        assert hashlib.sha256(data).hexdigest() == record["sha256"], path
        source_verification.append(record["path"])
    checkpoint_support = recovery / "source-recovery/checkpoint-support/recovery-manifest.json"
    if checkpoint_support.exists():
        for checkpoint in json.loads(checkpoint_support.read_text())["checkpoints"]:
            for record in checkpoint["files"]:
                path = repo / record["path"]
                assert path.stat().st_size == record["bytes"]
                assert sha(path) == record["sha256"]
                source_verification.append(record["path"])
    prompt_schedules = {}
    for path in sorted((recovery / "source-recovery/prompt-schedules").rglob("*.json")):
        schedule = json.loads(path.read_text())
        assert schedule["matches_recorded_workload"]
        assert schedule["source_sha256"] == schedule["metadata_sha256"]
        prompt_schedules[schedule["source"]] = {row["id"]: row for row in schedule["rows"]}
    rows, cells, identities, validations = [], {}, {}, []
    for run, arms in RUNS.items():
        sweep = repo / "results/8b/sweeps" / run
        status = json.loads((sweep / "sweep-status.json").read_text())
        assert status == dict.fromkeys(arms, "ok")
        identities[run] = {
            "launch_git_revision": (sweep / "git-revision.txt").read_text().strip(),
            "image_receipt": (sweep / "image.sha256").read_text().strip(),
            "job_receipt": (sweep / "job.txt").read_text().strip(),
            "arms": {},
        }
        for arm in arms:
            arm_status = json.loads((sweep / arm / "status.json").read_text())
            assert arm_status["status"] == "ok"
            assert all(test["exit_code"] == 0 for test in arm_status["tests"].values())
            summaries = sorted((sweep / arm).glob("*/*/c*/summary.json"))
            assert len(summaries) == 36, (run, arm, len(summaries))
            first_metadata = json.loads((summaries[0].parent / "metadata.json").read_text())
            extra = first_metadata["extra"]
            identities[run]["arms"][arm] = {
                "deployment_id": extra["deployment_id"],
                "checkpoint_sha256": extra.get("checkpoint_sha256"),
                "target_model": extra.get("target_model"),
                "target_revision": extra.get("target_revision"),
                "target_tensor_parallel_size": extra.get("target_tensor_parallel_size"),
                "serve_argv": arm_status["serve_argv"],
            }
            for path in summaries:
                cell = path.parent
                summary = json.loads(path.read_text())
                meta = json.loads((cell / "metadata.json").read_text())
                for key in ["deployment_id", "checkpoint_sha256", "target_revision"]:
                    assert meta["extra"].get(key) == extra.get(key)
                label, workload, concurrency = (
                    cell.parents[1].name,
                    cell.parent.name,
                    int(cell.name[1:]),
                )
                assert meta["cell"]["concurrency"] == concurrency
                assert meta["cell"]["workload"] == workload
                assert summary["requests"]["failed"] == 0
                assert summary["requests"]["success_rate"] == 1
                assert summary["warmup"]["failed"] == 0
                cap_schedule = prompt_schedules[meta["workload_file"]]
                counts = Counter()
                tokens, prompt_tokens, lengths, tpot, ttft, e2e, schedule = 0, 0, [], [], [], [], []
                caps, at_cap = [], 0
                event_count, event_sum, decode_ms = 0, 0.0, 0.0
                for request in read_requests(cell / "requests.jsonl"):
                    assert request["success"] and request["status_code"] == 200
                    assert (
                        request["completion_tokens"] is not None
                        and request["completion_tokens"] > 0
                    )
                    assert request["output"] is None, (
                        "Unexpected text payload; review redistribution scope"
                    )
                    n = request["completion_tokens"]
                    cap = (
                        meta["cell"]["generation"]["max_tokens"]
                        or cap_schedule[request["prompt_id"]]["max_tokens"]
                    )
                    assert n <= cap
                    if request["finish_reason"] == "length":
                        assert n == cap
                    caps.append(cap)
                    at_cap += n == cap
                    tokens += n
                    prompt_tokens += request["prompt_tokens"]
                    lengths.append(n)
                    counts[request["finish_reason"]] += 1
                    schedule.append((request["request_index"], request["prompt_id"]))
                    e2e.append(request["e2e_ms"])
                    ttft.append(request["ttft_ms"])
                    if request["tpot_ms"] is not None:
                        tpot.append(request["tpot_ms"])
                        assert_close(
                            request["tpot_ms"],
                            (request["e2e_ms"] - request["ttft_ms"]) / (n - 1),
                            "request TPOT",
                        )
                    decode_ms += request["e2e_ms"] - request["ttft_ms"]
                    event_count += len(request["stream_event_gaps_ms"])
                    event_sum += sum(request["stream_event_gaps_ms"])
                attempted = len(lengths)
                assert attempted == summary["requests"]["attempted"]
                assert tokens == summary["tokens"]["completion"]
                assert prompt_tokens == summary["tokens"]["prompt"]
                assert event_count == summary["latency_ms"]["stream_event_gap"]["count"]
                if event_count:
                    assert_close(
                        event_sum / event_count,
                        summary["latency_ms"]["stream_event_gap"]["mean"],
                        "event mean",
                    )
                for kind, values in [("e2e", e2e), ("ttft", ttft), ("tpot", tpot)]:
                    assert_close(
                        sum(values) / len(values),
                        summary["latency_ms"][kind]["mean"],
                        kind + " mean",
                    )
                    for quantile in [50, 95, 99]:
                        assert_close(
                            percentile(values, quantile / 100),
                            summary["latency_ms"][kind][f"p{quantile}"],
                            kind + f" p{quantile}",
                        )
                rate = summary["tokens"]["output_tokens_per_second"]
                assert_close(tokens / summary["wall_seconds"], rate, "throughput")
                before = parse_speculative_snapshot(
                    (cell / "metrics_before.prom").read_text(), meta["model"]
                )
                after = parse_speculative_snapshot(
                    (cell / "metrics_after.prom").read_text(), meta["model"]
                )
                delta = speculative_delta(before, after)
                k = meta["variant"]["num_speculative_tokens"]
                if arm != "plain":
                    assert delta["enabled"] and delta["drafts"] > 0
                    assert_close(delta["draft_tokens"] / delta["drafts"], k, "fixed draft length")
                    for key in [
                        "drafts",
                        "draft_tokens",
                        "accepted_tokens",
                        "acceptance_rate",
                        "mean_acceptance_length",
                    ]:
                        assert_close(delta[key], summary["speculative_decoding"][key], key)
                generation = meta["cell"]["generation"]
                assert generation["seed"] == 1
                row = {
                    "run": run,
                    "mode": "on" if "think" in run else "off",
                    "arm": arm,
                    "method": meta["variant"]["method"],
                    "draft_k": k,
                    "label": label,
                    "workload": workload,
                    "sampling": setting(label),
                    "concurrency": concurrency,
                    "temperature": generation["temperature"],
                    "top_p": generation["top_p"],
                    "sampling_top_k": generation["top_k"],
                    "seed_base": generation["seed"],
                    "requests": attempted,
                    "completion_tokens": tokens,
                    "mean_output_tokens": tokens / attempted,
                    "median_output_tokens": percentile(lengths, 0.5),
                    "p95_output_tokens": percentile(lengths, 0.95),
                    "min_output_tokens": min(lengths),
                    "max_output_tokens": max(lengths),
                    "min_output_cap": min(caps),
                    "max_output_cap": max(caps),
                    "mean_output_cap": sum(caps) / attempted,
                    "at_output_cap_rate": at_cap / attempted,
                    "length_finishes": counts["length"],
                    "length_finish_rate": counts["length"] / attempted,
                    "stop_finishes": counts["stop"],
                    "wall_seconds": summary["wall_seconds"],
                    "output_tok_s": rate,
                    "gpu_s_per_million_output_tokens": 1e6 / rate,
                    "e2e_mean_ms": sum(e2e) / attempted,
                    "e2e_p50_ms": percentile(e2e, 0.5),
                    "e2e_p95_ms": percentile(e2e, 0.95),
                    "ttft_p50_ms": percentile(ttft, 0.5),
                    "ttft_p95_ms": percentile(ttft, 0.95),
                    "tpot_p50_ms": percentile(tpot, 0.5),
                    "tpot_p95_ms": percentile(tpot, 0.95),
                    "token_weighted_decode_ms": decode_ms / max(tokens - attempted, 1),
                    "stream_gap_p50_ms": summary["latency_ms"]["stream_event_gap"]["p50"],
                    "stream_gap_p95_ms": summary["latency_ms"]["stream_event_gap"]["p95"],
                    "accepted_per_step": delta.get("mean_acceptance_length")
                    if arm != "plain"
                    else None,
                    "acceptance_rate": delta.get("acceptance_rate") if arm != "plain" else None,
                    "running_mean": summary["load"]["running"]["mean"],
                    "running_max": summary["load"]["running"]["max"],
                    "preemptions": summary["load"]["preemptions"],
                    "kv_usage_max": summary["load"]["kv_cache_usage"]["max"],
                    "cell_path": str(cell.relative_to(repo)),
                }
                assert counts["length"] + counts["stop"] == attempted, counts
                rows.append(row)
                cells[run, arm, label, concurrency] = (row, meta, schedule)
                validations.append(
                    {
                        "cell": row["cell_path"],
                        "requests": attempted,
                        "counter_decrease": False,
                        "seed_base": generation["seed"],
                        "harness_revision_metadata": meta["harness_git_revision"],
                    }
                )
    pairs = []
    for key, (row, meta, schedule) in cells.items():
        run, arm, label, c = key
        if arm == "plain":
            continue
        baseline, bm, bs = cells[run, "plain", label, c]
        assert meta["cell"]["generation"] == bm["cell"]["generation"]
        assert meta["workload_file_sha256"] == bm["workload_file_sha256"]
        assert meta["prompt_ids"] == bm["prompt_ids"]
        assert schedule == bs
        assert json.loads(meta["extra"]["engine"]) == json.loads(bm["extra"]["engine"])
        pair = {
            k: row[k]
            for k in [
                "run",
                "mode",
                "arm",
                "draft_k",
                "label",
                "workload",
                "sampling",
                "concurrency",
            ]
        }
        for key in [
            "output_tok_s",
            "gpu_s_per_million_output_tokens",
            "mean_output_tokens",
            "e2e_mean_ms",
            "tpot_p50_ms",
            "ttft_p95_ms",
            "stream_gap_p95_ms",
            "length_finish_rate",
        ]:
            pair["plain_" + key] = baseline[key]
            pair["candidate_" + key] = row[key]
        pair["throughput_ratio"] = row["output_tok_s"] / baseline["output_tok_s"]
        pair["e2e_latency_speedup"] = baseline["e2e_mean_ms"] / row["e2e_mean_ms"]
        pair["tpot_speedup"] = baseline["tpot_p50_ms"] / row["tpot_p50_ms"]
        pair["output_length_ratio"] = row["mean_output_tokens"] / baseline["mean_output_tokens"]
        pair["accepted_per_step"] = row["accepted_per_step"]
        pairs.append(pair)
    output.mkdir(parents=True)
    write_csv(output / "absolute.csv", rows)
    write_csv(output / "matched.csv", pairs)
    (output / "validation.json").write_text(
        json.dumps(
            {
                "source_files_verified": len(source_verification),
                "cells": len(rows),
                "requests": sum(r["requests"] for r in rows),
                "failed_requests": 0,
                "snapshot_counter_decreases": 0,
                "validations": validations,
                "identities": identities,
            },
            indent=2,
        )
        + "\n"
    )
    (output / "summary.json").write_text(
        json.dumps(
            {
                "absolute": rows,
                "matched": pairs,
                "claim_limits": [
                    "One deployment per arm/mode; no deployment-repeat confidence intervals.",
                    "Recorded seed base 1 and launch client source imply per-request seed "
                    "1 + request_index, contrary to YAML comments; no wire capture exists.",
                    "Sampling changes output distributions; no quality benchmark was run.",
                    "TPOT is per-request average decode time per token; "
                    "stream gaps measure multi-token event cadence.",
                    "Original stale harness Git metadata remains unmodified; "
                    "launch receipts are separate.",
                ],
            },
            indent=2,
        )
        + "\n"
    )
    (output / "analysis-provenance.json").write_text(
        json.dumps(
            {
                "created_at": dt.datetime.now(dt.UTC).isoformat(),
                "argv": sys.argv,
                "python": platform.python_version(),
                "git_revision": subprocess.check_output(
                    ["git", "-C", str(repo), "rev-parse", "HEAD"], text=True
                ).strip(),
                "analysis_source_sha256": sha(Path(__file__)),
                "manifest_sha256": sha(recovery / "source-recovery/collection-manifest.json"),
                "analysis_library_sha256": {
                    str(path.relative_to(repo)): sha(path)
                    for path in [
                        repo / "src/apertus_bench/prometheus.py",
                        repo / "src/apertus_bench/stats.py",
                    ]
                },
                "validation": (
                    "Original request counts/tokens/latencies and speculative snapshots "
                    "recomputed; matching requires exact generation, corpus, request schedule "
                    "and engine settings."
                ),
            },
            indent=2,
        )
        + "\n"
    )
    print(
        json.dumps(
            {
                "cells": len(rows),
                "requests": sum(r["requests"] for r in rows),
                "matched_pairs": len(pairs),
                "output": str(output),
            }
        )
    )


if __name__ == "__main__":
    main()
