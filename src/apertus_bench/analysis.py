from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


def _nested(record: dict[str, Any], *keys: str) -> Any:
    value: Any = record
    for key in keys:
        value = value[key]
    return value


def collect_rows(results_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for summary_path in sorted(results_dir.rglob("summary.json")):
        metadata_path = summary_path.with_name("metadata.json")
        if not metadata_path.exists():
            continue
        summary = json.loads(summary_path.read_text())
        metadata = json.loads(metadata_path.read_text())
        variant = metadata["variant"]
        cell = metadata["cell"]
        spec = summary["speculative_decoding"]
        rows.append(
            {
                "path": str(summary_path.parent),
                "variant": variant["name"],
                "method": variant["method"],
                "num_speculative_tokens": variant.get("num_speculative_tokens"),
                "draft_tensor_parallel_size": variant.get("draft_tensor_parallel_size"),
                "prompt_lookup_max": variant.get("prompt_lookup_max"),
                "workload": cell["workload"],
                "concurrency": cell["concurrency"],
                "repeat": cell["repeat"],
                "success_rate": _nested(summary, "requests", "success_rate"),
                "output_tokens_per_second": _nested(summary, "tokens", "output_tokens_per_second"),
                "ttft_p50_ms": _nested(summary, "latency_ms", "ttft", "p50"),
                "ttft_p95_ms": _nested(summary, "latency_ms", "ttft", "p95"),
                "tpot_p50_ms": _nested(summary, "latency_ms", "tpot", "p50"),
                "tpot_p95_ms": _nested(summary, "latency_ms", "tpot", "p95"),
                "acceptance_rate": spec.get("acceptance_rate"),
                "mean_acceptance_length": spec.get("mean_acceptance_length"),
            }
        )
    return rows


def add_speedups(rows: list[dict[str, Any]]) -> None:
    baseline: dict[tuple[str, int], list[float]] = defaultdict(list)
    for row in rows:
        if row["method"] == "none" and row["output_tokens_per_second"] is not None:
            baseline[(row["workload"], row["concurrency"])].append(
                float(row["output_tokens_per_second"])
            )
    baseline_means = {key: sum(values) / len(values) for key, values in baseline.items()}
    for row in rows:
        baseline_rate = baseline_means.get((row["workload"], row["concurrency"]))
        rate = row["output_tokens_per_second"]
        row["speedup_vs_baseline"] = (
            float(rate) / baseline_rate if rate is not None and baseline_rate else None
        )


def write_csv(rows: list[dict[str, Any]], output_path: Path) -> None:
    if not rows:
        raise ValueError("no result cells found")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
