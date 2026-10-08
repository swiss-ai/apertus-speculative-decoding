"""Tables from node-sweep results (src/apertus_bench/node_sweep.py).

    python3 -m apertus_bench.sweep_report results/8b/sweeps/A results/8b/sweeps/B \\
        --label summarization --reference plain-a [--metric output_tokens_per_second]

Collects every arm's `<label>/loadtest-summary.json` from the given sweep
directories and prints one markdown table per metric: concurrency rows, arm
columns, and (with --reference) each arm's ratio to the reference arm.
`--probe` prints the probe cell of every arm instead.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

METRICS = {
    "output_tokens_per_second": ("output tok/s", "{:,.0f}"),
    "mean_acceptance_length": ("accepted length", "{:.2f}"),
    "tpot_p50_ms": ("TPOT p50 (ms)", "{:.1f}"),
    "ttft_p50_ms": ("TTFT p50 (ms)", "{:,.0f}"),
    "running_max": ("running max", "{:.0f}"),
    "preemptions": ("preemptions", "{:.0f}"),
    "kv_cache_usage_max": ("KV use max", "{:.0%}"),
}


def collect(sweeps: list[Path], label: str) -> dict[str, dict[int, dict[str, Any]]]:
    """arm -> concurrency -> level row; a later sweep overrides an arm of the same name."""
    arms: dict[str, dict[int, dict[str, Any]]] = {}
    for sweep in sweeps:
        for summary in sorted(sweep.glob(f"*/{label}/loadtest-summary.json")):
            report = json.loads(summary.read_text())
            arm = summary.parent.parent.name
            arms[arm] = {int(level["concurrency"]): level for level in report["levels"]}
    return arms


def collect_probe(sweeps: list[Path]) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    for sweep in sweeps:
        for summary in sorted(sweep.glob("*/probe/summary.json")):
            data = json.loads(summary.read_text())
            spec = data.get("speculative_decoding") or {}
            rows[summary.parents[1].name] = {
                "output_tokens_per_second": data["tokens"]["output_tokens_per_second"],
                "mean_acceptance_length": spec.get("mean_acceptance_length"),
                "tpot_p50_ms": (data["latency_ms"].get("tpot") or {}).get("p50"),
            }
    return rows


def _cell(value: Any, fmt: str) -> str:
    if value is None:
        return "-"
    return fmt.format(value)


def table(
    arms: dict[str, dict[int, dict[str, Any]]],
    metric: str,
    order: list[str] | None = None,
    reference: str | None = None,
) -> str:
    title, fmt = METRICS[metric]
    names = order or sorted(arms)
    names = [name for name in names if name in arms]
    levels = sorted({c for rows in arms.values() for c in rows})
    header = ["C", *names]
    if reference and reference in arms:
        header += [f"{name} / {reference}" for name in names if name != reference]
    lines = [
        f"{title}:",
        "",
        "| " + " | ".join(header) + " |",
        "|" + " --- |" * len(header),
    ]
    for c in levels:
        cells = [str(c)] + [_cell(arms[name].get(c, {}).get(metric), fmt) for name in names]
        if reference and reference in arms:
            base = arms[reference].get(c, {}).get(metric)
            for name in names:
                if name == reference:
                    continue
                value = arms[name].get(c, {}).get(metric)
                cells.append(f"{value / base:.2f}x" if value is not None and base else "-")
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="apertus_bench.sweep_report")
    parser.add_argument("sweeps", type=Path, nargs="+")
    parser.add_argument("--label", default="summarization")
    parser.add_argument("--reference")
    parser.add_argument("--arms", nargs="+", help="column order (default: sorted)")
    parser.add_argument("--metric", action="append", choices=sorted(METRICS))
    parser.add_argument("--probe", action="store_true")
    args = parser.parse_args(argv)
    if args.probe:
        rows = collect_probe(args.sweeps)
        print("| arm | output tok/s | accepted length | TPOT p50 (ms) |")
        print("| --- | --- | --- | --- |")
        for name in args.arms or sorted(rows):
            row = rows.get(name)
            if row:
                print(
                    f"| {name} | {_cell(row['output_tokens_per_second'], '{:,.0f}')} | "
                    f"{_cell(row['mean_acceptance_length'], '{:.2f}')} | "
                    f"{_cell(row['tpot_p50_ms'], '{:.2f}')} |"
                )
        return
    arms = collect(args.sweeps, args.label)
    for metric in args.metric or ["output_tokens_per_second", "mean_acceptance_length"]:
        reference = args.reference if metric == "output_tokens_per_second" else None
        print(table(arms, metric, args.arms, reference))
        print()


if __name__ == "__main__":
    main()
