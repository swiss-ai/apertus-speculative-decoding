"""Tables and crossover chart of a pd-compute sweep (docs/pd-decode-compute-bound.md).

    python3 experiments/pd-decode-compute-bound/report.py SWEEP_DIR OUT_DIR

Per arm and concurrency: client throughput and TPOT, acceptance, the serving
instance's (P/D: the decode instance's) engine steps, verify width, vLLM's
analytic FLOP and byte counters (--enable-mfu-metrics) as a share of GH200
peak, and the nsys GPU metrics of that instance's GPU over the cell's steady
part. Speedup is k>0 over plain of the same layout (P/D or colocated); the
crossover is the first concurrency at which a k stops beating plain, linearly
interpolated in log2(C).
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Any

# GH200 (GH100, 132 SMs at 1.98 GHz, HBM3): dense BF16 tensor peak and HBM bandwidth.
PEAK_FLOPS = 989e12
PEAK_BYTES = 4.02e12  # nsys TARGET_INFO_GPU.memoryBandwidth
COUNTERS = {
    "steps": "vllm:iteration_tokens_total_count",
    "iteration_tokens": "vllm:iteration_tokens_total_sum",
    "generation_tokens": "vllm:generation_tokens_total",
    "prompt_tokens": "vllm:prompt_tokens_total",
    "drafts": "vllm:spec_decode_num_drafts_total",
    "draft_tokens": "vllm:spec_decode_num_draft_tokens_total",
    "accepted": "vllm:spec_decode_num_accepted_tokens_total",
    "flops": "vllm:estimated_flops_per_gpu_total",
    "read_bytes": "vllm:estimated_read_bytes_per_gpu_total",
    "write_bytes": "vllm:estimated_write_bytes_per_gpu_total",
}
GPU_METRICS = {
    "sm_active": "SMs Active [Throughput %]",
    "tensor_active": "Tensor Active [Throughput %]",
    "dram_read": "DRAM Read Bandwidth [Throughput %]",
    "dram_write": "DRAM Write Bandwidth [Throughput %]",
}


def _counter_totals(path: Path) -> dict[str, float]:
    totals: dict[str, float] = {}
    if not path.exists():
        return totals
    wanted = {metric: key for key, metric in COUNTERS.items()}
    for line in path.read_text().splitlines():
        if not line or line.startswith("#"):
            continue
        name = line.split("{", 1)[0].split(" ", 1)[0]
        if name in wanted:
            key = wanted[name]
            totals[key] = totals.get(key, 0.0) + float(line.rsplit(" ", 1)[1])
    return totals


def _k(arm: str) -> int:
    return int(arm.rsplit("-k", 1)[1]) if "-k" in arm else 0


def cell_row(arm: str, cell: Path, gpu: dict[str, Any] | None) -> dict[str, Any]:
    summary = json.loads((cell / "summary.json").read_text())
    before = _counter_totals(cell / "metrics_before.prom")
    after = _counter_totals(cell / "metrics_after.prom")
    delta = {key: after.get(key, 0.0) - before.get(key, 0.0) for key in COUNTERS}
    wall = summary["wall_seconds"]
    k = _k(arm)
    spec = summary.get("speculative_decoding") or {}
    steps = delta["steps"] or float("nan")
    # Target-forward decode width: every drafted request-step verifies k+1 tokens.
    verify = delta["drafts"] * (k + 1) if k else delta["generation_tokens"]
    row = {
        "arm": arm,
        "layout": "pd" if arm.startswith("pd-") else "colo",
        "k": k,
        "concurrency": int(cell.name[1:]),
        "successful": summary["requests"]["successful"],
        "failed": summary["requests"]["failed"],
        "tok_s": summary["tokens"]["output_tokens_per_second"],
        "tpot_p50_ms": summary["latency_ms"]["tpot"]["p50"],
        "ttft_p50_ms": summary["latency_ms"]["ttft"]["p50"],
        "acceptance_length": spec.get("mean_acceptance_length") if k else 1.0,
        "running_mean": (summary.get("load") or {}).get("running", {}).get("mean"),
        "preemptions": (summary.get("load") or {}).get("preemptions"),
        "wall_s": wall,
        "steps": delta["steps"],
        "step_ms": 1000 * wall / steps,
        "verify_tokens_per_step": verify / steps,
        "prompt_tokens_per_step": delta["prompt_tokens"] / steps,
        "flops_pct_peak": 100 * delta["flops"] / wall / PEAK_FLOPS,
        "bytes_pct_peak": 100 * (delta["read_bytes"] + delta["write_bytes"]) / wall / PEAK_BYTES,
        "arith_intensity": (
            delta["flops"] / (delta["read_bytes"] + delta["write_bytes"])
            if delta["read_bytes"]
            else None
        ),
    }
    if gpu:
        steady = gpu["steady"] if gpu["steady"]["seconds"] >= 3 else gpu["window"]
        row["gpu_seconds"] = steady["seconds"]
        for key, name in GPU_METRICS.items():
            row[key] = steady["metrics"].get(name)
    return row


def collect(sweep: Path) -> list[dict[str, Any]]:
    gpu_cells: dict[str, dict[str, Any]] = {}
    cells_path = sweep / "gpu-metrics" / "cells.json"
    if cells_path.exists():
        for record in json.loads(cells_path.read_text()):
            role = next(
                (
                    v
                    for key, v in record["gpus"].items()
                    if key.split(":")[0] in {"decode", "serve"}
                ),
                None,
            )
            gpu_cells[record["cell"]] = role
    rows = []
    for status_path in sorted(sweep.glob("*/status.json")):
        arm = status_path.parent.name
        for summary in sorted(status_path.parent.glob("*/*/c*/summary.json")):
            cell = summary.parent
            rows.append(cell_row(arm, cell, gpu_cells.get(str(cell.relative_to(sweep)))))
    plain = {
        (row["arm"].replace("-again", ""), row["concurrency"]): row["tok_s"]
        for row in rows if row["k"] == 0 and not row["arm"].endswith("-again")
    }  # fmt: skip
    for row in rows:
        base = plain.get((f"{row['layout']}-plain", row["concurrency"]))
        row["speedup_vs_plain"] = row["tok_s"] / base if base else None
    return rows


def crossovers(rows: list[dict[str, Any]]) -> dict[str, float | None]:
    """First C at which speedup <= 1, interpolated in log2(C); None if never."""
    result: dict[str, float | None] = {}
    arms = sorted({row["arm"] for row in rows if row["k"] > 0})
    for arm in arms:
        points = sorted(
            (row["concurrency"], row["speedup_vs_plain"])
            for row in rows
            if row["arm"] == arm and row["speedup_vs_plain"]
        )
        result[arm] = None
        for (c0, s0), (c1, s1) in zip(points, points[1:], strict=False):
            if s0 > 1 >= s1:
                x0, x1 = math.log2(c0), math.log2(c1)
                result[arm] = round(2 ** (x0 + (s0 - 1) * (x1 - x0) / (s0 - s1)))
                break
        if points and points[0][1] <= 1:
            result[arm] = points[0][0]
    return result


def chart(rows: list[dict[str, Any]], path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = {1: "#2a78d6", 3: "#eb6834", 7: "#1baf7a"}
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6), facecolor="#fcfcfb")
    for ax in axes:
        ax.set_facecolor("#fcfcfb")
        ax.grid(True, color="#e4e3df", linewidth=0.6)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.set_xscale("log", base=2)
        ax.set_xlabel("concurrency C (requests in flight)", color="#52514e")
    for ax, layout, title in (
        (axes[0], "colo", "Colocated (1 GPU)"),
        (axes[1], "pd", "P/D (1 prefill + 1 decode GPU)"),
    ):
        ax.axhline(1.0, color="#52514e", linewidth=1, linestyle="--")
        for k, color in colors.items():
            points = sorted(
                (r["concurrency"], r["speedup_vs_plain"])
                for r in rows
                if r["layout"] == layout and r["k"] == k and r["speedup_vs_plain"]
            )
            if not points:
                continue
            xs, ys = zip(*points, strict=True)
            ax.plot(
                xs, ys, color=color, linewidth=2, marker="o", markersize=5, label=f"DSpark k={k}"
            )
            ax.annotate(f"k={k}", (xs[-1], ys[-1]), xytext=(6, 0), textcoords="offset points",
                        color="#0b0b0b", fontsize=9, va="center")  # fmt: skip
        ax.set_title(f"{title}: throughput vs plain", color="#0b0b0b", fontsize=11, loc="left")
        ax.set_ylabel("output tok/s ÷ plain (same layout)", color="#52514e")
        ax.legend(frameon=False, fontsize=9)
    ax = axes[2]
    for k, color in {0: "#52514e", **colors}.items():
        points = sorted(
            (r["verify_tokens_per_step"], r.get("tensor_active"))
            for r in rows
            if r["layout"] == "pd" and r["k"] == k and r.get("tensor_active") is not None
        )
        if points:
            xs, ys = zip(*points, strict=True)
            label = "plain" if k == 0 else f"DSpark k={k}"
            ax.plot(xs, ys, color=color, linewidth=2, marker="o", markersize=5, label=label)
    ax.set_title("P/D decode GPU: Tensor Active vs verify width", color="#0b0b0b",
                 fontsize=11, loc="left")  # fmt: skip
    ax.set_xlabel("target tokens per decode step (requests × (k+1))", color="#52514e")
    ax.set_ylabel("Tensor Active, % (nsys, steady part)", color="#52514e")
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=150)


def markdown(rows: list[dict[str, Any]], cross: dict[str, float | None]) -> str:
    def fmt(value: Any, digits: int = 1) -> str:
        if value is None or (isinstance(value, float) and math.isnan(value)):
            return "–"
        return f"{value:.{digits}f}" if isinstance(value, float) else str(value)

    lines = [
        "| arm | C | ok/fail | tok/s | ×plain | TPOT p50 ms | acc len | running | step ms "
        "| verify tok/step | FLOP %peak | bytes %peak | FLOP/B | SM act % | tensor act % "
        "| DRAM rd % |",
        "|" + " --- |" * 16,
    ]
    for r in sorted(rows, key=lambda r: (r["layout"], r["arm"], r["concurrency"])):
        lines.append(
            f"| {r['arm']} | {r['concurrency']} | {r['successful']}/{r['failed']} "
            f"| {fmt(r['tok_s'], 0)} | {fmt(r['speedup_vs_plain'], 2)} | {fmt(r['tpot_p50_ms'])} "
            f"| {fmt(r['acceptance_length'], 2)} | {fmt(r['running_mean'])} | {fmt(r['step_ms'])} "
            f"| {fmt(r['verify_tokens_per_step'], 0)} | {fmt(r['flops_pct_peak'])} "
            f"| {fmt(r['bytes_pct_peak'])} | {fmt(r['arith_intensity'], 0)} "
            f"| {fmt(r.get('sm_active'))} | {fmt(r.get('tensor_active'))} "
            f"| {fmt(r.get('dram_read'))} |"
        )
    lines += ["", "Crossover (first C where k>0 no longer beats plain of the same layout):", ""]
    lines += [f"- {arm}: {value if value else 'not reached up to C=512'}" for arm, value in
              sorted(cross.items())]  # fmt: skip
    return "\n".join(lines) + "\n"


def main() -> None:
    sweep, out = Path(sys.argv[1]), Path(sys.argv[2])
    out.mkdir(parents=True, exist_ok=True)
    rows = collect(sweep)
    cross = crossovers(rows)
    (out / "cells.json").write_text(json.dumps(rows, indent=1) + "\n")
    (out / "crossover.json").write_text(json.dumps(cross, indent=1) + "\n")
    (out / "report.md").write_text(markdown(rows, cross))
    chart(rows, out / "crossover.png")
    print(markdown(rows, cross))


if __name__ == "__main__":
    main()
