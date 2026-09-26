"""A6: confirmation report over independent deployments per arm.

Speedups use the A5 same-block pairing (``apertus_eagle.a5_summary``): each
EAGLE cell is divided by the plain deployment of its own block. On top of that
this adds what plan section A6 asks to be reported: every block's effect per
(stratum, concurrency) cell, p95 ratios, output lengths and finish reasons,
acceptance by position, the served memory and KV capacity each deployment
logged at startup, and greedy output agreement per prompt.

Output agreement compares ``output_sha256`` for the same ``prompt_id``. EAGLE
against the plain deployment of the same block is the treatment; plain against
plain across blocks is the control, since independently launched deployments
do not reproduce each other's greedy tokens exactly (bf16 near-ties).
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path
from typing import Any

from apertus_eagle.a5_summary import geomean, load_cells, summarize

ENGINE_PATTERNS = {
    "model_load_gib": re.compile(r"Model loading took ([0-9.]+) GiB"),
    "kv_cache_gib": re.compile(r"Available KV cache memory: ([0-9.]+) GiB"),
    "kv_cache_tokens": re.compile(r"GPU KV cache size: ([0-9,]+) tokens"),
}


def deployment_resources(root: Path) -> dict[str, dict[str, Any]]:
    """Startup memory and KV capacity per deployment directory."""
    out = {}
    for excerpt in sorted(root.glob("*/engine-excerpt.txt")):
        text = excerpt.read_text(errors="replace")
        row: dict[str, Any] = {}
        for key, pattern in ENGINE_PATTERNS.items():
            match = pattern.search(text)
            if match:
                value = match.group(1).replace(",", "")
                row[key] = int(value) if key == "kv_cache_tokens" else float(value)
        status = excerpt.with_name("status.txt")
        row["status"] = status.read_text().split() if status.is_file() else None
        out[excerpt.parent.name] = row
    return out


def load_requests(root: Path) -> list[dict[str, Any]]:
    """One row per measured request, tagged with its cell's arm and block."""
    rows = []
    for requests_path in sorted(root.rglob("requests.jsonl")):
        metadata_path = requests_path.with_name("metadata.json")
        if not metadata_path.is_file():
            continue
        metadata = json.loads(metadata_path.read_text())
        extra = metadata.get("extra") or {}
        tag = {
            "variant": metadata["variant"]["name"],
            "method": metadata["variant"]["method"],
            "depth": metadata["variant"].get("num_speculative_tokens"),
            "block": extra.get("block_id"),
            "deployment": requests_path.relative_to(root).parts[0],
            "workload": metadata["cell"]["workload"],
            "concurrency": metadata["cell"]["concurrency"],
        }
        for line in requests_path.read_text().splitlines():
            if line.strip():
                request = json.loads(line)
                rows.append({**tag, **request})
    return rows


def arm_key(row: dict[str, Any]) -> str:
    return "baseline" if row["method"] == "none" else f"k{row['depth']}"


def output_stats(requests: list[dict[str, Any]]) -> dict[str, Any]:
    """Output length and finish reasons per arm, stratum and concurrency."""
    grouped: dict[tuple, list[dict[str, Any]]] = defaultdict(list)
    for r in requests:
        grouped[(arm_key(r), r["workload"], r["concurrency"])].append(r)
    out = {}
    for (arm, workload, conc), items in sorted(grouped.items()):
        lengths = [r["completion_tokens"] for r in items if r.get("success")]
        out[f"{arm}/{workload}/c{conc}"] = {
            "requests": len(items),
            "successful": len(lengths),
            "completion_tokens_mean": statistics.fmean(lengths) if lengths else None,
            "completion_tokens_p50": statistics.median(lengths) if lengths else None,
            "completion_tokens_max": max(lengths) if lengths else None,
            "finish_reasons": dict(Counter(str(r.get("finish_reason")) for r in items)),
        }
    return out


def agreement(requests: list[dict[str, Any]]) -> dict[str, Any]:
    """Share of prompts whose greedy output hash matches, treatment and control."""
    outputs: dict[tuple, dict[str, str]] = defaultdict(dict)
    for r in requests:
        if r.get("success") and r.get("output_sha256"):
            key = (arm_key(r), r["block"], r["workload"], r["concurrency"])
            outputs[key][r["prompt_id"]] = r["output_sha256"]

    def share(a: dict[str, str], b: dict[str, str]) -> tuple[int, int]:
        common = a.keys() & b.keys()
        return sum(a[p] == b[p] for p in common), len(common)

    cells: dict[tuple, list[tuple[int, int]]] = defaultdict(list)
    for (arm, block, workload, conc), mine in outputs.items():
        if arm == "baseline":
            continue
        base = outputs.get(("baseline", block, workload, conc))
        if base:
            cells[(f"{arm} vs baseline, same block", workload, conc)].append(share(mine, base))
    baseline_blocks = sorted({k[1] for k in outputs if k[0] == "baseline"})
    for first, second in combinations(baseline_blocks, 2):
        for (arm, block, workload, conc), mine in outputs.items():
            if arm == "baseline" and block == first:
                other = outputs.get(("baseline", second, workload, conc))
                if other:
                    label = "baseline vs baseline, across blocks"
                    cells[(label, workload, conc)].append(share(mine, other))
    out = {}
    for (label, workload, conc), pairs in sorted(cells.items()):
        same = sum(p[0] for p in pairs)
        total = sum(p[1] for p in pairs)
        out[f"{label}/{workload}/c{conc}"] = {
            "identical": same,
            "compared": total,
            "share": same / total if total else None,
        }
    return out


def per_cell_effects(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Every block's TPOT speedup per (depth, stratum, concurrency), plus summaries."""
    grouped: dict[tuple, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(row["depth"], row["workload"], row["concurrency"])].append(row)
    out = {}
    for (depth, workload, conc), items in sorted(grouped.items()):
        items = sorted(items, key=lambda r: r["block"])
        speedups = [r["tpot_speedup"] for r in items]
        out[f"k{depth}/{workload}/c{conc}"] = {
            "blocks": {r["block"]: round(r["tpot_speedup"], 4) for r in items},
            "geomean": geomean(speedups),
            "min": min(speedups),
            "max": max(speedups),
            "throughput_speedup_geomean": geomean([r["throughput_speedup"] for r in items]),
            "ttft_p50_ratio_geomean": geomean([r["ttft_p50_ratio"] for r in items]),
            "e2e_p50_ratio_geomean": geomean([r["e2e_p50_ratio"] for r in items]),
            "acceptance_rate_mean": statistics.fmean(
                [r["acceptance_rate"] for r in items if r["acceptance_rate"] is not None]
            ),
            "g_mean": statistics.fmean([r["g"] for r in items if r["g"] is not None]),
            "acceptance_per_position_mean": position_means(items),
        }
    return out


def position_means(items: list[dict[str, Any]]) -> list[float]:
    """Mean acceptance rate at each draft position (vLLM reports {"0": rate, ...})."""
    per_cell = [r["acceptance_per_position"] for r in items if r["acceptance_per_position"]]
    if not per_cell:
        return []
    positions = sorted(per_cell[0], key=int)
    return [statistics.fmean(float(cell[p]) for cell in per_cell) for p in positions]


def per_stratum(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Geometric-mean TPOT speedup per stratum over both concurrencies and all blocks."""
    grouped: dict[tuple, list[float]] = defaultdict(list)
    for row in rows:
        grouped[(row["depth"], row["workload"])].append(row["tpot_speedup"])
    return {f"k{d}/{w}": geomean(v) for (d, w), v in sorted(grouped.items())}


def tail_ratios(cells: list[dict[str, Any]], rows: list[dict[str, Any]]) -> dict[str, Any]:
    """p95 ratios of each EAGLE cell to its same-block baseline cell."""
    baseline = {
        (c["block"], c["workload"], c["concurrency"]): c for c in cells if c["method"] == "none"
    }
    out = {}
    for r in rows:
        base = baseline[(r["block"], r["workload"], r["concurrency"])]
        out[f"k{r['depth']}/{r['workload']}/c{r['concurrency']}/{r['block']}"] = {
            "tpot_p95_ratio": r["tpot_p95"] / base["tpot_p95"],
            "ttft_p95_ratio": r["ttft_p95"] / base["ttft_p95"],
            "e2e_p95_ratio": (r["e2e_p95"] / base["e2e_p95"])
            if r["e2e_p95"] and base["e2e_p95"]
            else None,
        }
    return out


def build_report(root: Path) -> dict[str, Any]:
    cells = load_cells(root)
    summary = summarize(cells)
    requests = load_requests(root)
    return {
        "root": str(root),
        "deployments": deployment_resources(root),
        "by_depth": summary["by_depth"],
        "ranking": summary["ranking"],
        "per_stratum": per_stratum(summary["rows"]),
        "per_cell": per_cell_effects(summary["rows"]),
        "latency_p95": tail_ratios(cells, summary["rows"]),
        "outputs": output_stats(requests),
        "greedy_agreement": agreement(requests),
        "min_success_rate": min(c["success_rate"] for c in cells) if cells else None,
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="apertus-eagle-a6-report")
    parser.add_argument("root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    report = build_report(args.root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    for depth in report["ranking"]:
        d = report["by_depth"][depth]
        blocks = ", ".join(f"{b} {v['all']:.3f}" for b, v in d["blocks"].items())
        print(
            f"k={depth}: all={d['geomean_tpot_speedup_all']:.3f} "
            f"c1={d['geomean_tpot_speedup_c1']:.3f} c8={d['geomean_tpot_speedup_c8']:.3f} "
            f"worst={d['worst_cell_speedup']:.3f} g={d['mean_g']:.2f} [{blocks}]"
        )
    for key, value in report["per_stratum"].items():
        print(f"  {key}: {value:.3f}")
    for key, value in report["greedy_agreement"].items():
        print(f"  agreement {key}: {value['identical']}/{value['compared']}")
    print("min success rate:", report["min_success_rate"])


if __name__ == "__main__":
    main()
