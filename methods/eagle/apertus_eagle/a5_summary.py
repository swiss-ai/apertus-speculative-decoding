"""A5: summarize screening cells and rank EAGLE depths.

Reads every ``summary.json`` + ``metadata.json`` under the screening root. For
each EAGLE cell the denominator is the operational baseline deployment of the
same block (same stratum and concurrency), so blocks are compared within
themselves. Selection metric (plan section A5): geometric-mean TPOT speedup
over the declared validation strata, equal weights; reported per concurrency
and over all six (stratum, concurrency) cells, averaged over blocks.

Committed tokens per round g = 1 + accepted / drafts from the engine counters
(an upper bound on delivered tokens when EOS or the output cap ends a round).
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any


def load_cells(root: Path) -> list[dict[str, Any]]:
    cells = []
    for summary_path in sorted(root.rglob("summary.json")):
        metadata_path = summary_path.with_name("metadata.json")
        if not metadata_path.is_file():
            continue
        summary = json.loads(summary_path.read_text())
        metadata = json.loads(metadata_path.read_text())
        extra = metadata.get("extra") or {}
        variant = metadata["variant"]
        spec = summary.get("speculative_decoding") or {}
        drafts = spec.get("drafts") or 0
        accepted = spec.get("accepted_tokens") or 0
        latency = summary["latency_ms"]
        cells.append(
            {
                "variant": variant["name"],
                "method": variant["method"],
                "depth": variant.get("num_speculative_tokens"),
                "block": extra.get("block_id"),
                "deployment": extra.get("deployment_id"),
                "workload": metadata["cell"]["workload"],
                "concurrency": metadata["cell"]["concurrency"],
                "success_rate": summary["requests"]["success_rate"],
                "requests": summary["requests"]["successful"],
                "output_tps": summary["tokens"]["output_tokens_per_second"],
                "completion_tokens": summary["tokens"]["completion"],
                "tpot_p50": latency["tpot"]["p50"],
                "tpot_p95": latency["tpot"]["p95"],
                "ttft_p50": latency["ttft"]["p50"],
                "ttft_p95": latency["ttft"]["p95"],
                "e2e_p50": (latency.get("e2e") or {}).get("p50"),
                "e2e_p95": (latency.get("e2e") or {}).get("p95"),
                "acceptance_rate": spec.get("acceptance_rate"),
                "g": (1 + accepted / drafts) if drafts else None,
                "acceptance_per_position": spec.get("acceptance_rate_per_position"),
            }
        )
    return cells


def mean(values: list[float | None]) -> float | None:
    values = [v for v in values if v is not None]
    return sum(values) / len(values) if values else None


def geomean(values: list[float]) -> float | None:
    values = [v for v in values if v and v > 0]
    return math.exp(sum(math.log(v) for v in values) / len(values)) if values else None


def summarize(cells: list[dict[str, Any]]) -> dict[str, Any]:
    baseline = {
        (c["block"], c["workload"], c["concurrency"]): c for c in cells if c["method"] == "none"
    }
    rows = []
    for cell in cells:
        if cell["method"] == "none":
            continue
        base = baseline.get((cell["block"], cell["workload"], cell["concurrency"]))
        if base is None:
            raise SystemExit(
                f"no same-block baseline for {cell['variant']} {cell['block']} "
                f"{cell['workload']} c{cell['concurrency']}"
            )
        rows.append(
            {
                **cell,
                "baseline_tpot_p50": base["tpot_p50"],
                "tpot_speedup": base["tpot_p50"] / cell["tpot_p50"],
                "throughput_speedup": cell["output_tps"] / base["output_tps"],
                "ttft_p50_ratio": cell["ttft_p50"] / base["ttft_p50"],
                "e2e_p50_ratio": (cell["e2e_p50"] / base["e2e_p50"])
                if cell["e2e_p50"] and base["e2e_p50"]
                else None,
            }
        )
    by_depth: dict[int, dict[str, Any]] = {}
    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[row["depth"]].append(row)
    for depth, items in sorted(grouped.items()):
        blocks = sorted({r["block"] for r in items})
        per_block = {}
        for block in blocks:
            mine = [r for r in items if r["block"] == block]
            per_block[block] = {
                "all": geomean([r["tpot_speedup"] for r in mine]),
                "c1": geomean([r["tpot_speedup"] for r in mine if r["concurrency"] == 1]),
                "c8": geomean([r["tpot_speedup"] for r in mine if r["concurrency"] == 8]),
                "worst_cell": min(r["tpot_speedup"] for r in mine),
            }
        by_depth[depth] = {
            "blocks": per_block,
            "geomean_tpot_speedup_all": mean([v["all"] for v in per_block.values()]),
            "geomean_tpot_speedup_c1": mean([v["c1"] for v in per_block.values()]),
            "geomean_tpot_speedup_c8": mean([v["c8"] for v in per_block.values()]),
            "worst_cell_speedup": min(v["worst_cell"] for v in per_block.values()),
            "mean_g": sum(r["g"] for r in items if r["g"])
            / max(1, sum(1 for r in items if r["g"])),
            "cells": len(items),
            "min_success_rate": min(r["success_rate"] for r in items),
        }
    ranking = sorted(by_depth, key=lambda d: -by_depth[d]["geomean_tpot_speedup_all"])
    best = by_depth[ranking[0]]["geomean_tpot_speedup_all"] if ranking else None
    selected = ranking[:1]
    if len(ranking) > 1:
        second = ranking[1]
        # Plan: a setting within 5% of the best with lower memory or better
        # worst-stratum latency may take the second slot.
        if by_depth[second]["geomean_tpot_speedup_all"] >= 0.95 * best:
            selected.append(second)
    return {"rows": rows, "by_depth": by_depth, "ranking": ranking, "selected": selected}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="apertus-eagle-a5-summary")
    parser.add_argument("root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    report = summarize(load_cells(args.root))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    fmt = lambda v: "n/a" if v is None else f"{v:.3f}"  # noqa: E731
    for depth in report["ranking"]:
        d = report["by_depth"][depth]
        print(
            f"k={depth}: geomean TPOT speedup all={fmt(d['geomean_tpot_speedup_all'])} "
            f"c1={fmt(d['geomean_tpot_speedup_c1'])} c8={fmt(d['geomean_tpot_speedup_c8'])} "
            f"worst={fmt(d['worst_cell_speedup'])} g={d['mean_g']:.2f} "
            f"success>={d['min_success_rate']}"
        )
    print("selected:", report["selected"])


if __name__ == "__main__":
    main()
