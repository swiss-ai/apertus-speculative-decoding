"""Where a DSpark draft step's time goes by depth, and the decode-only ceiling.

    python3 depth_profile.py --nsys SWEEP_NSYS --cost SWEEP_COST --grid SWEEP_A ... [--out X.json]

docs/small-drafter.md, phase A. From the Nsight sweep (5-layer DSpark e10
and its 1-layer truncation at k=4 and 7, eager draft) it reads the mean
per-step time of each draft range (apertus_bench.waste_report.dspark_samples),
fits the component model of waste_report (per-layer backbone and context
K/V, LM head, Markov loop) to the 5-layer windows and checks its 1-3 layer
extrapolation against the measured cost tables. It then gives the ceiling
and break-even acceptance of each depth in waste_report's decode-only step
model (CostModel.step_ms, no prefill), next to depth_ceiling.py's end-to-end
one.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from apertus_bench.waste_report import CostModel, dspark_samples, fit_dspark_components

LABEL = "opb"
NSYS_ARMS = {(5, 4): "dspark-e10-k4-nsys", (1, 4): "trunc-l1-k4-nsys",
             (5, 7): "dspark-e10-k7-nsys", (1, 7): "trunc-l1-k7-nsys"}
COST_ARMS = {5: "dspark-e10-l5", 3: "trunc-l3", 2: "trunc-l2", 1: "trunc-l1"}
RANGES = ("draft:propose", "draft:fc", "draft:context_kv", "draft:backbone", "draft:lm_head",
          "draft:markov")
KS = (1, 2, 3, 4, 5, 7)
# Mean cached tokens per running request on the held-out prompts (prompt
# about 563 tokens plus half of a 480-token answer).
CONTEXT_PER_REQUEST = 800


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--nsys", type=Path, required=True)
    parser.add_argument("--cost", type=Path, required=True)
    parser.add_argument("--grid", type=Path, nargs="+", required=True)
    parser.add_argument("--concurrencies", type=int, nargs="+", default=[64, 128, 256])
    parser.add_argument("--out", type=Path)
    args = parser.parse_args(argv)

    report: dict = {"windows": [], "extrapolation": [], "decode_only": {}}
    by_depth: dict[int, list[dict]] = {5: [], 1: []}
    for (layers, k), arm in NSYS_ARMS.items():
        stats = args.nsys / arm / "nsys-stats"
        manifest = json.loads((stats / "manifest.json").read_text())
        levels = manifest["profile"]["levels"]
        for window, level in enumerate(levels, start=1):
            sample = dspark_samples(stats, f"{arm}.{window}", k)
            if sample is None:
                continue
            by_depth[layers].append(sample)
            row = {"layers": layers, "k": k, "concurrency": level,
                   "requests": round(sample["requests"], 1)}
            row.update({key.split(":")[1]: round(sample.get(key) or 0.0, 3) for key in RANGES})
            report["windows"].append(row)
    print("draft step components, eager (ms per decode step):")
    print("| layers | k | C | decode reqs | propose | backbone | LM head | Markov | fc "
          "| context K/V |")
    print("| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    for r in sorted(report["windows"], key=lambda r: (r["k"], r["concurrency"], -r["layers"])):
        print(f"| {r['layers']} | {r['k']} | {r['concurrency']} | {r['requests']:.0f} | "
              f"{r['propose']:.2f} | {r['backbone']:.2f} | {r['lm_head']:.2f} | "
              f"{r['markov']:.2f} | {r['fc']:.2f} | {r['context_kv']:.2f} |")

    fit = fit_dspark_components(by_depth[5], layers=5)
    report["fit_5_layer"] = {k: v for k, v in fit.__dict__.items() if k != "source"}
    models = {n: CostModel.from_cost_table(args.cost / arm / "cost-table.json")
              for n, arm in COST_ARMS.items()}
    print("\nfit on the 5-layer windows, extrapolated by depth, vs the cost tables (ms):")
    for n in (5, 3, 2, 1):
        for reqs, k in ((64, 4), (256, 4), (256, 7)):
            parts = fit.draft_ms(reqs, k, layers=n)
            measured = models[n].dspark_draft_ms(reqs, k)
            row = {"layers": n, "requests": reqs, "k": k, "fit_ms": round(parts["total"], 3),
                   "table_ms": round(measured, 3)}
            report["extrapolation"].append(row)
            print(f"  N={n} R={reqs} k={k}: fit {parts['total']:.2f} "
                  f"(backbone {parts['backbone']:.2f}, LM head {parts['lm_head']:.2f}, "
                  f"Markov {parts['markov']:.2f}) table {measured:.2f}")

    grid: dict[tuple[int, int], dict] = {}
    for sweep in args.grid:
        for k in KS:
            summary = sweep / f"dspark-e10-k{k}" / LABEL / "loadtest-summary.json"
            if summary.exists():
                for lv in json.loads(summary.read_text())["levels"]:
                    grid[(k, lv["concurrency"])] = lv
    five = models[5]
    print("\ndecode-only step model (waste_report.CostModel, no prefill):")
    for c in args.concurrencies:
        acc = {k: grid[(k, c)]["mean_acceptance_length"] for k in KS if (k, c) in grid}
        reqs = c
        ctx = reqs * CONTEXT_PER_REQUEST
        rate5 = {k: acc[k] / five.step_ms(reqs, k, ctx) for k in acc}
        best_k = max(rate5, key=rate5.get)
        entry = {"best_k": best_k, "depths": {}}
        for n in (3, 2, 1):
            rows = {}
            for k in acc:
                step = five.step_ms(reqs, k, ctx, draft_ms=models[n].dspark_draft_ms(reqs, k))
                rows[k] = {"ceiling": round(acc[k] / step / rate5[best_k] - 1, 4),
                           "break_even_acceptance": round(rate5[best_k] * step, 3),
                           "acceptance_5": acc[k]}
            entry["depths"][n] = rows
            top = max(rows, key=lambda k: rows[k]["ceiling"])
            easy = min(rows, key=lambda k: rows[k]["break_even_acceptance"] / acc[k])
            print(f"  C={c} N={n}: ceiling {rows[top]['ceiling']:+.1%} at k={top} (5-layer best "
                  f"k={best_k}); break-even {rows[easy]['break_even_acceptance']:.2f} at k={easy} "
                  f"({rows[easy]['break_even_acceptance'] / acc[easy]:.0%} of {acc[easy]:.2f})")
        report["decode_only"][c] = entry
    if args.out:
        args.out.write_text(json.dumps(report, indent=1) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
