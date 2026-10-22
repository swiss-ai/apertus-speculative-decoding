"""Throughput ceiling and break-even acceptance of shallower DSpark drafters.

    python3 depth_ceiling.py --fit SWEEP_K7 --cost SWEEP_COST --grid SWEEP_A ... [--out X.json]

docs/small-drafter.md, phase A. Per output token, time is

    tau = (V_k + D_N(k, R)) / (R * a) + P

with R running requests, a accepted tokens per step, D_N the N-layer draft
step by draft length and request count (cost-table.json draft_ms_by_len of
the --cost sweep, one arm per depth), V_k the rest of a decode step at draft
length k, and P the prefill time per output token. V_7 and P are fitted per
concurrency on the --fit sweep (5/3/2/1-layer drafters at k=7, same job);
V_k for other k comes from the 5-layer grid (--grid) with that P. The
ceiling is an N-layer drafter at the 5-layer acceptance; the break-even
acceptance is what it needs at k to match the 5-layer drafter's best k.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

LABEL = "opb"
COST_ARMS = {5: "dspark-e10-l5", 3: "trunc-l3", 2: "trunc-l2", 1: "trunc-l1"}
FIT_ARMS = {5: "dspark-e10-k7", 3: "trunc-l3-k7", 2: "trunc-l2-k7", 1: "trunc-l1-k7"}
KS = (1, 2, 3, 4, 5, 7)


def interp(x: float, xs: list[float], ys: list[float]) -> float:
    if x <= xs[0]:
        return ys[0]
    for (x0, y0), (x1, y1) in zip(zip(xs, ys, strict=True), zip(xs[1:], ys[1:], strict=True),
                                  strict=False):
        if x <= x1:
            return y0 + (y1 - y0) * (x - x0) / (x1 - x0)
    return ys[-1]


def levels(path: Path) -> dict[int, dict]:
    return {lv["concurrency"]: lv for lv in json.loads(path.read_text())["levels"]}


def fit_line(points: list[tuple[float, float]]) -> tuple[float, float]:
    """Least squares y = v * x + p."""
    n = len(points)
    mx = sum(x for x, _ in points) / n
    my = sum(y for _, y in points) / n
    v = sum((x - mx) * (y - my) for x, y in points) / sum((x - mx) ** 2 for x, _ in points)
    return v, my - v * mx


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--fit", type=Path, required=True)
    parser.add_argument("--cost", type=Path, required=True)
    parser.add_argument("--grid", type=Path, nargs="+", required=True)
    parser.add_argument("--concurrencies", type=int, nargs="+", default=[64, 128, 256])
    parser.add_argument("--out", type=Path, help="write the full table as JSON")
    args = parser.parse_args(argv)

    tables = {n: json.loads((args.cost / arm / "cost-table.json").read_text())
              for n, arm in COST_ARMS.items()}

    def draft_ms(n: int, k: int, reqs: float) -> float:
        return interp(reqs, tables[n]["draft_reqs"], tables[n]["draft_ms_by_len"][str(k)])

    grid: dict[tuple[int, int], dict] = {}
    for sweep in args.grid:
        for k in KS:
            summary = sweep / f"dspark-e10-k{k}" / LABEL / "loadtest-summary.json"
            if summary.exists():
                for c, lv in levels(summary).items():
                    grid[(k, c)] = lv

    report: dict = {"model": "tau = (V_k + D_N) / (R a) + P", "concurrencies": {}}
    fitted: dict[int, tuple[float, float]] = {}
    fit_levels = {n: levels(args.fit / arm / LABEL / "loadtest-summary.json")
                  for n, arm in FIT_ARMS.items()}
    for c in sorted({c for lv in fit_levels.values() for c in lv}):
        if c < 8:
            continue
        points = []
        for n, lv in fit_levels.items():
            r, a = lv[c]["running_max"], lv[c]["mean_acceptance_length"]
            t = lv[c]["output_tokens_per_second"]
            points.append((1 / (r * a), 1000 / t - draft_ms(n, 7, r) / (r * a)))
        fitted[c] = fit_line(points)
    for c in args.concurrencies:
        near = min(fitted, key=lambda f: abs(f - c))
        lo = max((f for f in fitted if f <= c), default=near)
        hi = min((f for f in fitted if f >= c), default=near)
        p = fitted[lo][1] if lo == hi else interp(c, [lo, hi], [fitted[lo][1], fitted[hi][1]])
        five = {k: grid[(k, c)] for k in KS if (k, c) in grid}
        best_k = max(five, key=lambda k: five[k]["output_tokens_per_second"])
        best = five[best_k]["output_tokens_per_second"]
        entry = {"p_ms_per_token": p, "best_k": best_k, "best_tok_s": best, "depths": {}}
        for n in (3, 2, 1):
            rows = {}
            for k, lv in five.items():
                r, a = lv["running_max"], lv["mean_acceptance_length"]
                v = (1000 / lv["output_tokens_per_second"] - p) * r * a - draft_ms(5, k, r)
                d = draft_ms(n, k, r)
                rows[k] = {
                    "draft_ms": round(d, 3),
                    "draft_ms_5": round(draft_ms(5, k, r), 3),
                    "ceiling_tok_s": round(1000 / ((v + d) / (r * a) + p)),
                    "break_even_acceptance": round((v + d) / (r * (1000 / best - p)), 3),
                    "acceptance_5": a,
                }
            entry["depths"][n] = rows
        report["concurrencies"][c] = entry
        print(f"C={c}: 5-layer best k={best_k} {best:.0f} tok/s (P={p * 1000:.1f} us/token)")
        for n, rows in entry["depths"].items():
            top = max(rows, key=lambda k: rows[k]["ceiling_tok_s"])
            easy = min(
                rows, key=lambda k: rows[k]["break_even_acceptance"] / rows[k]["acceptance_5"]
            )
            print(f"  N={n}: ceiling {rows[top]['ceiling_tok_s']} at k={top} "
                  f"({rows[top]['ceiling_tok_s'] / best - 1:+.1%}); break-even at k={easy} "
                  f"{rows[easy]['break_even_acceptance']:.2f} "
                  f"({rows[easy]['break_even_acceptance'] / rows[easy]['acceptance_5']:.0%} "
                  f"of {rows[easy]['acceptance_5']:.2f})")
    report["fit"] = {c: {"v7_ms": v, "p_ms_per_token": p} for c, (v, p) in fitted.items()}
    print(json.dumps(report["fit"]))
    if args.out:
        args.out.write_text(json.dumps(report, indent=1) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
