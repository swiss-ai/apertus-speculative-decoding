"""Live k=3/5/7 speedups of the deployed pod, paired against the same-day plain window,
next to the controlled k-headline numbers.

  report-live.py --live LIVE_ROOT --controlled results/8b/k-headline/report-20261122 \
      --nov5 results/8b/task-speed/report-live-20261105 --output DIR

Windows: P (plain), K3, K5, K7, all direct to the Service. Writes live-k-results.json,
REPORT.md and k-headline-live.png.
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

from apertus_bench import task_speed

MATCH = ("corpus_sha256", "route", "node")
GROUPS = ("all", "id", "near-id", "ood-task", "ood-language")
KS = (3, 5, 7)
BAND = 0.07  # live within +-7% of controlled, as at k=3 on 2026-11-05
TIE = 0.03  # runner-up within 3% = tie (frozen k-headline rule)


def windows(root: Path) -> dict[str, dict]:
    found = {}
    for window in json.loads((root / "windows.json").read_text()):
        found[window["window"]] = window
    return found


def group_rows(rows: list[dict]) -> dict[tuple[str, int], dict]:
    out = {}
    for c in sorted({r["concurrency"] for r in rows}):
        members = [r for r in rows if r["concurrency"] == c]
        for group in GROUPS:
            chosen = [r for r in members if group == "all" or r["class"] == group]
            if not chosen:
                continue
            lengths = [r["acceptance_length"] for r in chosen if r["acceptance_length"]]
            pairs = [r["pairs"][0] for r in chosen]
            out[(group, c)] = {
                "slices": len(chosen),
                "speedup": statistics.geometric_mean([r["speedup_geomean"] for r in chosen]),
                "tpot_speedup": task_speed._geomean([r["tpot_speedup_geomean"] for r in chosen]),
                "acceptance_length": statistics.fmean(lengths) if lengths else None,
                "plain_tokens_s": statistics.geometric_mean([p["plain_tokens_s"] for p in pairs]),
                "dspark_tokens_s": statistics.geometric_mean([p["dspark_tokens_s"] for p in pairs]),
            }
    return out


def best_k(speedups: dict[int, float]) -> str:
    ranked = sorted(speedups, key=speedups.get, reverse=True)
    best, runner = ranked[0], ranked[1]
    tie = speedups[runner] >= (1 - TIE) * speedups[best]
    return f"k{best}" + (f"≈k{runner}" if tie else "")


def fmt(value, spec=".2f", suffix=""):
    return "–" if value is None else f"{value:{spec}}{suffix}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--live", type=Path, required=True)
    parser.add_argument("--controlled", type=Path, required=True)
    parser.add_argument("--nov5", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    suite = task_speed.load_suite(args.live / "suite.yaml")
    found = windows(args.live)
    plain = args.live / found["P"]["name"]
    controlled = json.loads((args.controlled / "k-grid-results.json").read_text())
    c_groups = {(g["group"], g["concurrency"]): g["k"] for g in controlled["groups"]}
    c_rows = {
        int(k): {(r["slice"], r["concurrency"]): r for r in v["rows"]}
        for k, v in controlled["per_k"].items()
    }
    result = {"suite": suite["name"], "windows": found, "per_k": {}, "groups": []}
    groups = {}
    for k in KS:
        name = f"K{k}"
        cells = task_speed.cell_rows(
            suite, {"live": {"plain": plain, "dspark": args.live / found[name]["name"]}},
            MATCH, guarded=True,
        )  # fmt: skip
        for row in cells["rows"]:
            ref = c_rows[k].get((row["slice"], row["concurrency"]))
            row["controlled_speedup"] = ref["speedup_geomean"] if ref else None
            row["live_over_controlled"] = (
                row["speedup_geomean"] / ref["speedup_geomean"] if ref else None
            )
            row["controlled_acceptance_length"] = ref["acceptance_length"] if ref else None
        result["per_k"][str(k)] = cells
        groups[k] = group_rows(cells["rows"])
    cs = suite["concurrencies"]
    for group in GROUPS:
        for c in cs:
            entry = {"group": group, "concurrency": c, "k": {}}
            for k in KS:
                live = groups[k].get((group, c))
                ref = c_groups[(group, c)][str(k)]
                entry["k"][str(k)] = {
                    **(live or {}),
                    "controlled_speedup": ref["speedup"],
                    "controlled_acceptance_length": ref["acceptance_length"],
                    "live_over_controlled": live["speedup"] / ref["speedup"] if live else None,
                }
            if all(entry["k"][str(k)].get("speedup") for k in KS):
                entry["best_live"] = best_k({k: entry["k"][str(k)]["speedup"] for k in KS})
            entry["best_controlled"] = best_k(
                {k: entry["k"][str(k)]["controlled_speedup"] for k in KS}
            )
            result["groups"].append(entry)
    # Anchors: K3 and the plain window against the 2026-11-05 live campaign (gateway route).
    nov5 = json.loads((args.nov5 / "paired-results.json").read_text())["live"]["rows"]
    nov5_rows = {(r["slice"], r["concurrency"]): r for r in nov5}
    nov5_groups = group_rows(nov5)
    anchors = []
    for group in GROUPS:
        for c in cs:
            live, old = groups[3].get((group, c)), nov5_groups.get((group, c))
            if not (live and old):
                continue
            anchors.append({
                "group": group, "concurrency": c,
                "k3_speedup": live["speedup"], "nov5_k3_speedup": old["speedup"],
                "k3_ratio": live["speedup"] / old["speedup"],
                "plain_tokens_s": live["plain_tokens_s"],
                "nov5_plain_tokens_s": old["plain_tokens_s"],
                "plain_ratio": live["plain_tokens_s"] / old["plain_tokens_s"],
            })  # fmt: skip
    result["anchors_vs_nov5"] = anchors
    within = {}
    for k in KS:
        ratios = [
            r["live_over_controlled"]
            for r in result["per_k"][str(k)]["rows"]
            if r["live_over_controlled"]
        ]
        within[str(k)] = {
            "cells": len(ratios),
            "within_band": sum(abs(x - 1) <= BAND for x in ratios),
            "median_ratio": statistics.median(ratios) if ratios else None,
            "min_ratio": min(ratios) if ratios else None,
            "max_ratio": max(ratios) if ratios else None,
        }
    result["cells_within_band"] = within
    result["nov5_cells"] = len(nov5_rows)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "live-k-results.json").write_text(json.dumps(result, indent=2) + "\n")
    write_report(args.output, result, cs)


def write_report(out: Path, result: dict, cs: list[int]) -> None:
    lines = ["# Live k=3/5/7 check of the deployed DSpark e10 pod", ""]
    lines += ["## Windows", "", "| Window | Variant | Status | Pod UID | Node | Start | End |",
              "|---|---|---|---|---|---|---|"]  # fmt: skip
    for name in ("P", "K3", "K5", "K7"):
        w = result["windows"].get(name, {})
        lines.append(
            f"| {w.get('name', name)} | {w.get('variant', '–')} | {w.get('status', 'missing')} | "
            f"{w.get('pod_uid', '–')} | {w.get('node', '–')} | {w.get('started', '–')} | "
            f"{w.get('ended', '–')} |"
        )
    for k in KS:
        cells = result["per_k"][str(k)]
        lines += ["", f"K{k}: {len(cells['rows'])} paired cells, {len(cells['missing'])} missing, "
                  f"{len(cells['rejected'])} rejected."]  # fmt: skip
        lines += [f"- rejected: {x}" for x in cells["rejected"]]
    lines += ["", "## Live vs controlled, by group (speedup over same-day live plain)", "",
              "Live x (controlled x) [live ÷ controlled]; L = live acceptance length.", "",
              "| Group | k | " + " | ".join(f"C={c}" for c in cs) + " | L live (ctrl) |",
              "|---|---|" + "---|" * len(cs) + "---|"]  # fmt: skip
    by = {(g["group"], g["concurrency"]): g for g in result["groups"]}
    for group in GROUPS:
        for k in KS:
            cells = []
            for c in cs:
                e = by[(group, c)]["k"][str(k)]
                cells.append(
                    f"{fmt(e.get('speedup'))}x ({fmt(e['controlled_speedup'])}x) "
                    f"[{fmt(e.get('live_over_controlled'))}]"
                )
            e1 = by[(group, cs[0])]["k"][str(k)]
            lines.append(
                f"| {group} | {k} | " + " | ".join(cells) + f" | "
                f"{fmt(e1.get('acceptance_length'))} ({fmt(e1['controlled_acceptance_length'])}) |"
            )
    lines += ["", "## Best k per load (runner-up within 3% = tie)", "",
              "| Group | " + " | ".join(f"C={c} live / controlled" for c in cs) + " |",
              "|---|" + "---|" * len(cs)]  # fmt: skip
    for group in GROUPS:
        lines.append(
            f"| {group} | "
            + " | ".join(
                f"{by[(group, c)].get('best_live', '–')} / {by[(group, c)]['best_controlled']}"
                for c in cs
            )
            + " |"
        )
    lines += ["", "## Cells within ±7% of controlled", "",
              "| k | Cells | Within | Median | Range |", "|---|---|---|---|---|"]  # fmt: skip
    for k, w in result["cells_within_band"].items():
        lines.append(
            f"| {k} | {w['cells']} | {w['within_band']} | {fmt(w['median_ratio'])} | "
            f"{fmt(w['min_ratio'])}–{fmt(w['max_ratio'])} |"
        )
    lines += ["", "## Anchors against the 2026-11-05 live campaign (gateway route)", "",
              "| Group | C | K3 now | K3 Nov 5 | ratio | plain tok/s now | Nov 5 | ratio |",
              "|---|---|---|---|---|---|---|---|"]  # fmt: skip
    for a in result["anchors_vs_nov5"]:
        lines.append(
            f"| {a['group']} | {a['concurrency']} | {fmt(a['k3_speedup'])}x | "
            f"{fmt(a['nov5_k3_speedup'])}x | {fmt(a['k3_ratio'])} | "
            f"{fmt(a['plain_tokens_s'], '.0f')} | {fmt(a['nov5_plain_tokens_s'], '.0f')} | "
            f"{fmt(a['plain_ratio'])} |"
        )
    header = " | ".join(f"k{k} live (ctrl)" for k in KS)
    lines += ["", "## Per slice", "", f"| Slice | Class | C | {header} |",
              "|---|---|---|" + "---|" * len(KS)]  # fmt: skip
    rows = {
        k: {(r["slice"], r["concurrency"]): r for r in result["per_k"][str(k)]["rows"]} for k in KS
    }
    for (label, c), r3 in sorted(rows[3].items(), key=lambda x: (x[1]["class"], x[0])):
        parts = []
        for k in KS:
            r = rows[k].get((label, c))
            parts.append(
                f"{fmt(r and r['speedup_geomean'])}x ({fmt(r and r['controlled_speedup'])}x)"
                if r
                else "–"
            )
        lines.append(f"| {label} | {r3['class']} | {c} | " + " | ".join(parts) + " |")
    (out / "REPORT.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
