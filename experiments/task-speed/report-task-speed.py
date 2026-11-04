"""Report per-task DSpark speedups: controlled pairs, live windows, or both.

  report-task-speed.py --controlled RUN [--controlled EXTENSION] [--live RUN] --output DIR

Writes paired-results.json, REPORT.md and select.json (the controlled cells
whose interval is inconclusive, as input for extension pairs).
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

from apertus_bench import task_speed

CONTROLLED_MATCH = (
    "source_revision",
    "physical_gpu",
    "node",
    "target_revision",
    "corpus_sha256",
    "engine",
)
LIVE_MATCH = ("corpus_sha256", "route", "node")


def controlled_pairs(roots: list[Path]) -> dict[str, dict[str, Path]]:
    pairs = {}
    for root in roots:
        for pair in sorted(root.glob("pair-*"), key=lambda p: int(p.name.split("-")[1])):
            pairs[f"{root.name}/{pair.name}"] = {"plain": pair / "plain", "dspark": pair / "dspark"}
    return pairs


def live_pairs(root: Path) -> dict[str, dict[str, Path]]:
    """Rounds of the live campaign: each pairs a plain and a DSpark gateway window."""
    windows = json.loads((root / "windows.json").read_text())
    rounds: dict[str, dict[str, Path]] = {}
    for window in windows:
        if window.get("status") != "ok" or window.get("route") != "gateway":
            continue
        rounds.setdefault(f"round-{window['round']}", {})[window["variant"]] = root / window["name"]
    return {name: arms for name, arms in rounds.items() if set(arms) == {"plain", "dspark"}}


def gateway_overhead(root: Path) -> list[dict]:
    """Gateway minus direct latency, per cell measured both ways in one window."""
    rows = []
    for direct in sorted(root.glob("*-direct")):
        gateway = root / direct.name.removesuffix("-direct")
        for summary_path in sorted(direct.glob("*/*/c*/summary.json")):
            relative = summary_path.relative_to(direct)
            other = gateway / relative
            if relative.parts[0] == "rejected" or not other.exists():
                continue
            via_direct = json.loads(summary_path.read_text())
            via_gateway = json.loads(other.read_text())
            rows.append(
                {
                    "window": gateway.name,
                    "slice": relative.parts[0],
                    "concurrency": int(relative.parts[2][1:]),
                    **{
                        f"{metric}_p50_ms_added": via_gateway["latency_ms"][metric]["p50"]
                        - via_direct["latency_ms"][metric]["p50"]
                        for metric in ("ttft", "e2e")
                    },
                    "tokens_s_gateway_over_direct": via_gateway["tokens"][
                        "output_tokens_per_second"
                    ]
                    / via_direct["tokens"]["output_tokens_per_second"],
                }
            )
    return rows


def fmt(value, spec=".2f", suffix=""):
    return "–" if value is None else f"{value:{spec}}{suffix}"


def speed_cell(row) -> str:
    if row is None:
        return "–"
    ci = row["bootstrap_95"]
    text = f"{row['speedup_geomean']:.2f}x"
    if ci:
        text += f" [{ci[0]:.2f}, {ci[1]:.2f}]"
    marks = {"slowdown": " ▼", "inconclusive": " ?", "within 5%": " ="}
    return text + marks.get(row["verdict"], "")


def render(suite, result) -> str:
    concurrencies = suite["concurrencies"]
    lines = [
        "# DSpark e10 k=3: per-task speedups",
        "",
        "Speedup = DSpark output tokens/s ÷ plain output tokens/s on the same prompts, "
        "geometric mean over deployment pairs, with the exact pair bootstrap 95% interval. "
        "▼ slowdown (interval below 0.95), = within ±5%, ? inconclusive; unmarked with an "
        "interval = speedup above 1.05. Greedy, thinking off, natural EOS at each source's "
        "output cap, 32 hash-selected prompts per slice, whole passes per level.",
        "",
    ]
    for track in ("controlled", "live"):
        data = result.get(track)
        if not data:
            continue
        title = {
            "controlled": "Controlled (Clariden, plain and DSpark on the same GPU)",
            "live": "Deployed (the Kubernetes pod, plain and DSpark windows via the gateway)",
        }[track]
        lines += [f"## {title}", "", f"Pairs: {', '.join(data['pairs'])}.", ""]
        lines += ["### By class", ""]
        header = "| Class | " + " | ".join(f"C={c}" for c in concurrencies) + " | Acc. length |"
        lines += [header, "|---" * (len(concurrencies) + 2) + "|"]
        by_class = {(r["class"], r["concurrency"]): r for r in data["classes"]}
        for cls in task_speed.CLASSES:
            cells = [by_class.get((cls, c)) for c in concurrencies]
            if not any(cells):
                continue
            length = next((c["acceptance_length_mean"] for c in cells if c), None)
            values = [
                "–"
                if c is None
                else f"{c['speedup_geomean']:.2f}x ({c['speedup_min']:.2f}–{c['speedup_max']:.2f})"
                for c in cells
            ]
            lines.append(f"| {cls} | " + " | ".join(values) + f" | {fmt(length)} |")
        lines += ["", "Class cells: geometric mean of the slices (range over slices).", ""]
        lines += ["### By slice", ""]
        header = "| Slice | Class | Acc. length | " + " | ".join(f"C={c}" for c in concurrencies)
        lines += [header + " |", "|---" * (len(concurrencies) + 3) + "|"]
        rows = {(r["slice"], r["concurrency"]): r for r in data["rows"]}
        for entry in suite["slices"]:
            cells = [rows.get((entry["label"], c)) for c in concurrencies]
            if not any(cells):
                continue
            length = next((c["acceptance_length"] for c in cells if c), None)
            lines.append(
                f"| {entry['label']} | {entry['class']} | {fmt(length)} | "
                + " | ".join(speed_cell(c) for c in cells)
                + " |"
            )
        lines += ["", "### Speedup against acceptance length", ""]
        lines += ["| C | Speedup ≈ a + b·length | R² | Break-even length |", "|---|---|---|---|"]
        for curve in data["curves"]:
            lines.append(
                f"| {curve['concurrency']} | {curve['intercept']:.2f} + "
                f"{curve['slope']:.2f}·L | {curve['r2']:.2f} | "
                f"{fmt(curve['break_even_acceptance_length'])} |"
            )
        lines += ["", "### Latency at C=1 (medians, ms)", ""]
        lines += [
            "| Slice | TTFT plain | TTFT DSpark | TPOT plain | TPOT DSpark | E2E plain "
            "| E2E DSpark |",
            "|---|---|---|---|---|---|---|",
        ]
        for entry in suite["slices"]:
            row = rows.get((entry["label"], concurrencies[0]))
            if row is None:
                continue

            def median(key, pairs=row["pairs"]):
                return statistics.median(p[key] for p in pairs)

            lines.append(
                f"| {entry['label']} | "
                + " | ".join(
                    fmt(median(f"{arm}_{metric}_p50_ms"), ".0f")
                    if metric != "tpot"
                    else fmt(median(f"{arm}_{metric}_p50_ms"), ".1f")
                    for metric in ("ttft", "tpot", "e2e")
                    for arm in ("plain", "dspark")
                )
                + " |"
            )
        if data["rejected"] or data["missing"]:
            lines += ["", f"**Rejected pairs ({len(data['rejected'])}):**", ""]
            lines += [f"- {r}" for r in data["rejected"]]
            lines += ["", f"**Missing cells: {len(data['missing'])}.**"]
        lines.append("")
    if result.get("gateway_overhead"):
        lines += ["## Gateway overhead (gateway minus direct, same window)", ""]
        lines += [
            "| Window | Slice | C | TTFT p50 added (ms) | E2E p50 added (ms) | tokens/s ratio |",
            "|---|---|---|---|---|---|",
        ]
        for row in result["gateway_overhead"]:
            lines.append(
                f"| {row['window']} | {row['slice']} | {row['concurrency']} | "
                f"{row['ttft_p50_ms_added']:.0f} | {row['e2e_p50_ms_added']:.0f} | "
                f"{row['tokens_s_gateway_over_direct']:.2f} |"
            )
        lines.append("")
    if result.get("transfer"):
        lines += ["## Live ÷ controlled speedup", ""]
        header = "| Slice | " + " | ".join(f"C={c}" for c in concurrencies) + " |"
        lines += [header, "|---" * (len(concurrencies) + 1) + "|"]
        transfer = {(t["slice"], t["concurrency"]): t["ratio"] for t in result["transfer"]}
        for entry in suite["slices"]:
            values = [transfer.get((entry["label"], c)) for c in concurrencies]
            if any(v is not None for v in values):
                lines.append(f"| {entry['label']} | " + " | ".join(fmt(v) for v in values) + " |")
        lines.append("")
    return "\n".join(lines)


def analyse(suite, pairs, match) -> dict:
    data = task_speed.cell_rows(suite, pairs, match)
    data["pairs"] = list(pairs)
    data["classes"] = task_speed.class_summary(data["rows"])
    data["curves"] = task_speed.acceptance_curve(data["rows"])
    return data


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", type=Path, default=Path(__file__).with_name("suite.yaml"))
    parser.add_argument("--controlled", type=Path, action="append", default=[])
    parser.add_argument("--live", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    suite = task_speed.load_suite(args.suite)
    result = {"suite": suite["name"]}
    if args.controlled:
        result["controlled"] = analyse(suite, controlled_pairs(args.controlled), CONTROLLED_MATCH)
    if args.live:
        result["live"] = analyse(suite, live_pairs(args.live), LIVE_MATCH)
        result["gateway_overhead"] = gateway_overhead(args.live)
    if args.controlled and args.live:
        controlled = {(r["slice"], r["concurrency"]): r for r in result["controlled"]["rows"]}
        result["transfer"] = [
            {
                "slice": row["slice"],
                "concurrency": row["concurrency"],
                "ratio": row["speedup_geomean"]
                / controlled[(row["slice"], row["concurrency"])]["speedup_geomean"],
            }
            for row in result["live"]["rows"]
            if (row["slice"], row["concurrency"]) in controlled
        ]
    select: dict[str, list[int]] = {}
    for row in result.get("controlled", {}).get("rows", []):
        if row["verdict"] in ("inconclusive", "incomplete"):
            select.setdefault(row["slice"], []).append(row["concurrency"])
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "paired-results.json").write_text(json.dumps(result, indent=2) + "\n")
    (args.output / "select.json").write_text(json.dumps({"select": select}, indent=2) + "\n")
    (args.output / "REPORT.md").write_text(render(suite, result))
    print(
        json.dumps(
            {
                track: {
                    "rows": len(result[track]["rows"]),
                    "missing": len(result[track]["missing"]),
                    "rejected": len(result[track]["rejected"]),
                }
                for track in ("controlled", "live")
                if track in result
            }
            | {"extend_cells": sum(len(v) for v in select.values())}
        )
    )


if __name__ == "__main__":
    main()
