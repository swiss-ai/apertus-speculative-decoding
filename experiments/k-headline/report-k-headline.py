"""Report the deployed DSpark e10 head's speedup at draft lengths k = 3, 5, 7.

  report-k-headline.py --run RUN [--run RUN ...] --output DIR

RUN is a controlled k-grid run (submit-controlled.sh with TASK_SPEED_KS): each
block-N holds plain and dspark-k<k> deployments on one GPU. For every k the
block's plain cells are paired with its dspark-k<k> cells by
apertus_bench.task_speed.cell_rows (same validation as the k=3 campaign), so a
speedup is always DSpark ÷ plain on the same GPU, prompts and engine settings.

Groups weigh their slices equally (geometric mean of slice speedups). A group's
interval is the exact bootstrap over blocks of the per-block group geomean.
Writes k-grid-results.json, headline.csv and REPORT.md.
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path

from apertus_bench import task_speed

MATCH = ("source_revision", "physical_gpu", "node", "target_revision", "corpus_sha256", "engine")
GROUPS = {
    "all": task_speed.CLASSES,
    "id": ("id",),
    "near-id": ("near-id",),
    "ood-task": ("ood-task",),
    "ood-language": ("ood-language",),
}
# The best k must beat the runner-up by more than this to be called a clear win.
TIE = 0.03


def blocks(runs: list[Path]) -> dict[str, Path]:
    found = {}
    for run in runs:
        for block in sorted(run.glob("block-*"), key=lambda p: int(p.name.split("-")[1])):
            found[f"{run.name}/{block.name}"] = block
    return found


def draft_lengths(found: dict[str, Path]) -> list[int]:
    ks = {int(d.name.removeprefix("dspark-k")) for b in found.values() for d in b.glob("dspark-k*")}
    return sorted(ks)


def geomean(values):
    values = [v for v in values if v]
    return statistics.geometric_mean(values) if values else None


def group_rows(rows, classes, concurrency):
    return [r for r in rows if r["class"] in classes and r["concurrency"] == concurrency]


def group_stats(rows, block_names) -> dict | None:
    """Group geomean of slice speedups, its per-block interval, latency and acceptance."""
    if not rows:
        return None
    per_block = []
    for name in block_names:
        ratios = [p["speedup"] for r in rows for p in r["pairs"] if p["pair"] == name]
        if len(ratios) == len(rows):  # the block measured every slice of the group
            per_block.append(statistics.geometric_mean(ratios))

    def median(row, key):
        return statistics.median(p[key] for p in row["pairs"])

    stats = {
        "slices": len(rows),
        "speedup": statistics.geometric_mean([r["speedup_geomean"] for r in rows]),
        "speedup_min": min(r["speedup_geomean"] for r in rows),
        "speedup_max": max(r["speedup_geomean"] for r in rows),
        "tpot_speedup": geomean([r["tpot_speedup_geomean"] for r in rows]),
        "bootstrap_95": task_speed.interval(per_block) if len(per_block) >= 2 else None,
        "blocks": len(per_block),
        "acceptance_length": statistics.fmean(
            [r["acceptance_length"] for r in rows if r["acceptance_length"]] or [0]
        )
        or None,
        "slowdown_slices": sum(r["verdict"] == "slowdown" for r in rows),
    }
    for arm in ("plain", "dspark"):
        for metric in ("ttft", "tpot"):
            stats[f"{arm}_{metric}_p50_ms"] = geomean(
                [median(r, f"{arm}_{metric}_p50_ms") for r in rows]
            )
        stats[f"{arm}_tokens_s"] = geomean([median(r, f"{arm}_tokens_s") for r in rows])
    return stats


def analyse(suite, found: dict[str, Path]) -> dict:
    ks = draft_lengths(found)
    names = list(found)
    result = {"suite": suite["name"], "blocks": names, "ks": ks, "per_k": {}, "groups": []}
    for k in ks:
        pairs = {n: {"plain": b / "plain", "dspark": b / f"dspark-k{k}"} for n, b in found.items()}
        data = task_speed.cell_rows(suite, pairs, MATCH)
        result["per_k"][str(k)] = data
    for group, classes in GROUPS.items():
        for concurrency in suite["concurrencies"]:
            entry = {"group": group, "concurrency": concurrency, "k": {}}
            for k in ks:
                rows = group_rows(result["per_k"][str(k)]["rows"], classes, concurrency)
                entry["k"][str(k)] = group_stats(rows, names)
            measured = {k: s for k, s in entry["k"].items() if s}
            if measured:
                ranked = sorted(measured, key=lambda k: measured[k]["speedup"], reverse=True)
                entry["best_k"] = int(ranked[0])
                entry["best_speedup"] = measured[ranked[0]]["speedup"]
                entry["runner_up_gap"] = (
                    measured[ranked[0]]["speedup"] / measured[ranked[1]]["speedup"] - 1
                    if len(ranked) > 1
                    else None
                )
            result["groups"].append(entry)
    return result


def fmt(value, spec=".2f", suffix=""):
    return "–" if value is None else f"{value:{spec}}{suffix}"


def speed(stats, best=False) -> str:
    if stats is None:
        return "–"
    text = f"{stats['speedup']:.2f}x"
    if stats["bootstrap_95"]:
        low, high = stats["bootstrap_95"]
        text += f" [{low:.2f}, {high:.2f}]"
    return f"**{text}**" if best else text


def render(suite, result) -> str:
    ks, cs = result["ks"], suite["concurrencies"]
    by = {(g["group"], g["concurrency"]): g for g in result["groups"]}
    lines = [
        "# DSpark e10 at k = " + ", ".join(map(str, ks)) + ": production speedups",
        "",
        "Production engine (Apertus-v1.5-8B, TP1 bf16, max-model-len 32768, gpu-mem 0.8, "
        "max-num-batched-tokens 16384, max-num-seqs 256, no prefix caching, chat template; "
        "live image and overlays), controlled on Clariden: every block runs plain and each k "
        "on one GPU, one after the other. Speedup = DSpark output tokens/s ÷ plain output "
        "tokens/s on the same prompts; groups are geometric means of their slices with the "
        "exact bootstrap 95% interval over blocks. Greedy, thinking off, natural EOS at each "
        "source's output cap, 32 prompts per slice, whole passes per level.",
        "",
        f"Blocks: {', '.join(result['blocks'])}.",
        "",
        "## Headline: best k per load",
        "",
        "| Group | " + " | ".join(f"C={c}" for c in cs) + " |",
        "|---" * (len(cs) + 1) + "|",
    ]
    for group in GROUPS:
        cells = []
        for c in cs:
            g = by.get((group, c))
            if not g or "best_k" not in g:
                cells.append("–")
                continue
            tie = g["runner_up_gap"] is not None and g["runner_up_gap"] <= TIE
            cells.append(f"{g['best_speedup']:.2f}x (k={g['best_k']}{', ~tie' if tie else ''})")
        lines.append(f"| {group} | " + " | ".join(cells) + " |")
    lines += [
        "",
        f"~tie: the runner-up k is within {TIE:.0%}. `all` = all 28 slices equally weighted.",
        "",
        "## Speedup by group, k and C",
        "",
        "| Group | k | " + " | ".join(f"C={c}" for c in cs) + " | Acc. length |",
        "|---" * (len(cs) + 3) + "|",
    ]
    for group in GROUPS:
        for k in ks:
            stats = [by[(group, c)]["k"][str(k)] if (group, c) in by else None for c in cs]
            best = [by.get((group, c), {}).get("best_k") == k for c in cs]
            length = next((s["acceptance_length"] for s in stats if s), None)
            lines.append(
                f"| {group} | {k} | "
                + " | ".join(speed(s, b) for s, b in zip(stats, best, strict=True))
                + f" | {fmt(length)} |"
            )
    lines += ["", "Bold: best k for that group and load.", ""]
    lines += ["## Latency by group (p50, ms; geomean over slices)", ""]
    for c in cs:
        lines += [
            f"### C={c}",
            "",
            "| Group | TTFT plain | "
            + " | ".join(f"TTFT k={k}" for k in ks)
            + " | TPOT plain | "
            + " | ".join(f"TPOT k={k}" for k in ks)
            + " | tok/s plain | "
            + " | ".join(f"tok/s k={k}" for k in ks)
            + " |",
            "|---" * (3 * len(ks) + 4) + "|",
        ]
        for group in GROUPS:
            entry = by.get((group, c))
            if not entry:
                continue
            first = next((s for s in entry["k"].values() if s), None)
            if first is None:
                continue
            row = [f"| {group}"]
            for metric, spec in (("ttft", ".0f"), ("tpot", ".2f"), ("tokens_s", ".0f")):
                key = f"_{metric}_p50_ms" if metric != "tokens_s" else "_tokens_s"
                row.append(fmt(first["plain" + key], spec))
                row += [fmt((entry["k"][str(k)] or {}).get("dspark" + key), spec) for k in ks]
            lines.append(" | ".join(row) + " |")
        lines.append("")
    lines += ["## Per-slice speedup", ""]
    rows = {
        (k, r["slice"], r["concurrency"]): r for k in ks for r in result["per_k"][str(k)]["rows"]
    }
    header = "| Slice | Class | " + " | ".join(f"C={c} k={k}" for c in cs for k in ks) + " |"
    lines += [header, "|---" * (len(cs) * len(ks) + 2) + "|"]
    for entry in suite["slices"]:
        values = []
        for c in cs:
            cells = {k: rows.get((k, entry["label"], c)) for k in ks}
            top = max((r["speedup_geomean"] for r in cells.values() if r), default=None)
            for k in ks:
                r = cells[k]
                text = fmt(r and r["speedup_geomean"], ".2f", "x")
                values.append(f"**{text}**" if r and r["speedup_geomean"] == top else text)
        lines.append(f"| {entry['label']} | {entry['class']} | " + " | ".join(values) + " |")
    lines += ["", "## Acceptance length per slice", ""]
    lines += ["| Slice | " + " | ".join(f"k={k}" for k in ks) + " |", "|---" * (len(ks) + 1) + "|"]
    for entry in suite["slices"]:
        values = []
        for k in ks:
            lengths = [
                rows[(k, entry["label"], c)]["acceptance_length"]
                for c in cs
                if (k, entry["label"], c) in rows
                and rows[(k, entry["label"], c)]["acceptance_length"]
            ]
            values.append(fmt(statistics.fmean(lengths) if lengths else None))
        lines.append(f"| {entry['label']} | " + " | ".join(values) + " |")
    for k in ks:
        data = result["per_k"][str(k)]
        if data["rejected"] or data["missing"]:
            lines += [
                "",
                f"**k={k}: rejected {len(data['rejected'])}, missing {len(data['missing'])}**",
                "",
            ]
            lines += [f"- {r}" for r in data["rejected"] + data["missing"]]
    lines.append("")
    return "\n".join(lines)


def write_headline(path: Path, suite, result) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            ["group", "concurrency", "k", "speedup", "ci_low", "ci_high", "tpot_speedup",
             "acceptance_length", "plain_tpot_ms", "dspark_tpot_ms", "plain_ttft_ms",
             "dspark_ttft_ms", "best_k"]
        )  # fmt: skip
        for g in result["groups"]:
            for k, s in g["k"].items():
                if not s:
                    continue
                ci = s["bootstrap_95"] or [None, None]
                writer.writerow(
                    [g["group"], g["concurrency"], k, f"{s['speedup']:.4f}",
                     ci[0] and f"{ci[0]:.4f}", ci[1] and f"{ci[1]:.4f}",
                     s["tpot_speedup"] and f"{s['tpot_speedup']:.4f}",
                     s["acceptance_length"] and f"{s['acceptance_length']:.3f}",
                     f"{s['plain_tpot_p50_ms']:.3f}", f"{s['dspark_tpot_p50_ms']:.3f}",
                     f"{s['plain_ttft_p50_ms']:.1f}", f"{s['dspark_ttft_p50_ms']:.1f}",
                     g.get("best_k")]
                )  # fmt: skip


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    default_suite = Path(__file__).parents[1] / "task-speed" / "suite.yaml"
    parser.add_argument("--suite", type=Path, default=default_suite)
    parser.add_argument("--run", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    suite = task_speed.load_suite(args.suite)
    result = analyse(suite, blocks(args.run))
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "k-grid-results.json").write_text(json.dumps(result, indent=2) + "\n")
    (args.output / "REPORT.md").write_text(render(suite, result))
    write_headline(args.output / "headline.csv", suite, result)
    summary = {
        k: {
            "rows": len(d["rows"]),
            "missing": len(d["missing"]),
            "rejected": len(d["rejected"]),
        }
        for k, d in result["per_k"].items()
    }
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
