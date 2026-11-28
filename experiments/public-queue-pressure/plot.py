"""Plot group completion times for the public Qwen queue-pressure experiment."""

import argparse
import json
from pathlib import Path
from statistics import median

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

JOBS = ("4060288", "4060290")
CONDITIONS = (("last", 17), ("last", 24), ("first", 17), ("first", 24))


def read_pairs(results_dir):
    """Match verifier timings within each node, receipt order, budget and rotation."""
    pairs = {condition: [] for condition in CONDITIONS}
    for job in JOBS:
        summary = json.loads((results_dir / f"job-{job}.json").read_text())
        for order, budget in CONDITIONS:
            for rotation in range(3):
                key = f"{order}-r{rotation}"
                arms = [
                    summary["cells"][f"{mode}-{budget}"][key] for mode in ("snapshot", "compact")
                ]
                if not all(arm["order_confirmed"] for arm in arms):
                    raise ValueError(f"Unexpected receipt order in job {job}, {budget} GiB, {key}")
                pairs[order, budget].append(tuple(arm["group_s"] for arm in arms))
    return pairs


def plot(pairs, output):
    ink, muted, rule = "#172C39", "#41596A", "#D3DDE0"
    dot, high_budget_dot = "#136E65", "#637D90"
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 13, "text.color": ink})
    fig = plt.figure(figsize=(12, 5.6), facecolor="white")
    fig.text(0.035, 0.91, "Group completion time with compact replay", weight="bold", fontsize=26)
    fig.text(
        0.035,
        0.836,
        "Qwen3.5-9B  ·  Native MTP, 3 draft steps  ·  64 requests  ·  T = 1, top-p = 1",
        color=muted,
        fontsize=14,
    )

    bounds = [0.30, 0.30, 0.40, 0.38]
    ax = fig.add_axes(bounds)
    ax.spines[["left", "top", "right"]].set_visible(False)
    ax.spines["bottom"].set_color(rule)
    ax.tick_params(axis="x", length=0, pad=7, colors=muted)
    ax.set_yticks([])
    ax.set_ylim(-0.5, 3.5)
    ax.set_xlim(-0.02, 1.22)
    ax.set_xticks([0.0, 0.3, 0.6, 0.9, 1.2])
    ax.grid(axis="x", color=rule, linewidth=0.7)
    ax.axvline(0, color="#9CABB2", linewidth=1)

    header_y = 0.71
    fig.text(0.035, header_y, "Long request", color=muted, fontsize=13)
    fig.text(0.19, header_y, "Cache", color=muted, fontsize=13)
    fig.text(bounds[0], header_y, "Time saved by compact replay (s)", color=muted, fontsize=13)
    fig.text(
        0.854, header_y + 0.048, "Median group time (s)", ha="center", weight="bold", fontsize=13
    )
    fig.text(0.796, header_y, "Snapshot", ha="center", color=muted, fontsize=13)
    fig.text(0.914, header_y, "Compact", ha="center", color=muted, fontsize=13)

    jitter = (-0.15, -0.09, -0.03, 0.03, 0.09, 0.15)
    for row, (order, budget) in enumerate(CONDITIONS):
        y = 3 - row
        fy = bounds[1] + bounds[3] * (y + 0.5) / 4
        snapshots, compacts = zip(*pairs[order, budget], strict=True)
        savings = [snapshot - compact for snapshot, compact in pairs[order, budget]]
        color = dot if budget == 17 else high_budget_dot
        ax.scatter(
            savings,
            [y + offset for offset in jitter],
            s=52,
            color=color,
            edgecolors="white",
            linewidths=0.8,
            zorder=3,
        )
        if budget == 17:
            fig.text(0.035, fy, f"Arrives {order}", va="center", weight="bold", fontsize=15)
        fig.text(0.19, fy, f"{budget} GiB", va="center", fontsize=15)
        fig.text(0.796, fy, f"{median(snapshots):.2f}", ha="center", va="center", fontsize=16)
        fig.text(
            0.914,
            fy,
            f"{median(compacts):.2f}",
            ha="center",
            va="center",
            weight="bold",
            fontsize=16,
        )

    fig.text(
        0.035,
        0.143,
        "Each dot is one matched group. Six pairs per row cover three prompt rotations "
        "on each of two GH200 nodes.",
        fontsize=12,
        color=muted,
    )
    fig.text(
        0.035,
        0.083,
        "All prompts contain 1,536 tokens. The long request generates 768 tokens "
        "and each of the other 63 generates 256.",
        fontsize=12,
        color=muted,
    )
    fig.savefig(output, dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "results_dir", type=Path, help="directory containing both job summary JSONs"
    )
    parser.add_argument("output", type=Path, help="output PNG path")
    args = parser.parse_args()
    plot(read_pairs(args.results_dir), args.output)
