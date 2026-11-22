"""Slide chart: DSpark e10 speedup over plain by load, one line per k, per group.

plot-k-headline.py REPORT_DIR   (reads k-grid-results.json, writes k-headline.png)
"""

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# Ordinal one-hue ramp (k grows light -> dark), validated with the dataviz validator.
COLORS = {"3": "#6da7ec", "5": "#2a78d6", "7": "#104281"}
MARKERS = {"3": "o", "5": "s", "7": "D"}
GROUPS = [
    ("all", "All 28 slices"),
    ("id", "In distribution"),
    ("near-id", "Near ID"),
    ("ood-task", "OOD task"),
    ("ood-language", "OOD language"),
]
INK, MUTED, GRID, SURFACE = "#1f1f1e", "#6b6a64", "#e6e5e0", "#fcfcfb"

root = Path(sys.argv[1])
result = json.loads((root / "k-grid-results.json").read_text())
by = {(g["group"], g["concurrency"]): g for g in result["groups"]}
cs = sorted({g["concurrency"] for g in result["groups"]})
fig, axes = plt.subplots(1, len(GROUPS), figsize=(15, 3.8), sharey=True, facecolor=SURFACE)
for ax, (group, title) in zip(axes, GROUPS, strict=True):
    ax.set_facecolor(SURFACE)
    ax.axhline(1.0, color=MUTED, lw=1, ls="--", zorder=1)
    for k in result["ks"]:
        ys = [by[(group, c)]["k"][str(k)]["speedup"] for c in cs]
        ax.plot(range(len(cs)), ys, color=COLORS[str(k)], lw=2, marker=MARKERS[str(k)],
                ms=6, label=f"k={k}", zorder=3)  # fmt: skip
    for i, c in enumerate(cs):
        best = by[(group, c)]
        ax.annotate(f"{best['best_speedup']:.2f}x", (i, best["best_speedup"]),
                    textcoords="offset points", xytext=(0, 8), ha="center",
                    fontsize=8, color=INK)  # fmt: skip
    ax.set_title(title, color=INK, fontsize=11, loc="left")
    ax.set_xticks(range(len(cs)), [f"C={c}" for c in cs], color=MUTED, fontsize=9)
    ax.tick_params(axis="y", colors=MUTED, labelsize=9)
    ax.grid(axis="y", color=GRID, lw=0.8)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
axes[0].set_ylabel("Output tok/s vs plain (x)", color=MUTED, fontsize=9)
axes[0].legend(frameon=False, fontsize=9, loc="upper right", labelcolor=INK)
fig.suptitle(
    "Apertus-v1.5-8B + DSpark e10, production engine: speedup by load and draft length "
    "(labels: best k; dashed: plain)",
    color=INK, fontsize=12, x=0.01, ha="left",
)  # fmt: skip
fig.tight_layout()
fig.savefig(root / "k-headline.png", dpi=160, facecolor=SURFACE)
