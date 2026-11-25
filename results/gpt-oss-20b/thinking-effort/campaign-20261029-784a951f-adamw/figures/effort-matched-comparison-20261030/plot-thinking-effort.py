#!/usr/bin/env python3
"""Compare vanilla, effort-matched FT and mixed FT using immutable effort reports."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import shutil
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from statistics import median

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402
from matplotlib.ticker import FuncFormatter, MaxNLocator  # noqa: E402

EFFORTS = ("low", "medium", "high")
WORKLOADS = ("math", "code")
SERIES = (
    ("phase1", "dflash", "Vanilla DFlash", "#606770"),
    ("phase2", "matched", "Effort-matched FT", "#009E73"),
    ("phase2", "mixed", "Mixed FT", "#0072B2"),
)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase1-report", type=Path, required=True)
    parser.add_argument("--phase2-report", type=Path, required=True)
    parser.add_argument("--phase1-lock", type=Path, required=True)
    parser.add_argument("--phase2-lock", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    reports = {
        phase: json.loads(getattr(args, f"{phase}_report").read_text())
        for phase in ("phase1", "phase2")
    }
    locks = {
        phase: json.loads(getattr(args, f"{phase}_lock").read_text())
        for phase in ("phase1", "phase2")
    }
    for report in reports.values():
        assert not report["problems"]
        assert report["decision"] not in ("invalid", "pending")
    for key in ("target", "workload"):
        assert locks["phase1"][key]["sha256"] == locks["phase2"][key]["sha256"]
    assert locks["phase1"]["image_sha256"] == locks["phase2"]["image_sha256"]
    for key in ("temperature", "top_p", "top_k", "min_p", "max_tokens", "fixed_date"):
        assert locks["phase1"]["spec"][key] == locks["phase2"]["spec"][key]
    rows = []
    for phase, comparison, label, _ in SERIES:
        for block in range(3):
            for effort in EFFORTS:
                variant = effort if comparison == "matched" else comparison
                selected = [
                    r
                    for r in reports[phase]["rows"]
                    if (r["block"], r["variant"], r["effort"]) == (block, variant, effort)
                ]
                baselines = [
                    r
                    for r in reports[phase]["rows"]
                    if (r["block"], r["variant"], r["effort"]) == (block, "plain", effort)
                ]
                assert len(selected) == len(baselines) == 1
                for workload in WORKLOADS:
                    measured = selected[0]["strata"][workload]
                    baseline = baselines[0]["strata"][workload]
                    assert measured["requests"] == baseline["requests"] == 32
                    values = {
                        "accepted_draft_tokens_per_round": measured[
                            "accepted_draft_tokens_per_round"
                        ],
                        "decode_speedup_vs_plain": baseline["decode_ms_per_token"]
                        / measured["decode_ms_per_token"],
                        "e2e_speedup_vs_plain": baseline["e2e_p50_ms"] / measured["e2e_p50_ms"],
                    }
                    assert all(math.isfinite(v) and v > 0 for v in values.values())
                    rows.append(
                        {
                            "phase": phase,
                            "comparison": comparison,
                            "variant": variant,
                            "label": label,
                            "block": block,
                            "effort": effort,
                            "workload": workload,
                            "requests": 32,
                            **values,
                            "e2e_p50_ms": measured["e2e_p50_ms"],
                            "plain_e2e_p50_ms": baseline["e2e_p50_ms"],
                            "decode_ms_per_token": measured["decode_ms_per_token"],
                            "plain_decode_ms_per_token": baseline["decode_ms_per_token"],
                        }
                    )
    assert len(rows) == 54
    assert all(r["variant"] == r["effort"] for r in rows if r["comparison"] == "matched")
    output = args.output
    output.mkdir(parents=True, exist_ok=False)
    sources = output / "sources"
    sources.mkdir()
    inputs = {}
    for name in ("phase1_report", "phase2_report", "phase1_lock", "phase2_lock"):
        source = getattr(args, name)
        target = sources / (name.replace("_", "-") + ".json")
        shutil.copyfile(source, target)
        inputs[name] = {
            "source_path": str(source.resolve()),
            "copy": str(target.relative_to(output)),
            "sha256": digest(target),
        }
    shutil.copyfile(__file__, output / "plot-thinking-effort.py")
    data = output / "plot-data.csv"
    with data.open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 11,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.titleweight": "bold",
            "axes.labelcolor": "#333333",
            "axes.edgecolor": "#BBBBBB",
            "pdf.fonttype": 42,
        }
    )
    metrics = (
        ("accepted_draft_tokens_per_round", "Accepted draft tokens / round", "Acceptance"),
        ("decode_speedup_vs_plain", "Speedup vs plain (×)", "Decode speedup"),
        ("e2e_speedup_vs_plain", "Speedup vs plain (×)", "End-to-end latency speedup"),
    )
    figure, axes = plt.subplots(2, 3, figsize=(17, 9.5), sharex=True)
    for ri, workload in enumerate(WORKLOADS):
        for ci, (metric, ylabel, title) in enumerate(metrics):
            ax = axes[ri, ci]
            width = 0.22
            for si, (phase, comparison, label, color) in enumerate(SERIES):
                x = [i + (si - 1) * width for i in range(3)]
                central = []
                for i, effort in enumerate(EFFORTS):
                    observations = [
                        r[metric]
                        for r in rows
                        if (r["phase"], r["comparison"], r["workload"], r["effort"])
                        == (phase, comparison, workload, effort)
                    ]
                    assert len(observations) == 3
                    central.append(median(observations))
                    ax.scatter(
                        [x[i] - 0.035, x[i], x[i] + 0.035],
                        observations,
                        s=22,
                        facecolor="white",
                        edgecolor="#222222",
                        linewidths=0.8,
                        zorder=4,
                    )
                ax.bar(
                    x,
                    central,
                    color=color,
                    width=width * 0.92,
                    alpha=0.88,
                    label=label,
                    zorder=2,
                )
            ax.set_title(f"{workload.capitalize()} · {title}", pad=13)
            ax.set_ylabel(ylabel)
            ax.set_xticks([0, 1, 2], ["Low", "Medium", "High"])
            ax.set_xlabel("Target thinking effort")
            ax.tick_params(axis="x", labelbottom=True)
            ax.set_xlim(-0.5, 2.5)
            ax.grid(axis="y", color="#E5E7EB", linewidth=0.8, zorder=0)
            maximum = max(r[metric] for r in rows)
            ax.set_ylim(0, maximum * 1.12)
            ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
            if ci:
                ax.axhline(1, color="#888888", linestyle=":", linewidth=1)
                ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.1f}×"))
    figure.suptitle(
        "GPT-OSS-20B: vanilla DFlash vs effort-matched FT vs mixed FT",
        fontsize=18,
        fontweight="bold",
        y=0.975,
    )
    figure.text(
        0.5,
        0.933,
        "32 math + 32 code prompts · temperature 1 · 3 paired repeats · "
        "fine-tunes: AdamW, 1.57M supervised tokens per head",
        ha="center",
        fontsize=11,
        color="#555555",
    )
    handles = [Patch(facecolor=color, alpha=0.88, label=label) for _, _, label, color in SERIES]
    figure.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.103),
        ncol=3,
        frameon=False,
        columnspacing=2.2,
        handlelength=2.6,
    )
    figure.text(
        0.5,
        0.069,
        "Bars: median of 3 repeats; dots: individual repeats. "
        "Matched FT: low-trained at low effort, medium-trained at medium, high-trained at high.",
        ha="center",
        fontsize=10,
        color="#444444",
    )
    figure.text(
        0.5,
        0.041,
        "Higher is better. Speedups use each run’s paired plain baseline. "
        "Vanilla comes from separate baseline runs; no contemporaneous vanilla-vs-FT pairing.",
        ha="center",
        fontsize=9.5,
        color="#666666",
    )
    figure.subplots_adjust(
        left=0.065, right=0.985, top=0.87, bottom=0.205, hspace=0.40, wspace=0.28
    )
    for suffix in ("png", "pdf", "svg"):
        figure.savefig(output / f"workload-effort-comparison.{suffix}", dpi=180, facecolor="white")
    plt.close(figure)
    files = {
        str(f.relative_to(output)): {"sha256": digest(f), "bytes": f.stat().st_size}
        for f in sorted(output.rglob("*"))
        if f.is_file()
    }
    provenance = {
        "created_at": datetime.now(UTC).isoformat(),
        "argv": sys.argv,
        "source_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "matplotlib_version": matplotlib.__version__,
        "sources": inputs,
        "target_sha256": locks["phase2"]["target"]["sha256"],
        "corpus_sha256": locks["phase2"]["workload"]["sha256"],
        "data_rows": len(rows),
        "comparison_scope": "At each effort: vanilla, that effort's FT head, and mixed FT only.",
        "matched_head_mapping": {effort: effort for effort in EFFORTS},
        "repeats": 3,
        "requests_per_workload_effort_head_repeat": 32,
        "aggregation": "Median of three per-block metrics, stratified by workload and effort.",
        "e2e_speedup": (
            "plain stratum p50 latency / drafter stratum p50 latency, same phase/block/effort"
        ),
        "decode_speedup": (
            "plain stratum decode ms/token / drafter stratum decode ms/token; "
            "same phase/block/effort"
        ),
        "cross_phase_limit": (
            "Default is contextual Phase 1 data; "
            "not a contemporaneous paired test against fine-tunes."
        ),
        "files": files,
    }
    (output / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    print(json.dumps({"output": str(output), "plot_data_rows": len(rows), "files": list(files)}))


if __name__ == "__main__":
    main()
