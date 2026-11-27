"""Gate, sampled drafting, decision rule and cross-model comparison (Qwen3-4B repeat).

    uv run --no-project --with numpy --with matplotlib --with pyyaml \
        python experiments/workload-shape/extend.py RESULTS_DIR --served-dir DIR \
        --apertus results/8b/workload-shape/full-20261122-4004306

Runs after ``analyze.py RESULTS_DIR --served ...`` (it reads ``analysis.json``,
``aggregate.json`` and ``sequences.csv``) and leaves those files untouched. Writes
``extended.json``, ``extended-tables.md`` and ``figures/{gate,sampled,cross-model}*.png``.

1. Gate (pre-registered): replayed vs served acceptance per slice, greedy k=3 and
   k=7 (Pearson r >= 0.98 and mean |diff| <= 0.1 tokens/step), and the T=1 replay
   (1 - TV chained) against the served T=1 arms (a check of that approximation).
2. R^2 of acceptance on mean target entropy, mean depth-1 TV and both, greedy and
   sampled, k = 1/3/5/7, across slices (with leave-one-out) and sequences.
3. The pre-registered decision rule, with the pooled non-English decomposition
   (12 ``lang-*`` slices except lang-eng) next to chat's; the same numbers for the
   Apertus run.
4. Cross-model: between-slice spread of target entropy, slice-wise correlation of
   entropy / TV / acceptance, and k=2 cluster membership on both targets.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import analyze as an  # noqa: E402

plt = an.plt
KS = (1, 3, 5, 7)
LANGS_NON_ENGLISH = tuple(
    f"lang-{c}"
    for c in ("deu", "fra", "ita", "spa", "roh", "zho", "jpn", "hin", "ara", "swa", "yor", "amh")
)


def corr(a, b) -> float:
    return float(np.corrcoef(a, b)[0, 1])


def gate_stats(replay: dict, served: dict, order: list[str]) -> dict:
    labels = [w for w in order if w in served and w in replay]
    if len(labels) < 3:
        return {"n": len(labels), "missing": [w for w in order if w not in served]}
    r = np.array([replay[w] for w in labels])
    s = np.array([served[w] for w in labels])
    return {
        "n": len(labels),
        "missing": [w for w in order if w not in served],
        "pearson_r": corr(r, s),
        "mean_abs_diff": float(np.abs(r - s).mean()),
        "max_abs_diff": float(np.abs(r - s).max()),
        "max_abs_diff_slice": labels[int(np.abs(r - s).argmax())],
        "mean_diff_replay_minus_served": float((r - s).mean()),
        "passes": bool(corr(r, s) >= 0.98 and np.abs(r - s).mean() <= 0.1 and len(labels) == 28),
        "per_slice": {w: {"replay": float(a), "served": float(b)} for w, a, b in zip(labels, r, s)},
    }


def chain_bins(wl: dict, names, d: str = "dspark", k: int = 3):
    nb = len(next(iter(wl.values()))[d]["chain"][f"k{k}"]["by_first_entropy"])
    n, a = np.zeros(nb), np.zeros(nb)
    for w in names:
        for j, b in enumerate(wl[w][d]["chain"][f"k{k}"]["by_first_entropy"]):
            if b["n"]:
                n[j] += b["n"]
                a[j] += b["n"] * b["acceptance_length"]
    with np.errstate(invalid="ignore"):
        return n, a / n


def decompose(wl: dict, group, d: str = "dspark") -> dict:
    """Shape/alignment split of the pooled k=3 step acceptance of ``group`` vs the ID
    reference (identical to analyze.py's per-slice terms, pooled over the group)."""
    n_ref, a_ref = chain_bins(wl, an.REFERENCE, d)
    s_ref = n_ref / n_ref.sum()
    n_w, a_w = chain_bins(wl, group, d)
    s_w = n_w / n_w.sum()
    aw = np.nan_to_num(a_w, nan=0.0)
    gap = float((s_w * aw).sum() - (s_ref * a_ref).sum())
    shape = float(((s_w - s_ref) * a_ref).sum())
    return {
        "slices": list(group),
        "acc_k3_steps": float((s_w * aw).sum()),
        "reference_acc_k3": float((s_ref * a_ref).sum()),
        "gap": gap,
        "shape": shape,
        "alignment": gap - shape,
        "shape_share": shape / gap if gap else None,
        "alignment_share": (gap - shape) / gap if gap else None,
    }


def decision(analysis: dict, wl: dict) -> dict:
    r2 = analysis["r2_slices"]["dspark_k3"]
    ent, tv = r2["target entropy (mean)"]["r2"], r2["TV d1 (mean)"]["r2"]
    chat = decompose(wl, ["chat"])
    langs = decompose(wl, [w for w in LANGS_NON_ENGLISH if w in wl])
    per_lang = {w: decompose(wl, [w]) for w in LANGS_NON_ENGLISH if w in wl}
    entropy_holds = (
        ent >= 0.8 and (chat["shape_share"] or 0) >= 0.5 and (langs["shape_share"] or 0) >= 0.5
    )
    alignment_holds = tv - ent >= 0.3 and (chat["alignment_share"] or 0) >= 0.7
    verdict = (
        "entropy hypothesis holds" if entropy_holds and not alignment_holds
        else "alignment generalises" if alignment_holds and not entropy_holds
        else "both criteria met (contradictory)" if entropy_holds and alignment_holds
        else "mixed"
    )  # fmt: skip
    return {
        "entropy_r2": ent,
        "entropy_loo_r2": r2["target entropy (mean)"]["loo_r2"],
        "tv_r2": tv,
        "tv_loo_r2": r2["TV d1 (mean)"]["loo_r2"],
        "tv_minus_entropy_r2": tv - ent,
        "chat": chat,
        "non_english_pooled": langs,
        "non_english_per_language_shape_share": {
            w: v["shape_share"] for w, v in per_lang.items()
        },
        "criteria": {
            "entropy: R2 >= 0.8": ent >= 0.8,
            "entropy: shape >= 50% of chat gap": (chat["shape_share"] or 0) >= 0.5,
            "entropy: shape >= 50% of non-English gap": (langs["shape_share"] or 0) >= 0.5,
            "alignment: TV R2 - entropy R2 >= 0.3": tv - ent >= 0.3,
            "alignment: alignment >= 70% of chat gap": (chat["alignment_share"] or 0) >= 0.7,
        },
        "entropy_hypothesis_holds": entropy_holds,
        "alignment_generalises": alignment_holds,
        "verdict": verdict,
    }


def spread(x: np.ndarray) -> dict:
    return {
        "mean": float(x.mean()),
        "sd": float(x.std(ddof=1)),
        "cv": float(x.std(ddof=1) / x.mean()),
        "min": float(x.min()),
        "max": float(x.max()),
        "range": float(x.max() - x.min()),
        "iqr": float(np.subtract(*np.quantile(x, [0.75, 0.25]))),
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("root", type=Path)
    p.add_argument("--served-dir", type=Path, required=True)
    p.add_argument("--apertus", type=Path, required=True)
    p.add_argument("--name", default="Red Hat DSpark")
    args = p.parse_args()
    root = args.root
    analysis = json.loads((root / "analysis.json").read_text())
    agg = json.loads((root / "aggregate.json").read_text())
    wl = agg["workloads"]
    order = analysis["workloads"]
    feat = analysis["features"]
    figs = root / "figures"
    out: dict = {}

    def col(name, f=feat):
        return np.array([f[w][name] for w in order], float)

    sampled = {
        k: {w: wl[w]["dspark"]["sampled"][f"k{k}"]["acceptance_length"] for w in order}
        for k in KS
    }
    served = {}
    for arm in ("k3-greedy", "k7-greedy", "k3-t1", "k7-t1"):
        path = args.served_dir / f"served-acceptance-{arm}.json"
        if path.is_file():
            served[arm] = {
                w: v["acceptance_length"] for w, v in json.loads(path.read_text())["slices"].items()
            }

    # 1. gate
    gate = {}
    for arm, replay in (
        ("k3-greedy", {w: feat[w]["dspark_acc_k3"] for w in order}),
        ("k7-greedy", {w: feat[w]["dspark_acc_k7"] for w in order}),
        ("k3-t1", sampled[3]),
        ("k7-t1", sampled[7]),
    ):
        if arm in served:
            gate[arm] = gate_stats(replay, served[arm], order)
    greedy = [gate.get(a, {}).get("passes", False) for a in ("k3-greedy", "k7-greedy")]
    gate["passes"] = bool(all(greedy))
    out["gate"] = gate

    # 2. R^2, greedy and sampled
    ent, tv = col("ent_mean"), col("dspark_tv_mean")
    models = {
        "target entropy (mean)": ent,
        "TV d1 (mean)": tv,
        "entropy + TV": np.column_stack([ent, tv]),
    }
    r2 = {}
    for k in KS:
        r2[f"greedy_k{k}"] = {m: an.ols(x, col(f"dspark_acc_k{k}")) for m, x in models.items()}
        y = np.array([sampled[k][w] for w in order])
        r2[f"sampled_k{k}"] = {m: an.ols(x, y) for m, x in models.items()}
    for arm in served:
        labels = [w for w in order if w in served[arm]]
        if len(labels) == len(order):
            y = np.array([served[arm][w] for w in order])
            r2[f"served_{arm}"] = {m: an.ols(x, y) for m, x in models.items()}
    out["r2_slices"] = r2
    seqs = list(csv.DictReader((root / "sequences.csv").open()))
    s_ent = np.array([float(s["target_ent_mean"]) for s in seqs])
    s_tv = np.array([float(s["dspark_tv1_mean"]) for s in seqs])
    seq_r2 = {}
    for kind, key in (("greedy", "acc"), ("sampled", "sacc")):
        for k in (3, 7):
            y = np.array([float(s[f"dspark_{key}_k{k}"]) for s in seqs])
            seq_r2[f"{kind}_k{k}"] = {
                "target entropy": an.ols(s_ent, y),
                "TV d1": an.ols(s_tv, y),
                "entropy + TV": an.ols(np.column_stack([s_ent, s_tv]), y),
            }
    out["r2_sequences"] = seq_r2

    # 3. decision rule, both targets
    out["decision"] = decision(analysis, wl)
    ap_analysis = json.loads((args.apertus / "analysis.json").read_text())
    ap_agg = json.loads((args.apertus / "aggregate.json").read_text())["workloads"]
    out["decision_apertus_8b_same_rule"] = decision(ap_analysis, ap_agg)

    # 4. cross-model
    ap_feat = ap_analysis["features"]
    cross = {}
    for name, key in (
        ("target entropy mean", "ent_mean"),
        ("target entropy median", "ent_median"),
        ("target top-1 mean", "top1p_mean"),
        ("DSpark TV d1 mean", "dspark_tv_mean"),
        ("DSpark acc k3 (replay)", "dspark_acc_k3"),
    ):
        q, a = col(key), col(key, ap_feat)
        cross[name] = {
            "qwen3_4b": spread(q),
            "apertus_8b": spread(a),
            "slice_correlation": corr(q, a),
        }
    # Within-workload entropy effect: per-slice correlation of entropy and acceptance.
    cross["entropy_vs_acceptance_slice_r"] = {
        "qwen3_4b": corr(ent, col("dspark_acc_k3")),
        "apertus_8b": corr(col("ent_mean", ap_feat), col("dspark_acc_k3", ap_feat)),
    }
    q2 = analysis["clusters"]["dspark"]["k2"]
    a2 = ap_analysis["clusters"]["dspark"]["k2"]
    ql, al = np.array(q2["labels"]), np.array(a2["labels"])
    same = int((ql == al).sum())
    swapped = same < len(order) - same  # cluster ids are arbitrary at k=2
    agree = len(order) - same if swapped else same
    switched = [w for w, x, y in zip(order, ql, al, strict=True) if (x == y) == swapped]
    cross["clusters_k2"] = {
        "qwen3_4b": {"members": q2["members"], "silhouette": q2["silhouette"],
                     "mean_acc_k3": q2["mean_acc_k3"], "mean_entropy": q2["mean_entropy"],
                     "mean_tv": q2["mean_tv"]},  # fmt: skip
        "apertus_8b": {"members": a2["members"], "silhouette": a2["silhouette"],
                       "mean_acc_k3": a2["mean_acc_k3"], "mean_entropy": a2["mean_entropy"],
                       "mean_tv": a2["mean_tv"]},  # fmt: skip
        "same_side_slices": int(agree),
        "switched": switched,
    }
    out["cross_model"] = cross
    (root / "extended.json").write_text(json.dumps(out, indent=1) + "\n")

    # ------------------------------------------------------------------ figures
    plt.rcParams.update(
        {"font.size": 9, "figure.facecolor": an.SURFACE, "savefig.facecolor": an.SURFACE}
    )
    classes = [feat[w]["class"] for w in order]
    gate_arms = [a for a in ("k3-greedy", "k7-greedy", "k3-t1", "k7-t1") if a in gate]
    if gate_arms:
        fig, axes = plt.subplots(1, len(gate_arms), figsize=(3.3 * len(gate_arms), 3.4),
                                 squeeze=False)  # fmt: skip
        for ax, arm in zip(axes[0], gate_arms, strict=True):
            an.style(ax)
            g = gate[arm]
            if "per_slice" not in g:
                continue
            for j, c in enumerate(an.CLASSES):
                sel = [w for w, cl in zip(order, classes, strict=True)
                       if cl == c and w in g["per_slice"]]  # fmt: skip
                ax.scatter([g["per_slice"][w]["served"] for w in sel],
                           [g["per_slice"][w]["replay"] for w in sel], s=26,
                           marker=an.MARKERS[c], color=an.SERIES[j], edgecolor=an.SURFACE,
                           linewidth=1, label=c, zorder=3)  # fmt: skip
            lo = min(min(v.values()) for v in g["per_slice"].values())
            hi = max(max(v.values()) for v in g["per_slice"].values())
            ax.plot([lo, hi], [lo, hi], color=an.INK2, lw=0.8, ls="--")
            kind = "greedy replay" if "greedy" in arm else "T=1 replay (1 - TV)"
            ax.set_title(f"{arm}: r = {g['pearson_r']:.3f}, mean |d| = {g['mean_abs_diff']:.3f}",
                         fontsize=8, loc="left")  # fmt: skip
            ax.set_xlabel("served acceptance length (vLLM)")
            ax.set_ylabel(kind)
        axes[0][0].legend(fontsize=6.5, frameon=False)
        fig.tight_layout()
        fig.savefig(figs / "gate-replay-vs-served.png", dpi=160)
        plt.close(fig)

    # greedy vs sampled acceptance against entropy and TV (k=3)
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.8), sharey=True)
    ys = {"greedy": col("dspark_acc_k3"), "sampled T=1": np.array([sampled[3][w] for w in order])}
    for ax, (x, xl, key) in zip(
        axes,
        ((ent, "target entropy, mean (nats)", "target entropy (mean)"),
         (tv, f"TV(target, {args.name}) depth 1, mean", "TV d1 (mean)")),
        strict=True,
    ):  # fmt: skip
        an.style(ax)
        for j, (kind, y) in enumerate(ys.items()):
            fit = r2["greedy_k3" if kind == "greedy" else "sampled_k3"][key]
            ax.scatter(x, y, s=24, color=an.SERIES[j], edgecolor=an.SURFACE, linewidth=1,
                       label=f"{kind}: R² {fit['r2']:.2f}", zorder=3)  # fmt: skip
            xs = np.linspace(x.min(), x.max(), 10)
            ax.plot(xs, fit["coef"][0] + fit["coef"][1] * xs, color=an.SERIES[j], lw=1, ls="--")
        ax.set_xlabel(xl)
        ax.legend(fontsize=7, frameon=False)
    axes[0].set_ylabel("acceptance length, k=3 (replay)")
    fig.suptitle(f"Qwen3-4B + {args.name}: greedy vs sampled (T=1) drafting across 28 slices",
                 fontsize=8.5, color=an.INK, x=0.01, ha="left")  # fmt: skip
    fig.tight_layout()
    fig.savefig(figs / "sampled-vs-shape.png", dpi=160)
    plt.close(fig)

    # cross-model: per-slice entropy and acceptance, both targets
    srt = sorted(order, key=lambda w: feat[w]["ent_mean"])
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 7.0), sharey=True)
    for ax, (key, xl) in zip(
        axes,
        (("ent_mean", "target entropy, mean over generated tokens (nats)"),
         ("dspark_acc_k3", "replayed DSpark acceptance length, k=3")),
        strict=True,
    ):  # fmt: skip
        an.style(ax)
        yy = np.arange(len(srt))
        ax.scatter([feat[w][key] for w in srt], yy, s=26, color=an.SERIES[0],
                   label="Qwen3-4B + Red Hat DSpark", zorder=3)  # fmt: skip
        ax.scatter([ap_feat[w][key] for w in srt], yy, s=26, color=an.SERIES[1], marker="D",
                   label="Apertus 1.5 8B + DSpark e10", zorder=3)  # fmt: skip
        for i, w in enumerate(srt):
            ax.plot([feat[w][key], ap_feat[w][key]], [i, i], color=an.GRID, lw=1, zorder=1)
        ax.set_xlabel(xl)
    axes[0].set_yticks(range(len(srt)))
    axes[0].set_yticklabels(srt, fontsize=7)
    axes[0].legend(fontsize=7, frameon=False, loc="lower right")
    fig.suptitle("Same 896 prompts on both targets; slices sorted by Qwen3-4B entropy",
                 fontsize=8.5, color=an.INK, x=0.01, ha="left")  # fmt: skip
    fig.tight_layout()
    fig.savefig(figs / "cross-model.png", dpi=160)
    plt.close(fig)

    # ------------------------------------------------------------------ tables
    lines = ["## R² across slices (mean target entropy / mean depth-1 TV / both; LOO)", ""]
    lines.append("| acceptance | entropy | TV | entropy + TV |")
    lines.append("|---|---|---|---|")
    for key, v in r2.items():
        cells = [f"{v[m]['r2']:.2f} ({v[m]['loo_r2']:.2f})" for m in models]
        lines.append(f"| {key} | " + " | ".join(cells) + " |")
    lines += ["", "## Sequence-level R²", "", "| acceptance | entropy | TV | both |",
              "|---|---|---|---|"]  # fmt: skip
    for key, v in seq_r2.items():
        lines.append(f"| {key} | " + " | ".join(f"{x['r2']:.2f}" for x in v.values()) + " |")
    lines += ["", "## Greedy vs sampled acceptance per slice (k=3 / k=7)", "",
              "| slice | greedy k3 | sampled k3 | served k3 | served T=1 k3 | greedy k7 "
              "| sampled k7 | served k7 | served T=1 k7 |", "|---" * 9 + "|"]  # fmt: skip

    def fmt(v):
        return "-" if v is None else f"{v:.2f}"

    for w in order:
        lines.append(
            f"| {w} | {feat[w]['dspark_acc_k3']:.2f} | {sampled[3][w]:.2f} "
            f"| {fmt(served.get('k3-greedy', {}).get(w))} | {fmt(served.get('k3-t1', {}).get(w))} "
            f"| {feat[w]['dspark_acc_k7']:.2f} | {sampled[7][w]:.2f} "
            f"| {fmt(served.get('k7-greedy', {}).get(w))} | {fmt(served.get('k7-t1', {}).get(w))} |"
        )
    (root / "extended-tables.md").write_text("\n".join(lines) + "\n")
    summary = {
        "gate": {a: {k: v for k, v in g.items() if k != "per_slice"} if isinstance(g, dict) else g
                 for a, g in gate.items()},  # fmt: skip
        "decision": {k: v for k, v in out["decision"].items()
                     if k not in ("non_english_per_language_shape_share",)},  # fmt: skip
        "apertus_same_rule": out["decision_apertus_8b_same_rule"]["verdict"],
    }
    print(json.dumps(summary, indent=1, default=str))


if __name__ == "__main__":
    main()
