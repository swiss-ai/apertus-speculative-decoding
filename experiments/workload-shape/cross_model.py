"""Cross-model comparison of the workload-shape study: Apertus 1.5 8B vs Qwen3-8B.

    uv run --no-project --with numpy --with matplotlib --with pyyaml \
        python experiments/workload-shape/cross_model.py APERTUS_RUN QWEN_RUN OUT_DIR

Each RUN holds ``aggregate.json`` and ``analysis.json`` of ``analyze.py``. Writes
``cross-model.json`` and ``figures/cross-model.png`` to OUT_DIR:

- between-slice spread of the target's mean (and median) entropy per model, and how
  the slices' entropies correlate across models (is the entropy trend clearer?);
- R² of k=3 acceptance on entropy, TV and both for every drafter;
- k-means (k=2) membership per drafter, and the cross-model agreement of the splits;
- per-slice contrasts EAGLE3 (Red Hat) vs EAGLE e10 and DFlash (Red Hat) vs DSpark e10:
  acceptance at k=3 and mean depth-1 TV (different targets, so these are
  drafter-and-target contrasts, not drafter-only ones);
- chat and non-English gap decompositions for all four drafters.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
SERIES = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100")
PAIRS = (("eagle3", "eagle"), ("dflash", "dspark"))
NAMES = {"dspark": "DSpark e10", "eagle": "EAGLE e10", "dflash": "DFlash", "eagle3": "EAGLE3"}
LANGS = ("lang-deu", "lang-fra", "lang-ita", "lang-spa", "lang-roh", "lang-zho", "lang-jpn",
         "lang-hin", "lang-ara", "lang-swa", "lang-yor", "lang-amh")  # fmt: skip


def spread(x: np.ndarray) -> dict:
    q25, q75 = np.quantile(x, [0.25, 0.75])
    return {
        "mean": float(x.mean()), "sd": float(x.std(ddof=1)), "min": float(x.min()),
        "max": float(x.max()), "range": float(x.max() - x.min()), "iqr": float(q75 - q25),
        "cv": float(x.std(ddof=1) / x.mean()),
    }  # fmt: skip


def rand_index(a: list[int], b: list[int]) -> float:
    n, agree = len(a), 0
    for i in range(n):
        for j in range(i + 1, n):
            agree += (a[i] == a[j]) == (b[i] == b[j])
    return agree / (n * (n - 1) / 2)


def main() -> None:
    ap_dir, qw_dir, out = (Path(p) for p in sys.argv[1:4])
    runs = {}
    for name, d in (("apertus-8b", ap_dir), ("qwen3-8b", qw_dir)):
        runs[name] = {
            "agg": json.loads((d / "aggregate.json").read_text()),
            "an": json.loads((d / "analysis.json").read_text()),
        }
    order = [w for w in runs["apertus-8b"]["an"]["workloads"] if w in runs["qwen3-8b"]["an"]["workloads"]]
    res: dict = {"slices": order, "models": {}}
    ent = {}
    for m, r in runs.items():
        f = r["an"]["features"]
        ent[m] = np.array([f[w]["ent_mean"] for w in order])
        med = np.array([f[w]["ent_median"] for w in order])
        res["models"][m] = {
            "target_entropy_mean_across_slices": spread(ent[m]),
            "target_entropy_median_across_slices": spread(med),
            "drafters": {},
        }
        for d in r["agg"].get("drafters", ("dspark", "eagle")):
            r2 = r["an"]["r2_slices"][f"{d}_k3"]
            k2 = r["an"]["clusters"][d]["k2"]
            dec = r["an"]["decomposition"][d]["slices"]
            gaps = {
                "chat": {k: dec["chat"][f"acc_k3_{k}"] for k in ("gap", "shape", "alignment")},
                "languages_pooled": {
                    k: sum(dec[w][f"acc_k3_{k}"] for w in LANGS) for k in ("gap", "shape", "alignment")
                },
            }
            for g in gaps.values():
                g["shape_share"] = g["shape"] / g["gap"] if g["gap"] < 0 else None
                g["alignment_share"] = g["alignment"] / g["gap"] if g["gap"] < 0 else None
            acc = np.array([f[w][f"{d}_acc_k3"] for w in order])
            res["models"][m]["drafters"][d] = {
                "r2_k3": {k: r2[k]["r2"] for k in ("target entropy (mean)", "TV d1 (mean)", "entropy + TV")},
                "loo_r2_k3": {k: r2[k]["loo_r2"] for k in ("target entropy (mean)", "TV d1 (mean)", "entropy + TV")},
                "acc_k3_across_slices": spread(acc),
                "k2_clusters": k2["members"],
                "k2_silhouette": k2["silhouette"],
                "k2_labels": k2["labels"],
                "gaps": gaps,
            }
    res["entropy_correlation_across_models"] = float(np.corrcoef(ent["apertus-8b"], ent["qwen3-8b"])[0, 1])
    contrasts = {}
    fa, fq = runs["apertus-8b"]["an"]["features"], runs["qwen3-8b"]["an"]["features"]
    for new, old in PAIRS:
        rows = {}
        for w in order:
            rows[w] = {
                f"{new}_acc_k3": fq[w][f"{new}_acc_k3"], f"{old}_acc_k3": fa[w][f"{old}_acc_k3"],
                f"{new}_tv1": fq[w][f"{new}_tv_mean"], f"{old}_tv1": fa[w][f"{old}_tv_mean"],
                "class": fa[w]["class"],
            }  # fmt: skip
        by_class = {}
        for c in ("id", "near-id", "ood-task", "ood-language"):
            ws = [w for w in order if fa[w]["class"] == c]
            by_class[c] = {
                "delta_acc_k3": float(np.mean([rows[w][f"{new}_acc_k3"] - rows[w][f"{old}_acc_k3"] for w in ws])),
                "delta_tv1": float(np.mean([rows[w][f"{new}_tv1"] - rows[w][f"{old}_tv1"] for w in ws])),
            }
        la = res["models"]["apertus-8b"]["drafters"][old]["k2_labels"]
        lq = res["models"]["qwen3-8b"]["drafters"][new]["k2_labels"]
        contrasts[f"{new}_vs_{old}"] = {
            "slices": rows,
            "by_class": by_class,
            "corr_acc_k3": float(np.corrcoef([rows[w][f"{new}_acc_k3"] for w in order],
                                             [rows[w][f"{old}_acc_k3"] for w in order])[0, 1]),
            "corr_tv1": float(np.corrcoef([rows[w][f"{new}_tv1"] for w in order],
                                          [rows[w][f"{old}_tv1"] for w in order])[0, 1]),
            "k2_rand_index": rand_index(la, lq),
        }  # fmt: skip
    res["contrasts"] = contrasts
    out.mkdir(parents=True, exist_ok=True)
    (out / "cross-model.json").write_text(json.dumps(res, indent=1) + "\n")

    # Figure: (a) slice mean entropy, Apertus vs Qwen; (b) R² of entropy and TV per drafter.
    plt.rcParams.update({"font.size": 9, "figure.facecolor": SURFACE, "savefig.facecolor": SURFACE})
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    ax = axes[0]
    ax.set_facecolor(SURFACE)
    ax.grid(color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    classes = [fa[w]["class"] for w in order]
    for j, c in enumerate(("id", "near-id", "ood-task", "ood-language")):
        sel = [i for i, k in enumerate(classes) if k == c]
        ax.scatter(ent["apertus-8b"][sel], ent["qwen3-8b"][sel], s=30, color=SERIES[j], label=c,
                   edgecolor=SURFACE, linewidth=1, zorder=3)  # fmt: skip
    for i, w in enumerate(order):
        ax.annotate(w, (ent["apertus-8b"][i], ent["qwen3-8b"][i]), xytext=(3, 2),
                    textcoords="offset points", fontsize=6, color=INK2)  # fmt: skip
    lim = [0, max(ent["apertus-8b"].max(), ent["qwen3-8b"].max()) * 1.05]
    ax.plot(lim, lim, color=INK2, lw=0.8, ls="--")
    ax.set_xlabel("Apertus 1.5 8B: target entropy, slice mean (nats)")
    ax.set_ylabel("Qwen3-8B: target entropy, slice mean (nats)")
    sa = res["models"]["apertus-8b"]["target_entropy_mean_across_slices"]["sd"]
    sq = res["models"]["qwen3-8b"]["target_entropy_mean_across_slices"]["sd"]
    ax.set_title(f"between-slice SD: Apertus {sa:.3f}, Qwen {sq:.3f}; r = "
                 f"{res['entropy_correlation_across_models']:.2f}", fontsize=8.5, loc="left")  # fmt: skip
    ax.legend(fontsize=7, frameon=False)
    ax = axes[1]
    ax.set_facecolor(SURFACE)
    ax.grid(color=GRID, linewidth=0.6, axis="y")
    ax.set_axisbelow(True)
    labels, e_r2, t_r2 = [], [], []
    for m, ds in (("apertus-8b", ("dspark", "eagle")), ("qwen3-8b", ("dflash", "eagle3"))):
        for d in ds:
            r2 = res["models"][m]["drafters"][d]["r2_k3"]
            labels.append(f"{NAMES[d]}\n({'Apertus' if m == 'apertus-8b' else 'Qwen3'})")
            e_r2.append(r2["target entropy (mean)"])
            t_r2.append(r2["TV d1 (mean)"])
    x = np.arange(len(labels))
    ax.bar(x - 0.2, e_r2, 0.38, color=SERIES[0], label="target entropy (mean)")
    ax.bar(x + 0.2, t_r2, 0.38, color=SERIES[1], label="TV(target, drafter) d1 (mean)")
    for i, (a, b) in enumerate(zip(e_r2, t_r2, strict=True)):
        ax.text(i - 0.2, a + 0.01, f"{a:.2f}", ha="center", fontsize=7, color=INK)
        ax.text(i + 0.2, b + 0.01, f"{b:.2f}", ha="center", fontsize=7, color=INK)
    ax.axhline(0.8, color=INK2, lw=0.8, ls=":")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=7)
    ax.set_ylim(0, 1.22)
    ax.set_ylabel("R² of k=3 acceptance across 28 slices")
    ax.legend(fontsize=7, frameon=False, loc="upper left", ncol=2)
    ax.set_title("dotted: pre-registered entropy threshold 0.8", fontsize=8.5, loc="left")
    for a in axes:
        for side in ("top", "right"):
            a.spines[side].set_visible(False)
    fig.tight_layout()
    (out / "figures").mkdir(exist_ok=True)
    fig.savefig(out / "figures" / "cross-model.png", dpi=160)
    plt.close(fig)
    print(json.dumps({m: {k: v for k, v in r.items() if k != "drafters"} for m, r in res["models"].items()}, indent=1))
    print("entropy corr", res["entropy_correlation_across_models"])
    for m, r in res["models"].items():
        for d, v in r["drafters"].items():
            print(m, d, {k: round(x, 3) for k, x in v["r2_k3"].items()}, v["gaps"])
    for k, v in contrasts.items():
        print(k, v["by_class"], round(v["corr_acc_k3"], 3), round(v["k2_rand_index"], 3))


if __name__ == "__main__":
    main()
