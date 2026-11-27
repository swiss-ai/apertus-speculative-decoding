"""Analysis and figures of the workload distribution-shape study.

    uv run --no-project --with numpy --with matplotlib --with pyyaml \
        python experiments/workload-shape/analyze.py RESULTS_DIR

RESULTS_DIR holds ``aggregate.json`` and ``sequences.csv`` (from
``apertus_tree.shape_aggregate``); ``inputs/`` next to it the served acceptance and
the suite. Writes ``analysis.json``, ``tables.md`` and ``figures/*.png`` there.

Other targets and drafters (the Qwen3-8B repeat with Red Hat's EAGLE-3 and DFlash):
the drafters come from ``aggregate.json`` (block drafter first), ``--served`` takes a
served-acceptance file with one entry per (drafter, k, sampling) cell
(``{"cells": [{"drafter", "k", "sampling", "slices": {slice: acceptance}}]}``) and
``--extended`` adds k = 1/3/5/7(/15) fits, the sampled (T=1) replay, the
pre-registered decision rule and per-drafter figures. Without them the output is
the Apertus run's.

A study with one drafter (the Qwen3-4B repeat: DSpark only) drops the second
drafter's columns and panels; ``--name dspark=LABEL`` relabels a drafter in figures.

Questions (Yu, 2026-11-20 sync):
1. How well do target entropy (shape) and TV(target, drafter) (alignment), alone
   and together, explain acceptance length across the 28 workload slices (R^2)?
2. Do they explain the chat and non-English deficits? Decomposition of each slice's
   gap to the ID reference into a shape term (different entropy mix at the ID
   slices' per-bin acceptance) and an alignment term (different acceptance within
   the same entropy bin).
3. A data-driven ID/OOD split: k-means clusters in (entropy, TV) space.
4. Are low-entropy tokens inside chat accepted as well as in math?
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib
import numpy as np
import yaml

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

DRAFTERS = ("dspark", "eagle")
NAMES = {
    "dspark": "DSpark e10",
    "eagle": "EAGLE e10",
    "dflash": "DFlash (Red Hat)",
    "eagle3": "EAGLE3 (Red Hat)",
}
SHORT = {"dspark": "DSpark", "eagle": "EAGLE", "dflash": "DFlash", "eagle3": "EAGLE3"}
LANGUAGES = ("lang-deu", "lang-fra", "lang-ita", "lang-spa", "lang-roh", "lang-zho", "lang-jpn",
             "lang-hin", "lang-ara", "lang-swa", "lang-yor", "lang-amh")  # fmt: skip
# Pre-registered decision rule (docs/workload-distribution-shape-qwen3-8b.md).
RULE = {
    "entropy_r2_min": 0.8,
    "entropy_shape_share_min": 0.5,
    "tv_minus_entropy_r2_min": 0.3,
    "alignment_chat_share_min": 0.7,
}
CLASSES = ("id", "near-id", "ood-task", "ood-language")
REFERENCE = ("opb", "math-gsm8k", "probe-math", "probe-humaneval", "multi-turn")
# Reference palette (dataviz skill), light mode.
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
SERIES = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100")
MARKERS = {"id": "o", "near-id": "s", "ood-task": "^", "ood-language": "D"}


def style(ax):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(INK2)
        ax.spines[side].set_linewidth(0.8)
    ax.tick_params(colors=INK2, labelsize=8)
    ax.grid(color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    ax.xaxis.label.set_color(INK)
    ax.yaxis.label.set_color(INK)
    ax.title.set_color(INK)


def ols(x: np.ndarray, y: np.ndarray) -> dict:
    """y ~ 1 + x (x [n, p]). R^2, adjusted R^2, coefficients, LOO R^2."""
    x = np.atleast_2d(x.T).T if x.ndim == 1 else x
    n, p = x.shape
    a = np.column_stack([np.ones(n), x])
    coef, *_ = np.linalg.lstsq(a, y, rcond=None)
    resid = y - a @ coef
    ss_tot = ((y - y.mean()) ** 2).sum()
    r2 = 1 - (resid**2).sum() / ss_tot
    hat = a @ np.linalg.pinv(a)
    loo = resid / (1 - np.diag(hat))
    return {
        "r2": float(r2),
        "adj_r2": float(1 - (1 - r2) * (n - 1) / (n - p - 1)),
        "loo_r2": float(1 - (loo**2).sum() / ss_tot),
        "coef": [float(c) for c in coef],
        "n": int(n),
    }


def kmeans(x: np.ndarray, k: int, seeds: int = 50):
    best = None
    for seed in range(seeds):
        rng = np.random.default_rng(seed)
        c = x[rng.choice(len(x), k, replace=False)]
        for _ in range(200):
            lab = ((x[:, None] - c[None]) ** 2).sum(-1).argmin(1)
            new = np.array([x[lab == j].mean(0) if (lab == j).any() else c[j] for j in range(k)])
            if np.allclose(new, c):
                break
            c = new
        inertia = ((x - c[lab]) ** 2).sum()
        if best is None or inertia < best[0] - 1e-12:
            best = (inertia, lab, c)
    return best


def silhouette(x: np.ndarray, lab: np.ndarray) -> float:
    d = np.sqrt(((x[:, None] - x[None]) ** 2).sum(-1))
    s = []
    for i in range(len(x)):
        same = lab == lab[i]
        if same.sum() <= 1:
            s.append(0.0)
            continue
        a = d[i, same].sum() / (same.sum() - 1)
        b = min(d[i, lab == j].mean() for j in set(lab.tolist()) if j != lab[i])
        s.append((b - a) / max(a, b))
    return float(np.mean(s))


def served_cells(path: Path, primary: str) -> list[dict]:
    """Served acceptance cells; the Apertus file is DSpark k=3 greedy only."""
    data = json.loads(path.read_text())
    if "cells" in data:
        return data["cells"]
    slices = {w: v.get("acceptance_length") for w, v in data["slices"].items()}
    return [{"drafter": primary, "k": 3, "sampling": "greedy", "slices": slices}]


def decision(r2: dict, decomp: dict, d: str) -> dict:
    """The pre-registered rule for one drafter (k=3 greedy replay, slice level)."""
    ent = r2[f"{d}_k3"]["target entropy (mean)"]["r2"]
    tv = r2[f"{d}_k3"]["TV d1 (mean)"]["r2"]
    rows = decomp[d]["slices"]

    def share(names, term):
        gap = sum(rows[w]["acc_k3_gap"] for w in names if w in rows)
        part = sum(rows[w][f"acc_k3_{term}"] for w in names if w in rows)
        # A share of a deficit; undefined when the slices are not below the reference.
        return {"gap": gap, term: part, "share": part / gap if gap < 0 else None}

    chat_shape, chat_align = share(["chat"], "shape"), share(["chat"], "alignment")
    lang_shape = share(LANGUAGES, "shape")
    per_lang = {w: share([w], "shape")["share"] for w in LANGUAGES if w in rows}

    def ge(x, t):
        return x is not None and x >= t

    entropy_holds = (
        ent >= RULE["entropy_r2_min"]
        and ge(chat_shape["share"], RULE["entropy_shape_share_min"])
        and ge(lang_shape["share"], RULE["entropy_shape_share_min"])
    )
    alignment_holds = tv - ent >= RULE["tv_minus_entropy_r2_min"] and ge(
        chat_align["share"], RULE["alignment_chat_share_min"]
    )
    verdict = (
        "entropy hypothesis holds" if entropy_holds and not alignment_holds
        else "alignment generalises" if alignment_holds and not entropy_holds
        else "both criteria met" if entropy_holds
        else "mixed"
    )  # fmt: skip
    return {
        "rule": RULE,
        "entropy_r2": ent,
        "tv_r2": tv,
        "tv_minus_entropy_r2": tv - ent,
        "chat_shape": chat_shape,
        "chat_alignment": chat_align,
        "languages_shape_pooled": lang_shape,
        "languages_shape_share_each": per_lang,
        "entropy_hypothesis_holds": entropy_holds,
        "alignment_generalises": alignment_holds,
        "verdict": verdict,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("inputs", type=Path, nargs="?")
    parser.add_argument("--served", type=Path, help="served acceptance (default: inputs/"
                        "served-acceptance-task-speed.json)")  # fmt: skip
    parser.add_argument("--extended", action="store_true")
    parser.add_argument("--name", action="append", default=[], metavar="DRAFTER=LABEL")
    args = parser.parse_args()
    root = args.root
    inputs = args.inputs or root.parent / "inputs"
    labels = dict(item.split("=", 1) for item in args.name)
    NAMES.update(labels)
    agg = json.loads((root / "aggregate.json").read_text())
    first = next(iter(agg["workloads"].values()))
    drafters = tuple(agg.get("drafters") or (d for d in DRAFTERS if d in first))
    primary, secondary = (drafters + (None,))[:2]
    extended = args.extended
    cells = served_cells(args.served or inputs / "served-acceptance-task-speed.json", primary)
    served = {
        w: {"acceptance_length": v}
        for c in cells
        if c["drafter"] == primary and c["k"] == 3 and c["sampling"] == "greedy"
        for w, v in c["slices"].items()
    }
    suite = yaml.safe_load((inputs / "suite.yaml").read_text())
    order = [s["label"] for s in suite["slices"] if s["label"] in agg["workloads"]]
    wl = agg["workloads"]
    bins = agg["entropy_bins"]
    figs = root / "figures"
    figs.mkdir(exist_ok=True)
    out: dict = {"workloads": order}

    # ------------------------------------------------------------------ per-slice features
    feat = {}
    for w in order:
        e = wl[w]
        row = {
            "class": e["class"],
            "ent_mean": e["target"]["ent"]["mean"],
            "ent_median": e["target"]["ent"]["q50"],
            "top1p_mean": e["target"]["top1p"]["mean"],
            "served_k3": served.get(w, {}).get("acceptance_length"),
        }
        for d in drafters:
            r = e[d]
            row[f"{d}_tv_mean"] = r["depth1"]["tv"]["mean"]
            row[f"{d}_tv_median"] = r["depth1"]["tv"]["q50"]
            row[f"{d}_tv13_mean"] = r["depth1to3"]["tv"]["mean"]
            row[f"{d}_kl_mean"] = r["depth1"]["kl"]["mean"]
            row[f"{d}_ent_mean"] = r["depth1"]["ent"]["mean"]
            row[f"{d}_match1"] = r["depth1"]["match"]["mean"]
            for k in agg.get("ks", (1, 3, 5, 7)):
                row[f"{d}_acc_k{k}"] = r["chain"][f"k{k}"]["acceptance_length"]
            for k, v in ((r.get("sampled") or {}) if extended else {}).items():
                row[f"{d}_sacc_{k}"] = v["acceptance_length"]
        if extended:
            for c in cells:
                key = f"served_{c['drafter']}_{c['sampling']}_k{c['k']}"
                row[key] = c["slices"].get(w)
        feat[w] = row
    out["features"] = feat

    # ------------------------------------------------------------------ replay vs served
    def compare(pairs):
        rep, srv = np.array(pairs).T
        return {
            "pearson_r": float(np.corrcoef(rep, srv)[0, 1]),
            "mean_abs_diff": float(np.abs(rep - srv).mean()),
            "mean_diff_replay_minus_served": float((rep - srv).mean()),
            "n": len(pairs),
        }

    if not extended:
        pairs = [
            (feat[w][f"{primary}_acc_k3"], feat[w]["served_k3"])
            for w in order
            if feat[w]["served_k3"]
        ]
        out[f"replay_vs_served_{primary}_k3"] = compare(pairs)
    else:
        # The validation gate: per drafter and served cell, replay vs served per slice.
        gate = {}
        for c in cells:
            d, k, mode = c["drafter"], c["k"], c["sampling"]
            replay = f"{d}_acc_k{k}" if mode == "greedy" else f"{d}_sacc_k{k}"
            pairs = [
                (feat[w][replay], c["slices"][w])
                for w in order
                if c["slices"].get(w) is not None and feat[w].get(replay) is not None
            ]
            if len(pairs) < 3:
                continue
            res = compare(pairs)
            if mode == "greedy":
                res["gate"] = {"pearson_r_min": 0.98, "mean_abs_diff_max": 0.1}
                res["passed"] = res["pearson_r"] >= 0.98 and res["mean_abs_diff"] <= 0.1
            gate[f"{d}_{mode}_k{k}"] = res
        out["replay_vs_served"] = gate

    # ------------------------------------------------------------------ R^2 across slices
    def col(name):
        return np.array([feat[w][name] for w in order], float)

    r2 = {}
    r2_ks = tuple(agg.get("ks", (1, 3, 5, 7))) if extended else (3, 7)
    for d in drafters:
        for k in r2_ks:
            y = col(f"{d}_acc_k{k}")
            models = {
                "target entropy (mean)": col("ent_mean"),
                "target entropy (median)": col("ent_median"),
                "target top-1 prob (mean)": col("top1p_mean"),
                "drafter entropy (mean)": col(f"{d}_ent_mean"),
                "TV d1 (mean)": col(f"{d}_tv_mean"),
                "TV d1-3 (mean)": col(f"{d}_tv13_mean"),
                "KL d1 (mean)": col(f"{d}_kl_mean"),
                "entropy + TV": np.column_stack([col("ent_mean"), col(f"{d}_tv_mean")]),
                "entropy + TV + entropy x TV": np.column_stack(
                    [col("ent_mean"), col(f"{d}_tv_mean"), col("ent_mean") * col(f"{d}_tv_mean")]
                ),
                "depth-1 match rate": col(f"{d}_match1"),
            }
            if not np.all(np.isfinite(col(f"{d}_kl_mean"))) or col(f"{d}_kl_mean").max() > 1e6:
                # Reduced draft vocabulary: KL(p||q) is infinite wherever the target puts
                # mass outside it, so the KL fit is not defined.
                del models["KL d1 (mean)"]
            r2[f"{d}_k{k}"] = {name: ols(x, y) for name, x in models.items()}
            if extended and f"{d}_sacc_k{k}" in feat[order[0]]:
                ys = col(f"{d}_sacc_k{k}")
                r2[f"{d}_sampled_k{k}"] = {
                    "target entropy (mean)": ols(col("ent_mean"), ys),
                    "TV d1 (mean)": ols(col(f"{d}_tv_mean"), ys),
                    "entropy + TV": ols(np.column_stack([col("ent_mean"), col(f"{d}_tv_mean")]), ys),
                }
    if not extended:
        y = np.array([feat[w]["served_k3"] for w in order], float)
        r2[f"served_{primary}_k3"] = {
            "target entropy (mean)": ols(col("ent_mean"), y),
            "TV d1 (mean)": ols(col(f"{primary}_tv_mean"), y),
            "entropy + TV": ols(np.column_stack([col("ent_mean"), col(f"{primary}_tv_mean")]), y),
        }
    else:
        for c in cells:
            key = f"served_{c['drafter']}_{c['sampling']}_k{c['k']}"
            if any(feat[w].get(key) is None for w in order):
                continue
            y = col(key)
            tv = col(f"{c['drafter']}_tv_mean")
            r2[key] = {
                "target entropy (mean)": ols(col("ent_mean"), y),
                "TV d1 (mean)": ols(tv, y),
                "entropy + TV": ols(np.column_stack([col("ent_mean"), tv]), y),
            }
    # Partial: does TV explain what entropy leaves unexplained, and vice versa?
    for d in drafters:
        y = col(f"{d}_acc_k3")
        e_res = (
            y
            - np.column_stack([np.ones(len(y)), col("ent_mean")])
            @ np.linalg.lstsq(np.column_stack([np.ones(len(y)), col("ent_mean")]), y, rcond=None)[0]
        )
        r2[f"{d}_k3"]["TV on entropy residual"] = ols(col(f"{d}_tv_mean"), e_res)
        r2[f"{d}_k3"]["corr(entropy, TV) across slices"] = {
            "r": float(np.corrcoef(col("ent_mean"), col(f"{d}_tv_mean"))[0, 1])
        }
    out["r2_slices"] = r2

    # Sequence level (896 points): same models, robustness to slice aggregation.
    seqs = list(csv.DictReader((root / "sequences.csv").open()))
    seq_r2 = {}
    for d in drafters:
        y = np.array([float(s[f"{d}_acc_k3"]) for s in seqs])
        ent = np.array([float(s["target_ent_mean"]) for s in seqs])
        tv = np.array([float(s[f"{d}_tv1_mean"]) for s in seqs])
        seq_r2[d] = {
            "target entropy": ols(ent, y),
            "TV d1": ols(tv, y),
            "entropy + TV": ols(np.column_stack([ent, tv]), y),
            "corr(entropy, TV)": float(np.corrcoef(ent, tv)[0, 1]),
        }
        if extended:
            for k in agg.get("ks", (1, 3, 5, 7)):
                for prefix, label in (("acc", "greedy"), ("sacc", "sampled")):
                    if f"{d}_{prefix}_k{k}" not in seqs[0]:
                        continue
                    yk = np.array([float(s[f"{d}_{prefix}_k{k}"]) for s in seqs])
                    seq_r2[d][f"{label}_k{k}"] = {
                        "target entropy": ols(ent, yk),
                        "TV d1": ols(tv, yk),
                        "entropy + TV": ols(np.column_stack([ent, tv]), yk),
                    }
    out["r2_sequences"] = seq_r2

    # ------------------------------------------------------------------ decomposition
    def bin_rates(names, d):
        n = np.zeros(len(bins))
        m = np.zeros(len(bins))
        tv = np.zeros(len(bins))
        for w in names:
            for j, b in enumerate(wl[w][d]["bins"]["depth1"]):
                if b["n"]:
                    n[j] += b["n"]
                    m[j] += b["n"] * b["match"]
                    tv[j] += b["n"] * b["tv"]
        with np.errstate(invalid="ignore"):
            return n, m / n, tv / n

    def chain_bins(names, d, k=3):
        n = np.zeros(len(bins))
        a = np.zeros(len(bins))
        for w in names:
            for j, b in enumerate(wl[w][d]["chain"][f"k{k}"]["by_first_entropy"]):
                if b["n"]:
                    n[j] += b["n"]
                    a[j] += b["n"] * b["acceptance_length"]
        with np.errstate(invalid="ignore"):
            return n, a / n

    decomp = {}
    for d in drafters:
        n_ref, m_ref, _ = bin_rates(REFERENCE, d)
        s_ref = n_ref / n_ref.sum()
        cn_ref, ca_ref = chain_bins(REFERENCE, d)
        cs_ref = cn_ref / cn_ref.sum()
        rows = {}
        for w in order:
            n_w, m_w, _ = bin_rates([w], d)
            s_w = n_w / n_w.sum()
            mw = np.nan_to_num(m_w, nan=0.0)
            total = float((s_w * mw).sum() - (s_ref * m_ref).sum())
            shape_term = float(((s_w - s_ref) * m_ref).sum())
            cn_w, ca_w = chain_bins([w], d)
            cs_w = cn_w / cn_w.sum()
            caw = np.nan_to_num(ca_w, nan=0.0)
            c_total = float((cs_w * caw).sum() - (cs_ref * ca_ref).sum())
            c_shape = float(((cs_w - cs_ref) * ca_ref).sum())
            rows[w] = {
                "match1": float((s_w * mw).sum()),
                "match1_gap": total,
                "match1_shape": shape_term,
                "match1_alignment": total - shape_term,
                "acc_k3_steps": float((cs_w * caw).sum()),
                "acc_k3_gap": c_total,
                "acc_k3_shape": c_shape,
                "acc_k3_alignment": c_total - c_shape,
                "share_low_entropy": float(s_w[0]),
                "match1_low_entropy": None if np.isnan(m_w[0]) else float(m_w[0]),
            }
        decomp[d] = {
            "reference": list(REFERENCE),
            "reference_match1": float((s_ref * m_ref).sum()),
            "reference_acc_k3": float((cs_ref * ca_ref).sum()),
            "reference_bin_match1": [float(x) for x in m_ref],
            "slices": rows,
        }
    out["decomposition"] = decomp
    if extended:
        out["decision"] = {d: decision(r2, decomp, d) for d in drafters}

    # Class-pooled per-bin rates
    pooled = {}
    for d in drafters:
        pooled[d] = {}
        for c in CLASSES:
            names = [w for w in order if feat[w]["class"] == c]
            n, m, tv = bin_rates(names, d)
            cn, ca = chain_bins(names, d)
            pooled[d][c] = {
                "n": n.tolist(),
                "match1": [None if np.isnan(x) else float(x) for x in m],
                "tv1": [None if np.isnan(x) else float(x) for x in tv],
                "acc_k3_by_first_entropy": [None if np.isnan(x) else float(x) for x in ca],
                "share": (n / n.sum()).tolist(),
            }
        for w in ("chat", "math-gsm8k", "probe-math", "summarization", "lang-deu"):
            n, m, tv = bin_rates([w], d)
            pooled[d][w] = {
                "n": n.tolist(),
                "match1": [None if np.isnan(x) else float(x) for x in m],
                "tv1": [None if np.isnan(x) else float(x) for x in tv],
            }
    out["entropy_bins"] = bins
    out["pooled_bins"] = pooled

    # ------------------------------------------------------------------ clusters
    clusters = {}
    for d in drafters:
        x = np.column_stack([col("ent_mean"), col(f"{d}_tv_mean")])
        z = (x - x.mean(0)) / x.std(0)
        res = {}
        for k in (2, 3, 4):
            _inertia, lab, _c = kmeans(z, k)
            # Name clusters by ascending mean acceptance.
            acc = col(f"{d}_acc_k3")
            rank = np.argsort([acc[lab == j].mean() for j in range(k)])[::-1]
            relabel = {int(old): new for new, old in enumerate(rank)}
            lab = np.array([relabel[int(v)] for v in lab])
            res[f"k{k}"] = {
                "silhouette": silhouette(z, lab),
                "members": {
                    str(j): [w for w, v in zip(order, lab, strict=True) if v == j] for j in range(k)
                },
                "mean_acc_k3": [float(acc[lab == j].mean()) for j in range(k)],
                "mean_entropy": [float(x[lab == j, 0].mean()) for j in range(k)],
                "mean_tv": [float(x[lab == j, 1].mean()) for j in range(k)],
                "labels": lab.tolist(),
            }
        clusters[d] = res
    out["clusters"] = clusters
    (root / "analysis.json").write_text(json.dumps(out, indent=1) + "\n")

    # ------------------------------------------------------------------ figures
    plt.rcParams.update({"font.size": 9, "figure.facecolor": SURFACE, "savefig.facecolor": SURFACE})

    # 1. (entropy, TV) map with k=2 or best-silhouette clusters
    for d in drafters:
        best_k = max((2, 3), key=lambda k: clusters[d][f"k{k}"]["silhouette"])
        lab = np.array(clusters[d][f"k{best_k}"]["labels"])
        cl_acc = clusters[d][f"k{best_k}"]["mean_acc_k3"]
        fig, ax = plt.subplots(figsize=(7.2, 5.0))
        style(ax)
        for c in CLASSES:
            for j in range(best_k):
                sel = [i for i, w in enumerate(order) if feat[w]["class"] == c and lab[i] == j]
                if not sel:
                    continue
                ax.scatter(
                    col("ent_mean")[sel], col(f"{d}_tv_mean")[sel], s=46, marker=MARKERS[c],
                    color=SERIES[j], edgecolor=SURFACE, linewidth=1.5, zorder=3,
                )  # fmt: skip
        for i, w in enumerate(order):
            ax.annotate(
                f"{w} {feat[w][f'{d}_acc_k3']:.2f}", (col("ent_mean")[i], col(f"{d}_tv_mean")[i]),
                xytext=(4, 3), textcoords="offset points", fontsize=6.5, color=INK2,
            )  # fmt: skip
        handles = [
            plt.Line2D([], [], marker="o", ls="", color=SERIES[j],
                       label=f"cluster {j + 1} (acc {cl_acc[j]:.2f})")
            for j in range(best_k)
        ] + [
            plt.Line2D([], [], marker=MARKERS[c], ls="", color=INK2, label=f"label: {c}")
            for c in CLASSES
        ]  # fmt: skip
        ax.legend(handles=handles, fontsize=7, frameon=False, loc="upper left")
        ax.set_xlabel("target entropy, mean over generated tokens (nats)")
        ax.set_ylabel(f"TV(target, {NAMES[d]}) at depth 1, mean")
        ax.set_title(
            f"{NAMES[d]}: slices in (entropy, TV) space, k-means k={best_k}\n"
            "point labels: replayed k=3 acceptance length",
            fontsize=8.5, loc="left",
        )  # fmt: skip
        fig.tight_layout()
        fig.savefig(figs / f"shape-map-{d}.png", dpi=160)
        plt.close(fig)

    # 2. acceptance vs entropy and vs TV (k=3), R^2 in the panel titles
    fig_drafters = drafters if extended else (primary,)
    for d in fig_drafters:
        fig, axes = plt.subplots(1, 2, figsize=(9, 3.8), sharey=True)
        for ax, (xname, xlabel, key) in zip(
            axes,
            (
                ("ent_mean", "target entropy, mean (nats)", "target entropy (mean)"),
                (f"{d}_tv_mean", f"TV(target, {SHORT[d]}) depth 1, mean", "TV d1 (mean)"),
            ),
            strict=True,
        ):
            style(ax)
            x, y = col(xname), col(f"{d}_acc_k3")
            for j, c in enumerate(CLASSES):
                sel = [i for i, w in enumerate(order) if feat[w]["class"] == c]
                ax.scatter(x[sel], y[sel], marker=MARKERS[c], color=SERIES[j], s=40,
                           edgecolor=SURFACE, linewidth=1.5, label=c, zorder=3)  # fmt: skip
            fit = r2[f"{d}_k3"][key]
            xs = np.linspace(x.min(), x.max(), 10)
            ax.plot(xs, fit["coef"][0] + fit["coef"][1] * xs, color=INK2, lw=1, ls="--")
            ax.set_title(f"R² = {fit['r2']:.2f} (LOO {fit['loo_r2']:.2f})", fontsize=8.5, loc="left")
            ax.set_xlabel(xlabel)
        axes[0].set_ylabel(f"replayed acceptance length, {SHORT[d]} k=3")
        axes[0].legend(fontsize=7, frameon=False)
        fig.tight_layout()
        fig.savefig(figs / f"acceptance-vs-shape-{d}.png", dpi=160)
        plt.close(fig)

    # 3. per-slice distributions (q10..q90 whiskers, IQR box, median) sorted by acceptance
    srt = sorted(order, key=lambda w: feat[w][f"{primary}_acc_k3"])
    panels = (("target entropy (nats)", lambda w: wl[w]["target"]["ent"]),) + tuple(
        (
            f"TV(target, {labels.get(d, SHORT[d])}) depth 1",
            lambda w, d=d: wl[w][d]["depth1"]["tv"],
        )
        for d in drafters
    )
    fig, axes = plt.subplots(
        1, len(panels), figsize=(3.5 * len(panels), 7.2), sharey=True, squeeze=False
    )
    axes = axes[0]
    for ax, (title, get) in zip(axes, panels, strict=True):
        style(ax)
        for i, w in enumerate(srt):
            q = get(w)
            color = SERIES[CLASSES.index(feat[w]["class"])]
            ax.plot([q["q10"], q["q90"]], [i, i], color=color, lw=1)
            ax.plot([q["q25"], q["q75"]], [i, i], color=color, lw=5, solid_capstyle="round")
            ax.plot([q["q50"]], [i], marker="|", color=INK, ms=8)
            ax.plot([q["mean"]], [i], marker="o", color=SURFACE, mec=INK, ms=3.5)
        ax.set_title(title, fontsize=8.5, loc="left")
    axes[0].set_yticks(range(len(srt)))
    axes[0].set_yticklabels([f"{w} ({feat[w][f'{primary}_acc_k3']:.2f})" for w in srt], fontsize=7)
    handles = [plt.Line2D([], [], color=SERIES[j], lw=5, label=c) for j, c in enumerate(CLASSES)]
    axes[0].legend(handles=handles, fontsize=7, frameon=False, loc="lower right")
    fig.suptitle(
        "Per-token distributions per slice (bar = IQR, line = 10-90%, tick = median, "
        f"dot = mean); slices sorted by replayed {SHORT[primary]} k=3 acceptance (in brackets)",
        fontsize=8.5, color=INK, x=0.01, ha="left",
    )  # fmt: skip
    fig.tight_layout()
    fig.savefig(figs / "distributions-per-slice.png", dpi=160)
    plt.close(fig)

    # 4. depth-1 match rate by target-entropy bin, per class, both drafters
    fig, axes = plt.subplots(
        1, len(drafters), figsize=(4.5 * len(drafters), 3.8), sharey=True, squeeze=False
    )
    axes = axes[0]
    for ax, d in zip(axes, drafters, strict=True):
        style(ax)
        for j, c in enumerate(CLASSES):
            m = [np.nan if v is None else v for v in pooled[d][c]["match1"]]
            ax.plot(range(len(bins)), m, color=SERIES[j], lw=2, marker="o", ms=5,
                    mec=SURFACE, mew=1.5, label=c)  # fmt: skip
        m = [np.nan if v is None else v for v in pooled[d]["chat"]["match1"]]
        ax.plot(range(len(bins)), m, color=INK2, lw=1.2, ls="--", marker="x", ms=5, label="chat")
        ax.set_xticks(range(len(bins)))
        ax.set_xticklabels(bins, fontsize=7)
        ax.set_xlabel("target entropy at the drafted position (nats)")
        ax.set_title(NAMES[d], fontsize=8.5, loc="left")
    axes[0].set_ylabel("depth-1 greedy match rate")
    axes[0].legend(fontsize=7, frameon=False)
    fig.tight_layout()
    fig.savefig(figs / "match-by-entropy.png", dpi=160)
    plt.close(fig)

    # 5. gap decomposition per slice (k=3 step acceptance)
    for d in fig_drafters:
        dd = decomp[d]["slices"]
        srt2 = sorted(order, key=lambda w: dd[w]["acc_k3_gap"])
        fig, ax = plt.subplots(figsize=(7.5, 6.8))
        style(ax)
        yy = np.arange(len(srt2))
        sh = np.array([dd[w]["acc_k3_shape"] for w in srt2])
        al = np.array([dd[w]["acc_k3_alignment"] for w in srt2])
        ax.barh(yy - 0.2, sh, height=0.38, color=SERIES[0], label="shape (entropy mix)")
        ax.barh(yy + 0.2, al, height=0.38, color=SERIES[1], label="alignment (within entropy bin)")
        ax.axvline(0, color=INK2, lw=0.8)
        ax.set_yticks(yy)
        ax.set_yticklabels(srt2, fontsize=7)
        ax.set_xlabel("contribution to the gap in k=3 acceptance length vs. the ID reference")
        ax.set_title(
            f"{NAMES[d]}: acceptance gap to the ID slices = shape term + alignment term",
            fontsize=8.5, loc="left",
        )  # fmt: skip
        ax.legend(fontsize=7, frameon=False, loc="upper left")
        fig.tight_layout()
        fig.savefig(figs / f"gap-decomposition-{d}.png", dpi=160)
        plt.close(fig)

    # 6. (extended) greedy vs sampled replay and served cells, per drafter
    if extended:
        fig, axes = plt.subplots(1, len(drafters), figsize=(9, 3.9), sharey=True)
        for ax, d in zip(axes, drafters, strict=True):
            style(ax)
            x = col("ent_mean")
            for j, (key, label) in enumerate(
                ((f"{d}_acc_k3", "greedy replay"), (f"{d}_sacc_k3", "sampled (T=1) replay"))
            ):
                if key not in feat[order[0]]:
                    continue
                y = col(key)
                fit = ols(x, y)
                ax.scatter(x, y, s=26, color=SERIES[j], edgecolor=SURFACE, linewidth=1,
                           label=f"{label}: R² vs entropy {fit['r2']:.2f}", zorder=3)  # fmt: skip
            for c in cells:
                if c["drafter"] == d and c["k"] == 3:
                    key = f"served_{d}_{c['sampling']}_k3"
                    y = np.array([np.nan if feat[w].get(key) is None else feat[w][key]
                                  for w in order], float)  # fmt: skip
                    ax.scatter(x, y, s=16, marker="x", color=INK2 if c["sampling"] == "greedy"
                               else SERIES[2], label=f"served {c['sampling']}", zorder=4)  # fmt: skip
            ax.set_title(f"{NAMES[d]}, k=3", fontsize=8.5, loc="left")
            ax.set_xlabel("target entropy, mean (nats)")
            ax.legend(fontsize=6.5, frameon=False)
        axes[0].set_ylabel("acceptance length (tokens per step)")
        fig.tight_layout()
        fig.savefig(figs / "greedy-vs-sampled.png", dpi=160)
        plt.close(fig)

    # ------------------------------------------------------------------ tables
    P, S = SHORT[primary], secondary and SHORT[secondary]
    lines = ["## Per-slice statistics", ""]
    lines.append(
        f"| slice | class | target H median [IQR] | target top-1 median | {P} TV d1 median [IQR] "
        f"| {P} match d1 | {P} acc k3 replay / served"
        + (
            f" | {S} TV d1 median [IQR] "
            f"| {S} match d1 | {S} acc k3{' replay / served' if extended else ''} |"
            if secondary
            else " |"
        )
    )
    lines.append("|---" * (10 if secondary else 7) + "|")

    def fmt(v):
        return "n/a" if v is None else f"{v:.2f}"

    for w in order:
        e = wl[w]
        t = e["target"]["ent"]
        ds = e[primary]["depth1"]
        served_k3 = feat[w]["served_k3"]
        row = (
            f"| {w} | {e['class']} | {t['q50']:.3f} [{t['q25']:.3f}, {t['q75']:.3f}] "
            f"| {e['target']['top1p']['q50']:.3f} "
            f"| {ds['tv']['q50']:.3f} [{ds['tv']['q25']:.3f}, {ds['tv']['q75']:.3f}] "
            f"| {ds['match']['mean']:.3f} | {feat[w][f'{primary}_acc_k3']:.2f} / "
            f"{fmt(served_k3)} "
        )
        if secondary:
            ea = e[secondary]["depth1"]
            s_served = (
                f" / {fmt(feat[w].get(f'served_{secondary}_greedy_k3'))}" if extended else ""
            )
            row += (
                f"| {ea['tv']['q50']:.3f} [{ea['tv']['q25']:.3f}, {ea['tv']['q75']:.3f}] "
                f"| {ea['match']['mean']:.3f} | {feat[w][f'{secondary}_acc_k3']:.2f}{s_served} |"
            )
        else:
            row += "|"
        lines.append(row)
    if extended:
        ks_all = [k for k in agg.get("ks", (1, 3, 5, 7))]
        lines += ["", "## Acceptance length per k (greedy replay; sampled T=1 replay in brackets)", ""]
        head = "| slice | " + " | ".join(f"{SHORT[d]} k{k}" for d in drafters for k in ks_all) + " |"
        lines += [head, "|---" * (1 + len(drafters) * len(ks_all)) + "|"]
        for w in order:
            cells_txt = []
            for d in drafters:
                for k in ks_all:
                    g = feat[w][f"{d}_acc_k{k}"]
                    sv = feat[w].get(f"{d}_sacc_k{k}")
                    cells_txt.append(f"{g:.2f}" + (f" ({sv:.2f})" if sv is not None else ""))
            lines.append(f"| {w} | " + " | ".join(cells_txt) + " |")
    (root / "tables.md").write_text("\n".join(lines) + "\n")
    if not extended:
        print(json.dumps({k: out[k] for k in (f"replay_vs_served_{primary}_k3",)}, indent=1))
        keys = tuple(f"{d}_k3" for d in drafters) + (f"served_{primary}_k3",)
    else:
        print(json.dumps(out["replay_vs_served"], indent=1))
        print(json.dumps(out["decision"], indent=1))
        keys = tuple(k for k in r2 if k.endswith("_k3"))
    for key in keys:
        print(key, {m: round(v.get("r2", v.get("r", 0)), 3) for m, v in r2[key].items()})
    for d in drafters:
        print(d, "seq", {m: round(v["r2"], 3) if isinstance(v, dict) and "r2" in v else
                         (round(v, 3) if not isinstance(v, dict) else "...")
                         for m, v in seq_r2[d].items()})  # fmt: skip
        for k in (2, 3, 4):
            c = clusters[d][f"k{k}"]
            print(
                d,
                k,
                round(c["silhouette"], 3),
                c["members"],
                [round(a, 2) for a in c["mean_acc_k3"]],
            )


if __name__ == "__main__":
    main()
