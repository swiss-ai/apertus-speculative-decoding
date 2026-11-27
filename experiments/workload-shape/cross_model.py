"""Cross-model comparisons of the workload-shape study against the Apertus 1.5 8B run.

    uv run --no-project --with numpy --with matplotlib --with pyyaml \
        python experiments/workload-shape/cross_model.py compare APERTUS_RUN RUN OUT_DIR \
        [--models apertus-8b qwen3-8b]
    ... cross_model.py report RUN --served-dir GATE/collected --apertus APERTUS_RUN \
        [--output-check RUN/output-check.json] [--label Qwen3-4B] [--name LABEL]

Both read each run's ``aggregate.json`` and ``analysis.json`` (analyze.py) and take the
drafters from the aggregate (block drafter first).

``compare`` writes ``cross-model.json`` and ``figures/cross-model.png`` to OUT_DIR:
between-slice spread of the target's mean (and median) entropy per model and how the
slices' entropies correlate across models; R² of k=3 acceptance on entropy, TV and
both for every drafter; k-means (k=2) membership per drafter and the cross-model
agreement of the splits; per-slice contrasts of each drafter with the reference
drafter in the same role (autoregressive head, block drafter; different targets, so
drafter-and-target contrasts); chat and non-English gap decompositions (summed
per-slice terms). Used for the Qwen3-8B run (EAGLE3 / DFlash vs EAGLE e10 / DSpark e10).

``report`` writes ``extended.json``, ``extended-tables.md`` and
``figures/{gate-replay-vs-served,sampled-vs-shape,cross-model}.png`` into RUN for a
run with one drafter and per-arm served files (collect_gate.py), as for the Qwen3-4B
run: (1) the pre-registered gate, replayed vs served acceptance per slice for greedy
k=3/k=7 (Pearson r >= 0.98, mean |diff| <= 0.1) and the T=1 replay (1 - TV chained)
against the T=1 arms; (2) R² of acceptance on entropy, TV and both, greedy and sampled,
k = 1/3/5/7, across slices (with LOO) and sequences; (3) the decision rule with the
non-English gap pooled over the 12 languages' step counts, for the run and for the
Apertus run, plus, with ``--output-check``, a post-hoc sensitivity without looping
slices; (4) cross-model spread, slice-wise correlation and k=2 cluster agreement.
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
SURFACE, INK, INK2, GRID, SERIES = an.SURFACE, an.INK, an.INK2, an.GRID, an.SERIES
KS = (1, 3, 5, 7)
NAMES = {"dspark": "DSpark e10", "eagle": "EAGLE e10", "dflash": "DFlash", "eagle3": "EAGLE3"}
FITS = ("target entropy (mean)", "TV d1 (mean)", "entropy + TV")
# Model key -> (axis label, short label, bar tag) for the compare figure.
MODELS = {
    "apertus-8b": ("Apertus 1.5 8B", "Apertus", "Apertus"),
    "qwen3-8b": ("Qwen3-8B", "Qwen", "Qwen3"),
    "qwen3-4b": ("Qwen3-4B", "Qwen", "Qwen3"),
}


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
        "per_slice": {
            w: {"replay": float(a), "served": float(b)}
            for w, a, b in zip(labels, r, s, strict=True)
        },
    }


def chain_bins(wl: dict, names, d: str, k: int = 3):
    nb = len(next(iter(wl.values()))[d]["chain"][f"k{k}"]["by_first_entropy"])
    n, a = np.zeros(nb), np.zeros(nb)
    for w in names:
        for j, b in enumerate(wl[w][d]["chain"][f"k{k}"]["by_first_entropy"]):
            if b["n"]:
                n[j] += b["n"]
                a[j] += b["n"] * b["acceptance_length"]
    with np.errstate(invalid="ignore"):
        return n, a / n


def decompose(wl: dict, group, d: str) -> dict:
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


def decision_pooled(
    analysis: dict, wl: dict, d: str, exclude: tuple[str, ...] = ()
) -> dict:
    """The pre-registered rule (Qwen3-4B write-up) on drafter ``d``'s replayed k=3
    acceptance, non-English gap pooled over the 12 languages; ``exclude`` drops slices
    (post-hoc sensitivity only)."""
    names = [w for w in analysis["workloads"] if w not in exclude]
    feat = analysis["features"]
    y = np.array([feat[w][f"{d}_acc_k3"] for w in names])
    r2 = {
        "target entropy (mean)": an.ols(np.array([feat[w]["ent_mean"] for w in names]), y),
        "TV d1 (mean)": an.ols(np.array([feat[w][f"{d}_tv_mean"] for w in names]), y),
    }
    ent, tv = r2["target entropy (mean)"]["r2"], r2["TV d1 (mean)"]["r2"]
    chat = decompose(wl, ["chat"], d)
    langs = decompose(wl, [w for w in an.LANGUAGES if w in wl and w not in exclude], d)
    per_lang = {w: decompose(wl, [w], d) for w in an.LANGUAGES if w in wl}
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
        "slices": len(names),
        "excluded": list(exclude),
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


def spread_report(x: np.ndarray) -> dict:
    return {
        "mean": float(x.mean()),
        "sd": float(x.std(ddof=1)),
        "cv": float(x.std(ddof=1) / x.mean()),
        "min": float(x.min()),
        "max": float(x.max()),
        "range": float(x.max() - x.min()),
        "iqr": float(np.subtract(*np.quantile(x, [0.75, 0.25]))),
    }


def spread_compare(x: np.ndarray) -> dict:
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


def compare(args: argparse.Namespace) -> None:
    ref, run = args.models
    out = args.out
    runs = {}
    for name, d in ((ref, args.reference), (run, args.run)):
        runs[name] = {
            "agg": json.loads((d / "aggregate.json").read_text()),
            "an": json.loads((d / "analysis.json").read_text()),
        }
    drafters = {m: an.drafters_of(r["agg"]) for m, r in runs.items()}
    # Pairs by role, autoregressive head first: (run's EAGLE, reference EAGLE), (block, block).
    pairs = list(zip(drafters[run], drafters[ref], strict=False))[::-1]
    order = [w for w in runs[ref]["an"]["workloads"] if w in runs[run]["an"]["workloads"]]
    res: dict = {"slices": order, "models": {}}
    ent = {}
    for m, r in runs.items():
        f = r["an"]["features"]
        ent[m] = np.array([f[w]["ent_mean"] for w in order])
        med = np.array([f[w]["ent_median"] for w in order])
        res["models"][m] = {
            "target_entropy_mean_across_slices": spread_compare(ent[m]),
            "target_entropy_median_across_slices": spread_compare(med),
            "drafters": {},
        }
        for d in drafters[m]:
            r2 = r["an"]["r2_slices"][f"{d}_k3"]
            k2 = r["an"]["clusters"][d]["k2"]
            dec = r["an"]["decomposition"][d]["slices"]
            gaps = {
                "chat": {k: dec["chat"][f"acc_k3_{k}"] for k in ("gap", "shape", "alignment")},
                "languages_pooled": {
                    k: sum(dec[w][f"acc_k3_{k}"] for w in an.LANGUAGES)
                    for k in ("gap", "shape", "alignment")
                },
            }
            for g in gaps.values():
                g["shape_share"] = g["shape"] / g["gap"] if g["gap"] < 0 else None
                g["alignment_share"] = g["alignment"] / g["gap"] if g["gap"] < 0 else None
            acc = np.array([f[w][f"{d}_acc_k3"] for w in order])
            res["models"][m]["drafters"][d] = {
                "r2_k3": {k: r2[k]["r2"] for k in FITS},
                "loo_r2_k3": {k: r2[k]["loo_r2"] for k in FITS},
                "acc_k3_across_slices": spread_compare(acc),
                "k2_clusters": k2["members"],
                "k2_silhouette": k2["silhouette"],
                "k2_labels": k2["labels"],
                "gaps": gaps,
            }
    res["entropy_correlation_across_models"] = float(np.corrcoef(ent[ref], ent[run])[0, 1])
    contrasts = {}
    fa, fq = runs[ref]["an"]["features"], runs[run]["an"]["features"]
    for new, old in pairs:
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
                "delta_acc_k3": float(
                    np.mean([rows[w][f"{new}_acc_k3"] - rows[w][f"{old}_acc_k3"] for w in ws])
                ),
                "delta_tv1": float(
                    np.mean([rows[w][f"{new}_tv1"] - rows[w][f"{old}_tv1"] for w in ws])
                ),
            }
        la = res["models"][ref]["drafters"][old]["k2_labels"]
        lq = res["models"][run]["drafters"][new]["k2_labels"]
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

    # Figure: (a) slice mean entropy, reference vs run; (b) R² of entropy and TV per drafter.
    plt.rcParams.update({"font.size": 9, "figure.facecolor": SURFACE, "savefig.facecolor": SURFACE})
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    ax = axes[0]
    ax.set_facecolor(SURFACE)
    ax.grid(color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    classes = [fa[w]["class"] for w in order]
    for j, c in enumerate(("id", "near-id", "ood-task", "ood-language")):
        sel = [i for i, k in enumerate(classes) if k == c]
        ax.scatter(ent[ref][sel], ent[run][sel], s=30, color=SERIES[j], label=c,
                   edgecolor=SURFACE, linewidth=1, zorder=3)  # fmt: skip
    for i, w in enumerate(order):
        ax.annotate(w, (ent[ref][i], ent[run][i]), xytext=(3, 2),
                    textcoords="offset points", fontsize=6, color=INK2)  # fmt: skip
    lim = [0, max(ent[ref].max(), ent[run].max()) * 1.05]
    ax.plot(lim, lim, color=INK2, lw=0.8, ls="--")
    ax.set_xlabel(f"{MODELS[ref][0]}: target entropy, slice mean (nats)")
    ax.set_ylabel(f"{MODELS[run][0]}: target entropy, slice mean (nats)")
    sa = res["models"][ref]["target_entropy_mean_across_slices"]["sd"]
    sq = res["models"][run]["target_entropy_mean_across_slices"]["sd"]
    r_models = res["entropy_correlation_across_models"]
    ax.set_title(f"between-slice SD: {MODELS[ref][1]} {sa:.3f}, {MODELS[run][1]} {sq:.3f}; "
                 f"r = {r_models:.2f}", fontsize=8.5, loc="left")  # fmt: skip
    ax.legend(fontsize=7, frameon=False)
    ax = axes[1]
    ax.set_facecolor(SURFACE)
    ax.grid(color=GRID, linewidth=0.6, axis="y")
    ax.set_axisbelow(True)
    labels, e_r2, t_r2 = [], [], []
    for m in (ref, run):
        for d in drafters[m]:
            r2 = res["models"][m]["drafters"][d]["r2_k3"]
            labels.append(f"{NAMES[d]}\n({MODELS[m][2]})")
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
    models = res["models"]
    summary = {m: {k: v for k, v in r.items() if k != "drafters"} for m, r in models.items()}
    print(json.dumps(summary, indent=1))
    print("entropy corr", res["entropy_correlation_across_models"])
    for m, r in res["models"].items():
        for d, v in r["drafters"].items():
            print(m, d, {k: round(x, 3) for k, x in v["r2_k3"].items()}, v["gaps"])
    for k, v in contrasts.items():
        print(k, v["by_class"], round(v["corr_acc_k3"], 3), round(v["k2_rand_index"], 3))


def report(args: argparse.Namespace) -> None:
    root = args.root
    label = args.label
    run_key = label.lower().replace("-", "_")
    analysis = json.loads((root / "analysis.json").read_text())
    agg = json.loads((root / "aggregate.json").read_text())
    wl = agg["workloads"]
    order = analysis["workloads"]
    feat = analysis["features"]
    d = an.drafters_of(agg)[0]
    figs = root / "figures"
    out: dict = {}

    def col(name, f=feat):
        return np.array([f[w][name] for w in order], float)

    sampled = {
        k: {w: wl[w][d]["sampled"][f"k{k}"]["acceptance_length"] for w in order}
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
        ("k3-greedy", {w: feat[w][f"{d}_acc_k3"] for w in order}),
        ("k7-greedy", {w: feat[w][f"{d}_acc_k7"] for w in order}),
        ("k3-t1", sampled[3]),
        ("k7-t1", sampled[7]),
    ):
        if arm in served:
            gate[arm] = gate_stats(replay, served[arm], order)
    greedy = [gate.get(a, {}).get("passes", False) for a in ("k3-greedy", "k7-greedy")]
    gate["passes"] = bool(all(greedy))
    out["gate"] = gate

    # 2. R^2, greedy and sampled
    ent, tv = col("ent_mean"), col(f"{d}_tv_mean")
    models = {
        "target entropy (mean)": ent,
        "TV d1 (mean)": tv,
        "entropy + TV": np.column_stack([ent, tv]),
    }
    r2 = {}
    for k in KS:
        r2[f"greedy_k{k}"] = {m: an.ols(x, col(f"{d}_acc_k{k}")) for m, x in models.items()}
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
    s_tv = np.array([float(s[f"{d}_tv1_mean"]) for s in seqs])
    seq_r2 = {}
    for kind, key in (("greedy", "acc"), ("sampled", "sacc")):
        for k in (3, 7):
            y = np.array([float(s[f"{d}_{key}_k{k}"]) for s in seqs])
            seq_r2[f"{kind}_k{k}"] = {
                "target entropy": an.ols(s_ent, y),
                "TV d1": an.ols(s_tv, y),
                "entropy + TV": an.ols(np.column_stack([s_ent, s_tv]), y),
            }
    out["r2_sequences"] = seq_r2

    # 3. decision rule, both targets
    out["decision"] = decision_pooled(analysis, wl, d)
    if args.output_check:
        check = json.loads(args.output_check.read_text())
        loops = tuple(
            w for w in order
            if check["slices"][w]["sequences_repeat4_over_0.5"] / check["slices"][w]["sequences"]
            > args.loop_threshold
        )  # fmt: skip
        sens = {
            "note": "post hoc, not pre-registered: slices where more than "
            f"{args.loop_threshold:.0%} of continuations loop (4-gram repeat share > 0.5) "
            "are dropped; sequence-level fits drop looping sequences",
            "looping_slices": list(loops),
            "decision": decision_pooled(analysis, wl, d, loops),
        }
        keep = [w for w in order if w not in loops]
        k_ent = np.array([feat[w]["ent_mean"] for w in keep])
        k_tv = np.array([feat[w][f"{d}_tv_mean"] for w in keep])
        sens["r2_slices"] = {}
        for kind, ys in (
            ("greedy_k3", [feat[w][f"{d}_acc_k3"] for w in keep]),
            ("sampled_k3", [sampled[3][w] for w in keep]),
            ("greedy_k7", [feat[w][f"{d}_acc_k7"] for w in keep]),
            ("sampled_k7", [sampled[7][w] for w in keep]),
        ):
            yv = np.array(ys)
            sens["r2_slices"][kind] = {
                "target entropy (mean)": an.ols(k_ent, yv),
                "TV d1 (mean)": an.ols(k_tv, yv),
                "entropy + TV": an.ols(np.column_stack([k_ent, k_tv]), yv),
            }
        ok = np.array([check["sequences"][s["seq"]]["rep4"] <= 0.5 for s in seqs])
        sens["sequences_kept"] = int(ok.sum())
        sens["r2_sequences"] = {}
        for kind, key in (("greedy_k3", "acc_k3"), ("sampled_k3", "sacc_k3")):
            yv = np.array([float(s[f"{d}_{key}"]) for s in seqs])[ok]
            sens["r2_sequences"][kind] = {
                "target entropy": an.ols(s_ent[ok], yv),
                "TV d1": an.ols(s_tv[ok], yv),
                "entropy + TV": an.ols(np.column_stack([s_ent[ok], s_tv[ok]]), yv),
            }
        out["sensitivity_without_loops"] = sens
    ap_analysis = json.loads((args.apertus / "analysis.json").read_text())
    ap_agg_full = json.loads((args.apertus / "aggregate.json").read_text())
    ap_agg = ap_agg_full["workloads"]
    ap_d = an.drafters_of(ap_agg_full)[0]
    out["decision_apertus_8b_same_rule"] = decision_pooled(ap_analysis, ap_agg, ap_d)

    # 4. cross-model
    ap_feat = ap_analysis["features"]
    cross = {}
    for name, key in (
        ("target entropy mean", "ent_mean"),
        ("target entropy median", "ent_median"),
        ("target top-1 mean", "top1p_mean"),
        (f"{an.SHORT[d]} TV d1 mean", f"{d}_tv_mean"),
        (f"{an.SHORT[d]} acc k3 (replay)", f"{d}_acc_k3"),
    ):
        q, a = col(key), col(key, ap_feat)
        cross[name] = {
            run_key: spread_report(q),
            "apertus_8b": spread_report(a),
            "slice_correlation": corr(q, a),
        }
    # Within-workload entropy effect: per-slice correlation of entropy and acceptance.
    cross["entropy_vs_acceptance_slice_r"] = {
        run_key: corr(ent, col(f"{d}_acc_k3")),
        "apertus_8b": corr(col("ent_mean", ap_feat), col(f"{d}_acc_k3", ap_feat)),
    }
    q2 = analysis["clusters"][d]["k2"]
    a2 = ap_analysis["clusters"][d]["k2"]
    ql, al = np.array(q2["labels"]), np.array(a2["labels"])
    same = int((ql == al).sum())
    swapped = same < len(order) - same  # cluster ids are arbitrary at k=2
    agree = len(order) - same if swapped else same
    switched = [w for w, x, y in zip(order, ql, al, strict=True) if (x == y) == swapped]
    cross["clusters_k2"] = {
        run_key: {"members": q2["members"], "silhouette": q2["silhouette"],
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
    ys = {"greedy": col(f"{d}_acc_k3"), "sampled T=1": np.array([sampled[3][w] for w in order])}
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
    fig.suptitle(f"{label} + {args.name}: greedy vs sampled (T=1) drafting across 28 slices",
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
         (f"{d}_acc_k3", f"replayed {an.SHORT[d]} acceptance length, k=3")),
        strict=True,
    ):  # fmt: skip
        an.style(ax)
        yy = np.arange(len(srt))
        ax.scatter([feat[w][key] for w in srt], yy, s=26, color=an.SERIES[0],
                   label=f"{label} + {args.name}", zorder=3)  # fmt: skip
        ax.scatter([ap_feat[w][key] for w in srt], yy, s=26, color=an.SERIES[1], marker="D",
                   label="Apertus 1.5 8B + DSpark e10", zorder=3)  # fmt: skip
        for i, w in enumerate(srt):
            ax.plot([feat[w][key], ap_feat[w][key]], [i, i], color=an.GRID, lw=1, zorder=1)
        ax.set_xlabel(xl)
    axes[0].set_yticks(range(len(srt)))
    axes[0].set_yticklabels(srt, fontsize=7)
    axes[0].legend(fontsize=7, frameon=False, loc="lower right")
    fig.suptitle(f"Same 896 prompts on both targets; slices sorted by {label} entropy",
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
            f"| {w} | {feat[w][f'{d}_acc_k3']:.2f} | {sampled[3][w]:.2f} "
            f"| {fmt(served.get('k3-greedy', {}).get(w))} | {fmt(served.get('k3-t1', {}).get(w))} "
            f"| {feat[w][f'{d}_acc_k7']:.2f} | {sampled[7][w]:.2f} "
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
    if "sensitivity_without_loops" in out:
        sd = out["sensitivity_without_loops"]["decision"]
        summary["sensitivity_without_loops"] = {
            k: sd[k] for k in ("slices", "excluded", "entropy_r2", "tv_r2", "verdict")
        } | {"non_english_shape_share": sd["non_english_pooled"]["shape_share"]}
    print(json.dumps(summary, indent=1, default=str))


def main() -> None:
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="command", required=True)
    c = sub.add_parser("compare", help="cross-model.json of a run against the reference run")
    c.add_argument("reference", type=Path)
    c.add_argument("run", type=Path)
    c.add_argument("out", type=Path)
    c.add_argument("--models", nargs=2, default=["apertus-8b", "qwen3-8b"], choices=MODELS,
                   metavar=("REFERENCE", "RUN"))  # fmt: skip
    c.set_defaults(func=compare)
    r = sub.add_parser("report", help="gate, sampled, decision and cross-model report of a run")
    r.add_argument("root", type=Path)
    r.add_argument("--served-dir", type=Path, required=True)
    r.add_argument("--apertus", type=Path, required=True)
    r.add_argument("--label", default="Qwen3-4B", help="the run's target, in figures and keys")
    r.add_argument("--name", default="Red Hat DSpark", help="the run's drafter, in figures")
    r.add_argument(
        "--output-check",
        type=Path,
        help="output_check.py result: adds a post-hoc sensitivity without looping slices",
    )
    r.add_argument("--loop-threshold", type=float, default=0.25)
    r.set_defaults(func=report)
    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
