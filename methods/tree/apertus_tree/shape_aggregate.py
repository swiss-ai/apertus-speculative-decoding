"""Aggregate the per-token shape shards of ``apertus_tree.shape``.

    python3 -m apertus_tree.shape_aggregate SHARD.npz ... --suite SUITE --output OUT.json \
        --sequences OUT-sequences.csv

Everything is reported per workload slice, as quantiles, never as a bare mean:

- target: entropy (nats) and top-1 probability at every generated position;
- per drafter at depth 1 (the next token after the anchor) and pooled over depths
  1..3 (the deployed k=3 chain), conditional on the depth being reached:
  entropy, top-1 probability, TV(target, drafter), KL(target || drafter), the
  greedy acceptance test (drafter top-1 == trajectory token), and agreement of
  the two top-1 tokens;
- realised acceptance: a greedy chain replayed on the trajectory at k = 1, 3, 5, 7
  (acceptance length per step = accepted drafts + 1, as vLLM counts it), with its
  per-step distribution and per-sequence quantiles;
- the depth-1 match rate and TV by target-entropy bin, per workload, and the
  step acceptance length by the entropy of the step's first drafted position, which
  separate the distribution-shape effect from the workload effect.

Shards of other drafters (``apertus_tree.shape --names``) and deeper replays
(``--depth`` 15 for DFlash) are detected from the arrays: the drafter keys come
from the ``*_match`` fields and k = 15 is added when the depth allows it.
``--sampled`` adds the sampled (T=1) acceptance replay: under rejection sampling
the drafted token at depth d is accepted with probability 1 - TV(p_d, q_d), so the
expected acceptance length of a step starting at anchor a is
1 + sum_{j<=k} prod_{d<=j} (1 - TV_d), evaluated along the greedy trajectory at
every anchor. Without these the output is the Apertus run's, byte for byte.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import yaml

DRAFTERS = ("dspark", "eagle")
FIELDS = ("ent", "top1p", "tv", "kl", "match", "agree", "ptruth")
KS = (1, 3, 5, 7)
QS = (0.1, 0.25, 0.5, 0.75, 0.9)
# Target entropy bins (nats). 0.05 nats ~ top-1 prob > 0.99.
ENT_EDGES = (0.0, 0.05, 0.2, 0.5, 1.0, 2.0, 1e9)
ENT_LABELS = ("<0.05", "0.05-0.2", "0.2-0.5", "0.5-1", "1-2", ">=2")


def quant(x: np.ndarray) -> dict:
    x = x[np.isfinite(x)]
    if not len(x):
        return {"n": 0}
    q = np.quantile(x, QS)
    return {
        "n": int(len(x)),
        "mean": float(x.mean()),
        **{f"q{int(p * 100)}": float(v) for p, v in zip(QS, q, strict=True)},
    }


def walk(match: np.ndarray, k: int) -> list[tuple[int, int]]:
    """Greedy chain on one sequence: match [anchors, 7] (nan past the end).
    Returns (anchor index, accepted drafts) per step."""
    steps, i, n = [], 0, match.shape[0]
    while i < n:
        acc = 0
        for d in range(k):
            if match[i, d] == 1.0:
                acc += 1
            else:
                break
        steps.append((i, acc))
        i += acc + 1
    return steps


def drafter_names(path: Path) -> tuple[str, ...]:
    with np.load(path) as z:
        names = {k[: -len("_match")] for k in z.files if k.endswith("_match")}
    return DRAFTERS if names == set(DRAFTERS) else tuple(sorted(names))


def sampled_lengths(tv: np.ndarray, k: int) -> np.ndarray:
    """Expected acceptance length per anchor (incl. the bonus token) under T=1
    rejection sampling: 1 + sum_{j<=k} prod_{d<=j} (1 - TV_d); depths past the
    sequence end (nan) end the chain."""
    acc = np.nan_to_num(1.0 - tv[:, :k].astype(np.float64), nan=0.0)
    return 1.0 + np.cumprod(acc, axis=1).sum(axis=1)


def load(paths: list[Path], drafters: tuple[str, ...] = DRAFTERS):
    seqs = []
    for path in paths:
        z = np.load(path)
        gen = z["seq_gen"]
        t_off = np.concatenate([[0], np.cumsum(gen)])
        a_off = np.concatenate([[0], np.cumsum(gen - 1)])
        for j in range(len(gen)):
            seqs.append(
                {
                    "workload": str(z["seq_workload"][j]),
                    "seq": str(z["seq_sha"][j]),
                    "prompt_tokens": int(z["seq_prompt"][j]),
                    "generated_tokens": int(gen[j]),
                    "target": {
                        f: z[f"target_{f}"][t_off[j] : t_off[j + 1]]
                        for f in ("ent", "top1p", "ptruth", "hf_agree")
                    },
                    **{
                        d: {f: z[f"{d}_{f}"][a_off[j] : a_off[j + 1]] for f in FIELDS}
                        for d in drafters
                    },
                }
            )
    return seqs


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="apertus-tree-shape-aggregate")
    parser.add_argument("shards", type=Path, nargs="+")
    parser.add_argument("--suite", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sequences", type=Path, required=True)
    parser.add_argument("--sampled", action="store_true", help="add the T=1 acceptance replay")
    args = parser.parse_args(argv)
    suite = yaml.safe_load(args.suite.read_text())
    classes = {s["label"]: s["class"] for s in suite["slices"]}
    drafters = drafter_names(args.shards[0])
    seqs = load(args.shards, drafters)
    depth = seqs[0][drafters[0]]["match"].shape[1]
    ks = KS + tuple(k for k in (15,) if k <= depth and k not in KS)
    extended = args.sampled or drafters != DRAFTERS or ks != KS
    labels = [s["label"] for s in suite["slices"] if any(q["workload"] == s["label"] for q in seqs)]

    out = {
        "entropy_bins": list(ENT_LABELS),
        "entropy_edges": list(ENT_EDGES[:-1]),
        "quantiles": list(QS),
        "workloads": {},
    }
    if extended:
        out = {"drafters": list(drafters), "depth": int(depth), "ks": list(ks), **out}
    seq_rows = []
    for label in labels:
        ws = [s for s in seqs if s["workload"] == label]
        t_ent = np.concatenate([s["target"]["ent"] for s in ws])
        t_top = np.concatenate([s["target"]["top1p"] for s in ws])
        entry = {
            "class": classes[label],
            "sequences": len(ws),
            "generated_tokens": int(sum(s["generated_tokens"] for s in ws)),
            "prompt_tokens": quant(np.array([s["prompt_tokens"] for s in ws], float)),
            "hf_argmax_agreement": float(
                np.concatenate([s["target"]["hf_agree"] for s in ws]).mean()
            ),
            "target": {"ent": quant(t_ent), "top1p": quant(t_top)},
            "target_entropy_hist": np.histogram(t_ent, ENT_EDGES)[0].tolist(),
        }
        for d in drafters:
            res = {"depth1": {}, "depth1to3": {}, "per_depth": {}, "chain": {}, "bins": {}}
            for f in FIELDS:
                res["depth1"][f] = quant(np.concatenate([s[d][f][:, 0] for s in ws]))
                res["depth1to3"][f] = quant(np.concatenate([s[d][f][:, :3].ravel() for s in ws]))
            for f in ("match", "tv", "agree"):
                stack = np.concatenate([s[d][f] for s in ws])
                res["per_depth"][f] = [float(np.nanmean(stack[:, j])) for j in range(depth)]
            # Depth-1 metrics by the target's entropy at the same position (anchor i
            # predicts generated position i + 1).
            ent1 = np.concatenate([s["target"]["ent"][1:] for s in ws])
            m1 = np.concatenate([s[d]["match"][:, 0] for s in ws])
            tv1 = np.concatenate([s[d]["tv"][:, 0] for s in ws])
            b = np.digitize(ent1, ENT_EDGES[1:-1])
            res["bins"]["depth1"] = [
                {
                    "n": int((b == j).sum()),
                    "match": float(m1[b == j].mean()) if (b == j).any() else None,
                    "tv": float(tv1[b == j].mean()) if (b == j).any() else None,
                }
                for j in range(len(ENT_LABELS))
            ]
            per_seq = {}
            for k in ks:
                lengths, firsts, seq_means = [], [], []
                for s in ws:
                    steps = walk(s[d]["match"], k)
                    acc = np.array([a for _i, a in steps], float) + 1
                    lengths.append(acc)
                    firsts.append(np.array([s["target"]["ent"][i + 1] for i, _a in steps]))
                    seq_means.append(acc.mean())
                    per_seq.setdefault(id(s), {})[k] = float(acc.mean())
                acc = np.concatenate(lengths)
                first = np.concatenate(firsts)
                fb = np.digitize(first, ENT_EDGES[1:-1])
                res["chain"][f"k{k}"] = {
                    "steps": int(len(acc)),
                    "acceptance_length": float(acc.mean()),
                    "tokens_per_step_hist": np.bincount(acc.astype(int), minlength=k + 2)[
                        1:
                    ].tolist(),
                    "per_sequence": quant(np.array(seq_means)),
                    "by_first_entropy": [
                        {
                            "n": int((fb == j).sum()),
                            "acceptance_length": float(acc[fb == j].mean())
                            if (fb == j).any()
                            else None,
                        }
                        for j in range(len(ENT_LABELS))
                    ],
                }
            if args.sampled:
                res["sampled"] = {}
                for k in ks:
                    per = [sampled_lengths(s[d]["tv"], k) for s in ws]
                    res["sampled"][f"k{k}"] = {
                        "anchors": int(sum(len(x) for x in per)),
                        "acceptance_length": float(np.concatenate(per).mean()),
                        "per_sequence": quant(np.array([x.mean() for x in per])),
                    }
                    for s, x in zip(ws, per, strict=True):
                        per_seq.setdefault(id(s), {})[f"s{k}"] = float(x.mean())
            entry[d] = res
            for s in ws:
                s.setdefault("_chain", {})[d] = per_seq[id(s)]
        out["workloads"][label] = entry
        for s in ws:
            row = {
                "workload": label,
                "class": classes[label],
                "seq": s["seq"],
                "prompt_tokens": s["prompt_tokens"],
                "generated_tokens": s["generated_tokens"],
                "target_ent_mean": float(s["target"]["ent"].mean()),
                "target_ent_median": float(np.median(s["target"]["ent"])),
                "target_top1p_mean": float(s["target"]["top1p"].mean()),
            }
            for d in drafters:
                row[f"{d}_ent1_mean"] = float(np.nanmean(s[d]["ent"][:, 0]))
                row[f"{d}_tv1_mean"] = float(np.nanmean(s[d]["tv"][:, 0]))
                row[f"{d}_kl1_mean"] = float(np.nanmean(s[d]["kl"][:, 0]))
                row[f"{d}_match1_mean"] = float(np.nanmean(s[d]["match"][:, 0]))
                for k in ks:
                    row[f"{d}_acc_k{k}"] = s["_chain"][d][k]
                if args.sampled:
                    for k in ks:
                        row[f"{d}_sacc_k{k}"] = s["_chain"][d][f"s{k}"]
            seq_rows.append(row)
    args.output.write_text(json.dumps(out, indent=1) + "\n")
    with args.sequences.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(seq_rows[0]))
        writer.writeheader()
        for row in seq_rows:
            writer.writerow({k: round(v, 5) if isinstance(v, float) else v for k, v in row.items()})
    print(
        json.dumps(
            {
                w: {d: e[d]["chain"]["k3"]["acceptance_length"] for d in drafters}
                for w, e in out["workloads"].items()
            },
            indent=1,
        )
    )


if __name__ == "__main__":
    main()
