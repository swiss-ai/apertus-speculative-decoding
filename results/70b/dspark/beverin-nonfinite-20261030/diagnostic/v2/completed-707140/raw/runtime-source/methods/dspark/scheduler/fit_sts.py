"""Sequential Temperature Scaling of the DSpark confidence head (arXiv:2607.05147,
section 3.2.1), from the calibration data the patched vLLM writes
(VLLM_DSPARK_CONF_DUMP, serving/patches/vllm-apertus-dspark-paper-scheduler.patch).

Each line is one verification of a request's drafts: the raw confidence logits
``z`` of its slots, how many were verified ``n`` and how many accepted ``a``.
Slot ``k`` (0-based) survived when ``a > k``; the head models
``c_k = sigmoid(z_k)``, the probability that slot ``k`` is accepted given the
slots before it were, so the prefix survival is the running product.

STS calibrates that product left to right: for each slot in turn, a 1D grid
search picks the temperature ``T_k`` (``c_k = sigmoid(z_k / T_k)``) that
minimises the expected calibration error of the running product against the
observed prefix survival, the earlier slots' temperatures fixed. ECE as in
DeepSpec's confidence evaluation: equal-width bins (20), weighted by count.
A request's last line is dropped (a stop can cut the accepted run short).

    python3 fit_sts.py OUT.json DUMP.jsonl [DUMP.jsonl ...]

The output holds the temperatures (VLLM_DSPARK_STS) and the per-slot ECE and
mean prediction before and after, against the observed rate.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

NUM_BINS = 20
GRID = np.unique(np.concatenate([np.geomspace(0.05, 20.0, 600), [1.0]]))


def load(paths: list[Path]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(logits [N, k], verified [N], accepted [N]) without each request's last line."""
    records: list[dict] = []
    last: dict[str, int] = {}
    for path in paths:
        with path.open() as handle:
            for line in handle:
                row = json.loads(line)
                # Requests are keyed per file: ids may repeat across servers.
                key = f"{path}:{row['req']}"
                last[key] = len(records)
                records.append(row)
    keep = sorted(set(range(len(records))) - set(last.values()))
    width = max(len(records[i]["z"]) for i in keep)
    logits = np.full((len(keep), width), np.nan)
    for out, index in enumerate(keep):
        z = records[index]["z"]
        logits[out, : len(z)] = z
    verified = np.array([records[i]["n"] for i in keep])
    accepted = np.array([records[i]["a"] for i in keep])
    return logits, verified, accepted


def ece(probs: np.ndarray, labels: np.ndarray, num_bins: int = NUM_BINS) -> float:
    """Expected calibration error over equal-width bins, weighted by count."""
    if probs.size == 0:
        return float("nan")
    bins = np.clip((probs * num_bins).astype(int), 0, num_bins - 1)
    count = np.bincount(bins, minlength=num_bins)
    pred = np.bincount(bins, weights=probs, minlength=num_bins)
    target = np.bincount(bins, weights=labels, minlength=num_bins)
    used = count > 0
    gap = np.abs(pred[used] - target[used])
    return float(gap.sum() / probs.size)


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def fit(logits: np.ndarray, verified: np.ndarray, accepted: np.ndarray) -> dict:
    """STS temperatures and per-slot calibration before and after."""
    num_slots = logits.shape[1]
    raw_prefix = np.ones(len(logits))
    sts_prefix = np.ones(len(logits))
    temperatures: list[float] = []
    slots: list[dict] = []
    for slot in range(num_slots):
        # Rows that verified this slot; its label is prefix survival through it.
        rows = (verified > slot) & ~np.isnan(logits[:, slot])
        labels = (accepted[rows] > slot).astype(np.float64)
        z = logits[rows, slot]
        raw_prefix[rows] *= sigmoid(z)
        best = 1.0  # No sample of this slot: keep the head as trained.
        if labels.size:
            scores = [ece(sts_prefix[rows] * sigmoid(z / t), labels) for t in GRID]
            best = float(GRID[int(np.argmin(scores))])
        temperatures.append(best)
        sts_prefix[rows] *= sigmoid(z / best)
        slots.append(
            {
                "slot": slot + 1,
                "samples": int(rows.sum()),
                "observed": float(labels.mean()) if labels.size else float("nan"),
                "pred_raw": float(raw_prefix[rows].mean()) if labels.size else float("nan"),
                "pred_sts": float(sts_prefix[rows].mean()) if labels.size else float("nan"),
                "ece_raw": ece(raw_prefix[rows], labels),
                "ece_sts": ece(sts_prefix[rows], labels),
                "temperature": best,
            }
        )
    return {"temperatures": temperatures, "num_bins": NUM_BINS, "slots": slots}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("output", type=Path)
    parser.add_argument("dumps", type=Path, nargs="+")
    args = parser.parse_args()
    logits, verified, accepted = load(args.dumps)
    result = fit(logits, verified, accepted)
    result["records"] = int(len(logits))
    result["sources"] = [str(path) for path in args.dumps]
    args.output.write_text(json.dumps(result, indent=1) + "\n")
    print(f"{len(logits)} verifications -> {args.output}")
    print("slot  samples  observed  pred raw  pred STS  ECE raw  ECE STS      T")
    for slot in result["slots"]:
        print(
            f"{slot['slot']:>4} {slot['samples']:>8} {slot['observed']:>9.3f} "
            f"{slot['pred_raw']:>9.3f} {slot['pred_sts']:>9.3f} "
            f"{slot['ece_raw']:>8.4f} {slot['ece_sts']:>8.4f} {slot['temperature']:>6.2f}"
        )


if __name__ == "__main__":
    main()
