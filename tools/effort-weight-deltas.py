#!/usr/bin/env python3
"""Weight deltas of effort-specific drafter heads against a reference head (CPU only).

For every tensor shared by the reference and the heads, report the relative
Frobenius norm ||W_head - W_ref|| / ||W_ref|| per head, and the cosine similarity
between the deltas of each pair of heads (do low/medium/high move the weights in
the same direction?). Also reports the same quantities aggregated over all
compared tensors (flattened). Frozen tensors that never change are listed but
excluded from the aggregate.

Usage:
  python3 tools/effort-weight-deltas.py --reference RELEASED_DIR \
      --head low=DIR --head medium=DIR --head high=DIR [--head mixed=DIR] \
      --output deltas.json
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from datetime import UTC, datetime
from pathlib import Path

import torch
from safetensors.torch import load_file


def load(directory: Path) -> dict[str, torch.Tensor]:
    files = sorted(directory.glob("*.safetensors"))
    if not files:
        raise SystemExit(f"no safetensors in {directory}")
    state: dict[str, torch.Tensor] = {}
    for path in files:
        state.update(load_file(str(path)))
    return state


def digest(directory: Path) -> dict[str, str]:
    return {
        p.name: hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(directory.glob("*.safetensors"))
    }


def compare(reference: dict, heads: dict[str, dict]) -> dict:
    names = sorted(set(reference).intersection(*[set(h) for h in heads.values()]))
    per_tensor, frozen = {}, []
    totals = {role: 0.0 for role in heads}
    dots = {pair: 0.0 for pair in itertools.combinations(heads, 2)}
    reference_total = 0.0
    for name in names:
        base = reference[name].double()
        deltas = {}
        for role, state in heads.items():
            if state[name].shape != base.shape:
                raise SystemExit(f"shape mismatch for {name} in {role}")
            deltas[role] = (state[name].double() - base).flatten()
        if all(float(d.abs().max()) == 0.0 for d in deltas.values()):
            frozen.append(name)
            continue
        base_norm = float(base.norm())
        reference_total += base_norm**2
        row = {"numel": base.numel(), "relative_frobenius": {}, "delta_cosine": {}}
        for role, delta in deltas.items():
            squared = float(delta.dot(delta))
            totals[role] += squared
            row["relative_frobenius"][role] = squared**0.5 / base_norm if base_norm else None
        for a, b in dots:
            dot = float(deltas[a].dot(deltas[b]))
            dots[(a, b)] += dot
            norms = float(deltas[a].norm()) * float(deltas[b].norm())
            row["delta_cosine"][f"{a}-{b}"] = dot / norms if norms else None
        per_tensor[name] = row
    aggregate = {
        "relative_frobenius": {
            role: (value / reference_total) ** 0.5 if reference_total else None
            for role, value in totals.items()
        },
        "delta_cosine": {
            f"{a}-{b}": dots[(a, b)] / (totals[a] * totals[b]) ** 0.5
            if totals[a] and totals[b]
            else None
            for a, b in dots
        },
    }
    return {
        "compared_tensors": len(per_tensor),
        "unchanged_tensors": frozen,
        "only_in_reference": sorted(set(reference) - set(names)),
        "aggregate": aggregate,
        "per_tensor": per_tensor,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--head", action="append", required=True, help="ROLE=DIR")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    paths = dict(item.split("=", 1) for item in args.head)
    heads = {role: load(Path(path)) for role, path in paths.items()}
    result = compare(load(args.reference), heads)
    result["inputs"] = {
        "reference": {"path": str(args.reference), "sha256": digest(args.reference)},
        "heads": {r: {"path": p, "sha256": digest(Path(p))} for r, p in paths.items()},
        "created_at": datetime.now(UTC).isoformat(),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as sink:
        json.dump(result, sink, indent=2)
        sink.write("\n")
    print(json.dumps(result["aggregate"], indent=2))


if __name__ == "__main__":
    main()
