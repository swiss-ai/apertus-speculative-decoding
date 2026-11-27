"""Collect the served acceptance of the Qwen3-4B gate (submit-gate-qwen3.sh) per slice.

    python3 collect_gate.py GATE_DIR OUT_DIR

GATE_DIR/<arm>/cells/<slice>/<slice>/c1/summary.json holds the vLLM spec-decode
counter deltas of one slice (32 requests, C=1). Writes, per arm,
``OUT_DIR/served-acceptance-<arm>.json`` in the format analyze.py reads
(``{"slices": {label: {"acceptance_length": ...}}}``) plus the raw counters, and
``OUT_DIR/gate-cells.json`` with every cell's status. Arms or slices without a
successful cell are listed as missing, never filled in.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ARMS = ("k3-greedy", "k7-greedy", "k3-t1", "k7-t1")


def main() -> None:
    gate, out = Path(sys.argv[1]), Path(sys.argv[2])
    out.mkdir(parents=True, exist_ok=True)
    cells = {}
    for arm in ARMS:
        root = gate / arm / "cells"
        if not root.is_dir():
            cells[arm] = {"missing": True}
            continue
        slices, missing = {}, []
        for cell in sorted(root.iterdir()):
            summary = cell / cell.name / "c1" / "summary.json"
            if not summary.is_file():
                missing.append(cell.name)
                continue
            s = json.loads(summary.read_text())
            spec = s.get("speculative_decoding") or {}
            if not spec.get("enabled") or not spec.get("drafts"):
                missing.append(cell.name)
                continue
            slices[cell.name] = {
                "acceptance_length": spec["mean_acceptance_length"],
                "acceptance_rate": spec["acceptance_rate"],
                "drafts": spec["drafts"],
                "draft_tokens": spec["draft_tokens"],
                "accepted_tokens": spec["accepted_tokens"],
                "acceptance_rate_per_position": spec.get("acceptance_rate_per_position"),
                "requests": s["requests"],
                "completion_tokens": s["tokens"]["completion"],
                "output_tokens_per_second": s["tokens"]["output_tokens_per_second"],
            }
        meta = {}
        status = gate / arm / "status.txt"
        cells[arm] = {
            "slices": len(slices),
            "missing": missing,
            "status": status.read_text().splitlines() if status.is_file() else None,
        }
        if slices:
            first = next(iter(slices))
            meta = json.loads((root / first / first / "c1" / "metadata.json").read_text())
        (out / f"served-acceptance-{arm}.json").write_text(
            json.dumps(
                {
                    "source": f"{gate}/{arm} (submit-gate-qwen3.sh), C=1, 512-token cap, "
                    "vLLM spec-decode counter deltas per slice",
                    "arm": arm,
                    "metadata": {k: v for k, v in meta.items() if k not in ("argv",)},
                    "slices": slices,
                },
                indent=1,
            )
            + "\n"
        )
    (out / "gate-cells.json").write_text(json.dumps(cells, indent=1) + "\n")
    print(json.dumps(cells, indent=1))


if __name__ == "__main__":
    main()
