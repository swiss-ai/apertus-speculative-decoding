"""Collect the served acceptance of a workload-shape gate (submit-gate.sh) per slice.

    python3 experiments/workload-shape/collect_gate.py GATE_DIR OUT_DIR \
        [--arms ARM ...] [--captures RUN_DIR/captures/shard-*.json]

GATE_DIR/<arm>/cells/<slice>/<slice>/c1/summary.json holds the vLLM spec-decode
counter deltas of one slice (apertus_bench loadtest, C=1). Each arm's drafter, k and
sampling come from its cells' metadata (speculative method, num_speculative_tokens,
temperature). ``--arms`` fixes the arms and their order (default: every arm directory,
sorted); the per-target .env files list them in GATE_ARMS. Writes to OUT_DIR:

- ``served-acceptance.json``: ``{"cells": [{"arm", "drafter", "k", "sampling",
  "slices": {slice: mean acceptance length}, "per_position", "requests", "failed"}]}``,
  the format analyze.py --served reads for several drafters and arms. With the
  replay's plain-vLLM captures it also checks, per arm, that the served prompt has the
  replay's prompt length (same chat rendering) and how often the served completion
  has the captured length (greedy arms: the same trajectory up to numerics);
- per arm ``served-acceptance-<arm>.json`` (``{"slices": {slice: {"acceptance_length",
  ...counters}}}``, one arm, as analyze.py and the cross_model.py report read it) and
  ``gate-cells.json`` with every arm's status. Slices without a successful cell are
  listed as missing, never filled in.
"""

from __future__ import annotations

import argparse
import json
import shlex
from pathlib import Path


def arm_identity(root: Path) -> tuple[str, int, str]:
    """(drafter, k, sampling) of an arm from the metadata of its first cell."""
    meta = json.loads(sorted(root.glob("*/*/c1/metadata.json"))[0].read_text())
    variant, gen = meta["variant"], meta["cell"]["generation"]
    sampling = "greedy" if not gen.get("temperature") else "t1"
    return variant["method"], int(variant["num_speculative_tokens"]), sampling


def submitter(gate: Path) -> str:
    """Name of the script that submitted the gate (first word of submit-command.txt)."""
    path = gate / "submit-command.txt"
    if path.is_file() and path.read_text().strip():
        return Path(shlex.split(path.read_text().splitlines()[0])[0]).name
    return "submit-gate.sh"


def served_cells(gate: Path, arms: list[str], captures: list[Path]) -> dict:
    captured = {}
    for path in captures:
        for c in json.loads(path.read_text())["cases"]:
            if "@cap" not in c["id"]:
                captured[c["id"]] = (c["prompt_tokens"], len(c["token_ids"]))
    cells = []
    for arm in arms:
        root = gate / arm / "cells"
        if not root.is_dir():
            continue
        drafter, k, mode = arm_identity(root)
        cell = {"arm": arm, "drafter": drafter, "k": k, "sampling": mode, "slices": {},
                "per_position": {}, "requests": {}, "failed": {}}  # fmt: skip
        for summary in sorted(root.glob("*/*/c1/summary.json")):
            label = summary.parent.parent.parent.name
            s = json.loads(summary.read_text())
            spec = s["speculative_decoding"]
            cell["slices"][label] = spec.get("mean_acceptance_length")
            pos = spec.get("acceptance_rate_per_position") or {}
            cell["per_position"][label] = [pos[str(i)] for i in range(len(pos))]
            req = s.get("requests") or {}
            cell["requests"][label] = req.get("successful")
            if req.get("failed"):
                cell["failed"][label] = req["failed"]
        if captured:
            n = same_prompt = same_len = 0
            for req in sorted(root.glob("*/*/c1/requests.jsonl")):
                for line in req.read_text().splitlines():
                    r = json.loads(line)
                    if not r.get("success") or r["prompt_id"] not in captured:
                        continue
                    p_len, g_len = captured[r["prompt_id"]]
                    n += 1
                    same_prompt += r["prompt_tokens"] == p_len
                    same_len += r["completion_tokens"] == g_len
            cell["vs_capture"] = {
                "requests": n,
                "same_prompt_tokens": same_prompt / n if n else None,
                "same_completion_tokens": same_len / n if n else None,
            }
        cells.append(cell)
    return {"source": str(gate), "cells": cells}


def per_arm(gate: Path, arms: list[str], out: Path) -> dict:
    by = submitter(gate)
    cells = {}
    for arm in arms:
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
                    "source": f"{gate}/{arm} ({by}), C=1, 512-token cap, "
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
    return cells


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("gate", type=Path)
    p.add_argument("out", type=Path)
    p.add_argument("--arms", nargs="+", help="arms in order (default: all, sorted)")
    p.add_argument("--captures", type=Path, nargs="*", default=[])
    args = p.parse_args()
    gate, out = args.gate, args.out
    arms = args.arms or sorted(d.name for d in gate.iterdir() if (d / "cells").is_dir())
    out.mkdir(parents=True, exist_ok=True)
    served = served_cells(gate, arms, args.captures)
    (out / "served-acceptance.json").write_text(json.dumps(served, indent=1) + "\n")
    status = per_arm(gate, arms, out)
    for c in served["cells"]:
        print(c["arm"], c["drafter"], c["k"], c["sampling"], len(c["slices"]), "slices",
              c["failed"] or "no failures", status[c["arm"]]["missing"] or "",
              c.get("vs_capture", ""))  # fmt: skip


if __name__ == "__main__":
    main()
