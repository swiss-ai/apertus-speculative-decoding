"""Collect the served acceptance of the Qwen3-8B shape gate into analyze.py's format.

    python3 experiments/workload-shape/gate_summary.py GATE_DIR OUT.json [CAPTURE.json ...]

GATE_DIR is an output of submit-gate-qwen3.sh: one directory per arm with
``cells/<slice>/<slice>/c1/summary.json`` (apertus_bench loadtest). Writes
``{"cells": [{"drafter", "k", "sampling", "arm", "slices": {slice: mean acceptance
length}, "per_position": {slice: [...]}, "requests": {slice: n}, "failed": {...}}]}``.
With the replay's plain-vLLM captures it also checks, per arm, that the served prompt
has the replay's prompt length (same chat rendering) and how often the served
completion has the captured length (greedy arms: the same trajectory up to numerics).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ARMS = {
    "eagle3-k3": ("eagle3", 3, "greedy"),
    "dflash-k3": ("dflash", 3, "greedy"),
    "dflash-k15": ("dflash", 15, "greedy"),
    "eagle3-k3-t1": ("eagle3", 3, "t1"),
    "dflash-k3-t1": ("dflash", 3, "t1"),
}


def main() -> None:
    gate, out = Path(sys.argv[1]), Path(sys.argv[2])
    captured = {}
    for path in sys.argv[3:]:
        for c in json.loads(Path(path).read_text())["cases"]:
            if "@cap" not in c["id"]:
                captured[c["id"]] = (c["prompt_tokens"], len(c["token_ids"]))
    cells = []
    for arm, (drafter, k, mode) in ARMS.items():
        root = gate / arm / "cells"
        if not root.is_dir():
            continue
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
    out.write_text(json.dumps({"source": str(gate), "cells": cells}, indent=1) + "\n")
    for c in cells:
        print(c["arm"], len(c["slices"]), "slices", c["failed"] or "no failures",
              c.get("vs_capture", ""))


if __name__ == "__main__":
    main()
