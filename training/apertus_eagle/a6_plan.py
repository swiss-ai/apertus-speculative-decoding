"""A6: write the confirmation plan from the A5 selection.

Three independent deployments per arm (plain baseline + at most two selected
depths), one of each per block, arm order rotated across blocks (Latin-square
rotation with a fixed seed) so no arm is always first after a fresh node.
Confirmation cells use the untouched test prompts (``PHASE=confirm``: 128
requests per stratum, C=1 and C=8, natural EOS).
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="apertus-eagle-a6-plan")
    parser.add_argument("summary", type=Path, help="a5_summary output JSON")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--blocks", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20260925)
    args = parser.parse_args(argv)
    report = json.loads(args.summary.read_text())
    selected = [int(d) for d in report["selected"]][:2]
    if not selected:
        raise SystemExit("A5 selected no depth")
    arms = [("baseline", None)] + [("eagle", depth) for depth in selected]
    rng = random.Random(args.seed)
    base_order = arms[:]
    rng.shuffle(base_order)
    lines = [
        f"# A6 confirmation: arms {arms}, {args.blocks} blocks, rotated order (seed {args.seed}).",
        f"# Selected from {args.summary} by geometric-mean TPOT speedup over validation strata.",
    ]
    for block in range(args.blocks):
        shift = block % len(base_order)
        order = base_order[shift:] + base_order[:shift]
        for arm, depth in order:
            lines.append(f"{arm} confirm c{block + 1}" + (f" {depth}" if depth else ""))
    args.output.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
