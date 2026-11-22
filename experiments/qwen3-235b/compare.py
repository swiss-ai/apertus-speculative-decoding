"""Compare the Qwen3-235B speculative arms against plain, cell by cell.

    python experiments/qwen3-235b/compare.py PLAIN_RUN SPEC_RUN [SPEC_RUN ...]

Each RUN is a results/qwen3-235b/<run> directory written by serving/qwen3/submit-arm.sh.
Prints one Markdown table per workload and the 10%-on-both-workloads verdict
from docs/qwen3-235b-specdec.md. Reads only the cells' summary.json files.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

THRESHOLD = 0.10
WORKLOADS = ("probe", "arena-hard-clean")


def load(run: Path) -> dict[tuple[str, int], dict]:
    cells = {}
    for path in sorted(run.glob("cells/*/*/c*/summary.json")):
        workload = path.parents[1].name
        concurrency = int(path.parent.name[1:])
        cells[(workload, concurrency)] = json.loads(path.read_text())
    return cells


def arm_name(run: Path) -> str:
    return run.name.split("-", 2)[-1]


def main(argv: list[str]) -> int:
    runs = [Path(arg) for arg in argv]
    plain, specs = load(runs[0]), [(arm_name(run), load(run)) for run in runs[1:]]
    gains: dict[tuple[str, int], dict[str, float]] = {}
    for workload in WORKLOADS:
        levels = sorted(c for (w, c) in plain if w == workload)
        print(f"\n### {workload}\n")
        header = "| C | plain tok/s | plain TPOT p50 ms |"
        rule = "| --- | --- | --- |"
        for name, _ in specs:
            header += f" {name} tok/s (vs plain) | {name} AL | {name} TPOT p50 ms |"
            rule += " --- | --- | --- |"
        print(header)
        print(rule)
        for level in levels:
            base = plain[(workload, level)]
            base_tps = base["tokens"]["output_tokens_per_second"]
            row = f"| {level} | {base_tps:.0f} | {base['latency_ms']['tpot']['p50']:.1f} |"
            for name, cells in specs:
                cell = cells.get((workload, level))
                if cell is None:
                    row += " missing | | |"
                    continue
                tps = cell["tokens"]["output_tokens_per_second"]
                gain = tps / base_tps - 1
                failed = cell["requests"]["failed"] + base["requests"]["failed"]
                value = gain if failed == 0 else float("nan")
                gains.setdefault((name, level), {})[workload] = value
                spec = cell.get("speculative_decoding") or {}
                al = spec.get("mean_acceptance_length")
                failed_note = f" FAILED={failed}" if failed else ""
                al_text = f"{al:.2f}" if al else "n/a"
                tpot = cell["latency_ms"]["tpot"]["p50"]
                row += f" {tps:.0f} ({gain:+.1%}){failed_note} | {al_text} | {tpot:.1f} |"
            print(row)
    print("\n### Verdict (>= +10% output throughput on both workloads, no failures)\n")
    for (name, level), by_workload in sorted(gains.items()):
        values = [by_workload.get(w) for w in WORKLOADS]
        ok = all(v is not None and v == v and v >= THRESHOLD for v in values)
        shown = ", ".join(
            f"{w} {v:+.1%}" if v is not None and v == v else f"{w} n/a"
            for w, v in zip(WORKLOADS, values, strict=True)
        )
        print(f"- {name} C={level}: {'PASS' if ok else 'fail'} ({shown})")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
