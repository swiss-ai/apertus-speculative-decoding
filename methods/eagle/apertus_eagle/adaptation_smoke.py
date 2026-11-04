"""Assess a short adaptation smoke without changing its scientific gates.

A four-step infrastructure check can update weights and export correctly while
its tiny stochastic training-loss window rises. Keep both outcomes explicit.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from pathlib import Path
from typing import Any


def assess(run_dir: Path, head_dir: Path, expected_steps: int) -> dict[str, Any]:
    sources = {
        "summary": run_dir / "train-summary.json",
        "history": run_dir / "history.json",
        "provenance": run_dir / "provenance.json",
        "evals": run_dir / "evals.json",
        "export": head_dir / "export-report.json",
        "config": head_dir / "config.json",
    }
    objects = {key: json.loads(path.read_text()) for key, path in sources.items()}
    summary, export, config = (objects[key] for key in ("summary", "export", "config"))
    scientific = summary["gates"]
    checks = {
        "expected_steps_completed": summary["steps"] == expected_steps,
        "finite_training": scientific.get("finite_losses") is True
        and summary.get("non_finite_grad_steps") == 0,
        "real_weight_updates": scientific.get("real_updates") is True,
        "finite_loss_summary": all(
            math.isfinite(summary[name])
            for name in ("train_loss_first_decile", "train_loss_last_decile")
        ),
        "export_reload_passed": export.get("reload", {}).get("pass") is True,
        "exported_tensors_equal": export.get("reload", {}).get("exported_tensors_bitwise_equal")
        is True,
        "parent_checkpoint_recorded": bool(config.get("apertus_training", {}).get("init_from")),
    }
    return {
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "run_dir": str(run_dir),
        "head_dir": str(head_dir),
        "steps": summary["steps"],
        "infrastructure_checks": checks,
        "infrastructure_passed": all(checks.values()),
        "scientific_gates": scientific,
        "scientific_gates_passed": all(scientific.values()),
        "claim_limit": "Infrastructure smoke only; no serving or adaptation-benefit claim.",
        "sources": {
            key: {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
            for key, path in sources.items()
        },
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--head-dir", type=Path, required=True)
    parser.add_argument("--expected-steps", type=int, default=4)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    report = assess(args.run_dir, args.head_dir, args.expected_steps)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as handle:
        handle.write(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    raise SystemExit(0 if report["infrastructure_passed"] else 1)


if __name__ == "__main__":
    main()
