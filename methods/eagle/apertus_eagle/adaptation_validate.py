"""Stop an owned recovery batch on infrastructure failure between independent arms."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from apertus_eagle.adaptation_recovery import curve_report


def inspect(config: Path, run_id: str) -> tuple[int, dict]:
    cfg = yaml.safe_load(config.read_text())
    name = cfg["run_name"] + "-" + run_id
    run = Path(cfg["output_root"]) / name
    report = curve_report(
        run, Path(cfg["head_root"]) / (name + "-se"), cfg["training"]["num_train_steps"]
    )
    if report["state"] == "complete":
        return 0, report  # Includes valid negative scientific outcomes, unchanged.
    marker = run / "training-incomplete.json"
    if (
        report["state"] == "incomplete"
        and marker.is_file()
        and (run / "checkpoints/resume").is_dir()
    ):
        return 4, report | {"resume_marker": str(marker)}
    return 2, report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()
    status, report = inspect(args.config, args.run_id)
    print(json.dumps(report), flush=True)
    raise SystemExit(status)


if __name__ == "__main__":
    main()
