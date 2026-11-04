#!/usr/bin/env python3
"""Package the two remaining terminal grids for the frozen CPU review helper."""

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path

HELPER = Path(__file__).parents[1] / "layout-recovery-20261027/review-helper-v1/review.py"
HELPER_SHA = "25d8f833b6377153780587efd5b6d58fa278ebe0e9cd51ad6c9c765d938e8989"
EXPECTED = {"tp2dp2": 120, "pd-off": 147}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--controller-state", type=Path, required=True)
    ap.add_argument("--path-map", action="append", default=[])
    ap.add_argument("--output", type=Path, required=True)
    a = ap.parse_args()
    if digest(HELPER) != HELPER_SHA:
        raise RuntimeError("Frozen helper changed")
    state = json.loads(a.controller_state.read_text())
    if [r["stage"] for r in state["stages"]] != list(EXPECTED):
        raise RuntimeError("Both original remaining stages must exist; no final package produced")
    maps = [tuple(value.split("=", 1)) for value in a.path_map]
    spec = importlib.util.spec_from_file_location("frozen_layout_review", HELPER)
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    rows = []
    sources = []
    outcomes = []
    for record in state["stages"]:
        if record.get("slurm", {}).get("state") != "COMPLETED":
            raise RuntimeError("Stage is not terminal COMPLETED; no final package produced")
        gate = record.get("gate", {})
        failures = [check for check in gate.get("checks", []) if not check.get("passed")]
        if not gate.get("passed") and (
            not failures
            or any(
                check.get("kind") != "cell"
                or check.get("requests", {}).get("success_rate", 1) >= 0.99
                for check in failures
            )
        ):
            raise RuntimeError("Unresolved infrastructure/provenance/counter failure")
        path = a.controller_state.parent / (
            "analysis-{}-{}.json".format(record["stage"], record["job_id"])
        )
        if path.exists():
            new = json.loads(path.read_text())["rows"]
            sources.append({"path": str(path), "sha256": digest(path)})
        else:
            new = helper.partial_rows(record, maps)
        if len(new) != EXPECTED[record["stage"]] or len({r["summary"] for r in new}) != len(new):
            raise RuntimeError("Original grid is incomplete or duplicated")
        rows.extend(new)
        outcomes.append(
            {key: record.get(key) for key in ["stage", "job_id", "state", "slurm", "gate"]}
        )
    a.output.mkdir(parents=True, exist_ok=False)
    report = {
        "rows": rows,
        "stage_outcomes": outcomes,
        "source_analyses": sources,
        "controller_state": str(a.controller_state),
        "controller_state_sha256": digest(a.controller_state),
        "helper_sha256": HELPER_SHA,
        "packager_sha256": digest(Path(__file__)),
        "claim_limit": (
            "CPU packaging only; all false gates and rejected cells retained. "
            "No scientific gate is promoted."
        ),
    }
    (a.output / "remaining-analysis.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"rows": len(rows), "output": str(a.output / "remaining-analysis.json")}))


if __name__ == "__main__":
    main()
