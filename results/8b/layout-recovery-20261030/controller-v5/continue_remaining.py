#!/usr/bin/env python3
"""Continue only original independent layouts; preserve strict rejected cells."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
from pathlib import Path

from apertus_bench import layout_recovery as recovery

REMAINING = ["tp2dp2", "pd-off"]


def read(path):
    return json.loads(path.read_text())


def request_rejections_only(record: dict) -> bool:
    """Reject reliability cells; never reinterpret them as gate passes."""
    failures = [r for r in record.get("gate", {}).get("checks", []) if not r.get("passed")]
    return bool(
        record.get("slurm", {}).get("state") == "COMPLETED"
        and failures
        and all(
            r.get("kind") == "cell"
            and r.get("requests", {}).get("attempted", 0) > 0
            and r["requests"].get("success_rate", 0) < 0.99
            for r in failures
        )
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--prior-state", type=Path, required=True)
    ap.add_argument("--expected-source", type=Path, required=True)
    ap.add_argument("--site", type=Path, required=True)
    a = ap.parse_args()
    previous = read(a.prior_state)
    if previous["campaign_state"] != "needs_attention":
        raise RuntimeError("Prior controller must be stopped at its explicit gate")
    stages = previous["stages"]
    if [r["stage"] for r in stages] != [
        "preflight",
        "tp4-dspark-off-k7",
        "tp4-eagle-k7",
        "tp4-dspark-on-k7",
        "think-pd",
    ]:
        raise RuntimeError("Unexpected prior stage set; inspect before continuation")
    if not all(r["state"] == "measured" and r["gate"]["passed"] for r in stages[:-1]):
        raise RuntimeError("Prior successful stages are not valid")
    if not request_rejections_only(stages[-1]):
        raise RuntimeError("Prior failure includes infrastructure/provenance/metric issues")
    expected = read(a.expected_source)
    if recovery.source_identity() != expected:
        raise RuntimeError("Frozen source changed; no new jobs submitted")
    original_stages = list(recovery.STAGES)
    recovery.STAGES = [r for r in original_stages if r[0] in REMAINING]
    if [r[0] for r in recovery.STAGES] != REMAINING:
        raise RuntimeError("Original remaining stage definitions changed")
    a.root.mkdir(parents=True, exist_ok=True)
    decision = a.root / "independent-continuation.json"
    if not decision.exists():
        record = {
            "created_at": dt.datetime.now(dt.UTC).isoformat(),
            "prior_state": str(a.prior_state),
            "prior_state_sha256": hashlib.sha256(a.prior_state.read_bytes()).hexdigest(),
            "source_identity": expected,
            "allowed_stages": REMAINING,
            "retained_rejections": [r for r in stages[-1]["gate"]["checks"] if not r["passed"]],
            "reason": (
                "Only request-reliability cells failed. Independent original layouts "
                "need no P/D passing gate; no repeat for a pass."
            ),
            "threshold": 0.99,
            "gate_policy": (
                "Original strict gates unchanged. Infrastructure/provenance/counter "
                "failures stop; request-only failures remain rejected."
            ),
            "wrapper_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        }
        with decision.open("x") as stream:
            json.dump(record, stream, indent=2)
            stream.write("\n")
    path = a.root / "state.json"
    if path.exists():
        existing = read(path)
        prior_tp2 = next((r for r in existing["stages"] if r["stage"] == "tp2dp2"), None)
        if (
            prior_tp2
            and request_rejections_only(prior_tp2)
            and any(r["stage"] == "pd-off" for r in existing["stages"])
        ):
            recovery.STAGES = [r for r in original_stages if r[0] == "pd-off"]
    recovery.controller(a.root, a.site, "normal", 60)
    path = a.root / "state.json"
    state = read(path)
    # If TP2 finishes but has request-only rejected cells, the independent
    # original P/D-off stage can still run. Keep its false gate and raw data.
    latest = state["stages"][-1]
    if latest["stage"] == "tp2dp2" and request_rejections_only(latest):
        recovery.write_json(
            a.root / "tp2dp2-reliability-continuation.json",
            {
                "rejected_gate": latest["gate"],
                "reason": "Preserve rejection; continue independent original PD-off once.",
            },
        )
        state["campaign_state"] = "active"
        state.pop("error", None)
        state["has_rejected_cells"] = True
        recovery.write_state(path, state)
        recovery.STAGES = [r for r in original_stages if r[0] == "pd-off"]
        recovery.controller(a.root, a.site, "normal", 60)
        state = read(path)
        latest = state["stages"][-1]
    all_finished = [r["stage"] for r in state["stages"]] == REMAINING
    usable_outcomes = all(
        r["state"] == "measured" or request_rejections_only(r) for r in state["stages"]
    )
    if all_finished and usable_outcomes:
        state["campaign_state"] = (
            "measurements_finished_with_rejected_cells_pending_scientific_review"
        )
        state["prior_think_pd_rejected_cells_preserved"] = True
        state["implementation_recommendation"] = "Unestablished; strict scientific review required"
        recovery.write_state(path, state)


if __name__ == "__main__":
    main()
