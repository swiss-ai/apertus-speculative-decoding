#!/usr/bin/env python3
"""Record checkpoint progress and validate resume after every chain link.

Runs on the batch node with Python 3.6; no torch imports or checkpoint loads.
The deployment directory holds immutable copies of the small run artifacts.
"""
import argparse
import json
import re
import shutil
import time
from pathlib import Path

from summarize_train import summarize


def latest_state(run):
    states = [p for p in (run / "checkpoints").glob("*/training_state.json")
              if p.parent.name.isdigit()]
    if not states:
        return None, {"global_step": 0}
    path = max(states, key=lambda p: int(p.parent.name))
    return path.parent, json.loads(path.read_text())


def audit(work, run_id, job_id, phase, return_code=0):
    run = work / "runs" / run_id
    deployment = run / "deployments" / job_id
    checkpoint, state = latest_state(run)
    if phase == "start":
        (deployment / "initial-training-state.json").write_text(json.dumps(state))
        return 0
    initial = json.loads((deployment / "initial-training-state.json").read_text())
    report = summarize(work, run_id, job_id)
    logs = [Path(a["log"]).read_text(errors="replace") for a in report["attempts"]]
    content = "\n".join(logs)
    start_step = initial.get("global_step", 0)
    end_step = state.get("global_step", 0)
    errors = []
    if return_code:
        errors.append("srun return code {}".format(return_code))
    if end_step <= start_step:
        errors.append("checkpoint global_step did not advance")
    if not any(a["checkpoint_saved"] for a in report["attempts"]):
        errors.append("no successful checkpoint save in the trainer log")
    if start_step:
        if initial.get("local_step", 0):
            resumed = any(int(n) == start_step for n in re.findall(
                r"Resuming mid-epoch from.*?global_step=(\d+)", content, re.DOTALL
            ))
        else:
            resumed = "Resuming training on epoch" in content
        if not resumed:
            errors.append("expected checkpoint resume was not logged")
    if checkpoint:
        for name in ("model.safetensors", "optimizer_state_dict.pt",
                     "scheduler_state_dict.pt", "training_state.json"):
            if not (checkpoint / name).is_file() or not (checkpoint / name).stat().st_size:
                errors.append("missing or empty checkpoint artifact: " + name)
        shutil.copy2(str(checkpoint / "training_state.json"),
                     str(deployment / "final-training-state.json"))
    for name in ("train_command.txt", "speculators.patch", "run.yaml"):
        source = run / "checkpoints" / name
        if source.is_file():
            shutil.copy2(str(source), str(deployment / name))
    report.update({
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "initial_state": initial, "final_state": state,
        "return_code": return_code, "status": "needs_attention" if errors else "ok",
        "errors": errors,
    })
    (deployment / "job-audit.json").write_text(json.dumps(report, indent=2))
    print("BEVERIN_AUDIT " + json.dumps({
        "job_id": job_id, "initial_step": start_step, "final_step": end_step,
        "status": report["status"], "errors": errors,
    }))
    return int(bool(errors))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("start", "finish"))
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--job-id", required=True)
    parser.add_argument("--return-code", type=int, default=0)
    args = parser.parse_args()
    raise SystemExit(audit(args.work, args.run_id, args.job_id, args.phase, args.return_code))
