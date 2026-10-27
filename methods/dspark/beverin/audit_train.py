#!/usr/bin/env python3
"""Record checkpoint progress and validate resume after every chain link.

Runs on the batch node with Python 3.6; no torch imports or checkpoint loads.
The deployment directory holds immutable copies of the small run artifacts.
"""
import argparse
import json
import os
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


def audit(work, run_id, job_id, phase, return_code=0, completion_skip=False):
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
    completed = None
    if os.environ.get("BEVERIN_SINGLE_EPOCH", "0") == "1":
        from single_epoch import read_completion
        completed = read_completion(run / "checkpoints", int(os.environ["BEVERIN_EXPECTED_EPOCH_STEPS"]))
    validation_only = bool(completed and initial.get("local_step") == 0
                           and start_step == end_step and "BEVERIN_VALIDATION_ONLY" in content)
    authenticated_skip = bool(completion_skip and completed)
    allow_unchanged = validation_only or authenticated_skip
    errors = []
    if return_code:
        errors.append("srun return code {}".format(return_code))
    if completion_skip and not authenticated_skip:
        errors.append("completion skip lacks authenticated final validation")
    if end_step <= start_step and not allow_unchanged:
        errors.append("checkpoint global_step did not advance")
    if not any(a["checkpoint_saved"] for a in report["attempts"]) and not allow_unchanged:
        errors.append("no successful checkpoint save in the trainer log")
    if start_step and not authenticated_skip:
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
    for source, name in ((run / "checkpoints" / "single-epoch-complete.json", "single-epoch-complete.json"),
                         (checkpoint / "val_metrics.json" if checkpoint else deployment / "missing", "val_metrics.json"),
                         (checkpoint / "config.json" if checkpoint else deployment / "missing", "checkpoint-config.json")):
        if source.is_file():
            shutil.copy2(str(source), str(deployment / name))
    report.update({
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "initial_state": initial, "final_state": state,
        "return_code": return_code, "status": "needs_attention" if errors else "ok",
        "validation_only": validation_only, "completion_skip": authenticated_skip,
        "single_epoch_complete": bool(completed),
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
    parser.add_argument("--completion-skip", action="store_true")
    args = parser.parse_args()
    raise SystemExit(audit(args.work, args.run_id, args.job_id, args.phase, args.return_code,
                          args.completion_skip))
