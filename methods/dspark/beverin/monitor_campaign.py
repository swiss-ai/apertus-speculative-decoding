#!/usr/bin/env python3
"""Read-only monitor for the ALPS one-epoch campaign.

Prints alerts and writes them next to the monitor state. It does not send
Slack or email, and it does not cancel or resubmit jobs. A non-finite loss is
matched explicitly; a checkpoint counter alone is not treated as health.
"""
import argparse
import json
import os
from pathlib import Path
import re
import statistics
import subprocess
import time

EXPECTED_STEPS = 4077
SAVE_EVERY = 64
LOSS_RATIO = 1.3
GRAD_RATIO = 5.0
STALL_S = 15 * 60
HEARTBEAT_S = 3 * 3600
LOW_MEM_GIB = 32
SSH = [
    "/usr/bin/ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=20",
    "-o", "ConnectionAttempts=1", "-i", os.path.expanduser("~/.ssh/cscs-key"),
    "-l", "faruk_zahiragic", "-J", "ela", "beverin.cscs.ch", "python3", "-",
]


def _finite(value):
    try:
        number = float(value)
    except (TypeError, ValueError):
        return False
    return number == number and number not in (float("inf"), float("-inf"))


def parse_log(text):
    losses, grads, steps = [], [], []
    nonfinite = False
    for match in re.finditer(r"train/loss=([^\s,]+)", text or ""):
        raw = match.group(1)
        if not _finite(raw):
            nonfinite = True
        else:
            losses.append(float(raw))
    for match in re.finditer(r"train/eal=([^\s,]+)", text or ""):
        if not _finite(match.group(1)):
            nonfinite = True
    for match in re.finditer(r"BEVERIN_GRAD_NORM\s+(\{.*\})", text or ""):
        try:
            payload = json.loads(match.group(1))
        except ValueError:
            nonfinite = True
            continue
        raw = payload.get("grad_norm")
        if not _finite(raw):
            nonfinite = True
        else:
            grads.append(float(raw))
        if "global_step" in payload:
            steps.append(int(payload["global_step"]))
    for match in re.finditer(r"global_step=(\d+)", text or ""):
        steps.append(int(match.group(1)))
    if re.search(r"\b(?:nan|inf|-inf)\b", (text or "").lower()) and re.search(
            r"train/loss=(?:nan|inf|-inf)|BEVERIN_GRAD_NORM", text or "", re.I):
        nonfinite = True
    return {"losses": losses, "grad_norms": grads, "steps": steps, "nonfinite": nonfinite,
            "failure": "BEVERIN_NUMERICAL_FAILURE" in (text or ""),
            "saving": "Saving checkpoint" in (text or "") or "BEVERIN_SAVE_START" in (text or "")}


def _spike(values, ratio, minimum=20):
    if len(values) <= minimum:
        return False
    baseline = values[-(minimum + 1):-1]
    if not baseline or not _finite(values[-1]):
        return False
    return values[-1] > ratio * statistics.median(baseline)


def assess(snapshot):
    """Return alert strings for one snapshot. Empty means quiet."""
    alerts = []
    if snapshot.get("nonfinite") or snapshot.get("failure_marker"):
        alerts.append("non-finite training values or a numerical failure marker")
    losses = snapshot.get("losses") or []
    grads = snapshot.get("grad_norms") or []
    if _spike(losses, LOSS_RATIO):
        alerts.append("loss exceeded 1.3 times its rolling median")
    if _spike(grads, GRAD_RATIO):
        alerts.append("gradient norm exceeded 5 times its rolling median")
    if snapshot.get("ledger_terminal"):
        alerts.append("rollback ledger recorded a terminal decision")
    verified_age = snapshot.get("verified_age_s")
    step_s = snapshot.get("step_seconds") or 20
    saving = snapshot.get("saving")
    if (snapshot.get("phase") == "training" and not saving and verified_age is not None
            and verified_age > 2 * SAVE_EVERY * float(step_s) and int(snapshot.get("global_step") or 0) >= SAVE_EVERY):
        alerts.append("no newly verified checkpoint for more than two save intervals")
    step_age = snapshot.get("step_age_s")
    if snapshot.get("phase") == "training" and not saving and step_age is not None and step_age > STALL_S:
        alerts.append("no training step for 15 minutes")
    avail = snapshot.get("min_avail_gib")
    if avail is not None and avail < LOW_MEM_GIB:
        alerts.append("available memory is low")
    return alerts


def heartbeat_due(last, now):
    return last is None or now - last >= HEARTBEAT_S


def format_heartbeat(snapshot):
    step = int(snapshot.get("global_step") or 0)
    rate = snapshot.get("step_seconds")
    remaining = max(0, EXPECTED_STEPS - step)
    eta = None if not rate else round(remaining * float(rate) / 3600.0, 2)
    return {
        "kind": "three-hour heartbeat",
        "utc": snapshot.get("utc"),
        "phase": snapshot.get("phase"),
        "progress": "%s/%s" % (step, EXPECTED_STEPS),
        "last_verified_step": snapshot.get("verified_step"),
        "step_seconds": rate,
        "training_eta_hours": eta,
        "note": "ETA excludes validation, queue, and restart overhead",
    }


def safe_token(value):
    if not re.match(r"^[A-Za-z0-9._-]+$", value or ""):
        raise ValueError("refusing an unsafe run or job id")
    return value


def remote_script(run_id, jobs):
    run_id = safe_token(run_id)
    jobs = [safe_token(job) for job in jobs]
    return r'''
import json, os, re, time
from pathlib import Path
run = Path("/iopsstor/scratch/cscs/faruk_zahiragic/beverin-feasibility/runs") / "%s"
jobs = %s
def tail(path, limit=400000):
    if not path.is_file():
        return ""
    data = path.read_bytes()
    return data[-limit:].decode("utf-8", "replace")
text = ""
logs = list((run / "logs").glob("train-*-node0.log")) if (run / "logs").is_dir() else []
log = max(logs, key=lambda p: p.stat().st_mtime) if logs else None
if log:
    text = tail(log)
state = {}
state_path = run / "checkpoints" / "0" / "training_state.json"
if state_path.is_file():
    try:
        state = json.loads(state_path.read_text())
    except ValueError:
        state = {}
marker = (run / "checkpoints" / "0" / "verified-finite.json").is_file()
verified_step = None
verified_age = None
if marker:
    try:
        payload = json.loads((run / "checkpoints" / "0" / "verified-finite.json").read_text())
        verified_step = payload.get("global_step")
        verified_age = time.time() - (run / "checkpoints" / "0" / "verified-finite.json").stat().st_mtime
    except (ValueError, OSError):
        pass
ledger = {}
ledger_path = run / "checkpoints" / "rollback-ledger.json"
if ledger_path.is_file():
    try:
        ledger = json.loads(ledger_path.read_text())
    except ValueError:
        ledger = {}
failure = (run / "checkpoints" / "numerical-failure.json").is_file() or ("BEVERIN_NUMERICAL_FAILURE" in text)
nonfinite = bool(re.search(r"train/loss=(?:nan|inf|-inf)", text, re.I))
losses = [float(v) for v in re.findall(r"train/loss=([0-9.eE+-]+)", text)[-60:]]
grads = []
for blob in re.findall(r"BEVERIN_GRAD_NORM (\{.*\})", text)[-60:]:
    try:
        raw = json.loads(blob).get("grad_norm")
        grads.append(float(raw))
    except (ValueError, TypeError):
        nonfinite = True
steps = [int(v) for v in re.findall(r"global_step=(\d+)", text)]
step = state.get("global_step") or (steps[-1] if steps else 0)
mem = None
for path in Path("/iopsstor/scratch/cscs/faruk_zahiragic/beverin-feasibility/logs").glob("mem-*-node*.log"):
    samples = re.findall(r"avail_gb=(\d+)", tail(path, 20000))
    if samples:
        mem = min([mem] + [int(s) for s in samples]) if mem is not None else min(int(s) for s in samples)
print(json.dumps({
    "utc": time.strftime("%%Y-%%m-%%dT%%H:%%M:%%SZ", time.gmtime()),
    "run": run.name, "global_step": step, "verified_step": verified_step,
    "verified_age_s": verified_age, "losses": losses, "grad_norms": grads,
    "nonfinite": nonfinite, "failure_marker": failure,
    "ledger_terminal": any(e.get("kind") == "terminal" for e in ledger.get("events") or [] if isinstance(e, dict)),
    "ledger_events": len(ledger.get("events") or []),
    "saving": "BEVERIN_SAVE_START" in text and "BEVERIN_SAVE_END" not in text[-2000:],
    "phase": "complete" if (run / "checkpoints" / "single-epoch-complete.json").is_file() else "training",
    "min_avail_gib": mem, "attempt": log.name if log else None,
    "jobs": jobs,
}))
''' % (run_id, json.dumps(jobs))


def poll(directory, run_id, jobs):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    state_path = directory / "state.json"
    state = json.loads(state_path.read_text()) if state_path.exists() else {}
    now = time.time()
    result = subprocess.run(SSH, input=remote_script(run_id, jobs), capture_output=True, text=True, timeout=90)
    if result.returncode:
        message = {"utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                   "error": (result.stderr or "")[-500:], "training_state": "unknown"}
        print("MONITOR_SSH " + json.dumps(message), flush=True)
        state["last_error"] = message
        state_path.write_text(json.dumps(state, indent=2))
        return message
    snapshot = json.loads(result.stdout.strip().splitlines()[-1])
    alerts = assess(snapshot)
    beat = None
    if heartbeat_due(state.get("last_heartbeat"), now):
        beat = format_heartbeat(snapshot)
        state["last_heartbeat"] = now
        print("MONITOR_HEARTBEAT " + json.dumps(beat), flush=True)
    if alerts:
        print("MONITOR_ALERT " + json.dumps({"alerts": alerts, "snapshot": snapshot}), flush=True)
        with (directory / "alerts.jsonl").open("a") as handle:
            handle.write(json.dumps({"utc": snapshot.get("utc"), "alerts": alerts}) + "\n")
    state["snapshot"] = snapshot
    state_path.write_text(json.dumps(state, indent=2))
    return {"alerts": alerts, "heartbeat": beat, "snapshot": snapshot}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state-dir", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--jobs", nargs="*", default=[])
    parser.add_argument("--check-log", help="Assess a local log instead of SSH")
    args = parser.parse_args()
    if args.check_log:
        parsed = parse_log(Path(args.check_log).read_text(errors="replace"))
        parsed["phase"] = "training"
        print(json.dumps({"alerts": assess(parsed), "parsed_nonfinite": parsed["nonfinite"]}, indent=2))
        return
    poll(args.state_dir, args.run_id, args.jobs)


if __name__ == "__main__":
    main()
