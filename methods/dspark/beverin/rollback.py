"""Bounded checkpoint rollback. Standard library, Python 3.6 safe.

Rules for one campaign:
- at most 3 numerical rollbacks;
- the same failing step rolls back to the newest verified copy, then to the
  older verified copy; a third occurrence at that step is terminal;
- crashes and hangs get 1 resume per allocation and 2 per campaign;
- no rollback in the last 30 minutes before DSPARK_STOP_AT;
- before every torchrun, checkpoints/0 is the newest verified checkpoint or
  the run refuses to start;
- after 2 numerical rollbacks an alert is recorded and nothing automatic is
  changed in the learning rate, batch, data, or optimizer.
"""
import argparse
import json
import os
import re
import time

from checkpoint_layout import (
    atomic_json,
    list_verified,
    prepare_for_start,
    read_json,
    read_marker,
    restore_step,
    retire,
)

LEDGER = "rollback-ledger.json"
DEADLINE_S = 1800
NUMERICAL_LIMIT = 3
CRASH_PER_JOB = 1
CRASH_PER_CAMPAIGN = 2
LOSS_WINDOW = 20
LOSS_RATIO = 1.3
PROPOSAL = (
    "Two numerical rollbacks have been used. A lower peak learning rate is a "
    "candidate for a separately reviewed campaign. This campaign does not change "
    "learning rate, batch, data, or optimizer."
)


def ledger_path(root):
    return os.path.join(root, LEDGER)


def empty_ledger():
    return {
        "numerical_rollbacks": 0,
        "crash_rollbacks": 0,
        "crash_by_job": {},
        "step_failures": {},
        "events": [],
        "alerts": [],
        "pending_continuity": None,
    }


def load_ledger(root):
    payload = read_json(ledger_path(root))
    if not isinstance(payload, dict):
        payload = empty_ledger()
    for key, value in empty_ledger().items():
        payload.setdefault(key, value if not isinstance(value, dict) else dict(value))
    payload["crash_by_job"] = dict(payload.get("crash_by_job") or {})
    payload["step_failures"] = dict(payload.get("step_failures") or {})
    payload["events"] = list(payload.get("events") or [])
    payload["alerts"] = list(payload.get("alerts") or [])
    return payload


def save_ledger(root, ledger):
    atomic_json(ledger_path(root), ledger)


def _tail(path, limit=1048576):
    if not path or not os.path.isfile(path):
        return ""
    with open(path, "rb") as handle:
        handle.seek(0, os.SEEK_END)
        size = handle.tell()
        handle.seek(max(0, size - limit))
        return handle.read().decode("utf-8", "replace")


def finite_losses(text):
    losses = []
    for match in re.finditer(r"train/loss=([^\s,]+)", text or ""):
        raw = match.group(1)
        try:
            value = float(raw)
        except ValueError:
            continue
        if value == value and value not in (float("inf"), float("-inf")):
            losses.append(value)
    return losses


def _median(values):
    if not values:
        return None
    ordered = sorted(values)
    mid = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[mid]
    return 0.5 * (ordered[mid - 1] + ordered[mid])


def failure_step(root, text):
    marker = read_json(os.path.join(root, "numerical-failure.json"))
    if isinstance(marker, dict) and "global_step" in marker:
        try:
            return int(marker["global_step"])
        except (TypeError, ValueError):
            return None
    found = re.findall(r"BEVERIN_NUMERICAL_FAILURE\s+(\{.*\})", text or "")
    for blob in reversed(found):
        try:
            return int(json.loads(blob)["global_step"])
        except (ValueError, KeyError, TypeError):
            continue
    return None


def _archive_failure(root, step):
    base = os.path.join(root, "evidence")
    os.makedirs(base, exist_ok=True)
    folder = None
    for nonce in range(1000):
        candidate = os.path.join(base, "failure-step-%s-%d-%d" % (step, int(time.time() * 1000), nonce))
        try:
            os.makedirs(candidate)
        except OSError:
            continue
        folder = candidate
        break
    if folder is None:
        raise RuntimeError("could not archive the numerical failure")
    marker = os.path.join(root, "numerical-failure.json")
    if os.path.exists(marker):
        os.rename(marker, os.path.join(folder, "numerical-failure.json"))
    batches = os.path.join(root, "numerical-failure-batches")
    if os.path.isdir(batches):
        os.rename(batches, os.path.join(folder, "numerical-failure-batches"))
    rejected = os.path.join(root, "save-rejected.json")
    if os.path.exists(rejected):
        os.rename(rejected, os.path.join(folder, "save-rejected.json"))
    return folder


def _event(kind, **fields):
    fields["kind"] = kind
    fields["utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    return fields


def _finish(root, ledger, action, token):
    save_ledger(root, ledger)
    payload = dict(action)
    payload["token"] = token
    atomic_json(os.path.join(root, ".rollback-action.json"), payload)
    print("BEVERIN_ROLLBACK " + json.dumps(payload), flush=True)
    return payload


def _terminal(root, ledger, token, reason, **fields):
    event = _event("terminal", reason=reason, **fields)
    ledger["events"].append(event)
    return _finish(root, ledger, {"action": "terminal", "reason": reason}, token)


def _time_left(stop_at):
    try:
        stop = float(stop_at or 0)
    except (TypeError, ValueError):
        stop = 0
    if stop <= 0:
        return None
    return stop - time.time()


def decide(root, return_code, log_path, job_id, stop_at, token):
    """Choose proceed, rollback, or terminal. Node 0 is the only caller."""
    os.makedirs(root, exist_ok=True)
    ledger = load_ledger(root)
    text = _tail(log_path)
    left = _time_left(stop_at)
    validation = "Final validation did not return finite loss and metrics" in text
    marker_exists = os.path.isfile(os.path.join(root, "numerical-failure.json"))
    numerical = validation or marker_exists or ("BEVERIN_NUMERICAL_FAILURE" in text)
    if validation:
        return _terminal(root, ledger, token, "validation failure is terminal", job_id=job_id)
    if numerical:
        return _numerical(root, ledger, token, text, job_id, left)
    if int(return_code) == 0:
        event = _event("proceed", job_id=job_id, return_code=0)
        ledger["events"].append(event)
        return _finish(root, ledger, {"action": "proceed", "reason": "clean exit"}, token)
    return _crash(root, ledger, token, job_id, left, int(return_code))


def _deadline(left):
    return left is not None and left < DEADLINE_S


def _numerical(root, ledger, token, text, job_id, left):
    step = failure_step(root, text)
    if step is None:
        return _terminal(root, ledger, token, "numerical failure has no step", job_id=job_id)
    key = str(step)
    ledger["step_failures"][key] = int(ledger["step_failures"].get(key, 0)) + 1
    occurrence = ledger["step_failures"][key]
    if occurrence >= 3:
        _archive_failure(root, step)
        return _terminal(root, ledger, token, "third occurrence at step %s" % step,
                         step=step, job_id=job_id)
    if int(ledger["numerical_rollbacks"]) >= NUMERICAL_LIMIT:
        _archive_failure(root, step)
        return _terminal(root, ledger, token, "numerical rollback budget exhausted",
                         step=step, job_id=job_id)
    if _deadline(left):
        _archive_failure(root, step)
        return _terminal(root, ledger, token, "no rollback within 30 minutes of the deadline",
                         step=step, job_id=job_id)
    verified = list_verified(root)
    candidates = [item for item in verified if item[0] < step]
    if not candidates:
        _archive_failure(root, step)
        return _terminal(root, ledger, token, "no verified checkpoint before the failure",
                         step=step, job_id=job_id)
    if occurrence == 1:
        target = candidates[-1]
    else:
        if len(candidates) < 2:
            _archive_failure(root, step)
            return _terminal(root, ledger, token, "no older verified copy for a repeated step",
                             step=step, job_id=job_id)
        target = candidates[-2]
    target_step = target[0]
    archived = _archive_failure(root, step)
    for verified_step, path, _marker in list(list_verified(root)):
        if verified_step > target_step:
            retire(root, path)
    restore_step(root, target_step)
    marker = read_marker(os.path.join(root, "0"))
    if marker is None or marker["global_step"] != target_step:
        return _terminal(root, ledger, token, "rollback restore failed", step=step, job_id=job_id)
    recent = finite_losses(text)[-50:]
    ledger["numerical_rollbacks"] = int(ledger["numerical_rollbacks"]) + 1
    ledger["pending_continuity"] = {
        "expected_global_step": target_step,
        "learning_rates": marker.get("learning_rates") or [],
        "pre_median_loss": _median(recent),
        "window": LOSS_WINDOW,
        "loss_ratio": LOSS_RATIO,
        "failure_step": step,
    }
    event = _event("numerical-rollback", step=step, resume_step=target_step, occurrence=occurrence,
                   job_id=job_id, evidence=archived, count=ledger["numerical_rollbacks"])
    ledger["events"].append(event)
    if ledger["numerical_rollbacks"] == 2:
        alert = {
            "utc": event["utc"],
            "after_rollbacks": 2,
            "evidence": archived,
            "failure_step": step,
            "resume_step": target_step,
            "proposed_change": PROPOSAL,
        }
        ledger["alerts"].append(alert)
        print("BEVERIN_ROLLBACK_ALERT " + json.dumps(alert), flush=True)
    return _finish(root, ledger, {
        "action": "rollback", "reason": "numerical", "resume_step": target_step,
        "failure_step": step, "clear_hidden_state_cache": True,
    }, token)


def _crash(root, ledger, token, job_id, left, return_code):
    job_key = str(job_id or "local")
    used_job = int(ledger["crash_by_job"].get(job_key, 0))
    used = int(ledger["crash_rollbacks"])
    if used_job >= CRASH_PER_JOB:
        return _terminal(root, ledger, token, "crash resume already used in this allocation",
                         job_id=job_key, return_code=return_code)
    if used >= CRASH_PER_CAMPAIGN:
        return _terminal(root, ledger, token, "crash resume budget exhausted",
                         job_id=job_key, return_code=return_code)
    if _deadline(left):
        return _terminal(root, ledger, token, "no rollback within 30 minutes of the deadline",
                         job_id=job_key, return_code=return_code)
    verified = list_verified(root)
    if not verified:
        return _terminal(root, ledger, token, "crash with no verified checkpoint",
                         job_id=job_key, return_code=return_code)
    target_step = verified[-1][0]
    restore_step(root, target_step)
    marker = read_marker(os.path.join(root, "0"))
    if marker is None or marker["global_step"] != target_step:
        return _terminal(root, ledger, token, "crash restore failed", job_id=job_key)
    ledger["crash_rollbacks"] = used + 1
    ledger["crash_by_job"][job_key] = used_job + 1
    ledger["pending_continuity"] = {
        "expected_global_step": target_step,
        "learning_rates": marker.get("learning_rates") or [],
        "pre_median_loss": None,
        "window": LOSS_WINDOW,
        "loss_ratio": LOSS_RATIO,
        "failure_step": None,
    }
    ledger["events"].append(_event(
        "crash-rollback", resume_step=target_step, job_id=job_key, return_code=return_code,
        count=ledger["crash_rollbacks"]))
    rejected = os.path.join(root, "save-rejected.json")
    if os.path.exists(rejected):
        os.makedirs(os.path.join(root, "evidence"), exist_ok=True)
        os.rename(rejected, os.path.join(root, "evidence", "save-rejected-%s.json" % job_key))
    return _finish(root, ledger, {
        "action": "rollback", "reason": "crash", "resume_step": target_step,
        "clear_hidden_state_cache": True,
    }, token)


def prepare(root, node_rank, token, log_path="", job_id="local", stop_at=0):
    os.makedirs(root, exist_ok=True)
    status_path = os.path.join(root, ".prepare-status.json")
    if int(node_rank) == 0:
        if os.path.isfile(os.path.join(root, "numerical-failure.json")):
            action = decide(root, 1, log_path, job_id, stop_at, token + "-marker")
            if action.get("action") == "terminal":
                result = {"ok": False, "reason": action.get("reason"), "token": token}
                atomic_json(status_path, result)
                return result
        result = prepare_for_start(root)
        result["token"] = token
        atomic_json(status_path, result)
        return result
    deadline = time.time() + 600
    while time.time() < deadline:
        payload = read_json(status_path)
        if isinstance(payload, dict) and payload.get("token") == token:
            return payload
        time.sleep(2)
    return {"ok": False, "reason": "prepare status timeout", "token": token}


def wait_action(root, token, timeout):
    path = os.path.join(root, ".rollback-action.json")
    deadline = time.time() + float(timeout)
    while time.time() < deadline:
        payload = read_json(path)
        if isinstance(payload, dict) and payload.get("token") == token and payload.get("action"):
            return payload
        time.sleep(2)
    return {"action": "terminal", "reason": "rollback action timeout", "token": token}


def _cli():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command")
    prepare_parser = sub.add_parser("prepare")
    prepare_parser.add_argument("--checkpoints", required=True)
    prepare_parser.add_argument("--node-rank", type=int, required=True)
    prepare_parser.add_argument("--token", required=True)
    prepare_parser.add_argument("--log", default="")
    prepare_parser.add_argument("--job-id", default="local")
    prepare_parser.add_argument("--stop-at", default="0")
    decide_parser = sub.add_parser("decide")
    decide_parser.add_argument("--checkpoints", required=True)
    decide_parser.add_argument("--return-code", type=int, required=True)
    decide_parser.add_argument("--log", default="")
    decide_parser.add_argument("--job-id", default="local")
    decide_parser.add_argument("--stop-at", default="0")
    decide_parser.add_argument("--token", required=True)
    wait_parser = sub.add_parser("wait-action")
    wait_parser.add_argument("--checkpoints", required=True)
    wait_parser.add_argument("--token", required=True)
    wait_parser.add_argument("--timeout", type=float, default=600)
    args = parser.parse_args()
    if args.command == "prepare":
        result = prepare(args.checkpoints, args.node_rank, args.token, args.log, args.job_id, args.stop_at)
        print(("PREPARE_OK " if result.get("ok") else "PREPARE_REFUSED ") + json.dumps(result))
        raise SystemExit(0 if result.get("ok") else 66)
    if args.command == "decide":
        result = decide(args.checkpoints, args.return_code, args.log, args.job_id, args.stop_at, args.token)
        raise SystemExit(0 if result.get("action") != "terminal" else 65)
    if args.command == "wait-action":
        result = wait_action(args.checkpoints, args.token, args.timeout)
        print(result.get("action", "terminal"))
        raise SystemExit(0 if result.get("action") != "terminal" else 65)
    parser.error("command required")


if __name__ == "__main__":
    _cli()
