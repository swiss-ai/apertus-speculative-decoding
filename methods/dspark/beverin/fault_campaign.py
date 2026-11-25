"""One-node demonstration of bounded rollback and rejected checkpoint writes.

This is the fault-injection campaign, not a production trainer. It refuses to
run unless RUN_ID starts with faultinject- and NODES/SLURM is 1.
"""
import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from checkpoint_layout import MARKER, atomic_json, list_verified, promote, read_marker
from fault_inject import injection_allowed
from rollback import decide, load_ledger


def _write_verified(root, step):
    staging = os.path.join(root, ".demo-staging-%d" % step)
    os.makedirs(staging)
    with open(os.path.join(staging, "model.safetensors"), "wb") as handle:
        handle.write(b"model-step-%d" % step)
    with open(os.path.join(staging, "optimizer_state_dict.pt"), "wb") as handle:
        handle.write(b"optimizer-step-%d" % step)
    with open(os.path.join(staging, "training_state.json"), "w") as handle:
        json.dump({"epoch": 0, "local_step": step, "global_step": step}, handle)
    atomic_json(os.path.join(staging, MARKER), {
        "finite": True,
        "global_step": step,
        "epoch": 0,
        "local_step": step,
        "learning_rates": [0.0001],
        "utc": "2026-10-31T00:00:00Z",
    })
    promote(root, staging)


def _log(path, step):
    with open(path, "a") as handle:
        for value in (0.50, 0.52, 0.48, 0.51):
            handle.write("train/loss=%.2f global_step=%d\n" % (value, step))
        handle.write('BEVERIN_NUMERICAL_FAILURE {"global_step": %d, "phase": "updated-parameters"}\n' % step)


def _fail(root, log_path, step, job_id):
    atomic_json(os.path.join(root, "numerical-failure.json"), {
        "global_step": step, "phase": "updated-parameters",
    })
    _log(log_path, step)
    token = "%s-%d-%d" % (job_id, step, load_ledger(root)["step_failures"].get(str(step), 0))
    return decide(root, 1, log_path, job_id, 0, token)


def demonstrate(root, log_path, job_id="fault-demo"):
    """NaN at step 12, a repeat, then a terminal third occurrence, then a bad write."""
    if not injection_allowed():
        raise RuntimeError(
            "fault campaign refused: RUN_ID must start with faultinject- and NODES must be 1")
    os.makedirs(root, exist_ok=True)
    _write_verified(root, 5)
    _write_verified(root, 10)
    first = _fail(root, log_path, 12, job_id)
    if first["action"] != "rollback" or first["resume_step"] != 10:
        raise RuntimeError("expected rollback to step 10, got %s" % first)
    if read_marker(os.path.join(root, "0"))["global_step"] != 10:
        raise RuntimeError("checkpoints/0 was not the step-10 checkpoint")
    second = _fail(root, log_path, 12, job_id)
    if second["action"] != "rollback" or second["resume_step"] != 5:
        raise RuntimeError("expected rollback to the older checkpoint, got %s" % second)
    if read_marker(os.path.join(root, "0"))["global_step"] != 5:
        raise RuntimeError("older verified copy was not restored")
    if not os.path.isdir(os.path.join(root, "retired")):
        raise RuntimeError("the repeated step did not retire the newer checkpoint")
    third = _fail(root, log_path, 12, job_id)
    if third["action"] != "terminal":
        raise RuntimeError("third occurrence should stop, got %s" % third)
    if read_marker(os.path.join(root, "0"))["global_step"] != 5:
        raise RuntimeError("terminal failure changed checkpoints/0")
    intact = open(os.path.join(root, "0", "model.safetensors"), "rb").read()
    staging = os.path.join(root, ".corrupt-staging")
    os.makedirs(staging)
    with open(os.path.join(staging, "model.safetensors"), "wb") as handle:
        handle.write(intact)
    from fault_inject import corrupt_checkpoint
    corrupt_checkpoint(staging)
    if open(os.path.join(staging, "model.safetensors"), "rb").read() == intact:
        raise RuntimeError("corruption did not change the staging file")
    if open(os.path.join(root, "0", "model.safetensors"), "rb").read() != intact:
        raise RuntimeError("corrupted write replaced checkpoints/0")
    ledger = load_ledger(root)
    if len(ledger["alerts"]) != 1:
        raise RuntimeError("expected one alert after two rollbacks")
    if "does not change" not in ledger["alerts"][0]["proposed_change"]:
        raise RuntimeError("alert did not record the proposed change")
    steps = [item[0] for item in list_verified(root)]
    report = {
        "first": first, "second": second, "third": third,
        "verified_steps": steps, "alerts": len(ledger["alerts"]),
        "zero_intact_after_corrupt": True,
    }
    print("FAULT_CAMPAIGN_OK " + json.dumps({
        "rolled_back_to": [10, 5], "third": "terminal", "zero_intact": True,
        "verified_steps": steps,
    }), flush=True)
    return report


def main():
    if not injection_allowed():
        raise SystemExit("FAULT_CAMPAIGN_REFUSED")
    root = os.environ.get("FAULT_ROOT") or tempfile.mkdtemp(prefix="faultinject-")
    log_path = os.path.join(root, "train.log")
    demonstrate(root, log_path, os.environ.get("SLURM_JOB_ID", "local"))
    out = os.environ.get("FAULT_REPORT")
    if out:
        with open(out, "w") as handle:
            json.dump({"root": root, "ok": True}, handle)


if __name__ == "__main__":
    main()
