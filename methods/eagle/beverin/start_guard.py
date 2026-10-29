"""Refuse to start a Beverin training link unless the run was handed over.

Usage (from the generated batch script, on the host, Python 3.6):
    python3 start_guard.py RUN_DIR

Passes only when RUN_DIR/cutover.json exists (written after every Clariden
link has ended) and checkpoints/resume is at or past its step, and no Clariden
link is recorded as having trained since: either training-incomplete.json is
present (the last link saved and stopped cleanly) or the last job recorded in
checkpoints/resume/state.json is not one of cutover.json's Clariden job ids.
Exit status 0 = start, 1 = refuse.
"""

import json
import os
import sys


def check(run_dir):
    cutover_path = os.path.join(run_dir, "cutover.json")
    state_path = os.path.join(run_dir, "checkpoints", "resume", "state.json")
    marker_path = os.path.join(run_dir, "training-incomplete.json")
    if not os.path.isfile(cutover_path):
        return False, "no cutover.json: the run has not been handed over to Beverin"
    with open(cutover_path) as handle:
        cutover = json.load(handle)
    if not os.path.isfile(state_path):
        return False, "no checkpoints/resume/state.json"
    with open(state_path) as handle:
        state = json.load(handle)
    step = int(state["step"])
    if step < int(cutover["step"]):
        return False, f"resume step {step} is before the cutover step {cutover['step']}"
    clariden = set(str(job) for job in cutover.get("clariden_job_ids", []))
    last_job = str((state.get("jobs") or [{}])[-1].get("slurm_job_id"))
    if os.path.isfile(marker_path):
        return True, f"resume step {step}, last link stopped cleanly (job {last_job})"
    if last_job in clariden:
        return False, f"no training-incomplete.json; the last recorded job {last_job} is Clariden's"
    return True, f"resume step {step}, last recorded job {last_job} is not a Clariden job"


def main(argv):
    if len(argv) != 2:
        sys.stderr.write("usage: start_guard.py RUN_DIR\n")
        return 2
    ok, reason = check(argv[1])
    print(f"START_GUARD {'pass' if ok else 'REFUSE'}: {reason}")
    sys.stdout.flush()
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv))
