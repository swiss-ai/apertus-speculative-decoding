"""Replace only this pilot's terminal/queued deployments; preserve all evidence."""
import json
import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

from apertus_bench import research_pilot as h

root = h.REPO / "results/8b/serving-pilot"
operation = root / "operations" / ("logging-flag-recovery-" + h.stamp())
operation.mkdir(parents=True)
shutil.copy2(__file__, operation / "maintenance.py")
assert h.git_info(h.REPO)["revision"] == "c18ea9c0b8e2e576766b8d788f1a67380c3e2808"
assert subprocess.check_output(["git", "branch", "--show-current"], text=True).strip() == "feat/dspark-research-serving-pilot"
items = [
    ("dev-dspark-20261028-a", "3783300", None, "dev-dspark-20261029-a", "dspark", "debug"),
    ("dev-plain-20261028-b", "3783531", 285639, "dev-plain-20261029-a", "plain", "normal"),
    ("dev-plain-control-20261028-a", "3783544", 285695, "dev-plain-control-20261029-a", "plain", "normal"),
]

def status(job):
    rows = subprocess.check_output(["sacct", "-X", "-n", "-P", "-j", job, "-o", "JobID,User,State"], text=True).strip().splitlines()
    assert len(rows) == 1, rows
    values = rows[0].split("|")
    assert values[0] == job and values[1] == "faruk_zahiragic", values
    return values[2].split()[0]

def cli(*argv, log=None):
    return subprocess.run([sys.executable, "-m", "apertus_bench.research_pilot", *map(str, argv)], check=True, stdout=log, stderr=subprocess.STDOUT if log else None)

# Read and verify all existing state before preparing or stopping anything.
for old, job, pid, new, variant, partition in items:
    run = root / old
    assert h.read_json(run / "job.json")["job_id"] == job
    state = status(job)
    assert state == ("FAILED" if variant == "dspark" else "PENDING"), (job, state)
    assert not (root / new).exists(), new
    if pid:
        proc = Path(f"/proc/{pid}")
        assert proc.stat().st_uid == os.getuid()
        args = (proc / "cmdline").read_bytes().split(b"\0")
        assert b"validate" in args and str(run).encode() in args, args
    h.write_json(operation / (old + "-before.json"), h.inspect_run(run))

# Render and verify concrete replacements before cancelling the queued jobs.
for old, job, pid, new, variant, partition in items:
    argv = ["prepare", "--variant", variant, "--partition", partition, "--time", "01:30:00", "--output", str(root / new)]
    if "control" in new:
        argv += ["--model-suffix", "control"]
    with (operation / (new + "-prepare.log")).open("x") as log:
        cli(*argv, log=log)
    run = root / new
    h.verify_manifest(run, "launch-manifest.json")
    for script in (run / "rendered").glob("*.sh"):
        subprocess.run(["bash", "-n", str(script)], check=True)
    master = (run / "rendered/master.sh").read_text()
    assert "--no-enable-log-requests" in master and "--disable-log-requests" not in master

for old, job, pid, new, variant, partition in items:
    run = root / old
    h.write_json(operation / (old + "-replacement-intent.json"), {
        "at": h.now(), "job": job, "controller_pid": pid, "replacement": str(root / new),
        "reason": "Pinned vLLM API parser rejects --disable-log-requests; use --no-enable-log-requests.",
    })
    if pid:
        assert status(job) == "PENDING", job
        os.kill(pid, signal.SIGTERM)
        for attempt in range(50):
            if not Path(f"/proc/{pid}").exists():
                break
            time.sleep(0.1)
        else:
            raise RuntimeError(f"Controller {pid} did not exit; no cancellation performed")
        # Recheck status at the point of cancellation. Never stop an unexpected live job.
        assert status(job) == "PENDING", job
        h.stop(run)
    else:
        h.collect(run)
    h.write_json(operation / (old + "-after.json"), h.inspect_run(run))

for old, job, pid, new, variant, partition in items:
    run = root / new
    record = h.submit(run)
    h.write_json(run / "replaces.json", {"at": h.now(), "run": str(root / old), "job_id": job, "operation": str(operation)})
    argv = [sys.executable, "-m", "apertus_bench.research_pilot", "validate", str(run), "--wait-seconds", "129600"]
    if "control" in new:
        argv += ["--smoke-only", "--stop-after"]
    with (run / "validation-controller.log").open("x") as log:
        proc = subprocess.Popen(argv, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True, env=h.job_environment())
    (run / "validation-controller.pid").write_text(str(proc.pid) + "\n")
    h.write_json(operation / (new + "-submission.json"), {"at": h.now(), "run": str(run), "job": record, "controller_pid": proc.pid, "controller_argv": argv})
    print(json.dumps({"run": new, "job": record["job_id"], "controller_pid": proc.pid}), flush=True)

print(operation, flush=True)
