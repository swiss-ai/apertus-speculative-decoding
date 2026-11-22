"""Extend only this campaign's waiting controllers, preserving all Slurm jobs."""
import datetime
import fcntl
import json
import os
import pathlib
import shutil
import signal
import subprocess
import sys
import time

root = pathlib.Path(sys.argv[1]).resolve()
tag = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
operation = root / "results/8b/serving-pilot/operations" / ("extend-waits-" + tag)
operation.mkdir(parents=True)
shutil.copy2(__file__, operation / "controller-maintenance.py")
targets = [
    ("dev-dspark-20261028-a", 243388, []),
]
events = []
for name, previous_pid, extra in targets:
    run = root / "results/8b/serving-pilot" / name
    job = json.loads((run / "job.json").read_text())["job_id"]
    state = subprocess.check_output(
        ["squeue", "-j", job, "-h", "-o", "%T"], text=True
    ).strip()
    event = {"run": str(run), "job_id": job, "job_state": state, "previous_pid": previous_pid}
    if state != "PENDING" or any((run / p).exists() for p in ("checks", "measurements", "open-loop")):
        event["action"] = "skipped_active_or_started"
        events.append(event)
        continue
    proc = pathlib.Path("/proc") / str(previous_pid)
    if not proc.exists():
        event["action"] = "skipped_missing_controller"
        events.append(event)
        continue
    args = (proc / "cmdline").read_bytes().split(b"\0")
    expected = [b"-m", b"apertus_bench.research_pilot", b"validate", str(run).encode()]
    if proc.stat().st_uid != os.getuid() or not any(
        args[i:i+4] == expected for i in range(len(args))
    ):
        raise RuntimeError("controller process identity differs; refusing to signal it")
    event["previous_argv"] = [a.decode() for a in args if a]
    with (operation / (name + "-intent.json")).open("x") as out:
        json.dump(event, out, indent=2)
    os.kill(previous_pid, signal.SIGTERM)
    with (run / ".validate.lock").open("a") as lock:
        for _ in range(100):
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                fcntl.flock(lock, fcntl.LOCK_UN)
                break
            except BlockingIOError:
                time.sleep(0.1)
        else:
            raise RuntimeError("old controller still owns its validation lock")
    argv = [str(root / "serving/research-pilot.sh"), "validate", str(run),
            "--no-gateway", "--wait-seconds", "129600", *extra]
    with (run / "validation-controller.log").open("ab") as log:
        child = subprocess.Popen(argv, cwd=root, stdin=subprocess.DEVNULL,
            stdout=log, stderr=subprocess.STDOUT, start_new_session=True, close_fds=True)
    (run / "validation-controller.pid").write_text(str(child.pid) + "\n")
    event.update(action="replaced_waiting_controller", new_pid=child.pid,
                 new_argv=argv, wait_seconds=129600)
    with (run / ("controller-replacement-" + tag + ".json")).open("x") as out:
        json.dump(event, out, indent=2)
    events.append(event)
with (operation / "events.json").open("x") as out:
    json.dump(events, out, indent=2)
print(json.dumps({"operation": str(operation), "events": events}, indent=2))
