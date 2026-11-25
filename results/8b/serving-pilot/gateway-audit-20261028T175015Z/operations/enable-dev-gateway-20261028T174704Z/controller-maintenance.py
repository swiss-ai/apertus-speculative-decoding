"""Enable gateway validation in this campaign's waiting controllers only."""
import datetime
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request

from apertus_bench.pilot_validation import gateway_key


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def save(path, value):
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")


def command(argv):
    return subprocess.check_output(argv, text=True).strip()


root = Path(sys.argv[1]).resolve()
base = root / "results/8b/serving-pilot"
branch = command(["git", "-C", str(root), "branch", "--show-current"])
revision = command(["git", "-C", str(root), "rev-parse", "HEAD"])
if branch != "feat/dspark-research-serving-pilot" or revision != sys.argv[2]:
    raise RuntimeError("expected pilot task branch and credential-resolver revision")
tag = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
operation = base / "operations" / ("enable-dev-gateway-" + tag)
operation.mkdir(parents=True)
shutil.copy2(__file__, operation / "controller-maintenance.py")
run = base / "dev-dspark-20261028-a"
key = gateway_key(run, "CSCS_SERVING_API", "dev")
request = urllib.request.Request(
    "https://apidev.swissai.svc.cscs.ch/v1/chat/completions",
    data=json.dumps({"model": "faruk_zahiragic/swiss-ai/pilot-auth-check-unavailable",
        "messages": [{"role": "user", "content": "synthetic authentication check"}],
        "max_tokens": 1}).encode(),
    headers={"Authorization": "Bearer " + key, "Content-Type": "application/json"})
try:
    with urllib.request.urlopen(request, timeout=15) as response:
        status, body = response.status, response.read(4096)
except urllib.error.HTTPError as error:
    status, body = error.code, error.read(4096)
accepted = status in (400, 503) and b"no provider" in body.lower()
save(operation / "credential-check.json", {
    "at": now(), "gateway": "https://apidev.swissai.svc.cscs.ch",
    "credential_source": "keyring:apertus-speculative-decoding/dev-api-key",
    "probe_model": "faruk_zahiragic/swiss-ai/pilot-auth-check-unavailable",
    "http_status": status, "no_provider_response": b"no provider" in body.lower(),
    "authentication_accepted": accepted, "live_completion_tested": False})
del key, request, body
if not accepted:
    raise RuntimeError("dev gateway did not accept the saved credential")
events = []
targets = [
    ("dev-dspark-20261028-a", 83195, []),
    ("dev-plain-20261028-b", 160699, []),
    ("dev-plain-control-20261028-a", 160822, ["--smoke-only", "--stop-after"]),
]
try:
    for name, previous_pid, extra in targets:
        run = base / name
        job = json.loads((run / "job.json").read_text())["job_id"]
        state = command(["squeue", "-j", job, "-h", "-o", "%T"])
        event = {"at": now(), "run": str(run), "job_id": job, "job_state": state,
            "previous_pid": previous_pid, "branch": branch, "revision": revision,
            "reason": "Enable authenticated dev gateway checks using the separate keyring entry"}
        if state != "PENDING" or any((run / p).exists() for p in
                ("checks", "measurements", "open-loop", "preflight.json", "preflight-failed.json")):
            event["action"] = "skipped_active_or_started"
            events.append(event)
            continue
        proc = Path("/proc") / str(previous_pid)
        if not proc.exists() or int((run / "validation-controller.pid").read_text()) != previous_pid:
            raise RuntimeError("waiting controller process identity changed")
        args = (proc / "cmdline").read_bytes().split(b"\0")
        expected = [b"-m", b"apertus_bench.research_pilot", b"validate", str(run).encode()]
        if proc.stat().st_uid != os.getuid() or b"--no-gateway" not in args or not any(
            args[i:i+4] == expected for i in range(len(args))):
            raise RuntimeError("refusing to signal an unrelated controller process")
        event["previous_argv"] = [a.decode() for a in args if a]
        save(operation / (name + "-intent.json"), event)
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
                raise RuntimeError("previous controller still owns its validation lock")
        argv = [str(root / "serving/research-pilot.sh"), "validate", str(run),
                "--wait-seconds", "129600", *extra]
        with (run / "validation-controller.log").open("ab") as log:
            child = subprocess.Popen(argv, cwd=root, stdin=subprocess.DEVNULL,
                stdout=log, stderr=subprocess.STDOUT, start_new_session=True, close_fds=True)
        (run / "validation-controller.pid").write_text(str(child.pid) + "\n")
        event.update(action="replaced_waiting_controller", new_pid=child.pid,
            new_argv=argv, wait_seconds=129600, gateway=True,
            serving_job_restarted=False, source_manifest_sha256=hashlib.sha256(
                (run / "source-manifest.json").read_bytes()).hexdigest())
        save(run / ("controller-replacement-" + tag + ".json"), event)
        events.append(event)
finally:
    save(operation / "events.json", events)
print(json.dumps({"operation": str(operation), "events": events}, indent=2))
