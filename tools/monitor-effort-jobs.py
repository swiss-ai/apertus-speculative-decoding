#!/usr/bin/env python3
"""Check the active Phase 2 run and wake its existing Codex thread on failures.

Installed through a scoped cron entry. No GPU work or repairs execute here:
repairs run in the original conversation with its instructions and tools.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import subprocess
import time
from datetime import UTC, datetime
from pathlib import Path

MARKER = "apertus-thinking-effort-monitor"
PREFIX = "[Effort job monitor]"
FAILURES = {"FAILED", "CANCELLED", "TIMEOUT", "NODE_FAIL", "OUT_OF_MEMORY", "PREEMPTED"}
REMOTE_PROBE = r"""
from pathlib import Path
import json,os,subprocess
base=Path("/iopsstor/scratch/cscs/faruk_zahiragic/thinking-effort/phase2")
candidates=[p for p in base.glob("campaign-*") if (p/"controller-config.json").is_file()]
if not candidates:raise RuntimeError("no Phase 2 campaign found")
root=max(candidates,key=lambda p:(p/"controller-config.json").stat().st_mtime_ns)
config=json.loads((root/"controller-config.json").read_text())["argv"]
status=json.loads((root/"controller-status.json").read_text())
pid=int((root/"controller.pid").read_text())
try:
 command=Path(f"/proc/{pid}/cmdline").read_bytes()
 alive=b"apertus_bench.effort_training_submit" in command and str(root).encode() in command
except (FileNotFoundError,PermissionError):alive=False
jobs={"qualification":str(config["qualification_job"])}
for path in sorted((root/"submissions").glob("*/job.json")):
 record=json.loads(path.read_text())
 jobs[path.parent.name]=str(record["job_id"])
result=subprocess.run(["sacct","-j",",".join(jobs.values()),"--noheader","--format=JobIDRaw,State%30,ExitCode","-P"],capture_output=True,text=True,check=True,timeout=30)
states={}
for line in result.stdout.splitlines():
 fields=line.split("|")
 if fields[0] in jobs.values():
  states[fields[0]]={"state":fields[1].split()[0],"exit_code":fields[2]}
print(json.dumps({
 "campaign":str(root),"controller":status,"controller_alive":alive,"controller_pid":pid,
 "jobs":{name:{"job_id":job,**states.get(job,{"state":"UNKNOWN"})} for name,job in jobs.items()},
 "qualification_directory":config["qualification"],"phase1_reference":config["phase1_reference"]
}))
"""


class AppServer:
    def __init__(self, codex: str):
        from websockets.sync.client import unix_connect

        daemon = subprocess.run(
            [codex, "app-server", "daemon", "version"],
            capture_output=True,
            text=True,
            check=True,
            timeout=15,
        )
        self.connection = unix_connect(
            json.loads(daemon.stdout)["socketPath"], uri="ws://localhost", open_timeout=10
        )
        self.identifier = 0
        self.request(
            "initialize",
            {
                "clientInfo": {"name": MARKER, "version": "1"},
                "capabilities": {"experimentalApi": True},
            },
        )
        self.connection.send(json.dumps({"method": "initialized"}))

    def request(self, method, params):
        self.identifier += 1
        identifier = self.identifier
        self.connection.send(json.dumps({"id": identifier, "method": method, "params": params}))
        deadline = time.monotonic() + 20
        while time.monotonic() < deadline:
            result = json.loads(self.connection.recv(timeout=max(0, deadline - time.monotonic())))
            if result.get("id") != identifier:
                continue
            if "error" in result:
                raise RuntimeError(f"Codex app-server {method}: {result['error']}")
            return result.get("result")
        raise TimeoutError(f"Codex app-server request timed out: {method}")

    def close(self):
        self.connection.close()


def incident(snapshot):
    control = snapshot["controller"]
    if control.get("status") == "complete":
        return {
            "kind": "complete",
            "report": control.get("report"),
            "decision": control.get("decision"),
        }
    failed = {name: job for name, job in snapshot["jobs"].items() if job["state"] in FAILURES}
    if failed or control.get("status") == "failed":
        return {"kind": "failure", "jobs": failed, "controller_error": control.get("error")}
    if not snapshot["controller_alive"]:
        return {"kind": "controller_stopped", "last_status": control.get("status")}
    return None


def notify(codex, thread, state_dir, finding):
    server = AppServer(codex)
    try:
        queued = server.request("thread/queue/list", {"threadId": thread, "limit": 100})
        for item in queued["data"]:
            if any(PREFIX in value.get("text", "") for value in item["input"]):
                return {"status": "already_queued", "id": item["id"]}
        prompt = (
            f"{PREFIX} Periodic check found {finding['kind']}. Read {state_dir / 'latest.json'} "
            "and tools/monitor-effort-jobs.py. Continue the GPT-OSS DFlash Phase 2 experiment. "
            "The user authorized immediate diagnosis, fixes and reruns without confirmation. "
            "Inspect the latest campaign and logs; fix and restart failed jobs/controllers now. "
            "Keep the experiment/budgets, completed raw cells and failed-attempt provenance; "
            "use fresh run directories when required, and touch only this experiment's jobs. "
            "Prefer debug when suitable. Do not merely report the error or wait for user input. "
            "Do not install another monitor; this five-minute cron monitor already exists. "
            "If complete, validate and collect the final report and stop the scoped monitor."
        )
        result = subprocess.run(
            [codex, "queue", "--thread", thread, "--message", prompt],
            capture_output=True,
            text=True,
            check=True,
            timeout=30,
        )
        metadata = server.request("thread/read", {"threadId": thread, "includeTurns": False})
        if metadata["thread"]["status"]["type"] == "idle":
            queue = server.request("thread/queue/list", {"threadId": thread, "limit": 100})
            selected = next(
                (
                    item
                    for item in queue["data"]
                    if any(PREFIX in value.get("text", "") for value in item["input"])
                ),
                None,
            )
            if selected:
                server.request(
                    "thread/queue/start", {"threadId": thread, "queuedSubmissionId": selected["id"]}
                )
        return {"status": "queued", "delivery": result.stdout.strip()}
    finally:
        server.close()


def remove_schedule():
    current = subprocess.run(["crontab", "-l"], capture_output=True, text=True)
    if current.returncode == 0:
        remaining = [line for line in current.stdout.splitlines() if MARKER not in line]
        subprocess.run(["crontab", "-"], input="\n".join(remaining) + "\n", text=True, check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state-dir", type=Path, required=True)
    parser.add_argument("--thread", required=True)
    parser.add_argument("--clariden", default="/home/agent/.local/bin/clariden")
    parser.add_argument("--codex", default="/home/agent/.local/bin/codex")
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    args.state_dir.mkdir(parents=True, exist_ok=True)
    with (args.state_dir / "monitor.lock").open("a") as guard:
        try:
            fcntl.flock(guard, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return
        state_path = args.state_dir / "monitor-state.json"
        state = json.loads(state_path.read_text()) if state_path.exists() else {}
        if state.get("disabled"):
            return
        observed_at = datetime.now(UTC).isoformat()
        try:
            response = subprocess.run(
                [args.clariden, "exec", "--", "python3 - <<'PY'\n" + REMOTE_PROBE + "\nPY"],
                capture_output=True,
                text=True,
                check=True,
                timeout=90,
            )
            snapshot = json.loads(response.stdout)
            finding = incident(snapshot)
        except (subprocess.SubprocessError, ValueError) as error:
            snapshot = {"connection_error": str(error)}
            finding = {"kind": "monitor_connection_error", "error": str(error)}
        latest = {"observed_at": observed_at, "snapshot": snapshot, "incident": finding}
        (args.state_dir / "latest.json").write_text(json.dumps(latest, indent=2) + "\n")
        if finding and not args.check_only:
            fingerprint = hashlib.sha256(
                json.dumps([snapshot.get("campaign"), finding], sort_keys=True).encode()
            ).hexdigest()
            if (
                fingerprint != state.get("last_fingerprint")
                or time.time() - state.get("last_notified", 0) >= 1800
            ):
                latest["notification"] = notify(args.codex, args.thread, args.state_dir, finding)
                state.update(last_fingerprint=fingerprint, last_notified=time.time())
                if finding["kind"] == "complete":
                    remove_schedule()
                    state["disabled"] = True
        state["last_checked"] = observed_at
        state_path.write_text(json.dumps(state, indent=2) + "\n")
        with (args.state_dir / "checks.jsonl").open("a") as log:
            log.write(json.dumps(latest) + "\n")
        print(json.dumps(latest))


if __name__ == "__main__":
    main()
