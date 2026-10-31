"""Freeze committed T5 code and replace only its owned waiting controller.

Run on the cluster login host after serialized bundle sync. This copies no
weights, corpora or result trees. Detached sparse worktrees pin Python/launch
sources while all inputs and mutable bootstrap dependencies stay isolated.
"""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import signal
import subprocess
import time
from pathlib import Path

from apertus_eagle.adaptation_decision import digest, immutable_json, read
from apertus_eagle.adaptation_recovery import write_state


def freeze(
    repo: Path,
    root: Path,
    trace_job: str | None,
    *,
    source_only: bool = False,
    run_id: str = "recovery-20261027",
    confirmation_only: bool = False,
    confirmation_attempt: str | None = None,
    confirmation_wall_time: str | None = None,
    repair_confirmation_job: str | None = None,
) -> dict:
    receipts = root / "receipts"
    receipts.mkdir(parents=True, exist_ok=True)
    replacement_lock = (receipts / "controller-replacement.lock").open("a")
    fcntl.flock(replacement_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    revision = subprocess.check_output(
        ["git", "-C", str(repo), "rev-parse", "HEAD"], text=True
    ).strip()
    if subprocess.check_output(
        ["git", "-C", str(repo), "diff", "HEAD", "--name-only"], text=True
    ).strip():
        raise RuntimeError("Commit owned changes before freezing controller code")
    snapshot = root / "runtime" / ("source-" + revision[:12])
    if not snapshot.exists():
        subprocess.run(
            [
                "git",
                "-C",
                str(repo),
                "worktree",
                "add",
                "--detach",
                "--no-checkout",
                str(snapshot),
                revision,
            ],
            check=True,
        )
        subprocess.run(
            [
                "git",
                "-C",
                str(snapshot),
                "sparse-checkout",
                "set",
                "--no-cone",
                "/methods/eagle/",
                "/src/",
                "/serving/",
                "/targets/",
                "/tools/",
                "/pyproject.toml",
                "/README.md",
                "/results/README.md",
                "/docs/protocol.md",
            ],
            check=True,
        )
        subprocess.run(["git", "-C", str(snapshot), "read-tree", "-mu", "HEAD"], check=True)
        (snapshot / "scratch").symlink_to(root / "runtime/scratch", target_is_directory=True)
        (snapshot / "workloads").symlink_to(repo / "workloads", target_is_directory=True)
        files = subprocess.check_output(
            ["git", "-C", str(snapshot), "ls-files"], text=True
        ).splitlines()
        manifest = {
            "revision": revision,
            "task_branch": subprocess.check_output(
                ["git", "-C", str(repo), "branch", "--show-current"], text=True
            ).strip(),
            "files": {f: digest(snapshot / f) for f in files if (snapshot / f).is_file()},
        }
        immutable_json(snapshot / "source-manifest.json", manifest)
        for folder in ("methods", "src", "serving", "targets", "tools"):
            for current, dirs, names in os.walk(snapshot / folder, followlinks=False):
                for name in names:
                    path = Path(current) / name
                    if not path.is_symlink():
                        path.chmod(path.stat().st_mode & ~0o222)
                for name in dirs:
                    path = Path(current) / name
                    if not path.is_symlink():
                        path.chmod(path.stat().st_mode & ~0o222)
            (snapshot / folder).chmod((snapshot / folder).stat().st_mode & ~0o222)
        (snapshot / "source-manifest.json").chmod(0o444)
    manifest = read(snapshot / "source-manifest.json")
    if manifest["revision"] != revision:
        raise RuntimeError("Snapshot revision differs")
    for name, sha in manifest["files"].items():
        if digest(snapshot / name) != sha:
            raise RuntimeError(f"Frozen source differs: {name}")
    immutable_json(
        root / "preregistration.json",
        read(snapshot / "methods/eagle/configs/8b/adapt-recovery-policy.json"),
    )
    if source_only:
        return {
            "revision": revision,
            "source": str(snapshot),
            "source_manifest": str(snapshot / "source-manifest.json"),
            "source_manifest_sha256": digest(snapshot / "source-manifest.json"),
            "policy_sha256": digest(root / "preregistration.json"),
            "task_branch": manifest.get("task_branch"),
        }
    if not trace_job:
        raise ValueError("A trace job is required unless freezing sources only")
    observed = subprocess.check_output(
        ["sacct", "-X", "-n", "-P", "-j", trace_job, "--format=JobIDRaw,State,ExitCode"], text=True
    )
    registration = receipts / ("freeze-" + revision[:12] + ".json")
    if not registration.exists():
        immutable_json(
            registration,
            {
                "revision": revision,
                "source": str(snapshot),
                "manifest_sha256": digest(snapshot / "source-manifest.json"),
                "policy_sha256": digest(root / "preregistration.json"),
                "registered_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "trace_observed": observed,
            },
        )
    previous = read(receipts / "campaign.json")
    repaired_attempts = []
    if confirmation_only:
        if previous.get("state") != "needs_repair":
            raise RuntimeError("Confirmation repair requires a stopped needs_repair controller")
        if (
            not (root / "adaptation-matrix-v1.json").is_file()
            or not (root / "selection.json").is_file()
        ):
            raise RuntimeError("Completed curves and frozen selection are required")
        if any(batch.get("active_job") for batch in previous.get("batches", {}).values()):
            raise RuntimeError("A training batch is still active")
        for name, entry in list(previous.get("confirmations", {}).items()):
            if entry.get("job_id") and entry.get("state") != "complete":
                job = entry["job_id"]
                if job != repair_confirmation_job:
                    raise RuntimeError(
                        "Inspect an already submitted serving job before controller repair"
                    )
                observed_failure = subprocess.check_output(
                    ["sacct", "-X", "-n", "-P", "-j", job,
                     "--format=JobIDRaw,State,ExitCode"], text=True
                )
                status = next(
                    row.split("|")[1].split()[0].rstrip("+")
                    for row in observed_failure.splitlines() if row.startswith(job + "|")
                )
                failure = Path(entry["report"]).parent / "failure.json"
                if status not in {"FAILED", "TIMEOUT", "CANCELLED", "NODE_FAIL", "OUT_OF_MEMORY"}:
                    raise RuntimeError(
                        "Only an inspected terminal failed confirmation can be repaired"
                    )
                if not failure.is_file():
                    raise RuntimeError("Preserved confirmation failure report is required")
                repaired_attempts.append({
                    "name": name, **entry, "observed_slurm": observed_failure,
                    "failure": read(failure), "failure_path": str(failure),
                    "previous_controller_source": previous["source_manifest"],
                })
                del previous["confirmations"][name]
        if repair_confirmation_job and not repaired_attempts:
            raise RuntimeError("Explicit failed confirmation is not present in campaign receipt")
    elif previous.get("state") != "waiting" or previous.get("current_job") != trace_job:
        raise RuntimeError("Replace only our trace-waiting controller; inspect current stage first")
    old_pid = previous["pid"]
    process = Path(f"/proc/{old_pid}/cmdline")
    if process.exists():
        cmd = process.read_bytes().replace(b"\0", b" ").decode()
        if "apertus_eagle.adaptation_recovery" not in cmd or str(root) not in cmd:
            raise RuntimeError("PID reuse or unexpected controller identity; refusing signal")
        os.kill(old_pid, signal.SIGTERM)
        for _ in range(100):
            state = subprocess.run(
                ["ps", "-p", str(old_pid), "-o", "stat="], capture_output=True, text=True
            )
            if state.returncode or state.stdout.strip().startswith("Z"):
                break
            time.sleep(0.1)
        else:
            raise RuntimeError("Owned controller did not stop")
    controller_lock = (receipts / "controller.lock").open("a")
    fcntl.flock(controller_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    controller_lock.close()
    for entry in repaired_attempts:
        immutable_json(receipts / f"failed-confirmation-{entry['job_id']}.json", entry)
        previous.setdefault("failed_confirmation_attempts", []).append(entry)
    if repaired_attempts:
        write_state(receipts / "campaign.json", previous)
    launcher = receipts / ("start-controller-" + revision[:12] + ".sh")
    controller_tmp = root / "submission-tmp" / ("controller-" + revision[:12])
    controller_tmp.mkdir(parents=True, exist_ok=True)
    extra = ""
    if confirmation_only:
        if (
            not confirmation_attempt
            or not confirmation_attempt.replace("-", "").replace("_", "").isalnum()
        ):
            raise ValueError("Confirmation repair needs a unique simple attempt name")
        extra = f" --confirmation-only --confirmation-attempt {confirmation_attempt}"
    if confirmation_wall_time:
        import re

        if not re.fullmatch(r"\d+:[0-5]\d:[0-5]\d", confirmation_wall_time):
            raise ValueError("Confirmation wall time must be HH:MM:SS")
        extra += f" --confirmation-wall-time {confirmation_wall_time}"
    script = f"""#!/bin/bash
set -euo pipefail
cd {snapshot}
export PYTHONPATH={snapshot}/methods/eagle:{snapshot}/src
export PATH=/users/faruk_zahiragic/venvs/sml-apertus/bin:$PATH
export PYTHONDONTWRITEBYTECODE=1
export TMPDIR={controller_tmp}
exec python3 -u -m apertus_eagle.adaptation_recovery \\
  --repo {snapshot} \\
  --campaign-root {root} \\
  --trace-job {trace_job} --run-id {run_id} --poll-seconds 120{extra}
"""
    if launcher.exists():
        if launcher.read_text() != script:
            raise RuntimeError("Refusing to change a frozen controller launcher")
    else:
        with launcher.open("x") as handle:
            handle.write(script)
    launcher.chmod(0o555)
    log = (receipts / "controller.log").open("ab")
    child = subprocess.Popen(
        ["bash", str(launcher)],
        stdin=subprocess.DEVNULL,
        stdout=log,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    (receipts / "controller.pid").write_text(str(child.pid) + "\n")
    time.sleep(2)
    if child.poll() is not None:
        raise RuntimeError(f"Frozen controller exited: {child.returncode}; inspect controller.log")
    identity = Path(f"/proc/{child.pid}/cmdline").read_bytes().replace(b"\0", b" ").decode()
    result = {
        "revision": revision,
        "source": str(snapshot),
        "source_manifest": str(snapshot / "source-manifest.json"),
        "source_manifest_sha256": digest(snapshot / "source-manifest.json"),
        "launcher": str(launcher),
        "pid": child.pid,
        "process_identity": identity,
        "previous_pid": old_pid,
        "state": read(receipts / "campaign.json"),
    }
    immutable_json(receipts / (f"controller-replacement-{revision[:12]}-{child.pid}.json"), result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--campaign-root", type=Path, required=True)
    parser.add_argument("--trace-job")
    parser.add_argument("--source-only", action="store_true")
    parser.add_argument("--run-id", default="recovery-20261027")
    parser.add_argument("--confirmation-only", action="store_true")
    parser.add_argument("--confirmation-attempt")
    parser.add_argument("--confirmation-wall-time")
    parser.add_argument("--repair-confirmation-job")
    args = parser.parse_args()
    print(
        json.dumps(
            freeze(
                args.repo,
                args.campaign_root,
                args.trace_job,
                source_only=args.source_only,
                run_id=args.run_id,
                confirmation_only=args.confirmation_only,
                confirmation_attempt=args.confirmation_attempt,
                confirmation_wall_time=args.confirmation_wall_time,
                repair_confirmation_job=args.repair_confirmation_job,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
