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


def freeze(repo: Path, root: Path, trace_job: str) -> dict:
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
    if previous.get("state") != "waiting" or previous.get("current_job") != trace_job:
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
    launcher = receipts / ("start-controller-" + revision[:12] + ".sh")
    script = f"""#!/bin/bash
set -euo pipefail
cd {snapshot}
export PYTHONPATH={snapshot}/methods/eagle:{snapshot}/src
export PATH=/users/faruk_zahiragic/venvs/sml-apertus/bin:$PATH
export PYTHONDONTWRITEBYTECODE=1
exec python3 -u -m apertus_eagle.adaptation_recovery \\
  --repo {snapshot} \\
  --campaign-root {root} \\
  --trace-job {trace_job} --run-id recovery-20261027 --poll-seconds 120
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
    parser.add_argument("--trace-job", required=True)
    args = parser.parse_args()
    print(json.dumps(freeze(args.repo, args.campaign_root, args.trace_job), indent=2))


if __name__ == "__main__":
    main()
