#!/usr/bin/env python3
"""Apply this explicit repair only after the original controller has stopped."""

import argparse
import datetime as dt
import hashlib
import json
import subprocess
from pathlib import Path


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", type=Path, required=True)
    ap.add_argument("--root", type=Path, required=True)
    a = ap.parse_args()
    freeze = json.loads((a.root / "source-freeze-v2.json").read_text())
    prior = Path(freeze["supersedes_state_path"])
    old = json.loads(prior.read_text())
    if (
        old["campaign_state"] != "needs_attention"
        or old["stages"][-1]["slurm"]["state"] != "FAILED"
    ):
        raise RuntimeError("Prior campaign is not the expected stopped failed attempt")
    if digest(prior) != freeze["supersedes_state_sha256"]:
        raise RuntimeError("Prior state changed")
    oldroot = str(prior.parent)
    for proc in Path("/proc").glob("[0-9]*/cmdline"):
        try:
            args = proc.read_bytes().decode().split("\0")
        except (OSError, UnicodeError):
            continue
        if oldroot in args and "apertus_bench.layout_recovery" in args and "controller" in args:
            raise RuntimeError("Prior controller is still alive")
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=a.repo, text=True).strip()
    if head != freeze["source_base_commit"]:
        raise RuntimeError("Unexpected remote HEAD")
    dirty = subprocess.check_output(
        ["git", "diff", "--name-only", "HEAD"], cwd=a.repo, text=True
    ).strip()
    if dirty:
        raise RuntimeError("Remote tracked files already changed; inspect before applying")
    before = {
        "commit": head,
        "files": {name: digest(a.repo / name) for name in old["source_identity"]["files"]},
    }
    if before != old["source_identity"]:
        raise RuntimeError("Original source hashes changed")
    patch = a.root / "source-overlay-v2.patch"
    if digest(patch) != freeze["source_overlay_sha256"]:
        raise RuntimeError("Patch hash mismatch")
    subprocess.run(["git", "apply", "--check", str(patch)], cwd=a.repo, check=True)
    subprocess.run(["git", "apply", str(patch)], cwd=a.repo, check=True)
    expected = json.loads((a.root / "expected-source-identity-v2.json").read_text())
    observed = {
        "commit": head,
        "files": {name: digest(a.repo / name) for name in expected["files"]},
    }
    if expected != observed:
        raise RuntimeError("Patched source identity mismatch; do not launch")
    record = {
        "applied_at": dt.datetime.now(dt.UTC).isoformat(),
        "source_base_commit": head,
        "source_overlay_sha256": digest(patch),
        "local_repair_commit": freeze["local_repair_commit"],
        "prior_state_path": str(prior),
        "prior_state_sha256": digest(prior),
        "verified_source_files": len(observed["files"]),
    }
    for name, value in [
        ("observed-source-identity-v2.json", observed),
        ("source-overlay-applied-v2.json", record),
    ]:
        with (a.root / name).open("x") as stream:
            json.dump(value, stream, indent=2)
            stream.write("\n")
    print(json.dumps(record))


if __name__ == "__main__":
    main()
