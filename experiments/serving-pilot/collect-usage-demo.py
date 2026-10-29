"""Fetch the completed demo for local review; never launch or cancel a job."""

from __future__ import annotations

import argparse
import hashlib
import json
import shlex
import shutil
import subprocess
import tarfile
import tempfile
import time
from pathlib import Path


def install_archive(archive: Path, output: Path) -> None:
    with tempfile.TemporaryDirectory(prefix="usage-demo-review-") as temporary:
        stage = Path(temporary)
        with tarfile.open(archive) as source:
            for member in source.getmembers():
                target = stage / member.name
                if not target.resolve().is_relative_to(stage) or not (
                    member.isfile() or member.isdir()
                ):
                    raise ValueError("unsafe result archive member")
                if member.isdir():
                    target.mkdir(parents=True, exist_ok=True)
                else:
                    target.parent.mkdir(parents=True, exist_ok=True)
                    with source.extractfile(member) as reader, target.open("xb") as writer:
                        shutil.copyfileobj(reader, writer)
        tree = stage / output.name
        manifest = json.loads((tree / "artifact-manifest.json").read_text())
        for name, digest in manifest.items():
            path = tree / name
            if not path.resolve().is_relative_to(tree):
                raise ValueError("unsafe artifact manifest path")
            if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
                raise ValueError(f"artifact hash mismatch: {name}")
        if any((output / p.name).exists() for p in tree.iterdir()):
            raise FileExistsError("local artifacts already exist; refusing to overwrite")
        shutil.copytree(tree, output, dirs_exist_ok=True)
        if (output / "report.md").exists():
            text = "The demo has completed. Read [the full answers and timings](report.md).\n"
        else:
            text = "The demo stopped before completion. See [failure details](failure.json).\n"
        (output / "README.md").write_text(text)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--remote-repo", required=True)
    parser.add_argument("--remote-output", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    archive = args.remote_output + ".tar.gz"
    deadline = time.monotonic() + 129600
    while time.monotonic() < deadline:
        result = subprocess.run(
            [
                "clariden",
                "--repo",
                args.remote_repo,
                "exec",
                "--",
                "test -f " + shlex.quote(archive),
            ],
            capture_output=True,
            timeout=60,
        )
        if result.returncode == 0:
            with tempfile.TemporaryDirectory(prefix="usage-demo-download-") as temporary:
                local_archive = Path(temporary) / "demo.tar.gz"
                subprocess.run(
                    ["clariden", "get", archive, str(local_archive)], check=True, timeout=120
                )
                install_archive(local_archive, args.output.resolve())
            print(args.output / "report.md", flush=True)
            return
        if result.returncode != 1:
            raise RuntimeError("Clariden access failed; results remain on the cluster")
        time.sleep(60)
    raise TimeoutError("results not ready within 36 hours; check the remote demo controller")


if __name__ == "__main__":
    main()
