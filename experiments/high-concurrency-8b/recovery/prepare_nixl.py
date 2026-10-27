#!/usr/bin/env python3
"""Restore the original NIXL wheels into a fresh campaign-owned prefix."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

WHEELS = {
    "nixl": "nixl-1.5.0-py3-none-any.whl",
    "nixl-cu13": "nixl_cu13-1.5.0-cp312-cp312-manylinux_2_28_aarch64.whl",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, required=True)
    args = ap.parse_args()
    root = args.root
    root.mkdir(parents=True, exist_ok=False)
    wheels = root / "wheels"
    wheels.mkdir()
    records = []
    for package, filename in WHEELS.items():
        metadata_url = f"https://pypi.org/pypi/{package}/1.5.0/json"
        with urllib.request.urlopen(metadata_url, timeout=60) as response:
            data = json.load(response)
        matches = [row for row in data["urls"] if row["filename"] == filename]
        if len(matches) != 1:
            raise RuntimeError(f"Missing unambiguous pinned wheel: {filename}")
        item = matches[0]
        dest = wheels / filename
        with (
            urllib.request.urlopen(item["url"], timeout=120) as response,
            dest.open("xb") as target,
        ):
            while chunk := response.read(1024 * 1024):
                target.write(chunk)
        digest = hashlib.sha256(dest.read_bytes()).hexdigest()
        if digest != item["digests"]["sha256"]:
            raise RuntimeError(f"Wheel hash mismatch: {filename}")
        records.append(
            {
                "package": package,
                "filename": filename,
                "metadata_url": metadata_url,
                "download_url": item["url"],
                "sha256": digest,
            }
        )
    site = root / "site"
    command = [
        sys.executable,
        "-m",
        "pip",
        "install",
        "--no-deps",
        "--only-binary=:all:",
        "--python-version",
        "3.12",
        "--platform",
        "manylinux_2_28_aarch64",
        "--implementation",
        "cp",
        "--target",
        str(site),
        *(str(wheels / name) for name in WHEELS.values()),
    ]
    subprocess.run(command, check=True)
    required = [
        "nixl/__init__.py",
        "nixl/_api.py",
        "nixl_meta_utils.py",
        "nixl_cu13/_bindings.cpython-312-aarch64-linux-gnu.so",
    ]
    if missing := [name for name in required if not (site / name).is_file()]:
        raise RuntimeError(f"Incomplete restored NIXL files: {missing}")
    manifest = [
        {
            "path": str(p.relative_to(site)),
            "sha256": hashlib.sha256(p.read_bytes()).hexdigest(),
            "bytes": p.stat().st_size,
        }
        for p in sorted(site.rglob("*"))
        if p.is_file()
    ]
    result = {
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "version": "1.5.0",
        "wheel_records": records,
        "install_command": command,
        "site": str(site),
        "files": manifest,
        "claim_limit": (
            "File repair only; native imports/connector startup require allocated-node preflight."
        ),
    }
    (root / "manifest.json").write_text(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps(
            {"site": str(site), "files": len(manifest), "manifest": str(root / "manifest.json")}
        )
    )


if __name__ == "__main__":
    main()
