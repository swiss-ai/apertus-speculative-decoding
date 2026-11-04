"""Capture each recovery training deployment inside its allocated container."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
import tomllib
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--deployment", type=Path, required=True)
    args = parser.parse_args()
    env_file = args.deployment / "environment.toml"
    for _ in range(30):
        if env_file.exists():
            break
        time.sleep(1)
    resolved = tomllib.loads(env_file.read_text())
    image = Path(resolved["image"])
    output = args.deployment / ("node" + os.environ.get("SLURM_NODEID", "0"))
    output.mkdir(parents=True, exist_ok=False)
    sha = hashlib.sha256()
    with image.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            sha.update(block)
    (output / "image.sha256").write_text(f"{sha.hexdigest()}  {image}\n")
    shutil.copy2(args.repo / "targets/8b/contract.json", output / "target-contract.json")
    if (args.repo / "source-manifest.json").exists():
        shutil.copy2(args.repo / "source-manifest.json", output / "source-manifest.json")
    if os.environ.get("T5_RUNTIME_CACHE_REPORT"):
        shutil.copy2(os.environ["T5_RUNTIME_CACHE_REPORT"], output / "runtime-cache-preflight.json")
    for name, argv in {
        "git-revision.txt": ["git", "-C", str(args.repo), "rev-parse", "HEAD"],
        "checkout.patch": ["git", "-C", str(args.repo), "diff", "HEAD", "--binary"],
        "torchspec-revision.txt": ["git", "-C", os.environ["TORCHSPEC_ROOT"], "rev-parse", "HEAD"],
        "torchspec.patch": ["git", "-C", os.environ["TORCHSPEC_ROOT"], "diff", "HEAD", "--binary"],
    }.items():
        (output / name).write_bytes(subprocess.check_output(argv))
    selected = [
        "SLURM_JOB_ID",
        "SLURM_JOB_NODELIST",
        "SLURMD_NODENAME",
        "CUDA_VISIBLE_DEVICES",
        "EAGLE_TRAIN_CONFIG",
        "EAGLE_RUN_ID",
        "EAGLE_STEPS",
        "TORCHSPEC_ROOT",
        "EAGLE_PYDEPS",
        "FLASHINFER_WORKSPACE_BASE",
        "VLLM_CONFIG_ROOT",
        "VLLM_CACHE_ROOT",
        "XDG_CACHE_HOME",
        "XDG_CONFIG_HOME",
        "HF_HOME",
        "TRITON_CACHE_DIR",
        "TORCHINDUCTOR_CACHE_DIR",
        "CUDA_CACHE_PATH",
        "TORCH_HOME",
    ]
    (output / "runtime.json").write_text(
        json.dumps(
            {
                "python": sys.version,
                "environment": {key: os.environ[key] for key in selected if key in os.environ},
                "overlay_mounts": resolved.get("mounts", []),
                "mount_entries": len(resolved.get("mounts", [])),
                "mount_specs_comma_joined_utf8_bytes": len(
                    ",".join(resolved.get("mounts", [])).encode()
                ),
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
