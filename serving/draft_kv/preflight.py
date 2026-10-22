"""Compute-node-only source/GPU audit for the drafter KV campaign."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
import os
import subprocess
from datetime import UTC, datetime
from pathlib import Path


def command(argv: list[str]) -> dict:
    result = subprocess.run(argv, capture_output=True, text=True, check=False)
    return {"argv": argv, "returncode": result.returncode,
            "stdout": result.stdout, "stderr": result.stderr}


def main() -> int:
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--source-manifest", required=True, type=Path)
    parser.add_argument("--gpu-check", action="store_true")
    args = parser.parse_args()
    if not os.environ.get("SLURM_JOB_ID"):
        parser.error("run source/GPU preflight inside a Slurm allocation")
    args.output.mkdir(parents=True, exist_ok=False)
    expected = json.loads(args.source_manifest.read_text())
    spec = importlib.util.find_spec("vllm")
    if spec is None or spec.origin is None:
        parser.error("vLLM package not found in this environment")
    root = Path(spec.origin).resolve().parent.parent
    source = {}
    for relative, digest in expected["files"].items():
        path = root / relative
        raw = path.read_bytes() if path.exists() else None
        actual = hashlib.sha256(raw).hexdigest() if raw is not None else None
        source[relative] = {"path": str(path), "sha256": actual,
                            "expected_sha256": digest, "matches_pin": actual == digest}
        if raw is not None:
            saved = args.output / "source" / relative
            saved.parent.mkdir(parents=True, exist_ok=True)
            saved.write_bytes(raw)
    packages = {}
    for name in ("torch", "triton", "vllm", "flash-attn", "numpy"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    report = {
        "timestamp": datetime.now(UTC).isoformat(),
        "job_id": os.environ["SLURM_JOB_ID"],
        "node": os.environ.get("SLURMD_NODENAME"),
        "node_list": os.environ.get("SLURM_JOB_NODELIST"),
        "packages": packages, "expected_revision": expected["revision"],
        "source": source,
        "git": command(["git", "-C", str(root), "rev-parse", "HEAD"]),
        "gpu": command(["nvidia-smi", "--query-gpu=index,name,uuid,memory.total,driver_version",
                        "--format=csv,noheader"]),
    }
    if args.gpu_check:
        check = Path(__file__).with_name("check_gpu.py")
        report["gpu_check"] = command([os.sys.executable, str(check)])
    (args.output / "audit.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    passed = all(row["matches_pin"] for row in source.values())
    passed = passed and report["gpu"]["returncode"] == 0
    if args.gpu_check:
        passed = passed and report["gpu_check"]["returncode"] == 0
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
