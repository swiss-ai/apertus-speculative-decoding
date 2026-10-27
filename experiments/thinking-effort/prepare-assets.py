"""Download the pinned head and inspect the GPU runtime inside a Slurm allocation."""

import argparse
import importlib.metadata
import json
import os
import platform
import subprocess
from datetime import UTC, datetime
from pathlib import Path

import yaml

parser = argparse.ArgumentParser()
parser.add_argument("spec", type=Path)
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()
args.output.mkdir(parents=True, exist_ok=False)
spec = yaml.safe_load(args.spec.read_text())
report = {
    "started_at": datetime.now(UTC).isoformat(),
    "python": platform.python_version(),
    "machine": platform.machine(),
    "node": platform.node(),
    "slurm_job_id": os.getenv("SLURM_JOB_ID"),
    "packages": {},
}
try:
    for name in ("torch", "vllm", "transformers", "huggingface-hub", "triton"):
        report["packages"][name] = importlib.metadata.version(name)
    import torch
    from huggingface_hub import snapshot_download

    report["cuda"] = torch.version.cuda
    report["gpu"] = torch.cuda.get_device_name(0)
    report["capability"] = list(torch.cuda.get_device_capability(0))
    report["drafter_revision"] = spec["drafter_revision"]
    snapshot_download(
        spec["drafter_model"],
        revision=spec["drafter_revision"],
        local_dir=spec["drafter_path"],
        allow_patterns=["*.json", "*.py", "*.safetensors", "README.md"],
    )
    command = ["python3", "-m", "vllm.entrypoints.openai.api_server", "--help"]
    result = subprocess.run(command, capture_output=True, text=True, timeout=180)
    (args.output / "vllm-help.txt").write_text(result.stdout + result.stderr)
    report["vllm_help_exit_code"] = result.returncode
    if result.returncode:
        raise RuntimeError("vLLM CLI failed; inspect vllm-help.txt")
    report["status"] = "ready"
except Exception as error:
    report["status"] = "failed"
    report["error"] = f"{type(error).__name__}: {error}"
    raise
finally:
    report["ended_at"] = datetime.now(UTC).isoformat()
    (args.output / "runtime.json").write_text(json.dumps(report, indent=2) + "\n")
