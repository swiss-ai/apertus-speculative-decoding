"""Download the pinned head and inspect the GPU runtime inside a Slurm allocation."""

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import urllib.request
from datetime import UTC, datetime
from pathlib import Path

import yaml

from apertus_bench.effort import stage_harmony_vocab

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
    for name in ("torch", "vllm", "transformers", "huggingface-hub", "triton", "openai-harmony"):
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
    vocab = Path(spec["harmony_vocab_path"])
    if not vocab.exists():
        with urllib.request.urlopen(spec["harmony_vocab_url"], timeout=60) as response:
            payload = response.read()
        if hashlib.sha256(payload).hexdigest() != spec["harmony_vocab_sha256"]:
            raise ValueError("downloaded Harmony vocabulary failed its published hash check")
        with vocab.open("xb") as sink:
            sink.write(payload)
    report["harmony_vocab"] = stage_harmony_vocab(spec, args.output / "harmony-cache")
    os.environ["TIKTOKEN_RS_CACHE_DIR"] = str(args.output / "harmony-cache")
    from openai_harmony import HarmonyEncodingName, load_harmony_encoding

    load_harmony_encoding(HarmonyEncodingName.HARMONY_GPT_OSS)
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
