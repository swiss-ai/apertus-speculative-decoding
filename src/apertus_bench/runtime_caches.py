"""Private runtime caches for owned GPU deployments; never changes HOME or kernels."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path
from typing import Any


def cache_environment(root: Path) -> dict[str, str]:
    return {
        key: str(root / directory)
        for key, directory in {
            "XDG_CACHE_HOME": "xdg-cache",
            "XDG_CONFIG_HOME": "xdg-config",
            "HF_HOME": "hf",
            "TORCH_HOME": "torch",
            "TRITON_CACHE_DIR": "triton",
            "TORCHINDUCTOR_CACHE_DIR": "inductor",
            "CUDA_CACHE_PATH": "cuda",
            "VLLM_CACHE_ROOT": "vllm",
            "VLLM_CONFIG_ROOT": "vllm-config",
            # Pinned FlashInfer appends .cache/flashinfer/version/arch to this base.
            "FLASHINFER_WORKSPACE_BASE": "flashinfer-base",
        }.items()
    }


def prepare(root: Path) -> dict[str, Any]:
    environment = cache_environment(root)
    for directory in environment.values():
        path = Path(directory)
        path.mkdir(parents=True, exist_ok=True)
        # Catch quota/permissions before loading a target model.
        with tempfile.TemporaryFile(dir=path) as handle:
            handle.write(b"owned runtime cache write probe\n")
            handle.flush()
    return {
        "environment": environment,
        "directory_write_probes_passed": True,
        "home_unchanged": os.environ.get("HOME"),
    }


def probe_image_runtime(root: Path) -> dict[str, str]:
    from filelock import FileLock
    from flashinfer.jit import env as flash_env
    from vllm import envs as vllm_env

    jit = Path(flash_env.FLASHINFER_JIT_DIR)
    config = Path(vllm_env.VLLM_CONFIG_ROOT)
    for path in (jit, config):
        if not path.is_relative_to(root):
            raise ValueError(f"Runtime cache escaped the owned root: {path}")
        path.mkdir(parents=True, exist_ok=True)
    lock = jit / "tmp" / "sampling.lock"
    lock.parent.mkdir(parents=True, exist_ok=True)
    with FileLock(str(lock), timeout=10):
        pass  # Exercise the exact lock path that failed with HOME EDQUOT.
    with (config / "runtime-write-probe").open("x") as handle:
        handle.write("owned vLLM config is writable\n")
    return {
        "flashinfer_jit_dir": str(jit),
        "sampling_lock": str(lock),
        "vllm_config_root": str(config),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--probe-flashinfer", action="store_true")
    parser.add_argument("--exec", dest="command", nargs=argparse.REMAINDER, required=True)
    args = parser.parse_args()
    report = prepare(args.root)
    os.environ.update(report["environment"])
    if args.probe_flashinfer:
        report["image_runtime"] = probe_image_runtime(args.root)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    with args.report.open("x") as handle:
        json.dump(report, handle, indent=2)
        handle.write("\n")
    print(json.dumps({"runtime_cache_preflight": str(args.report), **report}), flush=True)
    os.environ["T5_RUNTIME_CACHE_REPORT"] = str(args.report)
    if not args.command:
        parser.error("--exec requires a command")
    os.execvpe(args.command[0], args.command, os.environ)


if __name__ == "__main__":
    main()
