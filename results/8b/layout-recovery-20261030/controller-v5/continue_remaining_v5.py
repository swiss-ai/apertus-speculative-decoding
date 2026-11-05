#!/usr/bin/env python3
"""Original continuation with private scratch runtime and FlashInfer workspace."""

from __future__ import annotations

import sys
from pathlib import Path

import continue_remaining as original

from apertus_bench import layout_recovery as recovery


def scratch_launcher(real_run, runtime_root: Path):
    def run(command, *args, **kwargs):
        env = kwargs.get("env")
        if env is not None and str(command[0]) == str(recovery.REPO / "serving/node-sweep.sh"):
            root = runtime_root / Path(env["SWEEP_OUTPUT"]).name
            env["VLLM_PATCH_DIR"] = str(root / "vllm-patch")
            env["SML_ENV_DIR"] = str(root / "sml-env")
            extras = [
                part
                for part in env.get("EXTRA_ENV", "").split(";")
                if part and not part.startswith(("VLLM_CONFIG_ROOT=", "FLASHINFER_WORKSPACE_BASE="))
            ]
            extras.append("VLLM_CONFIG_ROOT=" + str(root / "vllm-config"))
            extras.append("FLASHINFER_WORKSPACE_BASE=" + str(root / "flashinfer-workspace"))
            env["EXTRA_ENV"] = ";".join(extras)
        return real_run(command, *args, **kwargs)

    return run


def main():
    root = Path(sys.argv[sys.argv.index("--root") + 1])
    recovery.subprocess.run = scratch_launcher(recovery.subprocess.run, root / "runtime")
    original.main()


if __name__ == "__main__":
    main()
