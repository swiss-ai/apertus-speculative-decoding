#!/usr/bin/env python3
"""Allocated-node native NIXL import check for the pinned serving image."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
import platform
import sys
import time
from pathlib import Path

# Registered by this image's kv_connector/factory.py, not an inferred API name.
NIXL_CONNECTOR_MODULE = "vllm.distributed.kv_transfer.kv_connector.v1.nixl"
MODULES = ["nixl_meta_utils", "nixl", "nixl._api", "nixl_cu13", NIXL_CONNECTOR_MODULE]


def inspect_dependencies(site: Path, loader=importlib.import_module) -> dict:
    sys.path.insert(0, str(site))
    result = {
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "job_id": os.getenv("SLURM_JOB_ID"),
        "node": platform.node(),
        "python": sys.version,
        "site": str(site),
        "modules": {},
        "passed": False,
    }
    try:
        for name in MODULES:
            module = loader(name)
            path = Path(module.__file__) if getattr(module, "__file__", None) else None
            result["modules"][name] = {
                "path": str(path),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest() if path else None,
            }
            if name == NIXL_CONNECTOR_MODULE:
                if not hasattr(module, "NixlConnector"):
                    raise RuntimeError("Pinned connector module does not export NixlConnector")
                result["connector_class"] = "NixlConnector"
        result["passed"] = True
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
    return result


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--site", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    result = inspect_dependencies(args.site)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        stream.write(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
