#!/usr/bin/env python3
"""Allocated-node native NIXL import check for this recovery campaign."""

import argparse
import hashlib
import importlib
import json
import os
import platform
import sys
import time
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--site", type=Path, required=True)
ap.add_argument("--output", type=Path, required=True)
a = ap.parse_args()
sys.path.insert(0, str(a.site))
modules = {}
for name in [
    "nixl_meta_utils",
    "nixl",
    "nixl._api",
    "nixl_cu13",
    "vllm.distributed.kv_transfer.kv_connector.v1.nixl_connector",
]:
    m = importlib.import_module(name)
    path = Path(m.__file__) if getattr(m, "__file__", None) else None
    modules[name] = {
        "path": str(path),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest() if path else None,
    }
result = {
    "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    "job_id": os.getenv("SLURM_JOB_ID"),
    "node": platform.node(),
    "python": sys.version,
    "site": str(a.site),
    "modules": modules,
    "passed": True,
}
a.output.parent.mkdir(parents=True, exist_ok=True)
with a.output.open("x") as f:
    f.write(json.dumps(result, indent=2) + "\n")
print(json.dumps(result))
