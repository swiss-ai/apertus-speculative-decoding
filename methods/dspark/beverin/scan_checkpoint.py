#!/usr/bin/env python3
"""Full finiteness scan of one saved checkpoint. Run inside the compute container."""
import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from checkpoint_ring import verify_directory


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--hash", action="store_true")
    args = parser.parse_args()
    started = time.time()
    finite = verify_directory(args.checkpoint, torch, 0, 1)
    report = {
        "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "checkpoint": str(args.checkpoint),
        "finite": bool(finite),
        "seconds": round(time.time() - started, 3),
    }
    if args.hash:
        report["sha256"] = {
            name: sha256(args.checkpoint / name)
            for name in ("model.safetensors", "optimizer_state_dict.pt", "config.json", "training_state.json")
            if (args.checkpoint / name).is_file()
        }
    marker = args.checkpoint / "verified-finite.json"
    if marker.is_file():
        report["verified_finite_marker"] = json.loads(marker.read_text())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print("FINITENESS_SCAN " + json.dumps({"finite": report["finite"], "output": str(args.output)}))
    if not finite:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
