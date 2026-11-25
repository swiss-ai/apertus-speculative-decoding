#!/usr/bin/env python3
"""Choose the ALPS math library from the Stage A muon timings."""
import argparse
import json
from pathlib import Path


def _load(directory):
    path = Path(directory) / "rank0.json"
    if not path.is_file():
        return {"status": "missing", "timing": {}}
    return json.loads(path.read_text())


def choose(default_dir, cublas_dir):
    default, cublas = _load(default_dir), _load(cublas_dir)
    default_s = (default.get("timing") or {}).get("median_s")
    cublas_s = (cublas.get("timing") or {}).get("median_s")
    default_ok = default.get("status") == "passed" and default_s
    cublas_ok = cublas.get("status") == "passed" and cublas_s
    if default_ok and cublas_ok and default_s <= 1.05 * cublas_s:
        backend, reason = "default", "default math library is finite and within 5% of rocBLAS"
    elif cublas_ok:
        backend, reason = "cublas", "keeping rocBLAS; the default library was not both finite and within 5%"
    else:
        backend, reason = None, "neither math library passed the tall Muon check"
    return {
        "backend": backend,
        "reason": reason,
        "default_status": default.get("status"),
        "cublas_status": cublas.get("status"),
        "default_median_s": default_s,
        "cublas_median_s": cublas_s,
        "adopt_default": backend == "default",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--default", type=Path, required=True)
    parser.add_argument("--cublas", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    decision = choose(args.default, args.cublas)
    args.output.write_text(json.dumps(decision, indent=2) + "\n")
    print("BLAS_DECISION " + json.dumps(decision))
    if decision["backend"] is None:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
