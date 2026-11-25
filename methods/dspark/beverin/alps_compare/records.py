"""Provenance and summaries for the unchanged job-694440 DTensor probes.

Host-side subcommands support Beverin's Python 3.6. Diagnose runs in the image.
"""

import datetime
import hashlib
import json
import os
import re
import statistics
import subprocess
import sys
from pathlib import Path


def write(path, data):
    Path(path).write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def selected_environment():
    exact = {
        "PATH",
        "LD_LIBRARY_PATH",
        "BASH_ENV",
        "CUDA_VISIBLE_DEVICES",
        "ROCR_VISIBLE_DEVICES",
        "GPU_DEVICE_ORDINAL",
        "OMP_NUM_THREADS",
        "PYTHONNOUSERSITE",
        "PYTHONPATH",
        "ALPS_EXTENDED_IMAGES_ENV_SOURCED",
        "SLURM_JOB_ID",
        "SLURM_JOB_NODELIST",
        "SLURM_CPUS_ON_NODE",
    }
    prefixes = ("NCCL_", "RCCL_", "FI_", "HSA_", "HIP_", "ROCM_", "TORCH_BLAS_", "OPENBLAS_")
    return {
        key: value for key, value in os.environ.items() if key in exact or key.startswith(prefixes)
    }


def prepare(deployment, repo, alps, upstream):
    identity = json.loads(
        (repo / "methods/dspark/beverin/alps_compare/image/identity.json").read_text()
    )
    revision = (deployment / "git-revision.txt").read_text().strip()
    branch = (deployment / "git-branch.txt").read_text().strip()
    for name in ("upstream-default", "upstream-rocblas", "alps-default", "alps-rocblas"):
        cell = deployment / "cells" / name
        cell.mkdir(parents=True)
        is_alps = name.startswith("alps-")
        image = alps if is_alps else upstream
        config = [
            "image = " + json.dumps(str(image)),
            'mounts = ["/capstor", "/iopsstor", "/users"]',
            "workdir = " + json.dumps(str(deployment)),
            "entrypoint = false",
            "",
            "[env]",
            'PYTHONNOUSERSITE = "1"',
            'HWLOC_COMPONENTS = "-gl"',
            "",
            "[annotations]",
        ]
        if is_alps:
            config.extend(
                [
                    'com.hooks.netstack.source = "host"',
                    'com.hooks.cxi.enabled = "false"',
                    'com.hooks.aws_ofi_nccl.enabled = "false"',
                ]
            )
        else:
            config.extend(
                [
                    'com.hooks.aws_ofi_nccl.enabled = "true"',
                    'com.hooks.aws_ofi_nccl.variant = "rocm6"',
                ]
            )
        # EDF annotation keys contain dots, so retain them as quoted TOML keys.
        config = [
            json.dumps(line.split(" = ", 1)[0]) + " = " + line.split(" = ", 1)[1]
            if line.startswith("com.hooks.")
            else line
            for line in config
        ]
        (cell / "environment.toml").write_text("\n".join(config) + "\n")
        probe = deployment / ("probe-rocblas.py" if name.endswith("-rocblas") else "probe.py")
        image_hash = (
            (deployment / ("alps-image.sha256" if is_alps else "upstream-image.sha256"))
            .read_text()
            .split()[0]
        )
        write(
            cell / "metadata.json",
            {
                "cell": name,
                "deployment_id": deployment.name + "/" + name,
                "git_revision": revision,
                "git_branch": branch,
                "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
                "nodes": os.environ.get("SLURM_JOB_NODELIST"),
                "image_path": str(image),
                "image_sha256": image_hash,
                "oci_reference": identity["reference"] if is_alps else None,
                "image_source_revision": identity["image_source_revision"] if is_alps else None,
                "blas_request": "cublas (rocBLAS on ROCm)"
                if name.endswith("-rocblas")
                else "image default",
                "probe_sha256": sha(probe),
                "environment_sha256": sha(cell / "environment.toml"),
                "local_shape": [1024, 1024],
                "dtype": "bfloat16",
                "ranks": 2,
                "visible_gpus": "2,3",
                "samples_per_placement_per_rank": 5,
                "cleared_visibility_variables": [
                    "ROCR_VISIBLE_DEVICES",
                    "GPU_DEVICE_ORDINAL",
                    "HIP_VISIBLE_DEVICES",
                ],
                "slurm_network": "disable_rdzv_get" if is_alps else None,
                "summary_excludes_iterations": [0],
                "source_overlays": [],
                "inputs": "Unseeded random tensors from the original probe.",
                "network": "image-bundled stack; external injection disabled"
                if is_alps
                else "original Beverin netstack hooks",
            },
        )
    write(deployment / "host-environment.json", selected_environment())


def diagnose(cell):
    import importlib.metadata

    import torch

    requested = "cublas" if cell.name.endswith("-rocblas") else None
    default = str(torch.backends.cuda.preferred_blas_library())
    if requested:
        torch.backends.cuda.preferred_blas_library(requested)
    packages = {}
    for name in ("torch", "vllm", "triton", "transformers", "amd-aiter", "rocm-sdk"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    data = {
        "python": sys.version,
        "torch": torch.__version__,
        "hip": torch.version.hip,
        "default_blas_preference": default,
        "selected_blas_preference": str(torch.backends.cuda.preferred_blas_library()),
        "packages": packages,
        "environment": selected_environment(),
        "cuda_available": torch.cuda.is_available(),
        "device_count": torch.cuda.device_count(),
        "devices": [
            str(torch.cuda.get_device_properties(i)) for i in range(torch.cuda.device_count())
        ],
        "torch_num_threads": torch.get_num_threads(),
    }
    write(cell / "runtime.json", data)
    (cell / "torch-config.txt").write_text(torch.__config__.show())
    (cell / "packages.txt").write_text(
        subprocess.check_output([sys.executable, "-m", "pip", "freeze"], text=True)
    )
    print(json.dumps(data, sort_keys=True), flush=True)


def summarize(deployment):
    cells = {}
    decoder = json.JSONDecoder()
    for cell in sorted((deployment / "cells").iterdir()):
        raw = (cell / "raw.log").read_text(errors="replace")
        samples = []
        for match in re.finditer(r'\{"rank":', raw):
            samples.append(decoder.raw_decode(raw[match.start() :])[0])
        with (cell / "samples.jsonl").open("w") as output:
            for sample in samples:
                output.write(json.dumps(sample, sort_keys=True) + "\n")
        placements = {}
        for placement in ("None", "R", "S(0)"):
            values = [
                s["wall_ms"] for s in samples if s["placement"] == placement and s["iteration"] > 0
            ]
            placements[placement] = {
                "samples": len(values),
                "median_wall_ms": statistics.median(values) if values else None,
                "per_rank_median_wall_ms": {
                    str(rank): statistics.median(
                        [
                            s["wall_ms"]
                            for s in samples
                            if s["rank"] == rank
                            and s["placement"] == placement
                            and s["iteration"] > 0
                        ]
                    )
                    for rank in sorted(set(s["rank"] for s in samples))
                    if any(
                        s["rank"] == rank and s["placement"] == placement and s["iteration"] > 0
                        for s in samples
                    )
                },
            }
        rc = int((cell / "exit-code.txt").read_text())
        summary = {
            "exit_code": rc,
            "total_samples": len(samples),
            "complete": rc == 0
            and len(samples) == 30
            and all(p["samples"] == 8 for p in placements.values()),
            "aggregation": "Median wall_ms pooled over two ranks, iterations 1 through 4",
            "placements": placements,
        }
        write(cell / "summary.json", summary)
        cells[cell.name] = summary
    report = {
        "cells": cells,
        "complete": len(cells) == 4 and all(c["complete"] for c in cells.values()),
        "created_at": datetime.datetime.utcnow().isoformat() + "Z",
    }
    write(deployment / "summary.json", report)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    action = sys.argv[1]
    args = [Path(value) for value in sys.argv[2:]]
    {"prepare": prepare, "diagnose": diagnose, "summarize": summarize}[action](*args)
