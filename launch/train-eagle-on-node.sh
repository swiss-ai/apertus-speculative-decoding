#!/usr/bin/env bash
# On-node EAGLE pipeline driver. Runs inside the allocated job and container.
# Everything target-specific comes from EAGLE_TRAIN_CONFIG and its contract;
# no 70B/TP=4 values are assumed here.
set -euo pipefail

if [ "${I_AM_ON_AN_ALLOCATED_GPU_JOB:-0}" != "1" ]; then
  echo "refusing to start a teacher or EAGLE training on this shell" >&2
  exit 2
fi
: "${EAGLE_TRAIN_CONFIG:?set EAGLE_TRAIN_CONFIG to a configs/eagle/<stage>/train-*.yaml}"

LAUNCH_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "${LAUNCH_DIR}/.." && pwd)"
cd "${REPO_ROOT}"
TORCHSPEC_ROOT="${TORCHSPEC_ROOT:-${REPO_ROOT}/scratch/TorchSpec}"
export TORCHSPEC_ROOT
PYTHON="${PYTHON:-python3}"
# TorchSpec imports wandb at package load. Use the offline stub, not pip wandb.
EAGLE_PYDEPS="${EAGLE_PYDEPS:-${REPO_ROOT}/scratch/pydeps}"
rm -rf "${EAGLE_PYDEPS}/wandb"
mkdir -p "${EAGLE_PYDEPS}/wandb"
cp "${REPO_ROOT}/training/apertus_eagle/wandb_offline_stub.py" "${EAGLE_PYDEPS}/wandb/__init__.py"
export PYTHONPATH="${EAGLE_PYDEPS}:${REPO_ROOT}/training:${REPO_ROOT}/src:${TORCHSPEC_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
# Sequential teacher-then-trainer on one GPU (plan A3 simplest resource path).
export CUDA_VISIBLE_DEVICES="${EAGLE_CUDA_DEVICES:-0}"

log() { printf '%s %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*"; }
log "host=$(hostname) job=${SLURM_JOB_ID:-none} config=${EAGLE_TRAIN_CONFIG} steps=${EAGLE_STEPS:-all} cuda=${CUDA_VISIBLE_DEVICES}"
nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv || true
bash "${LAUNCH_DIR}/apply-torchspec-patches.sh"
"${PYTHON}" - <<'PY'
import importlib, json, platform, sys
mods = {}
for name in ("torch", "transformers", "vllm", "safetensors", "yaml", "accelerate", "triton"):
    try:
        mods[name] = getattr(importlib.import_module(name), "__version__", "imported")
    except Exception as exc:  # noqa: BLE001
        mods[name] = f"MISSING {type(exc).__name__}: {exc}"
import torch
import vllm
mods["vllm_file"] = vllm.__file__
mods["cuda"] = torch.version.cuda
mods["gpu"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
mods["machine"] = platform.machine()
mods["python"] = sys.version.split()[0]
print(json.dumps({"environment": mods}))
PY

"${PYTHON}" -m apertus_eagle.pipeline --config "${EAGLE_TRAIN_CONFIG}" \
  ${EAGLE_STEPS:+--steps "${EAGLE_STEPS}"} ${EAGLE_RUN_ID:+--run-id "${EAGLE_RUN_ID}"}
log "pipeline finished"
