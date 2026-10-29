#!/usr/bin/env bash
# On-node EAGLE pipeline driver. Runs inside the allocated job and container.
# Everything target-specific comes from EAGLE_TRAIN_CONFIG and its contract;
# no 70B/TP=4 values are assumed here.
set -euo pipefail

if [ "${I_AM_ON_AN_ALLOCATED_GPU_JOB:-0}" != "1" ]; then
  echo "refusing to start a teacher or EAGLE training on this shell" >&2
  exit 2
fi
: "${EAGLE_TRAIN_CONFIG:?set EAGLE_TRAIN_CONFIG to a methods/eagle/configs/<stage>/train-*.yaml}"

LAUNCH_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "${LAUNCH_DIR}/../../.." && pwd)"
SERVING_DIR="${REPO_ROOT}/serving"
cd "${REPO_ROOT}"
TORCHSPEC_ROOT="${TORCHSPEC_ROOT:-${REPO_ROOT}/scratch/TorchSpec}"
export TORCHSPEC_ROOT
PYTHON="${PYTHON:-python3}"
EAGLE_PYDEPS="${EAGLE_PYDEPS:-${REPO_ROOT}/scratch/pydeps}"
# Multi-node jobs run this script once per node (srun). The setup below edits
# the shared filesystem, so only node 0 does it and the others wait for it.
NODE_RANK="${SLURM_NODEID:-0}"
SETUP_DONE="${REPO_ROOT}/scratch/.node-setup-${SLURM_JOB_ID:-local}"
setup_shared() {
  # TorchSpec imports wandb at package load. Use the offline stub, not pip wandb.
  rm -rf "${EAGLE_PYDEPS}/wandb"
  mkdir -p "${EAGLE_PYDEPS}/wandb"
  cp "${REPO_ROOT}/methods/eagle/apertus_eagle/wandb_offline_stub.py" "${EAGLE_PYDEPS}/wandb/__init__.py"
  bash "${LAUNCH_DIR}/apply-torchspec-patches.sh"
}
export PYTHONPATH="${EAGLE_PYDEPS}:${REPO_ROOT}/methods/eagle:${REPO_ROOT}/src:${TORCHSPEC_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
# Sequential teacher-then-trainer on one GPU (plan A3 simplest resource path).
export CUDA_VISIBLE_DEVICES="${EAGLE_CUDA_DEVICES:-0}"
# EAGLE_PLATFORM=rocm (Beverin MI300A; lessons of the DSpark 70B Beverin runs,
# docs/beverin-feasibility.md on branch dspark-70b-beverin).
if [ "${EAGLE_PLATFORM:-cuda}" = "rocm" ]; then
  # Slurm sets ROCR_VISIBLE_DEVICES; with CUDA_VISIBLE_DEVICES on top ROCm torch
  # miscounts devices (job 693749). Select GPUs with CUDA_VISIBLE_DEVICES only.
  unset ROCR_VISIBLE_DEVICES
  # JIT caches node-local (/tmp is RAM shared with the GPUs; removed at exit).
  BEV_CACHE="/tmp/bev-cache-${SLURM_JOB_ID:-local}"
  export TRITON_CACHE_DIR="${BEV_CACHE}/triton" TORCHINDUCTOR_CACHE_DIR="${BEV_CACHE}/inductor" \
    XDG_CACHE_HOME="${BEV_CACHE}/xdg" TORCH_EXTENSIONS_DIR="${BEV_CACHE}/torch_extensions" \
    MIOPEN_USER_DB_PATH="${BEV_CACHE}/miopen" MIOPEN_CUSTOM_CACHE_DIR="${BEV_CACHE}/miopen"
  mkdir -p "${BEV_CACHE}"
  export PYTORCH_HIP_ALLOC_CONF="${PYTORCH_HIP_ALLOC_CONF:-expandable_segments:True}"
  # Host and the 4 GPUs share one ~501 GB pool per node: log it every 10 s.
  MEM_LOG_DIR="${EAGLE_MEM_LOG_DIR:-/iopsstor/scratch/cscs/${USER}/beverin-eagle/logs}"
  mkdir -p "${MEM_LOG_DIR}"
  ( while sleep 10; do
      echo "$(date -u +%FT%TZ) $(free -g | awk '/Mem:/ {print "used_gb=" $3 " avail_gb=" $7}')"
    done ) >> "${MEM_LOG_DIR}/mem-${SLURM_JOB_ID:-local}-node${SLURM_NODEID:-0}.log" &
  MEM_MONITOR=$!
  trap 'kill "${MEM_MONITOR}" 2>/dev/null || true; rm -rf "${BEV_CACHE}"' EXIT
fi

log() { printf '%s %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*"; }
log "host=$(hostname) job=${SLURM_JOB_ID:-none} config=${EAGLE_TRAIN_CONFIG} steps=${EAGLE_STEPS:-all} cuda=${CUDA_VISIBLE_DEVICES}"
if [ "${EAGLE_PLATFORM:-cuda}" = "rocm" ]; then
  rocm-smi --showproductname --showmeminfo vram || true
else
  nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv || true
fi
if [ "${NODE_RANK}" = "0" ]; then
  setup_shared
  touch "${SETUP_DONE}"
else
  for _ in $(seq 1 120); do [ -f "${SETUP_DONE}" ] && break; sleep 5; done
  [ -f "${SETUP_DONE}" ] || { log "node ${NODE_RANK}: node 0 setup did not finish"; exit 1; }
fi
"${PYTHON}" - <<'PY'
import importlib, json, platform, sys
mods = {}
for name in ("torch", "transformers", "vllm", "safetensors", "yaml", "accelerate", "triton"):
    try:
        mods[name] = getattr(importlib.import_module(name), "__version__", "imported")
    except Exception as exc:  # noqa: BLE001
        mods[name] = f"MISSING {type(exc).__name__}: {exc}"
import torch
try:  # only the offline verifier needs vLLM; the Beverin image's is upstream ROCm vLLM
    import vllm
    mods["vllm_file"] = vllm.__file__
except Exception as exc:  # noqa: BLE001
    mods["vllm_file"] = f"MISSING {type(exc).__name__}: {exc}"
mods["cuda"] = torch.version.cuda
mods["hip"] = getattr(torch.version, "hip", None)
mods["gpu"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
mods["machine"] = platform.machine()
mods["python"] = sys.version.split()[0]
print(json.dumps({"environment": mods}))
PY

"${PYTHON}" -m apertus_eagle.pipeline --config "${EAGLE_TRAIN_CONFIG}" \
  ${EAGLE_STEPS:+--steps "${EAGLE_STEPS}"} ${EAGLE_RUN_ID:+--run-id "${EAGLE_RUN_ID}"}
log "pipeline finished"
