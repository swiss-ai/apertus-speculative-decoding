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
if [ -n "${EAGLE_RECOVERY_CACHE_ROOT:-}" ] && [ "${T5_CACHE_WRAPPER_STARTED:-0}" != 1 ]; then
  cache="${EAGLE_RECOVERY_CACHE_ROOT}/${SLURM_JOB_ID}/node${NODE_RANK}"
  exec "${PYTHON}" -m apertus_bench.runtime_caches --root "${cache}" \
    --report "${cache}/runtime-cache-preflight.json" --probe-flashinfer \
    --exec env T5_CACHE_WRAPPER_STARTED=1 bash "$0"
fi

log() { printf '%s %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*"; }
log "host=$(hostname) job=${SLURM_JOB_ID:-none} config=${EAGLE_TRAIN_CONFIG} steps=${EAGLE_STEPS:-all} cuda=${CUDA_VISIBLE_DEVICES}"
nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv || true
if [ "${NODE_RANK}" = "0" ]; then
  setup_shared
  touch "${SETUP_DONE}"
else
  for _ in $(seq 1 120); do [ -f "${SETUP_DONE}" ] && break; sleep 5; done
  [ -f "${SETUP_DONE}" ] || { log "node ${NODE_RANK}: node 0 setup did not finish"; exit 1; }
fi
if [ -n "${EAGLE_DEPLOYMENT_ROOT:-}" ]; then
  "${PYTHON}" -m apertus_eagle.adaptation_provenance --repo "${REPO_ROOT}" \
    --deployment "${EAGLE_DEPLOYMENT_ROOT}/${SLURM_JOB_ID}"
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
import vllm
mods["vllm_file"] = vllm.__file__
mods["cuda"] = torch.version.cuda
mods["gpu"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
mods["machine"] = platform.machine()
mods["python"] = sys.version.split()[0]
print(json.dumps({"environment": mods}))
PY

# Several space-separated configs run in turn; a failed one does not stop the
# rest, and the job fails if any did.
failed=0
for config in ${EAGLE_TRAIN_CONFIG}; do
  log "pipeline config=${config}"
  rc=0
  "${PYTHON}" -m apertus_eagle.pipeline --config "${config}" \
    ${EAGLE_STEPS:+--steps "${EAGLE_STEPS}"} ${EAGLE_RUN_ID:+--run-id "${EAGLE_RUN_ID}"} \
    || rc=$?
  if [ -n "${EAGLE_DEPLOYMENT_ROOT:-}" ]; then
    # A scientific negative must not stop independent arms. Nonfinite updates,
    # missing/corrupt exports and other infrastructure failures must stop here.
    outcome=0
    "${PYTHON}" -m apertus_eagle.adaptation_validate --config "${config}" \
      --run-id "${EAGLE_RUN_ID}" || outcome=$?
    if [ "${outcome}" = 4 ]; then
      log "recovery deadline stop; controller will resume the remaining bounded curves"
      failed=0
      break
    fi
    [ "${outcome}" = 0 ] || exit "${outcome}"
    [ "${rc}" = 0 ] || [ "${rc}" = 3 ] || exit "${rc}"
  fi
  if [ "${rc}" != 0 ]; then
    failed=${rc}
    log "pipeline failed rc=${rc} config=${config}"
  fi
done
log "pipeline finished"
exit "${failed}"
