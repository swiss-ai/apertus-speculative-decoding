#!/usr/bin/env bash
# Submit one EAGLE pipeline job on Clariden (normal partition by default).
#   EAGLE_TRAIN_CONFIG=methods/eagle/configs/8b/train-e31-overfit.yaml ./methods/eagle/launch/submit-eagle-train.sh
# The config names the target contract, draft config, data and output roots.
# Never uses the debug partition while a diagnostics replica holds debug-qos.
set -euo pipefail

LAUNCH_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "${LAUNCH_DIR}/../../.." && pwd)"
SERVING_DIR="${REPO_ROOT}/serving"
cd "${REPO_ROOT}"

: "${EAGLE_TRAIN_CONFIG:?set EAGLE_TRAIN_CONFIG (e.g. methods/eagle/configs/8b/train-e31-overfit.yaml)}"
if [ ! -f "${EAGLE_TRAIN_CONFIG}" ]; then
  echo "config not found: ${EAGLE_TRAIN_CONFIG}" >&2
  exit 2
fi
CONFIG_ABS="$(cd "$(dirname "${EAGLE_TRAIN_CONFIG}")" && pwd)/$(basename "${EAGLE_TRAIN_CONFIG}")"
ENV_SOURCE="${EAGLE_ENV_SOURCE:-${REPO_ROOT}/methods/eagle/configs/train-env.toml}"
PARTITION="${EAGLE_TRAIN_PARTITION:-normal}"
ACCOUNT="${EAGLE_TRAIN_ACCOUNT:-infra01}"
TIME_LIMIT="${EAGLE_TRAIN_TIME:-06:00:00}"
GPUS_PER_NODE="${EAGLE_GPUS_PER_NODE:-4}"
CUDA_DEVICES="${EAGLE_CUDA_DEVICES:-0}"
STEPS="${EAGLE_STEPS:-}"
RUN_ID="${EAGLE_RUN_ID:-}"
EXCLUDE_NODES="${EAGLE_TRAIN_EXCLUDE:-nid007129}"
# e.g. EAGLE_DEPENDENCY=afterok:3505554 so a stage starts only after its gate job passed.
DEPENDENCY="${EAGLE_DEPENDENCY:-}"
STAGE="$(sed -n 's/^stage: *\([^ #]*\).*/\1/p' "${CONFIG_ABS}" | head -1)"
RUN_NAME="$(sed -n 's/^run_name: *\([^ #]*\).*/\1/p' "${CONFIG_ABS}" | head -1)"
LOG_ROOT="${REPO_ROOT}/results/${STAGE:-unknown}/eagle/logs"

DEBUG_JOBS="$(squeue -u "${USER}" -p debug -h -o '%i:%N:%T' 2>/dev/null || true)"
if [ "${PARTITION}" = "debug" ] && [ -n "${DEBUG_JOBS}" ]; then
  echo "refusing debug submit: debug-qos MaxJobsPU=1 is already held" >&2
  exit 2
fi

# The offline EAGLE verifier needs the same vLLM overlays as methods/eagle/launch/eagle.sh.
# Own patch dir: patch-vllm.sh wipes its OUT_DIR, which a live replica may mount.
if [ "${EAGLE_APPLY_VLLM_PATCHES:-1}" = "1" ] && [ -z "${EXTRA_MOUNTS:-}" ]; then
  IMAGE="$(sed -n 's|^ *image *= *"\(/[^"]*\)".*|\1|p' "${ENV_SOURCE}" | head -1 | sed "s|{arch}|${SML_ARCH:-arm64}|")"
  if [ -f "${IMAGE}" ]; then
    EXTRA_MOUNTS="$(
      VLLM_PATCH_DIR="${EAGLE_VLLM_PATCH_DIR:-${HOME}/.sml/vllm-patch-train}" \
        "${SERVING_DIR}/patch-vllm.sh" "${IMAGE}" \
        "${REPO_ROOT}/serving/patches/vllm-apertus-image-token.patch" \
        "${REPO_ROOT}/serving/patches/vllm-apertus-eagle3-aux-layers.patch" \
        "${REPO_ROOT}/serving/patches/vllm-apertus-eagle-mrv2-inner.patch"
    )"
    export EXTRA_MOUNTS
  fi
fi
SML_ENVIRONMENT="$("${SERVING_DIR}/resolve-env.sh" "${ENV_SOURCE}")"
sed -i "s|^workdir = \".*\"|workdir = \"${REPO_ROOT}\"|" "${SML_ENVIRONMENT}"
mkdir -p "${LOG_ROOT}"
SBATCH_FILE="${LOG_ROOT}/submit-${RUN_NAME:-eagle}-$(date -u +%Y%m%dT%H%M%SZ).sbatch"

cat <<INFO
config=${CONFIG_ABS}
stage=${STAGE}
run_name=${RUN_NAME}
steps=${STEPS:-all}
partition=${PARTITION}
time=${TIME_LIMIT}
dependency=${DEPENDENCY:-none}
allocated_gpus=${GPUS_PER_NODE} (exclusive node)
used_cuda_devices=${CUDA_DEVICES}
environment=${SML_ENVIRONMENT}
vllm_overlays=${EXTRA_MOUNTS:-none}
sbatch_file=${SBATCH_FILE}
INFO

cat > "${SBATCH_FILE}" <<SBATCH
#!/bin/bash
#SBATCH --job-name=eagle-${STAGE}-${RUN_NAME}
#SBATCH --account=${ACCOUNT}
#SBATCH --partition=${PARTITION}
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=${GPUS_PER_NODE}
#SBATCH --cpus-per-task=288
#SBATCH --mem=450G
#SBATCH --time=${TIME_LIMIT}
#SBATCH --exclusive
#SBATCH --exclude=${EXCLUDE_NODES}
#SBATCH --environment=${SML_ENVIRONMENT}
#SBATCH --output=${LOG_ROOT}/%j.out
#SBATCH --error=${LOG_ROOT}/%j.err
#SBATCH --chdir=${REPO_ROOT}
${DEPENDENCY:+#SBATCH --dependency=${DEPENDENCY}}
#SBATCH --kill-on-invalid-dep=yes

set -euo pipefail
export I_AM_ON_AN_ALLOCATED_GPU_JOB=1
export EAGLE_TRAIN_CONFIG=${CONFIG_ABS}
export EAGLE_CUDA_DEVICES=${CUDA_DEVICES}
export EAGLE_STEPS=${STEPS}
export EAGLE_RUN_ID=${RUN_ID}
export TORCHSPEC_ROOT=${REPO_ROOT}/scratch/TorchSpec
export EAGLE_PYDEPS=${REPO_ROOT}/scratch/pydeps
export HF_HOME=/iopsstor/scratch/cscs/\${USER}/hf_home
export TMPDIR=/tmp
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export WANDB_MODE=offline
export WANDB_DISABLED=true
export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1
# No percent-strftime in this body: pyxis --environment sbatch mangled one (job 3491998).
echo "START \$(date -u -Iseconds) host=\$(hostname) job=\${SLURM_JOB_ID}"
bash ${REPO_ROOT}/methods/eagle/launch/train-eagle-on-node.sh
echo "END \$(date -u -Iseconds)"
SBATCH

bash -n "${SBATCH_FILE}"
if [ "${EAGLE_SUBMIT_DRY_RUN:-0}" = "1" ]; then
  echo "dry run; not submitting"
  exit 0
fi
sbatch "${SBATCH_FILE}"
