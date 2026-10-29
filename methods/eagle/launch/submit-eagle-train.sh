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
# Unset keeps the Clariden defaults; an empty value omits the line (Beverin:
# EAGLE_TRAIN_CPUS= EAGLE_TRAIN_MEM= EAGLE_TRAIN_EXCLUDE=).
EXCLUDE_NODES="${EAGLE_TRAIN_EXCLUDE-nid007129}"
CPUS_PER_TASK="${EAGLE_TRAIN_CPUS-288}"
MEMORY="${EAGLE_TRAIN_MEM-450G}"
# Extra sbatch arguments at submit time (e.g. "--nice=10000" for pilots, "--hold").
SBATCH_EXTRA="${EAGLE_SBATCH_EXTRA:-}"
# EAGLE_CHAIN=N queues N jobs, each starting after the previous one ends, all with
# the same run id: a job near its time limit saves checkpoints/resume and the next
# one continues (train_rollout exit status 4). Needs EAGLE_RUN_ID or sets one.
CHAIN="${EAGLE_CHAIN:-1}"
# EAGLE_NODES=N trains on N nodes (training.data_parallel ranks per node): the
# batch script runs outside the container and srun starts one container per node.
NODES="${EAGLE_NODES:-1}"
# EAGLE_RETRIES=N tries training up to N times inside one allocation (see train_once).
RETRIES="${EAGLE_RETRIES:-3}"
# Optional lines, each printed with a leading newline, so that nothing is added to
# the batch script while the variable is unset or empty.
sbatch_line() { [ -z "$2" ] || printf '\n#SBATCH %s=%s' "$1" "$2"; }
# Opt-in settings written into the batch script (unset: nothing written):
# EAGLE_PLATFORM=rocm (train-eagle-on-node.sh), EAGLE_BLAS_BACKEND (train_rollout),
# EAGLE_STOP_AFTER_STEPS (test hook), EAGLE_START_GUARD_RUN_DIR (start_guard.py),
# EAGLE_MEM_LOG_DIR (10 s free -g log per node on ROCm).
OPTIONAL_EXPORTS=""
for v in EAGLE_PLATFORM EAGLE_BLAS_BACKEND EAGLE_STOP_AFTER_STEPS EAGLE_START_GUARD_RUN_DIR EAGLE_MEM_LOG_DIR; do
  if [ -n "${!v:-}" ]; then OPTIONAL_EXPORTS+="$(printf '\nexport %s=%q' "${v}" "${!v}")"; fi
done
GUARD_LINE=""
if [ -n "${EAGLE_START_GUARD_RUN_DIR:-}" ]; then
  # Refuse to train unless the cutover record says this cluster owns the run.
  GUARD_LINE="$(printf '\npython3 %s/methods/eagle/beverin/start_guard.py "${EAGLE_START_GUARD_RUN_DIR}" || exit 3' "${REPO_ROOT}")"
fi
if [ "${CHAIN}" -gt 1 ] && [ -z "${RUN_ID}" ]; then RUN_ID="$(date -u +%Y%m%dT%H%M%SZ)"; fi
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
#SBATCH --nodes=${NODES}
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=${GPUS_PER_NODE}$(sbatch_line --cpus-per-task "${CPUS_PER_TASK}")$(sbatch_line --mem "${MEMORY}")
#SBATCH --time=${TIME_LIMIT}
#SBATCH --exclusive$(sbatch_line --exclude "${EXCLUDE_NODES}")
$([ "${NODES}" = 1 ] && echo "#SBATCH --environment=${SML_ENVIRONMENT}")
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
# Wall-clock deadline for train_rollout (it stops, saves and exits 4 before this).
limit_s=\$(echo "${TIME_LIMIT}" | awk -F: '{ if (NF == 3) print \$1 * 3600 + \$2 * 60 + \$3; else print \$1 * 60 + \$2 }')
export EAGLE_STOP_AT=\$(( \$(date +%s) + limit_s ))
export TORCHSPEC_ROOT=${REPO_ROOT}/scratch/TorchSpec
export EAGLE_PYDEPS=${REPO_ROOT}/scratch/pydeps
export HF_HOME=/iopsstor/scratch/cscs/\${USER}/hf_home
export TMPDIR=/tmp
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export WANDB_MODE=offline
export WANDB_DISABLED=true
export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1${OPTIONAL_EXPORTS}
# No percent-strftime in this body: pyxis --environment sbatch mangled one (job 3491998).
echo "START \$(date -u -Iseconds) host=\$(hostname) job=\${SLURM_JOB_ID} nodes=${NODES}"${GUARD_LINE}
train_once() {
$(if [ "${NODES}" = 1 ]; then
  echo "bash ${REPO_ROOT}/methods/eagle/launch/train-eagle-on-node.sh"
else
  cat <<MULTI
export EAGLE_NNODES=${NODES}
export EAGLE_MASTER_ADDR=\$(scontrol show hostnames "\${SLURM_JOB_NODELIST}" | head -1)
srun --ntasks-per-node=1 --environment=${SML_ENVIRONMENT} \\
  bash ${REPO_ROOT}/methods/eagle/launch/train-eagle-on-node.sh
MULTI
fi)
}
# A crash (not 0 = done, not 4 = saved at the deadline) relaunches on the same nodes
# from the last resume checkpoint instead of giving the allocation back to the queue.
attempt=1
while :; do
  set +e; train_once; rc=\$?; set -e
  if [ "\${rc}" = 0 ] || [ "\${rc}" = 4 ]; then break; fi
  left=\$(( EAGLE_STOP_AT - \$(date +%s) ))
  if [ "\${attempt}" -ge ${RETRIES} ] || [ "\${left}" -lt 1800 ]; then
    echo "END \$(date -u -Iseconds) rc=\${rc} attempts=\${attempt}"
    exit "\${rc}"
  fi
  echo "RETRY \$(date -u -Iseconds) rc=\${rc} attempt=\${attempt} left_s=\${left}"
  attempt=\$(( attempt + 1 ))
  sleep 30
done
echo "END \$(date -u -Iseconds) rc=\${rc}"
exit "\${rc}"
SBATCH

bash -n "${SBATCH_FILE}"
if [ "${EAGLE_SUBMIT_DRY_RUN:-0}" = "1" ]; then
  echo "dry run; not submitting"
  exit 0
fi
previous=""
for link in $(seq 1 "${CHAIN}"); do
  after="${DEPENDENCY}"
  if [ -n "${previous}" ]; then after="afterany:${previous}"; fi
  # shellcheck disable=SC2086
  previous="$(sbatch --parsable ${after:+--dependency=${after}} ${SBATCH_EXTRA} "${SBATCH_FILE}")"
  echo "Submitted batch job ${previous} (chain ${link}/${CHAIN}, run id ${RUN_ID:-per-job})"
done
