#!/usr/bin/env bash
# Submit DSpark 70B data preparation or training on Clariden (login node only submits).
#   MODE=prepare IN=... OUT=... ./methods/dspark/train/submit-train.sh   # or FULL=1
#   MODE=train RUN_ID=opb-thinkoff DATA=... [NODES=2] [CHAIN=12] ./methods/dspark/train/submit-train.sh
# MODE=train queues CHAIN jobs of TIME each, every one after the previous ends
# (afterany), all on the same RUN_ID: each resumes from the run's checkpoints.
# Pilot knobs passed through to train-node.sh: MAX_STEPS, CHECK_ROWS, GRAD_CKPT,
# EPOCHS, WORKERS, PREFETCH. Extra sbatch options go in SBATCH_EXTRA.
set -euo pipefail

TRAIN_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "${TRAIN_DIR}/../../.." && pwd)"
cd "${REPO_ROOT}"

MODE="${MODE:?MODE=prepare or train}"
ACCOUNT="${DSPARK_ACCOUNT:-infra01}"
PARTITION="${DSPARK_PARTITION:-normal}"
DSPARK_DIR="${DSPARK_DIR:-/iopsstor/scratch/cscs/${USER}/apertus-dspark70b}"
EXCLUDE="${DSPARK_EXCLUDE:-nid007129,nid007593,nid007611}"
ENV_SOURCE="${DSPARK_ENV_SOURCE:-${REPO_ROOT}/methods/eagle/configs/train-env.toml}"
IMAGE="$(sed -n 's|^ *image *= *"\(/[^"]*\)".*|\1|p' "${ENV_SOURCE}" | head -1 | sed "s|{arch}|arm64|")"
LOG_ROOT="${REPO_ROOT}/results/70b/dspark/logs"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
mkdir -p "${LOG_ROOT}"

# Extraction needs the aux-hidden-state fix for Apertus's multimodal wrapper
# (Yu's fork_patch.sh); its own overlay directory, on scratch since home is full.
EXTRA_MOUNTS="$(VLLM_PATCH_DIR="${DSPARK_DIR}/vllm-patch" "${REPO_ROOT}/serving/patch-vllm.sh" "${IMAGE}" \
  "${REPO_ROOT}/serving/patches/vllm-apertus-eagle3-aux-layers.patch")"
export EXTRA_MOUNTS
SML_ENVIRONMENT="$("${REPO_ROOT}/serving/resolve-env.sh" "${ENV_SOURCE}")"
sed -i "s|^workdir = \".*\"|workdir = \"${REPO_ROOT}\"|" "${SML_ENVIRONMENT}"

pass_env() {
  local v
  for v in DSPARK_DIR RUN_ID DATA EPOCHS MAX_STEPS CHECK_ROWS GRAD_CKPT WORKERS PREFETCH SEQ ANCHORS LR CKPT_FREQ \
           EXTRACT_MEM EXTRACT_EAGER IN OUT FULL ALLOW_MISSING; do
    if [ -n "${!v:-}" ]; then printf 'export %s=%q\n' "${v}" "${!v}"; fi
  done
}

case "${MODE}" in
  prepare)
    NODES=1
    NAME=dspark-70b-prepare
    TIME="${TIME:-04:00:00}"
    CMD="srun --ntasks-per-node=1 --environment=${SML_ENVIRONMENT} bash methods/dspark/train/prepare.sh"
    CHAIN=1
    ;;
  train)
    : "${RUN_ID:?set RUN_ID}" "${DATA:?set DATA (prepared dataset directory)}"
    NODES="${NODES:-2}"
    NAME="dspark-70b-${RUN_ID}"
    TIME="${TIME:-12:00:00}"
    # The trainer saves and exits at a step boundary STOP_MARGIN seconds before
    # the time limit (speculators-deadline-stop.patch); a save took up to 27 min.
    IFS=: read -r th tm ts <<< "${TIME}"
    LIMIT_S=$(( 10#${th} * 3600 + 10#${tm} * 60 + 10#${ts} ))
    CMD="export DSPARK_STOP_AT=\$(( \$(date +%s) + ${LIMIT_S} - ${STOP_MARGIN:-2400} ))
export HEAD_NODE=\$(scontrol show hostnames \"\${SLURM_JOB_NODELIST}\" | head -1)
srun --ntasks-per-node=1 --environment=${SML_ENVIRONMENT} bash methods/dspark/train/train-node.sh"
    CHAIN="${CHAIN:-1}"
    ;;
  *) echo "unknown MODE ${MODE}" >&2; exit 2 ;;
esac

SCRIPT="${LOG_ROOT}/submit-${MODE}-${STAMP}.sbatch"
cat > "${SCRIPT}" <<SBATCH
#!/bin/bash
#SBATCH --job-name=${NAME}
#SBATCH --account=${ACCOUNT}
#SBATCH --partition=${PARTITION}
#SBATCH --time=${TIME}
#SBATCH --nodes=${NODES}
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=288
#SBATCH --exclusive
#SBATCH --exclude=${EXCLUDE}
#SBATCH --chdir=${REPO_ROOT}
#SBATCH --output=${LOG_ROOT}/%j.out
set -euo pipefail
export REPO_ROOT=${REPO_ROOT}
$(pass_env)
echo "START \$(date -u -Iseconds) job=\${SLURM_JOB_ID} nodes=\${SLURM_JOB_NODELIST} git=$(git rev-parse --short HEAD)"
${CMD}
SBATCH

dep="${DEPENDENCY:-}"
for i in $(seq 1 "${CHAIN}"); do
  # shellcheck disable=SC2086
  id="$(sbatch --parsable ${dep:+--dependency=${dep}} ${SBATCH_EXTRA:-} "${SCRIPT}")"
  echo "submitted ${id} (${i}/${CHAIN}) ${SCRIPT}"
  dep="afterany:${id}"
done
