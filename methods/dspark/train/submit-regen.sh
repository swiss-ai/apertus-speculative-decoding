#!/usr/bin/env bash
# Submit the 70B open-perfectblend regeneration on Clariden (login node only submits).
#   MODE=prep  ./methods/dspark/train/submit-regen.sh   # side deps, dataset, shards, parts
#   MODE=pilot [TP=4] ./methods/dspark/train/submit-regen.sh   # throughput on one node
#   MODE=parts [TP=4] REGEN_NODES=8 ./methods/dspark/train/submit-regen.sh
# MODE=parts is an array with one node per part, at most REGEN_NODES at a time;
# resubmitting resumes every unfinished part (PARTS=3,17 limits it to some).
# Extra sbatch options (e.g. a dependency) go in SBATCH_EXTRA.
set -euo pipefail

TRAIN_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "${TRAIN_DIR}/../../.." && pwd)"
cd "${REPO_ROOT}"

MODE="${MODE:?MODE=prep, pilot or parts}"
ACCOUNT="${DSPARK_ACCOUNT:-infra01}"
PARTITION="${REGEN_PARTITION:-normal}"
DSPARK_DIR="${DSPARK_DIR:-/iopsstor/scratch/cscs/${USER}/apertus-dspark70b}"
NPARTS="${NPARTS:-32}"
TP="${TP:-4}"
REGEN_NODES="${REGEN_NODES:-8}"
EXCLUDE="${DSPARK_EXCLUDE:-nid007129,nid007593,nid007611}"
ENV_SOURCE="${DSPARK_ENV_SOURCE:-${REPO_ROOT}/methods/eagle/configs/train-env.toml}"
LOG_ROOT="${REPO_ROOT}/results/70b/dspark/logs"
SML_ENVIRONMENT="$("${REPO_ROOT}/serving/resolve-env.sh" "${ENV_SOURCE}")"
sed -i "s|^workdir = \".*\"|workdir = \"${REPO_ROOT}\"|" "${SML_ENVIRONMENT}"
mkdir -p "${LOG_ROOT}" "${DSPARK_DIR}/regen"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"

# The batch script stays outside the container and srun starts it, as Yu's and
# model-launch's jobs do: with the container on the batch step itself
# (#SBATCH --environment), vLLM's custom all-reduce hit illegal memory accesses
# during CUDA-graph capture (jobs 3680758, 3680807).
RUN="srun --ntasks-per-node=1 --environment=${SML_ENVIRONMENT}"

header() {  # name time output [extra #SBATCH line]; directives must precede commands
  cat <<SBATCH
#!/bin/bash
#SBATCH --job-name=$1
#SBATCH --account=${ACCOUNT}
#SBATCH --partition=${PARTITION}
#SBATCH --time=$2
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=288
#SBATCH --exclusive
#SBATCH --exclude=${EXCLUDE}
#SBATCH --chdir=${REPO_ROOT}
#SBATCH --output=$3
${4:-}
set -euo pipefail
export REPO_ROOT=${REPO_ROOT} DSPARK_DIR=${DSPARK_DIR}
SBATCH
}

SCRIPT="${LOG_ROOT}/submit-regen-${MODE}-${STAMP}.sbatch"
case "${MODE}" in
  prep)
    { header dspark-70b-prep 01:00:00 "${LOG_ROOT}/%j.out"
      echo "NPARTS=${NPARTS} ${RUN} bash methods/dspark/train/prep.sh"
    } > "${SCRIPT}"
    ;;
  pilot)
    LIMIT="${LIMIT:-6000}"
    { header dspark-70b-regen-pilot 02:30:00 "${LOG_ROOT}/%j.out"
      # Until one starts: as configured, without custom all-reduce, eager.
      echo "pilot=\"LIMIT=${LIMIT} IN=${DSPARK_DIR}/prompts/parts/part-000.jsonl BUDGET_SECONDS=2400 HEALTH_SECONDS=1500 TP=${TP}\""
      echo "env \$pilot OUT=${DSPARK_DIR}/pilot/tp${TP}.jsonl ${RUN} bash methods/dspark/train/regen-part.sh \\"
      echo "  || env \$pilot OUT=${DSPARK_DIR}/pilot/tp${TP}-nocar.jsonl VLLM_EXTRA=--disable-custom-all-reduce ${RUN} bash methods/dspark/train/regen-part.sh \\"
      echo "  || env \$pilot OUT=${DSPARK_DIR}/pilot/tp${TP}-eager.jsonl VLLM_EXTRA=--enforce-eager ${RUN} bash methods/dspark/train/regen-part.sh || true"
    } > "${SCRIPT}"
    mkdir -p "${DSPARK_DIR}/pilot"
    ;;
  parts)
    ARRAY="${PARTS:-0-$(( NPARTS - 1 ))}"
    { header dspark-70b-regen 12:00:00 "${LOG_ROOT}/%A_%a.out" "#SBATCH --array=${ARRAY}%${REGEN_NODES}"
      echo "part=\$(printf '%03d' \${SLURM_ARRAY_TASK_ID})"
      echo "TP=${TP} VLLM_EXTRA=${VLLM_EXTRA:-} BUDGET_SECONDS=42600 IN=${DSPARK_DIR}/prompts/parts/part-\${part}.jsonl OUT=${DSPARK_DIR}/regen/part-\${part}.jsonl ${RUN} bash methods/dspark/train/regen-part.sh"
    } > "${SCRIPT}"
    ;;
  *) echo "unknown MODE ${MODE}" >&2; exit 2 ;;
esac
# shellcheck disable=SC2086
sbatch ${SBATCH_EXTRA:-} "${SCRIPT}"
