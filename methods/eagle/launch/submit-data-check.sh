#!/usr/bin/env bash
# Submit the open-perfectblend download and data check as a compute job.
#   ./methods/eagle/launch/submit-data-check.sh
# Network, parsing and tokenization all run on the allocated node, not here.
set -euo pipefail

LAUNCH_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "${LAUNCH_DIR}/../../.." && pwd)"
SERVING_DIR="${REPO_ROOT}/serving"
cd "${REPO_ROOT}"

ENV_SOURCE="${EAGLE_ENV_SOURCE:-${REPO_ROOT}/methods/eagle/configs/train-env.toml}"
PARTITION="${DATA_CHECK_PARTITION:-debug}"
TIME_LIMIT="${DATA_CHECK_TIME:-01:30:00}"
OUTPUT_DIR="${DATA_CHECK_OUTPUT:-/iopsstor/scratch/cscs/${USER}/apertus-eagle/data/open-perfectblend}"
REPORT="${REPO_ROOT}/results/8b/eagle/data-perfectblend/check.json"
TOKENIZER="/capstor/store/cscs/swissai/infra01/hf_models/models/swiss-ai/Apertus-v1.5-8B"
LOG_ROOT="${REPO_ROOT}/results/8b/eagle/logs"
WORKLOADS="${REPO_ROOT}/workloads/8b/eagle-8b-validation.jsonl ${REPO_ROOT}/workloads/8b/eagle-8b-test.jsonl ${REPO_ROOT}/workloads/8b/eagle-8b-profile.jsonl"

SML_ENVIRONMENT="$("${SERVING_DIR}/resolve-env.sh" "${ENV_SOURCE}")"
sed -i "s|^workdir = \".*\"|workdir = \"${REPO_ROOT}\"|" "${SML_ENVIRONMENT}"
mkdir -p "${LOG_ROOT}" "${OUTPUT_DIR}"
SBATCH_FILE="${LOG_ROOT}/submit-data-check-$(date -u +%Y%m%dT%H%M%SZ).sbatch"

cat > "${SBATCH_FILE}" <<SBATCH
#!/bin/bash
#SBATCH --job-name=eagle-8b-data-check
#SBATCH --account=${EAGLE_TRAIN_ACCOUNT:-infra01}
#SBATCH --partition=${PARTITION}
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=288
#SBATCH --time=${TIME_LIMIT}
#SBATCH --exclusive
#SBATCH --environment=${SML_ENVIRONMENT}
#SBATCH --output=${LOG_ROOT}/%j.out
#SBATCH --error=${LOG_ROOT}/%j.err
#SBATCH --chdir=${REPO_ROOT}

set -euo pipefail
export HF_HOME=/iopsstor/scratch/cscs/\${USER}/hf_home
export HF_HUB_ENABLE_HF_TRANSFER=0
export PYTHONPATH=${REPO_ROOT}/methods/eagle:${REPO_ROOT}/src
export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1
echo "START \$(date -u -Iseconds) host=\$(hostname) job=\${SLURM_JOB_ID}"
python3 -m apertus_eagle.perfectblend_check --output-dir ${OUTPUT_DIR} --report ${REPORT} \\
  --tokenizer ${TOKENIZER} --workloads ${WORKLOADS}
echo "END \$(date -u -Iseconds)"
SBATCH

bash -n "${SBATCH_FILE}"
echo "sbatch_file=${SBATCH_FILE} output=${OUTPUT_DIR} report=${REPORT}"
if [ "${DATA_CHECK_DRY_RUN:-0}" = "1" ]; then
  echo "dry run; not submitting"
  exit 0
fi
sbatch "${SBATCH_FILE}"
