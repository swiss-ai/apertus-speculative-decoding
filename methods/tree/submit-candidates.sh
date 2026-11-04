#!/usr/bin/env bash
# Submit methods/tree/run-candidates.sh on one 4-GPU node in the serving image.
#   methods/tree/submit-candidates.sh RUN_DIR
# TREE_PARTITION (normal), TREE_TIME (02:00:00); other env passes through.
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "${REPO_ROOT}"
RUN_DIR="$1"
ENV_SOURCE="${REPO_ROOT}/methods/eagle/configs/train-env.toml"
SML_ENVIRONMENT="$("${REPO_ROOT}/serving/resolve-env.sh" "${ENV_SOURCE}")"
sed -i "s|^workdir = \".*\"|workdir = \"${REPO_ROOT}\"|" "${SML_ENVIRONMENT}"
IMAGE="$(sed -n 's/^image = "\(.*\)"/\1/p' "${SML_ENVIRONMENT}")"
mkdir -p "${RUN_DIR}/logs"
cp "${SML_ENVIRONMENT}" "${RUN_DIR}/logs/environment.toml"
SBATCH_FILE="${RUN_DIR}/logs/submit-$(date -u +%Y%m%dT%H%M%SZ).sbatch"
cat > "${SBATCH_FILE}" <<SBATCH
#!/bin/bash
#SBATCH --job-name=tree-drafters
#SBATCH --account=${TREE_ACCOUNT:-infra01}
#SBATCH --partition=${TREE_PARTITION:-normal}
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=288
#SBATCH --gpus-per-node=4
#SBATCH --time=${TREE_TIME:-02:00:00}
#SBATCH --exclusive
#SBATCH --environment=${SML_ENVIRONMENT}
#SBATCH --output=${RUN_DIR}/logs/%j.out
#SBATCH --chdir=${REPO_ROOT}
set -euo pipefail
export TREE_IMAGE=${IMAGE}
export LIMIT=${LIMIT:-0} MAX_ANCHORS=${MAX_ANCHORS:-0} DSPARK_HF_LAYERS=${DSPARK_HF_LAYERS:-0,7,14,21,28}
export STAGES="${STAGES:-capture candidates analyze}"
echo "START \$(date -u -Iseconds) host=\$(hostname) job=\${SLURM_JOB_ID} image=${IMAGE}"
bash methods/tree/run-candidates.sh ${RUN_DIR}
echo "END \$(date -u -Iseconds)"
SBATCH
bash -n "${SBATCH_FILE}"
echo "sbatch_file=${SBATCH_FILE}"
sbatch "${SBATCH_FILE}"
