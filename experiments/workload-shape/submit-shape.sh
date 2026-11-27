#!/usr/bin/env bash
# Submit experiments/workload-shape/run-shape.sh on one 4-GPU node in the serving image.
#   experiments/workload-shape/submit-shape.sh RUN_DIR
# SHAPE_ACCOUNT (d_research_serving), SHAPE_PARTITION (normal), SHAPE_TIME (02:00:00);
# LIMIT and STAGES pass through; so do the target overrides of run-shape.sh (TARGET,
# APERTUS_EAGLE_CONTRACT, EAGLE_HEAD, DSPARK, CORPUS_FROM, SHAPE_ARGS, AGG_ARGS, TREE_PYDEPS,
# HF_HOME). SHAPE_ENV_SOURCE picks the environment toml (default: the Apertus serving image).
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "${REPO_ROOT}"
RUN_DIR="$1"
ENV_SOURCE="${SHAPE_ENV_SOURCE:-${REPO_ROOT}/methods/eagle/configs/train-env.toml}"
SML_ENVIRONMENT="$("${REPO_ROOT}/serving/resolve-env.sh" "${ENV_SOURCE}")"
sed -i "s|^workdir = \".*\"|workdir = \"${REPO_ROOT}\"|" "${SML_ENVIRONMENT}"
IMAGE="$(sed -n 's/^image = "\(.*\)"/\1/p' "${SML_ENVIRONMENT}")"
mkdir -p "${RUN_DIR}/logs"
cp "${SML_ENVIRONMENT}" "${RUN_DIR}/logs/environment.toml"
SBATCH_FILE="${RUN_DIR}/logs/submit-$(date -u +%Y%m%dT%H%M%SZ).sbatch"
cat > "${SBATCH_FILE}" <<SBATCH
#!/bin/bash
#SBATCH --job-name=${SHAPE_JOB_NAME:-workload-shape}
#SBATCH --account=${SHAPE_ACCOUNT:-d_research_serving}
#SBATCH --partition=${SHAPE_PARTITION:-normal}
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=288
#SBATCH --gpus-per-node=4
#SBATCH --time=${SHAPE_TIME:-02:00:00}
#SBATCH --exclusive
#SBATCH --environment=${SML_ENVIRONMENT}
#SBATCH --output=${RUN_DIR}/logs/%j.out
#SBATCH --chdir=${REPO_ROOT}
set -euo pipefail
export SHAPE_IMAGE=${IMAGE}
export LIMIT=${LIMIT:-0} STAGES="${STAGES:-inputs capture shape aggregate}"
$(for v in TARGET APERTUS_EAGLE_CONTRACT EAGLE_HEAD DSPARK CORPUS_FROM SHAPE_ARGS AGG_ARGS TREE_PYDEPS HF_HOME MAX_MODEL_LEN; do
    [ -n "${!v:-}" ] && printf 'export %s=%q\n' "${v}" "${!v}"; done)
echo "START \$(date -u -Iseconds) host=\$(hostname) job=\${SLURM_JOB_ID} image=${IMAGE}"
bash experiments/workload-shape/run-shape.sh ${RUN_DIR}
echo "END \$(date -u -Iseconds)"
SBATCH
bash -n "${SBATCH_FILE}"
echo "sbatch_file=${SBATCH_FILE}"
sbatch "${SBATCH_FILE}"
