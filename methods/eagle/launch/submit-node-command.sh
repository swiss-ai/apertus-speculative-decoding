#!/usr/bin/env bash
# Run one command inside the serving image on a compute node (never on the login node).
#   ./methods/eagle/launch/submit-node-command.sh python3 -m apertus_eagle.select_long_rows ...
# NODE_CMD_PARTITION (debug), NODE_CMD_TIME (00:30:00), NODE_CMD_GPUS (0).
set -euo pipefail

LAUNCH_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "${LAUNCH_DIR}/../../.." && pwd)"
cd "${REPO_ROOT}"
[ "$#" -gt 0 ] || { echo "usage: $0 command [args...]" >&2; exit 2; }
ENV_SOURCE="${EAGLE_ENV_SOURCE:-${REPO_ROOT}/methods/eagle/configs/train-env.toml}"
SML_ENVIRONMENT="$("${REPO_ROOT}/serving/resolve-env.sh" "${ENV_SOURCE}")"
sed -i "s|^workdir = \".*\"|workdir = \"${REPO_ROOT}\"|" "${SML_ENVIRONMENT}"
LOG_ROOT="${REPO_ROOT}/results/8b/eagle/logs"
mkdir -p "${LOG_ROOT}"
SBATCH_FILE="${LOG_ROOT}/submit-node-command-$(date -u +%Y%m%dT%H%M%SZ).sbatch"
GPUS="${NODE_CMD_GPUS:-0}"
cat > "${SBATCH_FILE}" <<SBATCH
#!/bin/bash
#SBATCH --job-name=eagle-node-command
#SBATCH --account=${EAGLE_TRAIN_ACCOUNT:-infra01}
#SBATCH --partition=${NODE_CMD_PARTITION:-debug}
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=288
#SBATCH --time=${NODE_CMD_TIME:-00:30:00}
#SBATCH --exclusive
$([ "${GPUS}" -gt 0 ] && echo "#SBATCH --gpus-per-node=${GPUS}")
#SBATCH --environment=${SML_ENVIRONMENT}
#SBATCH --output=${LOG_ROOT}/%j.out
#SBATCH --error=${LOG_ROOT}/%j.err
#SBATCH --chdir=${REPO_ROOT}
set -euo pipefail
export PYTHONPATH=${REPO_ROOT}/methods/eagle:${REPO_ROOT}/src
export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1
echo "START \$(date -u -Iseconds) host=\$(hostname) job=\${SLURM_JOB_ID}"
$(printf '%q ' "$@")
echo "END \$(date -u -Iseconds)"
SBATCH
bash -n "${SBATCH_FILE}"
echo "sbatch_file=${SBATCH_FILE}"
sbatch "${SBATCH_FILE}"
