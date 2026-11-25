#!/usr/bin/env bash
# Submit the domain-drafters trace chain from a cluster checkout:
# a debug pilot job (pools + pilot slice), then generation jobs with three domains each.
#   methods/eagle/launch/domain-traces-submit.sh            # pilot + 2 generation nodes
#   DOMAIN_SKIP_PILOT=1 ...                                 # generation jobs only
# DOMAIN_PARTITION (normal), DOMAIN_TIME (04:00:00), DOMAIN_GROUPS ("math code tool;chat summarization multilingual").
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
cd "${REPO_ROOT}"
DATA_ROOT="${DOMAIN_DATA_ROOT:-/iopsstor/scratch/cscs/${USER}/spec-exp-data/domain-drafters}"
export SML_ENV_DIR="${SML_ENV_DIR:-${DATA_ROOT}/env}"
ENV_SOURCE="${REPO_ROOT}/methods/eagle/configs/train-env.toml"
ENV_FILE="$("${REPO_ROOT}/serving/resolve-env.sh" "${ENV_SOURCE}")"
sed -i "s|^workdir = \".*\"|workdir = \"${REPO_ROOT}\"|" "${ENV_FILE}"
LOGS="${DATA_ROOT}/logs"
mkdir -p "${LOGS}" "${DATA_ROOT}/submissions"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
IFS=';' read -r -a GROUPS_ <<< "${DOMAIN_GROUPS:-math code tool;chat summarization multilingual}"

write_job() {
  local name="$1" mode="$2" list="$3" gpus="$4" time="$5"
  local file="${DATA_ROOT}/submissions/${STAMP}-${name}.sbatch"
  cat > "${file}" <<SBATCH
#!/bin/bash
#SBATCH --job-name=domain-${name}
#SBATCH --account=${DOMAIN_ACCOUNT:-infra01}
#SBATCH --partition=${DOMAIN_PARTITION:-normal}
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=288
#SBATCH --time=${time}
#SBATCH --exclusive
$([ "${gpus}" -gt 0 ] && echo "#SBATCH --gpus-per-node=${gpus}")
#SBATCH --environment=${ENV_FILE}
#SBATCH --output=${LOGS}/%j-${name}.out
#SBATCH --error=${LOGS}/%j-${name}.err
#SBATCH --chdir=${REPO_ROOT}
set -euo pipefail
export PYTHONPATH=${REPO_ROOT}/methods/eagle:${REPO_ROOT}/src
export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1
export SBATCH_ENVIRONMENT_FILE=${ENV_FILE} DOMAIN_MODE=${mode} DOMAIN_LIST="${list}"
echo "START \$(date -u -Iseconds) host=\$(hostname) job=\${SLURM_JOB_ID}"
bash ${REPO_ROOT}/methods/eagle/launch/domain-traces.sh
echo "END \$(date -u -Iseconds)"
SBATCH
  bash -n "${file}"
  echo "${file}"
}

# Pilot (pools + 512/64 pilot slice of all six domains) on the debug partition,
# where a job starts quickly; full generation on normal, building the pools
# itself if the pilot job has not staged them yet.
if [ -z "${DOMAIN_SKIP_PILOT:-}" ]; then
  pilot_file="$(write_job pilot pilot "" 4 "${DOMAIN_PILOT_TIME:-01:30:00}")"
  sed -i "s|^#SBATCH --partition=.*|#SBATCH --partition=${DOMAIN_PILOT_PARTITION:-debug}|" "${pilot_file}"
  echo "pilot job $(sbatch --parsable "${pilot_file}")"
fi
for index in "${!GROUPS_[@]}"; do
  list="${GROUPS_[$index]}"
  file="$(write_job "gen${index}" generate "${list}" 4 "${DOMAIN_TIME:-04:00:00}")"
  echo "generate job $(sbatch --parsable "${file}"): ${list}"
done
