#!/usr/bin/env bash
# Submit the domain-drafters trace chain from a cluster checkout:
# one CPU pools job, then generation jobs (afterok) with three domains each.
#   methods/eagle/launch/domain-traces-submit.sh            # pools + 2 generation nodes
#   DOMAIN_SKIP_POOLS=1 ...                                 # pools already staged
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

dep=()
if [ -z "${DOMAIN_SKIP_POOLS:-}" ]; then
  pools_file="$(write_job pools pools "" 0 "${DOMAIN_POOLS_TIME:-01:30:00}")"
  pools_id="$(sbatch --parsable "${pools_file}")"
  echo "pools job ${pools_id}"
  dep=(--dependency="afterok:${pools_id}")
fi
for index in "${!GROUPS_[@]}"; do
  list="${GROUPS_[$index]}"
  gpus=$(wc -w <<< "${list}")
  file="$(write_job "gen${index}" generate "${list}" 4 "${DOMAIN_TIME:-04:00:00}")"
  id="$(sbatch --parsable ${dep[@]+"${dep[@]}"} "${file}")"
  echo "generate job ${id}: ${list} (${gpus} GPUs used)"
done
