#!/usr/bin/env bash
#SBATCH --job-name=drafter-kv-preflight
#SBATCH --account=infra01
#SBATCH --partition=debug
#SBATCH --time=00:20:00
#SBATCH --nodes=1
#SBATCH --exclusive
set -euo pipefail

REPO="${1:?isolated checkout}"
CAMPAIGN="${2:?campaign directory}"
OUT="${3:?reserved preflight output directory}"
ENVIRONMENT="${4:?resolved container environment}"
mkdir -p "${OUT}"
cp "${ENVIRONMENT}" "${OUT}/environment.toml"
cd "${REPO}"
git rev-parse HEAD > "${OUT}/code-revision.txt"
git diff HEAD --binary > "${OUT}/code.patch"
scontrol show job "${SLURM_JOB_ID}" > "${OUT}/slurm-job.txt"

srun --nodes=1 --ntasks=1 --environment="${ENVIRONMENT}" \
  bash "${REPO}/experiments/drafter-kv-8b/preflight-on-node.sh" "${REPO}" "${CAMPAIGN}" "${OUT}"
