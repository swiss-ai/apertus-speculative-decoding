#!/usr/bin/env bash
# Submit a node sweep (src/apertus_bench/node_sweep.py) on one Beverin MI300A
# node, inside the ALPS vLLM image with the apertus15_text plugin. Run from a
# packaged runtime checkout (${WORK}/repo-<sha>) on the Beverin login node.
#   methods/dspark/beverin/submit-sweep.sh experiments/system-70b/SPEC.yaml
# With EXPORT_SRC and EXPORT_DEST set, the job first exports that checkpoint
# (methods/dspark/train/export-head.sh) unless EXPORT_DEST already exists.
# Results go to results/<stage>/sweeps/<spec>-beverin-<stamp>/ in this checkout.
set -euo pipefail
BEV_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "${BEV_DIR}/../../.." && pwd)"
SPEC="$(cd "$(dirname "${1:?usage: submit-sweep.sh SPEC.yaml}")" && pwd)/$(basename "$1")"
STAGE="$(sed -n 's/^stage: *//p' "${SPEC}" | head -1)"
NAME="$(basename "${SPEC}" .yaml)"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
OUT="${SWEEP_OUTPUT:-${REPO_ROOT}/results/${STAGE:-8b}/sweeps/${NAME}-beverin-${STAMP}}"
EDF="${EDF:-${BEV_DIR}/env-vllm-alps.toml}"
mkdir -p "${OUT}"
cp "${EDF}" "${OUT}/environment.toml"
git -C "${REPO_ROOT}" rev-parse HEAD > "${OUT}/repo-revision.txt" 2> /dev/null \
  || cat "${REPO_ROOT}/REVISION" > "${OUT}/repo-revision.txt"

cat > "${OUT}/job.sh" <<EOF
#!/bin/bash
#SBATCH --job-name=sweep-${NAME}
#SBATCH --account=${SWEEP_ACCOUNT:-infra01}
#SBATCH --partition=${SWEEP_PARTITION:-mi300}
#SBATCH --time=${SWEEP_TIME:-05:00:00}
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --output=${OUT}/job.out
set -euo pipefail
echo "START \$(date -u -Iseconds) job=\${SLURM_JOB_ID} node=\${SLURM_JOB_NODELIST}"
if [ -n "${EXPORT_SRC:-}" ] && [ ! -e "${EXPORT_DEST:-}" ]; then
  bash "${REPO_ROOT}/methods/dspark/train/export-head.sh" "${EXPORT_SRC:-}" "${EXPORT_DEST:-}"
fi
srun --nodes=1 --ntasks=1 --environment="${EDF}" bash -c '
  export REPO_ROOT="${REPO_ROOT}"
  . "${BEV_DIR}/env.sh"
  export PYTHONPATH="${REPO_ROOT}/src:\${PYTHONPATH}" SWEEP_CACHE_ROOT=/tmp/bev-cache/sweep
  python3 -c "import vllm, torch; print(\"vllm\", vllm.__version__, \"torch\", torch.__version__)"
  exec python3 -m apertus_bench.node_sweep run "${SPEC}" --output "${OUT}"'
EOF
JOB="$(sbatch --parsable "${OUT}/job.sh")"
echo "job=${JOB}" > "${OUT}/job.txt"
echo "job=${JOB} output=${OUT}"
