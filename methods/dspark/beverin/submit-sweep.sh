#!/usr/bin/env bash
# Submit a node sweep (src/apertus_bench/node_sweep.py) on one Beverin MI300A
# node, inside the ALPS vLLM image with the apertus15_text plugin. Run from a
# packaged runtime checkout (${WORK}/repo-<sha>) on the Beverin login node.
#   methods/dspark/beverin/submit-sweep.sh experiments/system-70b/SPEC.yaml
# With EXPORT_SRC and EXPORT_DEST set, the job first exports that checkpoint
# (methods/dspark/train/export-head.sh, hashes checked against EXPORT_EXPECTED)
# unless EXPORT_DEST already exists. VLLM_PATCHES (space-separated patch files
# against vllm/, e.g. serving/patches/vllm031-*.patch) are applied to copies of
# the image's files, mounted over the originals for every arm.
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
VLLM_PATCHES="$(for p in ${VLLM_PATCHES:-}; do printf '%s ' "$(cd "$(dirname "${p}")" && pwd)/$(basename "${p}")"; done)"
mkdir -p "${OUT}"
cp "${EDF}" "${OUT}/environment.toml"
for p in ${VLLM_PATCHES}; do cp "${p}" "${OUT}/"; done
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
  bash "${REPO_ROOT}/methods/dspark/train/export-head.sh" "${EXPORT_SRC:-}" "${EXPORT_DEST:-}" "${EXPORT_EXPECTED:-}"
fi
EDF_RUN="${EDF}"
if [ -n "${VLLM_PATCHES}" ]; then
  # Patched copies of the image's vLLM files, bind-mounted over the originals.
  OVERLAY="${OUT}/vllm-overlay"
  mkdir -p "\${OVERLAY}"
  files=\$(cat ${VLLM_PATCHES} | sed -n 's|^+++ b/||p' | sort -u | tr '\n' ' ')
  srun --nodes=1 --ntasks=1 --environment="${EDF}" bash -c "cd /opt/venv/lib/python3.12/site-packages && cp --parents \${files} \${OVERLAY}/"
  for p in ${VLLM_PATCHES}; do patch -d "\${OVERLAY}" -p1 < "\${p}"; done
  EDF_RUN="${OUT}/environment-patched.toml"
  mounts=\$(for f in \${files}; do printf ', "%s:%s"' "\${OVERLAY}/\${f}" "/opt/venv/lib/python3.12/site-packages/\${f}"; done)
  sed "s|^mounts = \[\(.*\)\]|mounts = [\1\${mounts}]|" "${EDF}" > "\${EDF_RUN}"
fi
srun --nodes=1 --ntasks=1 --environment="\${EDF_RUN}" bash -c '
  export REPO_ROOT="${REPO_ROOT}"
  . "${BEV_DIR}/env.sh"
  export PYTHONPATH="${REPO_ROOT}/src:\${PYTHONPATH}" SWEEP_CACHE_ROOT=/tmp/bev-cache/sweep
  python3 -c "import vllm, torch; print(\"vllm\", vllm.__version__, \"torch\", torch.__version__)"
  exec python3 -m apertus_bench.node_sweep run "${SPEC}" --output "${OUT}"'
EOF
JOB="$(sbatch --parsable "${OUT}/job.sh")"
echo "job=${JOB}" > "${OUT}/job.txt"
echo "job=${JOB} output=${OUT}"
