#!/usr/bin/env bash
# Submit a node sweep: up to four deployments side by side on one 4-GPU node,
# one per GPU, load-tested at the same time (src/apertus_bench/node_sweep.py).
#   ./serving/node-sweep.sh experiments/system-8b/k-sweep.yaml
# Results go to results/<stage>/sweeps/<spec>-<stamp>/<arm>/, in the layout of
# serving/loadtest.sh (cells per workload, engine excerpt, status).
# Every arm runs with the same patched image as the launchers: the EAGLE and
# DSpark fixes in serving/patches are all applied, plus any SWEEP_PATCHES.
# Only submits; check the job with squeue and the output directory.
set -euo pipefail

SERVING_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "${SERVING_DIR}/.." && pwd)"
SPEC="${1:?usage: node-sweep.sh SPEC.yaml}"
SPEC="$(cd "$(dirname "${SPEC}")" && pwd)/$(basename "${SPEC}")"
STAGE="$(sed -n 's/^stage: *//p' "${SPEC}" | head -1)"
STAGE="${STAGE:-8b}"
NAME="$(basename "${SPEC}" .yaml)"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
OUT="${SWEEP_OUTPUT:-${REPO_ROOT}/results/${STAGE}/sweeps/${NAME}-${STAMP}}"
PARTITION="${SWEEP_PARTITION:-debug}"
TIME_LIMIT="${SWEEP_TIME:-01:30:00}"
ACCOUNT="${SWEEP_ACCOUNT:-infra01}"
IMAGE="${IMAGE:-/capstor/store/cscs/swissai/infra01/container-images/ci/vllm_apertus_1.5_release-arm64.sqsh}"
# This environment mounts /users, where the checkout and the bench client live.
ENV_SOURCE="${ENV_SOURCE:-${REPO_ROOT}/methods/eagle/configs/train-env.toml}"
# One overlay directory per sweep: patch-vllm.sh empties it, and a running
# sweep or launcher may still be reading another one.
export VLLM_PATCH_DIR="${VLLM_PATCH_DIR:-${HOME}/.sml/vllm-patch-sweep-${STAMP}}"
PATCHES=(
  "${REPO_ROOT}/serving/patches/vllm-apertus-image-token.patch"
  "${REPO_ROOT}/serving/patches/vllm-apertus-eagle3-aux-layers.patch"
  "${REPO_ROOT}/serving/patches/vllm-apertus-eagle-mrv2-inner.patch"
  "${REPO_ROOT}/serving/patches/vllm-apertus-dspark-mrv2-inner.patch"
  "${REPO_ROOT}/serving/patches/vllm-apertus-dspark-anchor-layout.patch"
)
for extra in ${SWEEP_PATCHES:-}; do
  PATCHES+=("$(cd "$(dirname "${extra}")" && pwd)/$(basename "${extra}")")
done

# Fail on a bad spec here, not after the queue wait.
PYTHONPATH="${REPO_ROOT}/src" python3 -m apertus_bench.node_sweep plan "${SPEC}" > /dev/null

mkdir -p "${OUT}"
if [ "${VALIDATE_ONLY:-0}" != "1" ]; then
  EXTRA_MOUNTS="$("${SERVING_DIR}/patch-vllm.sh" "${IMAGE}" "${PATCHES[@]}")"
  export EXTRA_MOUNTS
fi
SML_ENVIRONMENT="$("${SERVING_DIR}/resolve-env.sh" "${ENV_SOURCE}")"
CACHE_ROOT="/iopsstor/scratch/cscs/${USER}/apertus-loadtest/vllm-cache/sweep-${NAME}-${STAMP}"
printf '%s\n' "${PATCHES[@]}" > "${OUT}/patches.txt"
cp "${SML_ENVIRONMENT}" "${OUT}/environment.toml"

cat > "${OUT}/job.sh" <<EOF
#!/bin/bash
#SBATCH --job-name=sweep-${NAME}
#SBATCH --account=${ACCOUNT}
#SBATCH --partition=${PARTITION}
#SBATCH --time=${TIME_LIMIT}
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --output=${OUT}/job.out
set -euo pipefail
# GPU memory and power for the whole job, on the host.
srun --overlap --nodes=1 --ntasks=1 nvidia-smi \
  --query-gpu=timestamp,index,memory.used,memory.total,utilization.gpu,power.draw \
  --format=csv,noheader,nounits -lms 1000 > "${OUT}/gpu-memory.csv" 2> /dev/null &
SAMPLER=\$!
# The image's PYTHONPATH puts /workspace/vllm (where the patch overlays are
# mounted) ahead of the installed copy: extend it inside the container, never
# replace it, or vLLM runs unpatched.
srun --overlap --nodes=1 --ntasks=1 --environment="${SML_ENVIRONMENT}" \
  env PYTHONNOUSERSITE=1 SWEEP_CACHE_ROOT="${CACHE_ROOT}" bash -c \
  'export PYTHONPATH="${REPO_ROOT}/src:\${PYTHONPATH:-}"; exec python3 -m apertus_bench.node_sweep run "${SPEC}" --output "${OUT}"' \
  || status=\$?
kill \${SAMPLER} 2> /dev/null || true
exit \${status:-0}
EOF

if [ "${VALIDATE_ONLY:-0}" = "1" ]; then
  echo "validate_only=1; job script in ${OUT}/job.sh"
  exit 0
fi
JOB="$(sbatch --parsable "${OUT}/job.sh")"
echo "job=${JOB}" > "${OUT}/job.txt"
echo "job=${JOB} output=${OUT}"
