#!/bin/bash
#SBATCH --job-name=sweep-dspark-alps-1epoch
#SBATCH --account=infra01
#SBATCH --partition=mi300
#SBATCH --time=05:00:00
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --output=/iopsstor/scratch/cscs/faruk_zahiragic/beverin-feasibility/repo-277edd39/results/70b/sweeps/dspark-alps-1epoch-beverin-20261104T185916Z/job.out
set -euo pipefail
echo "START $(date -u -Iseconds) job=${SLURM_JOB_ID} node=${SLURM_JOB_NODELIST}"
if [ -n "" ] && [ ! -e "" ]; then
  bash "/iopsstor/scratch/cscs/faruk_zahiragic/beverin-feasibility/repo-277edd39/methods/dspark/train/export-head.sh" "" "" ""
fi
EDF_RUN="/iopsstor/scratch/cscs/faruk_zahiragic/beverin-feasibility/repo-277edd39/methods/dspark/beverin/env-vllm-alps.toml"
if [ -n "/iopsstor/scratch/cscs/faruk_zahiragic/beverin-feasibility/repo-277edd39/serving/patches/vllm031-dspark-apertus-pruned-head.patch " ]; then
  # Patched copies of the image's vLLM files, bind-mounted over the originals.
  OVERLAY="/iopsstor/scratch/cscs/faruk_zahiragic/beverin-feasibility/repo-277edd39/results/70b/sweeps/dspark-alps-1epoch-beverin-20261104T185916Z/vllm-overlay"
  mkdir -p "${OVERLAY}"
  files=$(cat /iopsstor/scratch/cscs/faruk_zahiragic/beverin-feasibility/repo-277edd39/serving/patches/vllm031-dspark-apertus-pruned-head.patch  | sed -n 's|^+++ b/||p' | sort -u | tr '\n' ' ')
  srun --nodes=1 --ntasks=1 --environment="/iopsstor/scratch/cscs/faruk_zahiragic/beverin-feasibility/repo-277edd39/methods/dspark/beverin/env-vllm-alps.toml" bash -c "cd /opt/venv/lib/python3.12/site-packages && cp --parents ${files} ${OVERLAY}/"
  for p in /iopsstor/scratch/cscs/faruk_zahiragic/beverin-feasibility/repo-277edd39/serving/patches/vllm031-dspark-apertus-pruned-head.patch ; do patch -d "${OVERLAY}" -p1 < "${p}"; done
  EDF_RUN="/iopsstor/scratch/cscs/faruk_zahiragic/beverin-feasibility/repo-277edd39/results/70b/sweeps/dspark-alps-1epoch-beverin-20261104T185916Z/environment-patched.toml"
  mounts=$(for f in ${files}; do printf ', "%s:%s"' "${OVERLAY}/${f}" "/opt/venv/lib/python3.12/site-packages/${f}"; done)
  sed "s|^mounts = \[\(.*\)\]|mounts = [\1${mounts}]|" "/iopsstor/scratch/cscs/faruk_zahiragic/beverin-feasibility/repo-277edd39/methods/dspark/beverin/env-vllm-alps.toml" > "${EDF_RUN}"
fi
srun --nodes=1 --ntasks=1 --environment="${EDF_RUN}" bash -c '
  export REPO_ROOT="/iopsstor/scratch/cscs/faruk_zahiragic/beverin-feasibility/repo-277edd39"
  . "/iopsstor/scratch/cscs/faruk_zahiragic/beverin-feasibility/repo-277edd39/methods/dspark/beverin/env.sh"
  export PYTHONPATH="/iopsstor/scratch/cscs/faruk_zahiragic/beverin-feasibility/repo-277edd39/src:${PYTHONPATH}" SWEEP_CACHE_ROOT=/tmp/bev-cache/sweep
  python3 -c "import vllm, torch; print(\"vllm\", vllm.__version__, \"torch\", torch.__version__)"
  exec python3 -m apertus_bench.node_sweep run "/iopsstor/scratch/cscs/faruk_zahiragic/beverin-feasibility/repo-277edd39/experiments/system-70b/dspark-alps-1epoch.yaml" --output "/iopsstor/scratch/cscs/faruk_zahiragic/beverin-feasibility/repo-277edd39/results/70b/sweeps/dspark-alps-1epoch-beverin-20261104T185916Z"'
