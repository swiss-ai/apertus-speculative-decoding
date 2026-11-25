#!/bin/bash
# Serve Apertus-v1.5-70B with stock ROCm vLLM + the apertus15_text plugin on
# one Beverin MI300A node and check a short greedy generation.
#   [TP=2 GPUS=0,1 STAGES="serve extract"] sbatch methods/dspark/beverin/smoke.sbatch
#SBATCH -A infra01 -p mi300 -N 1 --gpus-per-node=4 -t 01:00:00 --job-name=bev-smoke
#SBATCH -o /iopsstor/scratch/cscs/faruk_zahiragic/beverin-feasibility/logs/%x-%j.out
set -euo pipefail
export REPO_ROOT="${SLURM_SUBMIT_DIR}"
EDF="${EDF:-${REPO_ROOT}/methods/dspark/beverin/env-vllm023.toml}"
echo "job ${SLURM_JOB_ID} node $(hostname) TP=${TP:-2} GPUS=${GPUS:-0,1} EDF=${EDF}"
echo "cmd: $0 $*"; cat "${REPO_ROOT}/REVISION" 2>/dev/null || true
# STAGES: serve (vllm serve + greedy generation), extract (launch_vllm.py train
# + check_hidden_states.py); one container start each.
for stage in ${STAGES:-serve extract}; do
  case "${stage}" in
    serve) srun --environment="${EDF}" --export=ALL bash "${REPO_ROOT}/methods/dspark/beverin/serve-smoke.sh" ;;
    extract) srun --environment="${EDF}" --export=ALL bash "${REPO_ROOT}/methods/dspark/beverin/extract-check.sh" ;;
  esac || { echo "stage ${stage} failed"; exit 1; }
done
