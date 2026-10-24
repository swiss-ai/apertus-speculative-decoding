#!/bin/bash
# DSpark 70B training smoke on one Beverin node: the Clariden train-node.sh
# (TP2 extraction server on GPUs 0,1, two FSDP trainer ranks on GPUs 2,3) in the
# ROCm vLLM image, MAX_STEPS steps on the prepared 70B corpus.
#   [MAX_STEPS=30 SEQ=4096 ANCHORS=256] sbatch methods/dspark/beverin/train-smoke.sbatch
#SBATCH -A infra01 -p mi300 -N 1 --gpus-per-node=4 -t 01:30:00 --job-name=bev-train-smoke
#SBATCH -o /iopsstor/scratch/cscs/faruk_zahiragic/beverin-feasibility/logs/%x-%j.out
set -euo pipefail
export REPO_ROOT="${SLURM_SUBMIT_DIR}"
WORK=/iopsstor/scratch/cscs/${USER}/beverin-feasibility
EDF="${EDF:-${REPO_ROOT}/methods/dspark/beverin/env-vllm023.toml}"
export DSPARK_DIR="${WORK}" SPECULATORS="${WORK}/speculators"
export RUN_ID="${RUN_ID:-train-smoke-${SLURM_JOB_ID}}"
export DATA="${DATA:-/iopsstor/scratch/cscs/${USER}/apertus-dspark70b/data}"
export HEAD_NODE=$(scontrol show hostnames "${SLURM_JOB_NODELIST}" | head -1)
export MAX_STEPS="${MAX_STEPS:-30}" SEQ="${SEQ:-4096}" ANCHORS="${ANCHORS:-256}" CKPT_FREQ="${CKPT_FREQ:-1}"
export EXTRACT_MEM="${EXTRACT_MEM:-0.75}" EXTRACT_FLAGS="${EXTRACT_FLAGS---no-enable-chunked-prefill --no-enable-prefix-caching}"
export PY_PRELUDE="${REPO_ROOT}/methods/dspark/beverin/apertus15_run.py"
export PYTHONPATH="${REPO_ROOT}/methods/dspark/beverin"
export DSPARK_STOP_AT=$(( $(date +%s) + 5400 - 900 ))
echo "job ${SLURM_JOB_ID} node $(hostname) run ${RUN_ID} steps ${MAX_STEPS} seq ${SEQ} anchors ${ANCHORS}"
cat "${REPO_ROOT}/REVISION" 2>/dev/null || true
srun --environment="${EDF}" --export=ALL bash -c '
  . "${REPO_ROOT}/methods/dspark/beverin/env.sh"
  ( while sleep 60; do echo "$(date -u +%FT%TZ) $(free -g | awk "/Mem:/ {print \"used_gb=\" \$3 \" avail_gb=\" \$7}")"; done ) \
    > "${WORK}/logs/mem-${SLURM_JOB_ID}.log" &
  bash "${REPO_ROOT}/methods/dspark/train/train-node.sh"'
