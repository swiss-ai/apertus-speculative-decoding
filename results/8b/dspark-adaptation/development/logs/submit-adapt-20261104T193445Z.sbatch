#!/bin/bash
#SBATCH --job-name=dspark-8b-adapt
#SBATCH --account=infra01
#SBATCH --partition=debug
#SBATCH --time=01:30:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=288
#SBATCH --exclusive
#SBATCH --exclude=nid007129,nid007593,nid007611
#SBATCH --chdir=/iopsstor/scratch/cscs/faruk_zahiragic/spec-exp/dspark-adaptation
#SBATCH --output=/iopsstor/scratch/cscs/faruk_zahiragic/spec-exp-data/dspark-adaptation/adapt-mult-n8k-r0-20261104/logs/%j.out
set -euo pipefail
export REPO_ROOT=/iopsstor/scratch/cscs/faruk_zahiragic/spec-exp/dspark-adaptation
export DSPARK_DIR=/iopsstor/scratch/cscs/faruk_zahiragic/apertus-dspark8b
export WORKERS=8
export PREFETCH=4
export CKPT_FREQ=0.0625
export EXTRACT_MEM=0.85
export IN=/capstor/store/cscs/swissai/infra01/speculative-decoding/dspark-opb-thinking-off/open-perfectblend_apertus15-8b_thinking-off_seed0.jsonl
export TARGET_MODEL=/capstor/store/cscs/swissai/infra01/hf_models/models/swiss-ai/Apertus-v1.5-8B
export LAYERS=1\ 8\ 15\ 22\ 29
export FSDP=0
export EXTRACT_PER_RANK=1
export CHECK_SPLIT=1
export ADAPT_ROOT=/iopsstor/scratch/cscs/faruk_zahiragic/spec-exp-data/dspark-adaptation/adapt-mult-n8k-r0-20261104
echo "START $(date -u -Iseconds) job=${SLURM_JOB_ID} nodes=${SLURM_JOB_NODELIST} git=720d3624"
export ADAPT_DEADLINE=$(( $(date +%s) + 5400 - 300 ))
srun --ntasks-per-node=1 --environment=/iopsstor/scratch/cscs/faruk_zahiragic/spec-exp-data/dspark-adaptation/adapt-mult-n8k-r0-20261104/env/env_resolved_arm64_oxJR8G.toml bash methods/dspark/train/adapt-job.sh
