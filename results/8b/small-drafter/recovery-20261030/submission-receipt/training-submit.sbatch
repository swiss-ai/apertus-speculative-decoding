#!/bin/bash
#SBATCH --job-name=dspark-8b-depth-l1-warm-recovery-20261029
#SBATCH --account=infra01
#SBATCH --partition=normal
#SBATCH --time=06:00:00
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=288
#SBATCH --exclusive
#SBATCH --exclude=nid007129,nid007593,nid007611
#SBATCH --chdir=/iopsstor/scratch/cscs/faruk_zahiragic/spec-exp/small-drafter
#SBATCH --output=/iopsstor/scratch/cscs/faruk_zahiragic/spec-exp/small-drafter/results/8b/dspark/logs/%j.out
set -euo pipefail
export REPO_ROOT=/iopsstor/scratch/cscs/faruk_zahiragic/spec-exp/small-drafter
export DSPARK_DIR=/iopsstor/scratch/cscs/faruk_zahiragic/spec-exp-data/small-drafter
export RUN_ID=depth-l1-warm-recovery-20261029
export DATA=/iopsstor/scratch/cscs/faruk_zahiragic/spec-exp-data/small-drafter/data
export EPOCHS=1
export MAX_STEPS=6200
export WORKERS=8
export PREFETCH=4
export CKPT_FREQ=0.0625
export EXTRACT_MEM=0.85
export IN=/capstor/store/cscs/swissai/infra01/speculative-decoding/dspark-opb-thinking-off/open-perfectblend_apertus15-8b_thinking-off_seed0.jsonl
export TARGET_MODEL=/capstor/store/cscs/swissai/infra01/hf_models/models/swiss-ai/Apertus-v1.5-8B
export LAYERS=1\ 8\ 15\ 22\ 29
export SPECULATORS=/iopsstor/scratch/cscs/faruk_zahiragic/spec-exp-data/small-drafter/speculators
export NUM_LAYERS=1
export FSDP=0
export FROM_PRETRAINED=/iopsstor/scratch/cscs/faruk_zahiragic/spec-exp-data/small-drafter/drafters/dspark-e10-trunc-l1-shared-verifier-20261029
export EXTRACT_PER_RANK=1
export CHECK_SPLIT=1
export DSPARK_VALIDATE_WARM_START=1
echo "START $(date -u -Iseconds) job=${SLURM_JOB_ID} nodes=${SLURM_JOB_NODELIST} git=fb8f5c79"
srun --nodes=1 --ntasks=1 --environment=/iopsstor/scratch/cscs/faruk_zahiragic/spec-exp-data/small-drafter/env-20261029/env_resolved_arm64_LR8KMi.toml python3 /iopsstor/scratch/cscs/faruk_zahiragic/spec-exp/small-drafter/methods/dspark/train/rebind_verifier.py verify --initializer /iopsstor/scratch/cscs/faruk_zahiragic/spec-exp-data/small-drafter/drafters/dspark-e10-trunc-l1-shared-verifier-20261029 --contract /iopsstor/scratch/cscs/faruk_zahiragic/spec-exp/small-drafter/targets/8b/contract.json --image /capstor/store/cscs/swissai/infra01/container-images/ci/vllm_apertus_1.5_release-arm64.sqsh --source-revision fb8f5c795c90ab23c20fdb866c09174818949e2b --output /iopsstor/scratch/cscs/faruk_zahiragic/spec-exp-data/small-drafter/runs/depth-l1-warm-recovery-20261029/provenance/warm-start-${SLURM_JOB_ID}.json
export DSPARK_STOP_AT=$(( $(date +%s) + 21600 - 2400 ))
export HEAD_NODE=$(scontrol show hostnames "${SLURM_JOB_NODELIST}" | head -1)
srun --ntasks-per-node=1 --environment=/iopsstor/scratch/cscs/faruk_zahiragic/spec-exp-data/small-drafter/env-20261029/env_resolved_arm64_LR8KMi.toml bash methods/dspark/train/train-node.sh
