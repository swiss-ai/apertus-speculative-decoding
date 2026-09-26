#!/bin/bash
#SBATCH --job-name=eagle-8b-e31-pilot
#SBATCH --account=infra01
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=288
#SBATCH --mem=450G
#SBATCH --time=01:30:00
#SBATCH --exclusive
#SBATCH --exclude=nid007129
#SBATCH --environment=/users/faruk_zahiragic/.sml/env_resolved_arm64_M5Iusy.toml
#SBATCH --output=/users/faruk_zahiragic/apertus-bench-repo/results/eagle/8b/logs/%j.out
#SBATCH --error=/users/faruk_zahiragic/apertus-bench-repo/results/eagle/8b/logs/%j.err
#SBATCH --chdir=/users/faruk_zahiragic/apertus-bench-repo

#SBATCH --kill-on-invalid-dep=yes

set -euo pipefail
export I_AM_ON_AN_ALLOCATED_GPU_JOB=1
export EAGLE_TRAIN_CONFIG=/users/faruk_zahiragic/apertus-bench-repo/configs/eagle/8b/train-e31.yaml
export EAGLE_CUDA_DEVICES=0
export EAGLE_STEPS=verify
export EAGLE_RUN_ID=3508057
export TORCHSPEC_ROOT=/users/faruk_zahiragic/apertus-bench-repo/scratch/TorchSpec
export EAGLE_PYDEPS=/users/faruk_zahiragic/apertus-bench-repo/scratch/pydeps
export HF_HOME=/iopsstor/scratch/cscs/${USER}/hf_home
export TMPDIR=/tmp
export WANDB_MODE=offline
export WANDB_DISABLED=true
export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1
# No percent-strftime in this body: pyxis --environment sbatch mangled one (job 3491998).
echo "START $(date -u -Iseconds) host=$(hostname) job=${SLURM_JOB_ID}"
bash /users/faruk_zahiragic/apertus-bench-repo/launch/train-eagle-on-node.sh
echo "END $(date -u -Iseconds)"
