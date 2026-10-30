#!/bin/bash
#SBATCH --job-name=eagle-node-command
#SBATCH --account=infra01
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=288
#SBATCH --time=03:00:00
#SBATCH --exclusive
#SBATCH --gpus-per-node=4
#SBATCH --environment=/iopsstor/scratch/cscs/faruk_zahiragic/spec-exp-data/drafter-updates/recovery-20261030/receipts/trace-environments/env_resolved_arm64_9GYLhL.toml
#SBATCH --output=/iopsstor/scratch/cscs/faruk_zahiragic/spec-exp-data/drafter-updates/recovery-20261030/runtime/source-e9cc4842dc59/results/8b/eagle/logs/%j.out
#SBATCH --error=/iopsstor/scratch/cscs/faruk_zahiragic/spec-exp-data/drafter-updates/recovery-20261030/runtime/source-e9cc4842dc59/results/8b/eagle/logs/%j.err
#SBATCH --chdir=/iopsstor/scratch/cscs/faruk_zahiragic/spec-exp-data/drafter-updates/recovery-20261030/runtime/source-e9cc4842dc59
set -euo pipefail
export PYTHONPATH=/iopsstor/scratch/cscs/faruk_zahiragic/spec-exp-data/drafter-updates/recovery-20261030/runtime/source-e9cc4842dc59/methods/eagle:/iopsstor/scratch/cscs/faruk_zahiragic/spec-exp-data/drafter-updates/recovery-20261030/runtime/source-e9cc4842dc59/src
export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1
echo "START $(date -u -Iseconds) host=$(hostname) job=${SLURM_JOB_ID}"
env DRAFTER_DATA_ROOT=/iopsstor/scratch/cscs/faruk_zahiragic/spec-exp-data/drafter-updates/recovery-20261030/data DRAFTER_DOMAINS=chat bash methods/eagle/launch/drafter-traces.sh 
echo "END $(date -u -Iseconds)"
