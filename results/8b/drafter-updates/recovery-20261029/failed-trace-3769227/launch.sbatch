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
#SBATCH --environment=/users/faruk_zahiragic/.sml/env_resolved_arm64_Rf409V.toml
#SBATCH --output=/iopsstor/scratch/cscs/faruk_zahiragic/spec-exp/drafter-updates/results/8b/eagle/logs/%j.out
#SBATCH --error=/iopsstor/scratch/cscs/faruk_zahiragic/spec-exp/drafter-updates/results/8b/eagle/logs/%j.err
#SBATCH --chdir=/iopsstor/scratch/cscs/faruk_zahiragic/spec-exp/drafter-updates
set -euo pipefail
export PYTHONPATH=/iopsstor/scratch/cscs/faruk_zahiragic/spec-exp/drafter-updates/methods/eagle:/iopsstor/scratch/cscs/faruk_zahiragic/spec-exp/drafter-updates/src
export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1
echo "START $(date -u -Iseconds) host=$(hostname) job=${SLURM_JOB_ID}"
env DRAFTER_DATA_ROOT=/iopsstor/scratch/cscs/faruk_zahiragic/spec-exp-data/drafter-updates/recovery-20261027/data bash methods/eagle/launch/drafter-traces.sh 
echo "END $(date -u -Iseconds)"
