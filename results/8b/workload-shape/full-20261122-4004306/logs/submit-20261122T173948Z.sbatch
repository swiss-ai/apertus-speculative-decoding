#!/bin/bash
#SBATCH --job-name=workload-shape
#SBATCH --account=d_research_serving
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=288
#SBATCH --gpus-per-node=4
#SBATCH --time=02:30:00
#SBATCH --exclusive
#SBATCH --environment=/iopsstor/scratch/cscs/faruk_zahiragic/workload-shape/sml-env/env_resolved_arm64_oRA4Ar.toml
#SBATCH --output=/iopsstor/scratch/cscs/faruk_zahiragic/workload-shape/runs/full-20261122/logs/%j.out
#SBATCH --chdir=/iopsstor/scratch/cscs/faruk_zahiragic/workload-shape/code-f87ab168
set -euo pipefail
export SHAPE_IMAGE=/capstor/store/cscs/swissai/infra01/container-images/ci/vllm_apertus_1.5_release-arm64.sqsh
export LIMIT=0 STAGES="inputs capture shape aggregate"
echo "START $(date -u -Iseconds) host=$(hostname) job=${SLURM_JOB_ID} image=/capstor/store/cscs/swissai/infra01/container-images/ci/vllm_apertus_1.5_release-arm64.sqsh"
bash experiments/workload-shape/run-shape.sh /iopsstor/scratch/cscs/faruk_zahiragic/workload-shape/runs/full-20261122
echo "END $(date -u -Iseconds)"
