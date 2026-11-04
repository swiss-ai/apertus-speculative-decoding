#!/bin/bash
#SBATCH --job-name=tree-drafters
#SBATCH --account=infra01
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=288
#SBATCH --gpus-per-node=4
#SBATCH --time=03:00:00
#SBATCH --exclusive
#SBATCH --environment=/users/faruk_zahiragic/.sml/env_resolved_arm64_zrO0ge.toml
#SBATCH --output=/iopsstor/scratch/cscs/faruk_zahiragic/spec-exp-data/tree-drafters/runs/full-20261024T1000Z/logs/%j.out
#SBATCH --chdir=/iopsstor/scratch/cscs/faruk_zahiragic/spec-exp/tree-drafters
set -euo pipefail
export TREE_IMAGE=/capstor/store/cscs/swissai/infra01/container-images/ci/vllm_apertus_1.5_release-arm64.sqsh
export LIMIT=0 MAX_ANCHORS=0 DSPARK_HF_LAYERS=0,7,14,21,28
export STAGES="candidates analyze"
echo "START $(date -u -Iseconds) host=$(hostname) job=${SLURM_JOB_ID} image=/capstor/store/cscs/swissai/infra01/container-images/ci/vllm_apertus_1.5_release-arm64.sqsh"
bash methods/tree/run-candidates.sh /iopsstor/scratch/cscs/faruk_zahiragic/spec-exp-data/tree-drafters/runs/full-20261024T1000Z
echo "END $(date -u -Iseconds)"
