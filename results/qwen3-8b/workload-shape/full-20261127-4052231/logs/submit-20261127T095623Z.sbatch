#!/bin/bash
#SBATCH --job-name=shape-qwen3-8b
#SBATCH --account=d_research_serving
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=288
#SBATCH --gpus-per-node=4
#SBATCH --time=02:00:00
#SBATCH --exclusive
#SBATCH --environment=/iopsstor/scratch/cscs/faruk_zahiragic/workload-shape-qwen3-8b/sml-env/env_resolved_arm64_GKiypT.toml
#SBATCH --output=/iopsstor/scratch/cscs/faruk_zahiragic/workload-shape-qwen3-8b/runs/full-20261127/logs/%j.out
#SBATCH --chdir=/iopsstor/scratch/cscs/faruk_zahiragic/spec-exp/workload-shape-qwen3-8b/code-fabc7d1a73b0
set -euo pipefail
export SHAPE_IMAGE=/capstor/store/cscs/swissai/infra01/container-images/ci/vllm_nvidia_alps-arm64.sqsh
export LIMIT=0 STAGES="inputs capture shape aggregate"
export TARGET=/iopsstor/scratch/cscs/faruk_zahiragic/workload-shape-qwen3-8b/models/Qwen--Qwen3-8B
export APERTUS_EAGLE_CONTRACT=/iopsstor/scratch/cscs/faruk_zahiragic/spec-exp/workload-shape-qwen3-8b/code-fabc7d1a73b0/targets/qwen3-8b/contract.json
export EAGLE_HEAD=/iopsstor/scratch/cscs/faruk_zahiragic/workload-shape-qwen3-8b/models/RedHatAI--Qwen3-8B-speculator.eagle3
export DSPARK=/iopsstor/scratch/cscs/faruk_zahiragic/workload-shape-qwen3-8b/models/RedHatAI--Qwen3-8B-speculator.dflash
export CORPUS_FROM=/iopsstor/scratch/cscs/faruk_zahiragic/workload-shape/runs/full-20261122/inputs
export SHAPE_ARGS=--eagle-format\ speculators\ --block-format\ dflash\ --names\ dflash\,eagle3\ --depth\ 15\ --dspark-hf-layers\ 1\,9\,17\,25\,33
export AGG_ARGS=--sampled
export TREE_PYDEPS=/users/faruk_zahiragic/apertus-spec-decoding/scratch/pydeps:/iopsstor/scratch/cscs/faruk_zahiragic/workload-shape-qwen3-8b/pydeps
export HF_HOME=/iopsstor/scratch/cscs/faruk_zahiragic/workload-shape-qwen3-8b/hf_home
echo "START $(date -u -Iseconds) host=$(hostname) job=${SLURM_JOB_ID} image=/capstor/store/cscs/swissai/infra01/container-images/ci/vllm_nvidia_alps-arm64.sqsh"
bash experiments/workload-shape/run-shape.sh /iopsstor/scratch/cscs/faruk_zahiragic/workload-shape-qwen3-8b/runs/full-20261127
echo "END $(date -u -Iseconds)"
