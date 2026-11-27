#!/bin/bash
#SBATCH --job-name=workload-shape-q4b
#SBATCH --account=d_research_serving
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=288
#SBATCH --gpus-per-node=4
#SBATCH --time=02:00:00
#SBATCH --exclusive
#SBATCH --environment=/iopsstor/scratch/cscs/faruk_zahiragic/workload-shape-qwen3-4b/sml-env/env_resolved_arm64_3AtXrE.toml
#SBATCH --output=/iopsstor/scratch/cscs/faruk_zahiragic/workload-shape-qwen3-4b/runs/full-20261127/logs/%j.out
#SBATCH --chdir=/iopsstor/scratch/cscs/faruk_zahiragic/spec-exp/workload-shape-qwen3-4b/code-75e9f01d7df0
set -euo pipefail
export SHAPE_IMAGE=/capstor/store/cscs/swissai/infra01/container-images/ci/vllm_nvidia_alps-arm64.sqsh
export LIMIT=0 STAGES="inputs capture shape aggregate"
export APERTUS_EAGLE_CONTRACT=targets/qwen3-4b/contract.json
export TARGET=/iopsstor/scratch/cscs/faruk_zahiragic/workload-shape-qwen3-4b/models/Qwen--Qwen3-4B
export EAGLE_HEAD=''
export DSPARK=/iopsstor/scratch/cscs/faruk_zahiragic/workload-shape-qwen3-4b/models/RedHatAI--Qwen3-4B-speculator.dspark
export DSPARK_HF_LAYERS=0\,8\,16\,24\,32
export DSPARK_OWN_VOCAB=1
export SHAPE_SAMPLED=1
export MAX_MODEL_LEN=40960
export SPECULATORS_SRC=/iopsstor/scratch/cscs/faruk_zahiragic/workload-shape-qwen3-4b/speculators-e3041dd/src
export SHAPE_REF_DSPARK=/iopsstor/scratch/cscs/faruk_zahiragic/workload-shape-qwen3-4b/models/RedHatAI--Qwen3-4B-speculator.dspark
export SHAPE_REF_TARGET=/iopsstor/scratch/cscs/faruk_zahiragic/workload-shape-qwen3-4b/models/Qwen--Qwen3-4B
export TREE_PYDEPS=/iopsstor/scratch/cscs/faruk_zahiragic/workload-shape-qwen3-4b/pydeps
export HF_HOME=/iopsstor/scratch/cscs/faruk_zahiragic/workload-shape-qwen3-4b/hf_home

echo "START $(date -u -Iseconds) host=$(hostname) job=${SLURM_JOB_ID} image=/capstor/store/cscs/swissai/infra01/container-images/ci/vllm_nvidia_alps-arm64.sqsh"
bash experiments/workload-shape/run-shape.sh /iopsstor/scratch/cscs/faruk_zahiragic/workload-shape-qwen3-4b/runs/full-20261127
echo "END $(date -u -Iseconds)"
