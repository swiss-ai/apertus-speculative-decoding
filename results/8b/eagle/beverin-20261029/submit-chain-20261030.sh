#!/bin/bash
# Run on the Beverin login node: the held production chain for e31-opb-thinkon-stage1.
set -euo pipefail
W=/iopsstor/scratch/cscs/faruk_zahiragic/beverin-eagle
D=$W/repo-80d511ff
cd $D
env EAGLE_TRAIN_CONFIG=methods/eagle/configs/8b/train-e31-opb-thinkon-2n.yaml EAGLE_NODES=2 EAGLE_STEPS=train \
  EAGLE_CUDA_DEVICES=0,1,2,3 EAGLE_TRAIN_PARTITION=mi300 EAGLE_ENV_SOURCE=$D/methods/eagle/configs/train-env-beverin.toml \
  EAGLE_APPLY_VLLM_PATCHES=0 EAGLE_TRAIN_CPUS= EAGLE_TRAIN_MEM= EAGLE_TRAIN_EXCLUDE= EAGLE_PLATFORM=rocm \
  EAGLE_MEM_LOG_DIR=$W/logs EAGLE_BLAS_BACKEND=cublas EAGLE_RUN_ID=stage1 EAGLE_CHAIN=10 EAGLE_TRAIN_TIME=24:00:00 \
  EAGLE_START_GUARD_RUN_DIR=/iopsstor/scratch/cscs/faruk_zahiragic/apertus-eagle/8b/runs/e31-opb-thinkon-stage1 \
  EAGLE_SBATCH_EXTRA=--hold bash methods/eagle/launch/submit-eagle-train.sh
