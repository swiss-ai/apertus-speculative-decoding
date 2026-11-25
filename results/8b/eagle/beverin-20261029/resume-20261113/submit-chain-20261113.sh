#!/bin/bash
# Run on the Beverin login node: resume e31-opb-thinkon-stage1 from checkpoints/resume
# (step 299,964, written by link 707011) with 16 more 24 h links (~193k steps left at
# ~12.6k steps/link). Same command as ../submit-chain-20261030.sh except:
#  - D is a fresh snapshot of the same commit (80d511ff); the old repo-80d511ff was
#    mostly deleted by the scratch cleaner on 2026-11-12 (see README.md),
#  - EAGLE_CHAIN=16, and no --hold (cutover gates passed on 2026-10-30),
#  - SML_ENV_DIR on iopsstor: /users (shared with Clariden) is at its 50 GB quota.
set -euo pipefail
W=/iopsstor/scratch/cscs/faruk_zahiragic/beverin-eagle
D=$W/repo-80d511ff-restored-20261113
cd $D
export SML_ENV_DIR=$W/sml-env
env EAGLE_TRAIN_CONFIG=methods/eagle/configs/8b/train-e31-opb-thinkon-2n.yaml EAGLE_NODES=2 EAGLE_STEPS=train \
  EAGLE_CUDA_DEVICES=0,1,2,3 EAGLE_TRAIN_PARTITION=mi300 EAGLE_ENV_SOURCE=$D/methods/eagle/configs/train-env-beverin.toml \
  EAGLE_APPLY_VLLM_PATCHES=0 EAGLE_TRAIN_CPUS= EAGLE_TRAIN_MEM= EAGLE_TRAIN_EXCLUDE= EAGLE_PLATFORM=rocm \
  EAGLE_MEM_LOG_DIR=$W/logs EAGLE_BLAS_BACKEND=cublas EAGLE_RUN_ID=stage1 EAGLE_CHAIN=16 EAGLE_TRAIN_TIME=24:00:00 \
  EAGLE_START_GUARD_RUN_DIR=/iopsstor/scratch/cscs/faruk_zahiragic/apertus-eagle/8b/runs/e31-opb-thinkon-stage1 \
  ${DRY:+EAGLE_SUBMIT_DRY_RUN=1} bash methods/eagle/launch/submit-eagle-train.sh
