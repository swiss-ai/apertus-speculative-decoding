#!/bin/bash
# Run on the Beverin login node: P0 + P1a/P1b pilots from the snapshot.
set -euo pipefail
W=/iopsstor/scratch/cscs/faruk_zahiragic/beverin-eagle
D=$W/repo-80d511ff
cd $D
P0=$(sbatch --parsable --nice=10000 --chdir=$D methods/eagle/beverin/p0.sbatch)
echo "P0 $P0"
common() {
  env EAGLE_TRAIN_CONFIG=methods/eagle/configs/8b/train-e31-opb-thinkon-2n.yaml EAGLE_NODES=2 EAGLE_STEPS=train \
    EAGLE_CUDA_DEVICES=0,1,2,3 EAGLE_TRAIN_PARTITION=mi300 EAGLE_ENV_SOURCE=$D/methods/eagle/configs/train-env-beverin.toml \
    EAGLE_APPLY_VLLM_PATCHES=0 EAGLE_TRAIN_CPUS= EAGLE_TRAIN_MEM= EAGLE_TRAIN_EXCLUDE= EAGLE_PLATFORM=rocm \
    EAGLE_MEM_LOG_DIR=$W/logs EAGLE_SBATCH_EXTRA=--nice=10000 "$@" bash methods/eagle/launch/submit-eagle-train.sh
}
common EAGLE_RUN_ID=stage1-bev-pilot-lt EAGLE_TRAIN_TIME=01:30:00 EAGLE_DEPENDENCY=afterok:$P0 ${DRY:+EAGLE_SUBMIT_DRY_RUN=1}
out=$(common EAGLE_RUN_ID=stage1-bev-pilot-rb EAGLE_BLAS_BACKEND=cublas EAGLE_TRAIN_TIME=03:00:00 \
  EAGLE_STOP_AFTER_STEPS=167800 EAGLE_DEPENDENCY=afterok:$P0 ${DRY:+EAGLE_SUBMIT_DRY_RUN=1})
echo "$out"
RB1=$(echo "$out" | sed -n 's/^Submitted batch job \([0-9]*\).*/\1/p')
common EAGLE_RUN_ID=stage1-bev-pilot-rb EAGLE_BLAS_BACKEND=cublas EAGLE_TRAIN_TIME=01:00:00 \
  EAGLE_DEPENDENCY=afterany:${RB1:-0} ${DRY:+EAGLE_SUBMIT_DRY_RUN=1}
