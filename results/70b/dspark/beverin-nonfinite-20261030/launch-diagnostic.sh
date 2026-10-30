#!/usr/bin/env bash
# Instrumented reproduction of the failed recipe, bounded to 640 steps.
set -euo pipefail
revision="${1:?pass the committed runtime Git SHA}"
base=/iopsstor/scratch/cscs/faruk_zahiragic/beverin-feasibility
runtime="${base}/repo-${revision:0:8}"
[ "$(cat "${runtime}/REVISION")" = "${revision}" ]
export RUN_ID=opb-thinkoff-32n-nan-debug-20261030-v2
if [ -e "${base}/runs/${RUN_ID}" ]; then
  echo "Refusing to reuse an existing run: ${RUN_ID}" >&2
  exit 1
fi
unset MAX_STEPS BEVERIN_PILOT_RESUME BEVERIN_PROFILE_STEPS
unset BEVERIN_REDUCE_DTYPE BEVERIN_NCCL_PROTO BEVERIN_NCCL_MIN_NCHANNELS
unset BEVERIN_NCCL_NCHANNELS_PER_NET_PEER BEVERIN_FI_MR_CACHE_MONITOR DEPENDENCY
unset BEVERIN_SINGLE_EPOCH BEVERIN_EXPECTED_EPOCH_STEPS
export WORK="${base}" SPECULATORS="${base}/speculators-7046ac6-provenance"
export SIDE_DEPS="${base}/py312/pydeps"
export DATA=/iopsstor/scratch/cscs/faruk_zahiragic/apertus-dspark70b/data
export DATA_PROVENANCE="${base}/provenance/data-20261025"
export NODES=32 TIME=06:00:00 CHAIN=1 STOP_MARGIN=2400 MAX_ATTEMPTS=1
export SEQ=8192 ANCHORS=512 LR=8.48528137423857e-4 EPOCHS=1 CKPT_FREQ=0.015625
export DSPARK_DTENSOR_ALLGATHER=1 LAYERS='1 20 39 58 77'
export EXTRACT_MEM=0.62 EXTRACT_FLAGS='--api-server-count 1 --renderer-num-workers 1'
export WORKERS=1 PREFETCH=1 BEVERIN_EMPTY_CACHE_ON_SAVE=1
export BEVERIN_BLAS_BACKEND=cublas BEVERIN_SEGMENT_STEPS=640
export BEVERIN_CHECK_ACTIVATIONS=1 BEVERIN_CAPTURE_BAD_BATCH=1
export NCCL_DEBUG=WARN SBATCH_EXTRA=
bash "${runtime}/methods/dspark/beverin/submit-train.sh"
