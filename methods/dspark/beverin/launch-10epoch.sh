#!/usr/bin/env bash
# Ten-epoch DSpark 70B on 16 Beverin nodes, ALPS image, fresh initialization.
# The recipe of the one-epoch 32-node run (results/70b/dspark/
# beverin-32n-alps-1epoch-20261101) with three deliberate changes: 16 nodes
# (262,144 tokens/step, 4x Yu's 8B batch, about 8,150 updates per epoch),
# LR 6e-4 (3e-4 x sqrt(4), the Clariden 16-node choice) and a linear schedule
# over all ten epochs (EPOCHS=10). Multi-epoch resume and per-epoch archives
# come from multi_epoch.py. Run from a packaged runtime on the Beverin login node:
#   STAGE=smoke-data                       1 node: 600-row slice of the corpus
#   STAGE=smoke RUN_ID=...                 1 node, 3 epochs on the slice, stopped
#                                          mid-epoch then resumed (two jobs)
#   STAGE=b RUN_ID=...                     first 640 updates on 16 nodes (6 h)
#   STAGE=chain RUN_ID=... CHAIN=N [DEPENDENCY=afterok:J]   24 h links to the end
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
STAGE="${STAGE:?STAGE=smoke-data|smoke|b|chain}"
export WORK="${WORK:-/iopsstor/scratch/cscs/${USER}/beverin-feasibility}"
export RUNTIME="${REPO_ROOT}"
EDF="${REPO_ROOT}/methods/dspark/beverin/env-vllm-alps.toml"
SMOKE_DATA="${SMOKE_DATA:-${WORK}/data-smoke-multiepoch-600}"
FULL_DATA=/iopsstor/scratch/cscs/${USER}/apertus-dspark70b/data

common() {
  export BEVERIN_ALPS=1 EDF SPECULATORS="${WORK}/speculators-7046ac6-provenance"
  export SIDE_DEPS="${WORK}/py312/pydeps"
  export DATA_PROVENANCE="${WORK}/provenance/data-20261025"
  export MAX_ATTEMPTS=1 EXTRACT_START_ATTEMPTS=1 EXTRACT_HEALTH_SECONDS=7200
  export SBATCH_EXTRA=--no-requeue
  export SEQ=8192 ANCHORS=512 EPOCHS=10 LR=6e-4
  export DSPARK_DTENSOR_ALLGATHER=1 LAYERS='1 20 39 58 77'
  export EXTRACT_MEM=0.62 EXTRACT_FLAGS='--api-server-count 1 --renderer-num-workers 1'
  export WORKERS=1 PREFETCH=1 BEVERIN_EMPTY_CACHE_ON_SAVE=1
  export BEVERIN_CHECK_ACTIVATIONS=1 BEVERIN_CAPTURE_BAD_BATCH=1 NCCL_DEBUG=WARN
  export BEVERIN_BLAS_BACKEND=cublas BEVERIN_SINGLE_EPOCH=0
  # About every 64 updates at ~8,150 per epoch; model-only milestones every
  # 4,096 updates (epoch ends are kept separately under checkpoints/epochs).
  export CKPT_FREQ=0.0078125 BEVERIN_MILESTONE_INTERVAL=4096
  unset MAX_STEPS BEVERIN_PILOT_RESUME BEVERIN_PROFILE_STEPS BEVERIN_EXPECTED_EPOCH_STEPS
  unset BEVERIN_REDUCE_DTYPE BEVERIN_NCCL_PROTO BEVERIN_NCCL_MIN_NCHANNELS
  unset BEVERIN_NCCL_NCHANNELS_PER_NET_PEER BEVERIN_FI_MR_CACHE_MONITOR BEVERIN_STABLE_STEP
}

case "${STAGE}" in
  smoke-data)
    test ! -e "${SMOKE_DATA}"
    sbatch --parsable -A infra01 -p mi300 -N 1 -t 00:30:00 --exclusive --no-requeue \
      -J dspark-smoke-data -o "${WORK}/logs/%x-%j.out" --wrap "srun --environment=${EDF} \
      env PYTHONNOUSERSITE=1 PYTHONPATH=${WORK}/py312/pydeps:${WORK}/speculators-7046ac6-provenance/src \
      python3 ${REPO_ROOT}/methods/dspark/beverin/data_slice.py ${FULL_DATA} ${SMOKE_DATA} 600"
    ;;
  smoke)
    : "${RUN_ID:?}"
    common
    test ! -e "${WORK}/runs/${RUN_ID}"
    export NODES=1 DATA="${SMOKE_DATA}" EPOCHS=3 CKPT_FREQ=0.25 BEVERIN_MILESTONE_INTERVAL=0
    export TIME=02:00:00 STOP_MARGIN=600 BEVERIN_POST_SCAN=0
    first="$(BEVERIN_SEGMENT_STEPS="${SMOKE_STOP:-80}" bash "${REPO_ROOT}/methods/dspark/beverin/submit-train.sh" | tail -1 | awk '{print $2}')"
    unset BEVERIN_SEGMENT_STEPS
    DEPENDENCY="afterok:${first}" bash "${REPO_ROOT}/methods/dspark/beverin/submit-train.sh"
    ;;
  b)
    : "${RUN_ID:?}"
    common
    test ! -e "${WORK}/runs/${RUN_ID}"
    export NODES=16 DATA="${FULL_DATA}" TIME=06:00:00 STOP_MARGIN=2400 BEVERIN_POST_SCAN=1
    export BEVERIN_SEGMENT_STEPS=640 BEVERIN_STABLE_STEP=640
    bash "${REPO_ROOT}/methods/dspark/beverin/submit-train.sh"
    ;;
  chain)
    : "${RUN_ID:?}" "${CHAIN:?}"
    common
    export NODES=16 DATA="${FULL_DATA}" TIME=24:00:00 STOP_MARGIN=2400 BEVERIN_POST_SCAN=1
    unset BEVERIN_SEGMENT_STEPS
    bash "${REPO_ROOT}/methods/dspark/beverin/submit-train.sh"
    ;;
  *)
    echo "unknown STAGE ${STAGE}" >&2
    exit 2
    ;;
esac
