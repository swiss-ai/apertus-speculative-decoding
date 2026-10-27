#!/usr/bin/env bash
# Separate scaling pilot; never writes production checkpoints.
set -euo pipefail
base=/iopsstor/scratch/cscs/faruk_zahiragic/beverin-feasibility
# Written only after the finished 16-node job has passed review.
python3 - "${base}/provenance/scaling-20261027/16n-gates.json" <<'GATE'
import json,sys
with open(sys.argv[1]) as stream:
    gates=json.load(stream)
assert gates['eligible_for_32n'], gates
assert gates['job_id']=='697549', gates
print('16-node gate passed:', gates['decision_reason'])
GATE
export RUN_ID=scaling-32n-20261027-rocblas
if [ -e "${base}/runs/${RUN_ID}" ]; then
  echo "Refusing to reuse an existing run directory: ${RUN_ID}" >&2
  exit 1
fi
export WORK="${base}"
export SPECULATORS="${base}/speculators-7046ac6-provenance"
export DATA_PROVENANCE="${base}/provenance/data-20261025"
export NODES=32 TIME=03:00:00 CHAIN=1 STOP_MARGIN=1200
export SEQ=8192 ANCHORS=512 LR=8.48528137423857e-4 EPOCHS=10 CKPT_FREQ=0.015625
export EXTRACT_MEM=0.62 EXTRACT_FLAGS='--api-server-count 1 --renderer-num-workers 1'
export WORKERS=1 PREFETCH=1 BEVERIN_EMPTY_CACHE_ON_SAVE=1
export BEVERIN_BLAS_BACKEND=cublas BEVERIN_SEGMENT_STEPS=100 BEVERIN_PILOT_RESUME=1
export BEVERIN_PROFILE_STEPS=0 NCCL_DEBUG=WARN
export SBATCH_EXTRA=--nice=10000
bash "${base}/repo-ada15200/methods/dspark/beverin/submit-train.sh"
