#!/usr/bin/env bash
# Inside the container: the Clariden per-node extraction server (train-node.sh)
# on Beverin -- speculators' launch_vllm.py train, TP2 on GPUs 0,1, file backend
# to node-local /tmp -- then check_hidden_states.py on GPU 2. Stock vLLM 0.23's
# ExampleHiddenStatesConnector refuses chunked prefill (job 693719); the Swiss
# AI fork's newer connector accepts it.
set -euo pipefail
WORK="${WORK:-/iopsstor/scratch/cscs/${USER}/beverin-feasibility}"
export DSPARK_DIR="${WORK}" SPECULATORS="${SPECULATORS:-${WORK}/speculators}"
. "${REPO_ROOT}/methods/dspark/train/common.sh"   # SIDE_DEPS=${WORK}/pydeps, caches
. "${REPO_ROOT}/methods/dspark/beverin/env.sh"
LAYERS="${LAYERS:-1 20 39 58 77}"
EXTRACT_GPUS="${EXTRACT_GPUS:-0,1}"
EXTRACT_TP=$(( $(tr -cd , <<< "${EXTRACT_GPUS}" | wc -c) + 1 ))
CHECK_ROWS="${CHECK_ROWS:-/iopsstor/scratch/cscs/${USER}/apertus-dspark70b/pilot/tp4-seqs512.jsonl}"
PORT=8001 HS=/tmp/hidden_states JOB="${SLURM_JOB_ID:-local}"
RUN="${WORK}/runs/extract-${JOB}"
mkdir -p "${RUN}" && rm -rf "${HS}" && mkdir -p "${HS}"
[ -d "${SIDE_DEPS}" ] || install_side_deps 2>&1 | tail -5
check_side_deps
stamp "extract TP${EXTRACT_TP} on ${EXTRACT_GPUS}, layers ${LAYERS}"
t0=$(date +%s)
CUDA_VISIBLE_DEVICES="${EXTRACT_GPUS}" setsid python3 "${REPO_ROOT}/methods/dspark/beverin/apertus15_run.py" \
  "${SPECULATORS}/scripts/launch_vllm.py" train "${TARGET_MODEL}" \
  --target-layer-ids ${LAYERS} --hidden-states-backend file --hidden-states-path "${HS}" \
  --provenance-dir "${RUN}" -- \
  --tensor-parallel-size "${EXTRACT_TP}" --enforce-eager --trust-remote-code \
  --port "${PORT}" --gpu-memory-utilization "${EXTRACT_MEM:-0.80}" --max-model-len 8192 \
  ${EXTRACT_ARGS---no-enable-chunked-prefill} > "${RUN}/extract.log" 2>&1 &
server=$!
trap 'kill -TERM -- -${server} 2> /dev/null || true; sleep 5; kill -KILL -- -${server} 2> /dev/null || true' EXIT
if ! wait_health "${PORT}" "${server}" 3000; then
  stamp "EXTRACTION_SERVER_FAILED"; grep -nE "Error|error:|Traceback" "${RUN}/extract.log" | tail -20; tail -30 "${RUN}/extract.log"; exit 1
fi
stamp "extraction server up after $(( $(date +%s) - t0 )) s"
free -g | head -2; amd-smi metric -m --csv 2> /dev/null | head -6 || true
t1=$(date +%s)
python3 "${REPO_ROOT}/methods/dspark/train/check_hidden_states.py" \
  --endpoint "http://localhost:${PORT}/v1" --target "${TARGET_MODEL}" --rows "${CHECK_ROWS}" \
  --layers "$(wc -w <<< "${LAYERS}")" --device "cuda:${CHECK_GPU:-2}" --n "${CHECK_N:-16}" \
  2>&1 | tee "${RUN}/hs-check.log"
stamp "check took $(( $(date +%s) - t1 )) s"
ls "${RUN}"
