#!/usr/bin/env bash
# Container entry for Stage A. One mode per allocation: a1, a2, or a3.
set -euo pipefail
MODE="${1:?a1, a2, or a3}"
: "${REPO_ROOT:?}" "${BEVERIN_DEPLOYMENT:?}"
# shellcheck disable=SC1091
. "${REPO_ROOT}/methods/dspark/beverin/env.sh"
mkdir -p "${BEVERIN_DEPLOYMENT}"
cd "${REPO_ROOT}"

case "${MODE}" in
  a1)
    export BEVERIN_PROBE_DIR="${BEVERIN_DEPLOYMENT}/guard"
    export CUDA_VISIBLE_DEVICES=2,3
    torchrun --standalone --nproc_per_node=2 \
      "${REPO_ROOT}/methods/dspark/beverin/check_numerical_guard.py"
    unset CUDA_VISIBLE_DEVICES
    export EXTRACT_MEM=0.62
    export EXTRACT_ARGS='--api-server-count 1 --renderer-num-workers 1'
    export MIN_AGREEMENT=0.96
    export CHECK_ROWS="${CHECK_ROWS:-/iopsstor/scratch/cscs/${USER}/apertus-dspark70b/pilot/tp4-seqs512.jsonl}"
    export SPECULATORS SIDE_DEPS
    bash "${REPO_ROOT}/methods/dspark/beverin/extract-check.sh"
    export CUDA_VISIBLE_DEVICES=2,3 MUON_TIMING_STEPS=40
    default_rc=0
    MUON_BLAS=default torchrun --standalone --nproc_per_node=2 \
      "${REPO_ROOT}/methods/dspark/beverin/muon_oracle.py" \
      --output "${BEVERIN_DEPLOYMENT}/muon-default" --shapes 80x16 43008x8192 || default_rc=$?
    cublas_rc=0
    MUON_BLAS=cublas torchrun --standalone --nproc_per_node=2 \
      "${REPO_ROOT}/methods/dspark/beverin/muon_oracle.py" \
      --output "${BEVERIN_DEPLOYMENT}/muon-cublas" --shapes 80x16 43008x8192 || cublas_rc=$?
    python3 "${REPO_ROOT}/methods/dspark/beverin/blas_decision.py" \
      --default "${BEVERIN_DEPLOYMENT}/muon-default" \
      --cublas "${BEVERIN_DEPLOYMENT}/muon-cublas" \
      --output "${BEVERIN_DEPLOYMENT}/blas-decision.json"
    echo "STAGE_A1_MUON default_rc=${default_rc} cublas_rc=${cublas_rc}"
    if [ -n "${CAMPAIGN_DECISION:-}" ]; then
      cp "${BEVERIN_DEPLOYMENT}/blas-decision.json" "${CAMPAIGN_DECISION}"
    fi
    ;;
  a2)
    : "${RUN_ID:?}"
    case "${RUN_ID}" in
      faultinject-*) ;;
      *) echo "A2 requires a faultinject- run id" >&2; exit 2 ;;
    esac
    export NODES=1 BEVERIN_FAULT_INJECT=1
    export FAULT_ROOT="${BEVERIN_DEPLOYMENT}/fault" FAULT_REPORT="${BEVERIN_DEPLOYMENT}/fault-report.json"
    python3 "${REPO_ROOT}/methods/dspark/beverin/fault_campaign.py"
    ;;
  a3)
    : "${SPECULATORS:?}" "${TARGET_MODEL:?}"
    prompts="${BEVERIN_DEPLOYMENT}/pilot-prompts.jsonl"
    python3 - "${prompts}" <<'PY'
import json, sys
path = sys.argv[1]
with open(path, "w") as handle:
    for i in range(8):
        handle.write(json.dumps({
            "id": "pilot-%d" % i,
            "messages": [{"role": "user", "content": "Explain checkpoint %d in three short sentences." % i}],
        }) + "\n")
PY
    export DSPARK_DIR="${WORK}" LIMIT=8 BUDGET_SECONDS=3000 HEALTH_SECONDS=3600
    # shellcheck disable=SC1091
    . "${REPO_ROOT}/methods/dspark/train/common.sh"
    # shellcheck disable=SC1091
    . "${REPO_ROOT}/methods/dspark/beverin/env.sh"
    # Job 710632 was TP4 at gpu-memory-utilization 0.90 with CUDA graphs. The
    # 70B loaded, then the host OOM killer fired during JIT warmup. This retry
    # is one TP2 x DP2 server at the memory fraction the extraction servers
    # already ran, with eager mode and a small batch.
    export GPU_MEMORY_UTILIZATION=0.62 MAX_NUM_SEQS=16 CONCURRENCY=8
    export VLLM_EXTRA="--enforce-eager"
    echo "STAGE_A3_SETTINGS tp=2 dp=2 mem=0.62 seqs=16 concurrency=8 eager=1 limit=8"
    rc=0
    TP=2 IN="${prompts}" OUT="${BEVERIN_DEPLOYMENT}/tp2-eager.jsonl" \
      bash "${REPO_ROOT}/methods/dspark/train/regen-part.sh" || rc=$?
    echo "STAGE_A3_REGEN rc=${rc}"
    exit "${rc}"
    ;;
  *)
    echo "unknown mode ${MODE}" >&2
    exit 2
    ;;
esac
echo "STAGE_A_DONE ${MODE}"
