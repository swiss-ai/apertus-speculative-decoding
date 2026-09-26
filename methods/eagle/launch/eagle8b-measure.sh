#!/usr/bin/env bash
# A5/A6 measurement driver for Stage A: one independent deployment per call.
#   ARM=baseline PHASE=screen BLOCK_ID=b1 ./methods/eagle/launch/eagle8b-measure.sh
#   ARM=eagle DEPTH=3 PHASE=screen BLOCK_ID=b1 EAGLE_HEAD=... ./methods/eagle/launch/eagle8b-measure.sh
# PHASE: profile (32 fixed-256 prompts, ignore_eos, C=1; PROFILE_TRACE=1 adds a
#        torch-profiler window after the unprofiled headline cells)
#        screen  (validation strata, natural EOS, C=1 and C=8, 64 requests)
#        confirm (untouched test strata, natural EOS, C=1 and C=8, 128 requests)
# Every deployment gets a fresh VLLM_CACHE_ROOT, so compile-cache state is the
# same (cold) for every arm and repeat. Runs on a Clariden login node.
set -euo pipefail

LAUNCH_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "${LAUNCH_DIR}/../../.." && pwd)"
SERVING_DIR="${REPO_ROOT}/serving"
export STAGE=8b
export MODEL_LAUNCH_ROOT="${MODEL_LAUNCH_ROOT:-${HOME}/model-launch-apertus}"
export PATH="${HOME}/venvs/sml-apertus/bin:${PATH}"
export SML_PARTITION="${SML_PARTITION:-debug}"
export SML_TIME="${SML_TIME:-01:30:00}"
ARM="${ARM:?set ARM=eagle or ARM=baseline}"
PHASE="${PHASE:?set PHASE=profile, screen or confirm}"
BLOCK_ID="${BLOCK_ID:?set BLOCK_ID (deployment block label)}"
BENCH="${BENCH:-${HOME}/venvs/apertus-bench/bin/apertus-bench}"
WORKLOAD_DIR="${WORKLOAD_DIR:-${REPO_ROOT}/workloads/8b}"
CONTRACT="${REPO_ROOT}/targets/8b/contract.json"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
case "${ARM}" in
  eagle) : "${DEPTH:?set DEPTH for ARM=eagle}"; : "${EAGLE_HEAD:?set EAGLE_HEAD}"
         export NUM_SPECULATIVE_TOKENS="${DEPTH}" ALGORITHM=eagle31
         [ "${DEPTH}" = 1 ] && export ALLOW_DEPTH_1=1
         VARIANT="apertus15-8b-eagle31-k${DEPTH}"; LAUNCHER="${LAUNCH_DIR}/eagle.sh" ;;
  baseline) VARIANT="apertus15-8b-baseline"; LAUNCHER="${SERVING_DIR}/baseline.sh" ;;
  *) echo "ARM must be eagle or baseline" >&2; exit 2 ;;
esac
DEPLOYMENT_ID="${VARIANT}-${PHASE}-${BLOCK_ID}-${STAMP}"
OUT="${REPO_ROOT}/results/8b/eagle/${PHASE}/${DEPLOYMENT_ID}"
mkdir -p "${OUT}"
# STAGEA_LOCK: a driver stuck on a hung login node (ln004, 2026-09-25) keeps the default lock.
exec 9>"${STAGEA_LOCK:-${HOME}/.sml/eagle8b-smoke.lock}"
flock -n 9 || { echo "another Stage A deployment driver is running" >&2; exit 3; }
export VLLM_PATCH_DIR="${HOME}/.sml/vllm-patch-${DEPLOYMENT_ID}"
# The serving environment mounts /capstor and /iopsstor, not /users: a path under
# $HOME exists only inside the container (the first traced profile was lost).
SCRATCH_ROOT="${SCRATCH_ROOT:-/iopsstor/scratch/cscs/${USER}/apertus-eagle/8b}"
CACHE_ROOT="${SCRATCH_ROOT}/vllm-cache/${DEPLOYMENT_ID}"
mkdir -p "${CACHE_ROOT}"
export EXTRA_ENV="VLLM_CACHE_ROOT=${CACHE_ROOT}"
if [ "${PROFILE_TRACE:-0}" = "1" ]; then
  export PROFILER_DIR="${SCRATCH_ROOT}/traces/${DEPLOYMENT_ID}"
  mkdir -p "${PROFILER_DIR}"
fi

HEAD_REPORT=""
CHECKPOINT_SHA=""
if [ "${ARM}" = "eagle" ]; then
  HEAD_REPORT="$(PYTHONPATH="${REPO_ROOT}/src" python3 -m apertus_bench.eagle "${EAGLE_HEAD}" \
    --contract "${CONTRACT}" --algorithm eagle31 --target-model \
    /capstor/store/cscs/swissai/infra01/hf_models/models/swiss-ai/Apertus-v1.5-8B)"
  printf '%s\n' "${HEAD_REPORT}" > "${OUT}/head-report.json"
  CHECKPOINT_SHA="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["checkpoint_manifest_sha256"])' "${OUT}/head-report.json")"
fi

RUN_SUFFIX="${PHASE}-${BLOCK_ID}-${STAMP}" "${LAUNCHER}" --no-tui > "${OUT}/launch.log" 2>&1
JOB="$(grep -aoE 'Job submitted: [0-9]+' "${OUT}/launch.log" | tail -1 | awk '{print $NF}')"
[ -n "${JOB}" ] || { echo "no job id; see ${OUT}/launch.log" >&2; exit 1; }
echo "job=${JOB}" > "${OUT}/job.txt"
trap 'scancel "${JOB}" 2>/dev/null || true' EXIT

started="$(date +%s)"
INFO="$("${SERVING_DIR}/wait-replica.sh" "${JOB}")" || { echo "status=not_ready" > "${OUT}/status.txt"; exit 1; }
echo "${INFO}" > "${OUT}/replica.txt"
eval "${INFO}"
echo "ready_seconds=$(( $(date +%s) - started ))" >> "${OUT}/replica.txt"
BASE="http://${IP}:8080"
LOGS="${HOME}/.sml/logs/${JOB}"
cp "${LOGS}/log.out" "${OUT}/sml-log.out" 2>/dev/null || true
grep -aE 'non-default args|Initializing a V1|KV cache|Maximum concurrency|Model loading took|Eagle3 auxiliary|async|cudagraph|CUDA graph|max_num_batched_tokens|max_num_scheduled_tokens' \
  "${LOGS}/replica_0.out" > "${OUT}/engine-excerpt.txt" 2>/dev/null || true
IMAGE_PATH="/capstor/store/cscs/swissai/infra01/container-images/ci/vllm_apertus_1.5_release-arm64.sqsh"

METADATA=(
  --metadata "target_model=swiss-ai/Apertus-v1.5-8B"
  --metadata "target_revision=a411d838600baf0e3635a3daf66fb7c55fc97bb6"
  --metadata "tokenizer_sha256=1f2f6198ea5789e5a90ec7c5ec5cf0d5242cf5b8de007105537c91d777581582"
  --metadata "target_tensor_parallel_size=1"
  --metadata "precision=bfloat16"
  --metadata "container_image=${IMAGE_PATH}"
  --metadata "container_sha256=f5fc017ff794fa8d80b73dd9a788028a1872dbfc2e21c0de043fb7738c11a0b5"
  --metadata "vllm_revision=a601a9d998ddeb488f0c17e8512874b116aa7658"
  --metadata "model_launch_revision=909026a990454557f1b54d26f24ec3ad92e51e35"
  --metadata "max_model_len=32768"
  --metadata "gpu_memory_utilization=0.8"
  --metadata "prefix_caching=false"
  --metadata "async_scheduling=engine-default"
  --metadata "compile_cache=fresh-per-deployment"
  --metadata "compile_cache_root=${CACHE_ROOT}"
  --metadata "deployment_id=${DEPLOYMENT_ID}"
  --metadata "block_id=${BLOCK_ID}"
  --metadata "slurm_job_id=${JOB}"
  --metadata "slurm_node_id=${NODE}"
  --metadata "hardware=1x Clariden GH200 node (4 GPUs allocated, 1 used)"
  --metadata "load_generator=clariden login node, direct to replica IP"
  --metadata "phase=${PHASE}"
)
VARIANT_FLAGS=(--variant "${VARIANT}")
if [ "${ARM}" = "eagle" ]; then
  VARIANT_FLAGS+=(--method eagle3 --algorithm eagle31 --num-speculative-tokens "${DEPTH}" --draft-tensor-parallel-size 1)
  METADATA+=(--metadata "checkpoint_sha256=${CHECKPOINT_SHA}" --metadata "eagle_head=${EAGLE_HEAD}")
  METADATA+=(--metadata "training_revision=torchspec-6c042a8+apertus_eagle.train_rollout")
else
  VARIANT_FLAGS+=(--method none)
  METADATA+=(--metadata "baseline_role=operational")
fi

cell() {  # workload_file workload concurrency requests extra-args...
  local file="$1" workload="$2" conc="$3" requests="$4"; shift 4
  local warmup=$(( conc * 2 > 8 ? conc * 2 : 8 ))
  "${BENCH}" run --base-url "${BASE}" --metrics-url "${BASE}/metrics" --model "${MODEL}" \
    "${VARIANT_FLAGS[@]}" --workloads "${file}" --workload "${workload}" \
    --concurrency "${conc}" --repeat 1 --requests "${requests}" --warmup-requests "${warmup}" \
    --timeout-seconds 1800 "${METADATA[@]}" "$@" \
    --output "${OUT}/cells/${workload}/c${conc}" >> "${OUT}/cells.log" 2>&1 \
    || echo "cell failed: ${workload} c=${conc}" | tee -a "${OUT}/status.txt"
}

case "${PHASE}" in
  profile)
    cell "${WORKLOAD_DIR}/eagle-8b-profile.jsonl" profile_fixed256 1 32 --ignore-eos --max-tokens 256
    if [ "${PROFILE_TRACE:-0}" = "1" ]; then
      "${BENCH}" start-profile --base-url "${BASE}" || true
      "${BENCH}" run --base-url "${BASE}" --no-metrics --model "${MODEL}" "${VARIANT_FLAGS[@]}" \
        --workloads "${WORKLOAD_DIR}/eagle-8b-profile.jsonl" --workload profile_fixed256 \
        --concurrency 1 --repeat 1 --requests 4 --warmup-requests 8 --ignore-eos --max-tokens 256 \
        "${METADATA[@]}" --metadata "profiled=true" --output "${OUT}/cells/profile-trace" >> "${OUT}/cells.log" 2>&1 || true
      "${BENCH}" stop-profile --base-url "${BASE}" || true
    fi ;;
  screen|confirm)
    file="${WORKLOAD_DIR}/eagle-8b-$([ "${PHASE}" = screen ] && echo validation || echo test).jsonl"
    requests=$([ "${PHASE}" = screen ] && echo 64 || echo 128)
    for workload in chat code summarization; do
      for conc in 1 8; do cell "${file}" "${workload}" "${conc}" "${requests}"; done
    done ;;
  *) echo "unknown PHASE ${PHASE}" >&2; exit 2 ;;
esac

curl -sS --max-time 60 "${BASE}/metrics" > "${OUT}/metrics-final.txt" || true
cp "${LOGS}/replica_0.out" "${OUT}/replica_0.out" 2>/dev/null || true
grep -q "cell failed" "${OUT}/status.txt" 2>/dev/null || echo "status=ok" >> "${OUT}/status.txt"
echo "${OUT}"
