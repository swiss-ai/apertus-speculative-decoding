#!/usr/bin/env bash
# Measure one diagnostic deployment: readiness artifacts, optional capture, then
# the fixed-output 32-prompt cells at concurrency 1 and 8.
set -euo pipefail

DIAG_CONFIG="${1:?usage: drive-diagnostics.sh CONFIG JOB NODE IP MODEL BLOCK}"
JOB="${2:?}"
NODE="${3:?}"
IP="${4:?}"
MODEL="${5:?}"
BLOCK="${6:?}"

LAUNCH_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "${LAUNCH_DIR}/../.." && pwd)"
SERVING_DIR="${REPO_ROOT}/serving"
cd "${REPO_ROOT}"

BENCH="${BENCH:-${HOME}/venvs/apertus-bench/bin/apertus-bench}"
PYTHON="${PYTHON:-python3}"
BASE="http://${IP}:8080"
DATE_TAG="${DATE_TAG:-$(date -u +%Y%m%d)}"
RUN_ROOT="${RUN_ROOT:-results/diagnostics-${DATE_TAG}}"
WORKLOADS="${WORKLOADS:-workloads/diagnostics-fixed-256.jsonl}"
WORKLOAD_NAME="${WORKLOAD_NAME:-mechanistic_fixed256}"
IMAGE="${IMAGE:-/capstor/store/cscs/swissai/infra01/container-images/ci/vllm_apertus_1.5_release-arm64.sqsh}"
LOG_DIR="${SML_LOG_DIR:-${HOME}/.sml/logs/${JOB}}"
DEPLOYMENT_ID="${JOB}"
OUT_ROOT="${RUN_ROOT}/${DIAG_CONFIG}/${DEPLOYMENT_ID}"
PROV="${RUN_ROOT}/provenance/${DIAG_CONFIG}-${DEPLOYMENT_ID}"
mkdir -p "${PROV}" "${OUT_ROOT}"

eval "$(
  PYTHONPATH="${REPO_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}" \
    "${PYTHON}" -c "from apertus_bench.diagnostics import format_env_exports; print(format_env_exports('${DIAG_CONFIG}'))"
)"

parse_log() {
  local file="$1"
  local pattern="$2"
  local default="${3:-unknown}"
  if [ -f "${file}" ]; then
    grep -aE "${pattern}" "${file}" | tail -1 || echo "${default}"
  else
    echo "${default}"
  fi
}

ASYNC_LOG="$(parse_log "${LOG_DIR}/replica_0.out" 'async scheduling|Asynchronous scheduling|Async scheduling' unknown)"
BATCHED_LOG="$(parse_log "${LOG_DIR}/replica_0.out" 'max_num_batched_tokens=' unknown)"
SCHEDULED_LOG="$(parse_log "${LOG_DIR}/replica_0.out" 'max_num_scheduled_tokens' unknown)"
PREFIX_LOG="$(parse_log "${LOG_DIR}/replica_0.out" 'enable_prefix_caching|prefix caching' unknown)"
KV_LOG="$(parse_log "${LOG_DIR}/replica_0.out" 'GPU KV cache size|kv_cache_size' unknown)"
VERSION_LOG="$(parse_log "${LOG_DIR}/replica_0.out" 'Initializing a V1 LLM engine' unknown)"
EFFECTIVE_SCHEDULED_TOKENS="$(
  PYTHONPATH="${REPO_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}" \
    "${PYTHON}" -c "
from pathlib import Path
from apertus_bench.diagnostics import extract_effective_scheduled_tokens
log = Path('${LOG_DIR}/replica_0.out')
text = log.read_text(errors='replace') if log.is_file() else ''
print(extract_effective_scheduled_tokens(text, requested_batched_tokens='${MAX_NUM_BATCHED_TOKENS}'))
"
)"

echo "### /v1/models"
curl -sS --max-time 60 "${BASE}/v1/models" | tee "${PROV}/v1-models.json"
echo

echo "### first chat completion"
curl -sS --max-time 300 "${BASE}/v1/chat/completions" \
  -H 'Content-Type: application/json' \
  -d "{\"model\":\"${MODEL}\",\"messages\":[{\"role\":\"user\",\"content\":\"In one sentence, what is speculative decoding?\"}],\"max_tokens\":64,\"temperature\":0,\"seed\":1}" \
  | tee "${PROV}/first-chat-completion.json" >/dev/null
python3 - "${PROV}/first-chat-completion.json" <<'PY'
import json, sys
payload = json.load(open(sys.argv[1]))
choice = payload["choices"][0]
print("CHAT_COMPLETION_CONTENT:", json.dumps(choice["message"]["content"]))
print("usage:", json.dumps(payload.get("usage")))
PY

if [ -f "${LOG_DIR}/log.out" ]; then
  cp "${LOG_DIR}/log.out" "${PROV}/sml-log.out"
fi
if [ -f "${LOG_DIR}/replica_0.out" ]; then
  grep -aE 'async schedul|max_num_batched_tokens|max_num_scheduled_tokens|prefix cach|CUDA graph|cudagraph|KV cache|non-default args|Initializing a V1' \
    "${LOG_DIR}/replica_0.out" > "${PROV}/engine-config-excerpt.txt" || true
fi
printf '%s\n' "${ASYNC_LOG}" > "${PROV}/async-scheduling.txt"
printf '%s\n' "${BATCHED_LOG}" > "${PROV}/max-num-batched-tokens.txt"
printf '%s\n' "${SCHEDULED_LOG}" > "${PROV}/max-num-scheduled-tokens.txt"
printf '%s\n' "${EFFECTIVE_SCHEDULED_TOKENS}" > "${PROV}/effective-scheduled-tokens.txt"
printf '%s\n' "${KV_LOG}" > "${PROV}/kv-cache.txt"
printf '%s\n' "${VERSION_LOG}" > "${PROV}/engine-version.txt"

METHOD_FLAG=(--method "${DIAG_METHOD}")
VARIANT_FLAGS=()
if [ -n "${NUM_SPECULATIVE_TOKENS:-}" ]; then
  VARIANT_FLAGS+=(--num-speculative-tokens "${NUM_SPECULATIVE_TOKENS}")
fi
if [ -n "${DRAFT_TP:-}" ]; then
  VARIANT_FLAGS+=(--draft-tensor-parallel-size "${DRAFT_TP}")
fi
if [ -n "${PROMPT_LOOKUP_MAX:-}" ]; then
  VARIANT_FLAGS+=(--prompt-lookup-max "${PROMPT_LOOKUP_MAX}")
fi

METADATA=(
  --metadata "diagnostic_id=${DIAG_CONFIG}"
  --metadata "deployment_id=${DEPLOYMENT_ID}"
  --metadata "block_id=${BLOCK}"
  --metadata "slurm_job_id=${JOB}"
  --metadata "slurm_node_id=${NODE}"
  --metadata "slurm_partition=${SML_PARTITION:-debug}"
  --metadata "replica_head_ip=${IP}"
  --metadata "load_generator=clariden login node, direct to replica node IP (no public gateway)"
  --metadata "max_model_len=${MAX_MODEL_LEN:-131072}"
  --metadata "gpu_memory_utilization=${GPU_MEMORY_UTILIZATION:-0.8}"
  --metadata "tensor_parallel_size=4"
  --metadata "hardware=1x Clariden GH200 node (4 GPUs)"
  --metadata "model_launch_revision=909026a990454557f1b54d26f24ec3ad92e51e35"
  --metadata "vllm_revision=a601a9d998ddeb488f0c17e8512874b116aa7658"
  --metadata "container_image=${IMAGE}"
  --metadata "async_scheduling=${ASYNC_SCHEDULING}"
  --metadata "prefix_caching=${ENABLE_PREFIX_CACHING}"
  --metadata "max_num_batched_tokens=${MAX_NUM_BATCHED_TOKENS}"
  --metadata "effective_scheduled_tokens=${EFFECTIVE_SCHEDULED_TOKENS}"
  --metadata "scheduled_tokens_log=${SCHEDULED_LOG}"
  --metadata "ignore_eos=true"
  --metadata "fixed_output_tokens=256"
)
if [ -n "${BASELINE_ROLE:-}" ]; then
  METADATA+=(--metadata "baseline_role=${BASELINE_ROLE}")
fi

if [ "${SKIP_CAPTURE:-0}" != "1" ]; then
  echo "### greedy capture (8 prompts, natural EOS, max_tokens=64)"
  CAPTURE_WORKLOADS="${PROV}/capture-subset.jsonl"
  head -8 "${WORKLOADS}" > "${CAPTURE_WORKLOADS}"
  "${BENCH}" capture \
    --base-url "${BASE}" \
    --model "${MODEL}" \
    --workloads "${CAPTURE_WORKLOADS}" \
    --max-tokens 64 \
    --output "${RUN_ROOT}/correctness/${DIAG_CONFIG}-${DEPLOYMENT_ID}.json"
fi

if [ "${PROFILE_ONLY:-0}" != "1" ]; then
  echo "### fixed-output matrix"
  for concurrency in 1 8; do
    output="${OUT_ROOT}/${WORKLOAD_NAME}/c${concurrency}/repeat-$(printf '%02d' "${BLOCK}")"
    echo "running ${DIAG_CONFIG} c=${concurrency} block=${BLOCK} -> ${output}"
    "${BENCH}" run \
      --base-url "${BASE}" \
      --metrics-url "${BASE}/metrics" \
      --model "${MODEL}" \
      --variant "${DIAG_CONFIG}" \
      "${METHOD_FLAG[@]}" \
      "${VARIANT_FLAGS[@]}" \
      --workloads "${WORKLOADS}" \
      --workload "${WORKLOAD_NAME}" \
      --concurrency "${concurrency}" \
      --repeat "${BLOCK}" \
      --requests 32 \
      --warmup-requests 8 \
      --ignore-eos \
      --max-tokens 256 \
      --timeout-seconds 1200 \
      "${METADATA[@]}" \
      --output "${output}"
  done
fi

if [ "${PROFILE_AFTER:-0}" = "1" ]; then
  for conc in ${PROFILE_CONCURRENCIES:-${PROFILE_CONCURRENCY:-1}}; do
    echo "### torch profile window c=${conc}"
    "${BENCH}" start-profile --base-url "${BASE}" || echo "start-profile failed" >&2
    "${BENCH}" run \
      --base-url "${BASE}" \
      --no-metrics \
      --model "${MODEL}" \
      --variant "${DIAG_CONFIG}-profile" \
      "${METHOD_FLAG[@]}" \
      "${VARIANT_FLAGS[@]}" \
      --workloads "${WORKLOADS}" \
      --workload "${WORKLOAD_NAME}" \
      --concurrency "${conc}" \
      --repeat "${BLOCK}" \
      --requests "${PROFILE_REQUESTS:-4}" \
      --warmup-requests 8 \
      --ignore-eos \
      --max-tokens 256 \
      --timeout-seconds 1200 \
      "${METADATA[@]}" \
      --metadata "profiled=true" \
      --metadata "profile_concurrency=${conc}" \
      --output "${OUT_ROOT}/${WORKLOAD_NAME}/profile-c${conc}" \
      || echo "profile cell failed" >&2
    "${BENCH}" stop-profile --base-url "${BASE}" || echo "stop-profile failed" >&2
  done
fi

echo "### done ${DIAG_CONFIG} job ${JOB}"
