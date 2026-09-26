#!/usr/bin/env bash
# Serve Apertus 1.5 8B alone at TP=4 as a bound on embedded draft-step cost.
set -euo pipefail

LAUNCH_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "${LAUNCH_DIR}/../.." && pwd)"
SERVING_DIR="${REPO_ROOT}/serving"
MODEL_LAUNCH_ROOT="${MODEL_LAUNCH_ROOT:-../model-launch}"
TARGET_MODEL="${TARGET_MODEL:-/capstor/store/cscs/swissai/infra01/hf_models/models/swiss-ai/Apertus-v1.5-8B}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-131072}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.8}"
SML_PARTITION="${SML_PARTITION:-normal}"
SML_TIME="${SML_TIME:-04:00:00}"
ENV_SOURCE="${ENV_SOURCE:-${MODEL_LAUNCH_ROOT}/src/swiss_ai_model_launch/assets/envs/vllm_apertus_1.5_release.toml}"
RUN_SUFFIX="${RUN_SUFFIX:-$(id -un)-$(date -u +%Y%m%dT%H%M%SZ)}"
SERVED_MODEL="${SERVED_MODEL:-swiss-ai/Apertus-v1.5-8B-standalone-${RUN_SUFFIX}}"
EXTRA_VLLM="$("${SERVING_DIR}/vllm-extra-flags.sh")"

SML_ENVIRONMENT="$("${SERVING_DIR}/resolve-env.sh" "${ENV_SOURCE}")"

cd "${MODEL_LAUNCH_ROOT}"
echo "served_model=${SERVED_MODEL}"
echo "target_model=${TARGET_MODEL}"
echo "environment=${SML_ENVIRONMENT}"
echo "async_scheduling=${ASYNC_SCHEDULING:-engine-default}"
echo "enable_prefix_caching=${ENABLE_PREFIX_CACHING:-engine-default}"
echo "max_num_batched_tokens=${MAX_NUM_BATCHED_TOKENS:-engine-default}"

# shellcheck disable=SC2086
sml advanced \
  --system clariden \
  --partition "${SML_PARTITION}" \
  --nodes-per-replica 1 \
  --framework vllm \
  --time "${SML_TIME}" \
  --environment "${SML_ENVIRONMENT}" \
  --framework-args "--model ${TARGET_MODEL} \
    --served-model-name ${SERVED_MODEL} \
    --chat-template-content-format string \
    --tensor-parallel-size 4 \
    --gpu-memory-utilization ${GPU_MEMORY_UTILIZATION} \
    --max-model-len ${MAX_MODEL_LEN} \
    --enable-auto-tool-choice \
    --tool-call-parser apertus \
    --compilation-config.pass_config.fuse_allreduce_rms false \
    ${EXTRA_VLLM}" \
  "$@"
