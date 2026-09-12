#!/usr/bin/env bash
set -euo pipefail

LAUNCH_DIR="$(cd "$(dirname "$0")" && pwd)"
MODEL_LAUNCH_ROOT="${MODEL_LAUNCH_ROOT:-../model-launch}"
TARGET_MODEL="${TARGET_MODEL:-/capstor/store/cscs/swissai/infra01/hf_models/models/swiss-ai/Apertus-v1.5-70B}"
DRAFT_MODEL="${DRAFT_MODEL:-/capstor/store/cscs/swissai/infra01/hf_models/models/swiss-ai/Apertus-v1.5-8B}"
NUM_SPECULATIVE_TOKENS="${NUM_SPECULATIVE_TOKENS:-3}"
DRAFT_TP="${DRAFT_TP:-4}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-131072}"
SML_PARTITION="${SML_PARTITION:-normal}"
SML_TIME="${SML_TIME:-04:00:00}"
ENV_SOURCE="${ENV_SOURCE:-${MODEL_LAUNCH_ROOT}/src/swiss_ai_model_launch/assets/envs/vllm_apertus_1.5_release.toml}"
RUN_SUFFIX="${RUN_SUFFIX:-$(id -un)-$(date -u +%Y%m%dT%H%M%SZ)}"
SERVED_MODEL="${SERVED_MODEL:-swiss-ai/Apertus-v1.5-70B-draft-n${NUM_SPECULATIVE_TOKENS}-tp${DRAFT_TP}-${RUN_SUFFIX}}"

case "${NUM_SPECULATIVE_TOKENS}" in
  2|3|5|8) ;;
  *) echo "NUM_SPECULATIVE_TOKENS must be one of: 2, 3, 5, 8" >&2; exit 2 ;;
esac
case "${DRAFT_TP}" in
  1|4) ;;
  *) echo "DRAFT_TP must be 1 or 4" >&2; exit 2 ;;
esac

SML_ENVIRONMENT="$("${LAUNCH_DIR}/resolve-env.sh" "${ENV_SOURCE}")"

cd "${MODEL_LAUNCH_ROOT}"
echo "served_model=${SERVED_MODEL}"
echo "target_model=${TARGET_MODEL}"
echo "draft_model=${DRAFT_MODEL}"
echo "environment=${SML_ENVIRONMENT}"

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
    --gpu-memory-utilization 0.8 \
    --max-model-len ${MAX_MODEL_LEN} \
    --enable-auto-tool-choice \
    --tool-call-parser apertus \
    --speculative-config.method draft_model \
    --speculative-config.model ${DRAFT_MODEL} \
    --speculative-config.num_speculative_tokens ${NUM_SPECULATIVE_TOKENS} \
    --speculative-config.draft_tensor_parallel_size ${DRAFT_TP} \
    --compilation-config.pass_config.fuse_allreduce_rms false" \
  "$@"
