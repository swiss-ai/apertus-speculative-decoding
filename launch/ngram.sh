#!/usr/bin/env bash
set -euo pipefail

MODEL_LAUNCH_ROOT="${MODEL_LAUNCH_ROOT:-../model-launch}"
TARGET_MODEL="${TARGET_MODEL:-/capstor/store/cscs/swissai/infra01/hf_models/models/swiss-ai/Apertus-v1.5-70B}"
NUM_SPECULATIVE_TOKENS="${NUM_SPECULATIVE_TOKENS:-3}"
PROMPT_LOOKUP_MAX="${PROMPT_LOOKUP_MAX:-4}"
PROMPT_LOOKUP_MIN="${PROMPT_LOOKUP_MIN:-1}"
SML_TIME="${SML_TIME:-04:00:00}"
RUN_SUFFIX="${RUN_SUFFIX:-$(id -un)-$(date -u +%Y%m%dT%H%M%SZ)}"
SERVED_MODEL="${SERVED_MODEL:-swiss-ai/Apertus-v1.5-70B-ngram-n${NUM_SPECULATIVE_TOKENS}-${RUN_SUFFIX}}"

case "${NUM_SPECULATIVE_TOKENS}" in
  2|3|5|8) ;;
  *) echo "NUM_SPECULATIVE_TOKENS must be one of: 2, 3, 5, 8" >&2; exit 2 ;;
esac

cd "${MODEL_LAUNCH_ROOT}"
echo "served_model=${SERVED_MODEL}"
echo "target_model=${TARGET_MODEL}"

sml advanced \
  --firecrest-system clariden \
  --partition normal \
  --slurm-nodes-per-replica 1 \
  --serving-framework vllm \
  --slurm-time "${SML_TIME}" \
  --slurm-environment src/swiss_ai_model_launch/assets/envs/vllm_apertus_1.5_release.toml \
  --framework-args "--model ${TARGET_MODEL} \
    --served-model-name ${SERVED_MODEL} \
    --chat-template-content-format string \
    --tensor-parallel-size 4 \
    --gpu-memory-utilization 0.8 \
    --max-model-len 262144 \
    --enable-auto-tool-choice \
    --tool-call-parser apertus \
    --speculative-config.method ngram \
    --speculative-config.num_speculative_tokens ${NUM_SPECULATIVE_TOKENS} \
    --speculative-config.prompt_lookup_min ${PROMPT_LOOKUP_MIN} \
    --speculative-config.prompt_lookup_max ${PROMPT_LOOKUP_MAX} \
    --compilation-config.pass_config.fuse_allreduce_rms false"
