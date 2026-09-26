#!/usr/bin/env bash
# Stage presets from docs/eagle-execution-plan.md section 2, sourced by the
# baseline and EAGLE launchers. Every value can be overridden from the
# environment; both arms of one stage must use the same values.
#   STAGE=8b  -> Apertus-v1.5-8B,  target TP=1, 32k context
#   STAGE=70b -> Apertus-v1.5-70B, target TP=4, 131k context
# Sourced, not executed: expects REPO_ROOT to be set.

HF_MODELS_ROOT="${HF_MODELS_ROOT:-/capstor/store/cscs/swissai/infra01/hf_models/models/swiss-ai}"
case "${STAGE:-}" in
  8b)
    _stage_model="${HF_MODELS_ROOT}/Apertus-v1.5-8B"
    _stage_name="swiss-ai/Apertus-v1.5-8B"
    _stage_tp=1
    _stage_len=32768
    ;;
  70b)
    _stage_model="${HF_MODELS_ROOT}/Apertus-v1.5-70B"
    _stage_name="swiss-ai/Apertus-v1.5-70B"
    _stage_tp=4
    _stage_len=131072
    ;;
  *)
    echo "STAGE must be 8b or 70b (got '${STAGE:-}')" >&2
    exit 2
    ;;
esac
TARGET_MODEL="${TARGET_MODEL:-${_stage_model}}"
SERVED_BASE="${SERVED_BASE:-${_stage_name}}"
TARGET_TP="${TARGET_TP:-${_stage_tp}}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-${_stage_len}}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.8}"
TARGET_CONTRACT="${TARGET_CONTRACT:-${REPO_ROOT}/results/eagle/${STAGE}/preflight/compatibility.json}"
RESULTS_ROOT="${RESULTS_ROOT:-${REPO_ROOT}/results/eagle/${STAGE}}"
# Controlled measurements run with prefix caching off in both arms.
ENABLE_PREFIX_CACHING="${ENABLE_PREFIX_CACHING:-0}"
export ENABLE_PREFIX_CACHING
case "${TARGET_TP}" in
  1|2|4) ;;
  *) echo "TARGET_TP must be 1, 2 or 4 on a four-GPU node (got ${TARGET_TP})" >&2; exit 2 ;;
esac
