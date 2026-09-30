#!/usr/bin/env bash
# Launch an Apertus 1.5 target with its own EAGLE-3 / EAGLE 3.1 head.
# Engine method is always eagle3; ALGORITHM selects the trained variant.
#   STAGE=8b  EAGLE_HEAD=... ./methods/eagle/launch/eagle.sh   (target TP=1, draft TP=1)
#   STAGE=70b EAGLE_HEAD=... ./methods/eagle/launch/eagle.sh   (target TP=4, draft TP=4)
set -euo pipefail

LAUNCH_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "${LAUNCH_DIR}/../../.." && pwd)"
SERVING_DIR="${REPO_ROOT}/serving"
MODEL_LAUNCH_ROOT="${MODEL_LAUNCH_ROOT:-../model-launch}"
# shellcheck source=serving/stage-defaults.sh
. "${SERVING_DIR}/stage-defaults.sh"
EAGLE_HEAD="${EAGLE_HEAD:-}"
ALGORITHM="${ALGORITHM:-eagle31}"
NUM_SPECULATIVE_TOKENS="${NUM_SPECULATIVE_TOKENS:-3}"
DRAFT_TP="${DRAFT_TP:-${TARGET_TP}}"
PARALLEL_DRAFTING="${PARALLEL_DRAFTING:-0}"
SML_PARTITION="${SML_PARTITION:-normal}"
SML_TIME="${SML_TIME:-04:00:00}"
ENV_SOURCE="${ENV_SOURCE:-${MODEL_LAUNCH_ROOT}/src/swiss_ai_model_launch/assets/envs/vllm_apertus_1.5_release.toml}"
IMAGE="${IMAGE:-/capstor/store/cscs/swissai/infra01/container-images/ci/vllm_apertus_1.5_release-arm64.sqsh}"
APPLY_VLLM_PATCHES="${APPLY_VLLM_PATCHES:-1}"
VALIDATE_ONLY="${VALIDATE_ONLY:-0}"
PYTHON="${PYTHON:-python3}"
RUN_SUFFIX="${RUN_SUFFIX:-$(id -un)-$(date -u +%Y%m%dT%H%M%SZ)}"

case "${ALGORITHM}" in
  eagle3|eagle31|peagle) ;;
  *) echo "ALGORITHM must be one of: eagle3, eagle31, peagle" >&2; exit 2 ;;
esac
# The plan's depth grid (2/3/5/8) plus 7, the DSpark comparison's deepest depth
# and the training rollout length (ttt_length 7).
if [ "${ALLOW_DEPTH_1:-0}" = "1" ]; then
  case "${NUM_SPECULATIVE_TOKENS}" in
    1|2|3|5|7|8) ;;
    *) echo "NUM_SPECULATIVE_TOKENS must be one of: 1, 2, 3, 5, 7, 8" >&2; exit 2 ;;
  esac
else
  case "${NUM_SPECULATIVE_TOKENS}" in
    2|3|5|7|8) ;;
    *) echo "NUM_SPECULATIVE_TOKENS must be one of: 2, 3, 5, 7, 8" >&2; exit 2 ;;
  esac
fi
if [ "${DRAFT_TP}" != "${TARGET_TP}" ]; then
  echo "DRAFT_TP (${DRAFT_TP}) must equal TARGET_TP (${TARGET_TP}) until a mismatched-TP path is proven" >&2
  exit 2
fi
if [ "${ALGORITHM}" = "peagle" ] && [ "${PARALLEL_DRAFTING}" != "1" ]; then
  echo "ALGORITHM=peagle requires PARALLEL_DRAFTING=1 and a P-EAGLE-trained checkpoint" >&2
  exit 2
fi
if [ "${ALGORITHM}" != "peagle" ] && [ "${PARALLEL_DRAFTING}" = "1" ]; then
  echo "PARALLEL_DRAFTING=1 is not a valid E3/E3.1 conversion; train a P-EAGLE head" >&2
  exit 2
fi
if [ -z "${EAGLE_HEAD}" ]; then
  echo "EAGLE_HEAD is required (directory of a target-specific EAGLE checkpoint)" >&2
  exit 2
fi
if [ ! -d "${EAGLE_HEAD}" ]; then
  echo "EAGLE_HEAD is not a directory: ${EAGLE_HEAD}" >&2
  exit 2
fi

EXPECTED_ALGORITHM="${ALGORITHM}"
if [ "${ALGORITHM}" = "peagle" ]; then
  # P-EAGLE is served through method=eagle3; its architecture flags are not E3.1 norms.
  EXPECTED_ALGORITHM=""
fi

VALIDATE_ARGS=( "${EAGLE_HEAD}" --contract "${TARGET_CONTRACT}" --target-model "${TARGET_MODEL}" )
if [ -n "${EXPECTED_ALGORITHM}" ]; then
  VALIDATE_ARGS+=( --algorithm "${EXPECTED_ALGORITHM}" )
fi
if [ "${VALIDATE_ONLY}" = "1" ]; then
  VALIDATE_ARGS+=( --allow-config-only )
fi
HEAD_REPORT="$(
  PYTHONPATH="${REPO_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}" \
    "${PYTHON}" -m apertus_bench.eagle "${VALIDATE_ARGS[@]}"
)"

SERVED_MODEL="${SERVED_MODEL:-${SERVED_BASE}-${ALGORITHM}-n${NUM_SPECULATIVE_TOKENS}-tp${TARGET_TP}d${DRAFT_TP}-${RUN_SUFFIX}}"
EXTRA_VLLM="$("${SERVING_DIR}/vllm-extra-flags.sh")"

SPEC_FLAGS="--speculative-config.method eagle3 \
    --speculative-config.model ${EAGLE_HEAD} \
    --speculative-config.num_speculative_tokens ${NUM_SPECULATIVE_TOKENS} \
    --speculative-config.draft_tensor_parallel_size ${DRAFT_TP}"
if [ "${PARALLEL_DRAFTING}" = "1" ]; then
  SPEC_FLAGS="${SPEC_FLAGS} --speculative-config.parallel_drafting true"
fi

if [ "${APPLY_VLLM_PATCHES}" = "1" ] && [ -z "${EXTRA_MOUNTS:-}" ]; then
  if [ -f "${IMAGE}" ]; then
    EXTRA_MOUNTS="$(
      "${SERVING_DIR}/patch-vllm.sh" "${IMAGE}" \
        "${REPO_ROOT}/serving/patches/vllm-apertus-image-token.patch" \
        "${REPO_ROOT}/serving/patches/vllm-apertus-eagle3-aux-layers.patch" \
        "${REPO_ROOT}/serving/patches/vllm-apertus-eagle-mrv2-inner.patch"
    )"
    export EXTRA_MOUNTS
  elif [ "${VALIDATE_ONLY}" != "1" ]; then
    echo "container image not found: ${IMAGE}" >&2
    exit 1
  fi
fi

SML_ENVIRONMENT="$("${SERVING_DIR}/resolve-env.sh" "${ENV_SOURCE}")"

cat <<EOF
stage=${STAGE}
served_model=${SERVED_MODEL}
target_model=${TARGET_MODEL}
target_contract=${TARGET_CONTRACT}
target_tensor_parallel_size=${TARGET_TP}
results_root=${RESULTS_ROOT}
eagle_head=${EAGLE_HEAD}
algorithm=${ALGORITHM}
engine_method=eagle3
num_speculative_tokens=${NUM_SPECULATIVE_TOKENS}
draft_tensor_parallel_size=${DRAFT_TP}
parallel_drafting=${PARALLEL_DRAFTING}
max_model_len=${MAX_MODEL_LEN}
gpu_memory_utilization=${GPU_MEMORY_UTILIZATION}
enable_prefix_caching=${ENABLE_PREFIX_CACHING}
async_scheduling=${ASYNC_SCHEDULING:-engine-default}
max_num_batched_tokens=${MAX_NUM_BATCHED_TOKENS:-engine-default}
environment=${SML_ENVIRONMENT}
image=${IMAGE}
apply_vllm_patches=${APPLY_VLLM_PATCHES}
head_report=${HEAD_REPORT}
EOF

if [ "${VALIDATE_ONLY}" = "1" ]; then
  echo "validate_only=1; not submitting"
  exit 0
fi

cd "${MODEL_LAUNCH_ROOT}"
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
    --tensor-parallel-size ${TARGET_TP} \
    --gpu-memory-utilization ${GPU_MEMORY_UTILIZATION} \
    --max-model-len ${MAX_MODEL_LEN} \
    --enable-auto-tool-choice \
    --tool-call-parser apertus \
    ${SPEC_FLAGS} \
    --compilation-config.pass_config.fuse_allreduce_rms false \
    ${EXTRA_VLLM}" \
  "$@"
