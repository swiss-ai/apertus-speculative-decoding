#!/usr/bin/env bash
# Launch an Apertus 1.5 target with the colleague's DSpark drafter
# (speculators checkpoint, see docs/CONFIG_DSpark_Apertus_1.5_8B_09_29.md).
#   STAGE=8b DSPARK_CHECKPOINT=.../checkpoint_best ./methods/dspark/launch/dspark.sh
# Target settings come from serving/stage-defaults.sh, as for the baseline and
# EAGLE launchers, so the three arms of one stage are served alike.
set -euo pipefail

LAUNCH_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "${LAUNCH_DIR}/../../.." && pwd)"
SERVING_DIR="${REPO_ROOT}/serving"
MODEL_LAUNCH_ROOT="${MODEL_LAUNCH_ROOT:-../model-launch}"
# shellcheck source=serving/stage-defaults.sh
. "${SERVING_DIR}/stage-defaults.sh"
DSPARK_CHECKPOINT="${DSPARK_CHECKPOINT:-}"
# Block size 8 with the anchor as the first slot: at most 7 draft tokens.
NUM_SPECULATIVE_TOKENS="${NUM_SPECULATIVE_TOKENS:-7}"
SML_PARTITION="${SML_PARTITION:-normal}"
SML_TIME="${SML_TIME:-04:00:00}"
ENV_SOURCE="${ENV_SOURCE:-${MODEL_LAUNCH_ROOT}/src/swiss_ai_model_launch/assets/envs/vllm_apertus_1.5_release.toml}"
IMAGE="${IMAGE:-/capstor/store/cscs/swissai/infra01/container-images/ci/vllm_apertus_1.5_release-arm64.sqsh}"
APPLY_VLLM_PATCHES="${APPLY_VLLM_PATCHES:-1}"
# Its own overlay directory: patch-vllm.sh empties it, and an EAGLE launch may
# be using the default one.
export VLLM_PATCH_DIR="${VLLM_PATCH_DIR:-${HOME}/.sml/vllm-patch-dspark}"
VALIDATE_ONLY="${VALIDATE_ONLY:-0}"
PYTHON="${PYTHON:-python3}"
RUN_SUFFIX="${RUN_SUFFIX:-$(id -un)-$(date -u +%Y%m%dT%H%M%SZ)}"

case "${NUM_SPECULATIVE_TOKENS}" in
  1|2|3|4|5|6|7) ;;
  *) echo "NUM_SPECULATIVE_TOKENS must be 1..7 for a block-8 DSpark drafter" >&2; exit 2 ;;
esac
if [ -z "${DSPARK_CHECKPOINT}" ]; then
  echo "DSPARK_CHECKPOINT is required (speculators DSpark checkpoint directory)" >&2
  exit 2
fi
if [ ! -f "${DSPARK_CHECKPOINT}/config.json" ]; then
  echo "no config.json in DSPARK_CHECKPOINT: ${DSPARK_CHECKPOINT}" >&2
  exit 2
fi

# The fields the serving path depends on; fail before submitting if the
# checkpoint is not a DSpark speculator.
CHECKPOINT_REPORT="$(
  "${PYTHON}" - "${DSPARK_CHECKPOINT}" <<'PY'
import hashlib, json, pathlib, sys
root = pathlib.Path(sys.argv[1])
raw = (root / "config.json").read_bytes()
config = json.loads(raw)
kind = config.get("speculators_model_type")
if kind != "dspark":
    sys.exit(f"speculators_model_type is {kind!r}, not 'dspark'")
weights = sorted(p.name for p in root.glob("*.safetensors"))
if not weights:
    sys.exit(f"no *.safetensors in {root}")
report = {
    "config_sha256": hashlib.sha256(raw).hexdigest(),
    "weights": {name: (root / name).stat().st_size for name in weights},
}
for key in ("aux_hidden_state_layer_ids", "block_size", "sample_from_anchor",
            "num_hidden_layers", "draft_vocab_size", "markov_rank",
            "enable_confidence_head"):
    if key in config:
        report[key] = config[key]
print(json.dumps(report, sort_keys=True))
PY
)"

SERVED_MODEL="${SERVED_MODEL:-${SERVED_BASE}-dspark-n${NUM_SPECULATIVE_TOKENS}-tp${TARGET_TP}-${RUN_SUFFIX}}"
EXTRA_VLLM="$("${SERVING_DIR}/vllm-extra-flags.sh")"

SPEC_FLAGS="--speculative-config.method dspark \
    --speculative-config.model ${DSPARK_CHECKPOINT} \
    --speculative-config.num_speculative_tokens ${NUM_SPECULATIVE_TOKENS}"

if [ "${APPLY_VLLM_PATCHES}" = "1" ] && [ -z "${EXTRA_MOUNTS:-}" ]; then
  if [ -f "${IMAGE}" ]; then
    EXTRA_MOUNTS="$(
      "${SERVING_DIR}/patch-vllm.sh" "${IMAGE}" \
        "${REPO_ROOT}/serving/patches/vllm-apertus-image-token.patch" \
        "${REPO_ROOT}/serving/patches/vllm-apertus-eagle3-aux-layers.patch" \
        "${REPO_ROOT}/serving/patches/vllm-apertus-dspark-mrv2-inner.patch" \
        "${REPO_ROOT}/serving/patches/vllm-apertus-dspark-anchor-layout.patch"
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
dspark_checkpoint=${DSPARK_CHECKPOINT}
engine_method=dspark
num_speculative_tokens=${NUM_SPECULATIVE_TOKENS}
max_model_len=${MAX_MODEL_LEN}
gpu_memory_utilization=${GPU_MEMORY_UTILIZATION}
enable_prefix_caching=${ENABLE_PREFIX_CACHING}
async_scheduling=${ASYNC_SCHEDULING:-engine-default}
max_num_batched_tokens=${MAX_NUM_BATCHED_TOKENS:-engine-default}
max_num_seqs=${MAX_NUM_SEQS:-engine-default}
environment=${SML_ENVIRONMENT}
image=${IMAGE}
apply_vllm_patches=${APPLY_VLLM_PATCHES}
checkpoint_report=${CHECKPOINT_REPORT}
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
