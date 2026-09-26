#!/usr/bin/env bash
# Map DIAG_CONFIG (B0, B1, B2, N3, N3m, D3, ...) onto the existing launchers.
set -euo pipefail

LAUNCH_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "${LAUNCH_DIR}/../.." && pwd)"
SERVING_DIR="${REPO_ROOT}/serving"

if [ "${1:-}" != "" ] && [ "${1:0:1}" != "-" ]; then
  DIAG_CONFIG="$1"
  shift
fi
DIAG_CONFIG="${DIAG_CONFIG:?set DIAG_CONFIG or pass it as the first argument}"

PYTHON="${PYTHON:-python3}"
eval "$(
  PYTHONPATH="${REPO_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}" \
    "${PYTHON}" -c "from apertus_bench.diagnostics import format_env_exports; print(format_env_exports('${DIAG_CONFIG}'))"
)"

RUN_SUFFIX="${RUN_SUFFIX:-$(id -un)-$(date -u +%Y%m%dT%H%M%SZ)}"
SERVED_MODEL="${SERVED_MODEL:-swiss-ai/Apertus-v1.5-70B-diag-${DIAG_CONFIG}-${RUN_SUFFIX}}"
export SERVED_MODEL RUN_SUFFIX
IMAGE="${IMAGE:-/capstor/store/cscs/swissai/infra01/container-images/ci/vllm_apertus_1.5_release-arm64.sqsh}"

if [ "${APPLY_IMAGE_TOKEN_PATCH}" = "1" ] && [ -z "${EXTRA_MOUNTS:-}" ]; then
  if [ -f "${IMAGE}" ]; then
    EXTRA_MOUNTS="$("${SERVING_DIR}/patch-vllm.sh" "${IMAGE}" "${REPO_ROOT}/serving/patches/vllm-apertus-image-token.patch")"
    export EXTRA_MOUNTS
  else
    echo "container image not found for draft overlay: ${IMAGE}" >&2
    exit 1
  fi
fi

echo "diagnostic_config=${DIAG_CONFIG}"
echo "launcher=${LAUNCHER}"
echo "served_model=${SERVED_MODEL}"

case "${LAUNCHER}" in
  baseline.sh) LAUNCHER_PATH="${SERVING_DIR}/baseline.sh" ;;
  ngram.sh) LAUNCHER_PATH="${REPO_ROOT}/methods/ngram/ngram.sh" ;;
  draft-model.sh|standalone-8b.sh) LAUNCHER_PATH="${REPO_ROOT}/methods/draft_model/${LAUNCHER}" ;;
  *) echo "unknown launcher ${LAUNCHER}" >&2; exit 2 ;;
esac
exec "${LAUNCHER_PATH}" "$@"
