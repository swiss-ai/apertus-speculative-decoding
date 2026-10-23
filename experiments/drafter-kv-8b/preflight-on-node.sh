#!/usr/bin/env bash
# Hash/audit only: no serving deployment or training is started here.
set -euo pipefail
REPO="${1:?isolated checkout}"
CAMPAIGN="${2:?campaign directory}"
OUT="${3:?preflight output directory}"
export PYTHONNOUSERSITE=1
export PYTHONPATH="${REPO}/src:${REPO}/serving:${PYTHONPATH:-}"
IMAGE=/capstor/store/cscs/swissai/infra01/container-images/ci/vllm_apertus_1.5_release-arm64.sqsh
cd "${REPO}"
for head in off on eagle; do
  image_args=()
  if [ "${head}" = off ]; then image_args=(--image "${IMAGE}"); fi
  python3 -m apertus_bench.drafter_kv_experiment freeze "${CAMPAIGN}" \
    --repo "${REPO}" --head "${head}" --output "${OUT}/${head}-inputs.json" \
    --hash-weights "${image_args[@]}" > "${OUT}/${head}-freeze.log"
done
python3 "${REPO}/experiments/drafter-kv-8b/preflight.py" \
  --output "${OUT}/source-audit" \
  --source-manifest "${REPO}/serving/draft_kv/pinned-source.json" \
  --gpu-check > "${OUT}/audit.log" 2>&1
