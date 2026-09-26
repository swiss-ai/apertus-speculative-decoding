#!/usr/bin/env bash
# Tokenizer/BOS probe on login CPU. Does not talk to the D3 replica.
set -euo pipefail
LAUNCH_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "${LAUNCH_DIR}/.." && pwd)"
PYTHON="${PYTHON:-${HOME}/miniconda/envs/apertus-bench/bin/python}"
# Login python has pyarrow; the conda env has transformers/tokenizers.
LOOKUP_PYTHON="${LOOKUP_PYTHON:-python3}"
PARQUET_DIR="${PARQUET_DIR:-/capstor/store/cscs/swissai/infra01/datasets/Apertus-1.5-SFT-mix-pretrain-v1/data}"
INPUT="${INPUT:-${REPO_ROOT}/results/eagle/data/overfit.jsonl}"
OUTPUT="${OUTPUT:-${REPO_ROOT}/results/eagle/preflight/renderer-parity.json}"
SIDECAR="${MIX_TEXT_JSON:-${REPO_ROOT}/results/eagle/preflight/mix-text-by-id.json}"
export PYTHONPATH="${REPO_ROOT}/training${PYTHONPATH:+:${PYTHONPATH}}"
mkdir -p "$(dirname "${OUTPUT}")"
if [ ! -s "${SIDECAR}" ]; then
  echo "looking up mix text with ${LOOKUP_PYTHON} (pyarrow)"
  "${LOOKUP_PYTHON}" -m apertus_eagle.prepare_data \
    --lookup-ids-from "${INPUT}" \
    --parquet-dir "${PARQUET_DIR}" \
    --lookup-output "${SIDECAR}" \
    || echo "parquet lookup skipped; probe will reconstruct mix control spans"
fi
PROBE_ARGS=(--input "${INPUT}" --limit 32 --output "${OUTPUT}")
if [ -s "${SIDECAR}" ]; then
  PROBE_ARGS+=(--mix-text-json "${SIDECAR}")
fi
"${PYTHON}" -m apertus_eagle.probe_renderer "${PROBE_ARGS[@]}"
