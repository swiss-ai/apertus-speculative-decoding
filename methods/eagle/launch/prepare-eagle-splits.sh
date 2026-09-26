#!/usr/bin/env bash
# Materialize hashed SFT-mix splits on a Clariden login node. CPU only.
set -euo pipefail
LAUNCH_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "${LAUNCH_DIR}/../../.." && pwd)"
SERVING_DIR="${REPO_ROOT}/serving"
PYTHON="${PYTHON:-python3}"
PARQUET_DIR="${PARQUET_DIR:-/capstor/store/cscs/swissai/infra01/datasets/Apertus-1.5-SFT-mix-pretrain-v1/data}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/results/70b/eagle/data}"
export PYTHONPATH="${REPO_ROOT}/methods/eagle${PYTHONPATH:+:${PYTHONPATH}}"
"${PYTHON}" -m apertus_eagle.prepare_data \
  --from-parquet \
  --parquet-dir "${PARQUET_DIR}" \
  --output-dir "${OUTPUT_DIR}" \
  --seed 1
