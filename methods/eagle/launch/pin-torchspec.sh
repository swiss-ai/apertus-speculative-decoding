#!/usr/bin/env bash
# Clone or reset TorchSpec to the experiment pin. Safe to re-run.
set -euo pipefail
LAUNCH_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "${LAUNCH_DIR}/../../.." && pwd)"
SERVING_DIR="${REPO_ROOT}/serving"
TORCHSPEC_ROOT="${TORCHSPEC_ROOT:-${REPO_ROOT}/scratch/TorchSpec}"
PIN="${TORCHSPEC_PIN:-6c042a87140a84d13839e341ece2c5c3ada918bc}"
mkdir -p "$(dirname "${TORCHSPEC_ROOT}")"
if [ -d "${TORCHSPEC_ROOT}/.git" ]; then
  git -C "${TORCHSPEC_ROOT}" fetch --depth 1 origin "${PIN}"
  git -C "${TORCHSPEC_ROOT}" checkout --force "${PIN}"
elif [ -d "${TORCHSPEC_ROOT}/torchspec" ] && [ ! -d "${TORCHSPEC_ROOT}/.git" ]; then
  echo "snapshot at ${TORCHSPEC_ROOT} has no git metadata; cloning pin beside it" >&2
  TMP="${TORCHSPEC_ROOT}.pin-${PIN:0:8}"
  rm -rf "${TMP}"
  git clone --filter=blob:none https://github.com/lightseekorg/TorchSpec.git "${TMP}"
  git -C "${TMP}" checkout --force "${PIN}"
  rm -rf "${TORCHSPEC_ROOT}"
  mv "${TMP}" "${TORCHSPEC_ROOT}"
else
  git clone --filter=blob:none https://github.com/lightseekorg/TorchSpec.git "${TORCHSPEC_ROOT}"
  git -C "${TORCHSPEC_ROOT}" checkout --force "${PIN}"
fi
git -C "${TORCHSPEC_ROOT}" rev-parse HEAD
echo "TorchSpec pin ${PIN} at ${TORCHSPEC_ROOT}"
