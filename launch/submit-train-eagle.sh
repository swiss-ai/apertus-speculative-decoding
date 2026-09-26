#!/usr/bin/env bash
# Compatibility wrapper: the training config must be explicit.
set -euo pipefail
LAUNCH_DIR="$(cd "$(dirname "$0")" && pwd)"
: "${EAGLE_TRAIN_CONFIG:?set EAGLE_TRAIN_CONFIG; stage labels no longer select a 70B config}"
exec "${LAUNCH_DIR}/submit-eagle-train.sh"
