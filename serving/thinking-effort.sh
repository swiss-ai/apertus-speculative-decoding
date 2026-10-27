#!/usr/bin/env bash
# Launch the fixed native-effort experiment via Slurm. See experiments/thinking-effort/README.md.
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
export PYTHONPATH="${REPO_ROOT}/src:${PYTHONPATH:-}"
exec python3 -m apertus_bench.effort_submit "$@"
