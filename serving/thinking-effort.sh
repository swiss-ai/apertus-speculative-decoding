#!/usr/bin/env bash
# Launch the fixed native-effort experiment via Slurm. See experiments/thinking-effort/README.md.
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
export PYTHONPATH="${REPO_ROOT}/src:${PYTHONPATH:-}"
EFFORT_PYTHON="${EFFORT_PYTHON:-}"
if [[ -z "$EFFORT_PYTHON" ]]; then
    if [[ -x "${REPO_ROOT}/.venv/bin/python" ]]; then
        EFFORT_PYTHON="${REPO_ROOT}/.venv/bin/python"
    elif [[ -x "${HOME}/venvs/apertus-bench/bin/python" ]]; then
        EFFORT_PYTHON="${HOME}/venvs/apertus-bench/bin/python"
    else
        EFFORT_PYTHON=python3
    fi
fi
exec "$EFFORT_PYTHON" -m apertus_bench.effort_submit "$@"
