#!/usr/bin/env bash
# Reproducible DSpark research pilot. See experiments/serving-pilot/README.md.
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
export PYTHONPATH="${REPO_ROOT}/src:${PYTHONPATH:-}"
PILOT_PYTHON="${PILOT_PYTHON:-}"
if [[ -z "${PILOT_PYTHON}" ]]; then
  if [[ -x "${REPO_ROOT}/.venv/bin/python" ]]; then
    PILOT_PYTHON="${REPO_ROOT}/.venv/bin/python"
  elif [[ -x "${HOME}/venvs/apertus-bench/bin/python" ]]; then
    PILOT_PYTHON="${HOME}/venvs/apertus-bench/bin/python"
  else
    PILOT_PYTHON=python3
  fi
fi
exec "${PILOT_PYTHON}" -m apertus_bench.research_pilot "$@"
