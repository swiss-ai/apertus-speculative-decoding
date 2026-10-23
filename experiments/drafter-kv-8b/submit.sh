#!/usr/bin/env bash
# Explicitly submit one already-reserved allocation. Never starts another phase.
# Reserve first using apertus_bench.drafter_kv_experiment reserve. Unknown sbatch
# outcomes retain the reservation; use the recorded comment to reconcile them.
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
if [ "$#" != 3 ]; then
  echo "usage: $0 CAMPAIGN ALLOCATION GENERATED_SPEC.yaml" >&2
  exit 2
fi
export PYTHONPATH="${REPO_ROOT}/src:${PYTHONPATH:-}"
exec python3 -m apertus_bench.drafter_kv_experiment submit "$1" \
  --allocation "$2" --spec "$3" --repo "${REPO_ROOT}"
