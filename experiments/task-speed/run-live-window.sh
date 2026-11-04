#!/usr/bin/env bash
# Start one live window job (run-live.py) on a Clariden debug node, detached.
# Run on a login node from the task-speed checkout on scratch, after the toggle
# for this window is verified (toggle.md):
#   experiments/task-speed/run-live-window.sh D1 dspark 1 results/8b/task-speed/live-<stamp> \
#     results/8b/task-speed/controlled-<stamp>/corpus.jsonl [run-live.py options]
# The gateway key is read from SML here and reaches the client only through the
# environment of the srun step; it is never written. Debug jobs end after 1.5 h:
# the client stops starting cells after TASK_SPEED_BUDGET minutes, and starting
# the same window again measures only its missing cells.
set -euo pipefail
WINDOW="${1:?window}" VARIANT="${2:?variant}" ROUND="${3:?round}"
LIVE="$(realpath -m "${4:?live root}")" CORPUS="$(realpath "${5:?corpus}")"
shift 5
CODE="$(cd "$(dirname "$0")/../.." && pwd)"
PYTHON="${TASK_SPEED_PYTHON:-${HOME}/venvs/apertus-bench/bin/python}"
KUBECTL="${TASK_SPEED_KUBECTL:-/iopsstor/scratch/cscs/${USER}/apertus-kubernetes-access/kubectl}"
export KUBECONFIG="${KUBECONFIG:-/iopsstor/scratch/cscs/${USER}/apertus-kubernetes-access/kubeconfig}"
# kubectl credential plugins (e.g. kubectl-oidc_login) staged next to kubectl.
export PATH="$(dirname "${KUBECTL}"):${PATH}"
CONTEXT="${TASK_SPEED_KUBE_CONTEXT:-breithorn-oidc}"
if [ -n "$(git -C "${CODE}" status --porcelain --untracked-files=no)" ]; then
  echo "tracked changes in ${CODE}; commit and sync first" >&2
  exit 2
fi
"${KUBECTL}" --context "${CONTEXT}" -n rob-poc auth can-i create pods/exec | grep -qx yes \
  || { echo "kube access expired or missing (ACCESS.md)" >&2; exit 2; }
if [ -z "${CSCS_SERVING_API:-}" ]; then
  CSCS_SERVING_API="$("${HOME}/venvs/sml-apertus/bin/python" -c 'import sys
from swiss_ai_model_launch.cli.configuration import InitConfig
sys.stdout.write(InitConfig.load().get_non_none_value("swissai_research_api_key"))')"
fi
export CSCS_SERVING_API
mkdir -p "${LIVE}"
LOG="${LIVE}/${WINDOW}$(case " $* " in *" --direct "*) echo -direct;; esac)-$(date -u +%Y%m%dT%H%M%SZ).log"
BRANCH="$(git -C "${CODE}" branch --show-current)"
REVISION="$(git -C "${CODE}" rev-parse HEAD)"
nohup setsid srun -p "${TASK_SPEED_PARTITION:-debug}" -A "${TASK_SPEED_ACCOUNT:-infra01}" \
  -t "${TASK_SPEED_TIME:-01:30:00}" -N1 -n1 --job-name="task-speed-${WINDOW}" \
  env PYTHONPATH="${CODE}/src" PYTHONNOUSERSITE=1 TASK_SPEED_BRANCH="${BRANCH}" \
  TASK_SPEED_REVISION="${REVISION}" KUBECONFIG="${KUBECONFIG}" PATH="${PATH}" \
  "${PYTHON}" "${CODE}/experiments/task-speed/run-live.py" \
  --window "${WINDOW}" --variant "${VARIANT}" --round "${ROUND}" --output "${LIVE}" \
  --corpus "${CORPUS}" --kubectl "${KUBECTL}" --context "${CONTEXT}" \
  --budget-minutes "${TASK_SPEED_BUDGET:-78}" "$@" \
  > "${LOG}" 2>&1 < /dev/null &
echo "window=${WINDOW} pid=$! log=${LOG}"
