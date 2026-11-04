#!/usr/bin/env bash
# Start one live window (run-live.py) on Clariden's xfer partition, detached.
# Run on a login node from the task-speed checkout on scratch, after the toggle
# for this window is verified (toggle.md):
#   experiments/task-speed/run-live-window.sh D1 dspark 1 results/8b/task-speed/live-<stamp> \
#     results/8b/task-speed/controlled-<stamp>/corpus.jsonl [run-live.py options]
# The gateway key is read from SML here (its venv is aarch64) and reaches the
# client only through the environment of the srun step; it is never written.
set -euo pipefail
WINDOW="${1:?window}" VARIANT="${2:?variant}" ROUND="${3:?round}"
LIVE="$(realpath -m "${4:?live root}")" CORPUS="$(realpath "${5:?corpus}")"
shift 5
CODE="$(cd "$(dirname "$0")/../.." && pwd)"
CLIENT="${TASK_SPEED_CLIENT_ROOT:-/iopsstor/scratch/cscs/${USER}/apertus-task-speed-client}"
export KUBECONFIG="${KUBECONFIG:-/iopsstor/scratch/cscs/${USER}/apertus-kubernetes-access/kubeconfig}"
CONTEXT="${TASK_SPEED_KUBE_CONTEXT:-breithorn-oidc}"
test -x "${CLIENT}/venv/bin/python" || { echo "run setup-live-client.sh first" >&2; exit 2; }
if [ -n "$(git -C "${CODE}" status --porcelain --untracked-files=no)" ]; then
  echo "tracked changes in ${CODE}; commit and sync first" >&2
  exit 2
fi
# kubectl on the login node is the aarch64 copy staged for the Kubernetes work.
LOGIN_KUBECTL="${TASK_SPEED_LOGIN_KUBECTL:-/iopsstor/scratch/cscs/${USER}/apertus-kubernetes-access/kubectl}"
"${LOGIN_KUBECTL}" --context "${CONTEXT}" -n rob-poc auth can-i create pods/exec | grep -qx yes \
  || { echo "kube access expired or missing (ACCESS.md)" >&2; exit 2; }
if [ -z "${CSCS_SERVING_API:-}" ]; then
  CSCS_SERVING_API="$("${HOME}/venvs/sml-apertus/bin/python" -c 'import sys
from swiss_ai_model_launch.cli.configuration import InitConfig
sys.stdout.write(InitConfig.load().get_non_none_value("swissai_research_api_key"))')"
fi
export CSCS_SERVING_API
mkdir -p "${LIVE}"
LOG="${LIVE}/${WINDOW}$(case " $* " in *" --direct "*) echo -direct;; esac).log"
test ! -e "${LOG}" || { echo "${LOG} exists; windows are never rerun in place" >&2; exit 2; }
BRANCH="$(git -C "${CODE}" branch --show-current)"
REVISION="$(git -C "${CODE}" rev-parse HEAD)"
nohup setsid srun -p xfer -A "${TASK_SPEED_ACCOUNT:-infra01}" -t "${TASK_SPEED_TIME:-04:00:00}" \
  -n1 -c8 --job-name="task-speed-${WINDOW}" \
  env PYTHONPATH="${CODE}/src" PYTHONNOUSERSITE=1 TASK_SPEED_BRANCH="${BRANCH}" \
  TASK_SPEED_REVISION="${REVISION}" \
  "${CLIENT}/venv/bin/python" "${CODE}/experiments/task-speed/run-live.py" \
  --window "${WINDOW}" --variant "${VARIANT}" --round "${ROUND}" --output "${LIVE}" \
  --corpus "${CORPUS}" --kubectl "${CLIENT}/kubectl" --context "${CONTEXT}" "$@" \
  > "${LOG}" 2>&1 < /dev/null &
echo "window=${WINDOW} pid=$! log=${LOG}"
