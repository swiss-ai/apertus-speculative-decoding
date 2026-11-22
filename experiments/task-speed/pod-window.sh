#!/usr/bin/env bash
# Run live windows from the in-cluster client pod (rob-poc/task-speed-client).
# Run on a Clariden login node from the live checkout, with the Breithorn
# kubeconfig and the corpus there:
#   experiments/task-speed/pod-window.sh stage            # code + corpus + gateway key
#   experiments/task-speed/pod-window.sh start P1 plain 1 [run-live.py options]
#   experiments/task-speed/pod-window.sh log P1
#   experiments/task-speed/pod-window.sh fetch "P1 P2"    # finished windows to Clariden
# The key is piped into a 0600 file in the pod's emptyDir (never in argv or Git);
# the pod identity file is captured here with kubectl, since the client pod has
# no Kubernetes credentials. `start` refreshes it before the window.
set -euo pipefail
CODE="$(cd "$(dirname "$0")/../.." && pwd)"
D=/iopsstor/scratch/cscs/${USER}/apertus-kubernetes-access
export KUBECONFIG="${KUBECONFIG:-${D}/kubeconfig}" PATH="${D}:${PATH}"
K=(kubectl --context breithorn-oidc -n rob-poc)
CLIENT=task-speed-client
APP=vllm-apertus-v1p5-8b-dspark-e10
CORPUS_DIR="${TASK_SPEED_CORPUS_DIR:-${CODE}/results/8b/task-speed/corpus-20261104}"
LIVE="${TASK_SPEED_LIVE:-${CODE}/results/8b/task-speed/live-20261104}"
METRICS="http://${APP}.rob-poc.svc:8080/metrics"

case "${1:?stage|start|log|fetch}" in
  stage)
    if [ -n "$(git -C "${CODE}" status --porcelain --untracked-files=no)" ]; then
      echo "tracked changes in ${CODE}; commit and sync first" >&2; exit 2
    fi
    REVISION="$(git -C "${CODE}" rev-parse HEAD)"
    "${K[@]}" exec "${CLIENT}" -- sh -c 'rm -rf /work/code && mkdir -p /work/code /work/corpus'
    git -C "${CODE}" archive "${REVISION}" src experiments/task-speed serving/kubernetes \
      | "${K[@]}" exec -i "${CLIENT}" -- tar -x -C /work/code
    tar -C "${CORPUS_DIR}" -c corpus.jsonl corpus-manifest.json \
      | "${K[@]}" exec -i "${CLIENT}" -- tar -x -C /work/corpus
    # The pod's live root continues the run measured so far (e.g. D1 from Slurm).
    if [ -d "${LIVE}" ]; then
      "${K[@]}" exec "${CLIENT}" -- sh -c 'test ! -e /work/live' \
        || { echo "pod already holds /work/live; fetch it first" >&2; exit 2; }
      tar -C "${LIVE}" -c . | "${K[@]}" exec -i "${CLIENT}" -- sh -c 'mkdir /work/live && tar -x -C /work/live'
    fi
    printf '%s\n' "branch=$(git -C "${CODE}" branch --show-current)" "revision=${REVISION}" \
      | "${K[@]}" exec -i "${CLIENT}" -- sh -c 'cat > /work/code/REVISION'
    "${HOME}/venvs/sml-apertus/bin/python" -c 'import sys
from swiss_ai_model_launch.cli.configuration import InitConfig
sys.stdout.write(InitConfig.load().get_non_none_value("swissai_research_api_key"))' \
      | "${K[@]}" exec -i "${CLIENT}" -- sh -c 'umask 077; cat > /work/key'
    "${K[@]}" exec "${CLIENT}" -- sh -c 'sha256sum /work/corpus/corpus.jsonl; cat /work/code/REVISION'
    ;;
  start)
    WINDOW="${2:?window}" VARIANT="${3:?variant}" ROUND="${4:?round}"
    shift 4
    "${K[@]}" get pods -l "app=${APP}" -o json \
      | python3 -c 'import json,sys; p=[x for x in json.load(sys.stdin)["items"] if not x["metadata"].get("deletionTimestamp")]; assert len(p)==1, len(p); json.dump(p[0], sys.stdout)' \
      | "${K[@]}" exec -i "${CLIENT}" -- sh -c "cat > /work/pod-${WINDOW}.json"
    SUFFIX="$(case " $* " in *" --direct "*) echo -direct;; esac)"
    "${K[@]}" exec "${CLIENT}" -- sh -c "cd /work && . code/REVISION 2>/dev/null; \
      export TASK_SPEED_BRANCH=\$branch TASK_SPEED_REVISION=\$revision PYTHONPATH=/work/code/src; \
      nohup python3 code/experiments/task-speed/run-live.py --window ${WINDOW} \
        --variant ${VARIANT} --round ${ROUND} --output /work/live \
        --corpus /work/corpus/corpus.jsonl --metrics-url ${METRICS} \
        --pod-json /work/pod-${WINDOW}.json --api-key-file /work/key $* \
        > /work/${WINDOW}${SUFFIX}-\$(date -u +%Y%m%dT%H%M%SZ).log 2>&1 < /dev/null &
      echo started"
    ;;
  log)
    "${K[@]}" exec "${CLIENT}" -- sh -c "tail -n ${3:-20} \$(ls -t /work/${2:?window}*.log | head -1)"
    ;;
  fetch)
    # One compressed stream per finished window (one large exec stream broke off);
    # a window already on Clariden is never overwritten, toggles stay as written here.
    mkdir -p "${LIVE}/pod-logs"
    for window in ${2:?windows to fetch, e.g. "P1 P2"}; do
      test ! -e "${LIVE}/${window}" || { echo "${LIVE}/${window} exists; skipped" >&2; continue; }
      tmp="$(mktemp -d "${LIVE}/.fetch-XXXX")"
      "${K[@]}" exec "${CLIENT}" -- tar -C /work/live -czf - "${window}" | tar -xzf - -C "${tmp}"
      mv "${tmp}/${window}" "${LIVE}/${window}" && rmdir "${tmp}"
    done
    "${K[@]}" exec "${CLIENT}" -- cat /work/live/windows.json > "${LIVE}/windows.json"
    "${K[@]}" exec "${CLIENT}" -- sh -c 'cd /work && tar -c *.log' | tar -x -C "${LIVE}/pod-logs"
    echo "fetched into ${LIVE}"
    ;;
esac
