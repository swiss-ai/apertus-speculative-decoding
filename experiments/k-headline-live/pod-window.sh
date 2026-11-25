#!/usr/bin/env bash
# Live k=3/5/7 windows from the in-cluster client pod (rob-poc/task-speed-client),
# driven from the workstation with the operator's own kubeconfig (context
# breithorn-oidc). Requests go straight to the DSpark Service, never through the
# shared gateway, so no gateway key is staged.
#   experiments/k-headline-live/pod-window.sh stage           # code + corpus into the pod
#   experiments/k-headline-live/pod-window.sh start K5 dspark-k5 1
#   experiments/k-headline-live/pod-window.sh status K5       # last log lines, running?
#   experiments/k-headline-live/pod-window.sh fetch K5        # finished window -> $K_LIVE_ROOT
# The prompt corpus holds restricted text: it is streamed from Clariden into the
# pod's emptyDir and never written on the workstation or into Git.
set -euo pipefail
CODE="$(cd "$(dirname "$0")/../.." && pwd)"
K=(kubectl --context breithorn-oidc -n rob-poc)
CLIENT=task-speed-client
APP=vllm-apertus-v1p5-8b-dspark-e10
SERVICE="http://${APP}.rob-poc.svc:8080"
CORPUS_HOST="${K_LIVE_CORPUS_HOST:-clariden}"
CORPUS_DIR="${K_LIVE_CORPUS_DIR:-/iopsstor/scratch/cscs/faruk_zahiragic/k-headline/code/results/8b/k-headline/controlled-20261122T172956Z}"
CORPUS_SHA256=7d8e60ef1ae813b2201e71ffe5bd3358da6625e64805a51ec2c869d4a68c833d

case "${1:?stage|start|status|fetch}" in
  stage)
    if [ -n "$(git -C "${CODE}" status --porcelain --untracked-files=no)" ]; then
      echo "tracked changes in ${CODE}; commit first" >&2; exit 2
    fi
    REVISION="$(git -C "${CODE}" rev-parse HEAD)"
    git -C "${CODE}" merge-base --is-ancestor "${REVISION}" "@{upstream}" \
      || { echo "push ${REVISION} first (provenance must point at a pushed commit)" >&2; exit 2; }
    "${K[@]}" exec "${CLIENT}" -- sh -c 'test ! -e /work/live' \
      || { echo "pod already holds /work/live" >&2; exit 2; }
    "${K[@]}" exec "${CLIENT}" -- sh -c 'rm -rf /work/code && mkdir -p /work/code /work/corpus'
    git -C "${CODE}" archive "${REVISION}" src experiments/task-speed experiments/k-headline-live \
      serving/kubernetes | "${K[@]}" exec -i "${CLIENT}" -- tar -x -C /work/code
    ssh -o BatchMode=yes "${CORPUS_HOST}" \
      "tar -C '${CORPUS_DIR}' -c corpus.jsonl corpus-manifest.json" \
      | "${K[@]}" exec -i "${CLIENT}" -- tar -x -C /work/corpus
    printf '%s\n' "branch=$(git -C "${CODE}" branch --show-current)" "revision=${REVISION}" \
      | "${K[@]}" exec -i "${CLIENT}" -- sh -c 'cat > /work/code/REVISION'
    got="$("${K[@]}" exec "${CLIENT}" -- sha256sum /work/corpus/corpus.jsonl | cut -d' ' -f1)"
    [ "${got}" = "${CORPUS_SHA256}" ] || { echo "corpus sha256 ${got} != ${CORPUS_SHA256}" >&2; exit 2; }
    echo "staged ${REVISION}, corpus ${got}"
    ;;
  start)
    WINDOW="${2:?window}" VARIANT="${3:?variant}" ROUND="${4:?round}"
    shift 4
    # The pod identity is captured here (the client pod has no credentials).
    "${K[@]}" get pods -l "app=${APP}" -o json \
      | python3 -c 'import json,sys; p=[x for x in json.load(sys.stdin)["items"] if not x["metadata"].get("deletionTimestamp")]; assert len(p)==1, len(p); json.dump(p[0], sys.stdout)' \
      | "${K[@]}" exec -i "${CLIENT}" -- sh -c "cat > /work/pod-${WINDOW}.json"
    "${K[@]}" exec "${CLIENT}" -- sh -c "cd /work && . code/REVISION; \
      export TASK_SPEED_BRANCH=\$branch TASK_SPEED_REVISION=\$revision PYTHONPATH=/work/code/src; \
      nohup python3 code/experiments/task-speed/run-live.py --window ${WINDOW} \
        --variant ${VARIANT} --round ${ROUND} --output /work/live --direct \
        --corpus /work/corpus/corpus.jsonl --metrics-url ${SERVICE}/metrics \
        --pod-json /work/pod-${WINDOW}.json $* \
        > /work/${WINDOW}-\$(date -u +%Y%m%dT%H%M%SZ).log 2>&1 < /dev/null &
      echo started"
    ;;
  status)
    "${K[@]}" exec "${CLIENT}" -- sh -c "tail -n ${3:-5} \$(ls -t /work/${2:?window}-*.log | head -1); \
      if pgrep -f '[r]un-live.py --window ${2}' >/dev/null; then echo RUNNING; else echo STOPPED; fi"
    ;;
  fetch)
    LIVE="${K_LIVE_ROOT:?set K_LIVE_ROOT to the local live root}"
    mkdir -p "${LIVE}/pod-logs"
    for window in ${2:?windows, e.g. "K3 P"}; do
      name="${window}-direct"
      test ! -e "${LIVE}/${name}" || { echo "${LIVE}/${name} exists; skipped" >&2; continue; }
      tmp="$(mktemp -d "${LIVE}/.fetch-XXXX")"
      "${K[@]}" exec "${CLIENT}" -- tar -C /work/live -czf - "${name}" | tar -xzf - -C "${tmp}"
      mv "${tmp}/${name}" "${LIVE}/${name}" && rmdir "${tmp}"
    done
    for f in windows.json suite.yaml corpus-manifest.json; do
      "${K[@]}" exec "${CLIENT}" -- cat "/work/live/${f}" > "${LIVE}/${f}"
    done
    "${K[@]}" exec "${CLIENT}" -- sh -c 'cd /work && tar -c *.log pod-*.json code/REVISION' \
      | tar -x -C "${LIVE}/pod-logs"
    echo "fetched into ${LIVE}"
    ;;
esac
