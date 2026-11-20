#!/usr/bin/env bash
# Local watcher (runs on the workstation, not a login node): every POLL seconds
# it looks for the data campaign's ready markers over ssh and submits, once
#   READY-pilot           -> WHAT=smoke AFTER=<runtime job>
#   READY-<every domain>  -> WHAT=full  AFTER=<smoke job, else runtime job>
# from the cluster checkout. State (job ids) lives in STATE_DIR; delete a file
# there to allow a resubmission. Exits after the full submission.
#   RUNTIME_JOB=<id> nohup methods/dspark/train/watch-domain-data.sh &
set -uo pipefail
HOST="${HOST:-clariden1}"
CHECKOUT="${CHECKOUT:-/iopsstor/scratch/cscs/faruk_zahiragic/spec-exp/domain-dspark}"
TRACES="${TRACES:-/iopsstor/scratch/cscs/faruk_zahiragic/spec-exp-data/domain-drafters/traces}"
STATE_DIR="${STATE_DIR:-${HOME}/.local/state/domain-drafters-dspark}"
POLL="${POLL:-300}"
DOMAINS="math code chat summarization multilingual tool"
: "${RUNTIME_JOB:?}"
mkdir -p "${STATE_DIR}"
log() { echo "[$(date -u +%FT%TZ)] $*" >> "${STATE_DIR}/watch.log"; }
remote() { timeout 120 ssh -o BatchMode=yes "${HOST}" "$@"; }
ready() {  # ready NAME: any of the marker spellings
  case "$1" in
    pilot) remote "test -e ${TRACES}/pilot/READY-pilot -o -e ${TRACES}/READY-pilot" ;;
    *) remote "test -e ${TRACES}/READY-$1 -o -e ${TRACES}/$1/READY-$1 -o -e ${TRACES}/$1/READY" ;;
  esac
}
submit() {  # submit WHAT AFTER -> job id
  remote "cd ${CHECKOUT} && WHAT=$1 AFTER=$2 ./methods/dspark/train/submit-domain-drafters.sh" 2>&1 \
    | tee -a "${STATE_DIR}/watch.log" | sed -n "s/^$1 \(baseline=\)\{0,1\}\([0-9]*\).*/\2/p" | tail -1
}
log "watching (runtime ${RUNTIME_JOB})"
while true; do
  if [ ! -s "${STATE_DIR}/smoke" ] && ready pilot; then
    id="$(submit smoke "${RUNTIME_JOB}")"
    [ -n "${id}" ] && echo "${id}" > "${STATE_DIR}/smoke" && log "smoke submitted ${id}"
  fi
  all=1
  for d in ${DOMAINS}; do ready "${d}" || { all=0; break; }; done
  if [ "${all}" = 1 ] && [ ! -s "${STATE_DIR}/full" ]; then
    after="$(cat "${STATE_DIR}/smoke" 2> /dev/null || echo "${RUNTIME_JOB}")"
    id="$(submit full "${after}")"
    [ -n "${id}" ] && echo "${id}" > "${STATE_DIR}/full" && log "full submitted, baseline ${id}" && exit 0
  fi
  sleep "${POLL}"
done
