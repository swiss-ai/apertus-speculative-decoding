#!/usr/bin/env bash
# Domain drafters (A3), EAGLE grid: run on the WORKSTATION (not a login node).
# Every INTERVAL seconds it asks the cluster whether all six domain trace sets are
# READY; when they are, it submits the seven arms once (submit-grid.sh full). It then
# waits for those jobs to leave the queue and copies runs/ and heads/ to capstor
# scratch, which the iopsstor cleaner does not touch. Stops after MAX_HOURS.
#   nohup methods/eagle/launch/domain/watch-and-submit.sh <cluster-repo> > watch.log 2>&1 &
set -uo pipefail
REPO="${1:?cluster snapshot directory}"
HOST="${HOST:-clariden1}"
INTERVAL="${INTERVAL:-600}"
MAX_HOURS="${MAX_HOURS:-36}"
DATA=/iopsstor/scratch/cscs/faruk_zahiragic/spec-exp-data/domain-drafters
BACKUP=/capstor/scratch/cscs/faruk_zahiragic/domain-drafters-backup/eagle
deadline=$(( $(date +%s) + MAX_HOURS * 3600 ))
log() { echo "$(date -u +%FT%TZ) $*"; }

submitted=""
while [ "$(date +%s)" -lt "${deadline}" ]; do
  if [ -z "${submitted}" ]; then
    status="$(ssh -o BatchMode=yes -o ConnectTimeout=30 "${HOST}" \
      "cd ${REPO} && methods/eagle/launch/domain/submit-grid.sh status" 2>&1)" || {
      log "ssh/status failed: ${status}"; sleep "${INTERVAL}"; continue; }
    if ! grep -q "not ready" <<<"$(grep -v '^pilot' <<<"${status}")"; then
      log "all domains ready; submitting"
      out="$(ssh -o BatchMode=yes "${HOST}" \
        "cd ${REPO} && methods/eagle/launch/domain/submit-grid.sh full" 2>&1)"
      echo "${out}"
      submitted="$(grep -o 'Submitted batch job [0-9]*' <<<"${out}" | awk '{print $4}' | paste -sd, -)"
      log "submitted jobs: ${submitted:-none}"
      [ -n "${submitted}" ] || { log "submission produced no job ids; stopping"; exit 1; }
    else
      log "waiting: $(tr '\n' ' ' <<<"${status}")"
    fi
  else
    left="$(ssh -o BatchMode=yes -o ConnectTimeout=30 "${HOST}" "squeue -h -j ${submitted} -o %i 2>/dev/null | wc -l" 2>/dev/null || echo "?")"
    if [ "${left}" = "0" ]; then
      log "grid jobs left the queue; backing up runs and heads"
      ssh -o BatchMode=yes "${HOST}" "mkdir -p ${BACKUP} && rsync -a ${DATA}/eagle/runs ${DATA}/eagle/heads ${BACKUP}/ && echo backup-ok"
      exit 0
    fi
    log "grid jobs still queued/running: ${left}"
  fi
  sleep "${INTERVAL}"
done
log "deadline reached"
