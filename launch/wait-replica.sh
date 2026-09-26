#!/usr/bin/env bash
# Wait until an sml replica answers /v1/models and one short chat completion.
# Prints KEY=VALUE lines: JOB NODE IP MODEL LOG_DIR
set -euo pipefail

JOB="${1:?usage: wait-replica.sh JOBID}"
LOG_DIR="${SML_LOG_DIR:-${HOME}/.sml/logs/${JOB}}"
TIMEOUT_SECONDS="${WAIT_REPLICA_TIMEOUT:-2400}"
started="$(date +%s)"

while true; do
  now="$(date +%s)"
  if [ $((now - started)) -ge "${TIMEOUT_SECONDS}" ]; then
    echo "timed out waiting for job ${JOB}" >&2
    exit 1
  fi
  state="$(squeue -j "${JOB}" -h -o '%T' 2>/dev/null || true)"
  if [ -z "${state}" ]; then
    echo "job ${JOB} is no longer in the queue" >&2
    exit 1
  fi
  if [ "${state}" = "RUNNING" ]; then
    break
  fi
  sleep 10
done

IP=""
NODE=""
while true; do
  now="$(date +%s)"
  if [ $((now - started)) -ge "${TIMEOUT_SECONDS}" ]; then
    echo "timed out waiting for replica IP of job ${JOB}" >&2
    exit 1
  fi
  if [ -f "${LOG_DIR}/log.out" ]; then
    IP="$(grep -aoE 'Replica 0 head IP: [0-9.]+' "${LOG_DIR}/log.out" 2>/dev/null | tail -1 | awk '{print $NF}' || true)"
    NODE="$(grep -aoE 'Node 0: nid[0-9]+' "${LOG_DIR}/log.out" 2>/dev/null | tail -1 | awk '{print $NF}' || true)"
  fi
  if [ -n "${IP}" ] && [ -n "${NODE}" ]; then
    break
  fi
  sleep 10
done

BASE="http://${IP}:8080"
MODEL=""
while true; do
  now="$(date +%s)"
  if [ $((now - started)) -ge "${TIMEOUT_SECONDS}" ]; then
    echo "timed out waiting for /v1/models on ${BASE}" >&2
    exit 1
  fi
  if MODEL="$(curl -sf --max-time 15 "${BASE}/v1/models" 2>/dev/null \
      | python3 -c 'import json,sys; print(json.load(sys.stdin)["data"][0]["id"])' 2>/dev/null)"; then
    break
  fi
  sleep 10
done

while true; do
  now="$(date +%s)"
  if [ $((now - started)) -ge "${TIMEOUT_SECONDS}" ]; then
    echo "timed out waiting for a chat completion on ${BASE}" >&2
    exit 1
  fi
  if curl -sf --max-time 120 "${BASE}/v1/chat/completions" \
      -H 'Content-Type: application/json' \
      -d "{\"model\":\"${MODEL}\",\"messages\":[{\"role\":\"user\",\"content\":\"Reply with the single word ready.\"}],\"max_tokens\":8,\"temperature\":0,\"seed\":1}" \
      >/dev/null 2>/dev/null; then
    break
  fi
  sleep 10
done

printf 'JOB=%s\nNODE=%s\nIP=%s\nMODEL=%s\nLOG_DIR=%s\n' "${JOB}" "${NODE}" "${IP}" "${MODEL}" "${LOG_DIR}"
