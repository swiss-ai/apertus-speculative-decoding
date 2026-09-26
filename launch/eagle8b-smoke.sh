#!/usr/bin/env bash
# A4.4 serving smoke for Stage A: launch one independent deployment, prove it
# loads, warms up and answers, record provenance, capture a few greedy outputs,
# read the speculative counters, then cancel it. Run it twice per arm for the
# two independent load/restart cycles.
#   ARM=eagle EAGLE_HEAD=... ./launch/eagle8b-smoke.sh
#   ARM=baseline ./launch/eagle8b-smoke.sh
# Runs on a Clariden login node; the client talks to the replica IP directly.
set -euo pipefail

LAUNCH_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "${LAUNCH_DIR}/.." && pwd)"
export STAGE=8b
export MODEL_LAUNCH_ROOT="${MODEL_LAUNCH_ROOT:-${HOME}/model-launch-apertus}"
export PATH="${HOME}/venvs/sml-apertus/bin:${PATH}"
export SML_PARTITION="${SML_PARTITION:-debug}"
export SML_TIME="${SML_TIME:-01:00:00}"
ARM="${ARM:?set ARM=eagle or ARM=baseline}"
BENCH="${BENCH:-${HOME}/venvs/apertus-bench/bin/apertus-bench}"
PROMPTS="${PROMPTS:-${REPO_ROOT}/results/eagle/data/heldout_test.jsonl}"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
OUT="${REPO_ROOT}/results/eagle/8b/serving-smoke/${ARM}-${STAMP}"
mkdir -p "${OUT}"
# One driver at a time: job 3509784 died because a second launch's
# patch-vllm.sh wiped the overlay directory its container was mounting.
exec 9>"${HOME}/.sml/eagle8b-smoke.lock"
flock -n 9 || { echo "another eagle8b-smoke.sh is running" >&2; exit 3; }
# Per-deployment overlay copies, never shared with a live replica.
export VLLM_PATCH_DIR="${HOME}/.sml/vllm-patch-${ARM}-${STAMP}"

case "${ARM}" in
  eagle) LAUNCHER="${LAUNCH_DIR}/eagle.sh" ;;
  baseline) LAUNCHER="${LAUNCH_DIR}/baseline.sh" ;;
  *) echo "ARM must be eagle or baseline" >&2; exit 2 ;;
esac

RUN_SUFFIX="smoke-${STAMP}" "${LAUNCHER}" --no-tui > "${OUT}/launch.log" 2>&1
# Take the id from sml's own output: a squeue before/after diff also catches
# unrelated jobs submitted meanwhile (it once picked training job 3509855).
JOB="$(grep -aoE 'Job submitted: [0-9]+' "${OUT}/launch.log" | tail -1 | awk '{print $NF}')"
[ -n "${JOB}" ] || { echo "no job id; see ${OUT}/launch.log" >&2; exit 1; }
echo "job=${JOB}" | tee "${OUT}/job.txt"
cleanup() { scancel "${JOB}" 2>/dev/null || true; }
trap cleanup EXIT

started="$(date +%s)"
if ! INFO="$("${LAUNCH_DIR}/wait-replica.sh" "${JOB}")"; then
  echo "replica never became ready" | tee "${OUT}/status.txt"
  cp "${HOME}/.sml/logs/${JOB}/"*.out "${OUT}/" 2>/dev/null || true
  exit 1
fi
ready_s=$(( $(date +%s) - started ))
echo "${INFO}" > "${OUT}/replica.txt"
eval "${INFO}"
BASE="http://${IP}:8080"
echo "ready_seconds=${ready_s}" >> "${OUT}/replica.txt"

curl -sS --max-time 60 "${BASE}/v1/models" > "${OUT}/v1-models.json"
curl -sS --max-time 300 "${BASE}/v1/chat/completions" -H 'Content-Type: application/json' \
  -d "{\"model\":\"${MODEL}\",\"messages\":[{\"role\":\"user\",\"content\":\"In one sentence, what is speculative decoding?\"}],\"max_tokens\":64,\"temperature\":0,\"seed\":1}" \
  > "${OUT}/first-chat-completion.json"
head -32 "${PROMPTS}" | python3 -c '
import json, sys
for line in sys.stdin:
    row = json.loads(line)
    msgs = [m for m in row["messages"] if m["role"] != "developer"]
    print(json.dumps({"id": row["id"], "workload": row.get("domain") or "heldout", "messages": msgs, "max_tokens": 256}))
' > "${OUT}/capture-prompts.jsonl"
"${BENCH}" capture --base-url "${BASE}" --model "${MODEL}" \
  --workloads "${OUT}/capture-prompts.jsonl" --output "${OUT}/capture.json" > "${OUT}/capture.log" 2>&1 || true
curl -sS --max-time 60 "${BASE}/metrics" > "${OUT}/metrics.txt" || true
grep -E '^vllm:spec_decode' "${OUT}/metrics.txt" > "${OUT}/spec-decode-metrics.txt" || true
LOGS="${HOME}/.sml/logs/${JOB}"
cp "${LOGS}/log.out" "${OUT}/sml-log.out" 2>/dev/null || true
if [ -f "${LOGS}/replica_0.out" ]; then
  grep -aE 'speculative|eagle|Eagle|aux_hidden|draft|share|non-default args|KV cache|Loading weights took|Model loading took|Error|error' \
    "${LOGS}/replica_0.out" | head -200 > "${OUT}/engine-excerpt.txt" || true
fi
nvidia_mem="$(grep -aoE 'Model loading took [0-9.]+ Gi?B' "${OUT}/engine-excerpt.txt" | head -2 | tr '\n' ' ' || true)"
echo "model_loading=${nvidia_mem}" >> "${OUT}/replica.txt"
echo "status=ok" | tee "${OUT}/status.txt"
echo "${OUT}"
