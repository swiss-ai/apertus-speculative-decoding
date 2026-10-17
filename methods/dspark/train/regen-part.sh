#!/usr/bin/env bash
# Regenerate one prompt file with the 70B on one node, thinking off, with Yu's
# sampling (T 0.7, top-p 0.8, top-k 20, 4,096-token cap, 8,192 context).
#   REPO_ROOT=... IN=prompts.jsonl OUT=answers.jsonl [TP=2] [LIMIT=N] bash regen-part.sh
# TP=4 (default) serves one replica per node, TP=2 two data-parallel ones. --resume skips the
# conversations already in OUT, so a rerun continues; the client stops cleanly
# BUDGET_SECONDS after this script started (set it below the job's time limit).
set -euo pipefail
. "${REPO_ROOT}/methods/dspark/train/common.sh"
: "${IN:?}" "${OUT:?}"
TP="${TP:-4}"
DP=$(( 4 / TP ))
PORT="${PORT:-8000}"
CONCURRENCY="${CONCURRENCY:-512}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-256}"
BUDGET_SECONDS="${BUDGET_SECONDS:-42600}"
started=$(date +%s)
LOG_DIR="$(dirname "${OUT}")/logs"
NAME="$(basename "${OUT}" .jsonl)"
mkdir -p "${LOG_DIR}"
check_side_deps

done_rows=0
[ -f "${OUT}" ] && done_rows=$(wc -l < "${OUT}")
stamp "${NAME}: $(wc -l < "${IN}") conversations in ${IN}; ${done_rows} rows already; TP=${TP} DP=${DP} job=${SLURM_JOB_ID:-}"
setsid vllm serve "${TARGET_MODEL}" --tensor-parallel-size "${TP}" --data-parallel-size "${DP}" \
  --trust-remote-code --skip-mm-profiling --chat-template-content-format string \
  --max-model-len 8192 --gpu-memory-utilization 0.90 --max-num-seqs "${MAX_NUM_SEQS}" \
  --port "${PORT}" > "${LOG_DIR}/vllm-${NAME}-${SLURM_JOB_ID:-local}.log" 2>&1 &
server=$!
trap 'stop_server ${server}' EXIT
if ! wait_health "${PORT}" "${server}" 2400; then
  stamp "SERVER_FAILED"
  grep -E "Error|Traceback" "${LOG_DIR}/vllm-${NAME}-${SLURM_JOB_ID:-local}.log" | tail -5
  exit 1
fi
stamp "server up; regenerating"
t0=$(date +%s)
rc=0
timeout --signal=INT $(( started + BUDGET_SECONDS - t0 - 120 )) python3 -m speculators regenerate-responses \
  --dataset "${IN}" ${LIMIT:+--limit "${LIMIT}"} --concurrency "${CONCURRENCY}" --max-tokens 4096 \
  --model "${TARGET_MODEL}" \
  --sampling-params '{"temperature":0.7,"top_p":0.8,"top_k":20,"chat_template_kwargs":{"enable_thinking":false}}' \
  --outfile "${OUT}" --resume > "${LOG_DIR}/regen-${NAME}-${SLURM_JOB_ID:-local}.log" 2>&1 || rc=$?
elapsed=$(( $(date +%s) - t0 ))
stamp "regenerate-responses rc=${rc} in ${elapsed}s; rows $(wc -l < "${OUT}")"
grep -oE "ok=[0-9]+, err=[0-9]+, trunc=[0-9]+, rps=[0-9.]+, tps=[0-9]+" \
  "${LOG_DIR}/regen-${NAME}-${SLURM_JOB_ID:-local}.log" | tail -1 || true
python3 - "${OUT}" "${done_rows}" "${elapsed}" <<'PY'
import collections, json, statistics as st, sys
path, before, elapsed = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
comp, finish = [], collections.Counter()
for i, line in enumerate(open(path)):
    try:
        d = json.loads(line)
    except ValueError:
        continue
    if i < before:
        continue
    m = d.get("metadata", {})
    comp.append((m.get("usage") or {}).get("completion_tokens", 0))
    finish[m.get("finish_reason")] += 1
if comp:
    comp.sort()
    print("this job: rows %d finish %s | completion median %d mean %.0f p90 %d | %.1fM tokens, %.0f tok/s"
          % (len(comp), dict(finish), st.median(comp), st.mean(comp), comp[int(0.9 * len(comp))],
             sum(comp) / 1e6, sum(comp) / max(elapsed, 1)))
PY
# 124: the client was stopped at the budget; the next job resumes.
if [ "${rc}" = 0 ]; then stamp "REGEN_DONE ${NAME}"; elif [ "${rc}" = 124 ]; then stamp "REGEN_TIMEOUT ${NAME}"; else stamp "REGEN_FAILED ${NAME}"; fi
exit 0
