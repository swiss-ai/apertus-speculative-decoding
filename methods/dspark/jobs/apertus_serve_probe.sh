#!/bin/bash
# Serve one drafter checkpoint for Apertus 1.5 8B and probe acceptance. usage: NAME CKPT METHOD K [extra speculative-config json fields]
set +e; export SPEC=${SPEC:-/capstor/scratch/cscs/zyu/spec}; TAG=${TAG:-ghcr}; OUT=$SPEC/runs/apertus_serve_probe; mkdir -p $OUT; PORT=8030
NAME=$1; CK=$2; M=$3; K=$4; EXTRA=${5:-}
MODEL=$SPEC/hf/hub/models--swiss-ai--Apertus-v1.5-8B/snapshots/a411d838600baf0e3635a3daf66fb7c55fc97bb6
EX=$SPEC/venvs/apertus-extra-$TAG; export PYTHONPATH=$SPEC/repos/speculators/src:$SPEC/repos/speculators/hs_connectors/src:$EX:${PYTHONPATH:-}
source $SPEC/jobs/fork_patch.sh
wait_health() { for i in $(seq 1 150); do curl -sf localhost:$PORT/health >/dev/null && return 0; kill -0 $1 2>/dev/null || return 1; sleep 5; done; return 1; }
echo "=== [$(date +%T)] serve $NAME from $CK (method $M, k $K $EXTRA)"
CUDA_VISIBLE_DEVICES=0 vllm serve $MODEL --served-model-name apertus15-8b --trust-remote-code --skip-mm-profiling --max-model-len 8192 --max-num-seqs 64 --max-num-batched-tokens 16384 --gpu-memory-utilization 0.85 --port $PORT \
  --speculative-config "{\"method\":\"$M\",\"model\":\"$CK\",\"num_speculative_tokens\":$K$EXTRA}" > $OUT/${NAME}_vllm.log 2>&1 & P=$!
if wait_health $P; then
  for q in "Write a Python function that checks whether a number is prime and explain it." "Summarize the causes of the French Revolution in five sentences." "Solve step by step: a train travels 120 km in 1.5 hours; what is its average speed in m/s?" "Translate to German: The weather in Zurich is pleasant in September." "List five differences between TCP and UDP."; do
    curl -s localhost:$PORT/v1/chat/completions -H "Content-Type: application/json" -d '{"model":"apertus15-8b","max_tokens":400,"temperature":0,"chat_template_kwargs":{"enable_thinking":false},"messages":[{"role":"user","content":"'"$q"'"]}]}' > /dev/null; done
  python3 - <<PY
import urllib.request
t=urllib.request.urlopen("http://localhost:$PORT/metrics").read().decode()
g=lambda k: sum(float(l.rsplit(" ",1)[1]) for l in t.splitlines() if l.startswith("vllm:spec_decode_"+k+"_total"))
d,a,dt=g("num_drafts"),g("num_accepted_tokens"),g("num_draft_tokens"); print("   $NAME accepted length: %.2f | draft-token acceptance %.2f | drafts %d" % (1+a/d if d else 0, a/dt if dt else 0, d))
PY
  echo "SERVE_${NAME}_OK" >> $OUT/status.txt
else echo "SERVE_${NAME}_FAILED" >> $OUT/status.txt; grep -E "Error|assert" $OUT/${NAME}_vllm.log | grep -vE "INFO|^\s*File" | tail -4 | cut -c1-240; fi
kill $P 2>/dev/null; wait $P 2>/dev/null; echo "=== done $(date +%T)"
