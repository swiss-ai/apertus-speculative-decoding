#!/bin/bash
# Regenerate one Open-PerfectBlend shard with Apertus 1.5 8B, thinking ON (DeepSpec sampling, 8k answer cap), 4-way data-parallel server.
set +e; export SPEC=${SPEC:-/capstor/scratch/cscs/zyu/spec}; OUT=$SPEC/runs/apertus_opb/regen; mkdir -p $OUT; PORT=8000
MODEL=$SPEC/hf/hub/models--swiss-ai--Apertus-v1.5-8B/snapshots/a411d838600baf0e3635a3daf66fb7c55fc97bb6
SHARD=${SHARD:-0}; NSH=${NSH:-4}; IN=$SPEC/runs/apertus_opb/prompts/opb_shuffled_seed0_shard${SHARD}of${NSH}.jsonl; OUTFILE=$OUT/opb_think_shard${SHARD}.jsonl
TAG=ghcr; EX=$SPEC/venvs/apertus-extra-$TAG; export PYTHONPATH=$SPEC/repos/speculators/src:$SPEC/repos/speculators/hs_connectors/src:$EX:${PYTHONPATH:-}
command -v python >/dev/null || python() { python3 "$@"; }; export -f python 2>/dev/null
source $SPEC/jobs/side_deps.sh
command -v speculators >/dev/null || speculators() { python -m speculators "$@"; }
cd $SPEC/repos/speculators
stamp() { echo "=== [$(date +%T)] $(date +%s) $*"; }
wait_health() { for i in $(seq 1 $2); do curl -sf localhost:$PORT/health >/dev/null && return 0; kill -0 $1 2>/dev/null || return 1; sleep 5; done; return 1; }
stamp "shard $SHARD: $(wc -l < $IN) prompts; outfile $OUTFILE ($(wc -l < $OUTFILE 2>/dev/null || echo 0) rows already)"
CUDA_VISIBLE_DEVICES=0,1,2,3 vllm serve $MODEL --data-parallel-size 4 --trust-remote-code --skip-mm-profiling --chat-template-content-format string --max-model-len 16384 --gpu-memory-utilization 0.90 --max-num-seqs 256 --port $PORT > $OUT/vllm_think_shard$SHARD.log 2>&1 & P=$!
wait_health $P 180 || { stamp "SERVER FAILED"; grep -E "Error|Traceback" $OUT/vllm_think_shard$SHARD.log | tail -3; exit 1; }
stamp "server up; regenerating"; t0=$(date +%s)
HF_HUB_OFFLINE=0 HF_DATASETS_OFFLINE=0 speculators regenerate-responses --dataset $IN --concurrency 512 --max-tokens ${MAXTOK:-8192} --model $MODEL \
  --sampling-params '{"temperature":0.7,"top_p":0.8,"top_k":20,"chat_template_kwargs":{"enable_thinking":true}}' --outfile $OUTFILE --resume > $OUT/regen_think_shard$SHARD.log 2>&1; rc=$?
stamp "regen rc=$rc in $(( $(date +%s)-t0 ))s; rows $(wc -l < $OUTFILE)"; grep -oE "ok=[0-9]+, err=[0-9]+, trunc=[0-9]+, rps=[0-9.]+, tps=[0-9]+" $OUT/regen_think_shard$SHARD.log | tail -1
python3 - "$OUTFILE" <<'PY'
import json, sys, statistics as st, collections
C=[]; fr=collections.Counter()
for l in open(sys.argv[1]):
    try: d=json.loads(l)
    except Exception: continue
    u=d.get("metadata",{}).get("usage") or {}; C.append(u.get("completion_tokens",0)); fr[d.get("metadata",{}).get("finish_reason")]+=1
C.sort(); print("   rows %d finish %s | completion tokens median %d mean %d p90 %d | total %.1fM" % (len(C), dict(fr), st.median(C) if C else 0, st.mean(C) if C else 0, C[int(0.9*len(C))] if C else 0, sum(C)/1e6))
PY
kill $P 2>/dev/null; wait $P 2>/dev/null; [ $rc -eq 0 ] && echo "REGEN_THINK_SHARD${SHARD}_OK" >> $OUT/status.txt || echo "REGEN_THINK_SHARD${SHARD}_FAILED" >> $OUT/status.txt; stamp "done"
