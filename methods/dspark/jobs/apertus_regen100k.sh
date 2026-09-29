#!/bin/bash
# Regenerate 100k Magpie prompts with Apertus 1.5 8B, thinking off (DeepSpec sampling), for the DSpark scale demo.
# Tries a 4-way data-parallel server first (untested in this fork since the --ntasks fix); falls back to one GPU.
set +e; export SPEC=${SPEC:-/capstor/scratch/cscs/zyu/spec}; OUT=$SPEC/runs/apertus_p2/regen; mkdir -p $OUT; PORT=8000
MODEL=$SPEC/hf/hub/models--swiss-ai--Apertus-v1.5-8B/snapshots/a411d838600baf0e3635a3daf66fb7c55fc97bb6
N=${N:-100000}; OUTFILE=$OUT/magpie_nothink_${N}.jsonl
TAG=${TAG:-ghcr}; EX=$SPEC/venvs/apertus-extra-$TAG; export PYTHONPATH=$SPEC/repos/speculators/src:$SPEC/repos/speculators/hs_connectors/src:$EX:${PYTHONPATH:-}
command -v python >/dev/null || python() { python3 "$@"; }; export -f python 2>/dev/null
source $SPEC/jobs/side_deps.sh
command -v speculators >/dev/null || speculators() { python -m speculators "$@"; }
cd $SPEC/repos/speculators
stamp() { echo "=== [$(date +%T)] $(date +%s) $*"; }
wait_health() { for i in $(seq 1 $2); do curl -sf localhost:$PORT/health >/dev/null && return 0; kill -0 $1 2>/dev/null || return 1; sleep 5; done; return 1; }
nvidia-smi --query-gpu=timestamp,index,utilization.gpu,memory.used --format=csv,noheader -l 30 > $OUT/gpu_regen.csv 2>/dev/null & SMI=$!
[ -s $OUTFILE ] || { [ -s $SPEC/runs/apertus_pipe5k/regen_magpie.jsonl ] && cp $SPEC/runs/apertus_pipe5k/regen_magpie.jsonl $OUTFILE && echo "   seeded outfile with the 5k pilot rows (--resume)"; }
SERVE="vllm serve $MODEL --trust-remote-code --skip-mm-profiling --chat-template-content-format string --max-model-len 8192 --gpu-memory-utilization 0.90 --max-num-seqs 256 --port $PORT"
stamp "try data-parallel 4 server"; t0=$(date +%s)
CUDA_VISIBLE_DEVICES=0,1,2,3 $SERVE --data-parallel-size 4 > $OUT/regen_vllm_dp4.log 2>&1 & P=$!
if wait_health $P 150; then MODE=dp4; CONC=512; stamp "   dp4 up in $(( $(date +%s)-t0 ))s"; else
  kill $P 2>/dev/null; wait $P 2>/dev/null; sleep 5; stamp "   dp4 failed ($(grep -E 'Error|negative|Traceback' $OUT/regen_vllm_dp4.log | tail -2 | cut -c1-160)); falling back to one GPU"; t0=$(date +%s)
  CUDA_VISIBLE_DEVICES=0 $SERVE > $OUT/regen_vllm_1gpu.log 2>&1 & P=$!
  wait_health $P 150 && { MODE=1gpu; CONC=128; stamp "   1-GPU server up in $(( $(date +%s)-t0 ))s"; } || { echo REGEN_SERVER_FAILED >> $OUT/status.txt; kill $SMI; exit 1; }
fi
stamp "regenerate $N magpie (thinking off) mode=$MODE concurrency=$CONC"; t0=$(date +%s)
HF_HUB_OFFLINE=0 HF_DATASETS_OFFLINE=0 speculators regenerate-responses --dataset magpie --limit $N --concurrency $CONC --max-tokens 4096 --model $MODEL \
  --sampling-params '{"temperature":0.7,"top_p":0.8,"top_k":20,"chat_template_kwargs":{"enable_thinking":false}}' --outfile $OUTFILE --resume > $OUT/regen_${N}.log 2>&1; rc=$?
el=$(( $(date +%s)-t0 )); rows=$(wc -l < $OUTFILE)
stamp "regen rc=$rc rows=$rows in ${el}s ($MODE)"; tail -4 $OUT/regen_${N}.log | cut -c1-200
python3 - "$OUTFILE" <<'PY'
import json, sys, statistics as st
n=0; gen=[]; trunc=0
for l in open(sys.argv[1]):
    try: d=json.loads(l)
    except Exception: continue
    n+=1
    for k in ("completion_tokens","num_output_tokens","output_tokens"):
        if k in d: gen.append(d[k]); break
    else:
        c=d.get("conversations") or d.get("messages") or []
        t=sum(len(m.get("value") or m.get("content") or "")//4 for m in c if (m.get("from") or m.get("role")) in ("gpt","assistant")); gen.append(t)
print("   rows %d, generated tokens/sample median %.0f mean %.0f, total ~%.1fM" % (n, st.median(gen) if gen else 0, st.mean(gen) if gen else 0, sum(gen)/1e6))
PY
kill $P 2>/dev/null; wait $P 2>/dev/null; kill $SMI 2>/dev/null
[ $rc -eq 0 ] && echo "REGEN_OK $MODE $rows ${el}s" >> $OUT/status.txt || echo "REGEN_FAILED $MODE" >> $OUT/status.txt
stamp "done: $(cat $OUT/status.txt | tr '\n' ' ')"
