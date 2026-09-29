#!/bin/bash
# Phase 0 gate for Apertus 1.5 8B: (A) plain serving in both modes, (B) hidden-state extraction through speculators' launcher.
set +e; export SPEC=${SPEC:-/capstor/scratch/cscs/zyu/spec}; TAG=${TAG:-sqsh}; OUT=$SPEC/runs/apertus_gate_$TAG; mkdir -p $OUT; rm -f $OUT/status.txt; PORT=8000
MODEL=$SPEC/hf/hub/models--swiss-ai--Apertus-v1.5-8B/snapshots/a411d838600baf0e3635a3daf66fb7c55fc97bb6
echo "== python/vllm in image"; which python; python -c "import vllm, torch, transformers; print('vllm', vllm.__version__, 'torch', torch.__version__, 'transformers', transformers.__version__)"
# no nested venv: the image python is itself a venv; put speculators on PYTHONPATH and light deps in a --target dir
EX=$SPEC/venvs/apertus-extra-$TAG; mkdir -p $EX; export PYTHONPATH=$SPEC/repos/speculators/src:$SPEC/repos/speculators/hs_connectors/src:$EX:${PYTHONPATH:-}
command -v python >/dev/null || python() { python3 "$@"; }; export -f python 2>/dev/null
source $SPEC/jobs/side_deps.sh
python -c "import speculators, hs_connectors, vllm; print('speculators importable on vllm', vllm.__version__)" || echo "SPECULATORS_IMPORT_FAILED" >> $OUT/status.txt
echo "== image facts"; ls /workspace/vllm/vllm/model_executor/models/ 2>/dev/null | grep -i apertus | tr "\n" " "; echo
python - <<PY
import vllm.config.speculative as s, inspect; src=inspect.getsource(s)
print("   extract_hidden_states in SpeculativeConfig:", "extract_hidden_states" in src, "| dflash:", '"dflash"' in src, "| dspark:", '"dspark"' in src, "| parallel_drafting:", "parallel_drafting" in src, "| adaptive_verification:", "adaptive_verification" in src)
PY
wait_health() { for i in $(seq 1 150); do curl -sf localhost:$PORT/health >/dev/null && return 0; kill -0 $1 2>/dev/null || return 1; sleep 5; done; return 1; }
ask() { curl -s localhost:$PORT/v1/chat/completions -H "Content-Type: application/json" -d '{"model":"apertus15-8b","max_tokens":'$2',"temperature":0,"chat_template_kwargs":{"enable_thinking":'$1'},"messages":[{"role":"user","content":"What is 17*23? Then write one sentence about Zurich."}]}' | python3 -c "import sys,json; d=json.load(sys.stdin); c=d['choices'][0]['message']; print('   usage:',d['usage']); print('   reasoning:',((c.get('reasoning_content') or c.get('reasoning') or '')[:200]).replace(chr(10),' ')); print('   content:',(c.get('content') or '')[:300].replace(chr(10),' '))" 2>&1 | head -5; }
echo; echo "== A: plain serve ($(date +%T))"; t0=$(date +%s)
vllm serve $MODEL --served-model-name apertus15-8b --trust-remote-code --skip-mm-profiling --max-model-len 8192 --gpu-memory-utilization 0.85 --port $PORT > $OUT/A_vllm.log 2>&1 & P=$!
if wait_health $P; then echo "   up in $(( $(date +%s)-t0 ))s"; echo "-- thinking on"; ask true 300; echo "-- thinking off"; ask false 120; echo "A_PASS" >> $OUT/status.txt; else echo "A_FAIL" >> $OUT/status.txt; grep -E "Error|error|Traceback" $OUT/A_vllm.log | tail -5; fi
kill $P 2>/dev/null; wait $P 2>/dev/null; sleep 5
echo; echo "== fork patch"; source $SPEC/jobs/fork_patch.sh
echo; echo "== B: extraction serve via speculators launcher ($(date +%T))"; cd $SPEC/repos/speculators; rm -rf /tmp/hidden_states; t0=$(date +%s)
python scripts/launch_vllm.py train --help 2>&1 | grep -E "hidden-states|target-layer|usage" | head -6
python scripts/launch_vllm.py train $MODEL --target-layer-ids 2 16 29 --hidden-states-backend file -- --served-model-name apertus15-8b --trust-remote-code --skip-mm-profiling --renderer-num-workers 1 --mm-processor-cache-gb 0 --max-model-len 8192 --gpu-memory-utilization 0.85 --port $PORT > $OUT/B_vllm.log 2>&1 & P=$!
if wait_health $P; then echo "   up in $(( $(date +%s)-t0 ))s"; ask true 64; sleep 8; echo "-- hidden-state files:"; find /tmp/hidden_states -type f | head -5; python - <<PY
import glob, os
fs=sorted(glob.glob("/tmp/hidden_states/**/*", recursive=True)); fs=[f for f in fs if os.path.isfile(f)]
print("   files:", len(fs))
try:
    from safetensors import safe_open
    for f in fs[:2]:
        with safe_open(f, "pt") as s:
            for k in list(s.keys())[:8]: print("   ", os.path.basename(f), k, tuple(s.get_slice(k).get_shape()))
except Exception as e: print("   inspect error:", e)
PY
  [ -n "$(find /tmp/hidden_states -type f | head -1)" ] && echo "B_PASS" >> $OUT/status.txt || echo "B_NOFILES" >> $OUT/status.txt
else echo "B_FAIL" >> $OUT/status.txt; grep -E "Error|error|Traceback|not supported|Unsupported" $OUT/B_vllm.log | grep -v INFO | tail -8; fi
kill $P 2>/dev/null; wait $P 2>/dev/null
echo; echo "== status: $(cat $OUT/status.txt | tr '\n' ' ') ($(date +%T))"
