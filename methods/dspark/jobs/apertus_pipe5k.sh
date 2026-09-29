#!/bin/bash
# Apertus 1.5 8B drafter pipeline check (5k samples): regenerate -> prepare-data -> online train Eagle3, DFlash, DSpark -> serve + acceptance probe.
set +e; export SPEC=${SPEC:-/capstor/scratch/cscs/zyu/spec}; OUT=$SPEC/runs/apertus_pipe5k; mkdir -p $OUT; PORT=8000
MODEL=$SPEC/hf/hub/models--swiss-ai--Apertus-v1.5-8B/snapshots/a411d838600baf0e3635a3daf66fb7c55fc97bb6
LAYERS="2 16 29"; N=${N:-5000}; SEQ=8192
TAG=${TAG:-ghcr}; EX=$SPEC/venvs/apertus-extra-$TAG; mkdir -p $EX; export PYTHONPATH=$SPEC/repos/speculators/src:$SPEC/repos/speculators/hs_connectors/src:$EX:${PYTHONPATH:-}
command -v python >/dev/null || python() { python3 "$@"; }; export -f python 2>/dev/null
source $SPEC/jobs/side_deps.sh
command -v python >/dev/null || python() { python3 "$@"; }; export -f python 2>/dev/null
for pk in pandas pyarrow fsspec "datasets>=4.0.0,<=5.0.1" "loguru>=0.7.2,<=0.7.3" pydantic-settings dill multiprocess xxhash; do m=${pk%%[><=]*}; m=${m//-/_}; python -c "import $m" 2>/dev/null || pip install -q --no-deps --target $EX "$pk" 2>&1 | tail -1; done
python -c "import speculators, hs_connectors, vllm, transformers; print('stack ok: vllm', vllm.__version__, 'transformers', transformers.__version__)" || exit 1
command -v speculators >/dev/null || speculators() { python -m speculators "$@"; }
source $SPEC/jobs/fork_patch.sh
cd $SPEC/repos/speculators
wait_health() { for i in $(seq 1 180); do curl -sf localhost:$PORT/health >/dev/null && return 0; kill -0 $1 2>/dev/null || return 1; sleep 5; done; return 1; }
stamp() { echo "=== [$(date +%T)] $*"; }
# ---- Step 0: regenerate responses with the target (thinking off)
if [ ! -s $OUT/regen_${DATASET:-magpie}.jsonl ]; then
  stamp "regenerate $N ${DATASET:-magpie} responses"; t0=$(date +%s)
  CUDA_VISIBLE_DEVICES=0 vllm serve $MODEL --trust-remote-code --skip-mm-profiling --chat-template-content-format string --max-model-len 8192 --gpu-memory-utilization 0.90 --max-num-seqs 256 --port $PORT > $OUT/regen_vllm.log 2>&1 & P=$!
  wait_health $P || { echo REGEN_SERVER_FAILED >> $OUT/status.txt; tail -5 $OUT/regen_vllm.log; exit 1; }
  HF_HUB_OFFLINE=0 HF_DATASETS_OFFLINE=0 speculators regenerate-responses --dataset ${DATASET:-magpie} --limit $N --concurrency 128 --max-tokens 4096 --model $MODEL \
     --sampling-params '{"temperature":0.7,"top_p":0.8,"top_k":20,"chat_template_kwargs":{"enable_thinking":false}}' --outfile $OUT/regen_${DATASET:-magpie}.jsonl --resume > $OUT/regen.log 2>&1; rc=$?
  kill $P 2>/dev/null; wait $P 2>/dev/null; sleep 5
  echo "   regen rc=$rc rows=$(wc -l < $OUT/regen_${DATASET:-magpie}.jsonl) in $(( $(date +%s)-t0 ))s"; tail -3 $OUT/regen.log
  [ $rc -eq 0 ] || { echo REGEN_FAILED >> $OUT/status.txt; exit 1; }
fi
# ---- Step 1: prepare data (speculator format, no render endpoint needed)
stamp "prepare-data"; speculators prepare-data --model $MODEL --trust-remote-code --data $OUT/regen_${DATASET:-magpie}.jsonl --output $OUT/data --token-freq-path $OUT/data/token_freq.pt --no-skip-token-freq --max-samples $N --seq-length $SEQ > $OUT/prepare.log 2>&1; rc=$?
ls -la $OUT/data | head -8; [ -f $OUT/data/token_freq.pt ] && DV="--draft-vocab-size 32000" || { echo "   no token_freq.pt -> full vocabulary"; DV=""; }
tail -4 $OUT/prepare.log; [ $rc -eq 0 ] || { echo PREPARE_FAILED >> $OUT/status.txt; exit 1; }
# ---- Step 2: extraction server on GPUs 0-1
stamp "extraction server"; CUDA_VISIBLE_DEVICES=0 python scripts/launch_vllm.py train $MODEL --target-layer-ids $LAYERS --hidden-states-backend file -- --trust-remote-code --skip-mm-profiling --renderer-num-workers 1 --mm-processor-cache-gb 0 --port $PORT --gpu-memory-utilization 0.85 --max-model-len $SEQ > $OUT/extract_vllm.log 2>&1 & VP=$!
wait_health $VP || { echo EXTRACT_SERVER_FAILED >> $OUT/status.txt; grep -E "Error|Traceback" $OUT/extract_vllm.log | tail -5; exit 1; }
# ---- Step 3: train on GPUs 2-3
train() { name=$1; shift; if [ -d $OUT/ckpt_$name/checkpoint_best ]; then echo "=== skip train $name (checkpoint_best exists)"; echo "TRAIN_${name}_OK" >> $OUT/status.txt; return; fi; stamp "train $name"; t0=$(date +%s)
  CUDA_VISIBLE_DEVICES=1,2 torchrun --standalone --nproc_per_node 2 -m speculators.train --verifier-name-or-path $MODEL --data-path $OUT/data --loss-implementation eager $DV \
    --vllm-endpoint http://localhost:$PORT/v1 --save-path $OUT/ckpt_$name --total-seq-len ${TRAIN_SEQ:-4096} --target-layer-ids $LAYERS --on-missing generate --on-generate delete "$@" > $OUT/train_$name.log 2>&1; rc=$?
  echo "   $name rc=$rc in $(( $(date +%s)-t0 ))s"; grep -E "loss|epoch|acc" $OUT/train_$name.log | tail -3 | cut -c1-160; [ $rc -eq 0 ] && echo "TRAIN_${name}_OK" >> $OUT/status.txt || { echo "TRAIN_${name}_FAILED" >> $OUT/status.txt; grep -E "Error|Traceback" $OUT/train_$name.log | tail -4; }
}
train eagle3 --speculator-type eagle3 --epochs 5 --lr 1e-4
train dflash --speculator-type dflash --epochs 5 --lr 3e-4 --block-size 16 --max-anchors 1024 --num-layers 5 --per-position-loss-weight dpace --loss-fn ce
LAST=$OUT/ckpt_dflash/checkpoint_best; [ -d $LAST ] || LAST=$(ls -d $OUT/ckpt_dflash/[0-9]* 2>/dev/null | sort -V | tail -1)
# warm start from a DFlash checkpoint is rejected by this speculators version (speculators_model_type must be dspark); train from scratch for the check
train dspark --speculator-type dspark --epochs 3 --lr 3e-4 --block-size 8 --max-anchors 1024 --num-layers 5 --markov-rank 256 --enable-confidence-head --confidence-head-with-markov --loss-fn '{"ce":0.1,"tv":0.9}'
kill $VP 2>/dev/null; wait $VP 2>/dev/null; sleep 5
# ---- Step 4: serve each checkpoint and probe acceptance on a few prompts
for name in eagle3 dflash dspark; do
  CK=$OUT/ckpt_$name/checkpoint_best; [ -d $CK ] || CK=$(ls -d $OUT/ckpt_$name/[0-9]* 2>/dev/null | sort -V | tail -1); [ -n "$CK" ] || continue
  case $name in eagle3) M=eagle3; K=5;; dflash) M=dflash; K=15;; dspark) M=dspark; K=7;; esac
  stamp "serve $name from $CK"; CUDA_VISIBLE_DEVICES=0 vllm serve $MODEL --served-model-name apertus15-8b --trust-remote-code --skip-mm-profiling --max-model-len 8192 --max-num-seqs 64 --max-num-batched-tokens 16384 --gpu-memory-utilization 0.85 --port $PORT \
    --speculative-config "{\"method\":\"$M\",\"model\":\"$CK\",\"num_speculative_tokens\":$K}" > $OUT/serve_$name.log 2>&1 & P=$!
  if wait_health $P; then
    for q in "Write a Python function that checks whether a number is prime and explain it." "Summarize the causes of the French Revolution in five sentences." "Solve step by step: a train travels 120 km in 1.5 hours; what is its average speed in m/s?"; do
      curl -s localhost:$PORT/v1/chat/completions -H "Content-Type: application/json" -d '{"model":"apertus15-8b","max_tokens":400,"temperature":0,"chat_template_kwargs":{"enable_thinking":false},"messages":[{"role":"user","content":"'"$q"'"}]}' > /dev/null; done
    curl -s localhost:$PORT/metrics | grep -E "^vllm:spec_decode_(num_drafts|num_accepted_tokens)_total" | awk '{split($1,a,"{"); print "   "a[1], $NF}'
    python3 - <<PY
import urllib.request
t=urllib.request.urlopen("http://localhost:$PORT/metrics").read().decode()
g=lambda k: sum(float(l.rsplit(" ",1)[1]) for l in t.splitlines() if l.startswith("vllm:spec_decode_"+k+"_total"))
d,a=g("num_drafts"),g("num_accepted_tokens"); print("   $name accepted length: %.2f (drafts %d)" % (1+a/d if d else 0, d))
PY
    echo "SERVE_${name}_OK" >> $OUT/status.txt
  else echo "SERVE_${name}_FAILED" >> $OUT/status.txt; grep -E "Error|Traceback" $OUT/serve_$name.log | tail -4; fi
  kill $P 2>/dev/null; wait $P 2>/dev/null; sleep 5
done
stamp "done: $(cat $OUT/status.txt | tr '\n' ' ')"
