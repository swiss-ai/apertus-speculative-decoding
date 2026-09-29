#!/bin/bash
# DSpark scale demo on Apertus 1.5 8B, thinking off: 100k regenerated Magpie samples, one GH200 node, ~12 h.
# Run 1: best configuration from the 3 h benchmark (SEQ/OPT/NP/LAYOUT below), 5 epochs, with a forced interruption after
#        RESUME_AFTER seconds and a relaunch to exercise checkpoint resume.  Run 2: the pilot layout (online, delete, seq 4096,
#        Muon, 2 trainer GPUs) for 2 epochs as the scale reference.  Then: serve checkpoint_best, acceptance in both modes
#        (spec_accept.py) and greedy-equality against no speculation.  GPU trace every 30 s, capstor cache size, timings.
set +e; export SPEC=${SPEC:-/capstor/scratch/cscs/zyu/spec}; export OUT=$SPEC/runs/apertus_dspark_scale; mkdir -p $OUT; PORT=8000
MODEL=$SPEC/hf/hub/models--swiss-ai--Apertus-v1.5-8B/snapshots/a411d838600baf0e3635a3daf66fb7c55fc97bb6
REGEN=${REGEN:-$SPEC/runs/apertus_p2/regen/magpie_nothink_100k.jsonl}; DATA=$OUT/data; HS=$OUT/hs_cache; mkdir -p $HS
LAYERS="1 8 15 22 29"
# ---- knobs chosen from the benchmark (overridable by environment)
SEQ=${SEQ:-16384}; OPT=${OPT:-muon}; NP=${NP:-3}; TGPUS=${TGPUS:-1,2,3}; LAYOUT=${LAYOUT:-online_cache}; EPOCHS=${EPOCHS:-10}; ANCHORS=${ANCHORS:-1024}; VOCAB=${VOCAB:-full}; RESUME_AFTER=${RESUME_AFTER:-2700}
TAG=${TAG:-ghcr}; EX=$SPEC/venvs/apertus-extra-$TAG; export PYTHONPATH=$SPEC/repos/speculators/src:$SPEC/repos/speculators/hs_connectors/src:$EX:${PYTHONPATH:-}
export TORCHINDUCTOR_CACHE_DIR=$SPEC/cache/inductor; mkdir -p $TORCHINDUCTOR_CACHE_DIR
command -v python >/dev/null || python() { python3 "$@"; }; export -f python 2>/dev/null
source $SPEC/jobs/side_deps.sh
command -v python >/dev/null || python() { python3 "$@"; }; export -f python 2>/dev/null
command -v speculators >/dev/null || speculators() { python -m speculators "$@"; }
source $SPEC/jobs/fork_patch.sh
cd $SPEC/repos/speculators
TL=$OUT/timeline.txt; : > $TL; : > $OUT/status.txt
stamp() { echo "=== [$(date +%T)] $(date +%s) $*" | tee -a $TL; }
wait_health() { for i in $(seq 1 180); do curl -sf localhost:$PORT/health >/dev/null && return 0; kill -0 $1 2>/dev/null || return 1; sleep 5; done; return 1; }
nvidia-smi --query-gpu=timestamp,index,utilization.gpu,memory.used,power.draw --format=csv,noheader -l 30 > $OUT/gpu.csv 2>/dev/null & SMI=$!
stamp "knobs: SEQ=$SEQ OPT=$OPT NP=$NP TGPUS=$TGPUS LAYOUT=$LAYOUT EPOCHS=$EPOCHS ANCHORS=$ANCHORS VOCAB=$VOCAB RESUME_AFTER=$RESUME_AFTER; regen rows $(wc -l < $REGEN)"
[ "$VOCAB" = full ] && DV="" || DV="--draft-vocab-size 32000"
if python -m speculators.train --help 2>&1 | grep -q -- "--hidden-states-path"; then HSFLAG="--hidden-states-path $HS"; else HSFLAG=""; HS=/tmp/hidden_states; echo "   no --hidden-states-path on trainer; cache in $HS"; fi
# ---- prepare data
if [ ! -f $DATA/dataset_info.json ]; then stamp "prepare-data"; t0=$(date +%s)
  speculators prepare-data --model $MODEL --trust-remote-code --data $REGEN --output $DATA --token-freq-path $DATA/token_freq.pt --no-skip-token-freq --seq-length 8192 > $OUT/prepare.log 2>&1; rc=$?
  stamp "   prepare rc=$rc in $(( $(date +%s)-t0 ))s"; tail -3 $OUT/prepare.log | cut -c1-200; [ $rc -eq 0 ] || { echo PREPARE_FAILED >> $OUT/status.txt; kill $SMI; exit 1; }
fi
DSPARK="--speculator-type dspark --lr 3e-4 --block-size 8 --max-anchors $ANCHORS --num-layers 5 --markov-rank 256 --enable-confidence-head --confidence-head-with-markov --loss-fn {\"ce\":0.1,\"tv\":0.9}"
COMMON="--verifier-name-or-path $MODEL --data-path $DATA --hidden-states-backend file $HSFLAG --loss-implementation eager --vllm-endpoint http://localhost:$PORT/v1 --target-layer-ids $LAYERS"
poll_epochs() { log=$1; n=0; while sleep 20; do m=$(grep -c "Training epoch\|Validation epoch\|Checkpoint saved" $log 2>/dev/null); [ "$m" != "$n" ] && { n=$m; stamp "   $(basename $log .log): $(grep -E 'Training epoch|Validation epoch|Checkpoint saved' $log | tail -1 | grep -oE 'epoch [0-9]+/[0-9]+ [a-z]+|Checkpoint saved')"; }; kill -0 $2 2>/dev/null || break; done; }
start_server() { stamp "extraction server start"; t0=$(date +%s)
  CUDA_VISIBLE_DEVICES=0 python scripts/launch_vllm.py train $MODEL --target-layer-ids $LAYERS --hidden-states-backend file --hidden-states-path $HS -- --trust-remote-code --skip-mm-profiling --renderer-num-workers 1 --mm-processor-cache-gb 0 --port $PORT --gpu-memory-utilization 0.85 --max-model-len 8192 > $OUT/extract_vllm_$1.log 2>&1 & VP=$!
  wait_health $VP || { echo EXTRACT_SERVER_FAILED >> $OUT/status.txt; grep -E "Error|Traceback" $OUT/extract_vllm_$1.log | tail -5; return 1; }
  stamp "   up in $(( $(date +%s)-t0 ))s"; }
stop_server() { kill $VP 2>/dev/null; wait $VP 2>/dev/null; sleep 5; stamp "extraction server stopped"; }
# train NAME GPUS NP EPOCHS TIMEOUT extra...  (TIMEOUT=0 -> no limit)
train() { name=$1; gpus=$2; np=$3; epochs=$4; tmo=$5; shift 5; echo $np > $OUT/train_$name.ranks; stamp "train $name: gpus=$gpus np=$np epochs=$epochs timeout=$tmo extra=[$*]"; t0=$(date +%s)
  if [ "$tmo" -gt 0 ]; then timeout -s INT --kill-after=120 $tmo env CUDA_VISIBLE_DEVICES=$gpus torchrun --standalone --nproc_per_node $np -m speculators.train $COMMON $DSPARK $DV --epochs $epochs --save-path $OUT/ckpt_$name "$@" >> $OUT/train_$name.log 2>&1 & TP=$!
  else CUDA_VISIBLE_DEVICES=$gpus torchrun --standalone --nproc_per_node $np -m speculators.train $COMMON $DSPARK $DV --epochs $epochs --save-path $OUT/ckpt_$name "$@" >> $OUT/train_$name.log 2>&1 & TP=$!; fi
  poll_epochs $OUT/train_$name.log $TP; wait $TP; rc=$?
  stamp "   $name rc=$rc wall=$(( $(date +%s)-t0 ))s; checkpoints: $(ls $OUT/ckpt_$name 2>/dev/null | tr '\n' ' ')"
  [ $rc -eq 0 ] && echo "TRAIN_${name}_OK" >> $OUT/status.txt || { echo "TRAIN_${name}_rc$rc" >> $OUT/status.txt; grep -E "Error|Traceback|OutOfMemory|out of memory" $OUT/train_$name.log | tail -4 | cut -c1-200; }
}
# ---- Run 1: best configuration, interrupted once and resumed
if [ "$LAYOUT" = online_cache ]; then GEN="--on-missing generate --on-generate cache"; else GEN="--on-missing generate --on-generate delete"; fi
rm -rf $OUT/ckpt_run1; start_server run1 || { kill $SMI; exit 1; }
train run1 $TGPUS $NP $EPOCHS $RESUME_AFTER --total-seq-len $SEQ --optimizer $OPT $GEN --checkpoint-freq 0.5
stamp "interrupted run1 after ${RESUME_AFTER}s; checkpoints now: $(ls $OUT/ckpt_run1 2>/dev/null | tr '\n' ' ')"
if [ -d $OUT/ckpt_run1/interrupted ]; then ep=$(python3 -c "import json;print(json.load(open('$OUT/ckpt_run1/interrupted/training_state.json'))['epoch'])" 2>/dev/null)
  [ -n "$ep" ] && { rm -rf $OUT/ckpt_run1/$ep; mv $OUT/ckpt_run1/interrupted $OUT/ckpt_run1/$ep; stamp "   renamed interrupted -> $ep (epoch $ep, local_step $(python3 -c "import json;print(json.load(open('$OUT/ckpt_run1/$ep/training_state.json'))['local_step'])"))"; }
fi
stamp "relaunching the same command (resume test)"
train run1 $TGPUS $NP $EPOCHS 0 --total-seq-len $SEQ --optimizer $OPT $GEN --checkpoint-freq 0.5
grep -E "Resum|resum|interrupted" $OUT/train_run1.log | head -5 | cut -c1-200
stamp "cache: $(du -sh $HS 2>/dev/null | cut -f1), files $(find $HS -type f | wc -l); disk used by run dir: $(du -sh $OUT | cut -f1)"
stop_server
# ---- Run 2: pilot layout at scale (online, delete, seq 4096, Muon, 2 trainer GPUs), 2 epochs, separate cache dir so nothing is reused
HS2=$OUT/hs_tmp_run2; mkdir -p $HS2; COMMON2=${COMMON//$HS/$HS2}
stamp "run2 extraction server start"; t0=$(date +%s)
CUDA_VISIBLE_DEVICES=0 python scripts/launch_vllm.py train $MODEL --target-layer-ids $LAYERS --hidden-states-backend file --hidden-states-path $HS2 -- --trust-remote-code --skip-mm-profiling --renderer-num-workers 1 --mm-processor-cache-gb 0 --port $PORT --gpu-memory-utilization 0.85 --max-model-len 8192 > $OUT/extract_vllm_run2.log 2>&1 & VP=$!
if wait_health $VP; then stamp "   up in $(( $(date +%s)-t0 ))s"; COMMON_SAVE=$COMMON; COMMON=$COMMON2
  rm -rf $OUT/ckpt_run2; train run2 1,2 2 2 0 --total-seq-len 4096 --optimizer muon --on-missing generate --on-generate delete
  COMMON=$COMMON_SAVE; stop_server
else echo RUN2_SERVER_FAILED >> $OUT/status.txt; fi
# ---- Run 3: speed probe from the run-1 cache, AdamW + 16k packing on all four GPUs, 1 epoch
if [ "$LAYOUT" = online_cache ] && [ -d $OUT/ckpt_run1 ]; then rm -rf $OUT/ckpt_run3; train run3_adamw_seq16k_4gpu 0,1,2,3 4 1 0 --total-seq-len 16384 --optimizer adamw --on-missing raise; fi
# ---- Serve checkpoint_best of run1: acceptance both modes + greedy equality
CK=$OUT/ckpt_run1/checkpoint_best; [ -d $CK ] || CK=$(ls -d $OUT/ckpt_run1/[0-9]* 2>/dev/null | sort -V | tail -1)
PROMPTS=("Write a Python function that checks whether a number is prime and explain it." "Summarize the causes of the French Revolution in five sentences." "Solve step by step: a train travels 120 km in 1.5 hours; what is its average speed in m/s?" "Translate to German and French: The meeting is postponed to next Tuesday at 10 am." "List three differences between TCP and UDP and when to use each.")
ask_all() { tag=$1; : > $OUT/answers_$tag.txt; for q in "${PROMPTS[@]}"; do curl -s localhost:$PORT/v1/chat/completions -H "Content-Type: application/json" -d '{"model":"apertus15-8b","max_tokens":400,"temperature":0,"chat_template_kwargs":{"enable_thinking":false},"messages":[{"role":"user","content":"'"$q"'"}]}' | python3 -c "import sys,json; d=json.load(sys.stdin); print(d['choices'][0]['message']['content']); print('-----')" >> $OUT/answers_$tag.txt 2>&1; done; }
metrics() { python3 - <<PY | tee -a $TL
import urllib.request
t=urllib.request.urlopen("http://localhost:$PORT/metrics").read().decode()
g=lambda k: sum(float(l.rsplit(" ",1)[1]) for l in t.splitlines() if l.startswith("vllm:spec_decode_"+k+"_total"))
d,a=g("num_drafts"),g("num_accepted_tokens"); print("   $1 accepted length: %.2f (drafts %d, accepted %d)" % (1+a/d if d else 0, d, a))
PY
}
stamp "serve baseline"; t0=$(date +%s)
CUDA_VISIBLE_DEVICES=0 vllm serve $MODEL --served-model-name apertus15-8b --trust-remote-code --skip-mm-profiling --max-model-len 8192 --gpu-memory-utilization 0.85 --port $PORT > $OUT/serve_base.log 2>&1 & P=$!
if wait_health $P; then stamp "   up in $(( $(date +%s)-t0 ))s"; ask_all base; echo SERVE_base_OK >> $OUT/status.txt; else echo SERVE_base_FAILED >> $OUT/status.txt; fi
kill $P 2>/dev/null; wait $P 2>/dev/null; sleep 5
if [ -n "$CK" ]; then stamp "serve dspark from $CK"; t0=$(date +%s)
  CUDA_VISIBLE_DEVICES=0 vllm serve $MODEL --served-model-name apertus15-8b --trust-remote-code --skip-mm-profiling --max-model-len 8192 --max-num-seqs 64 --max-num-batched-tokens 16384 --gpu-memory-utilization 0.85 --port $PORT \
    --speculative-config "{\"method\":\"dspark\",\"model\":\"$CK\",\"num_speculative_tokens\":7}" > $OUT/serve_dspark.log 2>&1 & P=$!
  if wait_health $P; then stamp "   up in $(( $(date +%s)-t0 ))s"; ask_all dspark; metrics "probe(nothink,greedy)"
    if cmp -s $OUT/answers_base.txt $OUT/answers_dspark.txt; then stamp "   greedy outputs IDENTICAL to no-speculation"; else stamp "   greedy outputs DIFFER from no-speculation"; fi
    if [ -f $SPEC/jobs/spec_accept_apertus.py ] && ls $SPEC/hf/hub/datasets--RedHatAI--speculator_benchmarks/snapshots/*/math_reasoning.jsonl >/dev/null 2>&1; then
      for mode in false true; do stamp "spec_accept_apertus.py thinking=$mode (32 prompts math+HumanEval, greedy, 8 concurrent)"; MODEL_NAME=apertus15-8b THINK=$mode python3 $SPEC/jobs/spec_accept_apertus.py http://localhost:$PORT dspark_think_$mode $OUT/accept_think_$mode.json 32 $([ $mode = true ] && echo 1024 || echo 384) > $OUT/spec_accept_think_$mode.log 2>&1; python3 -c "import json; d=json.load(open('$OUT/accept_think_$mode.json')); print('   ', {k: (round(v,3) if isinstance(v,float) else v) for k,v in d.items() if k in ('name','prompts','gen_tokens','tok_per_s','accepted_length_incl_bonus','acceptance_rate')})" 2>&1 | tee -a $TL; done
    else stamp "spec_accept_apertus.py or the benchmark prompts are missing; skipped"; fi
    echo SERVE_dspark_OK >> $OUT/status.txt
  else echo SERVE_dspark_FAILED >> $OUT/status.txt; grep -E "Error|Traceback|size mismatch" $OUT/serve_dspark.log | tail -6 | cut -c1-200; fi
  kill $P 2>/dev/null; wait $P 2>/dev/null
fi
kill $SMI 2>/dev/null
python3 $SPEC/jobs/bench_summary.py $OUT | tee $OUT/summary.txt
stamp "done: $(cat $OUT/status.txt | tr '\n' ' ')"
