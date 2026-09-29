#!/bin/bash
# DSpark training-efficiency benchmark on Apertus 1.5 8B, thinking off, one GH200 node (4 GPUs, one Slurm task).
# Data: the 5k Magpie pilot set (non-thinking, prepared at seq 8192, 3.5M tokens/epoch). Five target layers (1 8 15 22 29),
# DSpark block 8, 5 draft layers, full 131k head unless stated. One epoch per variant; throughput is read from the trainer's
# per-step profile lines (steps >= 1, so torch.compile at step 0 is excluded).
#   B4  online, hidden states deleted after use (pilot layout: 1 extraction GPU + 2 trainer GPUs)
#   B1  online, hidden states cached to capstor, 2 epochs (epoch 2 reads the cache)
#   B2a cached, 2 trainer GPUs           B2b cached, 4 trainer GPUs (data-parallel scaling)
#   B3  cached, 32k draft vocabulary     B7 cached, --total-seq-len 8192     B8 cached, --total-seq-len 16384
#   B9  cached, --optimizer adamw (default is muon)
#   B5  serve the B1 checkpoint (5-layer DSpark loader check), acceptance on 5 prompts, greedy equality vs no speculation
#   B6  optional: CUDA xIELU kernel install, then one online epoch to measure the extraction-side gain
set +e; export SPEC=${SPEC:-/capstor/scratch/cscs/zyu/spec}; export OUT=$SPEC/runs/apertus_dspark_bench; mkdir -p $OUT; PORT=8000
MODEL=$SPEC/hf/hub/models--swiss-ai--Apertus-v1.5-8B/snapshots/a411d838600baf0e3635a3daf66fb7c55fc97bb6
LAYERS="1 8 15 22 29"; DATA=$SPEC/runs/apertus_pipe5k/data; HS=$OUT/hs_cache; mkdir -p $HS
TAG=${TAG:-ghcr}; EX=$SPEC/venvs/apertus-extra-$TAG; mkdir -p $EX; export PYTHONPATH=$SPEC/repos/speculators/src:$SPEC/repos/speculators/hs_connectors/src:$EX:${PYTHONPATH:-}
export TORCHINDUCTOR_CACHE_DIR=$SPEC/cache/inductor; mkdir -p $TORCHINDUCTOR_CACHE_DIR
command -v python >/dev/null || python() { python3 "$@"; }; export -f python 2>/dev/null
source $SPEC/jobs/side_deps.sh
command -v python >/dev/null || python() { python3 "$@"; }; export -f python 2>/dev/null
source $SPEC/jobs/fork_patch.sh
cd $SPEC/repos/speculators
TL=$OUT/timeline.txt; : > $TL; : > $OUT/status.txt
stamp() { echo "=== [$(date +%T)] $(date +%s) $*" | tee -a $TL; }
wait_health() { for i in $(seq 1 180); do curl -sf localhost:$PORT/health >/dev/null && return 0; kill -0 $1 2>/dev/null || return 1; sleep 5; done; return 1; }
nvidia-smi --query-gpu=timestamp,index,utilization.gpu,memory.used,power.draw --format=csv,noheader -l 10 > $OUT/gpu.csv 2>/dev/null & SMI=$!
python -c "import speculators, hs_connectors, vllm, transformers, torch; print('stack ok: vllm', vllm.__version__, 'transformers', transformers.__version__, 'torch', torch.__version__)" || exit 1
if python -m speculators.train --help 2>&1 | grep -q -- "--hidden-states-path"; then HSFLAG="--hidden-states-path $HS"; else HSFLAG=""; HS=/tmp/hidden_states; echo "   trainer has no --hidden-states-path; using default $HS (node RAM)"; fi
echo "   hidden-state cache: $HS"
DSPARK="--speculator-type dspark --lr 3e-4 --block-size 8 --max-anchors 1024 --num-layers 5 --markov-rank 256 --enable-confidence-head --confidence-head-with-markov --loss-fn {\"ce\":0.1,\"tv\":0.9}"
COMMON="--verifier-name-or-path $MODEL --data-path $DATA --hidden-states-backend file $HSFLAG --loss-implementation eager --vllm-endpoint http://localhost:$PORT/v1 --target-layer-ids $LAYERS"
poll_epochs() { log=$1; n=0; while sleep 5; do m=$(grep -c "Training epoch" $log 2>/dev/null); [ "$m" != "$n" ] && { n=$m; stamp "   $(basename $log .log): $(grep 'Training epoch' $log | tail -1 | grep -oE 'epoch [0-9]+/[0-9]+ [a-z]+')"; }; kill -0 $2 2>/dev/null || break; done; }
train() { name=$1; gpus=$2; np=$3; epochs=$4; shift 4; rm -rf $OUT/ckpt_$name; echo $np > $OUT/train_$name.ranks; stamp "train $name: gpus=$gpus np=$np epochs=$epochs extra=[$*]"; t0=$(date +%s)
  CUDA_VISIBLE_DEVICES=$gpus torchrun --standalone --nproc_per_node $np -m speculators.train $COMMON $DSPARK --epochs $epochs --save-path $OUT/ckpt_$name "$@" > $OUT/train_$name.log 2>&1 & TP=$!
  poll_epochs $OUT/train_$name.log $TP; wait $TP; rc=$?
  stamp "   $name rc=$rc wall=$(( $(date +%s)-t0 ))s"
  [ $rc -eq 0 ] && echo "TRAIN_${name}_OK" >> $OUT/status.txt || { echo "TRAIN_${name}_FAILED" >> $OUT/status.txt; grep -E "Error|Traceback|OutOfMemory|out of memory" $OUT/train_$name.log | tail -4 | cut -c1-200; }
}
# ---- extraction server on GPU 0 (5 target layers, file backend into the cache dir)
stamp "extraction server start"; t0=$(date +%s)
CUDA_VISIBLE_DEVICES=0 python scripts/launch_vllm.py train $MODEL --target-layer-ids $LAYERS --hidden-states-backend file --hidden-states-path $HS -- --trust-remote-code --skip-mm-profiling --renderer-num-workers 1 --mm-processor-cache-gb 0 --port $PORT --gpu-memory-utilization 0.85 --max-model-len 8192 > $OUT/extract_vllm.log 2>&1 & VP=$!
wait_health $VP || { echo EXTRACT_SERVER_FAILED >> $OUT/status.txt; grep -E "Error|Traceback" $OUT/extract_vllm.log | tail -5; kill $SMI; exit 1; }
stamp "extraction server up in $(( $(date +%s)-t0 ))s; xielu fallback warnings: $(grep -c 'CUDA-fused xIELU not available' $OUT/extract_vllm.log)"
train b4_online_delete 1,2 2 1 --total-seq-len 4096 --on-missing generate --on-generate delete
train b1_online_cache  1,2 2 2 --total-seq-len 4096 --on-missing generate --on-generate cache
stamp "cache size: $(du -sh $HS 2>/dev/null | cut -f1), files: $(find $HS -type f | wc -l)"
kill $VP 2>/dev/null; wait $VP 2>/dev/null; sleep 5; stamp "extraction server stopped"
train b2a_cached_2gpu    1,2     2 1 --total-seq-len 4096  --on-missing raise
train b2b_cached_4gpu    0,1,2,3 4 1 --total-seq-len 4096  --on-missing raise
train b3_cached_vocab32k 1,2     2 1 --total-seq-len 4096  --on-missing raise --draft-vocab-size 32000
train b7_cached_seq8192  1,2     2 1 --total-seq-len 8192  --on-missing raise
train b8_cached_seq16384 1,2     2 1 --total-seq-len 16384 --on-missing raise
train b9_cached_adamw    1,2     2 1 --total-seq-len 4096  --on-missing raise --optimizer adamw
# ---- B5: serve the B1 checkpoint and compare greedy outputs with no speculation
CK=$OUT/ckpt_b1_online_cache/checkpoint_best; [ -d $CK ] || CK=$(ls -d $OUT/ckpt_b1_online_cache/[0-9]* 2>/dev/null | sort -V | tail -1)
PROMPTS=("Write a Python function that checks whether a number is prime and explain it." "Summarize the causes of the French Revolution in five sentences." "Solve step by step: a train travels 120 km in 1.5 hours; what is its average speed in m/s?" "Translate to German and French: The meeting is postponed to next Tuesday at 10 am." "List three differences between TCP and UDP and when to use each.")
ask_all() { tag=$1; : > $OUT/answers_$tag.txt; for q in "${PROMPTS[@]}"; do curl -s localhost:$PORT/v1/chat/completions -H "Content-Type: application/json" -d '{"model":"apertus15-8b","max_tokens":400,"temperature":0,"chat_template_kwargs":{"enable_thinking":false},"messages":[{"role":"user","content":"'"$q"'"}]}' | python3 -c "import sys,json; d=json.load(sys.stdin); print(d['choices'][0]['message']['content']); print('-----')" >> $OUT/answers_$tag.txt 2>&1; done; }
stamp "serve baseline (no speculation)"; t0=$(date +%s)
CUDA_VISIBLE_DEVICES=0 vllm serve $MODEL --served-model-name apertus15-8b --trust-remote-code --skip-mm-profiling --max-model-len 8192 --gpu-memory-utilization 0.85 --port $PORT > $OUT/serve_base.log 2>&1 & P=$!
if wait_health $P; then stamp "   up in $(( $(date +%s)-t0 ))s"; ask_all base; echo SERVE_base_OK >> $OUT/status.txt; else echo SERVE_base_FAILED >> $OUT/status.txt; fi
kill $P 2>/dev/null; wait $P 2>/dev/null; sleep 5
if [ -n "$CK" ]; then stamp "serve dspark from $CK"; t0=$(date +%s)
  CUDA_VISIBLE_DEVICES=0 vllm serve $MODEL --served-model-name apertus15-8b --trust-remote-code --skip-mm-profiling --max-model-len 8192 --max-num-seqs 64 --max-num-batched-tokens 16384 --gpu-memory-utilization 0.85 --port $PORT \
    --speculative-config "{\"method\":\"dspark\",\"model\":\"$CK\",\"num_speculative_tokens\":7}" > $OUT/serve_dspark.log 2>&1 & P=$!
  if wait_health $P; then stamp "   up in $(( $(date +%s)-t0 ))s"; ask_all dspark
    python3 - <<PY | tee -a $TL
import urllib.request
t=urllib.request.urlopen("http://localhost:$PORT/metrics").read().decode()
g=lambda k: sum(float(l.rsplit(" ",1)[1]) for l in t.splitlines() if l.startswith("vllm:spec_decode_"+k+"_total"))
d,a=g("num_drafts"),g("num_accepted_tokens"); print("   dspark accepted length: %.2f (drafts %d, accepted %d)" % (1+a/d if d else 0, d, a))
PY
    if cmp -s $OUT/answers_base.txt $OUT/answers_dspark.txt; then stamp "   greedy outputs IDENTICAL to no-speculation"; else stamp "   greedy outputs DIFFER from no-speculation"; diff <(head -c 3000 $OUT/answers_base.txt) <(head -c 3000 $OUT/answers_dspark.txt) | head -20; fi
    echo SERVE_dspark_OK >> $OUT/status.txt
  else echo SERVE_dspark_FAILED >> $OUT/status.txt; grep -E "Error|Traceback|size mismatch" $OUT/serve_dspark.log | tail -6 | cut -c1-200; fi
  kill $P 2>/dev/null; wait $P 2>/dev/null; sleep 5
fi
# ---- B6 (optional): CUDA xIELU kernel, then one online epoch
if [ "${TRY_XIELU:-1}" = 1 ]; then stamp "xielu kernel install attempt (10 min cap)"; t0=$(date +%s)
  timeout 600 pip install -q --no-deps --target $EX "git+https://github.com/nickjbrowning/XIELU" > $OUT/xielu_install.log 2>&1; rc=$?; stamp "   pip rc=$rc in $(( $(date +%s)-t0 ))s"; tail -3 $OUT/xielu_install.log | cut -c1-200
  if python -c "import xielu" 2>/dev/null; then stamp "   xielu importable; extraction server with xielu"; t0=$(date +%s); mkdir -p $OUT/hs_b6
    CUDA_VISIBLE_DEVICES=0 python scripts/launch_vllm.py train $MODEL --target-layer-ids $LAYERS --hidden-states-backend file --hidden-states-path $OUT/hs_b6 -- --trust-remote-code --skip-mm-profiling --renderer-num-workers 1 --mm-processor-cache-gb 0 --port $PORT --gpu-memory-utilization 0.85 --max-model-len 8192 > $OUT/extract_vllm_xielu.log 2>&1 & VP=$!
    if wait_health $VP; then stamp "   up in $(( $(date +%s)-t0 ))s; fallback warnings: $(grep -c 'CUDA-fused xIELU not available' $OUT/extract_vllm_xielu.log)"
      COMMON=${COMMON//$HS/$OUT\/hs_b6}
      train b6_online_xielu 1,2 2 1 --total-seq-len 4096 --on-missing generate --on-generate delete
    else echo XIELU_SERVER_FAILED >> $OUT/status.txt; fi
    kill $VP 2>/dev/null; wait $VP 2>/dev/null
  else stamp "   xielu not importable; skipped"; echo XIELU_SKIPPED >> $OUT/status.txt; fi
fi
kill $SMI 2>/dev/null
# ---- summary table from the per-step profile lines (steps >= 1)
python3 - <<'PY' | tee $OUT/summary.txt
import glob, re, os, statistics as st
print("%-22s %6s %9s %9s %9s %9s %9s %9s %8s %9s" % ("variant","ranks","tok/s/rk","tok/s","fetch_ms","fwd_ms","bwd_ms","opt_ms","step0_s","val_eal"))
for f in sorted(glob.glob(os.environ["OUT"]+"/train_*.log")):
    s=open(f, errors="ignore").read(); name=os.path.basename(f)[6:-4]
    steps=re.findall(r"profile/fetch_ms=([0-9.e+-]+) profile/fwd_ms=([0-9.e+-]+) profile/bwd_ms=([0-9.e+-]+) profile/opt_ms=([0-9.e+-]+) profile/step_ms=([0-9.e+-]+) profile/tokens_per_s=([0-9.e+-]+) profile/fetch_frac=([0-9.e+-]+) epoch=(\d+) step=(\d+)", s.replace("\n"," "))
    if not steps: print("%-22s no profile lines" % name); continue
    rows=[tuple(map(float,x)) for x in steps]; later=[r for r in rows if r[8]>=1] or rows
    m=lambda i: st.median(r[i] for r in later)
    rk=f[:-4]+".ranks"; ranks=int(open(rk).read().strip()) if os.path.exists(rk) else 0
    step0=rows[0][4]/1000
    eal=re.findall(r"val/eal_epoch=([0-9.]+)", s); eal=eal[-1] if eal else "-"
    print("%-22s %6s %9.0f %9.0f %9.1f %9.1f %9.1f %9.1f %8.0f %9s" % (name, ranks or "?", m(5), m(5)*(ranks or 1), m(0), m(1), m(2), m(3), step0, eal))
PY
stamp "done: $(cat $OUT/status.txt | tr '\n' ' ')"
