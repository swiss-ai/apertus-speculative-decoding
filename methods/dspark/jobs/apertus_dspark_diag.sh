#!/bin/bash
# Diagnose the train->serve gap of the 100k DSpark drafter and redo the AdamW+16k speed probe.
# Four servers in parallel on GPUs 0-3, same 32 math+HumanEval prompts, greedy, thinking off:
#   S0 baseline (no speculation)         S1 dspark path, draft_sample_method greedy (as in the scale job)
#   S2 dspark path through a fork copy whose translation honours sample_from_anchor (anchor-as-first)         S3 the same weights through the fork's dflash path (config copy, model type dflash)
# Then: greedy-equality on 32 prompts (S0 vs S1, S0 vs S3), per-position acceptance for S1-S3, thinking-on repeat for S1/S3,
# then run 3 (AdamW, 16k packing, 4 GPUs, from the run-1 cache, 1 epoch, 4 dataloader workers) on all four GPUs.
set +e; export SPEC=${SPEC:-/capstor/scratch/cscs/zyu/spec}; export OUT=$SPEC/runs/apertus_dspark_diag; mkdir -p $OUT
MODEL=$SPEC/hf/hub/models--swiss-ai--Apertus-v1.5-8B/snapshots/a411d838600baf0e3635a3daf66fb7c55fc97bb6
SC=$SPEC/runs/apertus_dspark_scale; CK=$SC/ckpt_run1/checkpoint_best; LAYERS="1 8 15 22 29"
TAG=${TAG:-ghcr}; EX=$SPEC/venvs/apertus-extra-$TAG; export PYTHONPATH=$SPEC/repos/speculators/src:$SPEC/repos/speculators/hs_connectors/src:$EX:${PYTHONPATH:-}
export TORCHINDUCTOR_CACHE_DIR=$SPEC/cache/inductor
command -v python >/dev/null || python() { python3 "$@"; }; export -f python 2>/dev/null
source $SPEC/jobs/side_deps.sh
command -v python >/dev/null || python() { python3 "$@"; }; export -f python 2>/dev/null
source $SPEC/jobs/fork_patch.sh
cd $SPEC/repos/speculators
TL=$OUT/timeline.txt; : > $TL; : > $OUT/status.txt
stamp() { echo "=== [$(date +%T)] $(date +%s) $*" | tee -a $TL; }
wait_port() { for i in $(seq 1 200); do curl -sf localhost:$1/health >/dev/null && return 0; kill -0 $2 2>/dev/null || return 1; sleep 5; done; return 1; }
nvidia-smi --query-gpu=timestamp,index,utilization.gpu,memory.used --format=csv,noheader -l 30 > $OUT/gpu.csv 2>/dev/null & SMI=$!
# dflash-path copy of the checkpoint: symlinked weights, config with the model type changed
CKD=$OUT/ckpt_as_dflash; rm -rf $CKD; mkdir -p $CKD; ln -s $CK/model.safetensors $CKD/model.safetensors; [ -f $CK/config.py ] && cp $CK/config.py $CKD/
python3 - "$CK/config.json" "$CKD/config.json" <<'PY'
import json, sys
c=json.load(open(sys.argv[1])); c["speculators_model_type"]="dflash"
for k in ("markov_rank","enable_confidence_head","confidence_head_with_markov","markov_head_type","confidence_head_alpha"): c.pop(k, None)
json.dump(c, open(sys.argv[2],"w"), indent=1); print("   dflash-path config written; keys dropped where present")
PY
# S2: a second copy of the patched fork whose speculators->HF translation honours the checkpoint's sample_from_anchor
# (the fork hard-codes dspark_bonus_anchor=True, i.e. the 1+N layout, while speculators' DSpark default trains anchor-as-first)
PV2=$OUT/vllm_fork_bonusfix; if [ ! -f $PV2/vllm/__init__.py ]; then mkdir -p $PV2 && cp -r $SPEC/vllm_fork_patched_ghcr/vllm $PV2/; fi
AL=$PV2/vllm/transformers_utils/configs/speculators/algos.py
if grep -q 'dspark_bonus_anchor"\] = True' $AL; then
  sed -i 's|    pre_trained_config\["dspark_bonus_anchor"\] = True|    _sfa = config_dict.get("sample_from_anchor"); _sfa = True if _sfa is None else bool(_sfa)  # patched: speculators dspark default is anchor-as-first\n    pre_trained_config["dspark_bonus_anchor"] = not _sfa  # patched: follow the checkpoint instead of assuming the 1+N layout|' $AL
  grep -n "dspark_bonus_anchor" $AL | cut -c1-160; python3 -m py_compile $AL && echo "   fork copy patched and compiles"
else echo "   fork copy already patched or text differs"; grep -n "dspark_bonus_anchor" $AL | cut -c1-160; fi
python3 -c "import json; c=json.load(open('$CK/config.json')); print('   checkpoint sample_from_anchor =', c.get('sample_from_anchor'))"
SERVE="vllm serve $MODEL --served-model-name apertus15-8b --trust-remote-code --skip-mm-profiling --max-model-len 8192 --max-num-seqs 64 --max-num-batched-tokens 16384 --gpu-memory-utilization 0.85"
stamp "starting four servers"
CUDA_VISIBLE_DEVICES=0 $SERVE --port 8000 > $OUT/serve_S0_base.log 2>&1 & P0=$!
CUDA_VISIBLE_DEVICES=1 $SERVE --port 8001 --speculative-config "{\"method\":\"dspark\",\"model\":\"$CK\",\"num_speculative_tokens\":7}" > $OUT/serve_S1_dspark_greedy.log 2>&1 & P1=$!
CUDA_VISIBLE_DEVICES=2 PYTHONPATH=$PV2:$PYTHONPATH $SERVE --port 8002 --speculative-config "{\"method\":\"dspark\",\"model\":\"$CK\",\"num_speculative_tokens\":7}" > $OUT/serve_S2_dspark_anchorfix.log 2>&1 & P2=$!
CUDA_VISIBLE_DEVICES=3 $SERVE --port 8003 --speculative-config "{\"method\":\"dflash\",\"model\":\"$CKD\",\"num_speculative_tokens\":7}" > $OUT/serve_S3_dflashpath.log 2>&1 & P3=$!
declare -A UP
for s in "8000 S0 $P0" "8001 S1 $P1" "8002 S2 $P2" "8003 S3 $P3"; do set -- $s; if wait_port $1 $3; then UP[$2]=1; stamp "   $2 up on port $1"; else UP[$2]=0; stamp "   $2 FAILED to start"; grep -E "Error|Traceback|size mismatch|KeyError|Missing" $OUT/serve_$2_*.log | tail -4 | cut -c1-200; fi; done
# per-server acceptance on 32 prompts, both modes, and captured outputs for the equality check
cat > $OUT/probe.py <<'PY'
import json, sys, os, glob, time, threading, queue, requests
url, name, out_json, n, max_tokens, think = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]), int(sys.argv[5]), sys.argv[6]=="true"
ds = glob.glob(os.path.expandvars("$SPEC/hf/hub/datasets--RedHatAI--speculator_benchmarks/snapshots/*/"))[0]
prompts=[]
for f in ["math_reasoning.jsonl","HumanEval.jsonl"]:
    rows=[json.loads(l) for l in open(os.path.join(ds,f))][: n//2]
    prompts += [(f, r.get("prompt") or r.get("text") or next(v for v in r.values() if isinstance(v,str))) for r in rows]
def m():
    t=requests.get(url+"/metrics", timeout=30).text; c={}
    for line in t.splitlines():
        if line.startswith("vllm:spec_decode"):
            k,v=line.rsplit(" ",1); c[k]=c.get(k,0.0)+float(v)
    return c
def sm(c,key): return sum(v for k,v in c.items() if k.startswith(key))
before=m(); outs=[None]*len(prompts); q=queue.Queue(); [q.put(i) for i in range(len(prompts))]
def worker():
    while True:
        try: i=q.get_nowait()
        except queue.Empty: return
        f,p=prompts[i]
        r=requests.post(url+"/v1/chat/completions", json={"model":"apertus15-8b","max_tokens":max_tokens,"temperature":0,"chat_template_kwargs":{"enable_thinking":think},"messages":[{"role":"user","content":p}]}, timeout=900).json()
        outs[i]={"file":f,"text":r["choices"][0]["message"]["content"],"tokens":r.get("usage",{}).get("completion_tokens",0)}
t0=time.time(); th=[threading.Thread(target=worker) for _ in range(8)]; [t.start() for t in th]; [t.join() for t in th]; wall=time.time()-t0
after=m(); drafts=sm(after,"vllm:spec_decode_num_drafts")-sm(before,"vllm:spec_decode_num_drafts"); acc=sm(after,"vllm:spec_decode_num_accepted_tokens_total")-sm(before,"vllm:spec_decode_num_accepted_tokens_total")
dt=sm(after,"vllm:spec_decode_num_draft_tokens")-sm(before,"vllm:spec_decode_num_draft_tokens")
import re
pp={}
for k,v in after.items():
    mm=re.search(r'accepted_tokens_per_pos.*position="(\d+)"',k)
    if mm: pp[int(mm.group(1))]=pp.get(int(mm.group(1)),0.0)+v-before.get(k,0.0)
gen=sum(o["tokens"] for o in outs)
res={"name":name,"think":think,"prompts":len(prompts),"gen_tokens":gen,"tok_per_s":gen/wall,"drafts":drafts,"draft_tokens":dt,"accepted_tokens":acc,"accepted_length_incl_bonus":(1+acc/drafts) if drafts else None,"survival_per_pos":{k:(pp[k]/drafts if drafts else None) for k in sorted(pp)},"outputs":outs}
json.dump(res, open(out_json,"w")); print("   %-22s think=%-5s accepted_length=%s  pos0=%s  gen=%d tok/s=%.0f" % (name, think, ("%.2f"%res["accepted_length_incl_bonus"]) if drafts else "n/a", ("%.2f"%res["survival_per_pos"][0]) if 0 in res["survival_per_pos"] and drafts else "n/a", gen, gen/wall))
PY
for mode in false true; do for s in "8000 S0_base" "8001 S1_dspark_greedy" "8002 S2_dspark_anchorfix" "8003 S3_dflashpath"; do set -- $s; k=${2%%_*}; [ "${UP[$k]}" = 1 ] || continue; stamp "probe $2 thinking=$mode"; python3 $OUT/probe.py http://localhost:$1 $2 $OUT/probe_${2}_think_$mode.json 32 $([ $mode = true ] && echo 1024 || echo 384) $mode 2>&1 | tee -a $TL; done; done
python3 - "$OUT" <<'PY' | tee -a $OUT/timeline.txt
import json, sys, os
O=sys.argv[1]
def load(n):
    p=f"{O}/probe_{n}.json"; return json.load(open(p)) if os.path.exists(p) else None
for mode in ("false","true"):
    base=load(f"S0_base_think_{mode}")
    if not base: continue
    for n in ("S1_dspark_greedy","S2_dspark_anchorfix","S3_dflashpath"):
        d=load(f"{n}_think_{mode}")
        if not d: continue
        same=sum(1 for a,b in zip(base["outputs"],d["outputs"]) if a and b and a["text"]==b["text"]); tot=len(base["outputs"])
        divs=[]
        for a,b in zip(base["outputs"],d["outputs"]):
            if a and b and a["text"]!=b["text"]:
                x,y=a["text"],b["text"]; k=next((j for j in range(min(len(x),len(y))) if x[j]!=y[j]), min(len(x),len(y))); divs.append(k)
        print("   greedy equality vs baseline, thinking=%s, %s: %d/%d identical; first divergence chars: %s" % (mode, n, same, tot, sorted(divs)[:10]))
PY
for P in $P0 $P1 $P2 $P3; do kill $P 2>/dev/null; done; sleep 10; for P in $P0 $P1 $P2 $P3; do wait $P 2>/dev/null; done
# run 3 redo: AdamW + 16k on 4 GPUs from the run-1 cache, 1 epoch, fewer dataloader workers (host RAM)
stamp "run3 redo: adamw seq16384 4 gpus, num-workers 4 prefetch 2"; rm -rf $OUT/ckpt_run3; echo 4 > $OUT/train_run3.ranks; t0=$(date +%s)
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --standalone --nproc_per_node 4 -m speculators.train --verifier-name-or-path $MODEL --data-path $SC/data --hidden-states-backend file --hidden-states-path $SC/hs_cache --loss-implementation eager --vllm-endpoint http://localhost:8000/v1 --target-layer-ids $LAYERS \
  --speculator-type dspark --lr 3e-4 --block-size 8 --max-anchors 1024 --num-layers 5 --markov-rank 256 --enable-confidence-head --confidence-head-with-markov --loss-fn '{"ce":0.1,"tv":0.9}' \
  --epochs 1 --total-seq-len 16384 --optimizer adamw --on-missing raise --num-workers 4 --prefetch-factor 2 --save-path $OUT/ckpt_run3 > $OUT/train_run3.log 2>&1; rc=$?
stamp "   run3 rc=$rc wall=$(( $(date +%s)-t0 ))s; MaxRSS: $(sstat -j $SLURM_JOB_ID --format=MaxRSS -n 2>/dev/null | head -1 | tr -d ' ')"
python3 $SPEC/jobs/bench_summary.py $OUT | tee -a $TL
kill $SMI 2>/dev/null
stamp "done"
