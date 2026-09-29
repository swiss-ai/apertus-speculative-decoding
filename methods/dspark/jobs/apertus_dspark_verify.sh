#!/bin/bash
# Verify the fixed DSpark serving path: baseline + fixed path at k=3/5/7 (staggered starts), 64 math+HumanEval prompts, both modes,
# greedy equality vs baseline, per-position survival, and single-stream speed (ITL) per k.
set +e; export SPEC=${SPEC:-/capstor/scratch/cscs/zyu/spec}; export OUT=$SPEC/runs/apertus_dspark_verify; mkdir -p $OUT
MODEL=$SPEC/hf/hub/models--swiss-ai--Apertus-v1.5-8B/snapshots/a411d838600baf0e3635a3daf66fb7c55fc97bb6
CK=$SPEC/runs/apertus_dspark_scale/ckpt_run1/checkpoint_best
TAG=${TAG:-ghcr}; EX=$SPEC/venvs/apertus-extra-$TAG; export PYTHONPATH=$SPEC/repos/speculators/src:$SPEC/repos/speculators/hs_connectors/src:$EX:${PYTHONPATH:-}
command -v python >/dev/null || python() { python3 "$@"; }; export -f python 2>/dev/null
source $SPEC/jobs/side_deps.sh
command -v python >/dev/null || python() { python3 "$@"; }; export -f python 2>/dev/null
source $SPEC/jobs/fork_patch.sh          # now includes the dspark anchor-layout fix
cd $SPEC/repos/speculators
TL=$OUT/timeline.txt; : > $TL
stamp() { echo "=== [$(date +%T)] $(date +%s) $*" | tee -a $TL; }
wait_port() { for i in $(seq 1 200); do curl -sf localhost:$1/health >/dev/null && return 0; kill -0 $2 2>/dev/null || return 1; sleep 5; done; return 1; }
grep -n "dspark_bonus_anchor" $PV/vllm/transformers_utils/configs/speculators/algos.py | cut -c1-140
SERVE="vllm serve $MODEL --served-model-name apertus15-8b --trust-remote-code --skip-mm-profiling --max-model-len 8192 --max-num-seqs 64 --max-num-batched-tokens 16384 --gpu-memory-utilization 0.85"
stamp "servers (staggered by 75 s to avoid the shared compile-cache race)"
CUDA_VISIBLE_DEVICES=0 $SERVE --port 8000 > $OUT/serve_base.log 2>&1 & P0=$!; sleep 75
CUDA_VISIBLE_DEVICES=1 $SERVE --port 8001 --speculative-config "{\"method\":\"dspark\",\"model\":\"$CK\",\"num_speculative_tokens\":3}" > $OUT/serve_k3.log 2>&1 & P1=$!; sleep 75
CUDA_VISIBLE_DEVICES=2 $SERVE --port 8002 --speculative-config "{\"method\":\"dspark\",\"model\":\"$CK\",\"num_speculative_tokens\":5}" > $OUT/serve_k5.log 2>&1 & P2=$!; sleep 75
CUDA_VISIBLE_DEVICES=3 $SERVE --port 8003 --speculative-config "{\"method\":\"dspark\",\"model\":\"$CK\",\"num_speculative_tokens\":7}" > $OUT/serve_k7.log 2>&1 & P3=$!
declare -A UP
for s in "8000 base $P0" "8001 k3 $P1" "8002 k5 $P2" "8003 k7 $P3"; do set -- $s; if wait_port $1 $3; then UP[$2]=1; stamp "   $2 up on port $1"; else UP[$2]=0; stamp "   $2 FAILED to start"; grep -E "Error|Traceback" $OUT/serve_$2.log | tail -3 | cut -c1-200; fi; done
cp $SPEC/runs/apertus_dspark_diag/probe.py $OUT/probe.py
# batch probes: 64 prompts, 8 concurrent, both modes
for mode in false true; do for s in "8000 base" "8001 k3" "8002 k5" "8003 k7"; do set -- $s; [ "${UP[$2]}" = 1 ] || continue; stamp "probe $2 thinking=$mode (64 prompts, 8 concurrent)"; python3 $OUT/probe.py http://localhost:$1 $2 $OUT/probe_${2}_think_$mode.json 64 $([ $mode = true ] && echo 1024 || echo 384) $mode 2>&1 | tee -a $TL; done; done
# single-stream speed: 8 prompts sequentially, thinking off, 384 tokens -> ms per output token
cat > $OUT/single.py <<'PY'
import json, sys, time, requests, glob, os
url, name = sys.argv[1], sys.argv[2]
ds = glob.glob(os.path.expandvars("$SPEC/hf/hub/datasets--RedHatAI--speculator_benchmarks/snapshots/*/"))[0]
rows=[json.loads(l) for l in open(os.path.join(ds,"math_reasoning.jsonl"))][:8]
tot_t=0; tot_n=0
for r in rows:
    p=r.get("prompt") or r.get("text") or next(v for v in r.values() if isinstance(v,str))
    t=time.time(); j=requests.post(url+"/v1/chat/completions", json={"model":"apertus15-8b","max_tokens":384,"temperature":0,"chat_template_kwargs":{"enable_thinking":False},"messages":[{"role":"user","content":p}]}, timeout=600).json()
    tot_t+=time.time()-t; tot_n+=j.get("usage",{}).get("completion_tokens",0)
print("   %-5s single-stream: %d tokens in %.1f s -> %.2f ms/token (%.0f tok/s)" % (name, tot_n, tot_t, 1000*tot_t/max(tot_n,1), tot_n/max(tot_t,1e-9)))
PY
for s in "8000 base" "8001 k3" "8002 k5" "8003 k7"; do set -- $s; [ "${UP[$2]}" = 1 ] || continue; python3 $OUT/single.py http://localhost:$1 $2 2>&1 | tee -a $TL; done
python3 - "$OUT" <<'PY' | tee -a $OUT/timeline.txt
import json, sys, os
O=sys.argv[1]
def load(n):
    p=f"{O}/probe_{n}.json"; return json.load(open(p)) if os.path.exists(p) else None
for mode in ("false","true"):
    base=load(f"base_think_{mode}")
    if not base: print("   no baseline for thinking=%s" % mode); continue
    for n in ("k3","k5","k7"):
        d=load(f"{n}_think_{mode}")
        if not d: continue
        pairs=[(a,b) for a,b in zip(base["outputs"],d["outputs"]) if a and b]
        same=sum(1 for a,b in pairs if a["text"]==b["text"]); divs=[]
        for a,b in pairs:
            if a["text"]!=b["text"]:
                x,y=a["text"],b["text"]; k=next((j for j in range(min(len(x),len(y))) if x[j]!=y[j]), min(len(x),len(y))); divs.append(k)
        print("   greedy equality thinking=%s %s: %d/%d identical; divergence chars: %s" % (mode, n, same, len(pairs), sorted(divs)[:12]))
        print("   %s thinking=%s accepted_length=%.2f survival=%s" % (n, mode, d["accepted_length_incl_bonus"], {k: round(v,2) for k,v in list(d["survival_per_pos"].items())[:7]}))
PY
for P in $P0 $P1 $P2 $P3; do kill $P 2>/dev/null; done; sleep 5
stamp "done"
