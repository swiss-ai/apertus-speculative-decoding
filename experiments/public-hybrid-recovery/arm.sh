#!/bin/bash
set -euo pipefail
run=$1
code=$2
model=$3
arms=(plain snapshot compact plain-repeat)
context_length=2048
if [[ -f $code/configure.sh ]]; then
    source "$code/configure.sh"
fi
arm=${arms[$(( (SLURM_LOCALID + ${GDN_ROTATE:-0}) % 4 ))]}
dir=$run/$arm
mkdir "$dir"
export CUDA_VISIBLE_DEVICES=$SLURM_LOCALID
export HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export TRITON_CACHE_DIR=/tmp/gdn-$SLURM_JOB_ID-$arm/triton
export XDG_CACHE_HOME=/tmp/gdn-$SLURM_JOB_ID-$arm/cache
export SGLANG_CACHE_DIR=/tmp/gdn-$SLURM_JOB_ID-$arm/sglang
export SGLANG_JIT_CACHE_DIR=$SGLANG_CACHE_DIR/jit
export FLASHINFER_WORKSPACE_BASE=$SGLANG_CACHE_DIR/flashinfer
port=$((31100 + SLURM_LOCALID))
common=(--model-path "$model" --host 127.0.0.1 --port "$port" --tp-size 1 --dtype bfloat16
        --context-length "$context_length" --max-running-requests 64 --mem-fraction-static 0.8 --cuda-graph-max-bs-decode 64
        --disable-radix-cache --enable-metrics --random-seed 107)
spec=()
mode=${arm%%-*}
if [[ $mode == compact || $mode == snapshot ]]; then
    spec+=(--speculative-algorithm NEXTN --speculative-num-steps 3 --speculative-eagle-topk 1
           --speculative-num-draft-tokens 4 --speculative-use-rejection-sampling)
fi
if [[ $mode == compact ]]; then
    spec+=(--enable-linear-replayssm-spec)
fi
if [[ -f $code/cache_budget.py ]]; then
    cap=$(python "$code/cache_budget.py" "$mode" "${arm##*-}" "$dir/cache-budget.json")
    common+=(--max-total-tokens "$cap")
fi
printf '%q ' python -m sglang.launch_server "${common[@]}" "${spec[@]}" > "$dir/command.sh"
printf '\n' >> "$dir/command.sh"
python -m sglang.launch_server "${common[@]}" "${spec[@]}" > "$dir/server.log" 2>&1 &
pid=$!
trap 'kill "$pid" 2>/dev/null || true; wait "$pid" 2>/dev/null || true' EXIT
url=http://127.0.0.1:$port
for ((i=0; i<300; i++)); do
    kill -0 "$pid"
    if curl -fsS "$url/health" >/dev/null 2>&1; then break; fi
    sleep 2
done
curl -fsS "$url/server_info" > "$dir/server-info.json" || curl -fsS "$url/get_server_info" > "$dir/server-info.json"
curl -fsS -H 'Content-Type: application/json' -d '{"input_ids":[1,2,3,4],"sampling_params":{"temperature":0,"max_new_tokens":16}}' "$url/generate" > "$dir/warmup.json"
python "$code/groups.py" "$url" "$model" "$run/prompts.json" "$dir"
