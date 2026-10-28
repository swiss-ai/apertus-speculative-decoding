#!/bin/bash
# shellcheck disable=SC2046,SC2086
set -ex

export RAY_CGRAPH_get_timeout=1800
export no_proxy="0.0.0.0,$no_proxy"
export NO_PROXY="0.0.0.0,$NO_PROXY"

# User-supplied pre-launch commands
echo 'Running pre-launch commands...'
export PYTHONPATH=/iopsstor/scratch/cscs/faruk_zahiragic/apertus-pilot-code/results/8b/serving-pilot/dev-dspark-20261028-a/source/src:"${PYTHONPATH:-}"
export OPENTELA_BIN=/iopsstor/scratch/cscs/faruk_zahiragic/apertus-pilot-code/results/8b/serving-pilot/dev-dspark-20261028-a/bin/otela-arm64
python3 -m apertus_bench.research_pilot preflight /iopsstor/scratch/cscs/faruk_zahiragic/apertus-pilot-code/results/8b/serving-pilot/dev-dspark-20261028-a

# shellcheck disable=SC2034
replica_head_ip="$1"

$OPENTELA_BIN start \
    --bootstrap.addr "/ip4/148.187.108.177/tcp/43905/p2p/QmbUKJkCfotDzbFE5uoTsXD4GRyPHjzZC1f2yAGLoeBMn9" \
    --service.name llm \
    --service.port 8080 \
    --label launched_by=$USER \
    --label slurm_job_id=$SLURM_JOB_ID \
    --label slurm_partition=${SLURM_JOB_PARTITION:-unknown} \
    --label worker_group_id=$SLURM_JOB_ID \
    --label framework=vllm \
    --label served_model_name=faruk_zahiragic/swiss-ai/Apertus-v1.5-8B-dspark-e10-k3 \
    --label 'framework_args=--port 8080 --model /capstor/store/cscs/swissai/infra01/hf_models/models/swiss-ai/Apertus-v1.5-8B --served-model-name faruk_zahiragic/swiss-ai/Apertus-v1.5-8B-dspark-e10-k3 --chat-template-content-format string --tensor-parallel-size 1 --gpu-memory-utilization 0.8 --max-model-len 32768 --enable-auto-tool-choice --tool-call-parser apertus --speculative-config.method dspark --speculative-config.model /capstor/store/cscs/swissai/infra01/speculative-decoding/DSpark_Apertus_1.5_8B/dspark_apertus15-8b_open-perfectblend_thinking-off_2026-10-04_epoch10_final --speculative-config.num_speculative_tokens 3 --compilation-config.pass_config.fuse_allreduce_rms false --chat-template /iopsstor/scratch/cscs/faruk_zahiragic/apertus-pilot-code/results/8b/serving-pilot/dev-dspark-20261028-a/chat_template.jinja --disable-log-requests --dtype bfloat16 --no-enable-prefix-caching --max-num-batched-tokens 16384 --max-num-seqs 256' \
    --label started_at=$(date -u +%FT%TZ) \
    --label expires_at=$(date -u -d "+5400 seconds" +%FT%TZ) \
    --subprocess "python3 -m vllm.entrypoints.openai.api_server --port 8080 --model /capstor/store/cscs/swissai/infra01/hf_models/models/swiss-ai/Apertus-v1.5-8B     --served-model-name faruk_zahiragic/swiss-ai/Apertus-v1.5-8B-dspark-e10-k3     --chat-template-content-format string     --tensor-parallel-size 1     --gpu-memory-utilization 0.8     --max-model-len 32768     --enable-auto-tool-choice     --tool-call-parser apertus     --speculative-config.method dspark     --speculative-config.model /capstor/store/cscs/swissai/infra01/speculative-decoding/DSpark_Apertus_1.5_8B/dspark_apertus15-8b_open-perfectblend_thinking-off_2026-10-04_epoch10_final     --speculative-config.num_speculative_tokens 3     --compilation-config.pass_config.fuse_allreduce_rms false     --chat-template /iopsstor/scratch/cscs/faruk_zahiragic/apertus-pilot-code/results/8b/serving-pilot/dev-dspark-20261028-a/chat_template.jinja --disable-log-requests --dtype bfloat16 --no-enable-prefix-caching --max-num-batched-tokens 16384 --max-num-seqs 256"
