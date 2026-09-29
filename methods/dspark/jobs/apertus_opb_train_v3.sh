#!/bin/bash
# Multi-node DSpark training for Apertus 1.5 8B (thinking off, Open-PerfectBlend). Per node: 2 extraction servers (GPUs 0,1; ports 8001/8002)
# each trainer rank (GPUs 2,3) reads from its own server (local rank r -> port 8001+r) via train_rank_wrapper.sh; ranks joined across nodes with torchrun c10d rendezvous.
# TRIAL=1 -> pilot 5k data, 1 epoch, separate save path. Resume is automatic from the save path (numbered checkpoints).
set +e; export SPEC=${SPEC:-/capstor/scratch/cscs/zyu/spec}; R=$SPEC/runs/apertus_opb
MODEL=$SPEC/hf/hub/models--swiss-ai--Apertus-v1.5-8B/snapshots/a411d838600baf0e3635a3daf66fb7c55fc97bb6
LAYERS="1 8 15 22 29"; NODES=${SLURM_JOB_NUM_NODES:-1}; RANK=${SLURM_NODEID:-0}; HEAD=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -1)
if [ "${TRIAL:-0}" = 1 ]; then DATA=$SPEC/runs/apertus_pipe5k/data; EPOCHS=1; CK=$R/ckpt_trial_${NODES}n; else DATA=$R/data; EPOCHS=${EPOCHS:-10}; CK=$R/ckpt_dspark_nothink; fi
SEQ=${SEQ:-16384}; ANCHORS=${ANCHORS:-1024}; LR=${LR:-3e-4}
TAG=ghcr; EX=$SPEC/venvs/apertus-extra-$TAG; export PYTHONPATH=$SPEC/repos/speculators/src:$SPEC/repos/speculators/hs_connectors/src:$EX:${PYTHONPATH:-}
export SPEC_PG_TIMEOUT_MIN=${SPEC_PG_TIMEOUT_MIN:-60}; export TORCHINDUCTOR_CACHE_DIR=$SPEC/cache/inductor; mkdir -p $TORCHINDUCTOR_CACHE_DIR
command -v python >/dev/null || python() { python3 "$@"; }; export -f python 2>/dev/null
source $SPEC/jobs/side_deps.sh
command -v python >/dev/null || python() { python3 "$@"; }; export -f python 2>/dev/null
source $SPEC/jobs/fork_patch.sh
cd $SPEC/repos/speculators
LOG=$R/logs; mkdir -p $LOG $CK; HS=/tmp/hidden_states; rm -rf $HS; mkdir -p $HS
stamp() { echo "=== [$(date +%T)] node$RANK $(hostname) $*"; }
wait_port() { for i in $(seq 1 200); do curl -sf localhost:$1/health >/dev/null && return 0; kill -0 $2 2>/dev/null || return 1; sleep 5; done; return 1; }
stamp "nodes=$NODES head=$HEAD trial=${TRIAL:-0} data=$DATA epochs=$EPOCHS seq=$SEQ anchors=$ANCHORS save=$CK"
nvidia-smi --query-gpu=timestamp,index,utilization.gpu,memory.used --format=csv,noheader -l 60 > $LOG/gpu_${SLURM_JOB_ID}_node$RANK.csv 2>/dev/null & SMI=$!
for g in 0 1; do port=$((8001+g)); CUDA_VISIBLE_DEVICES=$g python scripts/launch_vllm.py train $MODEL --target-layer-ids $LAYERS --hidden-states-backend file --hidden-states-path $HS -- --trust-remote-code --skip-mm-profiling --renderer-num-workers 1 --mm-processor-cache-gb 0 --port $port --gpu-memory-utilization 0.85 --max-model-len 8192 > $LOG/extract_${SLURM_JOB_ID}_node${RANK}_gpu$g.log 2>&1 & eval VP$g=$!; sleep 60; done
PP=""
ok=1; for g in 0 1; do port=$((8001+g)); eval vp=\$VP$g; wait_port $port $vp && stamp "server gpu$g up on $port" || { stamp "server gpu$g FAILED"; grep -E "Error|Traceback" $LOG/extract_${SLURM_JOB_ID}_node${RANK}_gpu$g.log | tail -3; ok=0; }; done
[ $ok = 1 ] || { kill $VP0 $VP1 $SMI 2>/dev/null; exit 1; }
t0=$(date +%s); stamp "torchrun start"
CUDA_VISIBLE_DEVICES=2,3 torchrun --nnodes $NODES --nproc_per_node 2 --node_rank $RANK --rdzv_backend c10d --rdzv_endpoint $HEAD:29500 --rdzv_id $SLURM_JOB_ID --no-python bash $SPEC/jobs/train_rank_wrapper.sh \
  --verifier-name-or-path $MODEL --data-path $DATA --hidden-states-backend file --hidden-states-path $HS --on-missing generate --on-generate delete \
  --target-layer-ids $LAYERS --speculator-type dspark --num-layers 5 --block-size 8 --markov-rank 256 --enable-confidence-head --confidence-head-with-markov --loss-fn '{"ce":0.1,"tv":0.9}' \
  --loss-implementation eager --optimizer muon --lr $LR --epochs $EPOCHS --total-seq-len $SEQ --max-anchors $ANCHORS --num-workers ${WORKERS:-8} --prefetch-factor ${PREFETCH:-4} --checkpoint-freq 0.125 --save-path $CK \
  > $LOG/train_${SLURM_JOB_ID}_node$RANK.log 2>&1; rc=$?
stamp "torchrun rc=$rc wall=$(( $(date +%s)-t0 ))s; checkpoints: $(ls $CK 2>/dev/null | tr '\n' ' ' | cut -c1-200)"
[ $RANK = 0 ] && { python3 $SPEC/jobs/bench_summary.py $LOG 2>/dev/null | grep -E "variant|train_${SLURM_JOB_ID}" ; grep -oE "val/eal_epoch=[0-9.]+" $LOG/train_${SLURM_JOB_ID}_node0.log | tail -3 | paste -sd" "; }
kill $VP0 $VP1 $SMI 2>/dev/null; wait 2>/dev/null; stamp "done rc=$rc"; exit $rc
