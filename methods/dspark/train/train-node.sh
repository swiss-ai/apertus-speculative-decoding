#!/usr/bin/env bash
# One node of the DSpark 70B training (port of Yu's apertus_opb_train_v3.sh),
# started once per node by submit-train.sh inside the container. Per node: one
# TP2 extraction server on GPUs 0,1 writing target features to node-local /tmp,
# and trainer ranks on GPUs 2,3 that read from it; ranks of all nodes join one
# torchrun rendezvous. Resume is automatic from the run's checkpoints.
# Trainer settings are Yu's 8B ones; the 70B changes are the target layers,
# --fsdp-shard and the TP2 server.
set -euo pipefail
. "${REPO_ROOT}/methods/dspark/train/common.sh"
: "${RUN_ID:?}" "${DATA:?}" "${HEAD_NODE:?}"
NODES="${SLURM_JOB_NUM_NODES:-1}"
NODE_RANK="${SLURM_NODEID:-0}"
# Five layers evenly spaced from 1 to n-3 over the 70B's 80 (8B: 1 8 15 22 29).
LAYERS="${LAYERS:-1 20 39 58 77}"
EXTRACT_GPUS="${EXTRACT_GPUS:-0,1}"
TRAIN_GPUS="${TRAIN_GPUS:-2,3}"
# Prefill only, so CUDA graphs buy little; EXTRACT_EAGER=0 enables them.
eager=()
[ "${EXTRACT_EAGER:-1}" = 1 ] && eager=(--enforce-eager)
EXTRACT_TP=$(( $(tr -cd , <<< "${EXTRACT_GPUS}" | wc -c) + 1 ))
RANKS_PER_NODE=$(( $(tr -cd , <<< "${TRAIN_GPUS}" | wc -c) + 1 ))
PORT=8001
EPOCHS="${EPOCHS:-10}"
SEQ="${SEQ:-16384}"
ANCHORS="${ANCHORS:-1024}"
LR="${LR:-3e-4}"
WORKERS="${WORKERS:-4}"
PREFETCH="${PREFETCH:-2}"
# Yu saved every 1/16 epoch (~2 h here, longer than the hangs above allowed);
# 1/64 (~30 min) bounds the work a stalled job loses. A save takes ~1 min.
CKPT_FREQ="${CKPT_FREQ:-0.015625}"
RUN_DIR="${DSPARK_DIR}/runs/${RUN_ID}"
CKPT="${RUN_DIR}/checkpoints"
LOG="${RUN_DIR}/logs"
HS=/tmp/hidden_states
JOB="${SLURM_JOB_ID:-local}"
mkdir -p "${CKPT}" "${LOG}"
rm -rf "${HS}" && mkdir -p "${HS}"
export SPEC_PG_TIMEOUT_MIN="${SPEC_PG_TIMEOUT_MIN:-60}"
# Every hang of the first six jobs (3691347-52) was an NCCL all-to-all from
# Muon's sharded Newton-Schulz that one rank finished and the others never did.
# CSCS lists NCCL hangs on Alps as a known issue, worked around by disabling
# Slingshot eager messages (docs.cscs.ch/software/communication/nccl). Transport
# only; DSPARK_CXI_EAGER=1 keeps eager messages. DSPARK_DTENSOR_ALLGATHER=1
# (speculators-dtensor-allgather.patch) replaces the all-to-all altogether.
if [ "${DSPARK_CXI_EAGER:-0}" != 1 ]; then
  export FI_CXI_RDZV_GET_MIN=0 FI_CXI_RDZV_THRESHOLD=0 FI_CXI_RDZV_EAGER_SIZE=0
fi
export DSPARK_DTENSOR_ALLGATHER="${DSPARK_DTENSOR_ALLGATHER:-0}"
check_side_deps
cd "${SPECULATORS}"

stamp "node ${NODE_RANK}/${NODES} head=${HEAD_NODE} run=${RUN_ID} data=${DATA} extract=${EXTRACT_GPUS} (TP${EXTRACT_TP}) train=${TRAIN_GPUS} rdzv_threshold=${FI_CXI_RDZV_THRESHOLD:-default} dtensor_allgather=${DSPARK_DTENSOR_ALLGATHER}"
nvidia-smi --query-gpu=timestamp,index,utilization.gpu,memory.used --format=csv,noheader -l 60 \
  > "${LOG}/gpu-${JOB}-node${NODE_RANK}.csv" 2> /dev/null &
smi=$!
( while sleep 300; do echo "$(date -u +%FT%TZ) $(free -g | awk '/Mem:/ {print "used_gb=" $3}') tmp=$(du -sh ${HS} 2> /dev/null | cut -f1)"; done ) \
  > "${LOG}/host-${JOB}-node${NODE_RANK}.log" &
mon=$!
CUDA_VISIBLE_DEVICES="${EXTRACT_GPUS}" setsid python3 scripts/launch_vllm.py train "${TARGET_MODEL}" \
  --target-layer-ids ${LAYERS} --hidden-states-backend file --hidden-states-path "${HS}" -- \
  --tensor-parallel-size "${EXTRACT_TP}" "${eager[@]}" --trust-remote-code --skip-mm-profiling \
  --compilation-config.pass_config.fuse_allreduce_rms false \
  --renderer-num-workers 1 --mm-processor-cache-gb 0 --port "${PORT}" \
  --gpu-memory-utilization "${EXTRACT_MEM:-0.92}" --max-model-len 8192 \
  > "${LOG}/extract-${JOB}-node${NODE_RANK}.log" 2>&1 &
server=$!
trap 'kill ${smi} ${mon} 2> /dev/null || true; stop_server ${server}' EXIT
if ! wait_health "${PORT}" "${server}" 2400; then
  stamp "EXTRACTION_SERVER_FAILED"
  grep -E "Error|Traceback" "${LOG}/extract-${JOB}-node${NODE_RANK}.log" | tail -5
  exit 1
fi
stamp "extraction server up on ${PORT}"

if [ -n "${CHECK_ROWS:-}" ]; then
  python3 "${REPO_ROOT}/methods/dspark/train/check_hidden_states.py" \
    --endpoint "http://localhost:${PORT}/v1" --target "${TARGET_MODEL}" --rows "${CHECK_ROWS}" \
    --layers "$(wc -w <<< "${LAYERS}")" --device "cuda:${TRAIN_GPUS%%,*}" \
    2>&1 | tee "${LOG}/hs-check-${JOB}-node${NODE_RANK}.log" | tail -4
fi

# Every node's 70B server must be up before torchrun: load times differ by
# minutes and the head's rendezvous store refused late joiners after 60 s
# (job 3681626). Ready files on shared storage act as a barrier.
# A node whose container starts late (1.5 h in job 3691353) must not leave the
# others to time out: wait up to 2 h, then fail so the next job takes over.
touch "${LOG}/ready-${JOB}-node${NODE_RANK}"
ready=0
for i in $(seq 1 720); do
  ready=$(ls "${LOG}"/ready-"${JOB}"-node* 2> /dev/null | wc -l)
  [ "${ready}" -ge "${NODES}" ] && break
  sleep 10
done
if [ "${ready}" -lt "${NODES}" ]; then
  stamp "NODES_NOT_READY: ${ready} of ${NODES} extraction servers up after 2 h"
  exit 1
fi
stamp "all ${NODES} extraction servers up"

# Stall monitor (every node): a rank stuck outside NCCL left the others in a
# collective until the 60-min timeout, 1,300-2,200 steps into each of the first
# four jobs (3691347-50). When the rank-0 log shows no new event for longer than
# its phase allows, dump every trainer process's stacks (SIGUSR1, see
# speculators-stack-dump.patch) and stop torchrun, so the next job resumes.
stall_monitor() {
  local last="" changed now event limit
  changed=$(date +%s)
  while sleep 60; do
    event="$(cat "${LOG}"/train-"${JOB}"-node*.log 2> /dev/null | grep -aoE \
      "global_step=[0-9]+|Saving checkpoint|Checkpoint saved|Validation epoch [0-9]+/[0-9]+ (started|completed)|Training epoch [0-9]+/[0-9]+ started" \
      | tail -1 || true)"
    now=$(date +%s)
    if [ "${event}" != "${last}" ]; then last="${event}"; changed=${now}; continue; fi
    case "${event}" in
      "") limit=2700 ;;
      "Saving checkpoint") limit=2700 ;;
      *started) limit=14400 ;;
      *) limit=$(( ${STALL_MINUTES:-15} * 60 )) ;;
    esac
    if (( now - changed > limit )); then
      stamp "STALL: no event since '${last}' for $(( (now - changed) / 60 )) min; dumping stacks, stopping"
      pkill -USR1 -f "python3 -m speculators.train" || true
      sleep 60
      pkill -TERM -f "torchrun --nnodes" || true
      return
    fi
  done
}
stall_monitor &
monitor=$!

extra=()
[ "${GRAD_CKPT:-0}" = 1 ] && extra+=(--gradient-checkpointing)
[ -n "${MAX_STEPS:-}" ] && extra+=(--max-steps "${MAX_STEPS}")
t0=$(date +%s)
stamp "torchrun start"
rc=0
CUDA_VISIBLE_DEVICES="${TRAIN_GPUS}" DSPARK_PORT="${PORT}" torchrun --nnodes "${NODES}" \
  --nproc_per_node "${RANKS_PER_NODE}" --node_rank "${NODE_RANK}" --rdzv_backend c10d \
  --rdzv_endpoint "${HEAD_NODE}:29500" --rdzv_id "${JOB}" \
  --rdzv_conf join_timeout=1800,read_timeout=600 --no-python \
  bash "${REPO_ROOT}/methods/dspark/train/rank-wrapper.sh" \
  --verifier-name-or-path "${TARGET_MODEL}" --data-path "${DATA}" \
  --hidden-states-backend file --hidden-states-path "${HS}" --on-missing generate --on-generate delete \
  --target-layer-ids ${LAYERS} --speculator-type dspark --num-layers 5 --block-size 8 \
  --markov-rank 256 --enable-confidence-head --confidence-head-with-markov \
  --loss-fn '{"ce":0.1,"tv":0.9}' --loss-implementation eager --optimizer muon --lr "${LR}" \
  --epochs "${EPOCHS}" --total-seq-len "${SEQ}" --max-anchors "${ANCHORS}" \
  --num-workers "${WORKERS}" --prefetch-factor "${PREFETCH}" --checkpoint-freq "${CKPT_FREQ}" \
  --fsdp-shard "${extra[@]}" --save-path "${CKPT}" \
  > "${LOG}/train-${JOB}-node${NODE_RANK}.log" 2>&1 || rc=$?
kill "${monitor}" 2> /dev/null || true
stamp "torchrun rc=${rc} wall=$(( $(date +%s) - t0 ))s; checkpoints: $(ls "${CKPT}" 2> /dev/null | tail -3 | tr '\n' ' ')"
grep -oE "val/eal_epoch=[0-9.]+" "${LOG}/train-${JOB}-node${NODE_RANK}.log" | tail -3 | paste -sd" " || true
exit "${rc}"
