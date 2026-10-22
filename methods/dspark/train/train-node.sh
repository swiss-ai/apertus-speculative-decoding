#!/usr/bin/env bash
# One node of the DSpark training (port of Yu's apertus_opb_train_v3.sh),
# started once per node by submit-train.sh inside the container. Per node:
# extraction on GPUs 0,1 writing target features to node-local /tmp, and
# trainer ranks on GPUs 2,3 that read from it; ranks of all nodes join one
# torchrun rendezvous. Resume is automatic from the run's checkpoints.
# Trainer settings are Yu's 8B ones. 70B (the defaults): target layers
# 1/20/39/58/77, --fsdp-shard, one TP2 server. 8B (submit-train.sh STAGE=8b):
# Yu's layout, one 1-GPU server per trainer rank (EXTRACT_PER_RANK=1), no FSDP.
# NUM_LAYERS sets the drafter depth (Yu's 5; docs/small-drafter.md trains
# 1-3), FROM_PRETRAINED a warm start (methods/dspark/train/truncate_drafter.py).
set -euo pipefail
. "${REPO_ROOT}/methods/dspark/train/common.sh"
: "${RUN_ID:?}" "${DATA:?}" "${HEAD_NODE:?}"
NODES="${SLURM_JOB_NUM_NODES:-1}"
NODE_RANK="${SLURM_NODEID:-0}"
# Five layers evenly spaced from 1 to n-3 over the 70B's 80 (8B: 1 8 15 22 29).
LAYERS="${LAYERS:-1 20 39 58 77}"
EXTRACT_GPUS="${EXTRACT_GPUS:-0,1}"
TRAIN_GPUS="${TRAIN_GPUS:-2,3}"
NUM_LAYERS="${NUM_LAYERS:-5}"
FSDP="${FSDP:-1}"
EXTRACT_PER_RANK="${EXTRACT_PER_RANK:-0}"
# Prefill only, so CUDA graphs buy little; EXTRACT_EAGER=0 enables them.
eager=()
[ "${EXTRACT_EAGER:-1}" = 1 ] && eager=(--enforce-eager)
EXTRACT_TP=$(( $(tr -cd , <<< "${EXTRACT_GPUS}" | wc -c) + 1 ))
RANKS_PER_NODE=$(( $(tr -cd , <<< "${TRAIN_GPUS}" | wc -c) + 1 ))
# Extraction servers: one TP server on all EXTRACT_GPUS, or (EXTRACT_PER_RANK=1)
# one per GPU, local rank r reading from port PORT + r (rank-wrapper.sh).
if [ "${EXTRACT_PER_RANK}" = 1 ]; then
  IFS=, read -r -a SERVER_GPUS <<< "${EXTRACT_GPUS}"
  if [ "${#SERVER_GPUS[@]}" != "${RANKS_PER_NODE}" ]; then
    echo "EXTRACT_PER_RANK=1 needs one extraction GPU per trainer rank" >&2
    exit 2
  fi
  EXTRACT_TP=1
else
  SERVER_GPUS=("${EXTRACT_GPUS}")
fi
export DSPARK_PER_RANK="${EXTRACT_PER_RANK}"
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

stamp "node ${NODE_RANK}/${NODES} head=${HEAD_NODE} run=${RUN_ID} data=${DATA} target=${TARGET_MODEL} layers=${NUM_LAYERS} from=${FROM_PRETRAINED:-scratch} extract=${EXTRACT_GPUS} (TP${EXTRACT_TP}, per_rank=${EXTRACT_PER_RANK}) train=${TRAIN_GPUS} fsdp=${FSDP} rdzv_threshold=${FI_CXI_RDZV_THRESHOLD:-default} dtensor_allgather=${DSPARK_DTENSOR_ALLGATHER}"
nvidia-smi --query-gpu=timestamp,index,utilization.gpu,memory.used --format=csv,noheader -l 60 \
  > "${LOG}/gpu-${JOB}-node${NODE_RANK}.csv" 2> /dev/null &
smi=$!
( while sleep 300; do echo "$(date -u +%FT%TZ) $(free -g | awk '/Mem:/ {print "used_gb=" $3}') tmp=$(du -sh ${HS} 2> /dev/null | cut -f1)"; done ) \
  > "${LOG}/host-${JOB}-node${NODE_RANK}.log" &
mon=$!
# vLLM provenance (vllm_command.txt, vllm.patch, the target's sha256) per
# server under the run; the target is hashed once per run, by node 0.
PROV="${RUN_DIR}/provenance"
servers=()
trap 'kill ${smi} ${mon} 2> /dev/null || true; for s in "${servers[@]}"; do stop_server "${s}"; done' EXIT
for i in "${!SERVER_GPUS[@]}"; do
  suffix=""
  [ "${EXTRACT_PER_RANK}" = 1 ] && suffix="-server${i}"
  hash=(--no-hash-checkpoints)
  if [ "${NODE_RANK}" = 0 ] && [ "${i}" = 0 ] && ! ls "${PROV}"/*/checkpoint_sha256.txt > /dev/null 2>&1; then
    hash=()
  fi
  CUDA_VISIBLE_DEVICES="${SERVER_GPUS[${i}]}" setsid python3 scripts/launch_vllm.py train "${TARGET_MODEL}" \
    --target-layer-ids ${LAYERS} --hidden-states-backend file --hidden-states-path "${HS}" \
    --provenance-dir "${PROV}/${JOB}-node${NODE_RANK}-server${i}" "${hash[@]}" -- \
    --tensor-parallel-size "${EXTRACT_TP}" "${eager[@]}" --trust-remote-code --skip-mm-profiling \
    --compilation-config.pass_config.fuse_allreduce_rms false \
    --renderer-num-workers 1 --mm-processor-cache-gb 0 --port "$(( PORT + i ))" \
    --gpu-memory-utilization "${EXTRACT_MEM:-0.92}" --max-model-len 8192 \
    > "${LOG}/extract-${JOB}-node${NODE_RANK}${suffix}.log" 2>&1 &
  servers+=($!)
  # Yu started his per-rank servers a minute apart.
  if [ "${EXTRACT_PER_RANK}" = 1 ] && [ $(( i + 1 )) -lt "${#SERVER_GPUS[@]}" ]; then sleep 60; fi
done
for i in "${!servers[@]}"; do
  if ! wait_health "$(( PORT + i ))" "${servers[${i}]}" 2400; then
    stamp "EXTRACTION_SERVER_FAILED (server ${i})"
    grep -hE "Error|Traceback" "${LOG}"/extract-"${JOB}"-node"${NODE_RANK}"*.log | tail -5
    exit 1
  fi
  stamp "extraction server ${i} up on $(( PORT + i ))"
done

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
[ "${FSDP}" = 1 ] && extra+=(--fsdp-shard)
[ -n "${FROM_PRETRAINED:-}" ] && extra+=(--from-pretrained "${FROM_PRETRAINED}")
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
  --target-layer-ids ${LAYERS} --speculator-type dspark --num-layers "${NUM_LAYERS}" --block-size 8 \
  --markov-rank 256 --enable-confidence-head --confidence-head-with-markov \
  --loss-fn '{"ce":0.1,"tv":0.9}' --loss-implementation eager --optimizer muon --lr "${LR}" \
  --epochs "${EPOCHS}" --total-seq-len "${SEQ}" --max-anchors "${ANCHORS}" \
  --num-workers "${WORKERS}" --prefetch-factor "${PREFETCH}" --checkpoint-freq "${CKPT_FREQ}" \
  "${extra[@]}" --save-path "${CKPT}" \
  > "${LOG}/train-${JOB}-node${NODE_RANK}.log" 2>&1 || rc=$?
kill "${monitor}" 2> /dev/null || true
stamp "torchrun rc=${rc} wall=$(( $(date +%s) - t0 ))s; checkpoints: $(ls "${CKPT}" 2> /dev/null | tail -3 | tr '\n' ' ')"
grep -oE "val/eal_epoch=[0-9.]+" "${LOG}/train-${JOB}-node${NODE_RANK}.log" | tail -3 | paste -sd" " || true
exit "${rc}"
