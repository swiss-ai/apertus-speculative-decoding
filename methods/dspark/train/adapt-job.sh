#!/usr/bin/env bash
# DSpark repeat of the T5 best adaptation setting, inside one 1-node allocation
# (submit-train.sh STAGE=8b MODE=adapt). Frozen settings:
# methods/dspark/configs/8b/adapt-mult-n8k-r0.json. Every step writes a done
# marker under ADAPT_ROOT/stages; a resubmission skips finished steps, so the
# whole sequence can run as a chain of short jobs (ADAPT_DEADLINE, below).
#   rebind -> convert -> prepare (+ row check) -> lr-0 evaluations of e10 on the
#   multilingual dev traces and OPB held-out -> 384-step fine-tune -> export ->
#   lr-0 evaluations of the exported head -> development report.
set -euo pipefail
: "${REPO_ROOT:?}" "${ADAPT_ROOT:?}"
PLAN="${ADAPT_PLAN:-${REPO_ROOT}/methods/dspark/configs/8b/adapt-mult-n8k-r0.json}"
export DSPARK_DIR="${ADAPT_ROOT}"
export SIDE_DEPS="${ADAPT_ROOT}/runtime/pydeps"
export SPECULATORS="${ADAPT_ROOT}/runtime/speculators"
. "${REPO_ROOT}/methods/dspark/train/common.sh"
check_side_deps
export HEAD_NODE="$(hostname)"
TOOL=(python3 "${REPO_ROOT}/methods/dspark/train/adapt_dspark.py")
STAGES="${ADAPT_ROOT}/stages"
mkdir -p "${STAGES}" "${ADAPT_ROOT}/checks"
plan() { python3 -c "import json,sys; print(json.load(open('${PLAN}'))$1)"; }
SEQ_LEN="$(plan "['training']['total_seq_len']")"
SEED="$(plan "['training']['seed']")"

# Chunked allocations (debug, 1.5 h): ADAPT_DEADLINE (unix s, set by
# submit-train.sh MODE=adapt) stops the job before a step whose conservative
# duration (minutes, below) no longer fits, so no step is cut mid-way. Outputs
# of a step that was cut anyway (node failure, time limit) are kept under
# ADAPT_ROOT/incomplete/ and the step reruns from scratch: every step is
# deterministic (seeded) and at most ~40 min.
declare -A MINUTES=([rebind]=10 [convert]=10 [prepare-train]=20 [prepare-opb]=15
  [eval-e10-dev]=25 [eval-e10-opb]=25 [train]=55 [export]=15 [eval-final-dev]=25
  [eval-final-opb]=25 [report]=5)
declare -A OUTPUTS=([rebind]=init/e10-rebound [convert]=data/rows [prepare-train]=data/prepared-train
  [prepare-opb]=data/prepared-opb [eval-e10-dev]=runs/eval-e10-dev [eval-e10-opb]=runs/eval-e10-opb
  [train]=runs/train [export]=head [eval-final-dev]=runs/eval-final-dev
  [eval-final-opb]=runs/eval-final-opb [report]=development-report.json)

step() {  # step NAME CMD...: run once, if it fits before ADAPT_DEADLINE
  local name=$1 out; shift
  if [ -f "${STAGES}/${name}.done" ]; then stamp "skip ${name} (done)"; return 0; fi
  if [ -n "${ADAPT_DEADLINE:-}" ] && (( $(date +%s) + ${MINUTES[${name}]} * 60 > ADAPT_DEADLINE )); then
    stamp "ADAPT_PAUSED before ${name}: needs ${MINUTES[${name}]} min, $(( (ADAPT_DEADLINE - $(date +%s)) / 60 )) left"
    exit 0
  fi
  for out in ${OUTPUTS[${name}]}; do
    if [ -e "${ADAPT_ROOT}/${out}" ]; then
      mkdir -p "${ADAPT_ROOT}/incomplete"
      mv "${ADAPT_ROOT}/${out}" "${ADAPT_ROOT}/incomplete/$(basename "${out}")-$(date -u +%Y%m%dT%H%M%SZ)"
      stamp "set aside unfinished ${out} of ${name}"
    fi
  done
  stamp "begin ${name}"
  "$@"
  date -u +%FT%TZ > "${STAGES}/${name}.done"
  stamp "done ${name}"
}

prepare() {  # prepare IN OUT RATIO TRAIN_ROWS
  # Not `speculators prepare-data`: it shuffles rows, and the trainer holds
  # out the trailing rows by position (job 3861814's row check caught it).
  "${TOOL[@]}" build-dataset --input "$1" --output "$2"
  "${TOOL[@]}" check-prepared --data "$2" --expected "$1" --ratio "$3" \
    --expected-train-rows "$4" --output "$2/row-check.json"
}

trainer() {  # trainer RUN_ID DATA FROM LR EPOCHS MAX_STEPS RATIO [EXTRA...]
  local run=$1 data=$2 from=$3 lr=$4 epochs=$5 steps=$6 ratio=$7; shift 7
  RUN_ID="${run}" DATA="${data}" FROM_PRETRAINED="${from}" LR="${lr}" EPOCHS="${epochs}" \
    MAX_STEPS="${steps}" SEQ="${SEQ_LEN}" CKPT_FREQ=1 \
    TRAIN_EXTRA="--seed ${SEED} --train-data-ratio ${ratio} $*" \
    bash "${REPO_ROOT}/methods/dspark/train/train-node.sh"
}

evaluate() {  # evaluate NAME FROM DATA RATIO: one step at lr 0, then validation
  trainer "$1" "$3" "$2" 0 1 1 "$4"
  "${TOOL[@]}" compare --left "$2" --right "${ADAPT_ROOT}/runs/$1/checkpoints/0" \
    --output "${ADAPT_ROOT}/checks/$1-unchanged.json" --require-equal
}

INIT="${ADAPT_ROOT}/init/e10-rebound"
ROWS="${ADAPT_ROOT}/data/rows"
TRAIN_DATA="${ADAPT_ROOT}/data/prepared-train"
OPB_DATA="${ADAPT_ROOT}/data/prepared-opb"
HEAD="${ADAPT_ROOT}/head"
TRAIN_RATIO=0.969
OPB_RATIO=0.3334

step rebind "${TOOL[@]}" rebind --plan "${PLAN}" --contract "${REPO_ROOT}/targets/8b/contract.json" --output "${INIT}"
step convert "${TOOL[@]}" convert --plan "${PLAN}" --output "${ROWS}"
step prepare-train prepare "${ROWS}/train-dev.jsonl" "${TRAIN_DATA}" "${TRAIN_RATIO}" 8000
step prepare-opb prepare "${ROWS}/opb-eval.jsonl" "${OPB_DATA}" "${OPB_RATIO}" 256
step eval-e10-dev evaluate eval-e10-dev "${INIT}" "${TRAIN_DATA}" "${TRAIN_RATIO}"
step eval-e10-opb evaluate eval-e10-opb "${INIT}" "${OPB_DATA}" "${OPB_RATIO}"
step train trainer train "${TRAIN_DATA}" "${INIT}" "$(plan "['training']['learning_rate']")" \
  "$(plan "['training']['epochs']")" "$(plan "['training']['max_steps']")" "${TRAIN_RATIO}" \
  --scheduler-type cosine --scheduler-warmup-steps 19 --scheduler-num-cosine-cycles 0
step export "${TOOL[@]}" export --checkpoint "${ADAPT_ROOT}/runs/train/checkpoints/1" \
  --initializer "${INIT}" --output "${HEAD}"
step eval-final-dev evaluate eval-final-dev "${HEAD}" "${TRAIN_DATA}" "${TRAIN_RATIO}"
step eval-final-opb evaluate eval-final-opb "${HEAD}" "${OPB_DATA}" "${OPB_RATIO}"
step report "${TOOL[@]}" report --plan "${PLAN}" --root "${ADAPT_ROOT}" \
  --output "${ADAPT_ROOT}/development-report.json"
stamp "ADAPT_OK"
