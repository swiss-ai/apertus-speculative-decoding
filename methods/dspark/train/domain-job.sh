#!/usr/bin/env bash
# Per-domain DSpark drafters (canvas A3), inside one 1-node allocation
# (submit-train.sh STAGE=8b MODE=domain). Plan: DOMAIN_PLAN
# (methods/dspark/configs/8b/domain-drafters.json). HEADS lists what this job
# does, in order:
#   baseline  rebind e10, lock the traces, build every evaluation set, side deps,
#             lr-0 evaluations of e10 on every set
#   <domain>  build its training file, fine-tune, export, back up, evaluate on
#   / mixed   every set
# Steps write done markers (resubmission skips them) and stop before a step
# that no longer fits DOMAIN_DEADLINE, as adapt-job.sh does.
set -euo pipefail
: "${REPO_ROOT:?}" "${DOMAIN_ROOT:?}" "${DOMAIN_PLAN:?}" "${HEADS:?}"
ROOT="${DOMAIN_ROOT}"
export DSPARK_DIR="${ROOT}"
export SIDE_DEPS="${DOMAIN_SIDE_DEPS:-${ROOT}/runtime/pydeps}"
export SPECULATORS="${SPECULATORS:-${REPO_ROOT}/scratch/speculators}"
. "${REPO_ROOT}/methods/dspark/train/common.sh"
export HEAD_NODE="$(hostname)"
ADAPT=(python3 "${REPO_ROOT}/methods/dspark/train/adapt_dspark.py")
TOOL=(python3 "${REPO_ROOT}/methods/dspark/train/domain_drafters.py")
plan() { python3 -c "import json; p=json.load(open('${DOMAIN_PLAN}')); print($1)"; }
SEQ_LEN="$(plan "p['training']['total_seq_len']")"
SEED="$(plan "p['training']['seed']")"
MAX_STEPS_PLAN="$(plan "p['training']['max_steps']")"
EPOCHS_PLAN="$(plan "p['training']['epochs']")"
LR_PLAN="$(plan "p['training']['learning_rate']")"
SETS="$(plan "' '.join((p.get('eval_domains') or p['domains']) + ['opb'])")"
INIT="${ROOT}/init/e10-rebound"
EVAL_DATA="${ROOT}/shared/eval-data"
LOCK="${ROOT}/shared/data-lock.json"
BACKUP="${DOMAIN_BACKUP:-/capstor/scratch/cscs/${USER}/domain-drafters-backup/$(basename "${ROOT}")}"
mkdir -p "${ROOT}/checks" "${ROOT}/shared/stages"

step() {  # step STAGES_DIR MINUTES NAME OUTPUT CMD...: run once, if it fits
  local stages=$1 minutes=$2 name=$3 out=$4; shift 4
  mkdir -p "${stages}"
  if [ -f "${stages}/${name}.done" ]; then stamp "skip ${name} (done)"; return 0; fi
  if [ -n "${DOMAIN_DEADLINE:-}" ] && (( $(date +%s) + minutes * 60 > DOMAIN_DEADLINE )); then
    stamp "DOMAIN_PAUSED before ${name}: needs ${minutes} min"; exit 0
  fi
  if [ -n "${out}" ] && [ -e "${out}" ]; then
    mkdir -p "${ROOT}/incomplete"
    mv "${out}" "${ROOT}/incomplete/$(basename "${out}")-$(date -u +%Y%m%dT%H%M%SZ)"
    stamp "set aside unfinished ${out}"
  fi
  stamp "begin ${name}"
  "$@"
  date -u +%FT%TZ > "${stages}/${name}.done"
  stamp "done ${name}"
}

prepare() {  # prepare IN OUT FRONT_ROWS: build without shuffle, check rows and split
  local total ratio
  total="$(grep -c . "$1")"
  ratio="$(python3 -c "print(($3 + 0.5) / ${total})")"
  "${ADAPT[@]}" build-dataset --input "$1" --output "$2"
  "${ADAPT[@]}" check-prepared --data "$2" --expected "$1" --ratio "${ratio}" \
    --expected-train-rows "$3" --output "$2/row-check.json"
}
ratio_of() { python3 -c "import json; print(json.load(open('$1/row-check.json'))['train_data_ratio'])"; }

trainer() {  # trainer RUN_ID DATA FROM LR EPOCHS MAX_STEPS RATIO [EXTRA...]
  local run=$1 data=$2 from=$3 lr=$4 epochs=$5 steps=$6 ratio=$7; shift 7
  RUN_ID="${run}" DATA="${data}" FROM_PRETRAINED="${from}" LR="${lr}" EPOCHS="${epochs}" \
    MAX_STEPS="${steps}" SEQ="${SEQ_LEN}" CKPT_FREQ=1 \
    TRAIN_EXTRA="--seed ${SEED} --train-data-ratio ${ratio} $*" \
    bash "${REPO_ROOT}/methods/dspark/train/train-node.sh"
}

evaluate() {  # evaluate HEAD_NAME FROM SET: one lr-0 step, then validation
  local data="${EVAL_DATA}/prepared-$3"
  trainer "eval-$1-$3" "${data}" "$2" 0 1 1 "$(ratio_of "${data}")"
  "${ADAPT[@]}" compare --left "$2" --right "${ROOT}/runs/eval-$1-$3/checkpoints/0" \
    --output "${ROOT}/checks/eval-$1-$3-unchanged.json" --require-equal
}

front_rows() { python3 -c "import json; print(json.load(open('${EVAL_DATA}/eval-sets.json'))['sets']['$1']['front_rows'])"; }

baseline() {
  local S="${ROOT}/shared/stages"
  if ! check_side_deps 2> /dev/null; then
    if [ -n "${DOMAIN_SIDE_DEPS:-}" ] && [ "${INSTALL_SIDE_DEPS:-0}" != 1 ]; then
      echo "shared side deps ${SIDE_DEPS} incomplete; run the runtime check first" >&2; exit 1
    fi
    install_side_deps
  fi
  check_side_deps
  step "$S" 10 rebind "${INIT}" "${ADAPT[@]}" rebind --plan "${DOMAIN_PLAN}" \
    --contract "${REPO_ROOT}/targets/8b/contract.json" --output "${INIT}"
  step "$S" 10 lock-data "${LOCK}" "${TOOL[@]}" lock-data --plan "${DOMAIN_PLAN}" --output "${LOCK}"
  step "$S" 30 convert-evals "${EVAL_DATA}" "${TOOL[@]}" convert-evals --plan "${DOMAIN_PLAN}" \
    --lock "${LOCK}" --output "${EVAL_DATA}"
  local set
  for set in ${SETS}; do
    step "$S" 10 "prepare-${set}" "${EVAL_DATA}/prepared-${set}" \
      prepare "${EVAL_DATA}/${set}.jsonl" "${EVAL_DATA}/prepared-${set}" "$(front_rows "${set}")"
  done
  for set in ${SETS}; do
    step "$S" 20 "eval-e10-${set}" "${ROOT}/runs/eval-e10-${set}" evaluate e10 "${INIT}" "${set}"
  done
}

head_run() {  # head_run DOMAIN|mixed
  local h=$1 H="${ROOT}/heads/$1" S="${ROOT}/heads/$1/stages" ckpt set
  mkdir -p "${H}"
  check_side_deps
  [ -f "${ROOT}/shared/stages/eval-e10-opb.done" ] || { echo "baseline not finished" >&2; exit 1; }
  step "$S" 15 convert "${H}/data" "${TOOL[@]}" convert-head --plan "${DOMAIN_PLAN}" --lock "${LOCK}" \
    --domain "${h}" --output "${H}/data"
  local front
  front="$(python3 -c "import json; print(json.load(open('${H}/data/conversion.json'))['train_rows'])")"
  step "$S" 20 prepare "${H}/prepared" prepare "${H}/data/train-dev.jsonl" "${H}/prepared" "${front}"
  step "$S" "${TRAIN_MINUTES:-90}" train "${ROOT}/runs/train-${h}" trainer "train-${h}" "${H}/prepared" "${INIT}" "${LR_PLAN}" \
    "${EPOCHS_PLAN}" "${MAX_STEPS_PLAN}" "$(ratio_of "${H}/prepared")" \
    --scheduler-type cosine --scheduler-warmup-steps 19 --scheduler-num-cosine-cycles 0.5
  ckpt="$("${TOOL[@]}" pick-checkpoint --run "${ROOT}/runs/train-${h}" --max-steps "${MAX_STEPS_PLAN}")"
  step "$S" 15 export "${H}/head" "${ADAPT[@]}" export --checkpoint "${ckpt}" --initializer "${INIT}" \
    --output "${H}/head"
  step "$S" 10 backup "" bash -c "mkdir -p '${BACKUP}/${h}' && cp -r --preserve=timestamps '${H}/head/.' '${BACKUP}/${h}/' \
    && (cd '${BACKUP}/${h}' && sha256sum model.safetensors) > '${H}/backup-sha256.txt' \
    && grep -q \"\$(python3 -c \"import json;print(json.load(open('${H}/head/export-report.json'))['files']['model.safetensors']['sha256'])\")\" '${H}/backup-sha256.txt'"
  for set in ${SETS}; do
    step "$S" 20 "eval-${set}" "${ROOT}/runs/eval-${h}-${set}" evaluate "${h}" "${H}/head" "${set}"
  done
}

for h in ${HEADS}; do
  if [ "${h}" = baseline ]; then baseline; else head_run "${h}"; fi
done
python3 "${REPO_ROOT}/methods/dspark/train/domain_drafters.py" matrix --plan "${DOMAIN_PLAN}" --root "${ROOT}" \
  --output "${ROOT}/matrix-$(date -u +%Y%m%dT%H%M%SZ)-${SLURM_JOB_ID:-local}.json" || true
stamp "DOMAIN_OK ${HEADS}"
