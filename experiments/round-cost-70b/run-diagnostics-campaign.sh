#!/usr/bin/env bash
# Sequential diagnostic campaign on Clariden. One serving job at a time.
#
# PHASES is a comma-separated list:
#   config     six-configuration block, two deployment blocks (default)
#   profile    B2, N3m, D3 with torch profiler after unprofiled cells
#   standalone P2: 8B served alone
#   repro      historical smoke arms with the original launchers
set -euo pipefail
# Login-node SSH round-robin must not SIGHUP the sequential driver.
trap '' HUP

LAUNCH_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "${LAUNCH_DIR}/../.." && pwd)"
SERVING_DIR="${REPO_ROOT}/serving"
cd "${REPO_ROOT}"

export PATH="${HOME}/venvs/sml-apertus/bin:${PATH}"
export MODEL_LAUNCH_ROOT="${MODEL_LAUNCH_ROOT:-${HOME}/model-launch-apertus}"
export SML_PARTITION="${SML_PARTITION:-debug}"
export SML_TIME="${SML_TIME:-01:20:00}"
export DATE_TAG="${DATE_TAG:-$(date -u +%Y%m%d)}"
export RUN_ROOT="${RUN_ROOT:-results/diagnostics-${DATE_TAG}}"
export BENCH="${BENCH:-${HOME}/venvs/apertus-bench/bin/apertus-bench}"
PYTHON="${PYTHON:-python3}"
PHASES="${PHASES:-config}"
LEDGER="${RUN_ROOT}/ledger.jsonl"
# Attach to an already-submitted replica instead of launching a duplicate.
# Consumed on the first matching (config, block), then later cells submit normally.
ATTACH_JOB="${ATTACH_JOB:-}"
ATTACH_CONFIG="${ATTACH_CONFIG:-}"
ATTACH_BLOCK="${ATTACH_BLOCK:-1}"
mkdir -p "${RUN_ROOT}"

log() { printf '%s %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*"; }

submit_job() {
  local before after job submit_log
  before="$(mktemp)"
  after="$(mktemp)"
  submit_log="${RUN_ROOT}/sml-submit.log"
  mkdir -p "${RUN_ROOT}"
  squeue -u "${USER}" -h -o '%i' | sort > "${before}" || true
  "$@" --no-tui >>"${submit_log}" 2>&1
  sleep 5
  squeue -u "${USER}" -h -o '%i' | sort > "${after}" || true
  job="$(comm -13 "${before}" "${after}" | head -1 || true)"
  if [ -z "${job}" ]; then
    job="$(grep -aoE 'Job submitted: [0-9]+' "${submit_log}" | tail -1 | awk '{print $NF}' || true)"
  fi
  rm -f "${before}" "${after}"
  if [ -z "${job}" ]; then
    echo "failed to determine new Slurm job id; see ${submit_log}" >&2
    return 1
  fi
  printf '%s\n' "${job}"
}

run_config() {
  local config="$1"
  local block="$2"
  local extra_env="${3:-}"
  local repeat_tag
  repeat_tag="$(printf '%02d' "${block}")"
  if [ "${PROFILE_ONLY:-0}" = "1" ] || [ "${PROFILE_AFTER:-0}" = "1" ]; then
    local profile_missing=0
    local conc
    for conc in ${PROFILE_CONCURRENCIES:-${PROFILE_CONCURRENCY:-1}}; do
      if ! find "${RUN_ROOT}/${config}" -path "*/profile-c${conc}/summary.json" 2>/dev/null | grep -q .; then
        profile_missing=1
      fi
    done
    if [ "${profile_missing}" -eq 0 ]; then
      log "skip ${config} block ${block}: already profiled"
      return 0
    fi
  elif find "${RUN_ROOT}/${config}" -path "*/c8/repeat-${repeat_tag}/summary.json" 2>/dev/null | grep -q .; then
    log "skip ${config} block ${block}: already measured"
    return 0
  fi
  log "launch ${config} block ${block}"
  # shellcheck disable=SC2086
  eval "${extra_env}"
  local job
  if [ -n "${ATTACH_JOB}" ] && [ "${config}" = "${ATTACH_CONFIG}" ] && [ "${block}" = "${ATTACH_BLOCK}" ]; then
    job="${ATTACH_JOB}"
    ATTACH_JOB=""
    log "attaching to existing ${config} job ${job} (block ${block})"
  else
    job="$(submit_job "${LAUNCH_DIR}/diagnostics.sh" "${config}")"
    log "submitted ${config} job ${job}"
  fi
  "${PYTHON}" -c "from pathlib import Path; from apertus_bench.diagnostics import append_ledger; append_ledger(Path('${LEDGER}'), {'event':'submitted','config':'${config}','block':${block},'job_id':'${job}'})"
  local info ip node model
  if ! info="$("${SERVING_DIR}/wait-replica.sh" "${job}")"; then
    scancel "${job}" || true
    return 1
  fi
  eval "${info}"
  ip="${IP}"
  node="${NODE}"
  model="${MODEL}"
  log "ready ${config} job ${job} node ${node} ip ${ip}"
  set +e
  "${LAUNCH_DIR}/drive-diagnostics.sh" "${config}" "${job}" "${node}" "${ip}" "${model}" "${block}"
  local status=$?
  set -e
  scancel "${job}" || true
  "${PYTHON}" -c "from pathlib import Path; from apertus_bench.diagnostics import append_ledger; append_ledger(Path('${LEDGER}'), {'event':'finished','config':'${config}','block':${block},'job_id':'${job}','status':${status}})"
  if [ "${status}" -ne 0 ]; then
    log "measurement failed for ${config} job ${job} (status ${status}); continuing"
  fi
}

analyze_now() {
  if [ -x "${BENCH}" ]; then
    "${BENCH}" analyze "${RUN_ROOT}" --output "${RUN_ROOT}/analysis.csv" || true
    "${BENCH}" break-even "${RUN_ROOT}" --t0-variant B2 --spec-variant D3 \
      --output "${RUN_ROOT}/break-even-D3-vs-B2.json" || true
    "${BENCH}" break-even "${RUN_ROOT}" --t0-variant B0 --spec-variant D3 \
      --output "${RUN_ROOT}/break-even-D3-vs-B0.json" || true
    "${BENCH}" break-even "${RUN_ROOT}" --t0-variant B2 --spec-variant N3m \
      --output "${RUN_ROOT}/break-even-N3m-vs-B2.json" || true
  fi
}

IFS=',' read -r -a PHASE_LIST <<< "${PHASES}"
export PYTHONPATH="${REPO_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"

for phase in "${PHASE_LIST[@]}"; do
  case "${phase}" in
    config)
      log "phase config: 24 screening cells over two deployment blocks"
      for block in 1 2; do
        mapfile -t order < <(
          "${PYTHON}" -c "from apertus_bench.diagnostics import BLOCK_ORDERS; print('\\n'.join(BLOCK_ORDERS[${block}]))"
        )
        for config in "${order[@]}"; do
          run_config "${config}" "${block}"
          analyze_now
        done
      done
      ;;
    profile)
      log "phase profile: B2, N3m, D3 with profiler configured at launch"
      PROFILE_ROOT="${SCRATCH:-/capstor/scratch/cscs/${USER}}/apertus-diagnostics/${DATE_TAG}"
      mkdir -p "${PROFILE_ROOT}"
      export PROFILE_CONCURRENCIES="${PROFILE_CONCURRENCIES:-1 8}"
      for config in B2 N3m D3; do
        export PROFILER_DIR="${PROFILE_ROOT}/${config}"
        export PROFILE_AFTER=1
        export PROFILE_ONLY=1
        export SKIP_CAPTURE=1
        mkdir -p "${PROFILER_DIR}"
        run_config "${config}" 1
        unset PROFILER_DIR PROFILE_AFTER PROFILE_ONLY SKIP_CAPTURE
      done
      unset PROFILE_CONCURRENCIES
      ;;
    standalone)
      log "phase standalone: P2 8B-only bound"
      run_config P2 1
      ;;
    repro)
      log "phase repro: historical smoke arms are not driven here; use serving/baseline.sh ngram.sh draft-model.sh"
      ;;
    *)
      echo "unknown phase ${phase}" >&2
      exit 2
      ;;
  esac
done

analyze_now
log "campaign complete"
