# Sourced inside the container by the DSpark 70B jobs (port of Yu's
# methods/dspark/jobs on branch DSpark_Apertus_1.5_8B). Expects REPO_ROOT.
# speculators (7046ac6 + Yu's patch) lives untracked in the checkout's scratch/;
# it is prepended to PYTHONPATH, never replacing it: the image's /workspace/vllm
# entry carries the vLLM overlays.

DSPARK_DIR="${DSPARK_DIR:-/iopsstor/scratch/cscs/${USER}/apertus-dspark70b}"
TARGET_MODEL="${TARGET_MODEL:-/capstor/store/cscs/swissai/infra01/hf_models/models/swiss-ai/Apertus-v1.5-70B}"
SPECULATORS="${SPECULATORS:-${REPO_ROOT}/scratch/speculators}"
SIDE_DEPS="${DSPARK_DIR}/pydeps"
OPB_REPO="mlabonne/open-perfectblend"
OPB_REVISION="af60f3c18201652a83a93f46fcfee1b646ba3df7"
# Yu's 8B corpus: its row order and val_ids.txt define the held-out split.
CORPUS_8B_DIR="/capstor/store/cscs/swissai/infra01/speculative-decoding/dspark-opb-thinking-off"

export HF_HOME="${HF_HOME:-/iopsstor/scratch/cscs/${USER}/hf_home}"
export PYTHONPATH="${SPECULATORS}/src:${SPECULATORS}/hs_connectors/src:${SIDE_DEPS}${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1

stamp() { echo "=== [$(date -u +%FT%TZ)] $(hostname) $*"; }

# Install, without dependencies, whatever `import speculators` reports missing
# (Yu's side_deps.sh). Only the prep job calls it: parallel installs into one
# directory race.
install_side_deps() {
  declare -A pkg=([dateutil]=python-dateutil [yaml]=pyyaml [PIL]=pillow [sklearn]=scikit-learn [attr]=attrs
    [google]=protobuf [aiohttp]=aiohttp [typer]=typer [click]=click [rich]=rich [loguru]=loguru
    [pydantic_settings]=pydantic-settings [datasets]="datasets>=4.0.0,<=5.0.1" [dotenv]=python-dotenv
    [markdown_it]=markdown-it-py [multiprocess]=multiprocess [xxhash]=xxhash [dill]=dill
    [shellingham]=shellingham [annotated_types]=annotated-types [pydantic_core]=pydantic-core)
  mkdir -p "${SIDE_DEPS}"
  # speculators reads its own version from package metadata, which a PYTHONPATH
  # checkout lacks; record the version Yu's checkpoints carry.
  local meta="${SIDE_DEPS}/speculators-0.9.0.dev36.dist-info"
  mkdir -p "${meta}"
  printf 'Metadata-Version: 2.1\nName: speculators\nVersion: 0.9.0.dev36\n' > "${meta}/METADATA"
  local round err name
  for round in $(seq 1 25); do
    err="$(python3 -c "import speculators, hs_connectors, speculators.train.data, speculators.cli" 2>&1 \
      | grep -oE "No module named '[A-Za-z0-9_.]+'" | head -1 | sed "s/No module named '//; s/'//" | cut -d. -f1 || true)"
    if [ -z "${err}" ]; then echo "side deps complete after $((round - 1)) installs"; break; fi
    name="${pkg[${err}]:-${err}}"
    echo "missing ${err} -> installing ${name}"
    pip install -q --no-deps --target "${SIDE_DEPS}" "${name}" 2>&1 | tail -1
  done
  python3 -c "import transformers, torch, vllm; print('image stack intact: transformers', transformers.__version__, 'torch', torch.__version__, 'vllm', vllm.__version__)"
}

check_side_deps() {
  python3 -c "import speculators, hs_connectors, speculators.train.data, speculators.cli" \
    || { echo "speculators does not import: run the prep job first" >&2; return 1; }
}

# wait_health PORT PID SECONDS: the 70B takes ~10 min to load from capstor.
wait_health() {
  local i
  for i in $(seq 1 $(( $3 / 10 ))); do
    curl -sf "localhost:$1/health" > /dev/null && return 0
    kill -0 "$2" 2> /dev/null || return 1
    sleep 10
  done
  return 1
}
