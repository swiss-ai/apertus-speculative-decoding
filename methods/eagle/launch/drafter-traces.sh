#!/usr/bin/env bash
# Drafter updates (T5), step A, on one allocated node inside the serving image:
# build the per-domain adaptation pools, then answer every pool and dev prompt
# with plain vLLM at Apertus sampling (thinking off) as stand-in served traces,
# one domain per GPU. Everything written here holds restricted text: it stays
# under DATA_ROOT on the cluster; commit only the manifests' counts and digests.
#   NODE_CMD_PARTITION=normal NODE_CMD_TIME=03:00:00 NODE_CMD_GPUS=4 \
#     methods/eagle/launch/submit-node-command.sh bash methods/eagle/launch/drafter-traces.sh
set -euo pipefail

if [ -z "${SLURM_JOB_ID:-}" ]; then
  echo "run inside a Slurm allocation (submit-node-command.sh)" >&2
  exit 2
fi
REPO_ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
cd "${REPO_ROOT}"
DATA_ROOT="${DRAFTER_DATA_ROOT:-/iopsstor/scratch/cscs/${USER}/spec-exp-data/drafter-updates}"
POOLS="${DATA_ROOT}/pools"
TRACES="${DATA_ROOT}/traces"
DOMAINS="${DRAFTER_DOMAINS:-chat summarization multilingual}"
# Smoke runs: e.g. DRAFTER_POOL_ARGS="--pool 64 --dev 16 --test 16 --summary-files 1"
# DRAFTER_GEN_ARGS="--limit 16" with DRAFTER_DATA_ROOT pointing at a scratch root.
read -r -a POOL_ARGS <<< "${DRAFTER_POOL_ARGS:-}"
read -r -a GEN_ARGS <<< "${DRAFTER_GEN_ARGS:-}"
# Apertus recommended sampling, as the colleague's open-perfectblend regeneration.
SAMPLING=(--temperature 0.7 --top-p 0.8 --top-k 20 --seed 1)
CONTRACT="${REPO_ROOT}/targets/8b/contract.json"
# Caches off the home filesystem (at quota).
export HF_HOME="${DATA_ROOT}/cache/hf" XDG_CACHE_HOME="${DATA_ROOT}/cache/xdg"
export VLLM_CACHE_ROOT="${DATA_ROOT}/cache/vllm" TRITON_CACHE_DIR="${DATA_ROOT}/cache/triton"
export TORCHINDUCTOR_CACHE_DIR="${DATA_ROOT}/cache/inductor"
mkdir -p "${POOLS}" "${TRACES}" "${HF_HOME}" "${XDG_CACHE_HOME}"

log() { printf '%s %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*"; }
log "job=${SLURM_JOB_ID} host=$(hostname) revision=$(git -C "${REPO_ROOT}" rev-parse HEAD) domains=${DOMAINS}"

# Hash the image on the allocated compute node, and keep the actual source and
# environment with each independently launched generation deployment.
python3 - "${REPO_ROOT}" "${DATA_ROOT}" <<'PY'
import hashlib, json, os, platform, shutil, subprocess, sys, tomllib
from pathlib import Path
repo, data = map(Path, sys.argv[1:])
out = data / "deployments" / os.environ["SLURM_JOB_ID"]
out.mkdir(parents=True, exist_ok=False)
source = repo / "methods/eagle/configs/train-env.toml"
image = Path(tomllib.loads(source.read_text())["image"].replace("{arch}", "arm64"))
digest = hashlib.sha256()
with image.open("rb") as handle:
    for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
        digest.update(block)
(out / "image.sha256").write_text(f"{digest.hexdigest()}  {image}\n")
shutil.copy2(source, out / "environment-source.toml")
shutil.copy2(repo / "targets/8b/contract.json", out / "target-contract.json")
for name, argv in {
    "git-revision.txt": ["git", "-C", str(repo), "rev-parse", "HEAD"],
    "checkout.patch": ["git", "-C", str(repo), "diff", "HEAD", "--binary"],
}.items():
    (out / name).write_bytes(subprocess.check_output(argv))
relevant = ["SLURM_JOB_ID", "SLURM_JOB_NODELIST", "SLURMD_NODENAME", "PYTHONPATH",
            "CUDA_VISIBLE_DEVICES", "HF_HOME", "VLLM_CACHE_ROOT", "TRITON_CACHE_DIR",
            "TORCHINDUCTOR_CACHE_DIR", "DRAFTER_DOMAINS", "DRAFTER_POOL_ARGS", "DRAFTER_GEN_ARGS"]
(out / "runtime.json").write_text(json.dumps({
    "python": sys.version, "machine": platform.machine(),
    "environment": {key: os.environ[key] for key in relevant if key in os.environ},
    "script": str(repo / "methods/eagle/launch/drafter-traces.sh"),
    "script_sha256": hashlib.sha256((repo / "methods/eagle/launch/drafter-traces.sh").read_bytes()).hexdigest(),
}, indent=2) + "\n")
PY

# Prompts no pool or dev row may duplicate: the test and validation workloads,
# and the workload-coverage slices (read-only evaluation sets of another branch).
PROTECT=("${REPO_ROOT}/workloads/8b/eagle-8b-test.jsonl" "${REPO_ROOT}/workloads/8b/eagle-8b-validation.jsonl")
COVERAGE="${DRAFTER_COVERAGE:-/iopsstor/scratch/cscs/${USER}/spec-exp/workload-coverage/workloads/coverage/coverage.jsonl}"
[ -f "${COVERAGE}" ] && PROTECT+=("${COVERAGE}")
if [ ! -f "${POOLS}/manifest.json" ]; then
  # pyarrow is not in the serving image; the data-check jobs use the same pydeps.
  PYTHONPATH="${REPO_ROOT}/scratch/pydeps-data:${PYTHONPATH:-}" python3 -m apertus_eagle.build_adaptation_pools --contract "${CONTRACT}" \
    --output-dir "${POOLS}" --infomaniak-dir "${DATA_ROOT}/src/infomaniak" --protect "${PROTECT[@]}" \
    --split-index "${DRAFTER_SPLIT_INDEX:-/users/${USER}/apertus-spec-decoding/results/70b/eagle/data/split-index.jsonl}" \
    --domains "$(echo "${DOMAINS}" | tr ' ' ',')" ${POOL_ARGS[@]+"${POOL_ARGS[@]}"}
else
  log "pools exist: ${POOLS}/manifest.json"
fi

gpu=0
pids=()
for domain in ${DOMAINS}; do
  out="${TRACES}/${domain}"
  if [ -f "${out}/generation-summary.json" ]; then
    log "skip ${domain}: ${out}/generation-summary.json exists"
    continue
  fi
  mkdir -p "${out}"
  CUDA_VISIBLE_DEVICES="${gpu}" python3 -m apertus_eagle.generate_targets \
    --contract "${CONTRACT}" --input "${POOLS}/${domain}/dev.jsonl" "${POOLS}/${domain}/pool.jsonl" \
    --output-dir "${out}" --max-seq-length 8192 --max-new-tokens 2048 \
    --max-prompt-tokens 4096 --gpu-memory-utilization 0.9 "${SAMPLING[@]}" \
    ${GEN_ARGS[@]+"${GEN_ARGS[@]}"} > "${out}/generate.log" 2>&1 &
  pids+=("$!")
  log "started ${domain} on GPU ${gpu} (pid $!)"
  gpu=$((gpu + 1))
done
status=0
for pid in "${pids[@]}"; do wait "${pid}" || status=1; done
for domain in ${DOMAINS}; do tail -n 3 "${TRACES}/${domain}/generate.log" 2>/dev/null || true; done
log "traces finished status=${status}"
exit "${status}"
