#!/usr/bin/env bash
# Domain drafters (A3): per-domain stand-in served traces for the 8B target.
# Runs inside the serving image on an allocated node (submit with
# methods/eagle/launch/domain-traces-submit.sh). Two modes:
#
#   DOMAIN_MODE=pools     build prompt pools for all six domains (no GPU):
#                         chat/summarization/multilingual with the T5 builder,
#                         math/code/tool with build_domain_pools; stage
#                         prompts/<domain>/{train,dev}.jsonl and a pilot slice.
#   DOMAIN_MODE=generate  DOMAIN_LIST="math code tool": one domain per GPU,
#                         pilot first (READY-pilot once all six exist), then
#                         the full split (READY-<domain>), MANIFEST.json and a
#                         capstor backup after each domain.
#
# Everything under DATA_ROOT holds source/served text: it stays on the
# cluster. Commit only manifests (counts, digests, revisions, job IDs).
set -euo pipefail

[ -n "${SLURM_JOB_ID:-}" ] || { echo "run inside a Slurm allocation" >&2; exit 2; }
REPO_ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
cd "${REPO_ROOT}"
DATA_ROOT="${DOMAIN_DATA_ROOT:-/iopsstor/scratch/cscs/${USER}/spec-exp-data/domain-drafters}"
BACKUP_ROOT="${DOMAIN_BACKUP_ROOT:-/capstor/scratch/cscs/${USER}/domain-drafters-backup}"
SRC="${DATA_ROOT}/src"
POOLS="${DATA_ROOT}/pools"
PROMPTS="${DATA_ROOT}/prompts"
TRACES="${DATA_ROOT}/traces"
WORKLOADS="${DOMAIN_WORKLOADS:-/users/${USER}/apertus-spec-decoding/workloads/8b}"
PYDEPS="${DOMAIN_PYDEPS:-/users/${USER}/apertus-spec-decoding/scratch/pydeps-data}"
SPLIT_INDEX="${DOMAIN_SPLIT_INDEX:-/users/${USER}/apertus-spec-decoding/results/70b/eagle/data/split-index.jsonl}"
CONTRACT="${REPO_ROOT}/targets/8b/contract.json"
TRAIN_N="${DOMAIN_TRAIN_N:-8000}"
DEV_N="${DOMAIN_DEV_N:-256}"
PILOT_TRAIN_N="${DOMAIN_PILOT_TRAIN_N:-512}"
PILOT_DEV_N="${DOMAIN_PILOT_DEV_N:-64}"
ALL_DOMAINS="math code chat summarization multilingual tool"
# Apertus recommended sampling, as the T5 traces and the open-perfectblend regeneration.
SAMPLING=(--temperature 0.7 --top-p 0.8 --top-k 20 --seed 1)
export HF_HOME="${DATA_ROOT}/cache/hf" XDG_CACHE_HOME="${DATA_ROOT}/cache/xdg"
export VLLM_CACHE_ROOT="${DATA_ROOT}/cache/vllm" TRITON_CACHE_DIR="${DATA_ROOT}/cache/triton"
export TORCHINDUCTOR_CACHE_DIR="${DATA_ROOT}/cache/inductor" HF_HUB_OFFLINE=1
mkdir -p "${POOLS}" "${PROMPTS}" "${TRACES}" "${HF_HOME}" "${XDG_CACHE_HOME}" "${DATA_ROOT}/deployments"

log() { printf '%s %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*"; }
log "job=${SLURM_JOB_ID} host=$(hostname) mode=${DOMAIN_MODE} revision=$(git -C "${REPO_ROOT}" rev-parse HEAD 2>/dev/null || cat "${REPO_ROOT}/REVISION")"

# Per-deployment provenance: source revision, launcher hashes, environment.
DEPLOY="${DATA_ROOT}/deployments/${SLURM_JOB_ID}"
mkdir -p "${DEPLOY}"
python3 - "${REPO_ROOT}" "${DEPLOY}" <<'PY'
import hashlib, json, os, platform, sys
from pathlib import Path
repo, out = map(Path, sys.argv[1:])
files = ["methods/eagle/launch/domain-traces.sh", "methods/eagle/apertus_eagle/build_domain_pools.py",
         "methods/eagle/apertus_eagle/build_adaptation_pools.py", "methods/eagle/apertus_eagle/generate_targets.py",
         "targets/8b/contract.json"]
keys = ["SLURM_JOB_ID", "SLURM_JOB_NODELIST", "SLURMD_NODENAME", "CUDA_VISIBLE_DEVICES", "PYTHONPATH",
        "DOMAIN_MODE", "DOMAIN_LIST", "DOMAIN_TRAIN_N", "DOMAIN_DEV_N", "SBATCH_ENVIRONMENT_FILE"]
revision = (repo / "REVISION").read_text().strip() if (repo / "REVISION").exists() else None
(out / "runtime.json").write_text(json.dumps({
    "python": sys.version, "machine": platform.machine(), "source_revision": revision,
    "task_branch": "experiment/domain-drafters-data",
    "file_sha256": {f: hashlib.sha256((repo / f).read_bytes()).hexdigest() for f in files},
    "environment": {k: os.environ[k] for k in keys if k in os.environ},
}, indent=2) + "\n")
PY

stage_prompts() {
  # prompts/<domain>/{train,dev}.jsonl and prompts-pilot/<domain>/... (nested prefixes:
  # pools are written in hash order, so the first N rows are the N-row subset).
  local domain="$1" pool_dir="$2"
  mkdir -p "${PROMPTS}/${domain}" "${DATA_ROOT}/prompts-pilot/${domain}"
  head -n "${TRAIN_N}" "${pool_dir}/pool.jsonl" > "${PROMPTS}/${domain}/train.jsonl"
  head -n "${DEV_N}" "${pool_dir}/dev.jsonl" > "${PROMPTS}/${domain}/dev.jsonl"
  head -n "${PILOT_TRAIN_N}" "${PROMPTS}/${domain}/train.jsonl" > "${DATA_ROOT}/prompts-pilot/${domain}/train.jsonl"
  head -n "${PILOT_DEV_N}" "${PROMPTS}/${domain}/dev.jsonl" > "${DATA_ROOT}/prompts-pilot/${domain}/dev.jsonl"
  [ "$(wc -l < "${PROMPTS}/${domain}/train.jsonl")" -eq "${TRAIN_N}" ] || { echo "${domain}: short train" >&2; return 1; }
  [ "$(wc -l < "${PROMPTS}/${domain}/dev.jsonl")" -eq "${DEV_N}" ] || { echo "${domain}: short dev" >&2; return 1; }
  touch "${PROMPTS}/${domain}/STAGED"
}

if [ "${DOMAIN_MODE}" = pools ]; then
  PROTECT=()
  for name in eagle-8b-test eagle-8b-validation eagle-8b-profile arena-hard opb-heldout opb-calib \
      probe-speculator-benchmarks probe-speculator-benchmarks-thinking; do
    PROTECT+=("${WORKLOADS}/${name}.jsonl")
  done
  # Container digest, off the generation critical path (generation jobs only need the path).
  ( image="$(python3 -c "import tomllib,sys;print(tomllib.load(open(sys.argv[1],'rb'))['image'])" "${SBATCH_ENVIRONMENT_FILE}")"
    sha256sum "${image}" > "${DATA_ROOT}/image.sha256" ) &
  hash_pid=$!
  export PYTHONPATH="${PYDEPS}:${PYTHONPATH:-}"
  if [ ! -f "${POOLS}/manifest-domain-pools.json" ]; then
    python3 -m apertus_eagle.build_domain_pools --contract "${CONTRACT}" --output-dir "${POOLS}" \
      --source-root "${SRC}" --pool "${TRAIN_N}" --dev "${DEV_N}" --protect "${PROTECT[@]}" \
      --domains math,code,tool
  fi
  for domain in math code tool; do stage_prompts "${domain}" "${POOLS}/${domain}"; done
  log "staged math code tool"
  if [ ! -f "${POOLS}/manifest.json" ]; then
    python3 -m apertus_eagle.build_adaptation_pools --contract "${CONTRACT}" --output-dir "${POOLS}" \
      --infomaniak-dir "${SRC}/Infomaniak-AI__speculators-multilingual-en-fr-de-it-es" \
      --split-index "${SPLIT_INDEX}" --pool "${TRAIN_N}" --dev "${DEV_N}" \
      --protect "${PROTECT[@]}" --domains chat,summarization,multilingual
  fi
  for domain in chat summarization multilingual; do stage_prompts "${domain}" "${POOLS}/${domain}"; done
  log "staged chat summarization multilingual"
  wait "${hash_pid}" || log "image hash failed"
  log "pools done"
  exit 0
fi

[ "${DOMAIN_MODE}" = generate ] || { echo "DOMAIN_MODE must be pools or generate" >&2; exit 2; }

finalize() {
  # MANIFEST.json over every finished split, READY markers, capstor backup.
  local finished="$1"
  (
    flock 9
    python3 - "${TRACES}" "${DATA_ROOT}" "${finished}" "${ALL_DOMAINS}" <<'PY'
import hashlib, json, sys
from pathlib import Path
traces, data, finished, domains = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3], sys.argv[4].split()
def sha(p):
    h = hashlib.sha256()
    with p.open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()
pool_manifests = {}
for name in ("manifest.json", "manifest-domain-pools.json"):
    path = data / "pools" / name
    if path.exists():
        pool_manifests[name] = {"sha256": sha(path), "domains": json.loads(path.read_text())["domains"]}
manifest = {"schema": "domain-drafters-traces-v1", "data_root": str(data), "pool_manifests": pool_manifests,
            "image_sha256": (data / "image.sha256").read_text().split()[0] if (data / "image.sha256").exists() else None,
            "splits": {}}
for summary_path in sorted(traces.glob("**/generation-summary.json")):
    summary = json.loads(summary_path.read_text())
    key = str(summary_path.parent.relative_to(traces))
    manifest["splits"][key] = {
        "generation_settings": {k: summary.get(k) for k in summary if k not in ("splits", "render_parity")},
        "render_parity": summary.get("render_parity"),
        "files": {split: {"rows": s["rows_out"], "sha256": s["output_sha256"], "input_sha256": s["input_sha256"],
                          "finish_reasons": s["finish_reasons"], "generated_tokens": s["generated_tokens"],
                          "generate_seconds": s["generate_seconds"]}
                  for split, s in summary["splits"].items()},
        "job_id": (summary_path.parent / "JOB_ID").read_text().strip() if (summary_path.parent / "JOB_ID").exists() else None,
    }
(traces / "MANIFEST.json.tmp").write_text(json.dumps(manifest, indent=2) + "\n")
(traces / "MANIFEST.json.tmp").replace(traces / "MANIFEST.json")
if finished == "pilot" and all((traces / "pilot" / d / "generation-summary.json").exists() for d in domains):
    (traces / "pilot" / "READY-pilot").write_text("all six pilot domains generated\n")
    (traces / "READY-pilot").write_text("all six pilot domains generated\n")
elif finished != "pilot":
    (traces / f"READY-{finished}").write_text(f"{finished} generated\n")
PY
  ) 9>"${TRACES}/.manifest.lock"
  mkdir -p "${BACKUP_ROOT}"
  if [ "${finished}" = pilot ]; then
    cp -a "${TRACES}/pilot" "${TRACES}/MANIFEST.json" "${BACKUP_ROOT}/" || log "backup failed (pilot)"
  else
    cp -a "${TRACES}/${finished}" "${TRACES}/MANIFEST.json" "${BACKUP_ROOT}/" || log "backup failed (${finished})"
    [ -f "${TRACES}/READY-${finished}" ] && cp "${TRACES}/READY-${finished}" "${BACKUP_ROOT}/"
  fi
  return 0
}

generate() {
  local input_dir="$1" out="$2" gpu="$3"
  mkdir -p "${out}"
  echo "${SLURM_JOB_ID}" > "${out}/JOB_ID"
  CUDA_VISIBLE_DEVICES="${gpu}" python3 -m apertus_bench.runtime_caches \
    --root "${DATA_ROOT}/cache/job-${SLURM_JOB_ID}/$(basename "${out}")-${gpu}" \
    --report "${out}/runtime-cache-preflight.json" --probe-flashinfer \
    --exec python3 -m apertus_eagle.generate_targets \
    --contract "${CONTRACT}" --input "${input_dir}/dev.jsonl" "${input_dir}/train.jsonl" \
    --output-dir "${out}" --max-seq-length 8192 --max-new-tokens 2048 \
    --max-prompt-tokens 4096 --gpu-memory-utilization 0.9 "${SAMPLING[@]}" > "${out}/generate.log" 2>&1
}

run_domain() {
  local domain="$1" gpu="$2"
  [ -f "${PROMPTS}/${domain}/STAGED" ] || { log "${domain}: prompts not staged"; return 1; }
  mkdir -p "${TRACES}/pilot"
  if [ ! -f "${TRACES}/pilot/${domain}/generation-summary.json" ]; then
    generate "${DATA_ROOT}/prompts-pilot/${domain}" "${TRACES}/pilot/${domain}" "${gpu}"
  fi
  finalize pilot
  log "${domain}: pilot done"
  if [ ! -f "${TRACES}/${domain}/generation-summary.json" ]; then
    generate "${PROMPTS}/${domain}" "${TRACES}/${domain}" "${gpu}"
  fi
  finalize "${domain}"
  log "${domain}: full done"
}

gpu=0
pids=()
for domain in ${DOMAIN_LIST}; do
  run_domain "${domain}" "${gpu}" &
  pids+=("$!")
  log "started ${domain} on GPU ${gpu}"
  gpu=$((gpu + 1))
done
status=0
for pid in "${pids[@]}"; do wait "${pid}" || status=1; done
log "generate finished status=${status}"
exit "${status}"
