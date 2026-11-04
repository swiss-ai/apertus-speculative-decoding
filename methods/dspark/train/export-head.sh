#!/usr/bin/env bash
# Export a finished speculators DSpark checkpoint as a servable drafter directory:
# the weights and config (no optimizer state) plus the run's provenance, with the
# weights' sha256 checked against the run's completion record when it has one.
#   sbatch -A infra01 -p normal -t 00:30:00 -N 1 methods/dspark/train/export-head.sh CHECKPOINT DEST
# CHECKPOINT is checkpoints/<epoch> of a run; its parent holds run.yaml,
# train_command.txt, speculators.patch and, for Beverin runs, single-epoch-complete.json.
set -euo pipefail
SRC="$(cd "${1:?usage: export-head.sh CHECKPOINT DEST}" && pwd)"
DEST="${2:?usage: export-head.sh CHECKPOINT DEST}"
RUN_CKPTS="$(dirname "${SRC}")"
if [ -e "${DEST}" ]; then echo "refusing to overwrite ${DEST}" >&2; exit 2; fi
mkdir -p "${DEST}"

for f in config.json config.py model.safetensors training_state.json val_metrics.json verified-finite.json; do
  if [ -f "${SRC}/${f}" ]; then cp "${SRC}/${f}" "${DEST}/"; fi
done
for f in run.yaml train_command.txt speculators.patch single-epoch-complete.json rollback-ledger.json; do
  if [ -f "${RUN_CKPTS}/${f}" ]; then cp "${RUN_CKPTS}/${f}" "${DEST}/"; fi
done

(cd "${DEST}" && sha256sum model.safetensors config.json) > "${DEST}/checkpoint_sha256.txt"
python3 - "${SRC}" "${DEST}" <<'PY'
import datetime, json, os, sys
src, dest = sys.argv[1:]
sums = dict(line.split()[::-1] for line in open(os.path.join(dest, "checkpoint_sha256.txt")))
report = {
    "utc": datetime.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
    "source": src,
    "dest": dest,
    "sha256": sums,
    "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
}
record = os.path.join(dest, "single-epoch-complete.json")
if os.path.exists(record):
    expected = json.load(open(record))["sha256"]
    report["expected_sha256"] = {k: expected[k] for k in sums}
    report["matches_completion_record"] = all(expected[k] == sums[k] for k in sums)
json.dump(report, open(os.path.join(dest, "export-report.json"), "w"), indent=2)
print(json.dumps(report, indent=2))
if report.get("matches_completion_record") is False:
    sys.exit("exported weights differ from the run's completion record")
PY
chmod -R a-w "${DEST}"
echo "exported ${SRC} -> ${DEST}"
