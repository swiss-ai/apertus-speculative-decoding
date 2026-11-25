#!/usr/bin/env bash
# Workstation waiter (2026-11-20). Faruk gave the debug partition to the domain
# experiments first, so smoke evaluation block-0 (3988022) was cancelled with its
# controller stopped. This waits for the DSpark domain-grid watcher to record its
# smoke job ID, then resubmits block-0 with the campaign's exact sbatch argv plus
# --dependency=afterany:<smoke>, retrying the per-user debug submit limit. It writes
# the new job.json (the controller reuses an existing job.json instead of
# resubmitting) and restarts the identical smoke controller, which waits on the new
# block-0 and submits blocks 1 and 2 itself.
set -uo pipefail
HOST=clariden1
STATE=${HOME}/.local/state/domain-drafters-dspark
LOG=${HOME}/.local/state/effort-scratch18/resubmit-block0.log
T=/iopsstor/scratch/cscs/faruk_zahiragic/thinking-effort
C=/iopsstor/scratch/cscs/faruk_zahiragic/apertus-effort-scratch
S=$T/scratch18/smoke-20261120-f7898022
B=$S/submissions/phase2-block-0
mkdir -p "$(dirname "$LOG")"
log() { echo "[$(date -u +%FT%TZ)] $*" >> "$LOG"; }
remote() { timeout 120 ssh -o BatchMode=yes "$HOST" "$@"; }

log "waiting for DSpark smoke id in $STATE/smoke"
deadline=$(( $(date +%s) + 8 * 3600 ))
until [ -s "$STATE/smoke" ]; do
  if [ "$(date +%s)" -ge "$deadline" ]; then
    log "no DSpark smoke after 8 h; block-0 left unsubmitted"; exit 1
  fi
  sleep 120
done
smoke=$(tr -dc 0-9 < "$STATE/smoke")
log "DSpark smoke job $smoke"

while true; do
  out=$(remote "set -e; cd $S; python3 - <<'EOF'
import json, subprocess, sys
from datetime import UTC, datetime
from pathlib import Path
b = Path('$B')
if (b / 'job.json').exists():
    print('EXISTS', json.loads((b / 'job.json').read_text())['job_id']); sys.exit(0)
argv = json.loads((b / 'submission.json').read_text())['argv']
argv = argv[:-1] + ['--dependency=afterany:$smoke', argv[-1]]
r = subprocess.run(argv, capture_output=True, text=True)
with (b / 'submission-attempts.jsonl').open('a') as f:
    f.write(json.dumps({'timestamp': datetime.now(UTC).isoformat(), 'argv': argv,
        'returncode': r.returncode, 'stdout': r.stdout, 'stderr': r.stderr}) + '\n')
if r.returncode:
    print('REJECTED', r.stderr.strip()); sys.exit(0)
job = r.stdout.strip().split(';')[0]
(b / 'job.json').write_text(json.dumps({'directory': str(b), 'campaign': '$S',
    'job_id': job, 'replaces': '3988022', 'dependency': 'afterany:$smoke'}, indent=2) + '\n')
rep = json.loads((b / 'replacement.json').read_text())
rep.update({'replacement_job_id': job, 'dependency': 'afterany:$smoke', 'argv': argv})
(b / 'replacement.json').write_text(json.dumps(rep, indent=2) + '\n')
print('SUBMITTED', job)
EOF" 2>&1)
  log "$out"
  case "$out" in
    SUBMITTED*|EXISTS*) break ;;
    *QOSMaxSubmitJobPerUserLimit*|*AssocMaxSubmitJobLimit*|"") sleep 60 ;;
    *) log "unexpected sbatch failure; stopping"; exit 1 ;;
  esac
done

remote "cd $C && PYTHONPATH=$C/src nohup ~/venvs/apertus-bench/bin/python -m apertus_bench.effort_training_submit --output $S --phase1-reference /users/faruk_zahiragic/apertus-spec-decoding/results/gpt-oss-20b/thinking-effort/campaign-20261027-f409b9da --upstream $T/upstream/speculators-e3041ddb --qualification /users/faruk_zahiragic/apertus-spec-decoding/results/gpt-oss-20b/thinking-effort/phase2-prerequisites/prereq-20261028-d --qualification-job 3782492 --config $T/configs/phase2-scratch-smoke-f7898022.yaml --partition debug --time 01:30:00 >> $T/scratch18/smoke-f7898022-controller.log 2>&1 < /dev/null & echo \$!" \
  | { read -r pid; log "smoke controller restarted on $HOST pid $pid"; }
