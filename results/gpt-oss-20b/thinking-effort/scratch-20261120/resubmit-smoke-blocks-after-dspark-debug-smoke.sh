#!/usr/bin/env bash
# Workstation waiter, second version (2026-11-20). Faruk's standing rule: domain jobs
# go first on debug. Blocks 3988416 and 3988706 were cancelled after they took both
# debug slots. This waits for the DSpark agent to move its smoke to debug: the
# ID in $STATE/smoke must differ from the superseded IDs and be a debug job. It then
# waits until that smoke has FINISHED. After that it submits the smoke evaluation
# blocks 0 -> 1 -> 2 as a chain: block-0 afterany the smoke, each later block
# afterany the previous one, so at most one of them uses debug at a time. It retries
# the per-user submit limit, then restarts the identical controller, which reuses
# all three job.json records.
set -uo pipefail
HOST=clariden1
STATE=${HOME}/.local/state/domain-drafters-dspark
LOG=${HOME}/.local/state/effort-scratch18/resubmit-blocks.log
SUPERSEDED="3988408 3988723"
T=/iopsstor/scratch/cscs/faruk_zahiragic/thinking-effort
C=/iopsstor/scratch/cscs/faruk_zahiragic/apertus-effort-scratch
S=$T/scratch18/smoke-20261120-f7898022
mkdir -p "$(dirname "$LOG")"
log() { echo "[$(date -u +%FT%TZ)] $*" >> "$LOG"; }
remote() { timeout 120 ssh -o BatchMode=yes "$HOST" "$@"; }
deadline=$(( $(date +%s) + 12 * 3600 ))
expired() { [ "$(date +%s)" -ge "$deadline" ]; }

log "waiting for a new debug DSpark smoke id in $STATE/smoke (superseded: $SUPERSEDED)"
while true; do
  expired && { log "no debug DSpark smoke within 12 h; blocks left unsubmitted"; exit 1; }
  smoke=$(tr -dc 0-9 < "$STATE/smoke" 2>/dev/null)
  if [ -n "$smoke" ] && ! [[ " $SUPERSEDED " == *" $smoke "* ]]; then
    part=$(remote "sacct -j $smoke -X -n -P -o Partition" | head -1)
    [ "$part" = debug ] && break
    log "smoke $smoke is on '${part}', not debug; waiting"
  fi
  sleep 120
done
log "DSpark debug smoke $smoke; waiting for it to finish"
while true; do
  expired && { log "DSpark smoke $smoke still not finished after 12 h; stopping"; exit 1; }
  st=$(remote "sacct -j $smoke -X -n -P -o State" | head -1 | cut -d' ' -f1)
  case "$st" in PENDING|RUNNING|REQUEUED|SUSPENDED|CONFIGURING|COMPLETING|"") sleep 120 ;; *) break ;; esac
done
log "DSpark smoke $smoke ended: $st"

after=$smoke
for b in 0 1 2; do
  B=$S/submissions/phase2-block-$b
  while true; do
    out=$(remote "python3 - <<'EOF'
import json, subprocess, sys
from datetime import UTC, datetime
from pathlib import Path
b = Path('$B')
if (b / 'job.json').exists():
    print('EXISTS', json.loads((b / 'job.json').read_text())['job_id']); sys.exit(0)
argv = json.loads((b / 'submission.json').read_text())['argv']
argv = argv[:-1] + ['--dependency=afterany:$after', argv[-1]]
r = subprocess.run(argv, capture_output=True, text=True)
with (b / 'submission-attempts.jsonl').open('a') as f:
    f.write(json.dumps({'timestamp': datetime.now(UTC).isoformat(), 'argv': argv,
        'returncode': r.returncode, 'stdout': r.stdout, 'stderr': r.stderr}) + '\n')
if r.returncode:
    print('REJECTED', r.stderr.strip()); sys.exit(0)
job = r.stdout.strip().split(';')[0]
(b / 'job.json').write_text(json.dumps({'directory': str(b), 'campaign': '$S',
    'job_id': job, 'dependency': 'afterany:$after'}, indent=2) + '\n')
p = b / 'replacement.json'
rep = json.loads(p.read_text()) if p.exists() else {}
rep.update({'replacement_job_id': job, 'dependency': 'afterany:$after', 'argv': argv})
p.write_text(json.dumps(rep, indent=2) + '\n')
print('SUBMITTED', job)
EOF" 2>&1)
    log "block-$b: $out"
    case "$out" in
      SUBMITTED*|EXISTS*) after=$(echo "$out" | awk '{print $2}'); break ;;
      *QOSMaxSubmitJobPerUserLimit*|*AssocMaxSubmitJobLimit*|"") sleep 60 ;;
      *) log "unexpected sbatch failure; stopping"; exit 1 ;;
    esac
  done
done

remote "cd $C && PYTHONPATH=$C/src nohup ~/venvs/apertus-bench/bin/python -m apertus_bench.effort_training_submit --output $S --phase1-reference /users/faruk_zahiragic/apertus-spec-decoding/results/gpt-oss-20b/thinking-effort/campaign-20261027-f409b9da --upstream $T/upstream/speculators-e3041ddb --qualification /users/faruk_zahiragic/apertus-spec-decoding/results/gpt-oss-20b/thinking-effort/phase2-prerequisites/prereq-20261028-d --qualification-job 3782492 --config $T/configs/phase2-scratch-smoke-f7898022.yaml --partition debug --time 01:30:00 >> $T/scratch18/smoke-f7898022-controller.log 2>&1 < /dev/null & echo \$!" \
  | { read -r pid; log "smoke controller restarted on $HOST pid $pid"; }
