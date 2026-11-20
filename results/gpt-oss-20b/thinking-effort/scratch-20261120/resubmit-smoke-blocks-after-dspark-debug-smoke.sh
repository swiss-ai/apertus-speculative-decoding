#!/usr/bin/env bash
# Workstation waiter (third version, 2026-11-20). Faruk's standing rule is that
# domain jobs go first on debug. This waits ONLY for the DSpark agent's marker
# $STATE/smoke-done, created after its smoke succeeds, and ignores smoke IDs and
# job end states. If the marker is absent for 12 h it logs and exits. It then
# submits the smoke evaluation blocks 0 -> 1 -> 2 as an afterany chain, so at most
# one block uses debug at a time, retrying the per-user submit limit, and restarts
# the identical controller, which reuses all three job.json records.
# Earlier versions and their cancelled jobs (3988022; 3988416 and 3988706;
# 3988916 and 3988925) are recorded in README.md.
set -uo pipefail
HOST=clariden1
STATE=${HOME}/.local/state/domain-drafters-dspark
LOG=${HOME}/.local/state/effort-scratch18/resubmit-blocks.log
T=/iopsstor/scratch/cscs/faruk_zahiragic/thinking-effort
C=/iopsstor/scratch/cscs/faruk_zahiragic/apertus-effort-scratch
S=$T/scratch18/smoke-20261120-f7898022
mkdir -p "$(dirname "$LOG")"
log() { echo "[$(date -u +%FT%TZ)] $*" >> "$LOG"; }
remote() { timeout 120 ssh -o BatchMode=yes "$HOST" "$@"; }
deadline=$(( $(date +%s) + 12 * 3600 ))
expired() { [ "$(date +%s)" -ge "$deadline" ]; }

# Third version (2026-11-20 14:50 CET, coordinator/Faruk): proceed ONLY once the
# DSpark agent creates $STATE/smoke-done after its smoke succeeds. Smoke IDs and
# job end states are ignored entirely (the second version submitted right after
# DSpark smoke 3988743 FAILED, which blocked that smoke's resume).
log "waiting for $STATE/smoke-done"
until [ -e "$STATE/smoke-done" ]; do
  expired && { log "smoke-done absent for 12 h; blocks left unsubmitted"; exit 1; }
  sleep 120
done
log "smoke-done present"
# A stray measurement directory would make the rerun refuse to start.
remote "ls $S/phase2" | grep -q . && { log "unexpected entries in $S/phase2; stopping"; exit 1; }
after=""  # block-0 has no dependency: the smoke already succeeded
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
argv = argv[:-1] + (['--dependency=afterany:$after'] if '$after' else []) + [argv[-1]]
r = subprocess.run(argv, capture_output=True, text=True)
with (b / 'submission-attempts.jsonl').open('a') as f:
    f.write(json.dumps({'timestamp': datetime.now(UTC).isoformat(), 'argv': argv,
        'returncode': r.returncode, 'stdout': r.stdout, 'stderr': r.stderr}) + '\n')
if r.returncode:
    print('REJECTED', r.stderr.strip()); sys.exit(0)
job = r.stdout.strip().split(';')[0]
(b / 'job.json').write_text(json.dumps({'directory': str(b), 'campaign': '$S',
    'job_id': job, 'dependency': 'afterany:$after' if '$after' else None}, indent=2) + '\n')
p = b / 'replacement.json'
rep = json.loads(p.read_text()) if p.exists() else {}
rep.update({'replacement_job_id': job, 'dependency': 'afterany:$after' if '$after' else None, 'argv': argv})
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
