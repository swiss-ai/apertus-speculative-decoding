#!/bin/bash
set -euo pipefail
repo=/iopsstor/scratch/cscs/faruk_zahiragic/apertus-spec-decoding-highconc
campaign="$repo/results/8b/layout-recovery-20261028/controller-v2"
site=/iopsstor/scratch/cscs/faruk_zahiragic/pydeps/nixl-layout-recovery-20261027-v1/site
cd "$repo"
export PYTHONPATH="/iopsstor/scratch/cscs/faruk_zahiragic/pydeps/layout-controller-20261027-v1/site:$repo/src"
if [ -e "$campaign/launch-receipt.json" ]; then
  echo 'Controller-v2 already registered; inspect existing PID/state before any restart' >&2
  exit 1
fi
"$HOME/venvs/sml-apertus/bin/python" - "$campaign" <<'VERIFY'
import json,pathlib,sys,hashlib
from apertus_bench.layout_recovery import source_identity
r=pathlib.Path(sys.argv[1]);expected=json.loads((r/'expected-source-identity-v2.json').read_text())
if source_identity()!=expected:raise RuntimeError('Source changed after overlay verification')
f=json.loads((r/'source-freeze-v2.json').read_text())
if hashlib.sha256((r/'source-overlay-v2.patch').read_bytes()).hexdigest()!=f['source_overlay_sha256']:raise RuntimeError('Overlay changed')
VERIFY
nohup setsid flock -n "$campaign/controller.lock" \
  "$HOME/venvs/sml-apertus/bin/python" -m apertus_bench.layout_recovery controller \
  --root "$campaign" --site "$site" --partition normal --poll 60 \
  < /dev/null > "$campaign/controller.log" 2>&1 &
controller_pid=$!
python3 - "$campaign" "$controller_pid" "$site" <<'RECEIPT'
import json,sys,pathlib,time
root=pathlib.Path(sys.argv[1]);freeze=json.loads((root/'source-freeze-v2.json').read_text())
record={'backend':'remote nohup setsid flock (independent of local session)',
        'pid':int(sys.argv[2]),'site':sys.argv[3],
        'source_base_commit':freeze['source_base_commit'],
        'local_repair_commit':freeze['local_repair_commit'],
        'source_overlay_sha256':freeze['source_overlay_sha256'],
        'source_freeze_path':str(root/'source-freeze-v2.json'),
        'source_identity_path':str(root/'observed-source-identity-v2.json'),
        'state_path':str(root/'state.json'),'log_path':str(root/'controller.log'),
        'job_id_schema':'top-level job_ids mirrors stages[*].job_id',
        'supersedes_state_path':freeze['supersedes_state_path'],
        'created_at':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())}
with (root/'launch-receipt.json').open('x') as f:json.dump(record,f,indent=2);f.write('\n')
print(json.dumps(record))
RECEIPT
sleep 2
kill -0 "$controller_pid"
tail -n 5 "$campaign/controller.log"
