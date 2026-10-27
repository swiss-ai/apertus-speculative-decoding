#!/bin/bash
set -euo pipefail
repo=/iopsstor/scratch/cscs/faruk_zahiragic/apertus-spec-decoding-highconc
campaign="$repo/results/8b/layout-recovery-20261027/controller-v1"
site=/iopsstor/scratch/cscs/faruk_zahiragic/pydeps/nixl-layout-recovery-20261027-v1/site
cd "$repo"
expected_commit=efad15a10609e615c5c8635be3c243d1ec5b9da7
case "$(git rev-parse HEAD)" in "$expected_commit"*) ;; *) echo 'Unexpected source revision' >&2; exit 1;; esac
mkdir -p "$campaign"
if [ -e "$campaign/launch-receipt.json" ]; then
  echo 'Launch already registered; inspect its state/PID rather than starting another controller' >&2
  exit 1
fi
export PYTHONPATH="/iopsstor/scratch/cscs/faruk_zahiragic/pydeps/layout-controller-20261027-v1/site:$repo/src"
nohup setsid flock -n "$campaign/controller.lock" \
  "$HOME/venvs/sml-apertus/bin/python" -m apertus_bench.layout_recovery controller \
  --root "$campaign" --site "$site" --partition normal --poll 60 \
  < /dev/null > "$campaign/controller.log" 2>&1 &
controller_pid=$!
python3 - "$campaign" "$controller_pid" "$site" <<'RECEIPT'
import json,sys,subprocess,pathlib,time
root=pathlib.Path(sys.argv[1])
record={'backend':'remote nohup setsid flock (independent of local agent/SSH session)',
        'pid':int(sys.argv[2]), 'site':sys.argv[3],
        'source_commit':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        'state_path':str(root/'state.json'),
        'log_path':str(root/'controller.log'),
        'job_id_schema':'top-level job_ids mirrors stages[*].job_id',
        'created_at':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())}
with (root/'launch-receipt.json').open('x') as f: json.dump(record,f,indent=2);f.write('\n')
print(json.dumps(record))
RECEIPT
sleep 2
kill -0 "$controller_pid"
tail -n 4 "$campaign/controller.log"
