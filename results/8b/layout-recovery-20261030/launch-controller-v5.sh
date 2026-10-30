#!/bin/bash
set -euo pipefail
repo=/iopsstor/scratch/cscs/faruk_zahiragic/apertus-spec-decoding-highconc
campaign="$repo/results/8b/layout-recovery-20261030/controller-v5"
prior="$repo/results/8b/layout-recovery-20261028/controller-v2/state.json"
site=/iopsstor/scratch/cscs/faruk_zahiragic/pydeps/nixl-layout-recovery-20261027-v1/site
cd "$repo"
export PYTHONPATH="/iopsstor/scratch/cscs/faruk_zahiragic/pydeps/layout-controller-20261027-v1/site:$repo/src"
if [ -e "$campaign/launch-receipt.json" ]; then
  echo 'Continuation already registered; inspect existing state/PID before restarting' >&2
  exit 1
fi
"$HOME/venvs/sml-apertus/bin/python" - "$campaign" "$prior" <<'VERIFY'
import json,pathlib,sys,hashlib,subprocess
from apertus_bench.layout_recovery import source_identity
root=pathlib.Path(sys.argv[1]);prior=pathlib.Path(sys.argv[2]);freeze=json.loads((root/'continuation-freeze-v5.json').read_text())
if source_identity()!=json.loads((root/'expected-source-identity-v5.json').read_text()):raise RuntimeError('Frozen source changed')
if subprocess.check_output(['git','branch','--show-current'],text=True).strip()!=freeze['source_branch']:raise RuntimeError('Source branch changed')
if hashlib.sha256((root/'continue_remaining.py').read_bytes()).hexdigest()!=freeze['wrapper_sha256']:raise RuntimeError('Wrapper changed')
if hashlib.sha256(prior.read_bytes()).hexdigest()!=freeze['prior_state_sha256']:raise RuntimeError('Stopped prior state changed')
if hashlib.sha256((root/'continue_remaining_v5.py').read_bytes()).hexdigest()!=freeze['bootstrap_sha256']:raise RuntimeError('Bootstrap changed')
p=pathlib.Path(freeze['prior_submission_state_path']);old=json.loads(p.read_text())
if hashlib.sha256(p.read_bytes()).hexdigest()!=freeze['prior_submission_state_sha256'] or old['stages'] or old['job_ids']:raise RuntimeError('Prior submission state changed or contains jobs')
out=pathlib.Path(old['pending_submission']['output'])
if (out/'job.txt').exists():raise RuntimeError('Prior submission has a job receipt; inspect before retry')
p=pathlib.Path(freeze['prior_runtime_attempt_state_path']);attempt=json.loads(p.read_text())
if hashlib.sha256(p.read_bytes()).hexdigest()!=freeze['prior_runtime_attempt_state_sha256']:raise RuntimeError('Prior runtime state changed')
if attempt['state']!='needs_attention' or attempt['job_ids']!=['3802690']:raise RuntimeError('Unexpected prior runtime attempt')
for record in attempt['stages']:
 if record['slurm']['state'] not in {'FAILED','CANCELLED'}:raise RuntimeError('Prior allocation is not terminal')
 if list(pathlib.Path(record['output']).rglob('summary.json')):raise RuntimeError('Prior measurements exist; inspect instead of retrying')
print(json.dumps({'source_files_verified':75,'source_base_commit':freeze['source_base_commit'],'source_overlay_sha256':freeze['source_overlay_sha256'],'allowed_stages':freeze['allowed_stages']}))
VERIFY
export SWEEP_SOURCE_BRANCH=high-concurrency-8b
export SWEEP_CONTROLLER_WRAPPER_SHA=4b831ca38d5c18d242163ab6f21d2abf8a0b3b0e53d029ab1f52f3aa6f8aaa02
export SWEEP_CONTROLLER_BOOTSTRAP_SHA=a73da7bd29be2ffd0c5cb0cf73ef3cc396f867d1eb195ce89e0c72110ca7b10e
export SWEEP_CONTROLLER_FREEZE="$campaign/continuation-freeze-v5.json"
nohup setsid flock -n "$campaign/controller.lock" \
  "$HOME/venvs/sml-apertus/bin/python" "$campaign/continue_remaining_v5.py" \
  --root "$campaign" --prior-state "$prior" \
  --expected-source "$campaign/expected-source-identity-v5.json" --site "$site" \
  < /dev/null > "$campaign/controller.log" 2>&1 &
controller_pid=$!
python3 - "$campaign" "$controller_pid" <<'RECEIPT'
import json,pathlib,sys,time
root=pathlib.Path(sys.argv[1]);f=json.loads((root/'continuation-freeze-v5.json').read_text())
r={'created_at':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),'pid':int(sys.argv[2]),
   'backend':'remote nohup setsid flock','state_path':str(root/'state.json'),
   'log_path':str(root/'controller.log'),'source_branch':f['source_branch'],
   'source_base_commit':f['source_base_commit'],'source_overlay_sha256':f['source_overlay_sha256'],
   'local_wrapper_commit':f['local_wrapper_commit'],'wrapper_sha256':f['wrapper_sha256'],
   'local_bootstrap_commit':f['local_bootstrap_commit'],'bootstrap_sha256':f['bootstrap_sha256'],
   'runtime_artifacts_policy':f['runtime_artifacts_policy'],
   'prior_state_path':f['prior_state_path'],'allowed_stages':f['allowed_stages'],
   'job_id_schema':'top-level job_ids mirrors stages[*].job_id','scientific_state':'pending_review_with_preserved_rejections'}
with (root/'launch-receipt.json').open('x') as stream:json.dump(r,stream,indent=2);stream.write('\n')
print(json.dumps(r))
RECEIPT
sleep 2
kill -0 "$controller_pid"
tail -n 6 "$campaign/controller.log"
