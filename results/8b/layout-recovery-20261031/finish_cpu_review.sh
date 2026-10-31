#!/usr/bin/env bash
# CPU-only continuation. Never launches jobs/models or changes frozen engine code.
set -euo pipefail
cd /home/projects/apertus-speculative-decoding-highconc
exec 9>results/8b/layout-recovery-20261031/final-cpu-review.lock
flock -n 9 || exit 75
collector_pid=${1:-2888011}
while kill -0 "$collector_pid" 2>/dev/null; do sleep 10; done
py=/home/projects/apertus-speculative-decoding/.venv/bin/python
"$py" - <<'PY'
import json
from pathlib import Path
p=Path('results/8b/layout-recovery-20261031/controller-v5-collection.json')
r=json.loads(p.read_text())
assert r['returncode']==0, r
assert r['original_bytes']==2131469278, r['original_bytes']
assert len({f['source_path'] for f in r['files']})==len(r['files'])
assert sum(f['source_path'].endswith('/summary.json') for f in r['files'])==267
print('Complete lossless collection manifest:',len(r['files']),'files',flush=True)
PY
"$py" results/8b/layout-recovery-20261030/prepare_final_review.py \
 --controller-state results/8b/layout-recovery-20261030/controller-v5/state.json \
 --path-map /iopsstor/scratch/cscs/faruk_zahiragic/apertus-spec-decoding-highconc=/home/projects/apertus-speculative-decoding-highconc \
 --output results/8b/layout-recovery-20261031/remaining-review-inputs-v1
"$py" results/8b/layout-recovery-20261027/review-helper-v1/review.py \
 --analysis results/8b/layout-recovery-20261030/completed-v2-review-v1/review.json \
 --analysis results/8b/layout-recovery-20261031/remaining-review-inputs-v1/remaining-analysis.json \
 --path-map /iopsstor/scratch/cscs/faruk_zahiragic/apertus-spec-decoding-highconc=/home/projects/apertus-speculative-decoding-highconc \
 --verify-requests \
 --output results/8b/layout-recovery-20261031/final-review-v1
"$py" - <<'PY'
import datetime,hashlib,json
from pathlib import Path
r=Path('results/8b/layout-recovery-20261031')
p=r/'final-review-v1/review.json'
x=json.loads(p.read_text())
assert len(x['rows'])==1040,len(x['rows'])
assert not x['missing_stage_analyses'],x['missing_stage_analyses']
assert len({a['summary'] for a in x['rows']})==1040
out={'state':'cpu_review_finished_pending_qualified_human_verdict','finished_at':datetime.datetime.now(datetime.UTC).isoformat(),'review_sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'cells':len(x['rows']),'comparisons':len(x['comparisons']),'eligible_pairs':sum(a['screening_pair_eligible'] for a in x['comparisons']),'implementation_promotion':'unestablished','publication':'pending_qualified_review'}
(r/'cpu-review-completion.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps(out),flush=True)
PY
