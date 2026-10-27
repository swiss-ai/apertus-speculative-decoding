#!/usr/bin/env python3
"""Read-only bounded status snapshots for a Beverin scaling pilot (Python 3.6)."""
import argparse
import json
from pathlib import Path
import re
import subprocess
import time


def tail(path, limit=131072):
    if not path.is_file():
        return ''
    with path.open('rb') as stream:
        stream.seek(max(0, path.stat().st_size - limit))
        return stream.read().decode(errors='replace')


parser = argparse.ArgumentParser()
parser.add_argument('--work', type=Path, required=True)
parser.add_argument('--run-id', required=True)
parser.add_argument('--job-id', required=True)
parser.add_argument('--interval', type=int, default=60)
parser.add_argument('--once', action='store_true')
args = parser.parse_args()
run = args.work / 'runs' / args.run_id
for iteration in range(240):
    queue = subprocess.check_output(['squeue', '-h', '-j', args.job_id,
                                     '-o', '%T|%M|%D'], universal_newlines=True).strip()
    logs = sorted((run/'logs').glob('train-{}*-node0.log'.format(args.job_id)),
                  key=lambda p: p.stat().st_mtime)
    current = logs[-1] if logs else None
    content = tail(current) if current else ''
    steps = re.findall(r'global_step=(\d+)', content)
    timings = [float(x)/1000 for x in re.findall(r'profile/step_ms=([\deE.+-]+)', content)]
    state_file = run/'checkpoints/0/training_state.json'
    state = json.loads(state_file.read_text()) if state_file.is_file() else None
    memory = []
    for log in (args.work/'logs').glob('mem-{}-node*.log'.format(args.job_id)):
        samples = re.findall(r'used_gb=(\d+) avail_gb=(\d+)', tail(log, 4096))
        if samples:
            memory.append((int(samples[-1][0]), int(samples[-1][1])))
    report = {
        'utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
        'job': args.job_id, 'queue': queue,
        'ready_nodes': len(list((run/'logs').glob('ready-{}-node*'.format(args.job_id)))),
        'attempt': current.name if current else None,
        'last_step': int(steps[-1]) if steps else None,
        'recent_step_seconds': timings[-5:],
        'checkpoint': state,
        'max_current_used_gib': max(x[0] for x in memory) if memory else None,
        'min_current_available_gib': min(x[1] for x in memory) if memory else None,
    }
    print(json.dumps(report, sort_keys=True), flush=True)
    if not queue or args.once:
        break
    time.sleep(args.interval)
