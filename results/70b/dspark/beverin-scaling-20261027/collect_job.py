#!/usr/bin/env python3
"""Collect a finished Beverin pilot's small reproducibility artifacts (Python 3.6).

Leaves the run and all completed raw files untouched. Weights/optimizer state
remain at their recorded cluster paths. The tar archive contains byte hashes.
"""
import argparse
import hashlib
import io
import json
from pathlib import Path
import subprocess
import tarfile
import time


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        while True:
            data = stream.read(1024 * 1024)
            if not data:
                break
            digest.update(data)
    return digest.hexdigest()


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--work', type=Path, required=True)
parser.add_argument('--run-id', required=True)
parser.add_argument('--job-id', required=True)
parser.add_argument('--out', type=Path, required=True)
args = parser.parse_args()
if args.out.exists():
    raise SystemExit('Refusing to overwrite an earlier archive: ' + str(args.out))
query = subprocess.Popen(['squeue', '-h', '-j', args.job_id, '-o', '%T'],
                         stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                         universal_newlines=True)
stdout, stderr = query.communicate()
if query.returncode and 'Invalid job id specified' not in stderr:
    raise SystemExit(stderr)
queue = stdout.strip()
if queue:
    raise SystemExit('Job is still queued/running: ' + queue)
run = args.work/'runs'/args.run_id
files = {}
for subdir, archive_root in [(run/'deployments'/args.job_id, 'deployment'),
                              (run/'logs', 'logs')]:
    for path in subdir.rglob('*'):
        if path.is_file() and not path.is_symlink():
            if archive_root == 'logs' and args.job_id not in str(path.relative_to(subdir)):
                continue
            files[archive_root + '/' + str(path.relative_to(subdir))] = path
for path in (args.work/'logs').glob('*{}*'.format(args.job_id)):
    if path.is_file() and not path.is_symlink():
        files['logs/' + path.name] = path
for path in (run/'checkpoints').rglob('*'):
    if not path.is_file() or path.is_symlink():
        continue
    if (path.suffix in ('.json', '.yaml', '.txt', '.patch')
            or path.name == 'scheduler_state_dict.pt') and path.stat().st_size < 4*1024*1024:
        files['checkpoint-metadata/' + str(path.relative_to(run/'checkpoints'))] = path
runtime = args.work/'repo-ada15200'
for subdir in ('methods/dspark/beverin', 'methods/dspark/train', 'patches'):
    source = runtime/subdir
    if not source.exists():
        continue
    for path in source.rglob('*'):
        if (path.is_file() and not path.is_symlink() and '__pycache__' not in path.parts
                and path.suffix != '.pyc'):
            files['runtime-source/' + str(path.relative_to(runtime))] = path
files['runtime-source/REVISION'] = runtime/'REVISION'
target = Path('/capstor/store/cscs/swissai/infra01/hf_models/models/swiss-ai/Apertus-v1.5-70B')
for name in ('config.json', 'model.safetensors.index.json', 'generation_config.json'):
    path = target/name
    if path.is_file():
        files['target-metadata/' + name] = path
states = subprocess.check_output(['sacct', '-X', '-j', args.job_id, '--parsable2',
                                  '--format=JobIDRaw,State,Elapsed,ElapsedRaw,AllocNodes,Start,End,ExitCode,NodeList%1000'],
                                 universal_newlines=True,
                                 env=dict(__import__('os').environ, SLURM_TIME_FORMAT='standard'))
state_rows = [line.split('|') for line in states.splitlines()]
job_row = next((row for row in state_rows[1:] if row[0] == args.job_id), None)
if not job_row or job_row[1] in ('RUNNING', 'PENDING', 'COMPLETING'):
    raise SystemExit('Accounting is not final yet; retry after the next status tick')
manifest = {'collected_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
            'run_path': str(run), 'job_id': args.job_id, 'files': []}
args.out.parent.mkdir(parents=True, exist_ok=True)
with tarfile.open(str(args.out), 'w:gz', dereference=True) as archive:
    for name, path in sorted(files.items()):
        archive.add(str(path), arcname=name, recursive=False)
        manifest['files'].append({'archive_path': name, 'cluster_path': str(path),
                                  'size': path.stat().st_size, 'sha256': sha256(path)})
    extras = {'collection.json': json.dumps(manifest, indent=2).encode(),
              'sacct.txt': states.encode(),
              'SOURCE_SHA256SUMS': ''.join('{}  {}\n'.format(x['sha256'], x['archive_path'])
                                         for x in manifest['files']).encode()}
    for name, data in extras.items():
        info = tarfile.TarInfo(name)
        info.size = len(data)
        info.mtime = int(time.time())
        archive.addfile(info, io.BytesIO(data))
print(json.dumps({'archive': str(args.out), 'files': len(files),
                  'sha256': sha256(args.out), 'size': args.out.stat().st_size}))
