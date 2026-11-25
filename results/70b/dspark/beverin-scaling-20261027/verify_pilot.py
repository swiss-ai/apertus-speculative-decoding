#!/usr/bin/env python3
"""Verify completed scaling-pilot evidence without loading model weights."""
import argparse
import json
from pathlib import Path
import re

from analyze import analyze, read


def verify(cell, nodes, job_id):
    report = analyze(cell, nodes)
    audit = json.loads((cell/'deployment/job-audit.json').read_text())
    first = json.loads((cell/'logs'/('pilot-first-save-{}.json'.format(job_id))).read_text())
    initial = json.loads((cell/'deployment/initial-training-state.json').read_text())
    rows = [line.split('|') for line in (cell/'sacct.txt').read_text().splitlines()]
    accounting = dict(zip(rows[0], next(row for row in rows[1:] if row[0] == job_id)))
    attempts = report['attempts']
    first_attempt = next(a for a in attempts if a['resume_global_step'] == 0)
    resumed = next(a for a in attempts if a['resume_global_step'] == 100)
    hooks_by_node, errors, losses = {}, [], []
    for log in (cell/'logs').glob('train-*.log*'):
        content = read(log)
        errors += [float(v) for v in re.findall(r'train/error_records=([\deE.+-]+)', content)]
        losses += re.findall(r'train/loss=([^,\s]+)', content)
        node = re.search(r'-node(\d+)\.log', log.name).group(1)
        for payload in re.findall(r'BEVERIN_HOOKS (\{[^{}\n]*\})', content):
            hooks_by_node.setdefault(int(node), []).append(json.loads(payload))
    checks = {
        'slurm_completed': accounting['State'] == 'COMPLETED' and accounting['ExitCode'] == '0:0',
        'audit_ok': audit['status'] == 'ok' and audit['return_code'] == 0,
        'started_fresh': initial['global_step'] == 0,
        'first_save_at_100': first == {'epoch': 0, 'local_step': 100, 'global_step': 100},
        'first_segment_last_metric_99': first_attempt['last_logged_sample']['global_step'] == 99,
        'first_resumed_metric_100': resumed['first_logged_sample']['global_step'] == 100,
        'last_resumed_metric_199': resumed['last_logged_sample']['global_step'] == 199,
        'final_checkpoint_200': audit['final_state'] == {'epoch': 0, 'local_step': 200, 'global_step': 200},
        'epoch_length_consistent': report['estimate']['epoch_step_count_agrees_across_attempts'],
        'at_least_100_steady_samples': report['steady_phase_seconds']['step']['n'] >= 100,
        'all_nodes_confirm_rocblas_unprofiled': set(hooks_by_node) == set(range(nodes)) and all(
            all(h['blas_backend'] == '_BlasBackend.Cublas' and h['profile_steps'] == 0
                for h in hooks) for hooks in hooks_by_node.values()),
        'no_reported_data_recovery_errors': bool(errors) and max(errors) == 0,
        'no_nonfinite_losses': bool(losses) and not any(v.lower() in ('nan', 'inf', '-inf') for v in losses),
        'lr_resumed_above_zero': resumed['first_logged_sample']['lr/Muon'] > 0,
        'lr_continues_forward': resumed['first_logged_sample']['lr/Muon'] > first_attempt['last_logged_sample']['lr/Muon'],
    }
    return {'job_id': job_id, 'nodes': nodes, 'checks': checks,
            'status': 'passed' if all(checks.values()) else 'needs_attention',
            'last_lr_before_save': first_attempt['last_logged_sample']['lr/Muon'],
            'first_lr_after_resume': resumed['first_logged_sample']['lr/Muon'],
            'notes': ['Cursor and logged LR continuity are checked; bitwise RNG equivalence is not claimed.',
                      'Large weights/optimizer remain at the cluster paths; the batch-node audit verifies their presence.']}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('cell', type=Path)
    parser.add_argument('--nodes', type=int, required=True)
    parser.add_argument('--job-id', required=True)
    args = parser.parse_args()
    result = verify(args.cell, args.nodes, args.job_id)
    print(json.dumps(result, indent=2))
    raise SystemExit(result['status'] != 'passed')
