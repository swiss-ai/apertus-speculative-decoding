#!/usr/bin/env python3
"""Analyze immutable scaling-pilot logs; no training or checkpoint mutations."""
import argparse
import calendar
import datetime
import gzip
import json
from pathlib import Path
import re
import statistics


def read(path):
    opener = gzip.open if path.suffix == '.gz' else open
    with opener(path, 'rt', errors='replace') as stream:
        return stream.read()


def stats(values):
    if not values:
        return None
    ordered = sorted(values)
    return {'n': len(values), 'mean': statistics.mean(values),
            'median': statistics.median(values), 'min': ordered[0],
            'p90': ordered[int(.9 * (len(ordered) - 1))], 'max': ordered[-1]}


def analyze(cell, nodes, dataset_rows=1576555):
    attempts, saves, totals = [], {}, []
    for log in sorted((cell / 'logs').glob('train-*-node0.log*')):
        content = read(log)
        resume = re.findall(r'Resuming mid-epoch from.*?global_step=(\d+)',
                            content, re.DOTALL)
        skipped = int(resume[-1]) if resume else 0
        progress = re.findall(r'Epoch\s+\d+[^\r\n]*?(\d[\d,]*)/(\d[\d,]*)', content)
        epoch_steps = int(progress[-1][1].replace(',', '')) + skipped if progress else None
        if epoch_steps:
            totals.append(epoch_steps)
        local_saves = set()
        for event, payload in re.findall(r'BEVERIN_SAVE_(START|END) (\{[^{}\n]*\})', content):
            data = json.loads(payload)
            step = int(data['global_step'])
            local_saves.add(step)
            saves.setdefault(step, {'START': [], 'END': []})[event].append(data['unix_s'])
        steps = []
        for match in re.finditer(r'profile/fetch_ms=.*?global_step=\d+', content, re.DOTALL):
            steps.append({key: float(value) for key, value in re.findall(
                r'(profile/[a-z_]+|lr/[A-Za-z_]+|global_step)=([0-9.eE+\-]+)', match.group())})
        # The fetch timer for the step following an in-process checkpoint includes
        # its save latency. Report those samples separately from compute/feature work.
        steady = [s for s in steps[1:] if int(s['global_step']) not in local_saves]
        attempts.append({'log': log.name, 'epoch_steps': epoch_steps,
                         'resume_global_step': skipped, 'logged_steps': len(steps),
                         'first_logged_sample': steps[0] if steps else None,
                         'last_logged_sample': steps[-1] if steps else None,
                         'warmup_global_step': int(steps[0]['global_step']) if steps else None,
                         'post_save_global_steps': [int(s['global_step']) for s in steps[1:]
                                                   if int(s['global_step']) in local_saves],
                         'steady_samples': steady})
    samples = [step for attempt in attempts for step in attempt['steady_samples']]
    phases = {phase: stats([s['profile/' + phase + '_ms'] / 1000 for s in samples
                            if 'profile/' + phase + '_ms' in s])
              for phase in ('fetch', 'fwd', 'bwd', 'opt', 'step')}
    windows = []
    for step, events in sorted(saves.items()):
        if events['START'] and events['END']:
            start, end = min(events['START']), max(events['END'])
            windows.append({'global_step': step, 'start_unix_s': start,
                            'end_unix_s': end, 'duration_s': end - start})
    memory = {}
    for log in sorted((cell / 'logs').glob('mem-*-node*.log*')):
        samples_mem = [(calendar.timegm(datetime.datetime.strptime(stamp,
                        '%Y-%m-%dT%H:%M:%SZ').timetuple()), int(used), int(avail))
                       for stamp, used, avail in re.findall(
                           r'(\S+) used_gb=(\d+) avail_gb=(\d+)', read(log))]
        if samples_mem:
            save_mem = []
            for window in windows:
                during = [s for s in samples_mem if window['start_unix_s'] <= s[0]
                          <= window['end_unix_s']]
                if during:
                    save_mem.append({'global_step': window['global_step'],
                                     'samples': len(during),
                                     'peak_used_gib': max(s[1] for s in during),
                                     'min_available_gib': min(s[2] for s in during)})
            memory[log.name] = {'samples': len(samples_mem),
                                'peak_used_gib': max(s[1] for s in samples_mem),
                                'min_available_gib': min(s[2] for s in samples_mem),
                                'save_windows': save_mem}
    estimate = None
    if totals and phases['step']:
        epoch_steps = totals[0]
        mean_s = phases['step']['mean']
        median_s = phases['step']['median']
        save_s = statistics.mean(w['duration_s'] for w in windows) if windows else None
        estimate = {'epoch_steps': epoch_steps,
                    'epoch_step_count_agrees_across_attempts': len(set(totals)) == 1,
                    'train_hours_from_mean_step': epoch_steps * mean_s / 3600,
                    'train_hours_from_median_step': epoch_steps * median_s / 3600,
                    'train_node_hours_from_mean_step': nodes * epoch_steps * mean_s / 3600,
                    'ten_epoch_train_days_from_mean_step': 10 * epoch_steps * mean_s / 86400,
                    'nominal_global_token_budget_per_s': nodes * 16384 / mean_s,
                    'dataset_rows_per_s_projected': dataset_rows / (epoch_steps * mean_s),
                    'rank0_actual_tokens_per_s': stats([s['profile/tokens_per_s'] for s in samples
                                                       if 'profile/tokens_per_s' in s]),
                    'mean_save_duration_s': save_s,
                    'approx_64_saves_per_epoch_hours': 64 * save_s / 3600 if save_s else None,
                    'excludes': ['validation', 'startup', 'resume', 'queue waits',
                                 'regular saves (estimated separately)']}
    return {'nodes': nodes, 'attempts': attempts, 'steady_phase_seconds': phases,
            'saves': windows, 'memory_gib_10s_samples': memory, 'estimate': estimate,
            'notes': ['First sample per attempt and first sample after each save excluded.',
                      'Global token rate uses capacity, not measured summed tokens.',
                      'Actual token rate is logged for rank 0 only.',
                      'Short pilots do not establish training-quality equivalence.']}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('cell', type=Path)
    parser.add_argument('--nodes', type=int, required=True)
    args = parser.parse_args()
    print(json.dumps(analyze(args.cell, args.nodes), indent=2))
