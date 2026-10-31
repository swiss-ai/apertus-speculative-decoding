"""Standalone exact Muon Newton-Schulz correctness; no optimizer/model/training."""
import argparse
import ast
import datetime
import hashlib
import inspect
import json
import os
from pathlib import Path
import sys
import time

import torch
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import DTensor, Shard, _collective_utils, placement_types
import torch.distributed._functional_collectives as funcol

COEFFICIENTS = (3.4445, -4.7750, 2.0315)
EPS = 1e-7
STEPS = 5
FIXTURES = ('zero', 'tiny-random', 'random', 'rank-skewed-random',
            'rank-one-ones', 'rank-one-signed')


def parse_shape(value):
    rows, cols = map(int, value.split('x'))
    assert rows > 0 and cols > 0
    return rows, cols


def load_function(args):
    if args.function_source:
        source = Path(args.function_source).read_text()
        tree = ast.parse(source)
        function = next(node for node in tree.body if isinstance(node, ast.FunctionDef)
                        and node.name == '_zeropower_via_newtonschulz')
        scope = {'torch': torch, 'Tensor': torch.Tensor}
        exec(compile(ast.Module(body=[function], type_ignores=[]),
                     str(args.function_source), 'exec'), scope)
        return scope[function.name], ast.get_source_segment(source, function).rstrip() + '\n'
    from torch.optim._muon import _zeropower_via_newtonschulz
    return _zeropower_via_newtonschulz, inspect.getsource(_zeropower_via_newtonschulz).rstrip() + '\n'


def scalar_details(value):
    placements = [repr(x) for x in value.placements] if isinstance(value, DTensor) else None
    local = value.to_local() if isinstance(value, DTensor) else value
    full = value.full_tensor() if isinstance(value, DTensor) else value
    return {'placements': placements, 'local': float(local), 'materialized': float(full),
            'dtype': str(value.dtype), 'finite': bool(torch.isfinite(full))}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True)
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cuda')
    parser.add_argument('--shapes', nargs='+', type=parse_shape,
                        default=[(16, 80), (8192, 40960)])
    parser.add_argument('--function-source', help='CPU preflight only: exact pinned function AST')
    parser.add_argument('--relative-tolerance', type=float, default=.05)
    args = parser.parse_args()
    if args.device == 'cuda':
        assert not args.function_source, 'GPU must use actual installed private function'
        torch.cuda.set_device(int(os.environ['LOCAL_RANK']))
        torch.backends.cuda.preferred_blas_library('cublas')
    torch.set_num_threads(1)
    backend = 'nccl' if args.device == 'cuda' else 'gloo'
    dist.init_process_group(backend, timeout=datetime.timedelta(seconds=180))
    rank, world = dist.get_rank(), dist.get_world_size()
    mesh = init_device_mesh(args.device, (world,))

    def production_flip(value, gather_dim, shard_dim, device_mesh, mesh_dim):
        # Exact trainer workaround, also checked by the communication oracle.
        assert mesh_dim == 0
        gathered = funcol.all_gather_tensor(value, gather_dim, (device_mesh, mesh_dim))
        if isinstance(gathered, funcol.AsyncCollectiveTensor):
            gathered = gathered.wait()
        return torch.chunk(gathered, device_mesh.size(mesh_dim), dim=shard_dim)[
            device_mesh.get_local_rank(mesh_dim)].contiguous()

    _collective_utils.shard_dim_alltoall = production_flip
    placement_types.shard_dim_alltoall = production_flip
    device = torch.device(args.device, int(os.environ['LOCAL_RANK'])) if args.device == 'cuda' else torch.device('cpu')
    function, function_source = load_function(args)
    output = Path(args.output); output.mkdir(exist_ok=True, parents=True)
    record = {'rank': rank, 'world_size': world, 'local_rank': int(os.environ['LOCAL_RANK']),
              'argv': sys.argv, 'torch': torch.__version__, 'torch_git': torch.version.git_version,
              'hip': torch.version.hip, 'python': sys.version, 'device': str(device),
              'source_revision': os.environ.get('SOURCE_REVISION'),
              'source_branch': os.environ.get('SOURCE_BRANCH'),
              'probe_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'function_sha256': hashlib.sha256(function_source.encode()).hexdigest(),
              'function_source_override': args.function_source,
              'coefficients': COEFFICIENTS, 'eps': EPS, 'steps': STEPS,
              'production_shard_flip': True,
              'relative_tolerance': args.relative_tolerance, 'seed': 20261031,
              'blas': str(torch.backends.cuda.preferred_blas_library()) if args.device == 'cuda' else None,
              'cases': [], 'status': 'running'}
    (output / ('muon-function-rank%d.py' % rank)).write_text(function_source)
    from torch.distributed.tensor._ops import _pointwise_ops, _math_ops
    record['native_sources'] = {}
    for name, module in [('pointwise_ops', _pointwise_ops), ('math_ops', _math_ops)]:
        native = Path(inspect.getfile(module)).read_bytes()
        record['native_sources'][name] = {'path': inspect.getfile(module),
                                         'sha256': hashlib.sha256(native).hexdigest()}
        (output / ('%s-rank%d.py' % (name, rank))).write_bytes(native)
    filename = output / ('rank%d.json' % rank)

    def publish():
        temporary = filename.with_suffix('.tmp')
        temporary.write_text(json.dumps(record, indent=2) + '\n')
        temporary.replace(filename)

    def check(condition, case, stage):
        # Keep the local decision authoritative even if a tested collective fails.
        local_bad = not bool(condition)
        case['stage'] = stage
        if local_bad:
            case['local_failure'] = True
            record['status'] = 'failed'; publish()
        flag = torch.tensor([int(local_bad)], dtype=torch.int32, device=device)
        dist.all_reduce(flag, op=dist.ReduceOp.MAX)
        if local_bad or bool(flag.item()):
            record['status'] = 'failed'; publish()
            raise AssertionError('Muon correctness failure at ' + stage)

    publish()
    try:
        for shape in args.shapes:
            assert shape[0] % world == 0
            for fixture in FIXTURES:
                started = time.monotonic()
                torch.manual_seed(20261031)
                if args.device == 'cuda':
                    torch.cuda.manual_seed_all(20261031)
                if fixture == 'zero':
                    dense = torch.zeros(shape, device=device, dtype=torch.bfloat16)
                elif fixture == 'rank-one-ones':
                    dense = torch.ones(shape, device=device, dtype=torch.bfloat16)
                elif fixture == 'rank-one-signed':
                    rows = (torch.arange(shape[0], device=device) % 2 * 2 - 1).bfloat16()
                    cols = (torch.arange(shape[1], device=device) % 3 * 2 - 1).bfloat16()
                    dense = rows[:, None] * cols[None, :]
                else:
                    dense = torch.randn(shape, device=device, dtype=torch.bfloat16)
                    if fixture == 'tiny-random':
                        dense *= 1e-12
                    if fixture == 'rank-skewed-random':
                        dense[shape[0] // 2:] *= 32
                # Independent clones: pinned grad.bfloat16() aliases its argument.
                local = dense.chunk(world, dim=0)[rank].clone()
                sharded = DTensor.from_local(local, device_mesh=mesh,
                                             placements=[Shard(0)], run_check=False)
                case = {'shape': list(shape), 'local_shape': list(local.shape),
                        'fixture': fixture, 'input_nonfinite': int((~torch.isfinite(dense)).sum()),
                        'input_placements': [repr(x) for x in sharded.placements]}
                record['cases'].append(case)
                recovered = sharded.full_tensor()
                case['input_mismatches'] = int((recovered != dense).sum())
                check(case['input_nonfinite'] == 0 and case['input_mismatches'] == 0,
                      case, 'input-exact-reconstruction')
                del recovered
                raw_norm = sharded.norm()
                clamped_norm = raw_norm.clamp(min=EPS)
                case['norm_before_clamp'] = scalar_details(raw_norm)
                case['norm_after_clamp'] = scalar_details(clamped_norm)
                dense_norm = dense.norm()
                case['dense_norm'] = float(dense_norm)
                case['norm_relative_error'] = abs(case['norm_before_clamp']['materialized'] - float(dense_norm)) / max(abs(float(dense_norm)), 1e-30)
                dense_clamped = float(dense_norm.clamp(min=EPS))
                case['clamp_relative_error'] = abs(case['norm_after_clamp']['materialized'] - dense_clamped) / dense_clamped
                check(case['norm_before_clamp']['finite'] and case['norm_after_clamp']['finite']
                      and case['norm_relative_error'] <= .02 and case['clamp_relative_error'] <= .02,
                      case, 'materialized-global-norm')
                # Do not substitute the measured norm into the actual function.
                # Both paths execute the same installed private function unmodified.
                reference = function(dense.clone(), COEFFICIENTS, STEPS, EPS)
                actual = function(sharded.clone(), COEFFICIENTS, STEPS, EPS)
                case['output_placements'] = [repr(x) for x in actual.placements]
                actual_full = actual.full_tensor()
                case['nonfinite'] = {'dense': int((~torch.isfinite(reference)).sum()),
                                     'dtensor': int((~torch.isfinite(actual_full)).sum())}
                check(not any(case['nonfinite'].values()), case, 'raw-newton-schulz-finite')
                difference = actual_full.float() - reference.float()
                reference_l2 = float(torch.linalg.vector_norm(reference.float()))
                difference_l2 = float(torch.linalg.vector_norm(difference))
                case['reference_l2'] = reference_l2
                case['difference_l2'] = difference_l2
                case['relative_l2_error'] = difference_l2 / max(reference_l2, 1e-30)
                case['max_abs_error'] = float(difference.abs().max())
                case['actual_nonzero'] = int(torch.count_nonzero(actual_full)) if fixture == 'zero' else None
                check(case['relative_l2_error'] <= args.relative_tolerance
                      and (fixture != 'zero' or case['actual_nonzero'] == 0),
                      case, 'raw-newton-schulz-dense-reference')
                case['seconds'] = time.monotonic() - started
                case['status'] = 'passed'; publish()
                print('MUON_ORACLE_PASS ' + json.dumps(case), flush=True)
                del dense, local, sharded, raw_norm, clamped_norm, actual, actual_full, reference, difference
        record['status'] = 'passed'; publish()
    except Exception as exc:
        record['status'] = 'failed'; record['error'] = repr(exc); publish()
        raise
    finally:
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
