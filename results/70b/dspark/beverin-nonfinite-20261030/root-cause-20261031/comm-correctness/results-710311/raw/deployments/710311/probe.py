"""Bounded collective correctness probe; no language model or optimizer."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

import torch
import torch.distributed as dist
import torch.distributed._functional_collectives as funcol
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor import DTensor, Shard, _collective_utils, placement_types

SEED = 20261031


def fingerprint(indices, iteration):
    # Exact bounded integers; neither inputs nor expected sums grow with time.
    mixed = (indices * 1664525 + 1013904223 + iteration * 97) % 2147483648
    return (mixed ^ (mixed >> 13)) % 16


def pattern(indices, rank, iteration, dtype):
    return (fingerprint(indices, iteration) + rank * 16).to(dtype)


def shard_flip(value, gather_dim, shard_dim, mesh):
    # Exact implementation used by the production DTensor workaround.
    output = funcol.all_gather_tensor(value, gather_dim, (mesh, 0))
    if isinstance(output, funcol.AsyncCollectiveTensor):
        output = output.wait()
    return torch.chunk(output, mesh.size(0), dim=shard_dim)[mesh.get_local_rank(0)].contiguous()


def run(args):
    rank, world = dist.get_rank(), dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    device = torch.device("cpu" if args.cpu else "cuda", None if args.cpu else local_rank)
    mesh = DeviceMesh(device.type, list(range(world)))
    def production_flip(value, gather_dim, shard_dim, mesh, mesh_dim):
        assert mesh_dim == 0
        return shard_flip(value, gather_dim, shard_dim, mesh)
    _collective_utils.shard_dim_alltoall = production_flip
    placement_types.shard_dim_alltoall = production_flip
    root = Path(args.output)
    root.mkdir(parents=True, exist_ok=True)
    report = {"rank": rank, "local_rank": local_rank, "world_size": world,
              "seed": SEED, "iterations": args.iterations, "shapes": args.shapes,
              "argv": sys.argv, "python": sys.version,
              "probe_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "source_revision": os.environ.get("PROBE_SOURCE_REVISION"),
              "source_branch": os.environ.get("PROBE_SOURCE_BRANCH"),
              "torch": torch.__version__, "hip": torch.version.hip,
              "torch_git_revision": torch.version.git_version,
              "nccl_or_rccl_version": torch.cuda.nccl.version() if not args.cpu else None,
              "blas": str(torch.backends.cuda.preferred_blas_library()) if not args.cpu else None,
              "environment": {k: v for k, v in os.environ.items() if k.startswith(
                  ("NCCL_", "RCCL_", "FI_", "OFI_", "CUDA_", "HIP_", "ROCR_", "GPU_", "SLURM_JOB_"))},
              "checks": [], "buffer_addresses": [], "status": "running"}
    temporary = root / ("rank%d.json.tmp" % rank)

    def publish():
        temporary.write_text(json.dumps(report, indent=2) + "\n")
        temporary.replace(root / ("rank%d.json" % rank))

    def check(actual, expected, operation, mode, shape, dtype, iteration):
        finite = torch.isfinite(actual)
        mismatch = (~finite) | (actual != expected)
        local_mismatch_detected = bool(mismatch.any().item())
        local_bad = mismatch.any().to(torch.int32)
        dist.all_reduce(local_bad, op=dist.ReduceOp.MAX)
        item = {"operation": operation, "mode": mode, "shape": list(shape),
                "dtype": str(dtype), "iteration": iteration, "elements": actual.numel(),
                "nonfinite": int((~finite).sum().item()), "mismatches": int(mismatch.sum().item()),
                "local_mismatch_detected": local_mismatch_detected,
                "global_bad_flag": int(local_bad.item())}
        if local_mismatch_detected or local_bad.item():
            if item["mismatches"]:
                index = int(mismatch.flatten().nonzero()[0].item())
                item["first_mismatch"] = {"flat_index": index,
                    "observed": str(actual.flatten()[index].item()),
                    "expected": expected.flatten()[index].item()}
            report["checks"].append(item)
            report["status"] = "failed"
            publish()
            print("COMM_CORRECTNESS_FAILURE " + json.dumps(dict(item, rank=rank)), flush=True)
            # Keep all rank reports alive until every peer has written its result.
            dist.barrier()
            raise RuntimeError("collective correctness mismatch; no automatic retry")
        report["checks"].append(item)

    publish()
    started = time.monotonic()
    for rows, columns in args.shapes:
        assert columns % world == 0
        local_indices = torch.arange(rows * columns, device=device, dtype=torch.int64).reshape(rows, columns)
        full_indices = torch.arange(world * rows * columns, device=device, dtype=torch.int64).reshape(world * rows, columns)
        owners = torch.arange(world, device=device).repeat_interleave(rows).view(-1, 1)
        flipped_indices = ((torch.arange(world * rows, device=device) % rows).view(-1, 1) * columns
                           + torch.arange(columns // world, device=device).view(1, -1)
                           + rank * (columns // world))
        for dtype in (torch.bfloat16, torch.float32):
            for mode in ("reused", "fresh"):
                buffers = None
                b_buffer = None
                for iteration in range(args.iterations):
                    if buffers is None or mode == "fresh":
                        buffers = (torch.empty((rows, columns), device=device, dtype=dtype),
                                   torch.empty((world * rows, columns), device=device, dtype=dtype),
                                   torch.empty((world * rows, columns), device=device, dtype=dtype),
                                   torch.empty((rows, columns), device=device, dtype=dtype))
                    value, gathered, rs_input, scattered = buffers
                    report["buffer_addresses"].append({"mode": mode, "shape": [rows, columns],
                        "dtype": str(dtype), "iteration": iteration,
                        "addresses": [x.data_ptr() for x in buffers]})
                    value.copy_(pattern(local_indices, rank, iteration, dtype))
                    expected_gather = (fingerprint(full_indices % (rows * columns), iteration)
                                       + owners * 16).to(dtype)
                    work = dist.all_gather_into_tensor(gathered, value, async_op=True)
                    # Independent default-stream computation overlaps the collective;
                    # references remain live and output is consumed only after wait().
                    scratch = torch.ones(64, device=device).add_(iteration)
                    work.wait()
                    check(gathered, expected_gather, "all_gather", mode, (rows, columns), dtype, iteration)
                    rs_input.copy_(pattern(full_indices, rank, iteration, dtype))
                    expected_rs = (world * fingerprint(full_indices[rank * rows:(rank + 1) * rows], iteration)
                                   + 16 * sum(range(world))).to(dtype)
                    assert float(expected_rs.max().item()) < 256
                    work = dist.reduce_scatter_tensor(scattered, rs_input, async_op=True)
                    scratch.add_(1)
                    work.wait()
                    check(scattered, expected_rs, "reduce_scatter", mode, (rows, columns), dtype, iteration)
                    flipped = shard_flip(value, 0, 1, mesh)
                    expected_flip = (fingerprint(flipped_indices, iteration) + owners * 16).to(dtype)
                    check(flipped, expected_flip, "shard_flip", mode, (rows, columns), dtype, iteration)
                    roundtrip = shard_flip(flipped, 1, 0, mesh)
                    check(roundtrip, pattern(local_indices, rank, iteration, dtype),
                          "shard_roundtrip", mode, (rows, columns), dtype, iteration)
                    # Metrics are a separate FP32 vector in the real trainer.
                    metric_indices = torch.arange(64, device=device, dtype=torch.int64)
                    metrics = pattern(metric_indices, rank, iteration, torch.float32)
                    dist.all_reduce(metrics)
                    expected_metrics = (world * fingerprint(metric_indices, iteration)
                                        + 16 * sum(range(world))).float()
                    check(metrics, expected_metrics, "packed_metric_all_reduce", mode,
                          (64,), torch.float32, iteration)
                    # A[world*rows,K] @ B[K,128], entries +/-1, has an exact
                    # analytical outer-product result +/-K (40960 in the FC cell).
                    row_sign = torch.where(torch.arange(rows, device=device) % 2 == 0, 1., -1.)
                    value.copy_(row_sign[:, None].expand(rows, columns))
                    column_sign = torch.where((torch.arange(128 // world, device=device)
                                               + rank * (128 // world)) % 2 == 0, 1., -1.)
                    if b_buffer is None or mode == "fresh":
                        b_buffer = torch.empty((columns, 128 // world), device=device, dtype=dtype)
                    b_buffer.copy_(column_sign[None, :].expand(columns, -1))
                    b_local = b_buffer
                    a = DTensor.from_local(value, mesh, [Shard(0)], run_check=False)
                    b = DTensor.from_local(b_local, mesh, [Shard(1)], run_check=False)
                    product = (a @ b).full_tensor()
                    all_rows = torch.where(torch.arange(world * rows, device=device) % 2 == 0, 1., -1.)
                    all_columns = torch.where(torch.arange(128, device=device) % 2 == 0, 1., -1.)
                    expected_product = (all_rows[:, None] * all_columns[None, :] * columns).to(dtype)
                    check(product, expected_product, "structured_dtensor_gemm", mode,
                          (world * rows, columns, 128), dtype, iteration)
                publish()
    report["status"] = "passed"
    report["elapsed_seconds"] = time.monotonic() - started
    report["total_checks"] = len(report["checks"])
    report["total_elements_checked"] = sum(x["elements"] for x in report["checks"])
    publish()
    dist.barrier()
    if rank == 0:
        print("COMM_CORRECTNESS_PASS " + json.dumps({"world_size": world,
              "checks_per_rank": len(report["checks"]), "seed": SEED,
              "iterations": args.iterations, "elapsed_seconds": report["elapsed_seconds"]}), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--iterations", type=int, default=32)
    parser.add_argument("--shapes", nargs="+", default=["128x1024", "128x40960"])
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()
    args.shapes = [tuple(map(int, x.split("x"))) for x in args.shapes]
    assert 1 <= args.iterations <= 32
    torch.set_num_threads(1)
    torch.manual_seed(SEED)
    if not args.cpu:
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        torch.backends.cuda.preferred_blas_library("cublas")
    dist.init_process_group("gloo" if args.cpu else "nccl",
                            timeout=__import__("datetime").timedelta(seconds=90))
    assert dist.get_world_size() == 4
    try:
        run(args)
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
