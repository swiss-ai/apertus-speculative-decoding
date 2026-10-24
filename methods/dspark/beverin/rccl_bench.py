"""All-reduce / all-to-all / all-gather bandwidth across nodes with torch + RCCL.

Run under torchrun (one process per GPU). Prints the bus bandwidth per size
from rank 0; NCCL_DEBUG=INFO in the log shows which network plugin RCCL used.
"""
import os
import time

import torch
import torch.distributed as dist

dist.init_process_group("nccl")
rank, world = dist.get_rank(), dist.get_world_size()
local = int(os.environ["LOCAL_RANK"])
torch.cuda.set_device(local)
dev = torch.device("cuda", local)


def bench(name, fn, nbytes, factor, iters=20):
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    dist.barrier()
    t = time.time()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    dt = (time.time() - t) / iters
    if rank == 0:
        print(f"{name:10s} {nbytes / 2**20:8.0f} MiB  {dt * 1e3:8.2f} ms  busbw {nbytes * factor / dt / 1e9:7.1f} GB/s",
              flush=True)


for mib in (8, 64, 512):
    n = mib * 2**20 // 2
    x = torch.ones(n, dtype=torch.bfloat16, device=dev)
    y = torch.empty_like(x)
    g = torch.empty(n * world, dtype=torch.bfloat16, device=dev)
    bench("all_reduce", lambda: dist.all_reduce(x), x.nbytes, 2 * (world - 1) / world)
    bench("all_to_all", lambda: dist.all_to_all_single(y, x), x.nbytes, (world - 1) / world)
    bench("all_gather", lambda: dist.all_gather_into_tensor(g, x), g.nbytes, (world - 1) / world)
# Correctness of one all-to-all: rank r sends value r to everyone.
x = torch.full((world * 4,), float(rank), device=dev)
y = torch.empty_like(x)
dist.all_to_all_single(y, x)
ok = torch.equal(y, torch.arange(world, device=dev, dtype=x.dtype).repeat_interleave(4))
print(f"rank {rank} all_to_all correct {ok}", flush=True) if not ok or rank == 0 else None
dist.destroy_process_group()
