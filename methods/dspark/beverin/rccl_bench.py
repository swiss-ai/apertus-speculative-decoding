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


DTYPES = {"bf16": torch.bfloat16, "fp32": torch.float32}
for dname in os.environ.get("BENCH_DTYPES", "bf16").split(","):
    dt = DTYPES[dname]
    for mib in (8, 64, 512):
        n = mib * 2**20 // dt.itemsize
        n -= n % world
        x = torch.ones(n, dtype=dt, device=dev)
        y = torch.empty_like(x)
        g = torch.empty(n * world, dtype=dt, device=dev)
        r = torch.empty(n // world, dtype=dt, device=dev)
        if rank == 0:
            print(f"-- {dname}", flush=True)
        bench("all_reduce", lambda: dist.all_reduce(x), x.nbytes, 2 * (world - 1) / world)
        bench("all_to_all", lambda: dist.all_to_all_single(y, x), x.nbytes, (world - 1) / world)
        bench("all_gather", lambda: dist.all_gather_into_tensor(g, x), g.nbytes, (world - 1) / world)
        bench("red_scat", lambda: dist.reduce_scatter_tensor(r, x), x.nbytes, (world - 1) / world)
# Training-like buffers: a fresh allocation for every call (new addresses, so
# the network plugin's registration cache misses), as FSDP and Muon produce.
if os.environ.get("BENCH_FRESH", "0") == "1":
    for mib in (64, 512):
        n = mib * 2**20 // 4
        n -= n % world

        def fresh_rs():
            torch.cuda.empty_cache()
            x = torch.ones(n, dtype=torch.float32, device=dev)
            r = torch.empty(n // world, dtype=torch.float32, device=dev)
            dist.reduce_scatter_tensor(r, x)

        def fresh_ag():
            torch.cuda.empty_cache()
            x = torch.ones(n // world, dtype=torch.float32, device=dev)
            g = torch.empty(n, dtype=torch.float32, device=dev)
            dist.all_gather_into_tensor(g, x)

        if rank == 0:
            print("-- fresh buffers, fp32", flush=True)
        bench("fresh_rs", fresh_rs, n * 4, (world - 1) / world, iters=5)
        bench("fresh_ag", fresh_ag, n * 4, (world - 1) / world, iters=5)
# Correctness of one all-to-all: rank r sends value r to everyone.
x = torch.full((world * 4,), float(rank), device=dev)
y = torch.empty_like(x)
dist.all_to_all_single(y, x)
ok = torch.equal(y, torch.arange(world, device=dev, dtype=x.dtype).repeat_interleave(4))
print(f"rank {rank} all_to_all correct {ok}", flush=True) if not ok or rank == 0 else None
dist.destroy_process_group()
