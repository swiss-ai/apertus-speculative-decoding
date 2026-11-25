# Communication/GEMM correctness, job 710311

All three cells passed on 31 October 2026. Job 710311 used two Beverin nodes
(nid002484/nid002486), ran 229 seconds, and completed with exit code 0.
Runtime eebba3d7e0cd8fe5b64e4b09a8193a787761ee0e is immutable. This was a
non-training test: no model, corpus, optimizer or learning rate was loaded.

Each cell checked four ranks, 32 iterations, both 128x1024 and 128x40960
local shapes, BF16/FP32 and fresh/reused buffers. All six operations checked
every output against exact rank/iteration/index references: all-gather,
reduce-scatter, functional shard flip/round-trip, packed FP32 metrics
all-reduce, and structured DTensor GEMM with analytical +/-K output.

| Cell | RCCL | Intra-node transport in logs | Exact checks | Bad values/mismatches |
|---|---|---|---:|---:|
| Original default | 2.27.7 | P2P/IPC | 6,144 | 0 |
| Original P2P disabled | 2.27.7 | SHM | 6,144 | 0 |
| ALPS default | 2.30.4 | SHM (P2P disabled by image) | 6,144 | 0 |

There were 18,432 total checks and 57,982,255,104 element comparisons,
including repeated buffers and repeated analytical values. These are not
independent fault trials. The original/default cell really selected P2P/IPC;
the comparison does not rely only on the environment variable. Both complete
image SHA256 values were checked by the batch script before running cells.

The test did not reproduce corruption, and gives no evidence that changing
P2P or adopting ALPS fixes the original NaN. Four ranks on two nodes for
seconds do not clear the original 64-rank topology, training-time overlap,
large model, long-lived allocator state, or rare faults. The test also does
not exercise attention or an optimizer. It is a bounded correctness check,
not a stability certificate or throughput comparison.

All 59 collected raw files (14,498,343 bytes) match the remote SHA256 entries
in collection.json. Run `python3 -B verify_results.py` to independently
recheck every raw file and rank report; analysis.json contains the totals.
Raw bytes are retained under raw/, including exact source, launch argv,
resolved nonsecret environments, logs, node IDs, versions and source revision.
Slurm times in sacct.txt are cluster local (UTC+1); submission was
15:41:08 UTC, allocation start 15:41:09 UTC, end 15:44:58 UTC.
