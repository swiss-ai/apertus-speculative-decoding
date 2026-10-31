# Standalone DTensor Muon normalization/iteration oracle (prepared, not launched)

This tests the remaining configured math path without loading any model,
corpus or optimizer and without a training loop. The planned allocation is
one Beverin node, at most20minutes, two ranks on physical GPUs2/3, sequential
original and ALPS image cells (360-second timeout each), no automatic retry.
GPU launch requires parent review; no job has been submitted.

Each rank constructs identical known BF16 global input, takes its row shard,
checks exact reconstruction, and calls the actual installed private
`torch.optim._muon._zeropower_via_newtonschulz` on a cloned DTensor and a
separate cloned ordinary dense GPU tensor. This preserves the original
function's norm/clamp/in-place normalization and all five Newton-Schulz
iterations: coefficients(3.4445,-4.7750,2.0315), eps1e-7. The production
all-gather/chunk shard-redistribution workaround and forced rocBLAS are active.
No manually computed norm replaces the function's native calculation.

Shapes are16x80 and the captured FC global8192x40960. With two ranks, the
large local shape is4096x40960; original64-rank local shape128x40960 is not
replayed. Fixtures are zero, tiny random(1e-12), random, rank-skewed random,
rank-one uniform ones, and signed rank-one outer product. Rank-one cases
amplify an incorrect local norm into an observable Newton-Schulz failure;
wide random gradients alone can miss that mechanism.

Checks record local/materialized global norms, placements before/after
clamp, full output nonfinite counts, raw update relative L2 error<=5%,
max absolute error, and exactly zero output for zero input. Norm/clamp
relative error must be<=2% (BF16), and dense reference must also remain
finite. These compare raw NS updates so BF16 parameter rounding cannot
conceal a wrong small update. Local mismatch decisions remain authoritative
if the diagnostic flag collective is itself wrong. Any rank failure or
cell timeout fails the batch summary; there are no retries.

The final CPU/Gloo preflight in cpu-preflight-production used the exact
pinned2.13 function AST with host torch2.15.0+cpu and production shard flip.
All6cases on both ranks passed; rank-one differences were exactly zero.
The largest raw update difference was2.393% for rank-skewed random, where
BF16 materialized norm840 differs from dense844 (0.474%). This is below the
explicit5% tolerance and demonstrates why strict bitwise equality is not
required. Raw logs, native source captures, exact function, hashes and
per-rank JSON are retained. Earlier preflights are preserved separately.
CPU dispatch is not the complete native2.13 runtime.

source-audit retains exact image _pointwise_ops.py from read-only squashfs
extraction. Both images and the CPU2.15 host have identical bytes:
SHA256b25444743c8c707cf30aaa3e27a9b2d850a59f6309ed93eca4a466244b3a73db.
This removes a source-difference claim for that file; it does not prove
identity of the entire dispatcher/norm strategy framework. The GPU probe
also preserves native pointwise/math strategy source and actual Muon
function per rank, torch Git revision, full argv, source branch/revision,
resolved nonsecret environment, images and Slurm/node identity.

Passing would constrain deterministic errors for these inputs at two ranks;
it cannot establish the original519 trigger or64-rank training stability.
