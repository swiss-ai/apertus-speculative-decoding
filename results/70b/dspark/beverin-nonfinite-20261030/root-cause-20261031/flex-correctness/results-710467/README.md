# Attention forward/backward oracle, job 710467

Both image cells passed all eight numerical cases. Job 710467 used one
Beverin node (nid002410), physical GPU2, and completed in 271 seconds with
exit code 0 on 31 October 2026, 16:06:13-16:10:44 UTC. No model, optimizer,
corpus, training loop or NCCL group was loaded. Runtime
e9da7e874f1f9db3ad6957d836ca94bc521657f5 is immutable.

The pinned production mask source SHA256 is
166ddad80b67096f36bd490869cde2c650d5d35a5c3c4e7d940d561da8256bde.
The compiled mask builder and attention call use the captured causal
2048-token sliding window, two documents, and valid sorted anchors followed
by zero padding. Both complete image SHA256 values were verified before cells.
The two images used separate Triton/Inductor caches.

Per image, six production-size cases use Q4096/KV12288, GQA64:8 heads,
head dimension128, padded/unpadded anchors and three fixtures: uniform QK
with constant V, random QK with zero V, and saturated QK with constant V.
They check output, all three gradients, analytic uniform dV, masked-column
zeros and dV mass. Two smaller Q256/KV512, GQA8:2 random-QKV cases compare
output plus nonzero dQ/dK/dV with an independent FP32 math attention oracle.

| Metric | Original image | ALPS image |
|---|---:|---:|
| Passed cases | 8/8 | 8/8 |
| Nonfinite output/gradient values | 0 | 0 |
| Production fixture max output error | 0 | 0 |
| Max dV mass relative error | 0.00071847 | 0.00071847 |
| Max small FP32-reference relative L2 error | 0.00304606 | 0.00304610 |

The worst small-reference difference is 0.305%, below the explicit 3%
tolerance; dV mass error is at most 0.07185%, below 2%. These two images
give no evidence of a deterministic NaN for the tested padded-anchor and
masked-tile patterns. Neither result demonstrates that ALPS fixes the
original fault. This standalone test does not reproduce the original
training batch, dynamic hidden states, exact fused model graph, FSDP overlap,
optimizer, 64-rank topology, or a long-lived allocator state. Rare or
batch-dependent faults remain possible. The initiating fault is unproven.

The earlier 710459 import-registration harness failure is retained separately;
this is a new directory and runtime. No further job or automatic retry was
submitted. All 27 raw files (40,037 bytes) match remote hashes in
collection.json. `python3 -B verify_results.py` verifies raw provenance,
image hashes, case signatures, nonfinite fields and recorded numerical
criteria; analysis.json contains the computed summary. Raw commands, source,
versions, mask hashes, environments, node IDs, logs and both complete results remain
under raw/. sacct.txt uses cluster local UTC+1 timestamps.
