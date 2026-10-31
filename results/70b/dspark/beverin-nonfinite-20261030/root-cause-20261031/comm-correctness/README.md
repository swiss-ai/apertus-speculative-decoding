# Non-training communication correctness diagnostic

One authorized two-node,20-minute allocation; two ranks per node on physical
GPUs2,3; no language model, corpus, loss function, LR, optimizer or training.
Three independent240-second cells, no automatic retries:

1. Original vLLM0.31 image, original injected network stack, P2P default.
2. Same image/stack with NCCL_P2P_DISABLE=1.
3. Pinned ALPS image with its bundled stack and default P2P disable workaround.

Both complete image SHA256 values must match the existing recorded pins before
any cell runs. All cells select rocBLAS, clear conflicting visibility masks,
retain expected network setup and collect INFO transport logs. Allowing P2P
is not proof it was selected: inspect the recorded channel/transport paths.
ALPS changes the complete backend stack, so its result is not a one-factor
comparison with the original image.

Each rank tests BF16 and FP32, fresh and reused buffers,32 iterations and local
shapes128x1024 plus128x40960. The latter matches the FC local shard shape in
the64-rank training layout; world size here is only4. Six exact checks per
iteration give1536 checks per rank per cell:

- all-gather;
- reduce-scatter;
- production functional-all-gather/chunk shard flip;
- reverse shard flip roundtrip;
- a separate FP32 packed-metric-vector all-reduce;
- structured DTensor GEMM with +/-1 entries, analytical output+/-K.

Rank/iteration/index-encoded inputs are reset each iteration. Communication
values are<=63; sums<=156, exactly representable by BF16. GEMM outputs+/-40960
are also exactly representable. Every output element is checked, with explicit
nonfinite counts, local mismatch flags, and first bad index/expected/observed.
A local mismatch fails even if the diagnostic all-reduced flag is itself bad.
Any peer hang then becomes a failed timeout cell. The top-level exit code
requires all four rank reports in every cell, expected check counts, passed
status and every mismatch/nonfinite field zero. Data pointers are retained to
show actual fresh/reused allocation behavior.

The final tiny-shape CPU/Gloo preflight passed on4 ranks with2 iterations:
192 exact checks total, zero mismatches/nonfinite. See cpu-preflight-summary.json
and cpu-preflight-final/. This validates reference formulas/control flow with
host torch2.15CPU; actual ROCm2.13/RCCL compatibility is the GPU job's purpose.
No claim of training stability follows from passing this synthetic test.
