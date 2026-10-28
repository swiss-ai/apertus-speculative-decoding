# Serving-layout recovery continuation, 2026-10-28

Original recovery preflight 3769354 failed after 504seconds before readiness.
Its 31files/303,693bytes are preserved under the unchanged local
`../layout-recovery-20261027/controller-v1/`, with a collection manifest here.
`preflight 3769354-diagnosis.json` records the exact failures and source evidence.
No workload cells, speedup measurements or C1024 memory-stress results existed.

The import checker incorrectly named `v1.nixl_connector`; the pinned image's
factory registers `v1.nixl`. The four preceding NIXL package imports succeeded.
The repaired checker uses the factory module/export and retains partial module
identities if a later import fails. Separately, all three TP4 arms failed at
custom_all_reduce.cuh:455, the cudaIpcGetMemHandle export of CUDA graph buffers,
with expandable segments enabled. The retry explicitly sets
`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:False`, retains custom all-reduce
and CUDA graphs, and keeps gpu_memory_utilization 0.7 equally for paired plain
and speculative arms. The extra headroom costs KV capacity; actual capacity,
latency and resident memory remain measurements, not assumed savings.

Fresh preflight: **3782518**. Durable controller-v2 PID 136390. It follows the
same bounded sequence and original scientific gates as v1; no model, workload,
algorithm, k-grid or full-comparison scope was expanded. The fresh native
NIXL 1.5.0 prefix and controller-only dependencies are unchanged. No unrelated
checkout, shared install or dedicated job was changed.

The effective source freeze is base Git
`efad15a10609e615c5c8635be3c243d1ec5b9da7` plus the two-file overlay
SHA256 `51fdd333056fedaeafeb7cbc1df9c33bc71f24b84cdeb09e31bd98ca48ec9a8d`,
corresponding to local repair commit aa398fba. The guarded applier verified the
stopped v1 state/hash/process, clean original source, and all 75patched source
hashes before a new controller was launched. Mutation was serialized with the
shared sync.lock. Each normal sweep saves this source overlay as checkout.patch.
The image digest matches the original successful October 22/24 image.

Do not reset tracked remote source files or sync new highconc commits while
controller-v2 is active. Its deliberate tracked source changes are part of the
freeze. Local artifact commits can continue independently. Old v1 state, launch
receipt, source pin and raw deployment directory were never overwritten.

Durable remote root:
`/iopsstor/scratch/cscs/faruk_zahiragic/apertus-spec-decoding-highconc/results/8b/layout-recovery-20261028/controller-v2`.
The root contains source-freeze/overlay/application verification, launch script,
launch-receipt.json, controller.log and state.json. Read top-level job_ids or
stages[*].job_id to discover downstream jobs. Parent receipt retains 3769354
and points its monitor at the new state path. Slurm terminal state remains
separate from scientific completion. Stop-on-failure remains enabled.

The source repair passed 81focused tests plus Ruff; launcher shell syntax and
actual guarded source checks passed. These CPU checks do not establish native
compatibility or any performance benefit. That requires preflight and the
original comparisons to finish.

After measurement completion, collect the controller-v2 files and each stage's
output directory with hashes, original provenance and lossless compression as
needed. Follow the frozen CPU helper guide in
`../layout-recovery-20261027/review-helper-v1/README.md`, substituting this
controller-v2 state path in its local command:

```bash
/home/projects/apertus-speculative-decoding/.venv/bin/python \
  results/8b/layout-recovery-20261027/review-helper-v1/review.py \
  --analysis results/8b/layout-recovery-20261027/historical-analysis.json \
  --controller-state results/8b/layout-recovery-20261028/controller-v2/state.json \
  --path-map /iopsstor/scratch/cscs/faruk_zahiragic/apertus-spec-decoding-highconc=/home/projects/apertus-speculative-decoding-highconc \
  --verify-requests \
  --output results/8b/layout-recovery-20261028/final-review-v1
```

Keep failed/incomplete arms and losing strata explicit. Only substantial
repeatable benefits justify implementation language. Calibrated correctness,
repeat confidence and open-loop capacity claims remain unestablished unless
their original evidence exists. No further GPU work is needed for postprocessing.
