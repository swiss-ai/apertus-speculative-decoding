# Beverin 32-node, one-epoch production run

User authorized a fresh 32-node draft-model run for **one epoch** after the
validated scaling pilots. Run ID:
`opb-thinkoff-32n-1epoch-beverin-20261027`.

- 32 nodes, 64 FSDP trainer ranks; each node retains a TP2 target extraction
  server on GPUs 0/1 and trainer ranks on GPUs 2/3.
- DSpark 5 layers / block 8 / Markov rank 256 / confidence head, target
  layers 1/20/39/58/77, CE 0.1 + TV 0.9, Muon, same regenerated corpus.
- SEQ 8192, anchors 512/rank, global token budget 524,288; LR
  8.48528137423857e-4. **EPOCHS=1**, expected full epoch **4,077 steps**;
  the existing linear scheduler therefore has **4,077 total / 40 warmup
  steps**. No MAX_STEPS, pilot segment/resume or profiling settings.
- Extraction memory 0.62, API/renderer 1/1, workers/prefetch 1/1, empty cache
  before saving, trainer rocBLAS. No trained pilot checkpoint is reused.
- Two dependent 24-hour allocations, normal regular saves every 1/64 epoch,
  deadline save 40 minutes before the allocation limit, and existing
  in-allocation retries. The measured training+saves forecast is 23.83 h,
  before startup and validation, so a continuation is expected.

The opt-in Beverin completion hook preserves normal training/validation
flow. If training's final checkpoint exists but validation was interrupted,
continuation skips training and repeats only final validation. Completion
requires the final 0/0/4077 state, nonempty model/optimizer/scheduler/config
files and finite saved validation metrics. A rank-0 atomic marker plus a
barrier records success. Redundant chain links authenticate that marker and
skip containers; audits allow unchanged steps only for authenticated
completion or successfully completed, logged validation-only recovery.
The one-node run and every Clariden job remain untouched.

Nine CPU lifecycle/audit checks and shell syntax checks passed before
submission. Runtime source is committed and copied into an immutable new
snapshot; `launch.sh` requires its full Git SHA. Slurm launchers capture
source revisions, exact configuration, container/data/model digests,
applied patches, resolved non-secret environments, job/node IDs, raw logs
and checkpoint/validation metadata with each deployment.
