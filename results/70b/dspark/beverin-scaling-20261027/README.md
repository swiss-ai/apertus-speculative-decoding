# Beverin rocBLAS scaling pilots, 2026-10-27

User-authorized follow-up to the fixed two-node pilot 694470. The ongoing
one-node production run and every Clariden job remain unchanged.

## Pilot plan

- Job **697549**, run `scaling-16n-20261027-rocblas`: 16 nodes, 32 training
  ranks, 262,144 nominal tokens/step, LR 6e-4. Save at step 100, resume in
  the same allocation with target servers retained, then save/exit at 200.
  Three-hour allocation, 20-minute deadline-save margin, lower scheduling
  priority (`--nice=10000`). At submission, 74 nodes were idle.
- A 32-node comparison is conditional on reviewing the 16-node run's
  performance, memory and exact checkpoint resume. It is not submitted yet.
- These pilots retain the ten-epoch schedule and full 8,192-token contexts
  with 512 anchors per rank. The early stop hook preserves the LR horizon;
  it does not use the trainer's `--max-steps` option.
- The batch-scaled LR is a heuristic for this performance pilot. Neither a
  short pilot nor fewer seconds per epoch establishes equal final quality.

Runtime source is the immutable `repo-ada15200/` snapshot. Speculators is
`7046ac63fac4eeb218dc2e92a385c7d1ec8c4e00` plus the recorded patches, in
`speculators-7046ac6-provenance/`. The validated vLLM 0.31 image and trainer
rocBLAS preference are retained; CPU profiling is disabled.

Cluster root: `/iopsstor/scratch/cscs/faruk_zahiragic/beverin-feasibility`.
Each pilot uses `runs/<run_id>/`, with a deployment directory per Slurm job.
Large model and optimizer checkpoints remain on the cluster. The final
artifact collection retains raw logs and small provenance/configuration
files; no corpus text, credentials or weights are included here.

## Analysis

`analyze.py` reads the archived logs without changing them. It omits the
first compiling step of each attempt, and reports steps immediately after
in-process checkpoints separately: their fetch timer includes checkpoint
latency. Epoch projections use both mean and median steady step duration,
plus separately reported save costs. Validation and queue delays are not
measured by these short pilots. Global token throughput uses the configured
capacity; the trainer logs actual token throughput only for rank 0.

Raw files use `-text` attributes to preserve carriage returns and padding.
