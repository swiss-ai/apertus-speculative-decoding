# Beverin rocBLAS scaling pilots, 2026-10-27

User-authorized follow-up to the fixed two-node pilot 694470. The ongoing
one-node production run and every Clariden job remain unchanged.

## Results and current status

- **16 nodes: job 697549 completed**, 09:34:49–11:19:03 UTC. Saved at 100,
  resumed at 100 (next metric 100), saved periodically at 128, saved/exited
  at 200. All verification checks passed, including LR continuity and every
  node confirming unprofiled rocBLAS. 197 steady samples after excluding
  the first step of each attempt and the step immediately after the
  in-process save: mean **18.036 s**, median **18.2 s**. Exact epoch length
  **8,179 steps** implies **40.98 training hours/epoch**, **655.6 node-hours**,
  or **17.07 training days for ten epochs**. Regular saves add approximately
  **1.31 h/epoch**, based on the three observed saves. Whole-job 10-second
  samples: **340 GiB peak used**, **144 GiB minimum available**. First-save
  window: 330 GiB peak, 156 GiB minimum available. No data-recovery errors or
  nonfinite losses were reported. Full weights/optimizer remain under the
  recorded cluster run path.
- **32 nodes: job 697813 running**, submitted 13:17:22 UTC after the 16-node
  evidence passed review. Same source, context and anchors, 64 trainer
  ranks, 524,288 nominal tokens/step, LR **8.48528137423857e-4**. It uses the
  same save100/resume100/save200 plan, a three-hour allocation with a
  20-minute deadline-save margin and `--nice=10000`.
- The one-node production run is unchanged. A fresh 200-step reference
  (steps 10530–10729, captured 09:53:50 UTC) has mean **14.490 s**, median
  **14.8 s**, implying **569.43 training h/epoch** (23.73 days). Relative to
  this window, 16 nodes give **13.90x epoch speedup**, **86.9% parallel
  efficiency**, and **15.1% more training node-hours**. The earlier 38-sample
  one-node pilot was faster (12.45 s median); reference choice is material.
- The old two-node reference has just five steady samples, including two
  CPU-profiled steps: mean 14.36 s, median 14.8 s, **67,783 steps/epoch**.
  It is a weaker scaling reference than the new unprofiled 16-node pilot.

The first monitoring stream stopped before the 16-node job completed;
completion was discovered on a fresh check at 13:15 UTC. The final raw logs,
audits and memory samples cover the entire job. Monitoring now uses
independent status snapshots. No larger production chain was submitted.

All pilots retain the ten-epoch schedule, full 8,192-token contexts and 512
anchors per rank. Early stopping preserves the LR horizon; it does not use
`--max-steps`. Batch-scaled LR values are pilot heuristics; training-quality
and bitwise RNG equivalence are not established by these checks.

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

Times in launcher/watch records are UTC. Native trainer/vLLM logs and
`sacct.txt` use cluster local time (CET, UTC+1 on 2026-10-27). Each collected
cell has `SOURCE_SHA256SUMS`, verified against every recorded cluster file.
`verify_pilot.py` reproduces the cursor, LR, audit and runtime checks.
