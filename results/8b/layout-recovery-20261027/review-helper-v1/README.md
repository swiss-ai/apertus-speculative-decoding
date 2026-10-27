# Frozen CPU-only layout review helper v1

This standalone helper reads preserved JSON, metrics, GPU samples and bounded
startup-log prefixes. It never submits jobs, starts models, modifies raw cells,
imports active repository modules, or changes the controller. Output directories
must be fresh. `manifest.sha256` pins this helper and its regression tests.

The active remote checkout remains pinned to
`efad15a10609e615c5c8635be3c243d1ec5b9da7`. Do not sync this local artifact
commit into remote highconc while the controller is active. This helper is
staged locally only; it requires no remote source changes and no GPU work.

## What is already preserved and what this adds

Controller analyses contain throughput, full latency distributions, success,
preemptions, KV signals, engine settings, source hashes and request/counter gates.
This helper makes the joins explicit: speculative P/D against the same 1p3d/2p2d
plain P/D, plus each P/D layout against same-job plain DP4. It emits absolute
metrics and deltas, retaining failed or incomplete cells with exclusion reasons.

`cells.csv` includes every cell. `comparisons.csv` contains same-layout and
same-job layout comparisons. `review.json` retains provenance, individual GPU
memory samples' summaries, source scopes, startup capacities and scientific
limitations; `review.md` is a short index. No effect is automatically recommended
for implementation. Gains below 10% remain small effects, and larger gains remain
screening candidates until the original scientific gates are established.

KV-cache capacity in old summary.json is the last logical engine's value. The
helper reads all raw cache-config entries and sums distinct logical engines;
it never multiplies a TP engine's pool by GPU count. P/D metric counters and
capacity cover only the first decode instance. P/D startup capacities are read
from a bounded 1MiB prefix of every role log when unambiguous, with prefill and
decode totals kept separate. The combined role pools are not interchangeable.
Used-KV peak bounds account for unequal engine pools; the old summary peak stays
available as a legacy estimate. P/D-versus-DP4 capacity/preemption deltas with
different scopes are left unavailable.

Resident GPU memory includes preallocated KV pools. GPU CSV timestamps are
interpreted with the arm-status UTC offset; the report exposes coverage and
suppresses resident-memory deltas below 80% coverage. Missing values remain
unavailable. Compare 0.7-budget TP4 recovery pairs within their own allocation;
do not silently pool them with original 0.8-budget deployments.

## Collection and final review, without new measurements

1. Read the durable controller state and obtain every new ID from top-level
   `job_ids` or `stages[*].job_id`. Its absolute path is:
   `/iopsstor/scratch/cscs/faruk_zahiragic/apertus-spec-decoding-highconc/results/8b/layout-recovery-20261027/controller-v1/state.json`.
   A failed or incomplete state is still reviewable, but cannot complete the
   campaign. Final measurement state is
   `measurements_complete_pending_scientific_review`.
2. Copy state.json, all analysis/gate/submit JSON, frozen specs, launch receipt
   and controller log, preserving their controller-v1 directory structure.
   For every `stages[*].output`, inspect sizes before transfer, then collect one
   finished output directory at a time into the corresponding local path.
   Preserve metadata, summaries, requests, metric snapshots/timeseries, role
   logs, GPU samples, image digest, environment, patches and Slurm provenance.
   Do not collect weights, corpora or repository-root core dumps. Lossless gzip
   of requests/logs is supported; record original/stored SHA256 hashes and
   never overwrite an earlier raw cell. Avoid a whole-campaign temporary archive
   competing with the local disk budget.
3. Run the command below locally after collection. Path mapping resolves the
   absolute remote run paths recorded in controller analyses to their collected
   local equivalents. Use a new --output for each review. The optional
   --verify-requests pass recomputes exact successful-request usage totals and
   prompt-length min/median/p95/max from raw requests; it is bounded CPU work
   and should run locally, not on the Clariden login host. Historical usage
   totals were already independently checked in historical-analysis.json.

```bash
cd /home/projects/apertus-speculative-decoding-highconc
cd results/8b/layout-recovery-20261027/review-helper-v1
sha256sum -c manifest.sha256
cd /home/projects/apertus-speculative-decoding-highconc
/home/projects/apertus-speculative-decoding/.venv/bin/python \
  results/8b/layout-recovery-20261027/review-helper-v1/review.py \
  --analysis results/8b/layout-recovery-20261027/historical-analysis.json \
  --controller-state results/8b/layout-recovery-20261027/controller-v1/state.json \
  --path-map /iopsstor/scratch/cscs/faruk_zahiragic/apertus-spec-decoding-highconc=/home/projects/apertus-speculative-decoding-highconc \
  --verify-requests \
  --output results/8b/layout-recovery-20261027/final-review-v1
```

4. Check missing_stage_analyses, exclusion reasons, matching corpus/generation/
   prompts, failed cells and baseline-health gates before reading speedups.
   Inspect cells.csv/comparisons.csv for absolute throughput, request latency,
   completion volumes, resident memory, logical KV pools and preemptions.
   Preserve losing workload/concurrency strata; do not average them away.
5. The original deployment-repeat confidence interval, calibrated correctness,
   open-loop SLO and independent load-generator-capacity gates cannot be inferred
   from successful fixed-concurrency requests. Their absence is explicit in the
   report. A screening verdict can be written from the preserved outputs without
   further GPU work; implementation or sustainable-capacity claims remain
   unestablished where those gates have no evidence.

The historical-only output historical-review-v2 uses this final frozen helper.
The earlier historical-review-v1 preserves its exact prototype helper source;
it remains an earlier derived artifact, not an overwritten comparison.
