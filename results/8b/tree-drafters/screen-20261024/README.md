# Recovered tree-drafter screen

**Decision: do not implement multi-drafter serving from these results.**
The low-load acceptance gate versus chains passes, but the best graft adds
less than 0.1% projected throughput over a single DSpark tree. Actually
running both neural drafters is slower. Single-DSpark-tree low-load headroom
remains a decode-only prediction, not a measured serving result.

The report is [docs/tree-drafters.md](../../../../docs/tree-drafters.md).
[Corrected comparisons](analysis-v2/comparisons.csv),
[gate and sensitivity checks](analysis-v2/summary.json), and
[full corrected projection](analysis-v2/tree-sim.json) preserve the result.

## Original jobs and immutable evidence

- Validation **3737472**, completed 25 October 2026, 221 seconds on nid007618.
- Full candidate/replay **3737473**, completed 26 October, 411 seconds on
  nid007598: 128 OPB + 128 chat sequences, 86,844 candidate anchors.
- Both ran source `d69a9ac3ee5087afd20d39a4d7a1bf3b3bcac365` on the pinned
  vLLM 0.23.1rc1 image. The original provenance JSON, image digest, resolved
  environment, exact submit scripts, source patch, per-shard summaries and
  logs are under `raw/<original-run-name>/`. Slurm times in
  [sacct-local-time.psv](collection/sacct-local-time.psv) are cluster-local
  CET; the original provenance timestamps are UTC.
- Captures were reused from job 3730086, source `3f3f734c`, as recorded in
  their `SOURCE.txt`. Each candidate summary pins capture/prompt digests,
  target revision/config/tokenizer/template/weight-manifest hashes,
  drafter weight hashes, auxiliary layers and all candidate-builder args.
- [Collection manifest](collection/manifest.json) verifies **64 source files,
  77,637,642 bytes** against Clariden SHA-256 values. Original logs and
  summaries are unmodified. No weights or source corpus were copied.
- Restricted captures and token-level candidate shards are retained locally
  but ignored by Git; the same exact bytes remain at
  `/iopsstor/scratch/cscs/faruk_zahiragic/spec-exp-data/tree-drafters/runs/`.
  Their paths, lengths and hashes are committed in the collection manifest.
  Restore them with the recorded `clariden get --recursive` commands to
  replay candidate selection. They are not needed for cost-only reprojection.
- Supplemental target/drafter configs and EAGLE export report are in
  `inputs/`, collected on 27 October. Their hashes are recorded separately
  from the original as-run identities. The cost table is an exact copy of
  the one named in the original report, not a new measurement.

## Invalidated predictions and repair

`raw/full-20261024T1000Z/tree-sim.json` is preserved as the original output.
Its **throughput projections are superseded**, while its candidate replay
and acceptance aggregates are unchanged. The original adjacent-sample
extrapolation produced 48 negative verification-time points (minimum
-71.779 ms), including -18.339 ms at C=256 / 32 draft nodes. The apparent
13.37x OPB speedup is invalid.

Code fix `806d3be8` uses a cumulative-maximum timing envelope and a
nonnegative least-squares tail slope over the upper half of the measured
range: 31.52 us/token. Out-of-range predictions are flagged. Best selected
chain/single-tree/graft settings all remain within 1–2,048 measured batch
tokens, and their choices survive 0.5x/1.5x tail-slope sensitivity checks.
The original `sequences.all=0` bookkeeping error is also corrected to 256.

This does not validate tree-attention kernels, extra KV residency, prefill,
metadata processing or continuous-batching behavior. There is one greedy
capture per workload and no deployment-level confidence interval.

## Reproduce without another GPU run

From repository root with the harness environment, use a **new** analysis
directory. The commands below show how `analysis-v2/` was produced; output
files deliberately refuse to overwrite a completed version.

```bash
PYTHONPATH=src python -m apertus_bench.tree_sim \
  --reproject results/8b/tree-drafters/screen-20261024/raw/full-20261024T1000Z/tree-sim.json \
  --cost-table results/8b/tree-drafters/screen-20261024/inputs/cost-table.json \
  --output results/8b/tree-drafters/screen-20261024/analysis-v2/tree-sim.json
PYTHONPATH=src python results/8b/tree-drafters/screen-20261024/analysis-v2/analyze.py
```

The second script preserves acceptance aggregates, reports the original G4
thresholds and runs tail-slope sensitivity checks. It uses exact emitted-token
and step counts; other original means retain their four-decimal rounding.
[Analysis provenance](analysis-v2/provenance.json) records argv, Python,
Git revision and source/input hashes. `SHA256SUMS` covers committed artifacts;
the collection manifest additionally covers ignored restricted inputs.

Validation: **17 focused simulator tests pass**, including the actual noisy-tail
regression, explicit extrapolation flags, malformed-table rejection, immutable
reanalysis, and unchanged replay aggregates. Focused Ruff checks pass. No new
GPU job, serving implementation, Slack update or model publication was made.
