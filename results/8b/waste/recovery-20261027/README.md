# T1 completed summarization pair: recovery and verdict

Recovered on 27 Oct 2026. Jobs **3737492 / 3737493** finished successfully on
25 Oct. All **48 cells, 17,408 measured requests** succeeded, with no failed
warmups or speculative-counter resets. No new GPU jobs were launched.

The scheduler adds only **1.9% at C=128 and 2.8% at C=256** over plain
serving on this summarization workload. This does **not** establish a
substantial high-load benefit worth another serving mechanism. Retain the
existing low/moderate-load use of short drafts; use plain at saturation
unless a separate latency/capacity requirement is demonstrated.

| C | arm | output tok/s | GPU-s / million output tokens | paired throughput / plain |
| --- | --- | ---: | ---: | ---: |
| 128 | plain | 1,862 | 537.2 | 1.000 |
| 128 | DSpark k=1 | 1,853 | 539.7 | 0.995 (0.982–1.009) |
| 128 | DSpark k=3 | 1,837 | 544.5 | 0.987 (0.967–1.007) |
| 128 | paper scheduler, cut + KV | 1,897 | 527.1 | 1.019 (1.014–1.024) |
| 256 | plain | 1,862 | 537.1 | 1.000 |
| 256 | DSpark k=1 | 1,874 | 533.5 | 1.007 (1.005–1.008) |
| 256 | DSpark k=3 | 1,839 | 543.9 | 0.988 (0.975–1.001) |
| 256 | paper scheduler, cut + KV | 1,913 | 522.7 | 1.028 (1.027–1.028) |

Absolute metrics are arithmetic means of the two allocation-level metrics;
ratios pair each arm with plain from the **same job**, then take the geometric
mean. Parentheses show the two-allocation range, **not a confidence interval**.
There are only two independent allocations; hundreds of requests within a
cell do not supply hundreds of independent throughput replicates.

At C=256, k=1 lowers p95 request TPOT from 100 to 78 ms/token while p95
TTFT rises from 21.0 to 21.9 s and mean running requests falls from 151 to
116. This is a queueing/capacity trade, not a matched-throughput latency
win. At C=1 the scheduler's outputs are 9–18% longer than plain; its token
throughput cannot be read as an equivalent request-latency gain.

Measured module energy at C=128/256 is about 0.293 J/token for plain,
0.280–0.283 for fixed k, and 0.273–0.275 for the scheduler. These are
nvidia-smi module observations, confounded by resident load and other
module activity; they are not isolated GPU energy or a deployment
recommendation. No KV-memory saving is inferred from this cost experiment.

The original Nsight attribution remains separate: at k=7, C=128 on the
held-out prompts, about 22% of GPU time and 45–46% of speculative work is
wasted. These recovered serving cells are **unprofiled** and do not replace
or extend those profiler estimates. See [the complete T1 report](../../../../docs/waste-profiling.md).

## Artifacts and provenance

Canonical source runs, copied without changing completed cells:

- [waste-sum-a-20261024T133232Z](../../sweeps/waste-sum-a-20261024T133232Z/), job 3737492.
- [waste-sum-b-20261024T133235Z](../../sweeps/waste-sum-b-20261024T133235Z/), job 3737493.

Each preserves its job script/output, sweep specification, exact environment,
patch path list, arm validation/status, full engine logs, metadata, raw
requests, summaries and raw Prometheus snapshots/timeseries. Eight engine
logs are stored as deterministic `replica.out.gz`; their decompressed bytes
match the original files. [collection-verification.json](collection-verification.json)
records both original and stored SHA256 values. No weights or source corpus
were duplicated.

[source-recovery/collection-manifest.json](source-recovery/collection-manifest.json)
records source paths and SHA256 for **346 canonical files + 47 recovery
evidence files**. All 393 were verified locally. The unchanged source
archive is on Clariden at:

```text
/iopsstor/scratch/cscs/faruk_zahiragic/spec-exp-data/waste-profiling/recovery-20261027/completed-summarization-source-v1.tar.gz
SHA256 950c93d1db076177ac26aee7bd4a9e06e5120292c771ec24a86a83bc154b0255
```

The original cell metadata records harness revision
`a26ba85babb61fbd5522fe10eab2721a8f32f23b`, which predates these runs.
There is no contemporaneous top-level Git/image-digest receipt in these
sweep directories. That limitation cannot be repaired retrospectively.
Original metadata is retained unchanged. The manifest's recovery revision
`c4d60795bddec2479d0b00ce7c750991c2412021`, source patches, resolved overlay
files, target contract and scheduler configuration identify what was present
at recovery time, **not proven as-run identities**. The manifest keeps their
paths and SHA256 values; the byte copies (recovery-time repository files,
`serving/patches/` copies identical to the committed patches, resolved vLLM
overlay files and an unrelated run's identity files) were dropped from the
repository on merge and remain in the Clariden archive above and in branch
`waste-profiling` at `412eb2fc`.
Consequently close differences are descriptive, and these runs should not
be published as fully receipt-backed reproductions.

## Analysis and validation

[analysis-v2/summary.json](analysis-v2/summary.json) contains all 48 absolute
measurements and matched ratios at C=1,8,32,64,128,256. CSV tables are in
[absolute.csv](analysis-v2/absolute.csv) and [paired.csv](analysis-v2/paired.csv).
[validation.json](analysis-v2/validation.json) and
[input-sha256.json](analysis-v2/input-sha256.json) retain cell validation and
input hashes. The analysis checks source hashes (including decompressed
logs), arm statuses, request/token totals, zero failures, matched workload
hash/generation/prompt order, and monotone speculative counters. Its source
and shared cost-module hashes are recorded in
[analysis-provenance.json](analysis-v2/analysis-provenance.json).

Recompute into a **new** directory from the repository root:

```bash
PYTHONPATH=src .venv/bin/python results/8b/waste/recovery-20261027/analyze.py \
  --output results/8b/waste/recovery-20261027/analysis-v3
```

`analysis-v2` is the second execution; the first (`analysis-v1`) had
identical metric files and was dropped on merge together with the
Clariden-specific collector `collect_remote.py`. Both, and the exact executed
v1 sources, remain in branch `waste-profiling` at `412eb2fc`. Earlier
`results/8b/waste/{decomposition,drafter-costs,cost-per-token}*` outputs are
unchanged. The local `SHA256SUMS` covers all stored recovery and raw files.

Validation: `tests/test_waste_report.py` passes. No serving implementation changed.
