# Completed TP4 recovery and independent remaining layouts

31 Oct update: both remaining jobs are terminal (120 TP2DP2 +147 PD-off cells).
Nine additional PD-off reliability rejections are retained. Final collection/CPU
review status and durable continuation are in ../layout-recovery-20261031/README.md.

Historical status on 2026-10-30: preflight 3782518 and all three full TP4 recoveries
(3789188, 3798373, 3800281) passed their preserved gates. Thinking-on P/D 3800861
completed all 63 cells; two fail the unchanged 99% request-success gate and remain
rejected. The original independent TP2DP2 k3/k5 comparison is submitted as
job 3804065. Original thinking-off P/D k3 follows under the same durable owner.
No valid finished stage was rerun and no grid/model/workload was added.

The strict CPU view is completed-v2-review-v1/: 773 historical/recovered cells,
497 comparisons,455 eligible screening pairs. It rechecks raw successful-request
usage sums and exposes absolute latency, throughput, GPU resident memory,
logical cache capacity, scopes and exclusions. completed-v2-findings.json is a
compact index. See ../../../docs/high-concurrency-8b.md for the current
qualified verdict. Large C1024 TP4 gains coexist with C128 regressions, substantial
resident-memory overhead and reduced KV capacity; no implementation promotion
is inferred from one deployment or a minute gain.

## Preserved failed reliability cells

- Plain 1p3d P/D, contaminated-thinking C256: 1019/1032 successes (98.740%):
  7 HTTP500 and6 ReadError failures.
- DSpark-on k3 P/D, contaminated-thinking C1024: 4085/4128 (98.958%):
  28 HTTP500 and15 ReadError failures.

Proxy logs show upstream decode client.send raising httpx.ReadError before
response headers. There is no demonstrated accounting defect or proven compute
saturation cause. Background _report_usage_worker disk-quota errors and optional
nixl_ep import warnings are separate observations, not a claimed causal repair.
The two cells are retained/excluded; their gates were not changed to passing
and they were not retried for a pass. Independent original layouts do not depend
on a passing P/D operating point.

controller-v2-collection.json records all 1426 files: 3,181,270,517 original bytes
stored as 1,097,887,773 lossless bytes. Originals and compressed representations
have separate hashes; no completed raw result was overwritten. Source paths,
mode/mtime, all metadata, requests, snapshots/timeseries, logs, environment,
patches, checkpoint validation and Slurm provenance are preserved. Weights,
corpora and out-of-tree core dumps were not copied.

## Durable continuation and home-quota remedy

The first private continuation(controller-v3) failed before sbatch: the patch
launcher could not mkdir its HOME .sml overlay because the home quota was full.
It submitted no Slurm job. Its exact state/log/submission/source files and
controller-v3-submission-failure.json are preserved unchanged.

Fresh controller-v5 is detached with nohup/setsid/flock, PID 50602. It uses only
original TP2DP2 then PD-off specifications. Private host runtime overlays and
resolved environment files use controller-v5/runtime/RUN/vllm-patch and sml-env.
VLLM_CONFIG_ROOT uses RUN/vllm-config, the config/usage-reporting root supported
by the pinned image. HOME, shared caches, other checkouts and existing installs
are unchanged. Each actual job retains these resolved paths and source identity.
This fixes a demonstrated launch-write failure; it is not a retry of the P/D
reliability observations.

The host-runtime-only attempt controller-v4/job 3802690 then failed all TP2DP2
server initialization on the HOME FlashInfer sampling.lock quota (FAILED 914s,
zero cells). It is preserved. Controller-v5 adds only the pinned supported
FLASHINFER_WORKSPACE_BASE to RUN/flashinfer-workspace; FlashInfer 0.6.13 appends
.cache/flashinfer below it. Engine/image/precision/sampler settings stay frozen.
No completed valid measurement is rerun.

Registry:
`/iopsstor/scratch/cscs/faruk_zahiragic/apertus-spec-decoding-highconc/results/8b/layout-recovery-20261030/controller-v5/state.json`.
Read top-level job_ids or stages[*].job_id for downstream IDs. The parent receipt
points here and retains prior jobs/failed controllers. Source remains frozen at
base efad15a1 plus overlay SHA256
51fdd333056fedaeafeb7cbc1df9c33bc71f24b84cdeb09e31bd98ca48ec9a8d,
with all 75 files verified. No active engine source was changed. Do not reset its
tracked overlay or sync other commits while jobs run.

The private original-scope wrapper is commit 358ea55d, SHA256
4b831ca38d5c18d242163ab6f21d2abf8a0b3b0e53d029ab1f52f3aa6f8aaa02.
Scratch bootstrap is commit ee15d3b3, SHA256
a73da7bd29be2ffd0c5cb0cf73ef3cc396f867d1eb195ce89e0c72110ca7b10e.
continuation-freeze-v4.json and launch-controller-v5.sh preserve full commands,
branch/revisions and file identity. Every new submit-command.txt also records
SWEEP_SOURCE_BRANCH, SWEEP_CONTROLLER_WRAPPER_SHA, SWEEP_CONTROLLER_BOOTSTRAP_SHA
and SWEEP_CONTROLLER_FREEZE. Five focused continuation/cache checks, Ruff,
source verification and launch shell syntax pass.

Strict gates remain unchanged. Infrastructure/provenance/counter failures stop.
A request-only failure stays a false gate and excluded operating point while
an independent remaining layout can proceed. Final measurement state is
measurements_finished_with_rejected_cells_pending_scientific_review; it never
asserts scientific implementation readiness.

## Final CPU review after remaining jobs finish

Collect controller-v5's lightweight registry/analyses/gates, runtime source
artifacts and each completed stage output into the corresponding local path,
inspecting sizes first and retaining original/stored hashes with any lossless
compression. Use a fresh directory for every derived review. The original frozen
review helper remains unchanged.

The bounded packager below requires both original remaining grids to be terminal
and complete: 120 TP2DP2 cells plus 147 PD-off cells. It refuses missing or
infrastructure-invalid inputs. It retains false reliability gates and source
hashes, including when the source controller intentionally omitted analysis of
a rejected stage. The final review combines these 267 cells with the 773 already
reviewed; request counts are not independent deployment repeats.

```bash
cd /home/projects/apertus-speculative-decoding-highconc
/home/projects/apertus-speculative-decoding/.venv/bin/python \
  results/8b/layout-recovery-20261030/prepare_final_review.py \
  --controller-state results/8b/layout-recovery-20261030/controller-v5/state.json \
  --path-map /iopsstor/scratch/cscs/faruk_zahiragic/apertus-spec-decoding-highconc=/home/projects/apertus-speculative-decoding-highconc \
  --output results/8b/layout-recovery-20261030/remaining-review-inputs-v1
/home/projects/apertus-speculative-decoding/.venv/bin/python \
  results/8b/layout-recovery-20261027/review-helper-v1/review.py \
  --analysis results/8b/layout-recovery-20261030/completed-v2-review-v1/review.json \
  --analysis results/8b/layout-recovery-20261030/remaining-review-inputs-v1/remaining-analysis.json \
  --path-map /iopsstor/scratch/cscs/faruk_zahiragic/apertus-spec-decoding-highconc=/home/projects/apertus-speculative-decoding-highconc \
  --verify-requests \
  --output results/8b/layout-recovery-20261030/final-review-v1
```

Use cells.csv and comparisons.csv for absolute values, matched baselines,
losing strata and declared memory/metric scopes. P/D metrics cover the first
decode instance only. Original calibrated correctness, deployment-repeat
confidence, open-loop capacity and load-generator gates remain explicit limits.
Only substantial repeatable benefit warrants implementation language.
