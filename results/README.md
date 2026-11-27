# Results

Commit canonical raw result directories here. A cell contains:

- `metadata.json`: immutable configuration, timestamps, corpus digest, and provenance;
- `requests.jsonl`: one client-observed measurement per request;
- `summary.json`: cell aggregates and speculative-decoding counter deltas;
- `metrics_before.prom` and `metrics_after.prom`: raw vLLM metric snapshots.

Use a new top-level run directory for every deployed variant. Never overwrite a completed cell.
Secrets and bearer tokens are not written by the harness.

Correctness artifacts live under `correctness/<run>/`: one greedy capture per deployment, keeping the
full output text, plus the comparisons computed over them. Keep the captures. A comparison is a
derived file and a new criterion is a new file next to the old one, never an edit of it — the
exact-match comparisons recorded on 2026-09-13 are still present alongside the calibrated re-analysis
of the same captures. A comparison of the baseline against a speculative deployment establishes
nothing on its own: it needs a same-configuration calibration control recorded in the same directory,
because independently launched deployments do not reproduce each other's greedy tokens. The criterion
and both control kinds are specified in `../docs/protocol.md`.

## Recorded runs

- **8B deployment and usage.** [Controlled and live k=3/5/7](../docs/dspark-e10-k-headline.md),
  [per-task speed](../docs/dspark-e10-task-speed.md), and the run records under
  [speed/](8b/speed/), [usage-demo/](8b/usage-demo/), and [kubernetes/](8b/kubernetes/).
- **Domain drafters.** [DSpark](../docs/domain-drafters-dspark.md) and
  [EAGLE](../docs/domain-drafters-eagle.md) launch records, with the
  [shared corpus manifest and submissions](8b/domain-drafters/).
- **70B DSpark training on Beverin.** [Feasibility and runtime comparisons](../docs/beverin-feasibility.md)
  and [checkpoint and evaluation provenance](70b/dspark/).
- **GPT-OSS-20B.** [Thinking-effort protocol](../docs/thinking-effort-execution-plan.md)
  and [fine-tune, serving and scratch smoke records](gpt-oss-20b/thinking-effort/).
- **GLM-5.3-Flash.** [Run accounting and the location of the Clariden artifacts](../docs/glm53-flash-specdec.md#run-record).

- `qwen3-235b/`: plain versus EAGLE3 (RedHat speculator, k3 and k5) for
  Qwen3-235B-A22B-Instruct-2507-FP8 on one GH200 node at TP4 (jobs
  4004146 and 4004250-4004252). k3 clears the +10% rule at every C up to 128, and k5
  loses at C=128. See `qwen3-235b/README.md` and `../docs/qwen3-235b-specdec.md`.
- `8b/workload-shape/full-20261122-4004306/`: per-token target entropy and
  TV/KL/greedy match of DSpark e10 and EAGLE e10 on the 28-slice task-speed
  suite (teacher-forced on the target's greedy trajectories, job 4004306), with
  per-slice quantiles, a greedy-chain replay that reproduces the served k=3
  counters, and the analysis of `../docs/workload-distribution-shape.md`. The
  captures and per-token shards stay on Clariden; their hashes are in
  `cluster-only-sha256.txt`.
- `qwen3-4b/workload-shape/`: the workload-shape study repeated on Qwen3-4B with
  Red Hat's DSpark speculator (`../docs/workload-distribution-shape-qwen3-4b.md`).
  `full-20261127-4052179/` holds the replay (job 4052179) with its analyses,
  `gate-20261127-4052199/` the served vLLM cells of the replay-vs-served gate
  (k=3/k=7, greedy and T=1; job 4052199), and `models/` the pinned download
  hashes, configs and the drafter's `train_command.txt`. Captures, npz shards and
  the corpus stay on Clariden (`cluster-only-sha256.txt`).
- `8b/drafter-kv/kv-sparsity-20261022-a/`: two independent DSpark thinking-off
  screening repeats with physical KV-page eviction. The 512-token importance
  policy reduced accounted drafter storage by 69%, but throughput and
  summarization acceptance regressed. Its `README.md` links the frozen gates,
  raw cells, calibrated controls, allocation traces, failed-attempt history and
  reproducibility artifacts. Confirmation and other-head evaluations stopped
  at the negative screen.
- `8b/waste/`: wasted-compute analyses (`decomposition`, `drafter-costs`,
  `cost-per-token`) over the profiled sweeps `8b/sweeps/waste-prof-{a,b}-*` and
  the Nsight smoke test `8b/sweeps/nsys-smoke-*`. `recovery-20261027/` holds the
  paired analysis of the unprofiled summarization pair `8b/sweeps/waste-sum-{a,b}-*`
  (scheduler +1.9%/+2.8% over plain at C=128/256) and its provenance limits.
- `8b/tree-drafters/screen-20261024/`: offline multi-drafter tree screen (jobs
  3737472/3737473). Raw per-shard summaries, logs and the original projection are
  under `raw/`; `analysis-v2/` supersedes the original throughput projections,
  which extrapolated to negative verification time. Restricted captures stay on
  Clariden; their hashes are in `collection/manifest.json`.
- `8b/sweeps/{smoke,dp4-*,tp4-*,think-*,topk-*}-202610*` and
  `8b/layout-recovery-202610{27,28,30,31}/`: serving-layout and sampling sweeps for
  the final 8B drafters. The layout recovery directories hold the raw runs of the
  resumed controllers (`controller-v*/runs/`), their gates and submissions, and
  the CPU reviews; `layout-recovery-20261031/README.md` is the closure, including
  11 rejected P/D reliability cells.
- `8b/sampling/recovery-20261027/`: validated paired analysis (`analysis-v2/`)
  and verdict tables (`verdict-v1/`) of the `topk-off` and `topk-think` sweeps.
- `8b/drafter-updates/`: online drafter adaptation. `recovery-20261030-cache/`
  holds the 19 development curves, selection, paired serving-confirmation
  summaries and verdict; earlier `recovery-*` directories keep the failed trace
  attempts the report cites. The confirmation capsules' raw archives contain
  restricted served captures and stay on Clariden; each capsule's
  `removed-restricted-parts.json` gives their hashes and location. The live-swap
  smoke is `8b/sweeps/drafter-swap-smoke-*`.
- `8b/small-drafter/`: shallow DSpark drafter study. Phase-A depth cost and
  ceiling models, the dated recovery directories with submission receipts,
  preserved failures and the strict review, and `recovery-20261031/` with the
  correctness-blocked closure and the retrieved gate reports of job 3804151.
  Truncation sweeps are `8b/sweeps/small-trunc-*`.

- `smoke-screening-20260913-debug/`: paired baseline vs. 8B `draft_model` depth-3 TP=4 vs. model-free
  n-gram measurement, smoke corpus, concurrency 1 and 8, two independent deployment repeats of every
  arm. Its `README.md` carries the provenance, the headline comparison, and the deviations.
- `correctness/smoke-20260913-debug/`: the sequential greedy captures for all six deployments, the
  original exact-match comparisons, and the re-analysis under the calibrated gate that replaced them,
  including the baseline-vs-baseline control that shows plain vLLM does not reproduce its own greedy
  output across launches. Its `README.md` reports both generations side by side.
- `deployment-failures/`: launches that never reached a serving endpoint, with the log excerpt and
  the design consequence. A failure that removes a factor level from the design is a result.
- `8b/dspark-adaptation/`: DSpark e10 repeat of the T5 best adaptation setting (development
  records, verdict, absolute serving cells, confirmation cell archive and sha256 manifest; greedy
  captures, `requests.jsonl` and weights stay on the cluster). See `../docs/dspark-adaptation.md`.
