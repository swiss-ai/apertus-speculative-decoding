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

- `smoke-screening-20260913-debug/`: paired baseline vs. 8B `draft_model` depth-3 TP=4 vs. model-free
  n-gram measurement, smoke corpus, concurrency 1 and 8, two independent deployment repeats of every
  arm. Its `README.md` carries the provenance, the headline comparison, and the deviations.
- `correctness/smoke-20260913-debug/`: the sequential greedy captures for all six deployments, the
  original exact-match comparisons, and the re-analysis under the calibrated gate that replaced them,
  including the baseline-vs-baseline control that shows plain vLLM does not reproduce its own greedy
  output across launches. Its `README.md` reports both generations side by side.
- `deployment-failures/`: launches that never reached a serving endpoint, with the log excerpt and
  the design consequence. A failure that removes a factor level from the design is a result.
