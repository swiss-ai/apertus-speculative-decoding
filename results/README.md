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

- `smoke-screening-20260913-debug/`: first paired baseline vs. 8B `draft_model` depth-3 TP=4
  measurement, smoke corpus, concurrency 1 and 8, one deployment repeat per arm. Its `README.md`
  carries the provenance, the headline comparison, and the deviations.
- `correctness/smoke-20260913-debug/`: the sequential greedy captures for the baseline and both draft
  deployments, the original exact-match comparisons, and the re-analysis under the calibrated gate
  that replaced them. Its `README.md` reports both generations side by side.
- `deployment-failures/`: launches that never reached a serving endpoint, with the log excerpt and
  the design consequence. A failure that removes a factor level from the design is a result.
