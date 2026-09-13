# Results

Commit canonical raw result directories here. A cell contains:

- `metadata.json`: immutable configuration, timestamps, corpus digest, and provenance;
- `requests.jsonl`: one client-observed measurement per request;
- `summary.json`: cell aggregates and speculative-decoding counter deltas;
- `metrics_before.prom` and `metrics_after.prom`: raw vLLM metric snapshots.

Use a new top-level run directory for every deployed variant. Never overwrite a completed cell.
Secrets and bearer tokens are not written by the harness.

## Recorded runs

- `smoke-screening-20260913-debug/`: first paired baseline vs. 8B `draft_model` depth-3 TP=4
  measurement, smoke corpus, concurrency 1 and 8, one deployment repeat per arm. Its `README.md`
  carries the provenance, the headline comparison, and the deviations.
- `correctness/smoke-20260913-debug/`: the sequential greedy captures and comparison for that pair.
- `deployment-failures/`: launches that never reached a serving endpoint, with the log excerpt and
  the design consequence. A failure that removes a factor level from the design is a result.
