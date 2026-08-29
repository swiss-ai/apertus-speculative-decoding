# Results

Commit canonical raw result directories here. A cell contains:

- `metadata.json`: immutable configuration, timestamps, corpus digest, and provenance;
- `requests.jsonl`: one client-observed measurement per request;
- `summary.json`: cell aggregates and speculative-decoding counter deltas;
- `metrics_before.prom` and `metrics_after.prom`: raw vLLM metric snapshots.

Use a new top-level run directory for every deployed variant. Never overwrite a completed cell.
Secrets and bearer tokens are not written by the harness.
