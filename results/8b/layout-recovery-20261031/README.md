# Final bounded layout collection and CPU review

Status: all original measurements are terminal; final CPU review and publication
remain pending complete evidence collection. No further GPU work is authorized
in this closure. Existing completed/failed artifacts and all false gates remain.

- TP2DP2 job 3804065: COMPLETED 9146s; 120/120 cells, all gates pass.
- Thinking-off P/D job 3806226: COMPLETED 11896s; 147/147 cells, nine cells below
  the unchanged 99% success threshold. Infrastructure/provenance/counter gates
  do not fail. Two earlier thinking-on P/D exclusions remain: 11 total.
- PD-off rejected requests contain 239 HTTP500 and115 connection ReadError
  failures across the nine cells (354 failures). Proxy logs show upstream
  prefill `client.post` and decode `client.send` ReadError. Exact transport cause
  is unestablished; neither a benchmark bug nor compute-capacity limit is proven.
  See pd-off-rejection-signatures.txt. No retry for a passing point is planned.

## Preserved source and collection

Engine source stays branch high-concurrency-8b, base
`efad15a10609e615c5c8635be3c243d1ec5b9da7` plus guarded source overlay SHA256
`51fdd333056fedaeafeb7cbc1df9c33bc71f24b84cdeb09e31bd98ca48ec9a8d`.
All 75 source files were previously checked. No active/frozen engine code,
precision, sampler, workload or success gate changes in this CPU closure.

Remote registry:
`/iopsstor/scratch/cscs/faruk_zahiragic/apertus-spec-decoding-highconc/results/8b/layout-recovery-20261030/controller-v5/state.json`.
Final remote campaign state is
`measurements_finished_with_rejected_cells_pending_scientific_review`.
Slurm terminal state does not imply scientific implementation readiness.

collect_controller_v5.py streams the entire terminal controller-v5 tree to its
corresponding local results path. Original size inspected before transfer:
2,131,469,278 bytes, no symlinks. Requests/logs/CSV use lossless gzip; the completed
controller-v5-collection.json will record original/stored hashes, bytes,
mode/mtime and transfer status for every file. No weights/corpora are copied.
The stream never overwrites existing local files. Until its manifest closes,
this local controller-v5 copy is incomplete and must not be used as final input.

Lustre intermittently blocks remote tar in cl_sync_io_wait on even small source
metadata files. The live stream is preserved; no corrupted/failed copy is known.
Latest bounded observation and process IDs are in the persistent receipt:
`/home/projects/apertus-speculative-decoding-recovery-control/receipts/layouts.json`.

## Durable CPU continuation

Detached local finish_cpu_review.sh waits for the existing collector PID, then:

1. Requires a successful complete manifest with exact inspected byte total,
   267 summaries and unique source paths.
2. Runs the previously frozen prepare_final_review.py on exactly the120 TP2DP2
   and147 PD-off cells, retaining request-only false gates.
3. Runs unchanged review-helper-v1/review.py with --verify-requests, combining
   these267 cells with the773 reviewed historical/v2 cells into fresh outputs.
4. Requires1040 unique cells/no missing analyses and writes
   cpu-review-completion.json. No automatic implementation verdict/publication.

Paths: remaining-review-inputs-v1/remaining-analysis.json, final-review-v1/
(review.json,review.md,cells.csv,comparisons.csv), final-cpu-review.log.
The frozen helper's conservative scientific_state remains pending by design;
a separate qualified human-reviewed closure must explicitly assess all gates.

Once CPU review finishes, inspect absolute throughput and latency, successes,
preemptions, resident memory coverage, logical KV pools and declared P/D scopes.
Keep losing strata, small gains and11 reliability exclusions. Only substantial
repeatable benefits warrant implementation language. Original deployment-repeat
confidence, calibrated output correctness, open-loop capacity and independent
load-generator gates remain explicit limits. Final Slack/canvas publication is
pending concrete review, with no blanket promotion and item#6 remaining open.
