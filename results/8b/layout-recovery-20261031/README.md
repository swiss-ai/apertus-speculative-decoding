# Final bounded layout collection and CPU review

Status: **original bounded layout screening recovery complete**, with 11 new
P/D reliability exclusions and no implementation promotion. No additional GPU
work or repeats were used after the original grids finished. Broader study,
open-loop validation and delivery item #6 remain open.

The question was whether final 8B drafters materially improve useful serving
performance across DP4, TP4, 2×2 and P/D, after recovering unfinished deployments.
The answer is load-specific speculation, with material costs; this grid does
not establish a P/D adoption advantage or a blanket layout change.

- TP4 k7 at C1024 gives 1.44–1.60x off / 2.22–2.64x thinking-on paired throughput,
  but regresses Arena-Hard 28–45% at C128. OPB DSpark-off p95 falls 62.0→47.8s
  for 289.8→361.9GiB resident/node and 1.936M→1.439M logical KV; thinking-on
  p95 falls 151.4→68.3s for the same ~72GiB overhead. Conditional candidate only.
- TP2DP2 k3 at C128 gives 44–50% throughput gains. OPB p95 falls 7.41→6.09s
  DSpark / 5.39s EAGLE, resident 319.4→330.4/330.9GiB. At C1024 gains are 4–14%
  and do not establish a saturated-layout advantage. k5's very-low-load DSpark
  saving of ~0.2s costs ~10–15GiB additional resident memory; no general promotion.
- P/D is not worth implementing for this decode-heavy grid. Clean Arena C1024
  plain DP4 gives 28.96k tok/s, 43.55s p95; 1p3d DSpark gives 20.10k, 68.39s,
  saving only ~3GiB/node. Low-load drafter gains versus plain DP4 do not isolate
  disaggregation's value over speculative DP4. Preserve all 11 reliability failures.

Primary evidence:

- [Qualified verdict, exact selected comparisons and 11 exclusions](qualified-findings.json)
- [Full absolute metrics and both baseline types](final-review-v1/comparisons.csv)
- [All 1040 cells and exclusion reasons](final-review-v1/cells.csv)
- [Hashes, request checks, memory/metric scopes and scientific limits](final-review-v1/review.json)
- [Short review index](final-review-v1/review.md)
- [Complete original/stored file hashes](controller-v5-collection.json)
- [Final 75-file source verification](final-source-verification.json)
- [Context and interpretation](../../../docs/high-concurrency-8b.md)

The strict report has 1040 unique cells, 803 comparisons, 743 eligible screening
pairs, no missing analyses and 1040 passing raw usage re-sums. Original failed
arms, unhealthy allocations and preflight-only cells remain present/excluded.
The 11 new P/D exclusions are additional to historical invalid attempts. The
frozen helper intentionally leaves `scientific_state` pending; qualified-findings.json
records the completed bounded screening review separately. Calibrated output
correctness, deployment-repeat confidence, open-loop sustainable capacity and
independent load-generator sufficiency remain unestablished by this recovery.

- TP2DP2 job 3804065: COMPLETED 9146s; 120/120 cells, all gates pass.
- Thinking-off P/D job 3806226: COMPLETED 11896s; 147/147 cells, nine cells below
  the unchanged 99% success threshold. Infrastructure/provenance/counter gates
  do not fail. Two earlier thinking-on P/D exclusions remain: 11 total.
- PD-off rejected requests contain 239 HTTP500 and 115 connection ReadError
  failures across the nine cells (354 failures). Proxy logs show upstream
  prefill `client.post` and decode `client.send` ReadError. Exact transport cause
  is unestablished; neither a benchmark bug nor compute-capacity limit is proven.
  See pd-off-rejection-signatures.txt. No retry for a passing point is planned.

## Preserved source and collection

Engine source stays branch high-concurrency-8b, base
`efad15a10609e615c5c8635be3c243d1ec5b9da7` plus guarded source overlay SHA256
`51fdd333056fedaeafeb7cbc1df9c33bc71f24b84cdeb09e31bd98ca48ec9a8d`.
Final verification rechecked all 75 source files with zero mismatches. No active/frozen engine code,
precision, sampler, workload or success gate changes in this CPU closure.

Remote registry:
`/iopsstor/scratch/cscs/faruk_zahiragic/apertus-spec-decoding-highconc/results/8b/layout-recovery-20261030/controller-v5/state.json`.
Final remote campaign state is
`measurements_finished_with_rejected_cells_pending_scientific_review`.
Slurm terminal state does not imply scientific implementation readiness.

collect_controller_v5.py streams the entire terminal controller-v5 tree to its
corresponding local results path. Original size inspected before transfer:
2,131,469,278 bytes, no symlinks. Requests/logs/CSV use lossless gzip; the completed
controller-v5-collection.json records original/stored hashes, bytes,
mode/mtime where available and transfer status for every file. No weights/corpora are copied.
The stream never overwrites existing local files. The final manifest closes
successfully: 1,810 files, 2,131,469,278 original bytes stored as 637,697,020
lossless bytes; all 267 remaining summaries are present.

Lustre intermittently blocks remote tar in cl_sync_io_wait on even small source
metadata files. The first archive stream failed at the next tar header after 107 summaries.
collection-attempt1-failure.json preserves the exact error and evidence limits.
All finished member byte hashes were reconstructed (gzip CRC checked), with
original mode/mtime explicitly unavailable for this interrupted legacy collector.
resume_controller_v5.py now excludes those completed source paths, preserves
them in place, and journals each newly completed file before the next header.
Partial v2 writes use separate names; no finished file is overwritten.
The source EOF cause remains unestablished; Lustre waiting was directly observed.
Latest bounded observation and process IDs are in the persistent receipt:
`/home/projects/apertus-speculative-decoding-recovery-control/receipts/layouts.json`.

## Preserved CPU continuation

The completed local finish_cpu_review.sh waited for the resumed collector, then:

1. Requires a successful complete manifest with exact inspected byte total,
   267 summaries and unique source paths.
2. Runs the previously frozen prepare_final_review.py on exactly the 120 TP2DP2
   and 147 PD-off cells, retaining request-only false gates.
3. Runs unchanged review-helper-v1/review.py with --verify-requests, combining
   these 267 cells with the 773 reviewed historical/v2 cells into fresh outputs.
4. Requires 1040 unique cells/no missing analyses and writes
   cpu-review-completion.json. No automatic implementation verdict/publication.

Paths: remaining-review-inputs-v1/remaining-analysis.json, final-review-v1/
(review.json, review.md, cells.csv, comparisons.csv), final-cpu-review-v2.log.
The earlier final-cpu-review.log preserves the failed-input guard.
The frozen helper's conservative scientific_state remains pending by design;
a separate qualified human-reviewed closure must explicitly assess all gates.

The qualified review assesses absolute throughput and latency, successes,
preemptions, resident memory coverage, logical KV pools and declared P/D scopes.
Losing strata, small gains and 11 reliability exclusions remain visible. Only
substantial repeatable benefits warrant implementation language. Original deployment-repeat
confidence, calibrated output correctness, open-loop capacity and independent
load-generator gates remain explicit limits. The review supports the qualified screening verdict above. Publication retains
these limits, with no blanket promotion and item #6 remaining open.
