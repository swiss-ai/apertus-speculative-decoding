# DSpark drafter KV sparsity: negative validation screen

The implemented sparse backend physically frees drafter pages, but fails the
agreed performance and acceptance gates on the existing thinking-off head.
Keep the native 2,048-token drafter window for this head. No candidate qualified
for confirmation, DSpark thinking-on, or EAGLE thinking-off evaluation.

The work is on `experiment/drafter-kv-sparsity`, branched from main
`c15baf59aa35c10c9f36298695eba6078294b0a7`. Serving used
`c6675e217f9bc69352c5d5bcb273824a0f437390`; analysis used `27c3879b` plus the
hashed reporting scripts preserved here. The original checkout was left intact.

## Measurements and decision

Two independent GH200 node allocations, with swapped GPU assignments, compared
native-2048, recent-512, importance-1024, and importance-512. Each deployment
served validation chat, code, and summarization at concurrency 1, 32, and 256.
Every cell used whole prompt passes: 64, 128, and 1,024 measured requests,
respectively. All **29,184 measured requests in 72 cells succeeded**, and no
warmup request failed. Test and OPB-heldout inference were not run.

Ratios below are candidate/native. Throughput is the geometric mean of paired
node ratios across all three workloads at C=32/256. Acceptance is the worst
summarization cell across C=1/32/256. Storage is the conservative worst matched
steady-state ratio, including selection metadata.

| Policy | Throughput ratio, C32/256 | Summarization acceptance ratio | Drafter storage ratio | Combined target/draft storage ratio | Expansion gate |
| --- | ---: | ---: | ---: | ---: | --- |
| Importance-1024 | 0.8336 | 0.9473 | 0.5840 | 0.97004 | Fail |
| Importance-512 | 0.8315 | 0.8936 | 0.3082 | 0.95002 | Fail |
| Recent-512, diagnostic | 0.8299 | 0.8661 | 0.3082 | 0.94698 | Fail |

Every sparse policy regressed throughput by more than 3% and exceeded the p95
latency bound in every paired workload/concurrency cell. Importance-512 reduced
accounted drafter storage by 69.18%, but lost about 10% summarization acceptance.
Importance-1024 reduced storage by 41.60%, below the required 45%, and its C1
summarization acceptance was below 95% of native. Recent-512 was diagnostic and
was never eligible for selection. Its combined-storage improvement cannot
qualify it while performance and acceptance fail.

The importance-512 combined ratio is **0.9500209073**, just above the 0.95
capacity threshold; rounding it to 0.95 would change the decision incorrectly.
There is no speed or serving-capacity claim. Two allocations support this
descriptive screen; no confidence interval is inferred from two node blocks.

All three policies pass the repository's calibrated greedy-divergence controls.
Each deployment has two six-prompt captures. Native within-deployment controls
agree on 6/6 prompts, while cross-node native controls agree exactly on 5/6.
These small calibrated controls do not prove losslessness. The gate JSON's
top-level correctness flag refers only to a selected candidate; actual passing
per-policy flags are in `decisions` even though `candidate` is null.

Recorded preemptions across measured cells were native 28, importance-1024 22,
importance-512 9, and recent-512 11. They remain part of the serving measurements.

## Interpretation and physical accounting

The [design and literature review](../../../../docs/drafter-kv-sparsity.md)
explain the page-level H2O/SnapKV-inspired policy, MagicDec precedent, and the
canvas's DSpark/MineDraft references. Importance accumulates verified-anchor
attention mass across all five drafter layers, protects the latest 128 context
tokens, and prunes within the existing DSpark window. Original positions and
target attention are preserved. Selection physically releases pages; it does
not keep a full backing history.

This is a result about the **implemented backend**. In the C256 chat stress
control, the custom unpruned backend delivered 85.74% of native throughput while
retaining 100.90% of native acceptance. It therefore has substantial overhead
before eviction. It uses a custom attention path, host bookkeeping, and bounded
query CUDA graphs instead of the native whole-drafter compilation path. The
screen does not isolate an eviction-only speed effect or rule out a better
implementation. The target's native compilation path remains in use.

Storage matches allocator ownership and worker snapshots from the same output,
requiring all owners to fill the native window and matching 16-token length
buckets and target physical ownership. Native ownership above 130 pages per
request is treated as prefill/bootstrap, not the steady-state denominator.
There are 146 matched strata for importance-1024, 152 for importance-512, and
142 for recent-512 across both nodes. All policies have positive physical
release events. Excluded samples and reasons are retained in the memory report.

Selection metadata peaks at 7,221,248 bytes for importance-1024 and 6,041,600
bytes for the 512-token policies. Runtime-reported query and context-projection
scratch estimates peak at 36,700,160 and 304,087,040 bytes, respectively; these
are separate from persistent drafter storage and are not allocator peak-memory
measurements. Bootstrap is also separate. A shared pool's
reservation or `nvidia-smi` usage is not the storage denominator, and these
results do not demonstrate lower total GPU allocation.

## Artifacts and reproduction

- [Full paired tables](analysis/screen-report-v1/report.md),
  [absolute measurements](analysis/screen-report-v1/absolute.csv),
  [paired ratios](analysis/screen-report-v1/paired-ratios.csv), and
  [structured report](analysis/screen-report-v1/summary.json).
- [Frozen gate](analysis/screen-gate-v1.json),
  [evidence](analysis/screen-v1/evidence.json), and
  [correctness controls](analysis/screen-v1/correctness-controls.json).
- Each allocation retains the exact command, spec, source/overlay hashes,
  applied patch, resolved container environment, logs, deployment identities,
  and raw cell metadata/requests/summaries/metric snapshots.
- [Input manifest](inputs/manifest.json), frozen target template, original hashed
  head inventory, and `provenance-supplement/off-inputs-artifacts/` retain the
  target contract and existing training provenance. The container SHA-256 is
  `f5fc017ff794fa8d80b73dd9a788028a1872dbfc2e21c0de043fb7738c11a0b5`.
- [Slurm accounting in UTC](slurm-accounting-iso.psv) and its command retain job
  IDs, nodes, elapsed seconds, resources, and terminal states.

The full derived `memory.json` is preserved byte-for-byte in `memory.json.gz`
to keep the committed artifact below Git hosting's single-file size limit.
Decompress it before following evidence paths. The uncompressed local copy is
ignored. Source workload text follows the repository's existing restriction:
exact validation and six-prompt correctness copies remain on the cluster and
locally, with their hashes committed; the source text itself is ignored. Raw
measured request outputs and correctness captures are committed.

The executed [analysis runner](analysis/run-analysis-v1.py) records exact argv,
environment path mapping, source hashes, revision, package versions, logs and
exit codes in [pipeline provenance](analysis/pipeline-provenance-v1.json).
It maps collected cluster paths without rewriting any raw spec or metadata.
For a new analysis, use a fresh output version: the scripts deliberately refuse
to overwrite completed derived results. For replaying the current analysis on a
fresh checkout, restore the hashed correctness workload at its corresponding
path, decompress the memory artifact when reading existing reports, install
PyYAML and the harness dependencies, and use the path mapping recorded in the
pipeline provenance. Existing absolute report paths refer to this checkout.

## Runtime history and budget

All attempts remain separate. Failed attempts are excluded from decision cells
and included in allocated GPU-hour accounting.

| Job | Attempt | State | Allocated seconds | Note |
| --- | --- | --- | ---: | --- |
| 3723627 | Hash/kernel preflight | Complete | 56 | Source, image/head hashes, dense reference |
| 3723725 | Runtime-01 | Failed | 268 | Triton constants/config registry; fixed |
| 3723801 | Graphs-01 | Cancelled pending | 0 | Replaced before allocation |
| 3723803 | Graphs-02 | Failed | 11 | Long container mount paths; shortened |
| 3723844 | Graphs-03 | Failed | 53 | Unresolved compilation test fixture; fixed |
| 3723858 | Runtime-04 | Complete | 304 | Native/unpruned/sparse controls and physical release |
| 3725102 | Screen-01 | Cancelled after failure | 625 | C256 exposed int32 physical-cache address overflow |
| 3725104 | Screen-02 | Cancelled pending | 0 | Avoided running the same bad kernel |
| 3725531 | Int64 stress | Complete | 356 | Large-offset regression and 4 x 1,024 C256 requests |
| 3725591 | Screen-01-r2 | Complete | 792 | Final node block 1, nid007657 |
| 3725593 | Screen-02-r2 | Complete | 782 | Final node block 2, nid006078 |

The address bug was fixed by converting physical page IDs to int64 before
stride multiplication. A GPU regression covers IDs 65,535/65,536/65,537 and a
separate large-stride layout beyond 2^31 element offsets. Final screens use the
fixed source. Earlier tiny runtime controls are diagnostic and are not pooled
with validation cells. Early harness metadata inherited an old working-directory
Git SHA; their actual code is identified by launch provenance and overlay
hashes. The working directory was fixed before the final stress and screens.

Total consumption is **3.60777778 allocated GPU-hours out of 72**, counting all
four GPUs on each allocated node, failures, and idle time. Preflight used
1.16444444 and screening 2.44333333. Every recorded job is terminal. No usage
limit was reported. No further runs were launched after the negative screen.

Validation: 104 focused CPU tests pass; runtime/harness lint and shell syntax
checks pass. GPU reference, state/selection, graph replay, configuration,
sampling, and large-address regressions pass in the recorded preflights.

Possible follow-ups, requiring separate approval, are reducing custom-backend
overhead before another evaluation and sparse-aware drafter retraining. Neither
was attempted. No model, result, or Slack/GitHub/Hugging Face update was published.
