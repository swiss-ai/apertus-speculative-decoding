# Drafter KV sparsity experiment

This experiment asks whether the existing trained Apertus 1.5 8B drafter can
retain less KV without materially reducing acceptance, and whether doing so
improves serving speed or capacity. It is isolated on
`experiment/drafter-kv-sparsity`, based on main `c15baf59`. Only the drafter cache
changes; target KV and verification remain native.

The [completed thinking-off screen](../results/8b/drafter-kv/kv-sparsity-20261022-a/README.md)
failed the expansion gates: physical drafter storage falls, but throughput,
latency, and summarization acceptance regress. The experiment stopped before
confirmation or evaluation of the other heads, after 3.61 allocated GPU-hours.

Maintained campaign launchers and the fixed design live in
[`experiments/drafter-kv-8b/`](../experiments/drafter-kv-8b/README.md).
[`serving/draft_kv/`](../serving/draft_kv/README.md) contains the opt-in runtime
and overlay installer; controller and evidence analysis use the shared
`src/apertus_bench/` package. The results tree preserves executed scripts and
source snapshots byte-for-byte, so recorded launch paths refer to each run's
original Git revision. No further experimental runs are planned.

## Evidence and policy

The [action-items canvas](https://swissai-initiative.slack.com/docs/T0627JS49GT/F0C6431UBTP)
references [DSpark](https://arxiv.org/html/2607.05147v1) and
[MineDraft](https://arxiv.org/html/2603.18016v3). DSpark section 5.1 describes a
trained 128-token sliding-window drafter. MineDraft overlaps drafting and
verification across batches; it does not prescribe cache eviction.

Our five-layer DSpark head already uses a 2,048-token window. Earlier window
reductions to 1,024/512/256/128 degraded summarization acceptance without
improving throughput; see `system-optimizations-8b.md`, "Drafter KV cache".
Those completed results are background, not new runs in this campaign.

[MagicDec](https://arxiv.org/html/2408.11049v3) is the closest draft-cache
precedent: sparse drafting with unchanged full-context verification.
[SnapKV](https://arxiv.org/html/2404.14469v2) selects important prompt positions;
[H2O](https://arxiv.org/html/2306.14048v2) keeps accumulated-attention heavy hitters
and recent positions during generation. One-time prompt compression alone does
not bound generation cache growth. A sparse attention working set backed by a
full cache also does not establish memory savings; MagicDec's reference SnapKV
implementation allocates both.

We test one page-level adaptation: cumulative verified-anchor attention mass,
averaged across drafter heads/layers, retaining the highest-scoring pages plus
all pages intersecting the latest 128 verified context tokens. Ties prefer newer
pages. The two persistent budgets are 1,024 and 512 slots, in 16-token pages.
Selection stays within the original DSpark window. Original RoPE positions and
query visibility remain intact. This is not a faithful StreamingLLM reproduction:
StreamingLLM's cache-relative position shifting would be an additional change.

During prompt ingestion use the bounded native window. The first completed-prompt
draft pass provides initial all-layer anchor scores; release unselected physical
pages afterward. Subsequent decode needs the retained budget plus one staging
page and separate query scratch. Report bootstrap and staging separately; do not
keep a full backing cache. A 2,048-token unaligned interval can intersect 129
pages; the latest 128 tokens can intersect nine pages.

## Fixed design

- Target: Apertus-v1.5-8B revision
  `a411d838600baf0e3635a3daf66fb7c55fc97bb6`.
- Existing trained final DSpark off/on epoch-10 heads; existing
  `e31-opb-thinkoff-stage1-se` EAGLE off head. Freeze exact paths, configs, weight
  hashes, and their existing training/export provenance before running.
- One GH200 per deployment, TP1, fixed k=7, context 32768, memory utilization 0.8,
  prefix caching off, batch-token limit 16384, sequence limit 256. Preserve cache
  grouping and the image's existing compatibility patches across all arms.
- Screen native, recent-512, importance-1024, importance-512 on frozen validation
  chat/code/summarization at C=1/32/256, twice with GPU assignments swapped.
- Select on validation only. Among qualifying policies, maximize geometric-mean
  throughput across workloads at C=32/256; within 2%, prefer lower storage.
- Confirm one candidate on untouched test chat/code/summarization and OPB-heldout
  at C=1/8/32/128/256 in five independent node allocations. Keep same-configuration
  captures and native/candidate duplicates, and counterbalance GPU assignments.
- Each cell uses whole corpus passes until at least max(corpus size, 4*C)
  requests, identical request order, natural EOS, and existing per-prompt caps.
- Only after confirmation passes, compare that policy/budget with native on
  DSpark on and EAGLE off, each in two independent node allocations. For thinking
  on, derive explicit thinking-enabled frozen copies with cap 8192 and report
  truncation. Those two-allocation extension estimates are preliminary.

The generic workload manifest has stale validation/test hashes. Capture actual
bytes and prompt IDs in new immutable campaign manifests. Do not rewrite input
corpora or overwrite completed cells.

## Gates and interpretation

Before performance interpretation, test the full-coverage custom backend against
native, sparse attention against a dense retained-position reference, and page
crossings, chunked prefill, full/partial/immediate rejection, graph behavior,
preemption/resumption, completion and request-slot reuse. A different kernel or
eager-only runtime can add overhead; report the control and never attribute that
overhead solely to eviction quality.

Use `protocol.md`'s calibrated correctness gate, with both within-deployment and
cross-deployment controls. Existing margins are common-prefix fraction 0.05,
normalized edit distance 0.05 and completion-token difference 2. Exact text
agreement alone is not the gate.

Expansion requires all of:

- at least 45% lower steady-state live drafter storage, including policy
  metadata, for requests filling the native window;
- mean accepted length at least 95% of native in every workload;
- correctness passes and at least 99% successful requests;
- no workload loses more than 3% throughput or raises p95 TTFT/TPOT by over 5%;
- either at least 5% throughput gain in a declared workload/concurrency cell,
  with the node-block bootstrap interval excluding zero, or at least 5% lower
  combined live target/drafter KV storage at matched sequence lengths while
  meeting the performance bounds.

Report absolute metrics, all repeats, interval uncertainty, live blocks and bytes
by target/drafter group, metadata/staging/bootstrap bytes, attended rows,
selection overhead, acceptance length/vector, throughput, latency, errors and
preemptions. A fixed shared pool or unchanged nvidia-smi allocation does not
measure live per-request cache storage. Reduced storage is not automatically a
claim of sustainable request-rate improvement.

## Budget and provenance

The hard cap is 72 allocated GPU-hours, including failures and idle allocated
GPUs: preflight 6, screen 12, confirmation 30, DSpark-on 12, EAGLE-off 12. Reserve
pending allocations before submission; reconcile actual Slurm elapsed time when
terminal. All four GPUs of an exclusive node count, even when fewer are busy.
Never submit beyond reserved remaining budget. Stop promptly on a reported
agent/tool usage limit and cancel only this campaign's owned jobs.

Each deployment keeps raw requests, immutable metadata, summaries, metric
snapshots, launch logs, exact commands, source/image revisions and hashes,
checkpoint validation, original/resolved configs, applied patches, corpus
digests, resolved environments, Slurm jobs and nodes. Store code and analysis
alongside canonical results; leave large weights and raw cluster logs in place.

EAGLE's existing export/reload checks pass, but its training summary records
`real_updates=false`; preserve that caveat and require native serving validation.
Do not retrain, substitute a new head, or describe all training gates as passed.

Sparse-aware retraining is an unexecuted suggestion for later approval. Scheduler,
routing, disaggregation, model-training and unrelated benchmark experiments are
outside this campaign. No Slack/GitHub/Hugging Face publication is authorized.
