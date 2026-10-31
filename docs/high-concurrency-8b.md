# Final-checkpoint 8B serving-layout campaign

Status on 2026-10-31: the original bounded layout recovery and CPU screening
review are complete. All intended grids were measured; no further GPU work or
repeat-for-a-pass is planned. The strict view contains 1,040 unique cells,
803 comparisons and 743 eligible screening pairs, with no missing stage analyses.
All 1,040 raw successful-request usage re-sums pass. Eleven new P/D reliability
cells remain rejected by the unchanged 99% rule (two thinking-on, nine off).

Keep TP4 speculation as a load-specific candidate with explicit memory costs;
k3 is useful within TP2DP2 at moderate load. These measurements establish no
P/D adoption advantage for this decode-heavy grid and no blanket layout change.
Calibrated output correctness, deployment-repeat confidence, open-loop sustainable
capacity and independent load-generator sufficiency remain unestablished here.
Broader study/delivery item #6 is still open. Primary closure and evidence index:
[31 Oct final review](../results/8b/layout-recovery-20261031/README.md).

## Current TP4 verdict

TP4 speculation has substantial load-specific screening gains. At C1024,
DSpark-off produces 20.4k tok/s OPB and 22.3–22.6k Arena-Hard (1.44–1.60x
paired plain); EAGLE produces 21.4k and 23.0k (1.45–1.50x); DSpark-on
produces 26.9k and 27.8–28.0k (2.22–2.64x). The same settings regress at
C128: Arena-Hard throughput falls 35–45% for DSpark-off, about 28% for
EAGLE and 35–42% for DSpark-on. EAGLE's 6% OPB C128 gain is a small effect.
These results support keeping load sensitivity explicit, not a blanket setting.

The gains have material memory costs. The table shows OPB at C1024 (thinking
prompts for DSpark-on), one paired deployment per arm. Resident memory is the
sampled sum across all four GPUs, including reserved pools. Logical KV tokens
represent one TP4 engine pool; they are not multiplied by four TP shards.

| TP4 arm | Plain → spec tok/s | p95 E2E seconds | Logical KV tokens | Mean resident GiB/node |
|---|---:|---:|---:|---:|
| DSpark off k7 | 12,719 → 20, 379 | 62.0 → 47.8 | 1,936,080 → 1,438,768 | 289.8 → 361.9 |
| EAGLE k7 | 14,305 → 21,400 | 56.3 → 42.8 | 1,936,080 → 1,863,840 | 289.9 → 359.4 |
| DSpark on k7 | 12,107 → 26,908 | 151.4 → 68.3 | 1,936,080 → 1,438,768 | 290.0 → 362.0 |

At these cells, speculative execution uses roughly 69.5–72.0GiB more resident
memory per node than its matched plain deployment. DSpark has 25.7% less
logical KV capacity; EAGLE has 3.7% less. The 0.7 GPU memory-utilization setting
is a KV planning budget, not a hard total-process resident-memory cap. Request
latency and natural-EOS output volumes must be read alongside throughput;
full absolute metrics, scopes and losing strata are in the strict review files.

Historical DP4 data remains strong context: thinking-off k3 yields about 29.6k
OPB and 37.8–38.1k Arena-Hard tok/s; thinking-on DSpark k3 yields 45.8k
and 51.5–51.9k at C1024. Those are different configurations/allocations, so they
are descriptive context rather than a controlled layout effect. Assess TP4
against this absolute node-performance frontier before proposing deployment.

Recovered EAGLE k5 TP4 (3737762) also crossed over: 1.75–1.87x plain at
C1024, but 34–35% Arena-Hard regressions at C128. It was collected without a
rerun. Earlier valid DP4/TP4 arms and failed attempts remain preserved.

## Thinking-on P/D verdict

NIXL now starts and completes requests, and all 63 intended cells were produced. At C1024,
plain 1-prefill/3-decode P/D is 18–30% slower than same-job plain DP4. Adding
DSpark k3 improves P/D itself 26–30%, but versus plain DP4 the eligible clean
Arena-Hard gain is only 3.8%, while OPB loses 8.9%. The contaminated C1024
speculative point is rejected. No substantial high-load P/D deployment
advantage is established by these comparisons.

At C128, speculative P/D produces 25.5–28.4k tok/s (1.80–1.91x plain DP4),
but this comparison includes the drafter's benefit; it does not isolate a P/D
advantage over equivalent speculative DP4. Do not promote a small high-load
gain or attribute speculation's benefit to disaggregation.

Two cells fail reliability: plain P/D contaminated-thinking C256 has 1019/1032
successes (98.740%); speculative P/D contaminated-thinking C1024 has 4085/4128
(98.958%). Errors are upstream decode HTTP/connection-read failures through the
proxy. They remain rejected and are not rerun for a pass. No benchmark accounting
bug or proven compute-capacity cause was demonstrated. Separate home-quota
warnings arose in background usage-reporting threads; no causal link to request
failures is claimed. P/D counters cover the first decode instance only; the
report never extrapolates its preemptions into whole-node totals.

## TP2DP2 verdict

The original TP2DP2 k3/k5 grid completed all 120 cells with passing gates. At
C128, k3 improves throughput 44–50% over paired plain across the three workloads.
OPB p95 request latency falls 7.41→6.09s with DSpark and 7.41→5.39s with EAGLE.
Arena-Hard p95 falls 6.02–6.09→4.24–4.58s. Those are substantial screening
effects if this layout is otherwise needed; they do not establish superiority
over speculative DP4.

| TP2DP2 OPB C1024 | Plain → spec tok/s | p95 E2E seconds | Logical KV tokens | Mean resident GiB/node |
|---|---:|---:|---:|---:|
| DSpark off k3 | 22,036 → 24,578 | 31.24 → 28.61 | 2,152,000 → 1,648,470 | 319.4 → 336.2 |
| EAGLE k3 | 22,036 → 24,600 | 31.24 → 28.12 | 2,152,000 → 2,044,672 | 319.4 → 336.6 |

At C128 OPB, sampled resident memory is 319.4→330.4/330.9GiB (DSpark/EAGLE);
by C1024 it is about 17GiB higher than plain. DSpark loses 23.4% logical KV,
EAGLE 5.0%. C1024 throughput gains are only4–14% across workloads, including
small clean-Arena effects. k5 adds no consistent under-load benefit; k3 generally
performs better there. At C1/4, DSpark k5 does beat k3 by 15–23%, but this saves
only about 0.18–0.27s p95 for an extra 9.6–15.4GiB sampled resident memory. That
narrow tradeoff does not justify a general k5 implementation recommendation.

There is configured capacity headroom relative to historical DP4: TP2DP2 DSpark
k3 has 1.648M logical KV tokens versus 1.371M for DP4 k3 (about 20% more), with
OPB C1024 throughput 24.58k versus 29.60k and resident 336.2 versus 318.3GiB.
These are different allocations/configurations: this is a descriptive capacity/
throughput tradeoff, not a controlled layout effect. No preemption or capacity-
limited serving win was demonstrated at these points; extra configured KV alone
is not evidence to promote the layout.

## Thinking-off P/D verdict

All147 intended cells completed; nine reliability failures remain excluded.
Plain P/D never shows a substantial throughput gain over same-job plain DP4:
1p3d is 0.51–0.55x DP4 at C1024, 2p2d 0.48–0.49x. Adding a drafter improves
P/D itself, but at C1024 eligible 1p3d points are only 0.66–0.70x plain DP4;
2p2d is 0.53–0.57x. Low-load speculative P/D gains versus plain DP4 include
the drafter benefit and do not isolate disaggregation's value over speculative
DP4. Do not implement P/D on the strength of these comparisons.

On clean Arena-Hard C1024, plain DP4 delivers 28,956tok/s with 43.55s p95 E2E.
1p3d DSpark delivers 20,096tok/s with 68.39s p95 (24.85s longer), while sampled
resident memory falls only 315.5→312.5GiB (about 3.0GiB saved). TTFT p95 worsens
38.74→65.58s even though p95 inter-token time improves 14.21→6.42ms. EAGLE is
similar: 20,323tok/s,67.11s p95. 2p2d DSpark falls to 16,024tok/s and 82.34s
p95 for about 3.0GiB resident saving. The memory savings do not compensate for
these throughput and request-latency losses.

P/D raw counters and observed KV capacity cover the first decode instance only.
Startup role logs separately show1p3d DSpark decode capacity 1,028,367 tokens
and total role pools 1,371,156; the prefill pool cannot be used as extra decode
capacity. We do not subtract a single decode engine from whole-node DP4,
extrapolate preemptions, or claim that resident memory equals live KV bytes.

The nine thinking-off rejected cells contain 354 failed requests: 239 HTTP500 and
115 ReadError. They occur across 1p3d and 2p2d, plain and both drafters, mostly
OPB C128–1024 plus one contaminated Arena C512 point. Proxy logs show upstream
prefill and decode connection reads failing. Exact transport root cause remains
unestablished; neither a harness bug nor a compute-capacity threshold is proven.
All 11 new P/D exclusions (including the earlier two thinking-on cells) are
listed with raw hashes in qualified-findings.json. No success threshold was
relaxed and no cell was retried for a pass.

## Recovery and provenance

Original EAGLE/DSpark TP4 engine deaths were CUDA OOMs. Recovery reserves extra
headroom with gpu_memory_utilization 0.7 on both paired arms, retaining context
32768, max_num_seqs 1024 and batched tokens 16384. The first recovery preflight
3769354 failed before readiness because its checker used the wrong connector
module and expandable graph allocations failed cudaIpcGetMemHandle. Corrected
preflight 3782518 passed with the pinned factory module and
`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:False`, retaining custom all-reduce
and CUDA graphs. Full recovered jobs 3789188, 3798373 and 3800281 then passed.

P/D 3800861 completed with the two strict reliability rejections above. The
controller stopped as designed. Independent original layouts subsequently
completed without promoting those false gates. A subsequent submission attempt hit home quota
before sbatch; its files are preserved. Only owned patch/config runtime paths
were moved to scratch for the fresh continuation; shared HOME/caches/installs
and frozen engine source remain unchanged.

The effective engine source is base efad15a1 plus source overlay SHA256
51fdd333056fedaeafeb7cbc1df9c33bc71f24b84cdeb09e31bd98ca48ec9a8d;
all 75 source hashes were verified. Every deployment retains image/checkpoint/
corpus identity, launch command, source patch, environment, Slurm/node identity,
requests and raw metric snapshots. V2 collection preserved 3.181GB original
bytes as 1.098GB lossless stored artifacts with both hashes. No weights/corpora
were duplicated. The frozen source remains preserved; collection and
postprocessing do not modify it.

The earlier strict CPU report in
`results/8b/layout-recovery-20261030/completed-v2-review-v1/` covers 773 historical
and recovered cells, 497 comparisons and 455 eligible screening pairs. Raw usage
sums, metric continuity, matching corpus/generation/prompts and sampled resident
memory are explicit. Output equivalence, deployment-level confidence intervals,
open-loop capacity and load-generator sufficiency remain separate original
scientific gates. Only substantial repeatable benefit warrants implementation.

The final strict review and qualified closure are in
`results/8b/layout-recovery-20261031/`. Source/runtime freeze and historical
continuation instructions remain in `results/8b/layout-recovery-20261030/README.md`.
