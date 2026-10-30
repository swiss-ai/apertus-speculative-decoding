# Final-checkpoint 8B serving-layout campaign

Status on 2026-10-30: all three affected TP4 k7 comparisons are recovered and
pass their infrastructure, provenance, counter and request gates. Thinking-on
P/D completed every intended measurement, with two cells rejected by the
unchanged 99% request-success rule. The original TP2DP2 k3/k5 and thinking-off
P/D k3 comparisons continue independently. Scientific implementation promotion
remains unestablished: most configurations have one deployment, without the
protocol's repeated confirmation and calibrated per-layout correctness evidence.

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

NIXL now serves correctly, and all 63 intended cells were produced. At C1024,
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
controller stopped as designed. Independent original layouts continue without
promoting those false gates. A subsequent submission attempt hit home quota
before sbatch; its files are preserved. Only owned patch/config runtime paths
were moved to scratch for the fresh continuation; shared HOME/caches/installs
and frozen engine source remain unchanged.

The effective engine source is base efad15a1 plus source overlay SHA256
51fdd333056fedaeafeb7cbc1df9c33bc71f24b84cdeb09e31bd98ca48ec9a8d;
all 75 source hashes were verified. Every deployment retains image/checkpoint/
corpus identity, launch command, source patch, environment, Slurm/node identity,
requests and raw metric snapshots. V2 collection preserved 3.181GB original
bytes as 1.098GB lossless stored artifacts with both hashes. No weights/corpora
were duplicated. Do not sync/reset active source while the remaining jobs run.

The strict CPU report in
`results/8b/layout-recovery-20261030/completed-v2-review-v1/` covers 773 historical
and recovered cells, 497 comparisons and 455 eligible screening pairs. Raw usage
sums, metric continuity, matching corpus/generation/prompts and sampled resident
memory are explicit. Output equivalence, deployment-level confidence intervals,
open-loop capacity and load-generator sufficiency remain separate original
scientific gates. Only substantial repeatable benefit warrants implementation.

The current continuation, exact source/runtime freeze and remaining collection/
review commands are in `results/8b/layout-recovery-20261030/README.md`.
