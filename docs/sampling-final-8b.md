# Completed 8B sampling sweeps: verdict

The experiments completed successfully and their results are now recovered. **Do not
implement narrower sampling top-k as a speed optimization.** At fixed temperature 1
and top-p 1, top-k50 was always slower, top-k20 changed throughput by at most +0.6%,
and top-k5 by at most +8.0% with regressions elsewhere. These changes also change
the answer distribution; this experiment provides no quality evidence for them.

**Keep load-dependent speculative draft length in mind.** The final DSpark heads
deliver substantial latency gains at low load, and draft length 3 gives substantial
throughput gains at intermediate load. Draft length 7 loses badly at high load.
High-load gains depend on workload, thinking mode and sampling; this is not evidence
for enabling speculative decoding everywhere. `draft k` below is the number of
speculative tokens, separate from the sampler's `top_k`.

Evidence: [validated absolute measurements](../results/8b/sampling/recovery-20261027/analysis-v2/absolute.csv),
[252 matched speculative/plain comparisons](../results/8b/sampling/recovery-20261027/analysis-v2/matched.csv),
and [162 fixed-temperature sampling comparisons](../results/8b/sampling/recovery-20261027/verdict-v1/sampling-topk-effects.csv).
Every speculative arm is compared with plain at the same thinking mode, sampling,
workload, request schedule, concurrency and engine configuration.

| Sampling restriction at T=1, top-p=1 | Absolute output throughput change versus unfiltered sampling, across all arms/workloads/loads | Verdict |
|---|---:|---|
| top-k50 | −7.6% to −0.7% | No speed benefit |
| top-k20 | −6.2% to +0.6% | No substantial benefit |
| top-k5 | −5.1% to +8.0% | Mixed, modest; no implementation case |

For example, thinking-on Arena at concurrency 8 with DSpark draft k7 rises from
2,455 to 2,650 output tok/s with sampling top-k5: only 195 tok/s, despite acceptance
length rising from 2.75 to 3.12. Conversely, off-mode EAGLE draft k3 at Arena
concurrency 256 falls from 11,566 to 10,971 tok/s. Comparing each speculative arm's
*speedup over plain* alone can hide this: filtering also slows plain. The
`default` sampler changes temperature **and** top-p **and** top-k, so differences
between default and greedy/unfiltered cannot be attributed to top-k alone.

The following are absolute results from matched cells, not cross-sampler speedups.
GPU-seconds per million output tokens are `1,000,000 / output_tok_s` for the one-GPU
replica during measurement. They exclude deployment startup, warmup, idle allocation
and training; they are not billed job cost or maximum sustainable capacity.

| Mode / workload / sampling | C | DSpark draft k | Output tok/s, plain → spec | Ratio | GPU-s / M output tokens, plain → spec | Mean request s, plain → spec | Accepted / step |
|---|---:|---:|---:|---:|---:|---:|---:|
| off / OPB / default | 1 | 7 | 163 → 536 | 3.29× | 6,134.0 → 1,864.7 | 2.91 → 0.88 | 4.55 |
| off / Arena / default | 64 | 3 | 6,455 → 9,449 | 1.46× | 154.9 → 105.8 | 4.86 → 3.44 | 2.89 |
| off / Arena / default | 256 | 3 | 11,814 → 11,611 | 0.98× | 84.6 → 86.1 | 10.98 → 11.48 | 2.90 |
| off / Arena / T1 unfiltered | 256 | 7 | 12,452 → 7,743 | 0.62× | 80.3 → 129.1 | 10.89 → 17.89 | 3.00 |
| on / OPB / greedy | 1 | 7 | 171 → 647 | 3.77× | 5,831.4 → 1,546.0 | 5.56 → 1.47 | 5.15 |
| on / Arena / default | 64 | 3 | 6,080 → 10,207 | 1.68× | 164.5 → 98.0 | 14.60 → 8.92 | 2.95 |
| on / OPB / default | 256 | 3 | 9,109 → 11,454 | 1.26× | 109.8 → 87.3 | 23.00 → 18.91 | 3.11 |
| on / Arena / default | 256 | 3 | 10,283 → 12,207 | 1.19× | 97.2 → 81.9 | 35.43 → 30.39 | 2.96 |
| on / Arena / T1 unfiltered | 256 | 3 | 10,864 → 10,478 | 0.96× | 92.0 → 95.4 | 34.14 → 35.85 | 2.34 |

The first row saves about 2.03 seconds per request and 4,269 GPU-seconds per million
tokens at that low-load operating point. At Arena C64/default, off-mode saves
1.42 seconds/request and 49.1 GPU-seconds/M; thinking-on saves 5.68 seconds/request
and 66.5 GPU-seconds/M. Those are substantial. At off-mode Arena C256/default,
DSpark k3 regresses and EAGLE k3 gains just 2.1%; neither justifies implementation
there. At off-mode Arena C256/T1-unfiltered, k7 costs an extra 7.00 seconds/request
and 48.8 GPU-seconds/M despite accepting three tokens per step on average.

Across all six tested sampling settings, DSpark k7 delivers 2.62–3.47× on off-mode
OPB C1 and 2.36–3.77× on thinking-on OPB C1. At Arena C64, k3 delivers 1.30–1.52×
off and 1.34–1.83× on. These support retaining a low-/intermediate-load draft policy.
For off-mode Arena C256, even k3 ranges from 0.87 to 1.02×; keep plain there.
Thinking-on OPB C256/default has a substantial 26% gain, but the T1 variants gain
only 6–13%. Thinking-on Arena C256 is especially conditional: default gives 19%
(about 5.04 seconds/request), T1-unfiltered regresses, and greedy gives 31% while
recording 67 preemptions and essentially full KV usage. Preserve these as specific
operating-point evidence, rather than a high-load rollout recommendation. Full
[load ranges](../results/8b/sampling/recovery-20261027/verdict-v1/draft-load-ranges.csv)
include EAGLE3.1 and DSpark k5; their occasional small advantages do not establish
another implementation priority. In particular, k5 beating k7 by a few percent
at one operating point is not a general k5 recommendation.

The output-length and streaming qualifications materially affect interpretation.
Off-mode OPB uses prompt-specific caps of 16–1,024 tokens; 39.1–57.8% of requests
finish at their length limit. Off-mode Arena uses a 1,024-token cap and truncates
10.0–17.0%. Thinking-on OPB caps span 56–2,048 with 48.4–64.8% length finishes;
thinking-on Arena caps at 2,048 with 39.6–52.0% length finishes. These are timing
results on capped outputs, not time to a complete, quality-checked reasoning answer.
Across matched speculative/plain cells, mean output-length ratios range from
0.961–1.025 off and 0.912–1.039 on. The off OPB/default C1 example is closely
matched at 474 versus 471 tokens; thinking-on OPB/greedy C1 is 954 versus 951.
The CSV exposes length and truncation per cell so shorter answers cannot silently
stand in for faster decoding.

Conventional TPOT is each request's `(end − first token)/(completion_tokens − 1)`;
it is not the gap between streamed events, which may contain multiple tokens.
For off OPB/default C1 k7, median TPOT improves 6.07 → 1.62 ms while p95 event gap
worsens 6.27 → 8.39 ms, and p95 TTFT is 85.8 → 87.4 ms. At off Arena/default C64
k3, median TPOT improves 8.59 → 6.00 ms but p95 event gap worsens 15.3 → 22.8 ms.
Thus the demonstrated decode/whole-request benefit does not promise more frequent
stream updates or better first-token latency. At C256/default thinking Arena,
TPOT improves 23.39 → 19.78 ms while p95 event gap worsens 30.3 → 69.8 ms.

The recovered protocol is narrower than the spec comments suggest. All 324 cells
record seed base 1. The retained launch-revision
[CLI](../results/8b/sampling/recovery-20261027/source-recovery/launch-revision-source/a0597294cbe829aa8d6a1e577c462571df8bf572/src/apertus_bench/cli.py),
[runner](../results/8b/sampling/recovery-20261027/source-recovery/launch-revision-source/a0597294cbe829aa8d6a1e577c462571df8bf572/src/apertus_bench/runner.py)
and [HTTP client](../results/8b/sampling/recovery-20261027/source-recovery/launch-revision-source/a0597294cbe829aa8d6a1e577c462571df8bf572/src/apertus_bench/client.py)
show an omitted `--seed` defaulting to 1, `seed + request_index`, and a `seed` field
always included in the request payload. This is source-plus-metadata evidence for
seeded requests, not a retained wire capture. The YAML's “No seed” comment is
incorrect. Do not generalize sampler throughput to an unseeded path. The same seed
schedule was used within each matched comparison.

Both jobs ran on October 24–25 despite October 22 directory names: job 3721144 on
nid006035 produced 180 cells / 94,050 successes; job 3721145 on nid007582 produced
144 cells / 75,240 successes, with no request or warmup failures. Each arm had one
GH200, target TP1, bf16, context 32,768, token budget 16,384, sequence limit 256,
memory utilization 0.8 and prefix caching disabled. The off sweep scheduled five
arms on four GPUs, reusing GPU1 for EAGLE k7; the on sweep used four GPUs. Each
configuration has one deployment, with repeated prompts at C256 rather than
independent deployment repeats. These finite closed-loop samples do not measure
an arrival-rate SLO or establish confidence intervals. GPU/node placement and
sampler order were not counterbalanced. No quality evaluation accompanied them.

Original launch Git `a0597294cbe829aa8d6a1e577c462571df8bf572`, container digest,
Slurm/launch receipts, patches, resolved overlays, all cell records and checkpoint
identities are retained in the [recovery bundle](../results/8b/sampling/recovery-20261027/README.md).
The cell metadata's stale harness Git `a26ba85…` is preserved verbatim: the container
working directory points at an older checkout, while launch source prepends the
measurement repo's `src` to `PYTHONPATH`. Original launch receipts and separately
labelled recovery-time references are used together; no historical identity was
silently repaired. Validation recomputed all 169,290 request records, latency
quantiles, token totals and speculative metric deltas, and checked source hashes.
No successful sampling GPU job was rerun.
