# Plain Apertus 1.5 8B load test: memory and capacity

Shareable page: [loadtest-8b.html](loadtest-8b.html).

2026-09-29, one GH200 (95 GiB), pinned image `vllm_apertus_1.5_release-arm64`
(vLLM `0.23.1rc1.dev1029+ga601a9d99`), TP=1, bf16, `gpu_memory_utilization`
0.8, `max_model_len` 32768, prefix caching off. Run
`results/8b/loadtest/apertus15-8b-baseline-loadtest-20260929T192432Z/`
(job 3546378), driver `serving/loadtest.sh`.

## Memory

vLLM reserves its memory at start-up, so device memory does not follow load:
`nvidia-smi` shows **81.3 GiB used on the serving GPU for the whole test**, from
idle to 512 concurrent requests. The reservation, from the engine log:

| part | GiB |
| --- | --- |
| weights | 17.23 |
| peak activation workspace | 1.63 |
| CUDA graphs | 0.95 |
| KV-cache pool | 56.9 (466,144 tokens, 16-token blocks) |
| budget (0.8 x 95 GiB) | 76.0 |
| measured by `nvidia-smi` (budget + CUDA context and non-torch memory) | 81.3 |

What load consumes is the KV-cache pool. That is what the tables below report
(sampled from `/metrics` every second).

## Load sweep

(2026-09-29; its C=1 and C=8 rows used only the first 32 prompts. The
corrected plain sweep is in the 2026-10-03 section below.)

Workload: the 128 untouched summarization test prompts (~3.8k prompt tokens,
~200-token answers), natural EOS, closed loop at fixed concurrency C with
4 x C requests per level. The load generator runs on the replica's node
(`srun --overlap`), not on a login node.

| C | output tok/s | TTFT p50 / p95 (s) | TPOT p50 / p95 (ms) | KV use mean / max | running max | waiting max | preempted |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 156 | 0.09 / 0.13 | 5.9 / 5.9 | 0.6% / 0.9% | 1 | 0 | 0 |
| 8 | 690 | 0.13 / 0.57 | 9.8 / 11.1 | 4% / 6% | 8 | 0 | 0 |
| 32 | 1,431 | 0.17 / 2.2 | 19.1 / 28.2 | 16% / 22% | 32 | 19 | 0 |
| 64 | 1,653 | 0.23 / 4.5 | 34.1 / 46.8 | 32% / 42% | 64 | 52 | 0 |
| 128 | 1,839 | 0.37 / 8.6 | 65.2 / 78.1 | 66% / 80% | 128 | 101 | 0 |
| 256 | 1,844 | 10.9 / 21.3 | 86.1 / 99.1 | 91% / 100% | 167 | 232 | 40 |
| 512 | 1,872 | 40.1 / 44.0 | 86.3 / 93.3 | 95% / 100% | 167 | 476 | 66 |

- Throughput saturates at about 1.84k output tokens/s from C=128: with ~3.8k-token
  prompts the GPU spends most of its time on prefill.
- The KV cache fills at C=256. At most 167 requests then fit (~2.8k tokens
  each in the pool), the rest queue, and requests are preempted (evicted and
  recomputed later). Past this point more concurrency only adds waiting time.
- Every request succeeded at every level. GPU utilization was 100% under load,
  peak power 531 W.

## Caveats

- One workload shape (long prompts, short answers). Chat or code prompts are
  shorter, so more of them fit in the pool before it fills; rerun with
  `LOADTEST_WORKLOAD=chat` or `code` for those.
- Closed loop: the client keeps exactly C requests open. Real traffic arrives
  at a rate; an open-loop test (requests per second) is the next step for a
  capacity number.
- The first sweep (job 3546229) was capped at 100 concurrent requests by the
  client's HTTP connection pool, and the second (3546292) aborted on a
  keep-alive race; both are fixed in the harness.

## Plain vs DSpark vs EAGLE 3.1, corrected sweep (2026-10-03)

**These are the reference numbers.** The earlier sweeps below ran only the first
32 of the 128 prompts at C=1 and C=8 and all of them from C=32 on, which
inflated the low-load speedups and made acceptance look like it fell under
load. Since commit d9f4b06 every level runs whole passes over the prompts.

Setup, identical for all three arms: one GH200 (95 GiB) of a 4-GPU node, TP=1,
target and drafter on the same GPU (vLLM speculative decoding, drafter in the
same engine), pinned image `vllm_apertus_1.5_release` (vLLM 0.23.1rc1), bf16,
`gpu_memory_utilization` 0.8, `max_model_len` 32768, prefix caching off,
`max_num_batched_tokens` 16384, `max_num_seqs` 256, 7 draft tokens. Load
generator on the same node's CPUs (`srun --overlap`), closed loop at fixed
concurrency C, 128 x ceil(4C/128) requests per level after C warm-up requests,
streaming, greedy, natural EOS. Workload: the 128 summarization test prompts
(~3.8k prompt tokens, ~200-token answers).

| arm | checkpoint | run | job |
| --- | --- | --- | --- |
| plain | - | `apertus15-8b-baseline-loadtest-20261003T095425Z` | 3571677 |
| DSpark | Open-PerfectBlend thinking off, epoch 2 of 10 | `apertus15-8b-dspark-k7-loadtest-20261003T101305Z` | 3571734 |
| EAGLE 3.1 | same corpus and split, epoch 1 (`e31-opb-thinkoff-stage1-ep1-se`) | `apertus15-8b-eagle31-k7-loadtest-20261003T103254Z` | 3571815 |

Memory: weights 17.2 / 20.5 / 17.8 GiB; KV pool 469,840 / 342,693 (-27%) /
433,888 (-8%) tokens; `nvidia-smi` 78.2 / 80.8 / 80.5 GiB (plain / DSpark /
EAGLE).

| C | output tok/s plain / DSpark / EAGLE | vs plain DSpark / EAGLE | accepted length DSpark / EAGLE | TPOT p50 ms plain / DSpark / EAGLE | TTFT p50 s plain / DSpark / EAGLE | running max | preempted |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 159 / 212 / 192 | 1.34x / 1.21x | 1.83 / 2.05 | 5.9 / 4.3 / 4.8 | 0.08 / 0.09 / 0.08 | 1 / 1 / 1 | 0 / 0 / 0 |
| 8 | 802 / 934 / 893 | 1.16x / 1.11x | 1.81 / 2.03 | 9.1 / 7.7 / 8.3 | 0.10 / 0.11 / 0.11 | 8 / 8 / 8 | 0 / 0 / 0 |
| 32 | 1,397 / 1,401 / 1,398 | 1.00x / 1.00x | 1.82 / 2.03 | 19.1 / 19.8 / 19.9 | 0.18 / 0.25 / 0.28 | 32 / 32 / 32 | 0 / 0 / 0 |
| 64 | 1,649 / 1,521 / 1,548 | 0.92x / 0.94x | 1.82 / 2.02 | 34.2 / 36.9 / 37.1 | 0.24 / 0.37 / 0.39 | 64 / 64 / 64 | 0 / 0 / 0 |
| 128 | 1,841 / 1,566 / 1,636 | 0.85x / 0.89x | 1.82 / 2.02 | 64.9 / 69.3 / 68.9 | 0.38 / 0.96 / 0.62 | 128 / 128 / 128 | 0 / 7 / 0 |
| 256 | 1,837 / 1,579 / 1,645 | 0.86x / 0.90x | 1.82 / 2.02 | 87.3 / 75.1 / 86.6 | 10.8 / 17.9 / 13.8 | 169 / 129 / 158 | 46 / 15 / 18 |
| 512 | 1,860 / 1,579 / 1,647 | 0.85x / 0.89x | 1.82 / 2.02 | 87.4 / 77.1 / 90.2 | 40.1 / 52.6 / 46.8 | 170 / 129 / 158 | 73 / 36 / 42 |

Every request succeeded at every level. Probe in the same deployments (64
math/HumanEval prompts, C=8): 1,234 / 4,644 / 4,070 output tok/s, accepted
length 6.07 (DSpark) / 6.13 (EAGLE).

- Acceptance is flat under load (DSpark 1.82, EAGLE 2.03 on summarization).
  The speedup falls because the GPU becomes compute-bound: ~95% of the tokens
  it processes here are prompt tokens, and verifying 7 draft tokens per request
  per step costs more than the ~2 accepted return.
- DSpark is faster up to C=8 (one forward pass for the whole draft block);
  both break even at C=32; from C=64 on EAGLE is 3-5% ahead of DSpark (higher
  acceptance, smaller KV-cache cost: 158 vs 129 requests running at once).
- Both drafters are early checkpoints; rerun on the final ones.

## Earlier sweeps (superseded at C=1 and C=8)

Kept for the record; their C=1 and C=8 rows used only the first 32 prompts.

## DSpark drafter vs plain target (2026-09-30)

Both deployments with the same engine limits: `max_num_batched_tokens` 16384,
`max_num_seqs` 256, otherwise as above (0.8, 32k, prefix caching off). Plain:
`apertus15-8b-baseline-loadtest-20260930T161730Z` (job 3554644); DSpark: `apertus15-8b-dspark-k7-loadtest-20260930T155908Z` (job
3554517). Drafter: the colleague's Magpie 100k checkpoint (thinking off, epoch 7
best, 2026-09-25), depth 7, served with `methods/dspark/launch/dspark.sh`. This
is not the Open-PerfectBlend drafter of the EAGLE comparison.

**Serving check first.** On the 64 math/HumanEval probe prompts (C=8, 384
tokens) DSpark accepts 5.01 tokens per round (the colleague measured 4.98),
4,041 vs 1,217 output tokens/s for the plain target (3.3x), TPOT p50 1.7 vs
6.1 ms. The serving path is correct.

**Memory.** Weights 20.5 GiB (target 17.2 + drafter 3.3); KV pool 342,655
tokens vs 469,792 (-27%) inside the same 0.8 budget; `nvidia-smi` 80.8 GiB
DSpark vs 78.2 GiB plain. With the engine default `max_num_seqs` the DSpark
deployment used 95 of 95 GiB and one start ran out of memory
(`...20260930T124347Z/NOTE.md`), so `max_num_seqs` 256 is needed.

Summarization sweep (plain → DSpark):

| C | output tok/s | TTFT p50 (s) | TPOT p50 (ms) | TPOT p95 (ms) | KV max | running max | preempted | DSpark accepted length |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 155 → 221 | 0.09 → 0.09 | 5.9 → 4.0 | 5.9 → 4.5 | 1% → 1% | 1 → 1 | 0 → 0 | 1.99 |
| 8 | 691 → 824 | 0.12 → 0.15 | 9.6 → 7.6 | 11.2 → 10.1 | 6% → 7% | 8 → 8 | 0 → 0 | 1.99 |
| 32 | 1,408 → 1,291 | 0.16 → 0.22 | 18.9 → 21.6 | 28.4 → 27.7 | 22% → 28% | 32 → 32 | 0 → 0 | 1.52 |
| 64 | 1,665 → 1,386 | 0.24 → 0.32 | 33.7 → 40.1 | 46.8 → 54.1 | 41% → 55% | 64 → 64 | 0 → 0 | 1.51 |
| 128 | 1,861 → 1,440 | 0.40 → 0.99 | 63.7 → 76.5 | 77.6 → 107.3 | 79% → 100% | 128 → 128 | 0 → 10 | 1.51 |
| 256 | 1,883 → 1,450 | 10.4 → 19.6 | 84.9 → 82.2 | 99.9 → 107.8 | 100% → 100% | 170 → 129 | 29 → 20 | 1.51 |
| 512 | 1,885 → 1,451 | 39.5 → 57.6 | 86.3 → 85.1 | 93.9 → 108.4 | 100% → 100% | 171 → 129 | 75 → 61 | 1.51 |

- On summarization this drafter accepts only ~2 tokens per round (vs 5.0 on
  math/code): it was trained on 100k short Magpie chats, and these are
  ~3.8k-token documents.
- DSpark helps at low load (1.42x output tokens/s at C=1, 1.19x at C=8) and
  hurts from C=32 on (-8% at C=32, -23% at C=512). Once the GPU is
  compute-bound, verifying 7 draft tokens per request per step costs more
  than ~1.5 accepted tokens return.
- The smaller KV pool fills at C=128 instead of C=256; at most 129 requests
  run at once vs 171.
- Acceptance 1.99 at C<=8 vs 1.51 from C=32 on is a prompt-mix artifact,
  not batching: C=1 and C=8 ran 32 requests, i.e. only the first 32 of the
  128 prompts; C>=32 ran all of them (see "Acceptance under load" below).
- The first DSpark sweep (job 3552386) ran at the engine defaults, where k=7
  makes vLLM cap scheduled tokens at 2048 per step (at most 99 running); it is
  kept for the record (`...20260930T122216Z/NOTE.md`).

### Open-PerfectBlend drafter, epoch 2 (preliminary)

Same test on the colleague's Open-PerfectBlend thinking-off checkpoint after 2 of
10 epochs (`dspark_apertus15-8b_open-perfectblend_thinking-off_2026-09-30_epoch2_best`,
trainer validation 5.18 accepted tokens per round), run
`apertus15-8b-dspark-k7-loadtest-20260930T180508Z` (job 3555196). Final numbers
need the final checkpoint; memory and KV pool are identical to the Magpie
drafter (same architecture).

Probe: 6.10 accepted tokens per round, 4,798 output tokens/s (3.9x plain),
TPOT p50 1.4 ms.

| C | output tok/s plain → Magpie → OPB e2 | TPOT p50 plain → OPB e2 (ms) | accepted length Magpie → OPB e2 |
| --- | --- | --- | --- |
| 1 | 155 → 221 → 262 | 5.9 → 3.3 | 1.99 → 2.41 |
| 8 | 691 → 824 → 952 | 9.6 → 6.5 | 1.99 → 2.45 |
| 32 | 1,408 → 1,291 → 1,445 | 18.9 → 19.0 | 1.52 → 1.82 |
| 64 | 1,665 → 1,386 → 1,526 | 33.7 → 36.2 | 1.51 → 1.82 |
| 128 | 1,861 → 1,440 → 1,581 | 63.7 → 69.4 | 1.51 → 1.82 |
| 256 | 1,883 → 1,450 → 1,582 | 84.9 → 74.8 | 1.51 → 1.82 |
| 512 | 1,885 → 1,451 → 1,589 | 86.3 → 76.8 | 1.51 → 1.82 |

1.69x the plain target at C=1, 1.38x at C=8, even at C=32, 16% below from
C=128 on. Acceptance 2.45 → 1.82 from C=32 on is the same prompt-mix
artifact.

### EAGLE 3.1 epoch 1 vs DSpark epoch 2 (preliminary)

Our EAGLE 3.1 head trained on the same corpus and split, after 1 of 10 epochs
(`e31-opb-thinkoff-stage1-ep1-se`, step 25,450), served at depth 7 with the same
engine limits: run `apertus15-8b-eagle31-k7-loadtest-20261002T064001Z` (job
3565020). Both drafters are early checkpoints; the numbers will move.

| | plain | DSpark e2 | EAGLE 3.1 e1 |
| --- | --- | --- | --- |
| weights (GiB) | 17.23 | 20.51 | 17.78 |
| KV pool (tokens) | 469,792 | 342,653 (-27%) | 433,824 (-8%) |
| probe accepted length (C=8) | - | 6.10 | 6.13 |
| probe output tok/s (C=8) | 1,217 | 4,798 (3.9x) | 4,096 (3.4x) |
| probe TPOT p50 (ms) | 6.1 | 1.4 | 1.7 |

Summarization sweep, output tokens/s (accepted length):

| C | plain | DSpark e2 | EAGLE 3.1 e1 |
| --- | --- | --- | --- |
| 1 | 155 | 262 (2.41) | 228 (2.58) |
| 8 | 691 | 952 (2.45) | 860 (2.58) |
| 32 | 1,408 | 1,445 (1.82) | 1,409 (2.03) |
| 64 | 1,665 | 1,526 (1.82) | 1,538 (2.02) |
| 128 | 1,861 | 1,581 (1.82) | 1,619 (2.02) |
| 256 | 1,883 | 1,582 (1.82) | 1,625 (2.02) |
| 512 | 1,885 | 1,589 (1.82) | 1,638 (2.02) |

- Same acceptance on the probe; DSpark is 17% faster there, presumably because
  it drafts the block of 7 in one forward pass where EAGLE runs 7 draft steps.
- On summarization EAGLE accepts slightly more but DSpark is faster up to C=8;
  from C=64 on EAGLE is 2-3% ahead, helped by its larger KV pool (158 vs 129
  requests running at once). Both are below the plain target from C=64 on.
- Acceptance 2.58 → 2.02 from C=32 on is the same prompt-mix artifact.
  Within each table row both arms ran the same prompts, so per-level
  comparisons hold; across rows, C<=8 used an easier subset.

### Acceptance under load (probe prompts, 2026-10-03)

The 64 probe prompts with 384-token answers (short prompts, decode-bound), same
checkpoints and limits: DSpark `...dspark-k7-loadtest-20261003T092527Z`, EAGLE
`...eagle31-k7-loadtest-20261003T093127Z`, plain
`...baseline-loadtest-20261003T093720Z`. No request queued at any level.

| C | output tok/s plain / DSpark e2 / EAGLE e1 | speedup DSpark / EAGLE | accepted length DSpark / EAGLE | TPOT p50 plain / DSpark / EAGLE (ms) |
| --- | --- | --- | --- | --- |
| 1 | 172 / 773 / 636 | 4.5x / 3.7x | 6.51 / 6.48 | 5.8 / 1.2 / 1.5 |
| 8 | 1,170 / 4,822 / 3,945 | 4.1x / 3.4x | 6.51 / 6.43 | 6.1 / 1.3 / 1.6 |
| 32 | 3,962 / 10,183 / 9,196 | 2.6x / 2.3x | 6.10 / 6.14 | 6.7 / 2.6 / 2.9 |
| 64 | 6,669 / 11,585 / 11,226 | 1.7x / 1.7x | 6.10 / 6.15 | 8.1 / 4.6 / 4.8 |
| 128 | 10,154 / 12,741 / 12,744 | 1.25x / 1.26x | 6.09 / 6.15 | 11.0 / 8.5 / 8.5 |

- Acceptance does not depend on load: 6.1 at C=32, 64 and 128 alike. The
  step from 6.5 at C<=8 is the prompt mix (32 requests = the 32 math prompts;
  from C=32 on math and HumanEval), which the same per-level request rule
  caused on summarization. Fixed in the harness: every level now runs whole
  passes over the prompts.
- The shrinking speedup is real and expected: with constant acceptance it falls
  from 4.5x to 1.25x as the GPU goes from memory-bound to compute-bound and
  verifying 8 tokens per request per step stops being free. On summarization
  (~95% of processed tokens are prefill) that point comes much earlier.

Rerun, e.g. with another checkpoint:

```
STAGE=8b METHOD=dspark DSPARK_CHECKPOINT=<dir> NUM_SPECULATIVE_TOKENS=7 LOADTEST_PROBE=1 \
  MAX_NUM_BATCHED_TOKENS=16384 MAX_NUM_SEQS=256 \
  LOADTEST_CONCURRENCIES="1 8 32 64 128 256 512" serving/loadtest.sh
```

For an EAGLE head, replace `METHOD=dspark DSPARK_CHECKPOINT=<dir>` with
`METHOD=eagle EAGLE_HEAD=<head dir>`.
