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
- **Open:** acceptance falls from 1.99 (C<=8) to 1.51 (C>=32) on the same
  prompts. Greedy acceptance should not depend on batch size; the probe
  prompts at C=32+ would tell whether this is a serving bug or specific to
  this workload.
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
C=128 on. The same acceptance drop from C=32 on appears with this drafter
(2.45 → 1.82).

### EAGLE 3.1 head on the same corpus, epoch 1 (preliminary)

2026-10-02, same engine limits, run
`apertus15-8b-eagle31-k7-loadtest-20261002T064001Z` (job 3565020). Head
`e31-opb-thinkoff-stage1-ep1-se`: our EAGLE 3.1 run on the colleague's
Open-PerfectBlend thinking-off corpus and split, epoch 1 of the 2-epoch stage 1
(step 25,450, held-out simulated acceptance length 3.45), depth 7, served with
`methods/eagle/launch/eagle.sh`. Final numbers need the final checkpoints of
both drafters.

**Memory.** Weights 17.78 GiB (the head shares the target's embedding and
`lm_head`); KV pool 433,824 tokens: -8% vs plain, where DSpark loses 27%.
`nvidia-smi` 80.5 GiB.

Probe: 6.13 accepted tokens per round (DSpark OPB e2: 6.10), 4,096 output
tokens/s (3.4x plain; DSpark 4,798, 3.9x), TPOT p50 1.7 ms (DSpark 1.4).

| C | output tok/s plain → DSpark OPB e2 → EAGLE e1 | TPOT p50 plain → EAGLE (ms) | TTFT p50 plain → EAGLE (s) | KV max EAGLE | running max EAGLE | preempted EAGLE | accepted length DSpark → EAGLE |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 155 → 262 → 228 | 5.9 → 3.8 | 0.09 → 0.09 | 1% | 1 | 0 | 2.41 → 2.58 |
| 8 | 691 → 952 → 860 | 9.6 → 7.3 | 0.12 → 0.18 | 6% | 8 | 0 | 2.45 → 2.58 |
| 32 | 1,408 → 1,445 → 1,409 | 18.9 → 20.2 | 0.16 → 0.27 | 23% | 32 | 0 | 1.82 → 2.03 |
| 64 | 1,665 → 1,526 → 1,538 | 33.7 → 37.1 | 0.24 → 0.39 | 45% | 64 | 0 | 1.82 → 2.02 |
| 128 | 1,861 → 1,581 → 1,619 | 63.7 → 69.5 | 0.40 → 0.64 | 85% | 128 | 0 | 1.82 → 2.02 |
| 256 | 1,883 → 1,582 → 1,625 | 84.9 → 87.1 | 10.4 → 13.9 | 100% | 158 | 24 | 1.82 → 2.02 |
| 512 | 1,885 → 1,589 → 1,638 | 86.3 → 90.3 | 39.5 → 47.1 | 100% | 158 | 42 | 1.82 → 2.02 |

- EAGLE: 1.47x the plain target at C=1, 1.24x at C=8, even at C=32, 13% below
  from C=128 on. Every request succeeded at every level.
- DSpark is faster at low load (1.69x / 1.38x) although it accepts fewer
  tokens per round: it drafts all 7 tokens in one forward pass, EAGLE runs 7
  sequential draft steps.
- Under load EAGLE is slightly ahead of DSpark (+2-3% from C=128): higher
  acceptance on these long documents and a larger KV pool (158 vs 129
  requests running at once).
- The acceptance drop from C=32 on appears here too (2.58 → 2.02), with a
  different drafter architecture on the same prompts, so it comes from the
  engine's batched path rather than from either drafter.

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
- The acceptance drop from C=32 on appears with EAGLE too (2.58 → 2.02), so it
  is not specific to DSpark; it looks like an engine-level effect of batching.

Rerun, e.g. with another checkpoint:

```
STAGE=8b METHOD=dspark DSPARK_CHECKPOINT=<dir> NUM_SPECULATIVE_TOKENS=7 LOADTEST_PROBE=1 \
  MAX_NUM_BATCHED_TOKENS=16384 MAX_NUM_SEQS=256 \
  LOADTEST_CONCURRENCIES="1 8 32 64 128 256 512" serving/loadtest.sh
```

For an EAGLE head, replace `METHOD=dspark DSPARK_CHECKPOINT=<dir>` with
`METHOD=eagle EAGLE_HEAD=<head dir>`.
