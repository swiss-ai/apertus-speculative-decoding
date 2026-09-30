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

Rerun, e.g. with another checkpoint:

```
STAGE=8b METHOD=dspark DSPARK_CHECKPOINT=<dir> NUM_SPECULATIVE_TOKENS=7 LOADTEST_PROBE=1 \
  MAX_NUM_BATCHED_TOKENS=16384 MAX_NUM_SEQS=256 \
  LOADTEST_CONCURRENCIES="1 8 32 64 128 256 512" serving/loadtest.sh
```
