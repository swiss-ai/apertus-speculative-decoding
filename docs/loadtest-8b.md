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
