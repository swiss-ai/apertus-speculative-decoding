DSpark drafter from the colleague: Magpie 100k, thinking off, epoch 7 best (2026-09-25),
depth 7, summarization test prompts, same settings as the plain run 20260929T192432Z.
Caveats, read before comparing:
- With k=7 and the default max_num_batched_tokens 8192, vLLM caps max_num_scheduled_tokens
  at 2048 (engine-excerpt.txt). At most 99 requests ran at once, with the KV cache at most 78%
  full; throughput saturates at ~1.44k output tokens/s.
- Device memory was 97,269 MiB from start-up on (~95 GiB, GPU full), 14 GiB above the plain
  target at the same 0.8 budget. The next start on another node ran out of memory
  (20260930T124347Z/NOTE.md).
- Acceptance length is 1.97 at C=1/8 and 1.52 from C=32 on. Greedy acceptance should not
  depend on batch size, so the drop needs explaining before these numbers are used.
