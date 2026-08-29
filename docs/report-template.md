# Apertus-v1.5 speculative decoding: results

## Executive recommendation

State enable/disable/workload-specific, the exact setting, effect sizes, and confidence.

## Experimental contract

Record models, commits, image, hardware, workloads, sampling, warmup, repeats, and exclusions.

## Correctness sanity check

Report greedy exact matches and investigate every mismatch before performance claims.

## Fixed-concurrency results

Include absolute TTFT/TPOT/throughput and ratios by workload and concurrency, with repeat-level
uncertainty rather than request-level pseudo-replication.

## Why the result occurs

Connect acceptance length, draft cost/TP, batching, GPU telemetry, and memory/KV-cache evidence.

## Serving-capacity results

Report the declared SLO gates, passing knee, first failing point, repeatability, and binding gate.

## Limitations and failures

Include discarded runs, missing metrics, node effects, corpus scope, and implementation gotchas.

## Production change

Give the exact recommended launch diff, rollout guardrails, and rollback criterion.
