# Controlled DSpark e10 k=3 speed test

Same physical GH200 GPU, target, image, overlays, template, settings and loopback client. Only the speculative configuration changes. Three fresh deployment pairs use AB, BA, AB order.

32 deterministically selected prompts per stratum; natural EOS with a 256-token cap is primary. The separate fixed-length check generates exactly 256 tokens. C=128 is a single 128-request burst over four corpus passes, not a sustained-capacity measurement.

ID is the existing DSpark OpenPerfectBlend held-out validation workload. OOD chat and long-document summarization are shifts relative to that training mixture; they do not prove every individual topic was absent from training. Corpus hashes and chosen IDs are retained on cluster.

Speedup is DSpark output tokens/s divided by baseline output tokens/s. Verdicts require a 95% paired-deployment bootstrap interval entirely above 1.05 or below 0.95. Three pairs on one node give limited uncertainty estimates and do not establish universal speedup, losslessness, answer quality, multi-node performance, or gateway latency.

| Stratum | Generation | C | Paired speedups | Geomean [95% CI] | Verdict |
| --- | --- | ---: | --- | --- | --- |
| id-opb | natural | 1 | 2.50x, 2.45x, 1.71x | 2.19x [1.85, 2.49] | speedup |
| id-opb | natural | 8 | 2.16x, 2.15x, 1.93x | 2.08x [1.98, 2.15] | speedup |
| id-opb | natural | 128 | 1.15x, 1.17x, 1.11x | 1.14x [1.12, 1.17] | speedup |
| id-opb | fixed | 1 | 2.48x, 2.45x, 2.47x | 2.47x [2.46, 2.48] | speedup |
| id-opb | fixed | 128 | 1.18x, 1.18x, 1.17x | 1.18x [1.17, 1.18] | speedup |
| ood-chat | natural | 1 | 1.74x, 1.73x, 1.74x | 1.74x [1.73, 1.74] | speedup |
| ood-chat | natural | 8 | 1.49x, 1.50x, 1.51x | 1.50x [1.49, 1.50] | speedup |
| ood-chat | natural | 128 | 0.92x, 0.93x, 0.93x | 0.92x [0.92, 0.93] | slowdown |
| ood-chat | fixed | 1 | 1.84x, 1.83x, 1.84x | 1.84x [1.83, 1.84] | speedup |
| ood-chat | fixed | 128 | 0.94x, 0.95x, 0.95x | 0.95x [0.94, 0.95] | inconclusive |
| ood-summary | natural | 1 | 1.45x, 1.43x, 1.42x | 1.43x [1.42, 1.45] | speedup |
| ood-summary | natural | 8 | 1.29x, 1.29x, 1.32x | 1.30x [1.29, 1.31] | speedup |
| ood-summary | natural | 128 | 1.01x, 1.00x, 0.99x | 1.00x [0.99, 1.01] | within 5% |
| ood-summary | fixed | 1 | 1.51x, 1.50x, 1.51x | 1.51x [1.50, 1.51] | speedup |
| ood-summary | fixed | 128 | 0.99x, 0.97x, 0.97x | 0.98x [0.97, 0.99] | within 5% |

Absolute throughput, TTFT p95, TPOT p50, E2E p50, acceptance and completion lengths for every deployment pair are in `paired-results.json`. Raw cells retain request records and metric snapshots. Natural output-length differences must be interpreted alongside the fixed-length check.
