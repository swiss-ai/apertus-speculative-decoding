# 70B DSpark, one epoch: Clariden (GH200) vs Beverin (MI300A)

The same drafter (`dspark-70b-opb-thinkoff-32n-alps-epoch1`, `model.safetensors`
sha256 `30ce666d...695b`), prompts (128 held-out OPB, `workloads/8b/opb-heldout.jsonl`),
engine limits (16,384 batched tokens, 256 seqs), greedy decoding and TP4 on one node.
Every request succeeded on both clusters.

- Clariden: debug partition, one arm per job, runtime `6ba76553`, image
  `vllm_apertus_1.5_release-arm64.sqsh` with the patches in each `patches.txt`
  (no vocabulary patch needed). Jobs 3865306 (k3,
  `dspark-alps-1epoch-clariden-k3-20261105T142058Z`), 3865308 (k7,
  `...-k7-20261105T142101Z`), 3865568 (plain, `...-plain-20261105T145654Z`).
  A first k3 job (3863699) failed at engine start writing flashinfer's cache
  in the over-quota home directory (fixed in `6ba76553`); its k7 follower was cancelled.
- Beverin: job 720056, `dspark-alps-1epoch-beverin-20261104T185916Z`.

Output tokens/s (speedup over plain on the same cluster):

| C | Clariden plain | Clariden k3 | Clariden k7 | Beverin plain | Beverin k3 | Beverin k7 |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 69.9 | 170.1 (2.43) | 212.6 (3.04) | 59.8 | 82.6 (1.38) | 87.2 (1.46) |
| 8 | 481.9 | 1071.6 (2.22) | 1254.2 (2.60) | 268.7 | 310.7 (1.16) | 353.1 (1.31) |
| 32 | 1343.3 | 2415.8 (1.80) | 2278.0 (1.70) | 689.1 | 527.6 (0.77) | 516.0 (0.75) |
| 64 | 2234.3 | 3156.5 (1.41) | 2584.0 (1.16) | 1014.8 | 591.9 (0.58) | 521.5 (0.51) |

Acceptance length is the same on both clusters (k3 2.87, k7 3.67-3.72; per
position identical to 0.01). Cost of one speculative step relative to a plain
decode step (TPOT p50 x acceptance length / plain TPOT p50):

| C | Clariden k3 | Clariden k7 | Beverin k3 | Beverin k7 |
| --- | --- | --- | --- | --- |
| 1 | 1.07 | 1.03 | 1.70 | 2.13 |
| 8 | 1.18 | 1.26 | 2.24 | 2.42 |
| 32 | 1.42 | 1.95 | 3.49 | 4.17 |
| 64 | 1.85 | 2.74 | 4.63 | 6.05 |

Reading:

- Acceptance is a property of the drafter, not of the cluster: the gap to the
  8B drafter (4.78 served at k7) is training. Training's greedy validation
  figure (`eal`) is 4.49 for this drafter and 5.94 for the 10-epoch 8B; both
  serve at ~0.8 of it.
- The slowdown under load on Beverin is the MI300A serving stack (ALPS image,
  upstream vLLM 0.31 ROCm with the pruned-head patch): a speculative step costs
  2-2.5x more there than on Clariden, and plain decoding is also 1.2-2.2x slower.
  On Clariden the drafter wins at every load measured.
