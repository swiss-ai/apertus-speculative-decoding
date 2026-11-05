# 70B DSpark, one epoch: first serving eval (Beverin)

Beverin job 720056 on nid002410, 2026-11-04, runtime `repo-277edd39`
(spec `spec.yaml` = `experiments/system-70b/dspark-alps-1epoch.yaml`).
One MI300A node, every arm a whole-node TP4 engine run one after another;
ALPS image `vllm-rocm@sha256:7d95232c...` (vLLM 0.31.0, torch 2.13.0+rocm10.0.0)
with `vllm031-dspark-apertus-pruned-head.patch` mounted (`environment-patched.toml`).
Drafter: `/iopsstor/scratch/cscs/faruk_zahiragic/apertus-dspark70b/heads/dspark-70b-opb-thinkoff-32n-alps-epoch1`
(`model.safetensors` sha256 `30ce666d...695b`; training run
`results/70b/dspark/beverin-32n-alps-1epoch-20261101`). Workload: the 128
held-out OPB prompts of the 8B evaluation (`workloads/8b/opb-heldout.jsonl`,
sha256 `186d38cd...`), greedy, seed 1. All 1,536 requests succeeded.

| C | plain tok/s | k3 tok/s (x) | k7 tok/s (x) | plain TPOT p50 | k3 TPOT | k7 TPOT |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 59.8 | 82.6 (1.38) | 87.2 (1.46) | 16.3 ms | 9.6 | 9.4 |
| 8 | 268.7 | 310.7 (1.16) | 353.1 (1.31) | 28.1 | 21.9 | 18.3 |
| 32 | 689.1 | 527.6 (0.77) | 516.0 (0.75) | 37.5 | 45.8 | 42.4 |
| 64 | 1014.8 | 591.9 (0.58) | 521.5 (0.51) | 53.0 | 85.5 | 87.2 |

Acceptance (stable across concurrency):

- k3: mean acceptance length 2.87; per position 0.79 0.61 0.48.
- k7: 3.70; per position 0.76 0.56 0.42 0.32 0.26 0.21 0.18.

Training's greedy validation figure (`eal`) is 4.49 (4.36 is the
rejection-sampling `accept_len`); served k7 reaches 3.70, about 0.8 of it,
the same ratio as the 10-epoch 8B drafter (5.94 vs 4.78). DSpark wins at low
concurrency and loses from C=32 on this hardware. The same drafter on Clariden
GH200 has the same acceptance and wins at every load: see
`../dspark-alps-1epoch-clariden-vs-beverin.md`. The loss here is the MI300A
serving stack, not the drafter.

`replica.out` engine logs are kept on the cluster next to the original
output directory.
