# Native reasoning effort phase2 results

Decision: **retain_shared**

| Block | Drafter | Effort | Accepted draft tokens/round | Latency speedup | Decode speedup | Cap fraction |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| 0 | plain | low | unavailable | 1.000 | 1.000 | 0.000 |
| 0 | plain | medium | unavailable | 1.000 | 1.000 | 0.000 |
| 0 | plain | high | unavailable | 1.000 | 1.000 | 0.000 |
| 0 | low | low | 2.639 | 2.505 | 2.458 | 0.000 |
| 0 | low | medium | 1.868 | 1.765 | 1.937 | 0.000 |
| 0 | low | high | 1.627 | 2.012 | 1.809 | 0.000 |
| 0 | medium | low | 2.620 | 2.281 | 2.439 | 0.000 |
| 0 | medium | medium | 2.023 | 1.975 | 2.040 | 0.000 |
| 0 | medium | high | 1.555 | 2.148 | 1.763 | 0.000 |
| 0 | high | low | 2.641 | 2.334 | 2.449 | 0.000 |
| 0 | high | medium | 1.897 | 1.808 | 1.955 | 0.000 |
| 0 | high | high | 1.698 | 2.070 | 1.848 | 0.000 |
| 0 | mixed | low | 2.674 | 2.249 | 2.474 | 0.000 |
| 0 | mixed | medium | 1.931 | 2.139 | 1.969 | 0.000 |
| 0 | mixed | high | 1.632 | 1.723 | 1.819 | 0.000 |
| 1 | plain | low | unavailable | 1.000 | 1.000 | 0.000 |
| 1 | plain | medium | unavailable | 1.000 | 1.000 | 0.000 |
| 1 | plain | high | unavailable | 1.000 | 1.000 | 0.000 |
| 1 | low | low | 2.634 | 2.532 | 2.478 | 0.000 |
| 1 | low | medium | 1.964 | 1.955 | 2.010 | 0.000 |
| 1 | low | high | 1.417 | 1.883 | 1.710 | 0.000 |
| 1 | medium | low | 2.564 | 2.224 | 2.435 | 0.000 |
| 1 | medium | medium | 1.976 | 2.410 | 2.028 | 0.000 |
| 1 | medium | high | 1.476 | 2.033 | 1.760 | 0.000 |
| 1 | high | low | 2.651 | 2.666 | 2.497 | 0.000 |
| 1 | high | medium | 1.923 | 2.269 | 1.991 | 0.000 |
| 1 | high | high | 1.627 | 1.757 | 1.867 | 0.000 |
| 1 | mixed | low | 2.601 | 2.739 | 2.464 | 0.000 |
| 1 | mixed | medium | 1.987 | 2.281 | 2.034 | 0.000 |
| 1 | mixed | high | 1.526 | 1.891 | 1.790 | 0.000 |
| 2 | plain | low | unavailable | 1.000 | 1.000 | 0.000 |
| 2 | plain | medium | unavailable | 1.000 | 1.000 | 0.000 |
| 2 | plain | high | unavailable | 1.000 | 1.000 | 0.000 |
| 2 | low | low | 2.614 | 2.519 | 2.449 | 0.000 |
| 2 | low | medium | 1.965 | 1.774 | 2.051 | 0.000 |
| 2 | low | high | 1.576 | 1.745 | 1.807 | 0.000 |
| 2 | medium | low | 2.593 | 2.723 | 2.459 | 0.000 |
| 2 | medium | medium | 1.972 | 2.077 | 2.040 | 0.000 |
| 2 | medium | high | 1.612 | 1.736 | 1.836 | 0.000 |
| 2 | high | low | 2.640 | 2.723 | 2.488 | 0.000 |
| 2 | high | medium | 1.969 | 1.763 | 2.054 | 0.000 |
| 2 | high | high | 1.608 | 1.681 | 1.833 | 0.000 |
| 2 | mixed | low | 2.677 | 2.558 | 2.513 | 0.000 |
| 2 | mixed | medium | 1.940 | 1.575 | 2.040 | 0.000 |
| 2 | mixed | high | 1.613 | 1.682 | 1.827 | 0.000 |

Report storage/training costs before recommending multiple heads; this does not evaluate adapters, offloading or production capacity.

| Head | Training tokens | Stored checkpoint bytes | Draft tensor bytes |
| --- | ---: | ---: | ---: |
| low | 1572864 | 1569566723 | 1569534208 |
| medium | 1572864 | 1569566723 | 1569534208 |
| high | 1572864 | 1569566723 | 1569534208 |
| mixed | 1572864 | 1569566723 | 1569534208 |

serialized tensor payload at original dtype; projected weight-only residence, excludes KV cache, buffers, activations; deployments load one head at a time.
