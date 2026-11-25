# Native reasoning effort phase2 results

Decision: **retain_shared**

| Block | Drafter | Effort | Accepted draft tokens/round | Latency speedup | Decode speedup | Cap fraction |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| 0 | plain | low | unavailable | 1.000 | 1.000 | 0.000 |
| 0 | plain | medium | unavailable | 1.000 | 1.000 | 0.000 |
| 0 | plain | high | unavailable | 1.000 | 1.000 | 0.000 |
| 0 | low | low | 2.690 | 2.460 | 2.592 | 0.000 |
| 0 | low | medium | 1.972 | 2.101 | 2.033 | 0.000 |
| 0 | low | high | 1.612 | 1.785 | 1.796 | 0.000 |
| 0 | medium | low | 2.578 | 2.144 | 2.524 | 0.000 |
| 0 | medium | medium | 1.999 | 2.065 | 2.074 | 0.000 |
| 0 | medium | high | 1.608 | 1.747 | 1.798 | 0.000 |
| 0 | high | low | 2.573 | 2.295 | 2.516 | 0.000 |
| 0 | high | medium | 1.961 | 1.815 | 2.047 | 0.000 |
| 0 | high | high | 1.632 | 2.049 | 1.795 | 0.000 |
| 0 | mixed | low | 2.667 | 2.254 | 2.582 | 0.000 |
| 0 | mixed | medium | 1.995 | 2.032 | 2.075 | 0.000 |
| 0 | mixed | high | 1.553 | 1.957 | 1.744 | 0.000 |
| 1 | plain | low | unavailable | 1.000 | 1.000 | 0.000 |
| 1 | plain | medium | unavailable | 1.000 | 1.000 | 0.000 |
| 1 | plain | high | unavailable | 1.000 | 1.000 | 0.000 |
| 1 | low | low | 2.574 | 2.547 | 2.438 | 0.000 |
| 1 | low | medium | 1.939 | 2.363 | 2.044 | 0.000 |
| 1 | low | high | 1.541 | 1.800 | 1.786 | 0.000 |
| 1 | medium | low | 2.640 | 2.423 | 2.476 | 0.000 |
| 1 | medium | medium | 1.988 | 2.366 | 2.085 | 0.000 |
| 1 | medium | high | 1.557 | 1.766 | 1.791 | 0.000 |
| 1 | high | low | 2.593 | 2.456 | 2.452 | 0.000 |
| 1 | high | medium | 1.955 | 2.335 | 2.061 | 0.000 |
| 1 | high | high | 1.619 | 1.723 | 1.838 | 0.000 |
| 1 | mixed | low | 2.582 | 2.432 | 2.392 | 0.000 |
| 1 | mixed | medium | 1.988 | 2.143 | 2.085 | 0.000 |
| 1 | mixed | high | 1.638 | 1.747 | 1.853 | 0.000 |
| 2 | plain | low | unavailable | 1.000 | 1.000 | 0.000 |
| 2 | plain | medium | unavailable | 1.000 | 1.000 | 0.000 |
| 2 | plain | high | unavailable | 1.000 | 1.000 | 0.000 |
| 2 | low | low | 2.610 | 2.718 | 2.540 | 0.000 |
| 2 | low | medium | 1.970 | 2.044 | 2.082 | 0.000 |
| 2 | low | high | 1.665 | 1.740 | 1.849 | 0.000 |
| 2 | medium | low | 2.691 | 2.645 | 2.612 | 0.000 |
| 2 | medium | medium | 1.963 | 1.755 | 2.092 | 0.000 |
| 2 | medium | high | 1.555 | 1.861 | 1.785 | 0.000 |
| 2 | high | low | 2.656 | 2.713 | 2.574 | 0.000 |
| 2 | high | medium | 1.913 | 1.735 | 2.043 | 0.000 |
| 2 | high | high | 1.685 | 1.650 | 1.871 | 0.000 |
| 2 | mixed | low | 2.604 | 2.691 | 2.549 | 0.000 |
| 2 | mixed | medium | 1.992 | 1.726 | 2.106 | 0.000 |
| 2 | mixed | high | 1.618 | 1.730 | 1.831 | 0.000 |

Report storage/training costs before recommending multiple heads; this does not evaluate adapters, offloading or production capacity.

| Head | Training tokens | Stored checkpoint bytes | Draft tensor bytes |
| --- | ---: | ---: | ---: |
| low | 1572864 | 1569566723 | 1569534208 |
| medium | 1572864 | 1569566723 | 1569534208 |
| high | 1572864 | 1569566723 | 1569534208 |
| mixed | 1572864 | 1569566723 | 1569534208 |

serialized tensor payload at original dtype; projected weight-only residence, excludes KV cache, buffers, activations; deployments load one head at a time.
