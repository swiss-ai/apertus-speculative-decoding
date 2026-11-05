# DSpark e10 k=3: per-task speedups

Speedup = DSpark output tokens/s ÷ plain output tokens/s on the same prompts, geometric mean over deployment pairs, with the exact pair bootstrap 95% interval. ▼ slowdown (interval below 0.95), = within ±5%, ? inconclusive; unmarked with an interval = speedup above 1.05. Greedy, thinking off, natural EOS at each source's output cap, 32 hash-selected prompts per slice, whole passes per level.

## Controlled (Clariden, plain and DSpark on the same GPU)

Pairs: controlled-20261104T133618Z/pair-1, controlled-20261104T133618Z/pair-2, controlled-20261104T133618Z/pair-3, controlled-20261104T133618Z/pair-4.

### By class

| Class | C=1 | C=8 | C=32 | C=128 | Acc. length |
|---|---|---|---|---|---|
| id | 2.61x (2.20–2.86) | 2.57x (2.19–2.83) | 2.15x (1.81–2.41) | 1.38x (1.23–1.45) | 3.50 |
| near-id | 2.30x (2.00–2.63) | 2.16x (1.79–2.58) | 1.82x (1.47–2.12) | 1.25x (1.13–1.40) | 3.12 |
| ood-task | 1.58x (1.31–2.24) | 1.44x (1.25–2.14) | 1.25x (1.07–1.76) | 0.99x (0.82–1.13) | 2.25 |
| ood-language | 1.32x (1.09–1.63) | 1.36x (1.09–1.87) | 1.24x (1.00–1.88) | 0.85x (0.67–1.36) | 1.76 |

Class cells: geometric mean of the slices (range over slices).

### By slice

| Slice | Class | Acc. length | C=1 | C=8 | C=32 | C=128 |
|---|---|---|---|---|---|---|
| opb | id | 3.29 | 2.48x [2.47, 2.49] | 2.44x [2.39, 2.49] | 2.07x [2.03, 2.09] | 1.45x [1.43, 1.46] |
| math-gsm8k | id | 3.75 | 2.82x [2.80, 2.83] | 2.78x [2.72, 2.85] | 2.41x [2.38, 2.44] | 1.43x [1.43, 1.44] |
| probe-math | id | 3.82 | 2.86x [2.84, 2.89] | 2.83x [2.82, 2.84] | 2.31x [2.31, 2.32] | 1.36x [1.36, 1.36] |
| probe-humaneval | id | 3.71 | 2.76x [2.74, 2.77] | 2.66x [2.64, 2.67] | 2.21x [2.17, 2.25] | 1.42x [1.40, 1.44] |
| multi-turn | id | 2.95 | 2.20x [2.18, 2.22] | 2.19x [2.16, 2.22] | 1.81x [1.77, 1.86] | 1.23x [1.23, 1.24] |
| code-multi | near-id | 3.52 | 2.63x [2.62, 2.64] | 2.58x [2.51, 2.61] | 2.12x [2.10, 2.13] | 1.40x [1.39, 1.41] |
| json-schema | near-id | 3.73 | 2.58x [2.57, 2.59] | 2.23x [2.22, 2.23] | 1.75x [1.75, 1.76] | 1.16x [1.16, 1.16] |
| tool-call | near-id | 2.95 | 2.16x [2.15, 2.18] | 1.79x [1.76, 1.82] | 1.47x [1.46, 1.48] | 1.17x [1.16, 1.18] |
| lang-eng | near-id | 2.66 | 2.00x [1.99, 2.01] | 2.02x [1.99, 2.03] | 1.74x [1.71, 1.77] | 1.13x [1.12, 1.14] |
| arena-hard-clean | near-id | 2.95 | 2.25x [2.24, 2.25] | 2.25x [2.23, 2.25] | 2.01x [1.97, 2.05] | 1.31x [1.30, 1.32] |
| arena-hard-contaminated | near-id | 2.92 | 2.22x [2.20, 2.23] | 2.20x [2.16, 2.22] | 1.89x [1.84, 1.92] | 1.35x [1.34, 1.35] |
| chat | ood-task | 2.07 | 1.56x [1.55, 1.57] | 1.27x [1.26, 1.28] | 1.07x [1.06, 1.07] | 0.91x [0.91, 0.91] ▼ |
| summarization | ood-task | 1.95 | 1.45x [1.44, 1.45] | 1.36x [1.35, 1.37] | 1.20x [1.18, 1.21] | 1.02x [1.01, 1.03] = |
| long-context | ood-task | 2.44 | 1.51x [1.50, 1.52] | 1.25x [1.23, 1.27] | 1.14x [1.11, 1.18] | 1.13x [1.11, 1.14] |
| translate-to-en | ood-task | 3.08 | 2.24x [2.22, 2.26] | 2.14x [2.13, 2.15] | 1.76x [1.74, 1.78] | 1.11x [1.10, 1.11] |
| translate-from-en | ood-task | 1.74 | 1.31x [1.30, 1.32] | 1.34x [1.33, 1.35] | 1.20x [1.18, 1.22] | 0.82x [0.82, 0.82] ▼ |
| lang-deu | ood-language | 1.61 | 1.22x [1.21, 1.23] | 1.21x [1.12, 1.31] | 1.02x [1.01, 1.03] = | 0.74x [0.73, 0.75] ▼ |
| lang-fra | ood-language | 1.84 | 1.40x [1.40, 1.41] | 1.38x [1.35, 1.41] | 1.25x [1.23, 1.28] | 0.82x [0.81, 0.82] ▼ |
| lang-ita | ood-language | 1.66 | 1.25x [1.25, 1.26] | 1.31x [1.30, 1.31] | 1.14x [1.12, 1.16] | 0.84x [0.83, 0.85] ▼ |
| lang-spa | ood-language | 1.98 | 1.49x [1.49, 1.50] | 1.69x [1.65, 1.73] | 1.36x [1.34, 1.38] | 0.92x [0.91, 0.93] ▼ |
| lang-roh | ood-language | 1.76 | 1.32x [1.30, 1.34] | 1.23x [1.08, 1.41] | 1.49x [1.40, 1.62] | 0.97x [0.94, 0.99] ? |
| lang-zho | ood-language | 1.93 | 1.46x [1.46, 1.47] | 1.48x [1.47, 1.48] | 1.33x [1.30, 1.35] | 0.85x [0.82, 0.87] ▼ |
| lang-jpn | ood-language | 1.64 | 1.25x [1.24, 1.26] | 1.24x [1.21, 1.28] | 1.15x [1.13, 1.17] | 0.74x [0.74, 0.75] ▼ |
| lang-hin | ood-language | 1.63 | 1.24x [1.22, 1.25] | 1.25x [1.24, 1.26] | 1.10x [1.07, 1.12] | 0.78x [0.76, 0.81] ▼ |
| lang-ara | ood-language | 1.44 | 1.09x [1.09, 1.10] | 1.09x [1.09, 1.10] | 1.08x [1.03, 1.13] ? | 0.77x [0.76, 0.78] ▼ |
| lang-swa | ood-language | 1.47 | 1.12x [1.10, 1.13] | 1.12x [1.05, 1.18] | 1.00x [0.96, 1.05] = | 0.67x [0.67, 0.68] ▼ |
| lang-yor | ood-language | 2.00 | 1.44x [1.43, 1.44] | 1.87x [1.86, 1.88] | 1.88x [1.76, 2.00] | 1.36x [1.31, 1.41] |
| lang-amh | ood-language | 2.13 | 1.63x [1.62, 1.64] | 1.65x [1.62, 1.67] | 1.26x [1.24, 1.29] | 0.86x [0.85, 0.87] ▼ |

### Per-request decode speedup (TPOT p50, plain ÷ DSpark)

Unaffected by a cell's slowest stream; throughput above includes queueing and tails.

| Slice | C=1 | C=8 | C=32 | C=128 |
|---|---|---|---|---|
| opb | 2.75x | 2.62x | 2.07x | 1.41x |
| math-gsm8k | 2.90x | 2.86x | 2.32x | 1.37x |
| probe-math | 2.95x | 2.92x | 2.39x | 1.42x |
| probe-humaneval | 2.88x | 2.79x | 2.22x | 1.31x |
| multi-turn | 2.31x | 2.15x | 1.76x | 1.15x |
| code-multi | 2.78x | 2.70x | 2.12x | 1.31x |
| json-schema | 2.90x | 2.46x | 1.69x | 1.21x |
| tool-call | 2.29x | 1.91x | 1.46x | 1.09x |
| lang-eng | 2.08x | 2.10x | 1.78x | 1.03x |
| arena-hard-clean | 2.26x | 2.28x | 1.95x | 1.18x |
| arena-hard-contaminated | 2.25x | 2.22x | 1.91x | 1.23x |
| chat | 1.94x | 1.86x | 1.50x | 1.04x |
| summarization | 1.51x | 1.47x | 1.24x | 1.06x |
| long-context | 1.82x | 1.34x | 1.40x | 1.46x |
| translate-to-en | 2.37x | 2.26x | 1.74x | 1.09x |
| translate-from-en | 1.35x | 1.35x | 1.11x | 0.74x |
| lang-deu | 1.24x | 1.28x | 1.10x | 0.68x |
| lang-fra | 1.44x | 1.46x | 1.25x | 0.74x |
| lang-ita | 1.27x | 1.29x | 1.12x | 0.68x |
| lang-spa | 1.48x | 1.49x | 1.29x | 0.75x |
| lang-roh | 1.21x | 1.23x | 1.07x | 0.67x |
| lang-zho | 1.56x | 1.54x | 1.32x | 0.79x |
| lang-jpn | 1.25x | 1.26x | 1.07x | 0.64x |
| lang-hin | 1.21x | 1.21x | 1.08x | 0.65x |
| lang-ara | 1.04x | 1.05x | 0.92x | 0.57x |
| lang-swa | 1.01x | 1.01x | 0.90x | 0.54x |
| lang-yor | 1.27x | 1.24x | 0.97x | 0.83x |
| lang-amh | 1.62x | 1.58x | 1.38x | 0.87x |

### Speedup against acceptance length

| C | Speedup ≈ a + b·length | R² | Break-even length |
|---|---|---|---|
| 1 | 0.04 + 0.72·L | 0.98 | 1.32 |
| 8 | 0.17 + 0.66·L | 0.89 | 1.26 |
| 32 | 0.35 + 0.49·L | 0.79 | 1.33 |
| 128 | 0.39 + 0.28·L | 0.76 | 2.22 |

### Latency at C=1 (medians, ms)

| Slice | TTFT plain | TTFT DSpark | TPOT plain | TPOT DSpark | E2E plain | E2E DSpark |
|---|---|---|---|---|---|---|
| opb | 21 | 26 | 5.8 | 2.1 | 2998 | 1119 |
| math-gsm8k | 21 | 25 | 5.8 | 2.0 | 2124 | 742 |
| probe-math | 21 | 25 | 5.8 | 2.0 | 1930 | 668 |
| probe-humaneval | 21 | 24 | 5.8 | 2.0 | 1213 | 497 |
| multi-turn | 21 | 25 | 5.8 | 2.5 | 1086 | 499 |
| code-multi | 21 | 25 | 5.8 | 2.1 | 1398 | 554 |
| json-schema | 21 | 25 | 5.8 | 2.0 | 370 | 144 |
| tool-call | 36 | 38 | 5.8 | 2.5 | 1787 | 874 |
| lang-eng | 21 | 25 | 5.8 | 2.8 | 971 | 446 |
| arena-hard-clean | 21 | 25 | 5.8 | 2.6 | 2769 | 1360 |
| arena-hard-contaminated | 21 | 25 | 5.8 | 2.6 | 2915 | 1219 |
| chat | 26 | 28 | 5.8 | 3.0 | 1498 | 767 |
| summarization | 74 | 75 | 5.9 | 3.9 | 1320 | 989 |
| long-context | 486 | 488 | 6.2 | 3.4 | 1632 | 1104 |
| translate-to-en | 21 | 25 | 5.8 | 2.4 | 558 | 269 |
| translate-from-en | 21 | 25 | 5.8 | 4.3 | 628 | 503 |
| lang-deu | 21 | 25 | 5.8 | 4.6 | 684 | 623 |
| lang-fra | 21 | 25 | 5.8 | 4.0 | 1351 | 1159 |
| lang-ita | 21 | 25 | 5.8 | 4.5 | 726 | 570 |
| lang-spa | 20 | 25 | 5.8 | 3.9 | 900 | 607 |
| lang-roh | 21 | 25 | 5.7 | 4.7 | 432 | 445 |
| lang-zho | 21 | 25 | 5.8 | 3.7 | 1037 | 790 |
| lang-jpn | 21 | 25 | 5.8 | 4.6 | 1740 | 1451 |
| lang-hin | 21 | 25 | 5.8 | 4.7 | 789 | 588 |
| lang-ara | 21 | 25 | 5.8 | 5.5 | 639 | 567 |
| lang-swa | 21 | 25 | 5.8 | 5.7 | 607 | 610 |
| lang-yor | 21 | 25 | 5.7 | 4.5 | 178 | 143 |
| lang-amh | 29 | 30 | 5.8 | 3.6 | 2984 | 1736 |

## Deployed (the Kubernetes pod, plain and DSpark windows via the gateway)

Pairs: round-1, round-2.

### By class

| Class | C=1 | C=8 | C=32 | C=128 | Acc. length |
|---|---|---|---|---|---|
| id | 2.59x (2.20–2.81) | 2.45x (2.14–2.65) | 2.12x (1.80–2.32) | 1.40x (1.24–1.46) | 3.50 |
| near-id | 2.18x (1.94–2.48) | 1.84x (1.23–2.27) | 1.61x (1.02–1.99) | 1.20x (1.02–1.34) | 3.12 |
| ood-task | 1.57x (1.30–2.22) | 1.44x (1.26–2.10) | 1.07x (0.74–1.25) | 1.02x (0.87–1.19) | 2.25 |
| ood-language | 1.32x (1.08–1.61) | 1.28x (0.94–1.64) | 1.22x (0.98–1.74) | 0.87x (0.70–1.39) | 1.76 |

Class cells: geometric mean of the slices (range over slices).

### By slice

| Slice | Class | Acc. length | C=1 | C=8 | C=32 | C=128 |
|---|---|---|---|---|---|---|
| opb | id | 3.29 | 2.48x [2.47, 2.49] | 2.28x [2.24, 2.32] | 2.06x [2.04, 2.09] | 1.44x [1.43, 1.46] |
| math-gsm8k | id | 3.75 | 2.78x [2.76, 2.81] | 2.65x [2.59, 2.71] | 2.31x [2.25, 2.38] | 1.46x [1.42, 1.51] |
| probe-math | id | 3.79 | 2.81x [2.79, 2.84] | 2.65x [2.62, 2.68] | 2.32x [2.30, 2.33] | 1.43x [1.35, 1.53] |
| probe-humaneval | id | 3.72 | 2.72x [2.69, 2.75] | 2.59x [2.55, 2.64] | 2.16x [2.12, 2.20] | 1.44x [1.44, 1.45] |
| multi-turn | id | 2.96 | 2.20x [2.18, 2.21] | 2.14x [2.11, 2.17] | 1.80x [1.74, 1.85] | 1.24x [1.20, 1.28] |
| code-multi | near-id | 3.51 | 2.48x [2.37, 2.58] | 2.27x [1.98, 2.59] | 1.99x [1.83, 2.16] | 1.30x [1.18, 1.42] |
| json-schema | near-id | 3.73 | 2.26x [2.01, 2.54] | 1.67x [1.35, 2.09] | 1.02x [0.61, 1.73] ? | 1.06x [0.97, 1.16] ? |
| tool-call | near-id | 2.93 | 1.98x [1.87, 2.10] | 1.79x [1.79, 1.79] | 1.26x [1.13, 1.41] | 1.02x [0.90, 1.16] ? |
| lang-eng | near-id | 2.66 | 1.94x [1.91, 1.97] | 1.23x [0.78, 1.97] ? | 1.84x [1.83, 1.85] | 1.20x [1.19, 1.20] |
| arena-hard-clean | near-id | 2.95 | 2.23x [2.21, 2.24] | 2.19x [2.19, 2.20] | 1.93x [1.91, 1.95] | 1.31x [1.29, 1.33] |
| arena-hard-contaminated | near-id | 2.93 | 2.21x [2.20, 2.23] | 2.12x [2.10, 2.13] | 1.92x [1.91, 1.93] | 1.34x [1.32, 1.36] |
| chat | ood-task | 2.07 | 1.55x [1.54, 1.56] | 1.26x [1.25, 1.27] | 1.07x [1.07, 1.08] | 0.92x [0.91, 0.92] ▼ |
| summarization | ood-task | 1.95 | 1.44x [1.43, 1.44] | 1.37x [1.37, 1.37] | 1.22x [1.21, 1.24] | 1.03x [1.02, 1.03] = |
| long-context | ood-task | 2.44 | 1.49x [1.49, 1.50] | 1.27x [1.25, 1.31] | 1.17x [1.16, 1.17] | 1.14x [1.13, 1.14] |
| translate-to-en | ood-task | 3.07 | 2.22x [2.18, 2.26] | 2.10x [2.06, 2.13] | 0.74x [0.34, 1.68] ? | 1.19x [1.17, 1.21] |
| translate-from-en | ood-task | 1.73 | 1.30x [1.29, 1.31] | 1.34x [1.31, 1.37] | 1.25x [1.21, 1.30] | 0.87x [0.86, 0.89] ▼ |
| lang-deu | ood-language | 1.61 | 1.22x [1.21, 1.23] | 1.16x [1.16, 1.16] | 1.06x [1.05, 1.07] | 0.80x [0.79, 0.81] ▼ |
| lang-fra | ood-language | 1.85 | 1.40x [1.39, 1.41] | 1.41x [1.36, 1.47] | 1.19x [1.16, 1.23] | 0.85x [0.84, 0.86] ▼ |
| lang-ita | ood-language | 1.65 | 1.25x [1.24, 1.26] | 1.29x [1.26, 1.33] | 1.19x [1.17, 1.21] | 0.89x [0.86, 0.91] ▼ |
| lang-spa | ood-language | 1.98 | 1.49x [1.47, 1.50] | 1.56x [1.51, 1.60] | 1.30x [1.28, 1.32] | 0.95x [0.94, 0.95] ? |
| lang-roh | ood-language | 1.78 | 1.34x [1.32, 1.36] | 1.06x [1.06, 1.06] | 1.46x [1.44, 1.47] | 1.04x [1.00, 1.07] ? |
| lang-zho | ood-language | 1.92 | 1.46x [1.45, 1.47] | 1.50x [1.48, 1.51] | 1.31x [1.25, 1.37] | 0.79x [0.72, 0.88] ▼ |
| lang-jpn | ood-language | 1.65 | 1.26x [1.25, 1.27] | 1.21x [1.20, 1.22] | 1.13x [1.12, 1.14] | 0.75x [0.74, 0.76] ▼ |
| lang-hin | ood-language | 1.64 | 1.24x [1.23, 1.26] | 1.25x [1.24, 1.27] | 1.09x [1.09, 1.09] | 0.79x [0.77, 0.80] ▼ |
| lang-ara | ood-language | 1.43 | 1.08x [1.08, 1.09] | 1.16x [1.08, 1.26] | 1.08x [0.97, 1.20] ? | 0.82x [0.78, 0.87] ▼ |
| lang-swa | ood-language | 1.48 | 1.13x [1.11, 1.14] | 0.94x [0.84, 1.07] ? | 0.98x [0.94, 1.03] ? | 0.70x [0.68, 0.73] ▼ |
| lang-yor | ood-language | 2.00 | 1.42x [1.41, 1.44] | 1.34x [1.00, 1.81] ? | 1.74x [1.66, 1.81] | 1.39x [1.27, 1.53] |
| lang-amh | ood-language | 2.11 | 1.61x [1.59, 1.62] | 1.64x [1.63, 1.65] | 1.31x [1.31, 1.32] | 0.89x [0.87, 0.91] ▼ |

### Per-request decode speedup (TPOT p50, plain ÷ DSpark)

Unaffected by a cell's slowest stream; throughput above includes queueing and tails.

| Slice | C=1 | C=8 | C=32 | C=128 |
|---|---|---|---|---|
| opb | 2.75x | 2.67x | 2.06x | 1.41x |
| math-gsm8k | 2.90x | 2.80x | 2.33x | 1.39x |
| probe-math | 2.94x | 2.86x | 2.44x | 1.40x |
| probe-humaneval | 2.89x | 2.80x | 2.33x | 1.38x |
| multi-turn | 2.31x | 2.12x | 1.84x | 1.17x |
| code-multi | 2.73x | 2.76x | 2.20x | 1.33x |
| json-schema | 2.88x | 2.45x | 1.75x | 1.17x |
| tool-call | 2.26x | 1.92x | 1.37x | 1.03x |
| lang-eng | 2.06x | 2.13x | 1.79x | 1.14x |
| arena-hard-clean | 2.25x | 2.27x | 1.98x | 1.19x |
| arena-hard-contaminated | 2.23x | 2.26x | 1.91x | 1.24x |
| chat | 1.95x | 1.88x | 1.44x | 1.00x |
| summarization | 1.52x | 1.48x | 1.23x | 1.07x |
| long-context | 1.84x | 1.33x | 1.45x | 1.45x |
| translate-to-en | 2.37x | 2.24x | 1.88x | 1.13x |
| translate-from-en | 1.35x | 1.35x | 1.21x | 0.75x |
| lang-deu | 1.26x | 1.28x | 1.12x | 0.72x |
| lang-fra | 1.45x | 1.45x | 1.24x | 0.79x |
| lang-ita | 1.27x | 1.28x | 1.20x | 0.77x |
| lang-spa | 1.50x | 1.50x | 1.34x | 0.81x |
| lang-roh | 1.23x | 1.20x | 1.14x | 0.72x |
| lang-zho | 1.53x | 1.53x | 1.33x | 0.86x |
| lang-jpn | 1.25x | 1.25x | 1.09x | 0.67x |
| lang-hin | 1.23x | 1.22x | 1.09x | 0.72x |
| lang-ara | 1.01x | 1.04x | 0.96x | 0.56x |
| lang-swa | 1.02x | 0.99x | 0.92x | 0.61x |
| lang-yor | 1.29x | 1.28x | 1.06x | 0.84x |
| lang-amh | 1.59x | 1.57x | 1.36x | 0.88x |

### Speedup against acceptance length

| C | Speedup ≈ a + b·length | R² | Break-even length |
|---|---|---|---|
| 1 | 0.12 + 0.68·L | 0.96 | 1.29 |
| 8 | 0.25 + 0.58·L | 0.80 | 1.30 |
| 32 | 0.55 + 0.37·L | 0.44 | 1.21 |
| 128 | 0.46 + 0.25·L | 0.68 | 2.14 |

### Latency at C=1 (medians, ms)

| Slice | TTFT plain | TTFT DSpark | TPOT plain | TPOT DSpark | E2E plain | E2E DSpark |
|---|---|---|---|---|---|---|
| opb | 42 | 34 | 5.8 | 2.1 | 2999 | 1151 |
| math-gsm8k | 32 | 34 | 5.8 | 2.0 | 2152 | 760 |
| probe-math | 31 | 34 | 5.8 | 2.0 | 1939 | 679 |
| probe-humaneval | 31 | 34 | 5.8 | 2.0 | 1430 | 464 |
| multi-turn | 38 | 35 | 5.8 | 2.5 | 1236 | 493 |
| code-multi | 32 | 31 | 5.8 | 2.1 | 1484 | 567 |
| json-schema | 31 | 30 | 5.8 | 2.0 | 383 | 154 |
| tool-call | 53 | 54 | 5.8 | 2.6 | 1986 | 936 |
| lang-eng | 31 | 32 | 5.8 | 2.8 | 955 | 525 |
| arena-hard-clean | 35 | 35 | 5.8 | 2.6 | 2893 | 1333 |
| arena-hard-contaminated | 34 | 33 | 5.8 | 2.6 | 2868 | 1259 |
| chat | 44 | 45 | 5.8 | 3.0 | 1509 | 780 |
| summarization | 89 | 92 | 5.9 | 3.9 | 1370 | 997 |
| long-context | 520 | 532 | 6.1 | 3.3 | 1628 | 1136 |
| translate-to-en | 32 | 30 | 5.8 | 2.4 | 583 | 278 |
| translate-from-en | 30 | 31 | 5.8 | 4.3 | 638 | 514 |
| lang-deu | 29 | 33 | 5.8 | 4.6 | 706 | 617 |
| lang-fra | 29 | 33 | 5.8 | 4.0 | 1498 | 1024 |
| lang-ita | 27 | 32 | 5.8 | 4.6 | 719 | 591 |
| lang-spa | 28 | 32 | 5.8 | 3.9 | 906 | 620 |
| lang-roh | 28 | 31 | 5.8 | 4.7 | 450 | 428 |
| lang-zho | 29 | 33 | 5.8 | 3.8 | 1083 | 769 |
| lang-jpn | 30 | 33 | 5.8 | 4.6 | 1826 | 1470 |
| lang-hin | 30 | 32 | 5.8 | 4.7 | 814 | 600 |
| lang-ara | 27 | 31 | 5.8 | 5.7 | 604 | 628 |
| lang-swa | 28 | 32 | 5.8 | 5.7 | 597 | 616 |
| lang-yor | 27 | 30 | 5.8 | 4.5 | 184 | 150 |
| lang-amh | 41 | 41 | 5.8 | 3.6 | 2993 | 1804 |

## Live ÷ controlled speedup

| Slice | C=1 | C=8 | C=32 | C=128 |
|---|---|---|---|---|
| opb | 1.00 | 0.93 | 1.00 | 1.00 |
| math-gsm8k | 0.99 | 0.95 | 0.96 | 1.02 |
| probe-math | 0.98 | 0.94 | 1.00 | 1.05 |
| probe-humaneval | 0.99 | 0.98 | 0.98 | 1.02 |
| multi-turn | 1.00 | 0.98 | 0.99 | 1.01 |
| code-multi | 0.94 | 0.88 | 0.94 | 0.93 |
| json-schema | 0.87 | 0.75 | 0.58 | 0.92 |
| tool-call | 0.92 | 1.00 | 0.86 | 0.87 |
| lang-eng | 0.97 | 0.61 | 1.06 | 1.06 |
| arena-hard-clean | 0.99 | 0.98 | 0.96 | 1.00 |
| arena-hard-contaminated | 1.00 | 0.96 | 1.02 | 1.00 |
| chat | 1.00 | 0.99 | 1.00 | 1.01 |
| summarization | 0.99 | 1.01 | 1.02 | 1.01 |
| long-context | 0.99 | 1.02 | 1.02 | 1.01 |
| translate-to-en | 0.99 | 0.98 | 0.42 | 1.07 |
| translate-from-en | 0.99 | 1.00 | 1.05 | 1.07 |
| lang-deu | 1.00 | 0.96 | 1.04 | 1.08 |
| lang-fra | 1.00 | 1.02 | 0.95 | 1.04 |
| lang-ita | 1.00 | 0.99 | 1.04 | 1.05 |
| lang-spa | 1.00 | 0.92 | 0.96 | 1.02 |
| lang-roh | 1.01 | 0.86 | 0.97 | 1.08 |
| lang-zho | 0.99 | 1.02 | 0.99 | 0.93 |
| lang-jpn | 1.01 | 0.98 | 0.98 | 1.00 |
| lang-hin | 1.01 | 1.00 | 0.99 | 1.00 |
| lang-ara | 0.99 | 1.07 | 1.00 | 1.07 |
| lang-swa | 1.01 | 0.85 | 0.98 | 1.04 |
| lang-yor | 0.99 | 0.72 | 0.92 | 1.03 |
| lang-amh | 0.99 | 1.00 | 1.04 | 1.04 |
