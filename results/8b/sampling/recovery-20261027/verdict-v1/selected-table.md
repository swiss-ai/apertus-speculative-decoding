| Mode / workload / sampling | C | DSpark draft k | Output tok/s, plain → spec | Ratio | GPU-s / M output tokens, plain → spec | Mean request s, plain → spec | Accepted / step |
|---|---:|---:|---:|---:|---:|---:|---:|
| off / opb / default | 1 | 7 | 163 → 536 | 3.29× | 6,134.0 → 1,864.7 | 2.91 → 0.88 | 4.55 |
| off / arena-hard-clean / default | 64 | 3 | 6,455 → 9,449 | 1.46× | 154.9 → 105.8 | 4.86 → 3.44 | 2.89 |
| off / arena-hard-clean / default | 256 | 3 | 11,814 → 11,611 | 0.98× | 84.6 → 86.1 | 10.98 → 11.48 | 2.90 |
| off / arena-hard-clean / t1-kall | 256 | 7 | 12,452 → 7,743 | 0.62× | 80.3 → 129.1 | 10.89 → 17.89 | 3.00 |
| on / opb-think / greedy | 1 | 7 | 171 → 647 | 3.77× | 5,831.4 → 1,546.0 | 5.56 → 1.47 | 5.15 |
| on / arena-hard-clean / default | 64 | 3 | 6,080 → 10,207 | 1.68× | 164.5 → 98.0 | 14.60 → 8.92 | 2.95 |
| on / opb-think / default | 256 | 3 | 9,109 → 11,454 | 1.26× | 109.8 → 87.3 | 23.00 → 18.91 | 3.11 |
| on / arena-hard-clean / default | 256 | 3 | 10,283 → 12,207 | 1.19× | 97.2 → 81.9 | 35.43 → 30.39 | 2.96 |
| on / arena-hard-clean / t1-kall | 256 | 3 | 10,864 → 10,478 | 0.96× | 92.0 → 95.4 | 34.14 → 35.85 | 2.34 |
