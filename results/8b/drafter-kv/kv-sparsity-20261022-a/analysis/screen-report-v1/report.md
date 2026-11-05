# Paired screen measurements

Ratios are paired candidate/native geometric means across two allocations.
Above 1 means more throughput/acceptance or higher latency. No CI is inferred.

| Workload | C | Policy | Throughput | Acceptance | p95 TTFT | p95 TPOT |
| --- | ---: | --- | ---: | ---: | ---: | ---: |
| chat | 1 | importance1024 | 0.7607 | 0.9939 | 1.1833 | 1.3371 |
| chat | 1 | importance512 | 0.7525 | 0.9719 | 1.2365 | 1.3394 |
| chat | 1 | recent512 | 0.7516 | 0.9684 | 1.2617 | 1.3486 |
| chat | 32 | importance1024 | 0.7779 | 0.9874 | 1.3347 | 1.1846 |
| chat | 32 | importance512 | 0.8136 | 0.9709 | 1.3944 | 1.1333 |
| chat | 32 | recent512 | 0.8124 | 0.9755 | 1.3983 | 1.1216 |
| chat | 256 | importance1024 | 0.8214 | 0.9929 | 1.1675 | 1.1994 |
| chat | 256 | importance512 | 0.8066 | 0.9706 | 1.1872 | 1.2160 |
| chat | 256 | recent512 | 0.8206 | 0.9713 | 1.2044 | 1.1871 |
| code | 1 | importance1024 | 0.7525 | 0.9950 | 1.2953 | 1.3513 |
| code | 1 | importance512 | 0.7659 | 0.9850 | 1.2959 | 1.3566 |
| code | 1 | recent512 | 0.7402 | 0.9603 | 1.4143 | 1.3676 |
| code | 32 | importance1024 | 0.8095 | 0.9986 | 1.2230 | 1.1878 |
| code | 32 | importance512 | 0.8255 | 0.9837 | 1.2664 | 1.1667 |
| code | 32 | recent512 | 0.7982 | 0.9641 | 1.2027 | 1.2103 |
| code | 256 | importance1024 | 0.8295 | 0.9977 | 1.0762 | 1.1897 |
| code | 256 | importance512 | 0.8191 | 0.9886 | 1.1462 | 1.2428 |
| code | 256 | recent512 | 0.8270 | 0.9650 | 1.0876 | 1.2215 |
| summarization | 1 | importance1024 | 0.7497 | 0.9473 | 1.0670 | 1.3403 |
| summarization | 1 | importance512 | 0.7238 | 0.8936 | 1.0723 | 1.3431 |
| summarization | 1 | recent512 | 0.7011 | 0.8661 | 1.0718 | 1.4199 |
| summarization | 32 | importance1024 | 0.8680 | 0.9543 | 1.0183 | 1.1076 |
| summarization | 32 | importance512 | 0.8420 | 0.8974 | 1.0446 | 1.1353 |
| summarization | 32 | recent512 | 0.8415 | 0.8712 | 1.0416 | 1.1013 |
| summarization | 256 | importance1024 | 0.9006 | 0.9536 | 1.0614 | 1.1776 |
| summarization | 256 | importance512 | 0.8843 | 0.8973 | 1.0740 | 1.1903 |
| summarization | 256 | recent512 | 0.8821 | 0.8719 | 1.0675 | 1.1805 |

| Policy | High-C throughput score | Draft storage ratio | Correctness | Gate |
| --- | ---: | ---: | --- | --- |
| importance1024 | 0.8336 | 0.5840 | True | False |
| importance512 | 0.8315 | 0.3082 | True | False |
| recent512 | 0.8299 | 0.3082 | True | False |

Source gate: passed=False, candidate=None.
Absolutes, counts, preemptions and input hashes are in CSV/JSON.
