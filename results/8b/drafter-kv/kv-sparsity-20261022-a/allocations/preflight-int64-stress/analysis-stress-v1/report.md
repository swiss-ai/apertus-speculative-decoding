# Installed-runtime C256 stress diagnostic

Job `3725531`, launch `c6675e217f9bc69352c5d5bcb273824a0f437390`. All four arms completed1,024 measured requests without failures or preemptions. This allocation is stability stress data, excluded from policy selection and confirmation.

| Arm | tok/s | Acceptance length | p95 TTFT ratio | p95 TPOT ratio | Throughput ratio |
| --- | ---: | ---: | ---: | ---: | ---: |
| full-custom | 3296.50 | 2.5333 | 1.0589 | 1.1606 | 0.8574 |
| importance512 | 3137.89 | 2.4469 | 1.2021 | 1.2134 | 0.8162 |

Native arithmetic mean: 3844.59tok/s, acceptance length2.5106. Ratios use the two native deployment summaries as the denominator.

Large-offset GPU reference cases above2**31, state kernels and query graph replay passed. The full backend retains substantial overhead even with similar acceptance. No candidate or expansion decision follows from this diagnostic.

Earlier preflight04 tiny cells had full-custom/native throughput geometric mean0.8072 and importance512/native0.7922. Their different workload size and prior runtime revision prevent pooling with this stress cell. Exact sources and hashes are in the adjacent JSON.
