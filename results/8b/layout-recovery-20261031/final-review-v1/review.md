# Preserved layout comparison review

Campaign state: `historical_screening_only`. Scientific verdict remains pending.

All absolute metrics and both P/D baselines are in `comparisons.csv`; raw scopes/engine capacities/hashes are in `review.json`.
Effects of at least 10% are screening candidates only. Small gains never become implementation recommendations.

| Run | Arm | Comparison | Eligible/seen | Speedup range | Losing cells >10% |
|---|---|---|---:|---:|---:|
|dp4-a-20261023T084033Z|dspark-off-k3-dp4|same_layout_speculation|21/21|1.17–2.42x|0|
|dp4-a-20261023T084033Z|dspark-off-k5-dp4|same_layout_speculation|21/21|1.16–3.03x|0|
|dp4-b-20261023T122249Z|dspark-off-k7-dp4|same_layout_speculation|21/21|1.09–3.35x|0|
|dp4-b-20261023T122249Z|eagle-k3-dp4|same_layout_speculation|21/21|1.18–2.34x|0|
|dp4-c-20261023T182609Z|eagle-k5-dp4|same_layout_speculation|21/21|1.16–2.66x|0|
|dp4-c-20261023T182609Z|eagle-k7-dp4|same_layout_speculation|21/21|1.12–2.68x|0|
|pd-off-20261030T165007Z|pd-1p3d-dspark-off-k3|layout_vs_plain_dp4|18/21|0.67–2.34x|4|
|pd-off-20261030T165007Z|pd-1p3d-dspark-off-k3|same_layout_speculation|18/21|1.06–2.45x|0|
|pd-off-20261030T165007Z|pd-1p3d-eagle-k3|layout_vs_plain_dp4|19/21|0.68–2.30x|5|
|pd-off-20261030T165007Z|pd-1p3d-eagle-k3|same_layout_speculation|19/21|1.05–2.41x|0|
|pd-off-20261030T165007Z|pd-1p3d-plain|layout_vs_plain_dp4|20/21|0.51–1.02x|11|
|pd-off-20261030T165007Z|pd-2p2d-dspark-off-k3|layout_vs_plain_dp4|20/21|0.53–2.33x|6|
|pd-off-20261030T165007Z|pd-2p2d-dspark-off-k3|same_layout_speculation|19/21|1.08–2.40x|0|
|pd-off-20261030T165007Z|pd-2p2d-eagle-k3|layout_vs_plain_dp4|20/21|0.53–2.12x|6|
|pd-off-20261030T165007Z|pd-2p2d-eagle-k3|same_layout_speculation|19/21|1.01–2.14x|0|
|pd-off-20261030T165007Z|pd-2p2d-plain|layout_vs_plain_dp4|20/21|0.48–0.99x|11|
|preflight-20261028T122755Z|dspark-off-k7-tp4|same_layout_speculation|0/1|unestablished|0|
|preflight-20261028T122755Z|eagle-k7-tp4|same_layout_speculation|0/1|unestablished|0|
|think-layouts-20261022T122305Z|dspark-on-k3-dp4|same_layout_speculation|21/21|1.43–2.63x|0|
|think-layouts-20261022T122305Z|dspark-on-k3-tp4|layout_vs_plain_dp4|21/21|0.81–4.44x|4|
|think-layouts-20261022T122305Z|dspark-on-k3-tp4|same_layout_speculation|21/21|1.03–2.29x|0|
|think-layouts-20261022T122305Z|dspark-on-k7-dp4|same_layout_speculation|21/21|1.32–3.69x|0|
|think-layouts-20261022T122305Z|dspark-on-k7-tp4|layout_vs_plain_dp4|0/6|unestablished|0|
|think-layouts-20261022T122305Z|dspark-on-k7-tp4|same_layout_speculation|0/6|unestablished|0|
|think-layouts-20261022T122305Z|plain-tp4|layout_vs_plain_dp4|21/21|0.37–1.97x|9|
|think-pd-20261030T083755Z|pd-1p3d-dspark-on-k3|layout_vs_plain_dp4|20/21|0.91–2.51x|0|
|think-pd-20261030T083755Z|pd-1p3d-dspark-on-k3|same_layout_speculation|19/21|1.26–2.54x|0|
|think-pd-20261030T083755Z|pd-1p3d-plain|layout_vs_plain_dp4|20/21|0.70–1.00x|8|
|tp2dp2-20261030T131540Z|dspark-off-k3-tp2dp2|same_layout_speculation|24/24|1.02–2.38x|0|
|tp2dp2-20261030T131540Z|dspark-off-k5-tp2dp2|same_layout_speculation|24/24|0.92–2.93x|0|
|tp2dp2-20261030T131540Z|eagle-k3-tp2dp2|same_layout_speculation|24/24|1.04–2.38x|0|
|tp2dp2-20261030T131540Z|eagle-k5-tp2dp2|same_layout_speculation|24/24|0.96–2.65x|0|
|tp4-a-20261023T093209Z|dspark-off-k3-tp4|same_layout_speculation|24/24|0.84–2.29x|1|
|tp4-a-20261023T093209Z|dspark-off-k5-tp4|same_layout_speculation|0/17|unestablished|0|
|tp4-a2-20261023T141405Z|dspark-off-k5-tp4|same_layout_speculation|24/24|0.62–2.88x|3|
|tp4-b2-20261024T132206Z|eagle-k3-tp4|same_layout_speculation|24/24|0.74–2.04x|2|
|tp4-c1-20261024T145347Z|eagle-k5-tp4|same_layout_speculation|24/24|0.65–2.40x|2|
|tp4-c2-20261024T160931Z|eagle-k7-tp4|same_layout_speculation|0/7|unestablished|0|
|tp4-dspark-off-k7-20261029T042233Z|dspark-off-k7-tp4|same_layout_speculation|24/24|0.55–3.15x|3|
|tp4-dspark-on-k7-20261030T064246Z|dspark-on-k7-tp4|same_layout_speculation|21/21|0.58–3.32x|3|
|tp4-eagle-k7-20261030T022433Z|eagle-k7-tp4|same_layout_speculation|24/24|0.72–2.62x|2|

Original scientific gates/claim limits:

- Screening data only: independent deployment confirmation repeats and confidence intervals are not established.
- Calibrated per-layout correctness controls are not established by these load tests.
- Fixed-concurrency tests do not establish an open-loop SLO knee or sustainable traffic capacity.
- Load-generator capacity needs its own evidence; client success alone does not prove it.
- Natural-EOS outputs can differ across deployments; compare completion volumes alongside throughput.
- P/D load/acceptance counters cover only the first decode instance, not the whole node.
- GPU resident memory includes preallocated KV pools; it is not the minimum memory required.

Missing stage analyses: 0. A terminal Slurm state alone is not campaign completion.

GPU-memory windows assume the CSV clock shares the arm-status UTC offset. Verify the reported coverage; missing fields stay unavailable.
These are observed resident allocations, including preallocated KV pools. Negative deltas are savings only within the declared metric scope.
