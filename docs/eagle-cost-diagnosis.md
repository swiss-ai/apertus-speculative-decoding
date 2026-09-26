# EAGLE / 8B round-cost diagnosis

Updated 2026-09-23. Mix of **measured** T2 cells and **historical inferred**
smoke-screening numbers. Do not treat inferred round times as kernel traces.

## T2 campaign status

The config-phase driver finished on Clariden at 2026-09-19T22:10:51Z
(`campaign complete`). All **24 screening cells** (six configs × C=1/8 × two
deployment blocks) plus standalone **P2** exist under
`results/diagnostics-20260919/`. The campaign pid is dead; the queue is empty.

P1 profiling was **skipped** by a harness bug: `run_config` treated existing
unprofiled `c8/repeat-01/summary.json` files as “already measured.” That skip
is fixed. Training pilots still wait until P1 traces attribute the 103 ms
round. EAGLE 3.1 is **not** in this block (no 70B head).

| Config | Jobs | Blocks | Status |
| --- | --- | --- | --- |
| B0 operational baseline | 3441989, 3446578 | 1–2 | measured |
| B1 async-off, budget 8192 | 3446342, 3448414 | 1–2 | measured |
| B2 async-off, budget 7168 | 3446471, 3446749 | 1–2 | measured |
| N3 n-gram, budget 8192 | 3446532, 3446640 | 1–2 | measured |
| N3m n-gram, budget 7168 | 3446427, 3448363 | 1–2 | measured |
| D3 8B depth-3 draft | 3446273, 3447739 | 1–2 | measured (3442017 failed on empty `baseline_role=`) |
| P2 standalone 8B TP=4 | 3448470 | 1 | measured |
| P1 profiles B2/N3m/D3 | 3491208 (B2) | 1 | replica healthy 10:09Z; `/start_profile` 200; C=1 window running |

Canonical artifacts: `results/diagnostics-20260919/`. Break-even JSON files
pool both blocks’ B0 or B2 `t0`.

## Screening means (mechanistic_fixed256, ignore_eos, 256 tokens, 32 prompts)

Prefix caching off. Prompt tokens 26,002/cell (≈813/request). Success 32/32
on every cell below.

| Config | C=1 tok/s (blk1 / blk2) | vs B0 C=1 | C=8 tok/s (blk1 / blk2) | vs B0 C=8 | TPOT p50 C=1 (ms) | KV tokens | Async | Scheduled |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| B0 | 69.15 / 68.93 | 1.00× | 473.84 / 469.04 | 1.00× | 14.19 / 14.22 | 513,712 / 513,696 | on | 8192 |
| B1 | 63.73 / 63.74 | 0.92× | 434.81 / 439.10 | 0.92–0.93× | 15.42 / 15.38 | 513,696 | off | 8192 |
| B2 | 63.83 / 63.71 | 0.92× | 438.99 / 437.88 | 0.93× | 15.40 / 15.43 | 513,920 / 513,904 | off | 7168 |
| N3 | 75.28 / 78.06 | 1.09–1.13× | 392.77 / 439.79 | 0.83–0.93× | 13.71 / 12.02 | 416,624 | off | 8192 |
| N3m | 76.91 / 78.22 | 1.11–1.13× | 470.10 / 455.86 | 0.97–1.00× | 12.89 / 12.35 | 416,832 / 416,816 | off | 7168 |
| D3 | 23.53 / 19.72 | 0.34 / 0.29× | 173.11 / 179.38 | 0.37 / 0.38× | 41.86 / 50.28 | 251,488 | off | 7168 |
| P2 | 242.25 | 3.51× (8B) | 1723.07 | 3.65× (8B) | 4.00 | 2,249,424 | off | 8192 |

B2 versus B1 is a wash at this batch shape: shrinking the advertised budget
from 8192 to 7168 does **not** explain D3. Async-off (B1/B2 vs B0) is an
**8%** throughput tax and remains the operational bar’s gap, not the draft
collapse.

## D3 versus matched B2

| Cell | tok/s | vs B0 | vs B2 | Accept | `g` (prom.) | Inferred `t_round` (ms) | Break-even vs B2 (ms) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| blk1 C=1 | 23.53 | 0.341× | **0.369×** | 48.9% | 2.47 | 103.5 | 39.1 |
| blk1 C=8 | 173.11 | 0.367× | **0.395×** | 49.2% | 2.48 | 108.3 | 39.2 |
| blk2 C=1 | 19.72 | 0.286× | **0.309×** | 50.6% | 2.52 | 126.5 | 39.8 |
| blk2 C=8 | 179.38 | 0.380× | **0.409×** | 51.0% | 2.53 | 102.9 | 40.0 |

Pooled B2 `t0 = 15.821 ms` (four cells). Perfect-acceptance ceiling at
unchanged D3 cost is **0.46–0.62×**. The 103–126 ms round is **2.6–3.2×**
the matched break-even. Alignment training cannot close this by itself.

`t_round` is inferred as `TPOT_p50 × completion / (completion − accepted)`,
not from Nsight.

## N-gram (N3 / N3m) versus B2

| Cell | tok/s | vs B0 | vs B2 | Accept | `g` | Inferred `t_round` (ms) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| N3m blk1 C=1 | 76.91 | 1.114× | **1.206×** | 21.0% | 1.63 | 17.9 |
| N3m blk2 C=1 | 78.22 | 1.133× | **1.227×** | 22.2% | 1.67 | 17.5 |
| N3m blk1 C=8 | 470.10 | 0.997× | 1.072× | 22.9% | 1.69 | 20.1 |
| N3m blk2 C=8 | 455.86 | 0.967× | 1.040× | 20.8% | 1.63 | 22.7 |
| N3 blk1 C=1 | 75.28 | 1.090× | 1.181× | 21.1% | 1.63 | 19.1 |
| N3 blk2 C=1 | 78.06 | 1.131× | 1.225× | 22.8% | 1.68 | 17.1 |
| N3 blk1 C=8 | 392.77 | 0.833× | 0.899× | 20.3% | 1.61 | 25.2 |
| N3 blk2 C=8 | 439.79 | 0.933× | 1.006× | 20.1% | 1.60 | 22.9 |

On this non-copying corpus, n-gram still wins at C=1 because rounds are ~18 ms.
N3 C=8 is noisier (block 1 dipped to 0.83× B0). Do not treat the 11% C=1 gain
as the historical summarization result.

## P2 standalone 8B bound

Job `3448470` / `nid006904`. Same 8B checkpoint as the D3 drafter, TP=4,
async off, prefix cache off, batched/scheduled 8192. Served-name still says
`70B-diag-P2`; the 4.00 ms TPOT and 2.25M-token KV confirm the 8B weights.

| Cell | Output tok/s | TPOT p50 (ms) | TTFT p50 (ms) |
| --- | ---: | ---: | ---: |
| C=1 | 242.25 | **4.00** | 36.4 |
| C=8 | 1723.07 | 4.27 | 96.4 |

Three sequential standalone-like draft steps would be ~12 ms, plus one 70B
verify near B2’s 15 ms, an optimistic ~27 ms floor versus measured D3
`t_round` 103–126 ms. The 8B model/kernel is not the 103 ms by itself. P1
must attribute the embedded path (graphs, host sync, padding, NCCL,
verification). P2 is a bound, not an integrated TP experiment.

## Historical inferred cost (2026-09-13 smoke, not this campaign)

| Observation | Value | Status |
| --- | --- | --- |
| 8B depth-3 throughput vs operational baseline | 0.375–0.473× | historical |
| Acceptance | 60.5–79.7% | historical |
| Mean committed tokens / round `g` | 2.81–3.39 | historical |
| Inferred round time | 91–106 ms | inferred, not traced |
| Perfect-acceptance ceiling at unchanged cost | 0.53–0.65× | arithmetic |

Mechanistic D3 lands in the same round-cost band with lower acceptance.

## Decision table

| Observation | Next action |
| --- | --- |
| D3 `t_round` 103–126 ms vs ~39–40 ms B2 break-even | **P1** profiles of B2, N3m, D3 at C=1 then C=8 |
| Standalone 8B is 4.0 ms/token | Embedded path, not raw 8B FLOPs, is the first suspect |
| B2 ≈ B1 | Stop blaming the 7168 scheduled-token reservation at this shape |
| N3m C=1 is 1.21× B2 at `g ≈ 1.63` | Cheap proposer can win; confirm on copying workloads later |
| Perfect D3 acceptance still ≤0.62× at this cost | Do not start 8B or smaller-drafter training pilots yet |
| Graphs / CPU / collectives / verification | P1 traces decide which to patch first |
