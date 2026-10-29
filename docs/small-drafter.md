# Shallower DSpark drafters under load, Apertus 1.5 8B

Yu (Slack, 16 Oct): DSpark uses a 5-layer drafter to get a high
acceptance length, but past the saturation point every drafted token pays
for that depth. Can 3, 2 or even 1 layer win under load, accepting less
but costing less per step? He suggested a lightweight training run rather
than a 10-epoch one. EAGLE e10, a 1-layer autoregressive drafter trained
on the same data, is the existing 1-layer reference
([system-optimizations-8b.md](system-optimizations-8b.md), "EAGLE at
epoch 10").

Status 2026-10-29: phase A and the five-layer 6,200-step control are complete
(3722176, validation EAL 4.630). One-layer jobs 3722177 and 3769226 both failed
before training: conflicting CLI flags, then an inaccessible verifier path
embedded in the warm-start config. The latter is repaired by changing only that
path in a fresh initializer with the same physical drafter weights. New training
**3790212** first validates target/initializer/container hashes inside its allocation.
The afterok chain **3790216 → 3790218 → 3790219** performs validation/correctness,
the counterbalanced serving comparisons and automatic strict CPU review.
The comparison remains **incomplete**, with no serving recommendation. A future
5% G3 pass is a scientific result only; implementation evidence still requires
the separate fixed-k strict review (10% repeatable throughput plus latency gains,
with absolute TTFT/tail costs). Recovered
evidence, source hashes and current job/dependency receipts are in
[`results/8b/small-drafter/recovery-20261029/`](../results/8b/small-drafter/recovery-20261029/README.md).

## Summary

| question | result |
| --- | --- |
| does the fork serve a 1-3 layer DSpark drafter? | Not as shipped: it sizes `fc` by the drafter's layer count instead of its 5 target layers. Fixed by the opt-in overlay `serving/patches/vllm-apertus-dspark-num-layers.patch`; all depths then serve. |
| what does a layer cost? | At 256 requests the draft step falls from 10.5 ms (5 layers) to 7.8 / 6.3 / 5.1 ms (3/2/1) at k=4, and from 17.0 to 12.7 / 10.2 / 8.1 ms at k=7: 1.35-2.2 ms per layer. About 3.7-5.9 ms (fc, LM head over 131k tokens, Markov loop) does not shrink with depth. At low load the step is mostly fixed overhead (1 request, k=7: 2.0 ms for 5 layers, 1.1 for 1). |
| where the draft step goes (Nsight, eager draft) | At 256 requests and k=4 the 5-layer step is backbone 8.2 ms, LM head 1.9, Markov loop 0.2, fc and context K/V 0.8; the 1-layer step is backbone 1.5, LM head 1.7, Markov 0.2, fc and context K/V 0.4. A layer costs about 1.7 ms at k=4 and 2.5 ms at k=7; the LM head over the 131k-token vocabulary (1.7-3.3 ms) stays whatever the depth, and a 1-layer drafter spends more on it than on its layer. The Markov loop is negligible. |
| KV pool | 343k tokens (5 layers), 389k (3), 417k (2), 434k (1). At C=256 on the held-out prompts no arm is KV-bound (all run 256 requests), so the extra capacity does not raise throughput here. |
| ceiling: an N-layer drafter at the 5-layer acceptance, against DSpark e10's best k | 1 layer: +5.9% at C=64, +5.1% at C=128, +5.5% at C=256. 2 layers: +3.9-4.5%. 3 layers: +2.3-2.7%. |
| break-even acceptance | 92-93% of the 5-layer acceptance for 1 layer (C=256: 3.49 at k=4 vs 3.78), 94% for 2 layers, 96-97% for 3 layers. |
| untrained truncations (first N layers of e10) | 2.19 / 1.53 / 1.29 accepted tokens at k=7 for 3/2/1 layers vs 4.77: useless untrained, the warm start for training. |
| gate G3 (>= 5% over DSpark e10's best fixed k at some C >= 128, same job) | No verdict yet: the control trained, but the one-layer run failed before training and its replacement is queued. The projected ceiling remains about +5.5%; it is not a measured gain. |

In the waste-profiling cost model (`apertus_bench.waste_report`, decode
steps only, no prefill) the ceiling of a 1-layer drafter is +8.9-9.8% at
C=64-256 (2 layers +6.7-7.5%, 3 layers +4.0-4.6%), with the best length
shifted to k=2-3. That is the ceiling for decode-dominated traffic; the
held-out prompts (about 563 prompt and 480 output tokens) spend about
35-40 us of prefill per output token (fitted below), which brings it to the
5-6% above. The break-even acceptance comes out the same in both models:
91-92% of the 5-layer acceptance for 1 layer, 93-94% for 2, 96% for 3.

## Method

**Truncated drafters.** `methods/dspark/train/truncate_drafter.py` copies
DSpark e10 with only its first N decoder layers (plus fc, hidden_norm, norm,
the Markov and confidence heads; `num_hidden_layers` = N and the first N
`layer_types` in `config.json`). Weights are copied byte for byte; each copy
has `truncation.json` with both sha256 sums and the source's provenance files
as `source-*`. Sizes 1.10B / 0.80B / 0.49B parameters (3/2/1 layers) against
1.72B. Paths: `/iopsstor/scratch/cscs/faruk_zahiragic/spec-exp-data/small-drafter/drafters/dspark-e10-trunc-l{1,2,3}`
(sha256 2ce2b28f... l3, da48b949... l2, ea50bb1c... l1; source 002d1167...).

**Loader.** The fork's `DFlashQwen3Model` sizes `fc` as hidden size times the
number of target layers found in `eagle_config`/`dflash_config`, else times
its own `num_hidden_layers`. Speculators DSpark checkpoints carry their target
layers only as the top-level `target_layer_ids` (algos.py), so `fc` came out
as 4096 x N*4096 against the checkpoint's 4096 x 20480. The overlay uses
`target_layer_ids` in that case; 5-layer drafters are unaffected.

**Draft cost by depth.** `small-trunc-cost` serves the four depths with the
paper scheduler's engine-start profile (`cost-table.json`, `draft_ms_by_len`:
the draft step per draft length and request count, 20 repeats, dummy
batches). Only the table is used.

**Step model.** Per output token,

    tau = (V_k + D_N(k, R)) / (R * a) + P

with R running requests, a accepted tokens per step, D_N the N-layer draft
step from the cost table, V_k the rest of a decode step at draft length k
(verify, KV read, sampling, scheduling) and P the prefill time per output
token. `small-trunc-a` runs all four depths at k=7 in one job; their very
different acceptances and draft costs fit V_7 and P per concurrency, and the
model reproduces all four arms within 0.7%:

| C | V_7 (ms/step, without draft) | P (us/output token) | worst arm error |
| --- | --- | --- | --- |
| 64 | 23.4 | 40.4 | 0.7% |
| 256 | 82.3 | 35.0 | 0.6% |

V_k for the other lengths comes from DSpark e10's held-out grid
(k-grid-opb-a/b, jobs of 2026-10-10) with that P; the ceiling replaces D_5 by
D_N at the 5-layer acceptance, and the break-even acceptance is the a at which
the N-layer drafter matches e10's best k. Script:
`methods/dspark/scheduler/depth_ceiling.py`, output
`results/8b/small-drafter/depth-ceiling-phase-a.json`. The grid is from
another job (between-job spread up to ~5%), so these are bounds, not
measurements; phase D measures the trained drafters against e10 in the same
jobs.

## Results

### Draft step by depth (cost table, ms)

| requests | k | 5 layers | 3 layers | 2 layers | 1 layer |
| --- | --- | --- | --- | --- | --- |
| 1 | 3 | 1.74 | 1.33 | 1.08 | 0.86 |
| 1 | 7 | 1.99 | 1.54 | 1.29 | 1.06 |
| 64 | 3 | 2.73 | 2.05 | 1.65 | 1.35 |
| 64 | 7 | 5.54 | 4.15 | 3.34 | 2.61 |
| 128 | 3 | 4.53 | 3.46 | 2.74 | 2.24 |
| 128 | 7 | 9.29 | 6.79 | 5.55 | 4.47 |
| 256 | 1 | 4.07 | 3.03 | 2.47 | 1.95 |
| 256 | 3 | 8.32 | 6.26 | 5.00 | 4.06 |
| 256 | 4 | 10.47 | 7.84 | 6.34 | 5.08 |
| 256 | 7 | 16.95 | 12.68 | 10.22 | 8.13 |

Verify (target forward, same tables): 5.9 ms at 32 tokens, 10.9 at 256, 19.4
at 512, 34.4 at 1,024, 63.3 at 2,048, the same within 2% in all four arms.

### Where the draft step goes (small-trunc-nsys, Nsight Systems)

Mean GPU time per decode step of each NVTX range (`nvtx_gpu_proj_trace`,
decode-only steps of each 15 s window), with the draft run eagerly so that
its components are separate ranges (`eager_draft`; at 256 requests and
k=4 the eager step is 14% above the cost table for 5 layers and 7% below it
for 1 layer, so ratios rather than timings). Script `methods/dspark/scheduler/depth_profile.py`, output
`results/8b/small-drafter/depth-profile-phase-a.json`. At C=64 the 5-layer
windows have few decode-only steps (27-28 requests on average), so compare
the C=256 rows.

| layers | k | C | decode reqs | propose | backbone | LM head | Markov | fc | context K/V |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 5 | 4 | 1 | 1 | 3.15 | 1.35 | 0.30 | 0.10 | 0.05 | 0.06 |
| 1 | 4 | 1 | 1 | 1.05 | 0.23 | 0.30 | 0.10 | 0.05 | 0.03 |
| 5 | 4 | 256 | 254 | 11.99 | 8.19 | 1.86 | 0.22 | 0.37 | 0.41 |
| 1 | 4 | 256 | 255 | 4.74 | 1.49 | 1.73 | 0.21 | 0.35 | 0.09 |
| 5 | 7 | 1 | 1 | 4.66 | 1.76 | 0.30 | 0.17 | 0.06 | 0.06 |
| 1 | 7 | 1 | 1 | 1.76 | 0.25 | 0.29 | 0.17 | 0.06 | 0.03 |
| 5 | 7 | 256 | 253 | 18.75 | 12.34 | 3.22 | 0.40 | 0.60 | 0.61 |
| 1 | 7 | 256 | 255 | 8.40 | 2.45 | 3.26 | 0.38 | 0.60 | 0.14 |

(ms; `propose` is the whole draft step, the rest is sampling and launch
gaps.) Removing four layers saves 6.7 ms of backbone at k=4 and 9.9 ms at
k=7 at 256 requests; nothing else shrinks except the context K/V. Fitted
on the 5-layer windows, waste_report's component model
(`fit_dspark_components`) puts the backbone at 1.27 ms per layer at k=4 and
256 requests, close to the cost tables' 1.35 ms per layer; it misses about
1.4-1.9 ms of fixed per-step work, so it reads that much below the tables at
every depth.

### Truncated drafters at k=7 (small-trunc-a, held-out prompts)

| C | metric | 5 layers (e10) | 3 layers | 2 layers | 1 layer |
| --- | --- | --- | --- | --- | --- |
| 1 | tok/s | 584 | 292 | 213 | 185 |
| 64 | tok/s | 7,456 | 4,190 | 3,211 | 2,813 |
| 256 | tok/s | 8,605 | 4,871 | 3,717 | 3,236 |
| all | accepted / step | 4.75-4.80 | 2.19 | 1.53 | 1.29 |
| 256 | KV use max | 89% | 80% | 75% | 73% |
| - | KV pool (tokens) | 342,693 | 389,310 | 416,734 | 434,023 |

### Ceiling and break-even against DSpark e10's best fixed k

| C | e10 best k (tok/s) | N | ceiling (k) | vs best | break-even acceptance (k) |
| --- | --- | --- | --- | --- | --- |
| 64 | k=5 (7,895) | 3 | 8,109 (5) | +2.7% | 4.04 of 4.20 (5), 96% |
| | | 2 | 8,251 (5) | +4.5% | 3.93 of 4.20 (5), 94% |
| | | 1 | 8,361 (5) | +5.9% | 3.86 of 4.20 (5), 92% |
| 128 | k=3 (8,834) | 3 | 9,037 (3) | +2.3% | 3.17 of 3.28 (3), 97% |
| | | 2 | 9,181 (4) | +3.9% | 3.57 of 3.79 (4), 94% |
| | | 1 | 9,287 (4) | +5.1% | 3.51 of 3.79 (4), 93% |
| 256 | k=4 (9,453) | 3 | 9,702 (4) | +2.6% | 3.64 of 3.78 (4), 96% |
| | | 2 | 9,850 (4) | +4.2% | 3.56 of 3.78 (4), 94% |
| | | 1 | 9,977 (4) | +5.5% | 3.49 of 3.78 (4), 92% |

For comparison, EAGLE e10 (1 layer, autoregressive) accepts 1.87 / 2.60 /
3.22 / 3.72 tokens at k=1-4, 98-99% of DSpark e10, and its best k was
9,936 tok/s at C=256 in its own job (k-grid-opb-e/f): about where the 1-layer
DSpark ceiling is. A 1-layer *parallel* drafter would have to keep the
acceptance that EAGLE gets by drafting autoregressively.

## Decision: phase C cut to two runs (user, 2026-10-23)

The plan had seven equal-budget runs (about 0.25 epoch each: from scratch
with 1, 2, 3 and 5 layers, warm-started from e10's first 1, 2, 3 layers).
With the phase-A ceiling at about 5.5% end to end even at the full 5-layer
acceptance, and the gate at 5%, the user chose to run only two:

- `depth-l5-scratch` (job 3722176): 5 layers from scratch, the equal-budget
  control and the sanity check of the 8B training port;
- `depth-l1-warm` (job 3722177): 1 layer, warm-started from e10's first layer
  and heads (`dspark-e10-trunc-l1`).

Both: Yu's 8B recipe (`methods/dspark/train/submit-train.sh STAGE=8b`), 2
nodes, 6,200 steps with the cosine schedule over those steps (`EPOCHS=1
MAX_STEPS=6200`), then the validation epoch on Yu's held-out split. If the
1-layer drafter comes close to the 5-layer acceptance, further runs are a
new decision.

**Dataset** (job 3721370, 21 min): Yu's corpus prepared at 8,192 tokens;
1,809,684 rows, 180,969 held out, identical to his `val_ids.txt`
(`SPLIT_VERIFIED`; his `val_ids.py` matched 24/24 sampled rows). At
`/iopsstor/scratch/cscs/faruk_zahiragic/spec-exp-data/small-drafter/data`.

## Runs

| run | job | what |
| --- | --- | --- |
| truncation | 3721325 | `truncate_drafter.py` for N = 1, 2, 3 (debug) |
| `small-trunc-a-20261022T125905Z` | 3721350 | e10 and its 3/2/1-layer truncations at k=7, C=1/64/256 |
| `small-trunc-cost-20261022T125907Z` | 3721351 | failed at container start (overlay mount list too long for pyxis); not recorded |
| `small-trunc-cost-20261022T130123Z` | 3721367 | cost tables of the four depths (paper-scheduler profile; overlays at a short `VLLM_PATCH_DIR`) |
| `small-trunc-nsys-20261022T161244Z` | 3722940 | Nsight split of the draft step, 5 vs 1 layer at k=4/7, C=1/64/256, eager draft; reports (`.nsys-rep`) on the cluster under each arm's `nsys/` |
| dataset prepare | 3721370 | 8B dataset, Yu's split verified |
| `runs/depth-l5-scratch` | 3722176 | Completed 6,200 steps; validation EAL 4.630, accepted length 4.466. Training metrics, not serving acceptance. |
| `runs/depth-l1-warm` | 3722177 | Failed before training: `--from-pretrained` conflicts with `--num-layers`; no checkpoint. |
| `runs/depth-l1-warm-recovery-20261027` | 3769226 | Failed before optimizer step 0: initializer embeds an inaccessible private verifier path; no trained checkpoint. |
| first recovery continuation | 3769355/56/57 | Canceled after training failed; no benchmark measurements. Frozen source and receipts retained. |
| `runs/depth-l1-warm-recovery-20261029` | 3790212 | Rebound verifier path only, same physical initializer weights; allocated full hash gate then unchanged 6,200-step recipe. Normal, 2 nodes, 6 h. |
| recovery validation/correctness | 3790216 | Afterok 3790212; checkpoint/source/recipe and calibrated k7 correctness gates. |
| recovery comparison a | 3790218 | Afterok 3790216; original serving grid and full held-out prompt passes, two captures per deployment. |
| recovery comparison b and strict review | 3790219 | Afterok 3790218; counterbalanced measurements, calibrated correctness, original 5% G3 result, then automatic fixed-k >=10% throughput+latency implementation screen. |

The first queued continuation (3769344/45/46) was canceled while still waiting
on dependencies so the v2 jobs could use short, job-private pyxis overlay paths.
Its submissions and cancellation receipts are preserved; it produced no measurements.
