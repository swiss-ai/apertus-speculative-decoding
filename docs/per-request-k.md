# Per-request draft length for DSpark, Apertus 1.5 8B

Experiment T2 of the 16 Oct sync: control the draft length more finely
than the DSpark paper scheduler, per request and per task. The paper
scheduler on vLLM (with the per-step draft cut and the KV-read cost term)
tracks the best fixed draft length k at every load but never beats it,
and adjusting k with concurrency does no better
([dspark-scheduler-8b.md](dspark-scheduler-8b.md),
[system-optimizations-8b.md](system-optimizations-8b.md)). Yu (16 Oct):
rejected drafts are wasted compute once the GPU is compute-bound. The
sync also asks to predict the best k per prompt, and a serving gateway
that knows each request's task could set the draft length per task.

Status 2026-10-22: offline headroom (A1, A2) done on the held-out
calibration dump and a new labelled dump on a mixed workload; gate G2 not
passed, so no serving patch. The fixed k=1/2/4 arms on the mixed workload
(mixed-calib-b, job 3721369) are still queued; they check the simulator's
best fixed k there (k=2 at C>=128).

## Summary

| question | result |
| --- | --- |
| how much could per-request lengths gain? (oracle) | Verifying and drafting exactly each request's accepted run: **+13-18% over the best fixed k** at C=64-256, on the held-out prompts and on mixed traffic. Per-step lengths for the whole batch (the draft cut) get half of that even as an oracle (+5-12%). |
| implementable per-request policy | Per-request draft blocks set by the scheduler from each request's two-step-old survival, one slot of slack, the GPU verifying the best prefix on the current survival: **+2.4% to +6.1% over the best fixed k** (mixed), +4.1% to +7.8% (held-out), at C=64-256, in a simulator that overstates the measured paper scheduler by 3.5-7% on the held-out prompts; corrected, +0.6% to +4.1%. |
| task label from a gateway | A per-task length alone (no confidence head): +2-4% at C=64-128, nothing at C=256. Recalibrating the head per task adds nothing: it is 5-9% overconfident on summarization and chat, but the lengths follow its ranking, which recalibration hardly changes. |
| why the confidence head cannot cut more | The DSpark backbone drafts a request's whole block in parallel before the head can score it, and it is 2/3 of the per-slot drafting cost. Lengths that follow the *current* confidence can only cut the LM head and Markov steps (1/3): +2-4%, less than the lagged block. |
| gate G2 (>=5% over best fixed k and the load-based schedule at some C>=64) | **Not passed.** Raw simulator +6.1% (mixed) / +7.8% (held-out) at C=128, but it overstates the measured paper scheduler by 3.5-7%; corrected, +0.6% to +4.1%. Phase B (the overlay) not started. |

## Method

### Simulator

`methods/dspark/scheduler/headroom_sim.py` replays a confidence dump: the
lines `VLLM_DSPARK_CONF_DUMP` writes with DSpark e10 at k=7 verifying every
slot (per verification: raw head logits of the 7 drafts, accepted run). Each
request becomes a trajectory of (prefix survival from the STS-calibrated
head, accepted run). A closed loop of C slots replays them; every step a
policy picks per-request lengths, the request gains `1 + min(a, L_r)`
tokens (the run at a shorter length is the k=7 run cut to it; checked: the
dump's `1 + E[min(a, 3)]` = 3.18 against 3.27 measured at k=3) and a
finished request's slot takes the next request and pays its prefill.

Step time: `verify_ms(R + sum L_r) + draft + kv_ms_per_token * context +
overhead`, from the cost table the paper scheduler profiles at engine start
(kv-cost-a, GH200: verify 5.7 ms up to 32 tokens, 7.8 ms at 128, 34.6 ms at
1,024; draft per length and batch; KV read 36 us per 1,000 context tokens).
Drafting costs either the step's longest length (per-step cut, from the
table) or per (request, slot) pair, from a fit of the same table: `1.29 ms
+ 2.6 us * R + L * (6 us + 8.2 us * R)` (rms 0.2 ms). Of the 8.2 us per pair,
the 5-layer backbone is ~2/3 (2.4 of 3.5 GFLOP per query token: hidden 4096,
MLP 21504; LM head over the 131k draft vocab 1.07 GFLOP; Markov bias 0.07).
The per-step overhead and the prefill cost per prompt token are fitted to
measured plain and fixed-k throughput: held-out grid k-grid-opb-a/b, 2.8 ms
and 32 us/token, rms error 2.9% over 28 points; mixed, mixed-calib-a, 0.6 ms
and 31 us/token, rms 2.2%. The mixed fit predicts the k=7 arm it was not fitted
to within 2.5%.

### Policies

- **fixed k** (k=1..7). The load-based schedule ("adjust k with
  concurrency") is the best fixed k at each C, the envelope; the simulator
  runs one C at a time, so it equals the best fixed k.
- **paper top-K**: greedy admission by descending survival (Algorithm 1),
  the best prefix for tokens per ms, drafting all 7 slots; **+ cut**: drafting
  only up to the longest verified slot. On current survival (an upper bound
  for the real, lagged budget).
- **per-request, per step (bound)**: the same admission with per-pair drafting
  cost, current survival for both drafting and verifying: requires cutting
  the backbone by confidences the backbone itself produces, so not causal.
- **lagged budgets**: the per-slot row counts of that plan two steps earlier,
  applied slot by slot to the current survival (the slot-j confidence is known
  before slot j's token); "head cut only" keeps the backbone at 7 slots, the
  causal version.
- **stale lengths**: the scheduler sets each request's block from its
  two-step-old survival (the newest under async scheduling, as the paper's
  lagged budget); **stale +1, GPU verify**: one slot more is drafted and the
  GPU verifies the best prefix on the current survival. Both are
  implementable with a variable-length drafter batch.
- **task label**: the label's mean observed survival per slot instead of the
  head (one length per task and load), or as a per-task recalibration of the
  head.
- **oracle**: lengths = the accepted run.

### Mixed workload

`tools/make-mixed-workload.py` (manifest `workloads/8b/mixed.manifest.json`,
text on the cluster): 128 held-out Open-PerfectBlend, 128 chat, 128
summarization, 128 Arena-Hard clean, 32 probe math, 32 probe HumanEval
prompts, shuffled, each with `vllm_xargs: {"task": label}` sent with the
request (the bench client sends a row's `vllm_xargs`). The dump overlay
`serving/patches/vllm-apertus-dspark-conf-dump-task.patch` adds the label and
the context length to each dump line. Accepted tokens per step at k=7 by
label (dump, mixed-calib-a): summarization 2.12, chat 2.62, Arena-Hard 3.86,
held-out 4.79, probe math 6.94, probe code 6.43: the same as each workload
alone.

## Results

Throughput relative to the best fixed k at the same C in the simulator
(the load-based schedule equals it at each C). First row: which k that is
and its speedup over plain.

Held-out calibration dump (`paper-calib`, 3,466 requests, 298k
verifications, homogeneous Open-PerfectBlend traffic):

| policy | C=64 | C=128 | C=256 |
| --- | --- | --- | --- |
| best fixed k (sim) | 4: 1.66x | 5: 1.30x | 3: 1.15x |
| paper top-K + cut (per step) | +4.1% | +2.5% | +1.7% |
| oracle, one length per step | +11.5% | +10.0% | +9.4% |
| oracle, per request | +17.8% | +16.3% | +15.9% |
| per-request, current survival (not causal) | +9.5% | +12.0% | +8.2% |
| per-request, current survival, LM head cut only | +6.2% | +5.3% | +1.6% |
| per-request stale lengths | +3.3% | +4.5% | +1.1% |
| per-request stale +1, GPU verify | +5.9% | +7.8% | +4.1% |

Mixed workload (`mixed-calib-a`, 4,521 labelled requests, 445k
verifications):

| policy | C=64 | C=128 | C=256 |
| --- | --- | --- | --- |
| best fixed k (sim) | 3: 1.30x | 2: 1.13x | 2: 1.07x |
| paper top-K + cut (per step) | -0.4% | +1.8% | -2.7% |
| oracle, one length per step | +8.0% | +9.6% | +5.2% |
| oracle, per request | +16.5% | +17.8% | +13.2% |
| per-request, current survival (not causal) | +8.2% | +9.5% | +5.9% |
| per-request, current survival, LM head cut only | +2.3% | +4.2% | -1.0% |
| per-request stale lengths | +1.3% | +3.5% | -0.5% |
| per-request stale +1, GPU verify | +4.7% | +6.1% | +2.4% |
| per-task length (label only) | +2.2% | +3.8% | -0.1% |

(The lagged slot budgets with the backbone cut too, +5.8% to +9.4%, track
the non-causal bound within 0.2%: the lag of the budget itself costs
nothing. The per-task recalibrated head is within 0.2% of the plain head.)

- **The ceiling is per request, not per step.** On mixed traffic one
  length per step, even chosen by an oracle, leaves half of the per-request
  oracle's gain: a batch of 64+ requests nearly always contains one that
  should draft 7, so the per-step cut drafts 7 (in the simulator the paper
  scheduler with and without cut are identical to 0.1% at every C).
- **Most of the ceiling needs the confidences before the backbone.** The
  head scores slot j from the backbone's hidden state for slot j, which the
  backbone computes for the whole block at once; what the current
  confidence can still cut, the LM head and Markov steps, is a third of the
  per-slot cost, worth +2-6% at C=64-128 and nothing at C=256.
- **Lagged per-request blocks are the best causal option.** Setting each
  request's block from its survival two steps back, plus one slot, and
  trimming the verification on the GPU by the current survival gives
  +4.7% / +6.1% / +2.4% (mixed) and +5.9% / +7.8% / +4.1% (held-out) at
  C=64 / 128 / 256. Without the slack and the GPU trim (the first port's
  CPU lengths, now with a variable drafter batch) it is +1-4%.
- **A task label is a weak prior.** A per-task length captures the
  difference between summarization (2.1 accepted at k=7) and math (6.9),
  but not the spread within a task (on the held-out prompts alone the
  per-request mean accepted run ranges from 2.3 to 6.7 between the 10th and
  90th percentile), and it is only known at admission: +2-4% at C<=128,
  nothing at C=256. Per-request history overtakes it after two steps.

**How far to trust the simulator.** It reproduces measured fixed-k and
plain throughput within 3% (rms 2.2-2.9%; the mixed k=7 arm, not fitted,
within 2.5%), but adaptive policies run steps it does not model: decode
batches of uneven shape (FULL graphs only for decode-only steps, piecewise
with prefill), the GPU top-K and index building. On the held-out prompts
the simulator puts the paper scheduler with cut 1.7-6.3% above fixed k=3
at C=64-256, where kv-cost-a/b measured it at -0.8% to -1.7% (means of the
two jobs: 0.992x, 0.991x, 0.983x of k=3). Corrected by that gap (7.1%,
3.9%, 3.5%), the best causal policy predicts +1.0%, +4.1% and +0.6% over
k=3 on the held-out prompts, and the per-request oracle +12%. A
per-request patch also has to build the variable drafter batch on the GPU
and replay it in captured graphs, which the per-step patches did not
need.

## Gate

**G2 not passed.** The best implementable policy (lagged per-request
blocks with one slot of slack and GPU verification) predicts +6.1%
(mixed) and +7.8% (held-out) over the best fixed k at C=128 in the raw
simulator, but the same simulator overstates the existing paper scheduler
by 3.5-7% against its measurement; corrected, the prediction is +0.6% to
+4.1% at C=64-256, below the 5% gate and below the ~5% between-job noise.
The remaining headroom is in information the drafter does not have when
it must decide: the per-request oracle stays at +12-18%, and two thirds of
it needs acceptance estimates before the parallel backbone runs.

Phase B (the per-request draft overlay) was therefore not started. What
would change the outcome:

- a confidence signal available before the backbone (a per-request
  acceptance predictor on the previous step's hidden state, trained like
  the confidence head), which the simulator puts at up to +8-9% over the
  best fixed k at C=64-128 (the "not causal" row, before the correction);
- a cheaper backbone per slot (a smaller block or fewer layers), which
  shifts the cost to the part the current confidence can cut;
- measuring the adaptive-step overhead directly (the Nsight Systems arm of
  the waste-profiling work) to replace the correction above by a cost term.

## Runs

| sweep | job | what |
| --- | --- | --- |
| `paper-calib-20261015T145635Z` (existing) | 3669651 | held-out calibration dump, DSpark e10 k=7, 512 prompts, 298k verifications (no labels) |
| `k-grid-opb-{a,b}-20261010T1454*Z` (existing) | 3632467, 3632468 | measured fixed k=1..7 and plain on the held-out prompts (simulator fit) |
| `kv-cost-{a,b}-20261015T2306*Z` (existing) | 3672289, 3672290 | cost table; paper scheduler + cut + KV vs k=3 measured |
| `mixed-calib-b` (queued) | 3721369 | mixed workload: plain, DSpark e10 k=1, k=2, k=4 |
| `mixed-calib-a-20261022T130127Z` | 3721368 | mixed workload: plain, DSpark e10 k=3, k=5, k=7 with the labelled dump (450k verifications; on the cluster, sha256 in the record) |

Simulator inputs and outputs: `results/8b/per-request-k/` (cost table,
measured points of the fit, per-policy results per C). Reproduce with

    python3 methods/dspark/scheduler/headroom_sim.py OUT.json DUMP.jsonl \
        --cost-table results/8b/per-request-k/cost-table-kv-cost-a.json \
        --sts methods/dspark/scheduler/sts-dspark-e10-8b.json \
        --measured results/8b/per-request-k/measured-mixed.json \
        --labels opb chat summarization arena-hard probe-math probe-code

with the mixed dump (`/iopsstor/scratch/cscs/faruk_zahiragic/spec-exp-data/per-request-k/conf-dump-mixed-calib-a-3721368.jsonl`,
sha256 b4ee34d7...) or the paper-calib dump
(`results/8b/sweeps/paper-calib-20261015T145635Z/dspark-e10-k7-calib/conf-dump.jsonl`
in the main cluster checkout) with `measured-opb-heldout.json`.
