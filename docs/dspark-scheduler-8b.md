# The DSpark scheduler on vLLM, Apertus 1.5 8B

Part of the system-level work item from the 2 Oct sync (canvas "Spec
Decoding Sync 2 Oct: Action Items by Owner", Faruk's part): make speculative
decoding pay off under load. The load tests of 3 Oct
([loadtest-8b.md](loadtest-8b.md)) showed both drafters faster than the
plain target up to C=8, even at C=32 and 8-15% slower from C=64 on. This
document covers the draft length and DSpark's confidence-scheduled
verification: the fixed-length baselines, the two ports of the scheduler to
vLLM and their runs. The other canvas items (sampling, drafter KV cache,
several GPUs and disaggregation, MineDraft) are written up in
`docs/system-optimizations-8b.md` on branch `eagle-8b-stage-a`.

Status 2026-10-17: the per-step draft cut (`draft-cut-a/b`) and upstream
vLLM's adaptive verification (`upstream-av-a/b`) are recorded under Runs
but not yet written up here; the KV-read cost term (`kv-cost-a/b`) is
specified, its results not yet recorded.

## Summary

| item | result |
| --- | --- |
| draft length on the current checkpoints | **Draft length is the big lever.** DSpark e10 at k=3 instead of 7: summarization never below plain (1.01-1.06x at C>=128 instead of 0.89x), chat at 0.93-1.00x instead of 0.74x, at the cost of ~10% at C=1. EAGLE e1 is best at k=3 at low load and k=1 under load (1.02x at C=256). On the in-distribution held-out prompts acceptance is higher and the best lengths are longer: DSpark k=5 (3.07x at C=1, 1.18-1.33x at C>=128), EAGLE e3 k=4. |
| DSpark scheduler on GH200 | Ported to vLLM twice. The first port (per-request lengths on the CPU, online cost fit, raw head) does not beat a fixed k=3 on summarization/chat: variable lengths broke vLLM's CUDA-graph shapes, confidences were one step stale, DSpark drafts all 7 slots. The second follows the paper (STS-calibrated head, Algorithm 1 with the section 5.2 lagged budget and top-K on the GPU, cost table profiled at engine start, full graphs for uneven decode steps). On the held-out prompts it tracks the better of k=7 and k=5 at every load (3.38x at C=1, 1.17x at C=256) but does not beat the best fixed length at any load; the remaining cost is drafting all 7 slots under load. |

**Recommendation for serving DSpark e10:** fixed k=3 for general traffic
with much out-of-distribution content (summarization, chat); k=4-5 for
traffic like the training data (held-out prompts: k=5 keeps 85% of k=7's
C=1 gain and is within 4% of the best under load). k=7 only for
single-stream or math/code-heavy deployments (probe 6.7 accepted tokens,
4.2x). With EAGLE e3, k=4 on in-distribution traffic; out of
distribution (epoch 1) k=3 at low load, k=1 under load.

## Setup

Unless a section says otherwise: one GH200 (95 GiB) per deployment, pinned
image `vllm_apertus_1.5_release-arm64` (vLLM 0.23.1rc1, commit a601a9d9) with
the overlays in `serving/patches/`, TP=1, bf16, `gpu_memory_utilization` 0.8,
`max_model_len` 32768, prefix caching off, `max_num_batched_tokens` 16384,
`max_num_seqs` 256, greedy, streaming, closed loop at fixed concurrency C,
whole passes over the 128 prompts of each workload. Workloads:
summarization (~3.8k prompt tokens, ~200-token answers, the prefill-heavy
worst case for speculation) and chat (the 128 chat test prompts). Probe: 64
math/HumanEval prompts at C=8, 384 tokens.

| drafter | checkpoint | trained on |
| --- | --- | --- |
| DSpark e10 | `dspark_apertus15-8b_open-perfectblend_thinking-off_2026-10-04_epoch10_final` (Yu, final) | Open-PerfectBlend regenerated, thinking off |
| EAGLE e1 | `e31-opb-thinkoff-stage1-ep1-se` (ours, epoch 1 of 10) | same corpus and split |

The canvas notes that the epoch count does not change the system-level
questions; DSpark is at its final checkpoint, EAGLE at epoch 1 until the
10-epoch run finishes.

### Harness: node sweeps

`serving/node-sweep.sh SPEC.yaml` submits one job that serves up to four
arms side by side on the four GPUs of a node (one `vllm serve` per GPU) and
load-tests them at the same time with the same client as
`serving/loadtest.sh` (`src/apertus_bench/node_sweep.py`; specs in
`experiments/system-8b/`, results in `results/8b/sweeps/<spec>-<stamp>/`,
tables with `python3 -m apertus_bench.sweep_report`). Arms can also span
several GPUs (TP/DP) or run a one-off program (`command`).

Calibration (`calibrate`, job 3619818): two identical plain arms on two GPUs
of one node differ by at most 1.5% in output tokens/s at any C, and plain
reproduces the 3 Oct single-deployment run (job 3571677) within 1%. Arms do
not disturb each other.

**Run-to-run spread.** Between jobs (different nodes) the same arm varies by
up to ~6% at high load (fixed k=3, chat C=256: 0.97x in `sched-smoke`, 0.93x
in `verify-uniform`, both against the calibration plain). Ratios between
arms of different sweeps are good to about ±5%; decisions between close
alternatives come from single jobs (`verify-uniform`, `verify-final`).

## Draft length

Output tokens/s relative to plain (calibration), greedy (sweeps
`calibrate`, `sched-smoke`, `k-grid-a`, `k-grid-b`):

Summarization:

| C | DSpark k=1 | 2 | 3 | 4 | 5 | 7 | EAGLE k=1 | 2 | 3 | 4 | 5 | 7 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 1.20x | 1.39x | 1.47x | 1.49x | 1.51x | 1.52x | 1.31x | 1.40x | 1.40x | 1.36x | 1.31x | 1.19x |
| 8 | 1.15x | 1.26x | 1.34x | 1.28x | 1.31x | 1.28x | 1.21x | 1.24x | 1.26x | 1.22x | 1.18x | 1.13x |
| 32 | 1.08x | 1.15x | 1.21x | 1.14x | 1.13x | 1.06x | 1.10x | 1.14x | 1.12x | 1.08x | 1.07x | 1.04x |
| 64 | 1.06x | 1.09x | 1.13x | 1.06x | 1.04x | 0.96x | 1.07x | 1.07x | 1.09x | 1.03x | 0.99x | 0.98x |
| 128 | 1.01x | 1.04x | 1.05x | 0.98x | 0.96x | 0.89x | 1.03x | 1.03x | 1.01x | 0.98x | 0.94x | 0.92x |
| 256 | 1.01x | 1.05x | 1.06x | 0.99x | 0.96x | 0.89x | 1.02x | 1.03x | 1.00x | 0.97x | 0.94x | 0.93x |

Chat:

| C | DSpark k=1 | 2 | 3 | 4 | 5 | 7 | EAGLE k=1 | 2 | 3 | 4 | 5 | 7 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 1.26x | 1.54x | 1.70x | 1.79x | 1.83x | 1.87x | 1.39x | 1.54x | 1.56x | 1.53x | 1.48x | 1.36x |
| 8 | 1.24x | 1.43x | 1.60x | 1.57x | 1.62x | 1.61x | 1.35x | 1.45x | 1.49x | 1.41x | 1.38x | 1.29x |
| 32 | 1.12x | 1.18x | 1.31x | 1.19x | 1.17x | 1.12x | 1.19x | 1.19x | 1.23x | 1.10x | 1.05x | 1.01x |
| 64 | 1.08x | 1.09x | 1.12x | 1.02x | 1.00x | 0.88x | 1.12x | 1.07x | 1.05x | 0.96x | 0.91x | 0.84x |
| 128 | 1.01x | 1.01x | 1.00x | 0.91x | 0.88x | 0.77x | 1.02x | 0.99x | 0.95x | 0.89x | 0.82x | 0.76x |
| 256 | 0.99x | 0.99x | 0.97x | 0.89x | 0.84x | 0.74x | 1.02x | 0.96x | 0.92x | 0.86x | 0.80x | 0.74x |

Accepted tokens per round (flat across load), DSpark k=1/2/3/5/7:
summarization 1.57 / 1.86 / 2.00 / 2.10 / 2.12, chat 1.65 / 2.05 / 2.29 /
2.51 / 2.59; probe 1.96 / 2.89 / 3.75 / 5.31 / 6.69.

- At k=7 only 16% (summarization) and 23% (chat) of the verified slots are
  accepted. Once the GPU is compute-bound every rejected slot is lost
  throughput, so the best length falls with load: DSpark 5-7 at C=1, 3 from
  C=8 to 64, 1-3 at C>=128; EAGLE 2-3 at low load, 1 under load.
- One fixed setting: DSpark k=3 keeps 90-97% of k=7's C=1 gain and never
  costs more than ~7% at high load (chat); EAGLE k=1-2 is never below plain.
- On chat at C>=128 no setting is clearly faster than plain (best within
  ±2%); speculation pays below ~C=64 on this GPU.

### Draft length on the held-out prompts

The grid above uses summarization and chat, both out of distribution for
the drafters (2.1-2.6 accepted tokens at k=7). `k-grid-opb-a..d` (jobs
3632467, 3632468, 3632470, 3632471; four nodes, each with its own plain arm,
plain within 3% across them) repeat it on Yu's held-out prompts (next
section) with DSpark e10 and EAGLE e3. Output tok/s vs plain in the same
job:

| C | DSpark k=1 | 2 | 3 | 4 | 5 | 7 | EAGLE e3 k=1 | 2 | 3 | 4 | 5 | 7 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 1.47x | 2.01x | 2.46x | 2.80x | 3.07x | 3.42x | 1.66x | 2.08x | 2.37x | 2.50x | 2.59x | 2.62x |
| 8 | 1.46x | 1.92x | 2.33x | 2.55x | 2.72x | 3.00x | 1.63x | 1.96x | 2.22x | 2.34x | 2.40x | 2.43x |
| 32 | 1.32x | 1.55x | 1.90x | 1.95x | 2.12x | 2.14x | 1.48x | 1.61x | 1.89x | 1.76x | 1.84x | 1.83x |
| 64 | 1.28x | 1.42x | 1.63x | 1.66x | 1.70x | 1.58x | 1.40x | 1.52x | 1.63x | 1.64x | 1.57x | 1.49x |
| 128 | 1.19x | 1.29x | 1.36x | 1.34x | 1.33x | 1.24x | 1.28x | 1.37x | 1.38x | 1.38x | 1.33x | 1.23x |
| 256 | 1.13x | 1.18x | 1.22x | 1.20x | 1.18x | 1.09x | 1.20x | 1.27x | 1.25x | 1.24x | 1.19x | 1.11x |

Accepted tokens per step (flat across load), k=1/2/3/4/5/7: DSpark 1.89 /
2.64 / 3.27 / 3.78 / 4.18 / 4.78, EAGLE e3 1.85 / 2.55 / 3.13 / 3.59 / 3.97
/ 4.52.

- In distribution every setting is faster than plain at every load (C=256
  still 1.09-1.27x); out of distribution chat and summarization fall to or
  below plain above C~64.
- The best length still falls with load but sits higher, because more
  drafts are accepted: DSpark 7 at C<=8, 5-7 at C=32, 4-5 at C=64, 3-5 at
  C>=128.
- k=3 is no longer the best single setting here: it keeps only 60% of
  k=7's C=1 gain (2.46x vs 3.42x). DSpark k=5 keeps 85% of it (3.07x), is
  best at C=64 and within 4% of the best at C>=128; k=4 is similar.
- EAGLE e3 gains little beyond k=5 even at C=1 (2.59x vs 2.62x); k=4 is
  within 7% of the best at every load. Under load EAGLE e3 matches or
  beats DSpark (C=256: 1.27x vs 1.22x, separate jobs), at C<=8 DSpark
  leads by 23-31%.
- So the choice depends on the traffic: k=3 when much of it is out of
  distribution (summarization, chat), k=4-5 when it resembles the training
  data; a load-dependent length (7 at low load, 4-5 above C~32) would get
  the best of both on in-distribution traffic.

## In-distribution check (Yu's held-out prompts)

Yu expected larger single-request speedups than the summarization/chat
sweeps show and pointed out that Open-PerfectBlend has no long-document
summarization. `opb-check` (job 3622798, one node) serves 128 prompts from
the DSpark Open-PerfectBlend validation split (`tools/make-opb-workload.py`;
mean 563 / median 158 prompt tokens, ~480-token answers, thinking off),
otherwise the setup above:

| C | DSpark e10 k=7 | DSpark e10 k=3 | EAGLE e1 k=7 | plain tok/s |
| --- | --- | --- | --- | --- |
| 1 | 3.44x | 2.43x | 2.43x | 172 |
| 8 | 2.97x | 2.31x | 2.26x | 1,177 |
| 32 | 2.05x | 1.89x | 1.67x | 3,159 |
| 128 | 1.23x | 1.36x | 1.12x | 6,604 |

Accepted tokens per step (including the target's own token): 4.8 / 3.3 /
4.2, flat across load; TPOT at C=1 1.4 / 2.2 / 2.0 ms vs 5.7 plain. The
serving path reproduces Yu's numbers on his distribution; the 1.5-1.9x at
C=1 on summarization/chat is their low acceptance (2.1-2.6). Under load the
ordering is the same as on the other workloads: k=3 overtakes k=7 by C=128.


## Confidence-scheduled verification (the DSpark scheduler)

**Paper** ([arXiv:2607.05147](https://arxiv.org/abs/2607.05147)): a
confidence head predicts, for each draft slot, the probability that it is
accepted given its prefix was; the running product `a[r, j]` is the prefix
survival. Each step the scheduler chooses a verification length `l_r` per
request that maximises expected tokens per step times steps per second,
`tau * SPS(B)` with `tau = sum_r (1 + sum_{j<l_r} a[r, j])` and
`B = sum_r (1 + l_r)`, greedily by descending survival. Deployed in
DeepSeek's own engine (not vLLM), with a decode-only batch.

**State in vLLM.** The DSpark checkpoints carry the trained head
(`confidence_head.proj`, input `[hidden; markov_w1[previous token]]`, 4,352 -> 1;
trained against the analytic acceptance rate 1 - TV; Yu's epoch-10 run reports
a cumulative-product bias of +0.011), but the pinned vLLM skips its weights
("not wired into inference yet"): every request verifies all k slots.

**Port** (`serving/patches/vllm-apertus-dspark-confidence-verify.patch`,
opt-in with `VLLM_DSPARK_VERIFY_SCHED=1`):

- the drafter evaluates the head inside its captured draft step, next to the
  Markov sampling it already does, and writes the prefix survival to a
  persistent buffer;
- the model runner copies it to the host with the step's output
  (`ModelRunnerOutput.spec_survival`);
- the scheduler trims each running request's draft placeholders to its
  length before scheduling, so the token budget, KV allocation and the
  runner (which already handles per-request draft counts) follow.

Modes: `threshold` (keep slots with survival >= t), `greedy` (the paper's
search, per request), and `greedy` with `VLLM_DSPARK_VERIFY_UNIFORM=1` (one
length per step for all decode requests, chosen from
`VLLM_DSPARK_VERIFY_LENGTHS`, with a captured uniform-decode CUDA graph per
length). Departures from the paper:

1. **Prefill counts.** Our engine mixes prefill chunks into decode steps.
   The paper's objective treats them as free, so with 4,000 prefill tokens
   in the step it verifies all 7 slots, although longer steps delay that
   prefill too. The objective here is all tokens processed per second,
   `(P + tau) / t(B)`, with `P` the prefill tokens expected in the step; it
   equals the paper's when `P = 0`.
2. **Step time.** `t(B) = f(B) + o0 + o1 * B`: `f` is the forward time
   against batch tokens profiled at short contexts
   (`methods/dspark/scheduler/profile_step_time.py`, measured in
   `methods/dspark/scheduler/step-time-8b-gh200.json`: ~5.4 ms up to 32
   tokens, knee at 128-192, then ~26 us per token), `o0 + o1 * B` an online
   least-squares fit of the observed step interval minus `f` (attention over
   long contexts, drafting, scheduling).
3. **Staleness.** With async scheduling (the default) step t+1 is scheduled
   before step t's drafts exist, so lengths come from each request's
   previous drafts; the paper uses a two-step lag for the same reason.

### Results

**The head is informative** (`sched-smoke`). Threshold 0.5 on prefix
survival verified 1.8 (summarization) / 2.3 (chat) slots per step and 0.92 /
1.34 of them were accepted; fixed lengths interpolated to the same budget
accept ~0.74 / ~1.06: ~25% more accepted tokens per verified slot.

**First bottleneck: CUDA graphs.** The threshold arm ran at 0.59x plain at
C=1 (fixed k=1: 1.20x). With varying lengths, decode batches are no longer
the one uniform shape (k+1 tokens per request) vLLM captures full CUDA
graphs for, and fall back to piecewise graphs with eager attention, ~7 ms
more per step at small batch. The paper reports the same conflict between
dynamic lengths and graph replay. Two fixes measured: full graphs for every
batch shape (`cudagraph_mode FULL`; removes the overhead, the greedy mode
then matches k=7 on the probe, 5,188 vs 5,184 tok/s, but the mode alone
costs ~5% at high load), and one length per step with a uniform-decode graph
captured per allowed length (what vLLM does for its own batch-size
schedule).

**Final comparison** (`verify-final`, job 3620667, all arms on one node;
output tok/s vs plain in the same job):

| C | summ. k=3 | vLLM batch-size schedule 7/3/2 | confidence scheduler (uniform) | chat k=3 | schedule 7/3/2 | confidence scheduler |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 1.49x | 1.53x | 1.47x | 1.70x | 1.87x | 1.82x |
| 8 | 1.34x | 1.31x | 1.21x | 1.65x | 1.60x | 1.54x |
| 32 | 1.16x | 1.14x | 1.05x | 1.29x | 1.23x | 1.13x |
| 64 | 1.10x | 1.04x | 0.99x | 1.09x | 1.00x | 0.93x |
| 128 | 1.01x | 0.97x | 0.92x | 0.97x | 0.89x | 0.86x |
| 256 | 1.01x | 0.96x | 0.92x | 0.93x | 0.87x | 0.83x |

(vLLM's schedule: k=7 up to 4 running requests, 3 up to 64, 2 above, from
the grid above. Earlier variants: per-request greedy with full graphs
reaches 0.94-0.97x on summarization and 0.88-0.90x on chat at C=256, i.e.
also below k=3; without async scheduling acceptance is higher but throughput
8-11% lower.)

Both adaptive variants keep k=7's single-stream speed but trail fixed k=3
from C=8 on. Why, and what would fix it:

- **Drafting is paid for all 7 slots.** Both adaptive arms run DSpark with a
  7-slot block and verify fewer; at high batch the drafter's 7-slot LM head
  and 7-step Markov loop are a real part of the step (k=3 drafts only 3).
  Fix: let the length decision cut the draft too (stop the sequential
  Markov loop and the per-slot LM head at the step's maximum length).
- **Stale confidences.** Async scheduling decides lengths before the drafts
  exist; turning it off costs more than the fresher signal gains. Fix: the
  paper's split, a lagged batch budget plus a top-K selection on the GPU
  over current scores, which needs the varlen batch to be built on the GPU.
- **Cost model.** The per-token cost of verification at high load (~47 us
  measured from step times at chat C=256 vs ~26 us in the short-context
  profile) is not recovered by the online fit, whose slope stays ~0 at a
  fixed load. Fix: profile `f` with long contexts and the drafter in the
  loop instead of fitting online.

### The scheduler as the paper deploys it

The port above departs from the paper in its cost model, its use of stale
per-request lengths and the uncalibrated head. A second patch
(`serving/patches/vllm-apertus-dspark-paper-scheduler.patch`, applied after
the first; `VLLM_DSPARK_VERIFY_MODE=paper`) follows the paper instead:

- **Calibrated head (STS, section 3.2.1).** One temperature per slot,
  `c_j = sigmoid(z_j / T_j)`, fitted left to right by grid search to minimise
  the ECE (20 bins, as in DeepSpec's evaluation) of the running product
  against the observed prefix survival (`methods/dspark/scheduler/fit_sts.py`).
  Data: DSpark e10 at k=7 verifying everything with the head loaded, on 512
  held-out prompts disjoint from the 128 test prompts
  (`workloads/8b/opb-calib.jsonl`, `tools/make-opb-workload.py --offset 48
  --exclude opb-heldout.jsonl`), 298k verifications (`paper-calib`, job
  3669651). The trained head is already close: ECE 1.0-2.4% per slot, later
  slots underconfident; STS (T = 1.04, 0.97, 1.00, 0.89, 0.85, 0.87, 0.83;
  `methods/dspark/scheduler/sts-dspark-e10-8b.json`) brings it to 0.7-2.3%,
  ~1% from slot 4 on, as the paper reports.
- **Algorithm 1 with the section 5.2 adaptations.** The scheduler sets the
  step's batch capacity K from the newest survival it has (two steps old
  under async scheduling) by a global search over the greedy admission path,
  `Theta = tau * SPS(B)`, `B = R + K`, decode tokens only, no early stop.
  Every decode request is scheduled with all its draft placeholders, and the
  worker verifies the K drafts with the highest *current* survival: a top-K
  on the GPU, with per-request query lengths, `query_start_loc` and the
  rejection sampler's offsets built there (no host sync).
- **Cost table profiled once at engine start (section 3.2.2).** Target
  forward with logits over B tokens (the graph a decode step replays) plus
  the draft step for the step's R requests, which drafts all 7 slots whatever
  is verified. On GH200: verify 5.7 ms up to 32 tokens, 7.8 ms at 128, 19.6
  ms at 512, then a cliff to 27.3 ms at 576 where full CUDA graphs end (the
  "jagged" capacity curve of section 5.2); draft step 1.9 ms for one request,
  9.2 ms at 128, 17.1 ms at 256.
- **CUDA graphs.** Decode-only steps of uneven lengths replay FULL graphs
  captured per token count (FlashAttention 3 replays them for any split);
  steps with prefill stay piecewise. The extra graphs take 4.8 instead of
  1.1 GiB.

Drafts verified per request (scheduler log, both jobs): 7.0 at C=1, 5.9 at
C=8, 4.5 at C=32, 3.6 at C=64, 3.0-3.3 at C>=128.

**Result** (`paper-eval-a`/`-b`, jobs 3669829 / 3669831, the 128 held-out
test prompts, arms on swapped GPUs between the two jobs, means of both;
output tok/s vs plain in the same job):

| C | DSpark k=7 | DSpark k=5 | paper scheduler | vs k=7 | vs k=5 | k=7, job b vs a |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 3.43x | 3.06x | 3.38x | -1.5% | +10.3% | +0.0% |
| 8 | 2.98x | 2.72x | 2.93x | -1.5% | +7.6% | +1.4% |
| 32 | 2.15x | 2.05x | 2.14x | -0.6% | +4.2% | +4.6% |
| 64 | 1.59x | 1.63x | 1.60x | +1.0% | -1.8% | +3.8% |
| 128 | 1.28x | 1.32x | 1.30x | +1.7% | -2.0% | +5.6% |
| 256 | 1.13x | 1.17x | 1.17x | +3.5% | -0.6% | +6.5% |

Accepted tokens per step: 4.8 (k=7) and 4.2 (k=5) at every load; the
scheduler 4.8 / 4.7 / 4.4 / 4.3 / 3.6 / 3.8 from C=1 to 256.

- The paper's scheduler follows the better of the two fixed lengths at every
  load: k=7's single-stream speed (within 1.5%) and k=5's throughput under
  load (within 2%). As one setting for all loads it is the best tested here:
  never more than 2% below either, 10% above k=5 at C=1 and 3.5% above k=7
  at C=256.
- It does not beat the best fixed length at any one load. The differences to
  the better arm (-1.5% to +1.0%) are below the spread between identical arms
  on different GPUs (up to 6.5%, last column).
- At C>=128 the grid's k=3 and k=4 reached 1.34-1.36x and 1.20-1.22x
  (separate jobs), ~4% above the scheduler. The scheduler verifies ~3 drafts
  per request there but, as in the paper, drafts all 7: at 256 requests the
  draft step alone is 17 ms against 64 ms for verifying 2,048 tokens, a cost
  k=3 does not pay.

With the first port and this one the conclusion holds: on vLLM, at 8B, the
scheduler buys load-independence (one setting matching the best fixed length
within noise across C=1..256), not throughput beyond the best fixed length.
The remaining gap is the 7-slot drafting under load; cutting the draft at
the step's longest verified length is the next change (not in the paper).

## Runs

| sweep | job | what |
| --- | --- | --- |
| `calibrate-20261008T224745Z` | 3619818 | two plain arms, EAGLE e1 k=7, DSpark e10 k=7 |
| `sched-smoke-20261008T225336Z` | 3619828 | DSpark k=3, k=1, threshold scheduler (profile arm failed) |
| `profile-20261008T231013Z` | 3619913 | step-time profile |
| `verify-greedy-20261008T234317Z` | 3620217 | greedy scheduler, per request / uniform, full graphs, sync |
| `verify-fit-20261009T000309Z` | 3620259 | fitted overhead, k=3 with full graphs |
| `k-grid-a-20261009T000407Z` | 3620263 | DSpark k=2/5, EAGLE k=1/3 |
| `verify-uniform-20261009T002023Z` | 3620271 | uniform scheduler {1,3,7}, k=3, vLLM schedule 7/3 |
| `verify-final-20261009T013655Z` | 3620667 | plain, k=3, vLLM schedule 7/3/2, uniform scheduler |
| `k-grid-b-20261008T225658Z` | 3619837 | EAGLE k=2/4/5, DSpark k=4 |
| `opb-check-20261009T113600Z` | 3622798 | Open-PerfectBlend held-out prompts: plain, DSpark k=7/3, EAGLE k=7 |
| `k-grid-opb-{a,b,c,d}-20261010T1454*Z` | 3632467, 3632468, 3632470, 3632471 | draft-length grid on the held-out prompts: DSpark e10 k=1/2/3, 4/5/7, EAGLE e3 k=1/2/3, 4/5/7, each with plain |
| `paper-calib-20261015T145635Z` | 3669651 | paper scheduler: STS calibration data (k=7, head loaded), raw-head scheduler, k=7, plain on 512 calibration prompts |
| `paper-eval-a-20261015T153126Z`, `paper-eval-b-20261015T153130Z` | 3669829, 3669831 | paper scheduler with STS vs DSpark k=7, k=5, plain on the held-out prompts, arms on swapped GPUs |
| `draft-cut-a-20261015T165456Z`, `draft-cut-b-20261015T165501Z` | 3670246, 3670247 | paper scheduler cutting the draft per step vs the uncut paper scheduler, DSpark k=3, plain on the held-out prompts, arms on swapped GPUs |
| `upstream-av-a-20261015T231232Z`, `upstream-av-b-20261015T231233Z` | 3672315, 3672316 | upstream vLLM adaptive verification (#47808) on the v0.31.0 image vs DSpark k=3, k=7, plain on the same image, held-out prompts, arms on swapped GPUs |
