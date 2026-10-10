# System-level optimizations for speculative decoding, Apertus 1.5 8B

Work item from the 2 Oct sync (canvas "Spec Decoding Sync 2 Oct: Action
Items by Owner", Faruk's part): make speculative decoding pay off under load.
The load tests of 3 Oct ([loadtest-8b.md](loadtest-8b.md)) showed both
drafters faster than the plain target up to C=8, even at C=32 and 8-15%
slower from C=64 on. This document records what was tried against that,
with which runs, and what came out.

Status 2026-10-09: all canvas items measured once on the current
checkpoints; rerun the draft-length grid when EAGLE reaches its final epoch.

## Summary

| canvas item | result |
| --- | --- |
| system-level tricks on the current checkpoints | **Draft length is the big lever.** DSpark e10 at k=3 instead of 7: summarization never below plain (1.01-1.06x at C>=128 instead of 0.89x), chat at 0.93-1.00x instead of 0.74x, at the cost of ~10% at C=1. EAGLE e1 is best at k=3 at low load and k=1 under load (1.02x at C=256). |
| DSpark scheduler on GH200 | Ported to vLLM (the checkpoints carry the trained confidence head; vLLM ignored it). The head is informative (~25% more accepted tokens per verified slot than fixed lengths), but on vLLM the scheduler **does not beat a fixed k=3**: variable lengths break vLLM's CUDA-graph shapes (fixed by capturing graphs per length), async scheduling makes the confidences one step stale, and DSpark drafts all 7 slots whatever is verified. Bottlenecks and proposed fixes below. |
| one GPU / several GPUs / disaggregation | For throughput, four data-parallel engines per node (3.95x one GPU; speculation keeps its single-GPU pattern). For latency, TP4 + DSpark (C=4: 2.35x DP4 plain, 2.5 ms TPOT) but it saturates at 0.59x. Prefill/decode disaggregation on one node does not pay for an 8B target: 1P+1D is prefill-bound (0.54-0.74x of DP2), 3P+1D overflows the decode GPU's KV (0.11-0.16x of DP4 from C=256). |
| tune top-k | Sampling at Apertus' T 0.7 / top-p 0.8 costs ~2% acceptance against greedy; top-k 20 vs none makes no measurable difference next to top-p 0.8. The speedups follow the greedy ones. |
| sparsify the drafter KV cache | DSpark's drafter KV is already a 2048-token window. Shrinking the window loses more acceptance than the extra requests it fits. Removing vLLM's layer-group padding (patch) gives +8% KV capacity (139 vs 129 requests at C=256), throughput unchanged. |
| verify the papers on the cluster | DSpark scheduler: as above. MineDraft: needs a separate drafter GPU and targets vLLM 0.9; for an 8B target at most ~30% of a step is drafting and the extra GPU halves per-GPU throughput, so not pursued (analysis below). |

**Recommendation for serving DSpark e10 on general traffic:** fixed k=3.
k=7 only for single-stream or math/code-heavy deployments (probe 6.7
accepted tokens, 4.2x). With EAGLE, k=3 at low load, k=1 under load.

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
several GPUs (TP/DP), run a one-off program (`command`), or be a
prefill/decode pair behind a proxy (`pd`, `apertus_bench.pd_proxy`).

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

**EAGLE at epoch 3** (`eagle-e3`, job 3630557, one node; head exported from
step 76,344 like the epoch-1 head). Output tok/s vs plain in the same job,
7 draft tokens; accepted tokens per step in brackets:

| workload, C | DSpark e10 | EAGLE e1 | EAGLE e3 |
| --- | --- | --- | --- |
| held-out, 1 | 3.43x (4.78) | 2.46x (4.19) | 2.63x (4.52) |
| held-out, 32 | 2.06x | 1.72x | 1.82x |
| held-out, 128 | 1.23x | 1.13x | 1.20x |
| chat, 1 | 1.87x (2.60) | 1.37x (2.33) | 1.45x (2.49) |
| chat, 128 | 0.77x | 0.72x | 0.75x |
| summarization, 1 | 1.53x (2.14) | 1.20x (2.05) | 1.28x (2.20) |
| summarization, 128 | 0.89x | 0.87x | 0.91x |

Two more epochs raise EAGLE's acceptance by 4-8% (probe 6.13 -> 6.30) and its
speedup accordingly; it still trails DSpark e10 except on summarization,
where the two are level (2.20 vs 2.14 accepted). The load-dependence is
unchanged: at k=7 both lose to plain under load on chat and summarization.

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

## Sampling and top-k

`sampling` (job 3619860), chat and summarization at T 0.7 / top-p 0.8 with
top-k 20 (Apertus' recommended setting) and without a top-k cut:

| | chat, greedy | T0.7 p0.8 k20 | T0.7 p0.8 no top-k | summ., greedy | T0.7 p0.8 k20 |
| --- | --- | --- | --- | --- | --- |
| DSpark k=7 accepted | 2.59 | 2.55 | 2.54 | 2.12 | 2.08 |
| DSpark k=3 accepted | 2.29 | 2.26 | 2.26 | 2.00 | 1.98 |
| EAGLE k=7 accepted | 2.32 | 2.29 | 2.29 | 2.02 | 2.01 |
| DSpark k=3 vs plain, C=1 / 32 / 256 | 1.70 / 1.31 / 0.97x | 1.62 / 1.29 / 0.93x | 1.61 / 1.30 / 0.92x | 1.47 / 1.21 / 1.06x | 1.41 / 1.19 / 1.01x |

Top-k does not matter next to top-p 0.8 (identical acceptance, throughput
within 1%); sampling costs ~2% acceptance and a few percent of speedup.

## Drafter KV cache

The pool measured on 3 Oct: 469,840 tokens plain, 342,693 with DSpark
(-27%), 433,888 with EAGLE (-8%). Where DSpark's 27% goes (Apertus 8B: 32
full-attention layers; DSpark: 5 layers, all sliding-window attention with
window 2048):

- **Layer-group padding, ~9%.** vLLM's hybrid KV manager needs equally
  sized layer groups; with 32 full and 5 sliding-window layers it picks
  groups of 5 and pads the target's 32 layers to 35 slots
  (`_get_kv_cache_groups_uniform_page_size`).
- **The drafter's window, ~11% at these lengths.** Its 5 layers keep at most
  2,048 tokens each per request (vLLM frees blocks outside the window); with
  ~2.8k-token requests that is 5 x 2,048 against 32 x 2,800.
- **Drafter weights, ~6%** (3.3 GiB of the 0.8 x 95 GiB budget).

EAGLE's single full-attention layer costs 1/33 of the pool; nothing to gain
there.

**Smaller window** (`drafter-window`, job 3620352; config copies of the e10
checkpoint with the weights symlinked):

| window | accepted, summarization | accepted, chat | running at C=256 (summ.) | tok/s vs 2048, summ. C=1 / C=256 | chat C=1 / C=256 |
| --- | --- | --- | --- | --- | --- |
| 2048 (trained) | 2.12 | 2.59 | 129 | 1.00 / 1.00 | 1.00 / 1.00 |
| 1024 | 1.95 | 2.59 | 136 | 0.91 / 0.98 | 0.99 / 1.00 |
| 512 | 1.81 | 2.50 | 139 | 0.87 / 0.95 | 0.97 / 0.98 |
| 256 | 1.74 | 2.42 | 141 | 0.84 / 0.95 | 0.94 / 0.97 |
| 128 | 1.68 | 2.29 | 142 | 0.81 / 0.95 | 0.89 / 0.97 |

The drafter uses its long context: on ~3.8k-token prompts every halving of
the window costs acceptance, and the extra requests that fit do not pay for
it. Training with a shorter window would be the proper test.

**No padding** (`drafter-kv-groups`, job 3619869;
`serving/patches/vllm-apertus-kv-group-size.patch`, `VLLM_KV_CACHE_GROUP_SIZE`):
group size 1 lifts the pool to 370,477 tokens (+8%) and the requests that fit
at C=256 from 129 to 139 (151 with window 512), throughput within noise
(0.99-1.03x of the default at every C; group size 2: 360,723 tokens, one
padding layer). The gain is capacity (fewer queued requests, lower TTFT at
saturation), not tokens/s, since at that load the GPU is compute-bound.

## Several GPUs and disaggregation

**Data parallel x4** (`multigpu-dp4`, job 3619861, summarization, C per
node): plain 598 / 2,730 / 5,439 / 6,569 / 7,286 / 7,295 tok/s at C = 4 /
32 / 128 / 256 / 512 / 1024, i.e. 3.95x one GPU at saturation. DSpark e10
k=7: 1.55x / 1.34x / 1.10x / 0.98x / 0.90x / 0.89x of it, EAGLE e1 k=7:
1.20x / 1.17x / 1.01x / 0.92x / 0.88x / 0.89x: the single-GPU pattern at 4x
the concurrency. For an 8B model, a node is best used as four independent
engines; per-GPU conclusions above carry over.

**Tensor parallel** (`multigpu-tp4`, job 3620830; `multigpu-tp2dp2`,
3620826), output tok/s per node vs plain DP4, summarization:

| C (per node) | DSpark DP4 | plain TP4 | DSpark TP4 | plain TP2xDP2 | DSpark TP2xDP2 |
| --- | --- | --- | --- | --- | --- |
| 4 | 1.55x | 1.74x | 2.35x | 1.41x | 1.97x |
| 32 | 1.34x | 1.13x | 1.26x | 1.11x | 1.26x |
| 128 | 1.10x | 0.90x | 0.70x | 0.91x | 0.83x |
| 256 | 0.98x | 0.80x | 0.64x | 0.83x | 0.72x |
| 512 | 0.90x | 0.72x | 0.59x | 0.79x | 0.66x |
| 1024 | 0.89x | 0.72x | 0.59x | 0.79x | 0.67x |

TP4 is the latency layout: at C=4 a request decodes at 3.5 ms/token plain
and 2.5 ms with DSpark (6.0 / 3.7 ms on one GPU), and speculation adds 35% on
top of TP. From C=128 on it saturates below DP4 (all-reduce per layer, one
scheduler), and the drafter makes that worse. DP4 is the throughput layout;
DSpark on DP4 pays up to C~128 per node (~32 per GPU), as on one GPU.

**Prefill/decode disaggregation** (`pd-2gpu`, job 3620663): one prefill
and one decode GPU (NIXL 1.5.0 KV transfer, `apertus_bench.pd_proxy` in
front, the drafter on both roles so the KV layouts match) against two
data-parallel engines on the same two GPUs. Output tok/s vs plain DP2:

| C (per 2 GPUs) | summ. DSpark DP2 | 1P+1D plain | 1P+1D DSpark | chat DSpark DP2 | 1P+1D plain | 1P+1D DSpark |
| --- | --- | --- | --- | --- | --- | --- |
| 2 | 1.50x | 0.93x | 1.40x | 1.86x | 0.97x | 1.74x |
| 16 | 1.30x | 0.93x | 1.14x | 1.61x | 0.95x | 1.40x |
| 64 | 1.08x | 0.83x | 0.86x | 1.07x | 0.89x | 0.83x |
| 128 | 0.95x | 0.75x | 0.76x | 0.89x | 0.67x | 0.63x |
| 256 | 0.90x | 0.73x | 0.72x | 0.86x | 0.54x | 0.59x |
| 512 | 0.91x | 0.74x | 0.74x | 0.85x | 0.54x | 0.47x |

A 1:1 split is prefill-bound on these prompts: TTFT p50 grows to 15-37 s
while the decode GPU runs at ~16 ms TPOT (summarization C=512; 87 ms
colocated). The decode GPU is in the regime where speculation pays, but the
pipeline is limited by prefill, so the drafter adds nothing under load (and
the KV transfer adds ~80-100 ms TTFT at low load). On chat a single decode
instance is also capped at `max_num_seqs` 256.

Three prefill GPUs per decode GPU (`pd-3p1d`, job 3620834) fixes the
prefill side but breaks the decode side: on summarization the pipeline
collapses from C=256 on (814-1,200 tok/s per node, 0.11-0.16x of DP4, TTFT
up to 90 s, decode TPOT ~200 ms) because one decode GPU's KV pool holds only
~170 of these 4k-token requests and the transfers back up; up to C=32 it
runs at 0.77-0.91x of DP4 (DSpark: 0.89-1.31x). Chat is erratic between
neighbouring levels (0.64x to 1.07x for DSpark vs plain in the same split).
For an 8B model on one 4-GPU node, data parallelism beats both splits; a
useful disaggregated layout needs several decode GPUs' worth of KV per
prefill pool, i.e. more than one node, which is where it is usually
deployed. Kept as harness capability for the 70B/Apertus 2 work.

## MineDraft

[MineDraft](https://arxiv.org/abs/2603.18016) overlaps drafting of one batch
with verification of another by running the drafter on its own GPU (their
setups: target on TP4 plus one draft GPU; batches of 16-64), as a vLLM 0.9.2
plugin. Its gain is bounded by the drafting share of a step, and the paper
itself notes it vanishes when the target is compute-bound. For the 8B
target here: at C=1 a DSpark k=7 step takes ~8.7 ms against ~6.2 ms plain
(summarization), and at chat C=256 ~125 ms against ~41 ms plus ~48 ms for
the extra verified tokens; drafting and other per-step overhead are at most
~30% of a step, so hiding all of it gives at most ~1.4x per replica while
doubling the GPUs per replica. Per GPU that is a loss for an 8B model, and
the plugin does not run on the pinned vLLM (model runner v2). Not pursued;
it may be worth a look for the 70B target if a node-external draft GPU is
available.

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
| `drafter-window-20261009T005027Z` | 3620352 | DSpark window 1024..128 |
| `verify-final-20261009T013655Z` | 3620667 | plain, k=3, vLLM schedule 7/3/2, uniform scheduler |
| `sampling-20261008T230331Z` | 3619860 | T 0.7 / top-p 0.8 with and without top-k 20 |
| `multigpu-dp4-20261008T230335Z` | 3619861 | DP4 plain / DSpark / EAGLE |
| `k-grid-b-20261008T225658Z` | 3619837 | EAGLE k=2/4/5, DSpark k=4 |
| `drafter-kv-groups-20261008T230434Z` | 3619869 | KV group size 1 and 2, with windows 1024/512 |
| `pd-2gpu-20261009T013600Z` | 3620663 | 1 prefill + 1 decode GPU vs DP2, plain and DSpark |
| `multigpu-tp4-20261009T021403Z` | 3620830 | TP4 plain / DSpark |
| `multigpu-tp2dp2-20261009T021208Z` | 3620826 | TP2 x DP2 plain / DSpark |
| `pd-3p1d-20261009T021854Z` | 3620834 | 3 prefill + 1 decode GPU, plain / DSpark |
| `opb-check-20261009T113600Z` | 3622798 | Open-PerfectBlend held-out prompts: plain, DSpark k=7/3, EAGLE k=7 |
| `eagle-e3-20261010T080528Z` | 3630557 | EAGLE epoch 3 vs epoch 1, DSpark e10, plain on held-out, chat, summarization |
