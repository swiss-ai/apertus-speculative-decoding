# System-level optimizations for speculative decoding, Apertus 1.5 8B

Work item from the 2 Oct sync (canvas "Spec Decoding Sync 2 Oct: Action
Items by Owner", Faruk's part): make speculative decoding pay off under load.
The load tests of 3 Oct ([loadtest-8b.md](loadtest-8b.md)) showed both
drafters faster than the plain target up to C=8, even at C=32 and 8-15%
slower from C=64 on. This document records what was tried against that,
with which runs, and what came out.

Status: in progress (2026-10-09). Sections without numbers are pending runs.

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

Drafters:

| name | checkpoint | trained on |
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
prefill/decode pair behind a proxy (`pd`).

Calibration (`calibrate`, job 3619818): two identical plain arms on two GPUs
of one node differ by at most 1.5% in output tokens/s at any C, and plain
reproduces the 3 Oct single-deployment run (job 3571677) within 1%. Arms do
not disturb each other. The EAGLE e1 k=7 arm is 2-5% faster at C>=32 than
its 3 Oct run (3571815) at the same acceptance; comparisons below are always
within one sweep or between sweeps of this harness.

## Draft length

Output tokens/s relative to plain, DSpark e10, greedy (sweeps `calibrate`,
`sched-smoke`; full grid for both drafters pending in `k-grid-a/b`):

| C | summarization k=7 | k=3 | k=1 | chat k=7 | k=3 | k=1 |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 1.52x | 1.47x | 1.20x | 1.87x | 1.70x | 1.26x |
| 8 | 1.28x | 1.34x | 1.15x | 1.61x | 1.60x | 1.24x |
| 32 | 1.06x | 1.21x | 1.08x | 1.12x | 1.31x | 1.12x |
| 64 | 0.96x | 1.13x | 1.06x | 0.88x | 1.12x | 1.08x |
| 128 | 0.89x | 1.05x | 1.01x | 0.77x | 1.00x | 1.01x |
| 256 | 0.89x | 1.06x | 1.01x | 0.74x | 0.97x | 0.99x |

Accepted tokens per round, flat across load: summarization 2.12 / 2.00 / 1.57,
chat 2.59 / 2.29 / 1.65, probe (math/code) 6.69 / 3.75 / 1.96.

- k=7 wastes most of its verification on these workloads: 16% (summarization)
  and 23% (chat) of the verified slots are accepted. Once the GPU is
  compute-bound, every rejected slot is lost throughput.
- k=3 keeps nearly all of the low-load gain and turns the high-load loss into
  break-even or better: summarization never falls below plain (1.05-1.06x
  at C>=128), chat loses at most 3%. It is the better single setting for
  serving these workloads; k=7 wins only at C=1 and on high-acceptance
  (math/code) traffic.

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

**Implementation** (`serving/patches/vllm-apertus-dspark-confidence-verify.patch`,
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
search), and `greedy` with `VLLM_DSPARK_VERIFY_UNIFORM=1` (one length per
step for all requests). Two changes against the paper:

1. **Prefill counts.** Our engine mixes prefill chunks into decode steps.
   The paper's objective treats them as free, so with 4,000 prefill tokens
   in the step it verifies all 7 slots; but longer steps delay that
   prefill too. The objective here is all tokens processed per second,
   `(P + tau) / t(B)`, with `P` the prefill tokens expected in the step; it
   equals the paper's when `P = 0`.
2. **Step time.** `t(B) = f(B) + o`: `f` is the forward time against batch
   tokens profiled at short contexts
   (`methods/dspark/scheduler/profile_step_time.py`), `o` an online mean of
   the observed step interval minus `f`. At high concurrency with ~4k-token
   contexts most of a decode step is reading the KV cache, which barely
   depends on the number of draft slots; `o` captures that, and with it the
   scheduler keeps long drafts when verification is nearly free.

With async scheduling (the default) the scheduler sets step t+1's lengths
before step t's drafts exist, so it uses each request's previous drafts
(one step stale); the paper uses a two-step lag for the same reason.

## Drafter KV cache

The canvas asks whether the drafter needs its full KV cache. The pool
measured on 3 Oct: 469,840 tokens plain, 342,693 with DSpark (-27%), 433,888
with EAGLE (-8%). Where DSpark's 27% goes (Apertus 8B: 32 full-attention
layers; DSpark: 5 layers, all sliding-window attention with window 2048):

- **Grouping padding, ~9%.** vLLM's hybrid KV manager needs equally sized
  layer groups; with 32 full and 5 sliding-window layers it picks groups of
  5 and pads the target's 32 layers to 35 slots
  (`_get_kv_cache_groups_uniform_page_size`).
- **The drafter's window, ~11% at these lengths.** Its 5 layers keep at most
  2,048 tokens each per request (vLLM frees blocks outside the window); with
  ~2.8k-token requests that is 5 x 2,048 against 32 x 2,800.
- **Drafter weights, ~6%** (3.3 GiB taken from the 0.8 x 95 GiB budget).

The pool size vLLM logs (10.46 x 32k) is a conservative bound that treats
the window group as full length; the requests that actually fit (129 at
C>=256 vs 171 plain) match the padded-plus-window accounting.

So the drafter's KV is already sparse in time (a window); the levers are a
smaller window at inference and removing the padding. EAGLE's single
full-attention layer costs 1/33 of the pool; there is little to gain there.
Sweeps: `drafter-window` (window 1024/512/256/128 on the epoch-10 weights,
config copies with the weights symlinked) and `drafter-kv-groups` (group size
1 and 2 via `serving/patches/vllm-apertus-kv-group-size.patch`).

### Results so far

All DSpark e10, drafting 7 slots, output tokens/s relative to plain
(sweeps `sched-smoke`, `verify-greedy`, `verify-fit`).

**The head is informative.** Threshold 0.5 on prefix survival verified 1.8
(summarization) / 2.3 (chat) slots per step and accepted 0.92 / 1.34 of
them; fixed lengths interpolated to the same budget accept ~0.74 / ~1.06,
so the head buys ~25% more accepted tokens per verified slot.

**The engine ate it.** The threshold arm ran at 0.59x plain at C=1 (fixed
k=1: 1.20x). With per-request or per-step varying lengths, decode batches are
no longer the one uniform shape (k+1 tokens per request) that vLLM captures
full CUDA graphs for, and fall back to piecewise graphs with eager
attention, ~7 ms more per step at small batch. The paper reports the same
conflict between dynamic lengths and graph replay and solves it inside
DeepSeek's engine.

**With full graphs** (`cudagraph_mode FULL`, FlashAttention 3 handles varlen
batches in graphs) the overhead disappears: on the probe the greedy mode
matches fixed k=7 (5,188 vs 5,184 tok/s) and the uniform variant is 4%
faster (5,402). Under load:

| C | summ. k=7 | k=3 | k=3 full graphs | greedy uniform, full graphs | chat k=7 | k=3 | k=3 full graphs | greedy uniform, full graphs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 1.52x | 1.47x | 1.47x | 1.57x | 1.87x | 1.70x | 1.70x | 1.84x |
| 8 | 1.28x | 1.34x | 1.31x | 1.24x | 1.61x | 1.60x | 1.59x | 1.58x |
| 32 | 1.06x | 1.21x | 1.15x | 1.12x | 1.12x | 1.31x | 1.27x | 1.19x |
| 64 | 0.96x | 1.13x | 1.08x | 1.04x | 0.88x | 1.12x | 1.07x | 1.00x |
| 128 | 0.89x | 1.05x | 0.99x | 0.97x | 0.77x | 1.00x | 0.95x | 0.91x |
| 256 | 0.89x | 1.06x | 0.99x | 0.97x | 0.74x | 0.97x | 0.92x | 0.89x |

- The scheduler keeps k=7's low-load speed and recovers about half of k=7's
  high-load loss, without any tuning, but stays 2-3% (summarization) and
  ~3% (chat) behind fixed k=3 with the same graphs, and full graphs for
  every batch shape cost fixed k=3 ~5% at high load by themselves.
- Per-request lengths (the paper's form) do no better than one length per
  step here, and lose more to graph shapes; without async scheduling (exact
  instead of one-step-stale survival) acceptance is higher but throughput
  8-11% lower.
- Fitting the overhead as `o0 + o1 * B` did not change the picture: at
  fixed load the batch size varies too little for the slope (fitted ~0).

Next (`verify-uniform`): the default graph mode plus a captured
uniform-decode graph for each length the scheduler may pick, and vLLM's
built-in batch-size schedule (k=7 up to 8 running, k=3 above) as the
untuned-but-simple alternative.
