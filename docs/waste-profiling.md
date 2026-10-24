# Wasted compute past saturation, Apertus 1.5 8B

Yu (Slack, 16 Oct): once speculative decoding is compute-bound the speedup
drops; how much compute is wasted in that region (e.g. k=7 at C=128 accepting
~3.5 tokens), what does a token cost at high load, and how do both relate to
the drafter's architecture? Rob suggested Nsight Systems for the measurements.

Status 2026-10-24: profiled sweeps of every fixed-k arm done (waste-prof-a/b),
analysis in `src/apertus_bench/waste_report.py`. The unprofiled summarization
costs of DSpark k=1/3 and the paper scheduler are queued (waste-sum-a/b).

## Summary

| question | answer (Apertus 1.5 8B, one GH200, opb held-out prompts unless noted) |
| --- | --- |
| where does saturation start? | Tensor-pipe activity reaches 90% of its plateau (65-75%) at **C\*~128** on the held-out prompts for every speculative arm (plain: C=256) and at **C\*~64** on summarization, where prefill is 60-75% of GPU time. The target verify step turns compute-bound (marginal cost per token within 80% of its ~30 us asymptote) from ~770 tokens per step: C~96 at k=7, C~190 at k=3. |
| how much is wasted at k=7, C=128? | **DSpark e10: 22% of GPU time** (16% verifying rejected drafts, 6% drafting slots never accepted), **45% of the work speculation adds**. EAGLE e10: 22% and 46%. Yu's "half of the extra computation" holds at k=7 and C=64-128 (45-49%); at k=3 it is 24-26%, at k=1 11-13%. The rest of the step: prefill 29%, useful verify 36%, sampling 5%. At C=256 the waste drops to 12% of GPU time (prefill and the KV read grow). |
| and out of distribution? | Summarization (accepted length ~2.2 at k=7): 17-19% of GPU time and **77% (DSpark) / 81% (EAGLE) of the speculative work** is wasted at C=128, but prefill (61-63%) dominates the step, so no k beats plain by much there. |
| cost per token under load | GPU-seconds per 1M output tokens at C=128 / C=256: plain 153 / 128; DSpark e10 best k (k=3) 113 / 106, k=7 123 / 117; EAGLE e10 best k (k=3) 108 / 101, k=7 119 / 112; paper scheduler (cut + KV) 113-115 / 105-109. Summarization: plain 540 / 538, EAGLE e10 k=1-3 526-528, DSpark k=7 606, EAGLE k=7 599 (worse than plain). Energy ~0.054 J/token for the best k at C=256 against 0.066 for plain (module power, see caveats). |
| which slots pay? | The marginal rule (slot j pays iff p_accept(j) x step time per emitted token > its draft + verify cost) keeps all 7 slots up to C=8, ~5 at C=32 and 3 at C=64-128, 2 at C=256; the measured best fixed k is 7, 7, 7, 5, 3, 3-4 at C=1-256. |
| how much could be won back? | A load-dependent k (best fixed k per level) beats the single best fixed k by 4% geomean over C=1-256 (DSpark: +12% at C=1, +1-2% at C>=32). A per-request oracle that verifies exactly the accepted drafts (drafting unchanged) is 15-18% above the best fixed k at C=64-256 in decode-only steps, ~10-12% once prefill is counted. |
| and the drafter's architecture? | Drafting is 12-15% of GPU time for DSpark at k=3-7 and C>=128, mostly the backbone (1.29 us per query token per layer; the 131k LM head is 19%). EAGLE costs 0.53 ms + 4.1 us/request per pass, so k=7 pays 7 passes: 2.0x DSpark's draft time at C=1 (3.7 vs 1.85 ms). Fewer DSpark layers at the same acceptance would add (model) +10% at C=128 for 1 layer, +6% for 2, +3% for 3; under load the verify of rejected tokens, not the drafter, is the larger waste. |

## Method

**Runs.** waste-prof-a (plain, DSpark e10 k=1/3/7) and waste-prof-b (DSpark
e10 k=5, EAGLE e10 k=1/3/7), each arm under Nsight Systems with the NVTX
overlay and GPU-metrics sampling at 1 kHz, on the 128 held-out OPB prompts
and the 128 summarization prompts at C=1, 8, 32, 64, 128, 256. One 10 s
capture window per level, 8 s into its measured requests. Settings as in
`docs/system-optimizations-8b.md` ("Setup"). Profiled throughput is not used:
cost per token comes from the unprofiled records (k-grid-opb-a/b/e/f,
kv-cost-a/b, k-grid-e10-a/b, calibrate, k-grid-a), all on the same image
and settings.

**Per-step split** (`waste_report decompose`, cost table from kv-cost-a's
`dspark-e10-k7-paper-cut-kv` arm). Every engine step of a window (NVTX
`engine:step`, projected onto the GPU) is split into:

- *prefill*: the target forward's share of prompt tokens in that step
  (token share, from the worker's `execute_context_N(T)_generation_M(T)`
  annotation);
- *rejected verify*: the step's rejected draft tokens (its decode tokens x
  the window's rejected share from the server counters) x the marginal
  verify cost per token at the step's size, from the cost table; *useful
  verify* is the rest of the target forward;
- *draft accepted / wasted*: the draft step (`draft:propose`) less the
  DSpark context part (`draft:fc`, `draft:context_kv`, listed as fixed),
  split by the accepted share of drafted tokens;
- *sample* (sampling and rejection sampling, postprocess), other time inside
  the step, and GPU time between steps.

"Share of the speculative work wasted" divides rejected verify + wasted
drafting by all drafted-token verify (priced the same way) + all drafting.
Windows that caught only a level's drain (opb C=32 for k>=3: one pass of
128 prompts ends before the window) are left out.

**Saturation.** Mean `Tensor Active` / `SMs Active` per window (nsys GPU
metrics); C\* is the first level within 90% of the arm's maximum. The verify
slope comes from the cost table (`verify_saturation_tokens`).

**Analytic model** (`CostModel`, `marginal_rule`, `predicted_speedup`,
`oracle_verify`): decode-only steps from the cost table plus a KV read of
3.6e-5 ms per cached token, opb acceptance per position from the cells, 800
cached tokens per request. It ignores prefill, so it overstates the speedup
under load (predicted DSpark k=5 at C=256: 1.08x; measured 1.18x in
k-grid-opb-b, whose plain arm also prefills).

## Results

### Where the GPU time goes (held-out prompts)

Share of GPU time in the capture window (`results/8b/waste/decomposition.md`
has every arm, workload and level):

| arm | C | prefill | useful verify | rejected verify | draft, accepted slots | draft, wasted slots | draft fixed | sample | wasted (GPU time) | wasted share of speculative work |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| plain | 128 | 31% | 65% | - | - | - | - | 4% | - | - |
| DSpark e10 k=1 | 128 | 34% | 49% | 2% | 9% | 1% | 1% | 4% | 3% | 11% |
| DSpark e10 k=3 | 128 | 34% | 41% | 7% | 8% | 3% | 1% | 4% | 10% | 24% |
| DSpark e10 k=5 | 128 | 33% | 38% | 10% | 8% | 4% | 1% | 5% | 15% | 35% |
| DSpark e10 k=7 | 1 | 4% | 68% | 0% | 12% | 9% | 2% | 4% | 10% | 41% |
| DSpark e10 k=7 | 64 | 17% | 41% | 18% | 9% | 8% | 1% | 6% | 26% | 48% |
| DSpark e10 k=7 | 128 | 29% | 36% | 16% | 7% | 6% | 1% | 5% | 22% | 45% |
| DSpark e10 k=7 | 256 | 31% | 43% | 6% | 7% | 5% | 1% | 6% | 12% | 40% |
| EAGLE e10 k=1 | 128 | 36% | 52% | 2% | 5% | 1% | - | 4% | 3% | 13% |
| EAGLE e10 k=3 | 128 | 36% | 41% | 8% | 7% | 3% | - | 5% | 11% | 26% |
| EAGLE e10 k=7 | 1 | 3% | 55% | 1% | 16% | 22% | - | 4% | 22% | 58% |
| EAGLE e10 k=7 | 64 | 16% | 38% | 18% | 12% | 11% | - | 5% | 29% | 49% |
| EAGLE e10 k=7 | 128 | 32% | 34% | 16% | 7% | 6% | - | 5% | 22% | 46% |
| EAGLE e10 k=7 | 256 | 33% | 43% | 7% | 7% | 5% | - | 6% | 12% | 43% |

At low load rejected drafts cost almost nothing to verify (the step is
memory-bound: the marginal token is free), and the waste is drafting; EAGLE
k=7 at C=1 spends 38% of the GPU time drafting, 22% on slots that are never
accepted. Past C\* every rejected token costs ~30 us of verify, and verify
becomes the main waste. At C=256 the share falls again because prefill and
the KV read of the larger batch grow. GPU time between steps is below 1%
(async scheduling hides the CPU), so CPU and launch gaps are not a lever.

On summarization the same arms spend 61-75% of the GPU time in prefill from
C=64 on; at k=7 and C=128 11% goes to rejected verify and 6-8% to wasted
drafting, 77% (DSpark) and 81% (EAGLE) of the speculative work (accepted
length 2.2 of 8).

### Saturation

Tensor-pipe activity (% of peak) by concurrency, held-out prompts:

| arm | 1 | 8 | 64 | 128 | 256 | C\* |
| --- | --- | --- | --- | --- | --- | --- |
| plain | 2 | 6 | 36 | 49 | 58 | 256 |
| DSpark e10 k=3 | 4 | 14 | 45 | 65 | 69 | 128 |
| DSpark e10 k=7 | 4 | 23 | 56 | 70 | 72 | 128 |
| EAGLE e10 k=3 | 4 | 14 | 43 | 65 | 69 | 128 |
| EAGLE e10 k=7 | 5 | 20 | 47 | 67 | 71 | 128 |

Summarization saturates at C=64 for every arm (65-75%). The plateau is ~70%,
not 100%: attention, normalisation and sampling kernels do not use the
tensor cores. DRAM read bandwidth stays below 45% at C>=128.

### Cost per token

GPU-seconds per 1M output tokens (one GPU per deployment), unprofiled
records:

| arm | opb C=1 | opb C=128 | opb C=256 | summ. C=128 | summ. C=256 |
| --- | --- | --- | --- | --- | --- |
| plain | 5,850 | 153 | 128 | 540 | 538 |
| DSpark e10 k=1 | 4,015 | 129 | 114 | (waste-sum) | (waste-sum) |
| DSpark e10 k=3 | 2,388 | 113 | 106 | (waste-sum) | (waste-sum) |
| DSpark e10 k=5 | 1,904 | 114 | 108 | 564 | 563 |
| DSpark e10 k=7 | 1,705 | 123 | 117 | 606 | 607 |
| EAGLE e10 k=1 | 3,534 | 121 | 108 | 526 | 528 |
| EAGLE e10 k=3 | 2,426 | 108 | 101 | 528 | 525 |
| EAGLE e10 k=7 | 2,124 | 119 | 112 | 599 | 596 |
| paper scheduler (cut + KV) | 1,750 | 113-115 | 105-109 | (waste-sum) | (waste-sum) |

All arms draw 450-540 W at every load, so J/token follows GPU-seconds:
0.054 J/token for the best k at opb C=256, 0.066 for plain, 0.28-0.31 on
summarization (`results/8b/waste/cost-per-token.md`).

### The marginal rule and the best k

With the cost table, opb acceptance and the fitted EAGLE pass cost, the
slots that pay (`marginal_rule`) and the best whole draft length
(`predicted_speedup`, decode-only):

| C | DSpark k=7: slots that pay | model best k | measured best k | EAGLE k=7: slots that pay | model best k | measured best k |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 2-7 | 7 | 7 | 1-7 | 7 | 7 |
| 8 | 2-7 | 7 | 7 | 1-7 | 7 | 7 |
| 32 | 2-5 | 4 | 7 | 1-4 | 4 | 4 |
| 64 | 2-3 | 3 | 5 | 1-2 | 3 | 3 |
| 128 | 1-3 | 3 | 3 | 1-3 | 2 | 3 |
| 256 | 1-5 | 2 | 3-4 | 1-5 | 2 | 3 |

DSpark's slot 1 carries the drafter's whole fixed cost (1.6-1.9 ms at small
batch), so on its own it "does not pay" at low load; whole lengths are
judged by `predicted_speedup`. At C=256 the cost table's slope is noisy
near its end (2048 tokens), which is why the slot rule and the best length
disagree there.

### Drafter cost model

Fitted from the profiled windows (`waste_report fit`,
`results/8b/waste/drafter-costs.json`; projected GPU ms per draft step):

- **EAGLE e10** (34 windows): first pass (draft prefill over the step's
  target tokens) 0.509 ms + 3.61 us/request + 0.817 us/target token; each
  later pass **0.529 ms + 4.10 us/request**. Predictions within 12% of
  every window (largest: later passes at C>=128, +11% on held-out prompts,
  -12% on summarization).
- **DSpark e10** (45 windows, 9 eager): fixed floor 1.50 ms in graph mode
  (0.223 ms per layer + 0.391 ms head, split by parameter count), fc
  0.038 ms + 0.264 us/target token, context KV 0.011 ms + 0.054 us/target
  token per layer, backbone 1.29 us per query token per layer, attention
  over the context 0.25 ns per query token per cached token per layer
  (window 2048), LM head 1.47 us per query row (19% of the query-block
  slope), Markov loop ~0 per slot, 0.073 ms around the graph. Mean absolute
  error 6%, at most 22% (k=1 on summarization at C=128).
  When the query block exceeds the 512-token graph sizes (k=7 from C~74,
  k=3 from C~171) the draft step runs eagerly.

Depth extrapolation (`DSparkComponents.draft_ms(..., layers=L)`), k=7, 800
cached tokens: 1.69 / 3.75 / 10.2 / 18.7 ms at B=1/32/128/256 for 5 layers;
0.71 / 1.43 / 3.66 / 6.64 ms for 1 layer. At the same acceptance (an upper
bound: fewer layers accept less) a 1-layer DSpark would be 19% faster at
C=1 and ~10% at C=128 (model); under load the bigger lever is not verifying
rejected drafts.

### Ceilings

- Best k per level vs the best single fixed k (measured, held-out prompts):
  DSpark +4.2% geomean (+12% at C=1, +10% at C=8, +1-2% at C>=32); EAGLE
  +4.2% (+14%, +10%, +2.5% at C=32, 0 at C>=64).
- Per-request oracle (verify only the drafts that will be accepted, draft
  all 7 slots): 1.21x / 1.18x / 1.16x / 1.15x above the model's best fixed k
  at C=32/64/128/256 in decode-only steps; with prefill at ~30% of GPU time
  this is ~10-12%, matching the 16 Oct estimate.

### Caveats

- Power is nvidia-smi `power.draw`, which on GH200 appears to include more
  than the GPU (450 W at C=1); J/token is an upper bound.
- Prefill vs verify inside a mixed step is split by token count; attention
  over long prompts makes prefill tokens somewhat dearer, so prefill is a
  lower bound and verify an upper bound in mixed steps.
- The rejected-verify price is the cost table's marginal cost, measured at
  short contexts with the KV read added separately; it is extrapolated
  beyond 2048 tokens per step.
- The opb C=32 windows of k>=3 arms caught only the end of the level and
  are excluded; summarization C=256 runs at ~125-150 requests (KV capacity),
  so it repeats C=128.

## Profiling a sweep arm (Nsight Systems)

### What is installed

Nsight Systems is not in the serving image (only `ncu` 2025.3 is) and not on
the login node. The CLI comes from NVIDIA's sbsa repository, extracted (no
install) to scratch, which the serving container mounts (`/iopsstor` in
`methods/eagle/configs/train-env.toml`), so no extra mount is needed:

| item | value |
| --- | --- |
| package | `nsight-systems-2025.6.3_2025.6.3.541-1_arm64.deb` |
| URL | `https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/sbsa/nsight-systems-2025.6.3_2025.6.3.541-1_arm64.deb` |
| sha256 | `4d0e0dc88a940146da9880e62c34a2ccc6d3a4f973d687667d694a9b2bbd0c59` |
| extracted to | `/iopsstor/scratch/cscs/faruk_zahiragic/pydeps/nsight-systems-2025.6.3/` (`ar x` + `tar`; no `dpkg-deb` on the login node) |
| binary | `.../opt/nvidia/nsight-systems/2025.6.3/target-linux-sbsa-armv8/nsys` (`apertus_bench.profiling.DEFAULT_NSYS`) |

The `usr/local/bin/nsys` link in the package is absolute and does not
resolve from the extraction; use the binary path above.

### How a profiled arm runs

Add a `profile` block to a serving arm of a node-sweep spec:

```yaml
  - name: dspark-e10-k7-nsys
    method: dspark
    drafter: /capstor/.../dspark_apertus15-8b_open-perfectblend_thinking-off_2026-10-04_epoch10_final
    k: 7
    profile:
      tool: nsys
      window_s: 15        # capture length inside each level
      delay_s: 10         # from the start of the level's measured requests
      levels: [64, 128]   # levels to capture; default all
      # loadtests: [opb]  # load-test labels to capture; default all
      # gpu_metrics: true # adds --gpu-metrics-devices=cuda-visible
      # eager_draft: true # draft step without CUDA graphs: component ratios only
      # nvtx: false       # no NVTX overlay ranges
      # extra_args: [...] # more nsys profile flags
```

`src/apertus_bench/profiling.py` and `node_sweep.py` then:

1. run `nsys --version`, `nsys status --environment`, a 2-second GPU-metrics
   sample and an `ncu` counter check on the arm's GPU (`nsys-env.txt`);
2. serve the arm as before plus `--profiler-config.profiler cuda` (the fork's
   hook that turns `/start_profile` into `cudaProfilerStart` in the worker),
   under
   `nsys profile --trace cuda,nvtx,osrt --cuda-graph-trace node
   --capture-range cudaProfilerApi --capture-range-end repeat --sample none
   --cpuctxsw none -o {arm_dir}/nsys/{arm}`: nothing is collected outside the
   windows, and each window is its own report;
3. set `VLLM_APERTUS_NVTX=1`, which activates the NVTX overlay
   (`serving/patches/vllm-apertus-nvtx.patch`; `serving/node-sweep.sh` adds
   it to the overlays whenever the spec has a profiled arm);
4. pass `--profile-levels/--profile-window/--profile-delay` to the load-test
   client, which in each chosen level POSTs `/start_profile` `delay_s` into
   the measured requests and `/stop_profile` `window_s` later (or when the
   level runs out of requests: `truncated`), and writes
   `profile-window.json` into the cell with the server counters at both ends
   (generated and prompt tokens, drafts, accepted tokens) and the times;
5. after the load tests stop the server (nsys writes all reports when the
   server exits; it gets `report_timeout_s`, default 900 s, before SIGKILL),
   and export `nsys stats` CSVs to `{arm_dir}/nsys-stats/`:
   `cuda_gpu_kern_sum`, `cuda_gpu_mem_time_sum`, `cuda_api_sum`, `nvtx_sum`,
   `nvtx_gpu_proj_sum`, `nvtx_gpu_proj_trace`, `osrt_sum`, plus a
   GPU-metrics summary when sampled, and `manifest.json` pairing each report
   with its window.

Every cell of a profiled arm carries `profile` in its metadata. Profiled
numbers are never headline numbers: put the unprofiled twin in the same job.
Arms without `profile` are served and driven exactly as before (tests in
`tests/test_profiling.py`).

The `.nsys-rep` and `.sqlite` files stay on the cluster under
`{arm_dir}/nsys/` (git-ignored); their paths are in `nsys-stats/manifest.json`
and the arm's `status.json`. Commit `nsys-stats/`, `nsys-env.txt` and the
cells, never the reports.

### NVTX ranges

The fork's model runner v2 path has no NVTX ranges of its own (only the
worker's `execute_context_N(T)_generation_M(T)` annotation, which the cuda
profiler hook emits around each target forward and which carries the batch
composition). The overlay touches only `gpu_worker.py`, which at import calls
`apertus_bench.vllm_nvtx.install()` when `VLLM_APERTUS_NVTX=1`; that wraps
methods at run time, so it composes with the scheduler patches that edit
`model_runner.py` and the speculators. With one GPU per arm the worker lives
in the EngineCore process, so one install covers the scheduler too.

| range | what |
| --- | --- |
| `engine:step` | one EngineCore step |
| `engine:schedule` | `Scheduler.schedule` |
| `engine:update_from_output` | output processing in the engine |
| `target:execute[mixed]` / `[decode]` | target forward of a step with / without prefill tokens |
| `target:graph` | the target's CUDA-graph replay |
| `target:sample`, `target:rejection_sample` | sampling, rejection sampling |
| `target:postprocess` | `postprocess_sampled` |
| `draft:propose` | the whole draft step |
| `draft:fc`, `draft:context_kv` | DSpark aux-hidden combine and context K/V precompute (eager in every mode) |
| `draft:graph` | DSpark query-block graph: backbone, LM head and Markov loop |
| `draft:backbone`, `draft:lm_head`, `draft:markov`, `draft:markov_head` | DSpark components, eager draft only |
| `draft:pass0`, `draft:pass`, `draft:sample` | EAGLE first pass, each later pass, sampling |

The DSpark draft step is a single full CUDA graph. Ranges pushed while a
graph is captured are not replayed, so inside it only kernels are visible
(`--cuda-graph-trace node`); the per-component ranges need
`eager_draft: true`, which runs the draft step without graphs (the target
keeps its graphs). Use that for component ratios, not timings.

### Reading the stats

```bash
nsys stats --report nvtx_gpu_proj_sum --format csv REPORT.nsys-rep   # GPU time per range
nsys stats --report cuda_gpu_kern_sum --format csv REPORT.nsys-rep   # per kernel, graph nodes included
```

`profile-window.json` gives the tokens generated inside the window, so GPU
time per range divided by those tokens is GPU-ms per output token.

### Smoke test (nsys-smoke, job 3722011)

Plain and DSpark e10 k=7, each unprofiled and profiled (10 s windows, 5 s into
the level), opb held-out prompts at C=8 and C=128, one node, debug partition.
All four arms `ok`; record in `results/8b/sweeps/nsys-smoke-20261022T142254Z/`.

**Environment** (`nsys-env.txt` of the profiled arms): nsys 2025.6.3.541 runs
in the container; `nsys status --environment` OK (paranoid level 0, no root).
**GPU-metrics sampling is permitted** (`--gpu-metrics-devices cuda-visible`
produced "General Metrics for NVIDIA GH100" and a report; the check's exit
143 is `--duration` ending `sleep`), and **`ncu` may read counters**
(`sm__throughput` returned a value, no ERR_NVGPUCTRPERM). nsys warns that the
NVIDIA persistence daemon is not running (slower GPU initialisation only).

**Harness check**: the unprofiled arms against eagle-e10 (job 3709465):

| arm | C | this job, tok/s | eagle-e10, tok/s | diff |
| --- | --- | --- | --- | --- |
| plain | 8 | 1148 | 1176 | -2.4% |
| plain | 128 | 6468 | 6572 | -1.6% |
| DSpark e10 k=7 | 8 | 3526 | 3482 | +1.3% |
| DSpark e10 k=7 | 128 | 8063 | 8143 | -1.0% |

Acceptance length 4.77 in both jobs.

**Reports**: one per window (`{arm}.1.nsys-rep` at C=8, `.2` at C=128,
9-26 MB, 24-80 MB SQLite). `nsys stats` exported all CSVs (exit 0). Kernels
inside CUDA graphs are visible in `cuda_gpu_kern_sum` (e.g. 26k instances of
the target's nvjet GEMMs in the C=8 window, where the target and draft run as
full graphs); in `nvtx_gpu_proj_sum` a graph replay counts as one GPU op but
its projected time covers the graph. All NVTX ranges of the overlay appear.

**Overhead.** Inside a capture the engine steps run at the unprofiled pace:
the NVTX step counts give 10.5 ms per step for DSpark at C=8 (unprofiled,
from tok/s and acceptance: 10.8 ms), 69 vs 76 ms at C=128, and 17.2 vs
19.8 ms for plain at C=128. The cost is at the window edges:
`/start_profile` takes 0.3-1.7 s and the stop stalls the engine for
3-7 s while CUPTI flushes (`open_s` 13.1-17.1 s for 10 s windows). A
profiled level therefore runs 3.5-9.2 s longer than its twin, and its
client-side tok/s is 10-15% (plain) and 10-28% (DSpark) lower; p50 TPOT is
within 1-3% except DSpark C=128 (+13%). Rules: take throughput from
unprofiled arms; take time decompositions from the trace; keep levels long
enough (tens of seconds) that a window is in steady state.

**Caveats found:**

- nsys writes the reports only at server exit, not after each window
  (fixed in the harness: no wait before the stop, long SIGINT grace).
- At C=128 the DSpark draft step does not run as a CUDA graph: 128 requests
  x 7 query tokens exceed the graph sizes, so `draft:backbone`,
  `draft:lm_head`, `draft:markov` appear with ~70 kernel launches per step
  (C=8: one `draft:graph` replay per step). Under load the draft's component
  ranges are therefore available without `eager_draft`.
- Two profiled arms on one node are fine (separate GPUs, separate nsys
  sessions).

## Runs

| run | job | what |
| --- | --- | --- |
| `results/8b/sweeps/nsys-smoke-20261022T142254Z` | 3722011 | harness smoke test (debug); reports on the cluster under each `*-nsys/nsys/` |
| `results/8b/sweeps/waste-prof-a-20261022T161309Z` | 3722945 | profiled: plain, DSpark e10 k=1/3/7, opb + summarization, C=1-256 |
| `results/8b/sweeps/waste-prof-b-20261022T161311Z` | 3722946 | profiled: DSpark e10 k=5, EAGLE e10 k=1/3/7 |
| waste-sum-a / waste-sum-b (queued) | 3737492 / 3737493 | unprofiled summarization: plain, DSpark k=1/3, paper scheduler with cut + KV |
| k-grid-opb-a/b/e/f, kv-cost-a/b, k-grid-e10-a/b, calibrate, k-grid-a | earlier | unprofiled cost per token (reused) |

Analysis outputs: `results/8b/waste/` (`decomposition.{json,md}`,
`drafter-costs.json`, `cost-per-token.{json,md}`), from
`python -m apertus_bench.waste_report {decompose,fit,cost}` with the cost
table `results/8b/sweeps/kv-cost-a-20261015T230650Z/dspark-e10-k7-paper-cut-kv/cost-table.json`
(on `spec-exp-base`). Profiled reports (`.nsys-rep`, `.sqlite`) stay on the
cluster under `/iopsstor/scratch/cscs/faruk_zahiragic/spec-exp/waste-profiling/results/8b/sweeps/<run>/<arm>/nsys/`.
