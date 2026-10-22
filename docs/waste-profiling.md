# Wasted compute past saturation, Apertus 1.5 8B

Yu (Slack, 16 Oct): once speculative decoding is compute-bound the speedup
drops; how much compute is wasted in that region (e.g. k=7 at C=128 accepting
~3.5 tokens), what does a token cost at high load, and how do both relate to
the drafter's architecture? Rob suggested Nsight Systems for the measurements.

Status 2026-10-22: Nsight harness built and smoke-tested (nsys-smoke, job
3722011); see "Profiling a sweep arm". Sweeps and the waste analysis follow.

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
