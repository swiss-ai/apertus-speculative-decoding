# Wasted compute past saturation, Apertus 1.5 8B

Yu (Slack, 16 Oct): once speculative decoding is compute-bound the speedup
drops; how much compute is wasted in that region (e.g. k=7 at C=128 accepting
~3.5 tokens), what does a token cost at high load, and how do both relate to
the drafter's architecture? Rob suggested Nsight Systems for the measurements.

Status 2026-10-22: Nsight harness built; smoke test (nsys-smoke) queued. See
"Profiling a sweep arm" for how to use it; sweeps and the waste analysis follow.

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
5. after the load tests wait for nsys to finish writing the reports, stop the
   server, and export `nsys stats` CSVs to `{arm_dir}/nsys-stats/`:
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
