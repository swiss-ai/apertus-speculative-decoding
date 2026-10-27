# Draft CSCS support request — Beverin hipBLASLt sharded GEMM overhead

**Draft only; not sent. Updated 2026-10-27: the rocBLAS workaround also
passed full-context training and save/resume on 16 nodes (697549) and
32 nodes (697813). The underlying hipBLASLt-path cause remains unresolved.**

Subject: Beverin MI300A: ~500 ms sharded GEMM dispatch in ROCm vLLM 0.31
image; forcing rocBLAS restores sub-millisecond small-matmul latency

Hello CSCS support,

We found a reproducible software-path performance problem while investigating
PyTorch FSDP2/Muon training on Beverin (`infra01`, `mi300`). What initially
looked like an inter-node RCCL problem also occurs on one node. A minimal
sharded-matmul test isolates the difference, and selecting rocBLAS instead
of hipBLASLt removes it in that test.

Each training node runs TP2 vLLM extraction on GPUs 0,1 and two FSDP2 trainer
ranks on GPUs 2,3. The target is Apertus-v1.5-70B; the drafter is an
approximately 6B-parameter, five-layer DSpark model trained with Muon.

Training observations:

| Measurement | Jobs | Observed performance |
|---|---|---|
| Original one-node reference, vLLM 0.23 / torch 2.11, SEQ 4096 | 693757 | ~4.6 s/step: forward 0.68 s, backward 0.56 s, optimizer 3.37 s |
| Original two-node and sixteen-node runs, v0.31 | 693946–693947, 693916 | ~181 s and 191–214 s/step; both all-gather and all-to-all Muon slow |
| New one-node full-context run, v0.31 | 694425 | ~279 s/step: forward ~0.81 s, backward ~56 s, optimizer ~222 s |
| New two-node full-context baseline, v0.31 | 694426 | 183 s/step median: forward 0.72 s, backward 51.8 s, optimizer 128 s |
| Reused-buffer RCCL benchmarks, matching trainer GPU layout | 693943–693944 | ~68 GB/s all-reduce, all-gather and reduce-scatter; ~20 GB/s all-to-all |
| Fresh buffers, 512 MiB, including allocation | 693957–693960 | ~6–9 GB/s all-gather/reduce-scatter |

The original one-node reference used a different image and smaller batch;
its comparison with the slow multi-node jobs was confounded. Job 694425
establishes that inter-node traffic is not necessary for the slowdown.

The CPU torch profiler in 694426 attributes 51.55% of self CPU time to
`aten::addmm` and 46.50% to `aten::mm`, 533.6 s in three profiled steps.
Average CPU total durations are ~248–250 ms per call. In contrast,
`_c10d_functional::wait_tensor` totals 71.9 ms across 1,881 calls. These are
CPU timings; they do not directly time GPU network kernels.

An isolated minimal reproduction, job **694440**, compares plain,
replicated-DTensor and sharded-DTensor BF16 matrix products. Each rank starts
with a 1024×1024 local tensor; the global sharded shape scales with world
size. Both images were tested on the same allocated nodes. Warmed medians
omit iteration 0:

| Image / preference | One node, two ranks | Two nodes, four ranks |
|---|---|---|
| v0.31, default hipBLASLt, sharded | **478 ms** | **516 ms** |
| v0.23, default, sharded | **1.28 ms** | **4.81 ms** |
| v0.31, forced rocBLAS, sharded | **0.402 ms** | not separately microbenchmarked |

Plain GEMMs are ~0.05 ms and replicated DTensor GEMMs ~0.1 ms in both
images. Python cProfile puts the sharded delay inside the native ATen call.
The rocBLAS control uses
`torch.backends.cuda.preferred_blas_library("cublas")`; PyTorch retains the
CUDA API names on ROCm. The hypothesis is an issue in the newer image's
hipBLASLt path for gathered/sharded operands; we have not identified its
native implementation cause.

Full training confirms the workaround in the same v0.31 image. Job 694469
saved at step 20, resumed to 40, and saved again: median 12.45 s/step across
38 steady samples, backward 0.674 s, optimizer 2.89 s. Peak sampled node memory
was 407 GiB, with 77 GiB minimum available; the resumed save peaked at
403 GiB. Job 694470 ran six steps on two nodes and saved: median 14.8 s/step,
backward 0.674 s, optimizer 2.99 s. Feature production now limits much of the
step.

Follow-up unprofiled, full-context pilots on 2026-10-27 validated the same
workaround on **16 nodes (697549)** and **32 nodes (697813)**. Both saved at
step 100, resumed exactly at 100, and saved again at 200. Mean steady step
times were **18.04 s** (197 samples) and **19.71 s** (195 samples), respectively.
All nodes confirmed the rocBLAS preference. These results establish a
practical workaround at scale, while the native cause of the default-path
slowdown is still unknown.

Other controlled two-node trials (five steady samples after warm-up):

- 694427: bf16 FSDP gradient reduction — 193 s/step median.
- 694428: `NCCL_PROTO=Simple` — 178 s.
- 694429: `NCCL_MIN_NCHANNELS=16` — 179 s; actual collective channels remained 36.
- 694430: `NCCL_NCHANNELS_PER_NET_PEER=8` — 181 s.
- 694432: `FI_MR_CACHE_MONITOR=disabled`, applied after the enroot hook — 175 s.
- 694431: host netstack plus rocm6 plugin — fails before serving/training:
  injected `/lib64/libselinux.so.1` requires `GLIBC_2.38`, absent from the
  image's libc. There is no step-time result for this variant.
- Earlier `OFI_NCCL_DISABLE_DMABUF=1` and expandable HIP allocator segments
  did not restore benchmark bandwidth. The earlier intended disabled-MR
  controls 693944/693959 actually logged `userfaultfd`: enroot's hook
  overrode the external environment. 694432 is the valid disabled control.

Environment and artifacts:

- Four MI300A per node; host and GPUs share approximately 501 GiB.
- v0.31 image: `/capstor/scratch/cscs/faruk_zahiragic/container-images/vllm-openai-rocm-v0.31.0.sqsh`,
  SHA-256 `b58797bb6a430d7587f1c10ec24ac71f198bd8d590bde98dbbe68d7744a9816f`.
  Python 3.12, torch `2.13.0+git733fca1`, HIP `7.2.53211`, vLLM
  `0.31.0+rocm723`, RCCL `2.27.7-HEAD:96a25b5`.
- v0.23 image: `/capstor/scratch/cscs/faruk_zahiragic/container-images/rocm-vllm-0.23.0-cdna.sqsh`,
  SHA-256 `75f5553dbe4b4a8bc431bfd3fe78b38ebf9724d09dfc0a2a4e7c85b0f84868df`.
  Python 3.14, torch `2.11.0+rocm7.14.1`.
- Default enroot netstack:
  `x86_64/26.08.1/gpu_rocm7-cxi_13.1.0-ofi_2.6.0-aws_1.20.0`,
  aws-ofi-nccl 1.20.0, libfabric 2.6, provider `cxi`, four NICs per node.
  Training logs confirm this provider. No host/TCP fallback was identified.
- Speculators base `7046ac63fac4eeb218dc2e92a385c7d1ec8c4e00`, with recorded
  Apertus compatibility, deadline, stack/watchdog and DTensor all-gather
  patches. Trainer and extraction processes have all node CPUs available.
- Raw launch/trainer/collective logs, resolved environments, exact commands,
  patches and four-rank CPU Chrome traces are in
  `/iopsstor/scratch/cscs/faruk_zahiragic/beverin-feasibility/`, under
  `runs/<run>/logs/` and `runs/<run>/deployments/<job>/`.
- Minimal comparison: `runs/dtensor-compare-694440/`, including the exact
  scripts and image hashes.
- Repository branch `dspark-70b-beverin`, runtime workaround commit
  `ada15200`; `methods/dspark/beverin/dtensor_probe.py` is the small reproducer.
  `BEVERIN_BLAS_BACKEND=cublas` selects rocBLAS in the trainer prelude.
- Repository artifacts:
  [`results/70b/dspark/beverin-20261025/`](../results/70b/dspark/beverin-20261025/)
  and [`results/70b/dspark/beverin-scaling-20261027/`](../results/70b/dspark/beverin-scaling-20261027/),
  including raw logs, configuration/provenance records and SHA-256 manifests.

Could you advise whether this hipBLASLt behavior for sharded/gathered inputs
is known, and recommend a compatible ROCm/PyTorch image or library fix?
We would also appreciate guidance on native profiling inside the enroot
namespace and on a host-plugin combination that avoids the GLIBC mismatch.
