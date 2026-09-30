Probe workload (64 RedHatAI/speculator_benchmarks prompts, C=1/8/32/64, 384 tokens) on the
same DSpark deployment as 20260930T122216Z. Job 3552504 failed before the first cell:
the engine hit `RuntimeError: Triton Error [CUDA]: out of memory` at 14:48:34, ~75 s after
start-up. In the earlier run nvidia-smi showed 97,269 MiB used from start-up on (~95 GiB of
95 GiB, vs 83,263 MiB for the plain target), so there is almost no headroom for kernels
loaded lazily at run time.
