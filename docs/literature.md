# Literature map

This is the initial reading map, not the final literature review.

## Core algorithm

- Leviathan, Kalman, and Matias, [Fast Inference from Transformers via Speculative
  Decoding](https://proceedings.mlr.press/v202/leviathan23a.html), ICML 2023. Establishes an
  exact-distribution speculative-decoding algorithm and relates speed to the approximation model.
- Chen et al., [Accelerating Large Language Model Decoding with Speculative
  Sampling](https://arxiv.org/abs/2302.01318), 2023. Independently develops modified rejection
  sampling and evaluates a 70B target in a distributed setting.

Both papers motivate the Apertus 8B/70B pairing, but their reported speedups do not transfer
directly: speed depends on model ratio, kernels, parallel layout, batch shape, and workload.

## Serving context and alternative speculation

- Kwon et al., [Efficient Memory Management for Large Language Model Serving with
  PagedAttention](https://doi.org/10.1145/3600006.3613165), SOSP 2023. Provides the serving and
  KV-cache context behind vLLM; this project explicitly measures whether the draft reduces usable
  cache/concurrency.
- Miao et al., [SpecInfer: Accelerating Generative Large Language Model Serving with Tree-based
  Speculative Inference and Verification](https://arxiv.org/abs/2305.09781), ASPLOS 2024. Shows
  that serving conclusions depend on how candidates are constructed and verified under load.
- Cai et al., [Medusa: Simple LLM Inference Acceleration Framework with Multiple Decoding
  Heads](https://arxiv.org/abs/2401.10774), ICML 2024. Useful comparison for the operational cost
  of maintaining a separate draft model, though Medusa is not in the core experimental matrix.

## Implementation sources

- The experiment pins `swiss-ai/vllm` commit
  `a601a9d998ddeb488f0c17e8512874b116aa7658`. Its `SpecDecodingProm` implementation defines the
  aggregate counters and the bonus-token convention used by this harness.
- [vLLM production metrics](https://docs.vllm.ai/en/v0.22.1/usage/metrics/) documents the
  speculative counter family. The raw Prometheus exposition uses `_total` suffixes for counters.
- [vLLM online serving benchmark](https://docs.vllm.ai/en/v0.23.0/api/vllm/benchmarks/serve/)
  provides a useful cross-check for TTFT/TPOT and speculative-metric calculations.

## Local prior work

- `swiss-ai/bsezen-benchmark-reports`, especially the DeepSeek-V3 non-latent MoE sweep, provides
  reporting conventions: immutable raw data, explicit launch scripts, repeat disclosure,
  workload-specific capacity knees, and mechanistic GPU evidence.
- `swiss-ai/serving-metrics` and `swiss-ai/serving-plotter` provide the shared Prometheus/DCGM
  query and plotting layer.
- `swiss-ai/bench-agent` provides deterministic open-loop knee search, drain checks, repeated
  boundary soaks, and separation of serving failure from load-generator failure.

The main methodological inheritance is to hold the externally controllable contract fixed and
avoid treating one scalar such as acceptance rate, active parameters, or active FLOPs as a
sufficient performance model.
