# Drafters refused under pipeline parallelism (job 3721313, 2026-10-22)

Sweep `experiments/system-8b/pp-smoke.yaml` (record:
`results/8b/sweeps/pp-smoke-20261022T125029Z/`). It ran on the image
`vllm_apertus_1.5_release-arm64.sqsh` (vLLM 0.23.1rc1 fork, model runner v2) with the
standard `serving/patches` and no `SWEEP_PATCHES`. All arms used Apertus 1.5 8B with
`--pipeline-parallel-size 2 --tensor-parallel-size 1` on node nid006573.

| arm | drafter | runner | outcome |
| --- | --- | --- | --- |
| pp2-plain | none | v2 | served; chat cell at C=4 completed (128/128 requests, whole passes) |
| pp2-dspark-k3 | DSpark e10, k=3 | v2 | refused at config validation, before any GPU work |
| pp2-eagle-k3 | EAGLE e10, k=3 | v2 | refused at config validation |
| pp2-eagle-k3-mrv1 | EAGLE e10, k=3 | v1 (`VLLM_USE_V2_MODEL_RUNNER=0`) | refused at config validation |

The three drafter arms fail identically. `SpeculativeConfig` gives the draft model the target's
pipeline-parallel size (`vllm/config/speculative.py:1112`), and the draft architectures do not
implement `SupportsPP`:

```
File "/workspace/vllm/vllm/config/speculative.py", line 1171, in _verify_args
  self.draft_model_config.verify_with_parallel_config(
File "/workspace/vllm/vllm/config/model.py", line 1226, in verify_with_parallel_config
  raise NotImplementedError(
NotImplementedError: Pipeline parallelism is not supported for this model. Supported models implement the `SupportsPP` interface.
```

The image's source has two more refusals after this one. Getting past the config check alone
would not be enough:

- `vllm/v1/worker/gpu/model_runner.py:197-203` (model runner v2) raises
  `ValueError("<method> with pipeline parallel is not supported.")` for eagle3, dflash and dspark,
  because these drafters need auxiliary hidden states from the last pipeline stage.
- `vllm/v1/worker/gpu/spec_decode/dspark/utils.py:38-39` raises
  `NotImplementedError("DSpark does not support pipeline parallelism.")`.
- The v1 runner has no DSpark proposer (`vllm/v1/spec_decode/` has no dspark module).

**Design consequence.** PP > 1 with either drafter is not a deployable configuration on this
image. Removing that factor level would take an upstream change: a draft model that runs on the
last stage only, and the aux-hidden-state hand-off implemented across stages. The PP part of the
workload-coverage study is therefore plain PP2/PP4 for reference only (`docs/workload-coverage.md`).
