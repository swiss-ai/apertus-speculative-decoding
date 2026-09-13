# `draft_tensor_parallel_size=1` cannot be served at the pinned vLLM revision

Job 3392110, node nid006593, `debug` partition, 2026-09-13 18:59:08 CEST. No cell was measured:
the engine never started, so there are no request rows, no metric snapshots, and no KV-cache
numbers to report.

## What happened

The launch was the same as the working depth-3 draft arm except for `DRAFT_TP=1`, with the
`patches/vllm-apertus-image-token.patch` overlay bind-mounted as usual. The flag reached vLLM
intact — the engine's own config dump shows `'draft_tensor_parallel_size': 1` — and then every one
of the four workers raised, about 2.5 minutes after job start:

```text
ValueError: Currently, 'draft_tensor_parallel_size' and 'tensor_parallel_size' must be the same.
Got 1 and 4. Please pass 'draft_tensor_parallel_size' in the speculative_config.
```

`engine-error.log` has the full traceback from one worker. It fails in
`DraftModelProposer.__init__` at `vllm/v1/spec_decode/draft_model.py:32`, which calls
`_raise_if_draft_tp_mismatch` at line 74. The guard is unconditional and the source explains why it
exists:

> Note(Tomas Ruiz) If we run the target model with TP > 1 and the draft model with TP = 1, then the
> different TP ranks collide. Specifically when all ranks compile the draft model on rank 0
> (because TP=1), then the torch compile cache is overwritten and corrupted. We need a mechanism
> like this: https://github.com/vllm-project/vllm/pull/5414 To prevent this error, we assert that
> both TP sizes must be the same.

So mismatched draft TP is not implemented at revision `a601a9d998ddeb488f0c17e8512874b116aa7658`,
and the failure is at engine construction, before any memory profiling.

## What this means for the design

`configs/experiment.yaml` lists `draft_tensor_parallel_size: [1, 4]` and research question 3 asks
which combination of speculative depth and draft TP is best. **The TP=1 level is unreachable at the
pinned revision**, so RQ3 reduces to a depth sweep at draft TP = target TP until an upstream fix
lands. Only `draft_tensor_parallel_size=4` can be measured against this image.

The reason this cell was chosen — testing whether tensor-parallel communication on every draft step
explains the depth-3 TP=4 regression — therefore remains **untested**. The draft-cost explanation is
neither supported nor refuted by this attempt.

The anticipated failure mode was different and did not occur. The concern was that an unsharded 8B
drafter on rank 0 would leave too little room to reserve KV cache for one 131072-token request, the
check that killed job 3374717 at `max_model_len=262144`. The engine dies before KV-cache profiling,
so **`max_model_len` was never the binding constraint here and no context length was changed**. The
KV-capacity question at draft TP=1 is still open and cannot be answered without a vLLM that permits
the configuration.

## Not worked around

Deleting the guard with another bind-mount overlay would produce a run of unknown validity: the
comment says the ranks collide through a corrupted `torch.compile` cache, which is a silent
correctness and performance hazard rather than a clean failure. That is not a change to make
unreviewed for a measurement that is supposed to be trustworthy. The alternatives that do work
around it change a controlled factor — target TP=1 does not fit a 70B model on one GH200, and
target TP=2 with draft TP=2 is a different experiment.

The reasonable next steps are a depth sweep at draft TP=4 (`num_speculative_tokens` 2, 5, 8), which
tests draft cost by varying how much drafting happens per step, and the n-gram arm, which removes
the draft forward pass entirely and so isolates draft-model cost from verification cost.

## Provenance

- Slurm job 3392110, node nid006593, partition `debug`, `--time 01:00:00`, cancelled at 19:02 CEST
  once the failure was confirmed, about 3 minutes into the allocation.
- Served model name that would have been used:
  `faruk_zahiragic/swiss-ai/Apertus-v1.5-70B-draft-n3-tp1-faruk_zahiragic-20260913T165901Z`.
- Requested `max_model_len=131072`, `gpu_memory_utilization=0.8`, target TP=4, draft TP=1,
  `num_speculative_tokens=3`.
- `model-launch` revision `909026a990454557f1b54d26f24ec3ad92e51e35`; vLLM revision
  `a601a9d998ddeb488f0c17e8512874b116aa7658` (`0.23.1rc1.dev1029+ga601a9d99`) from image
  `/capstor/store/cscs/swissai/infra01/container-images/ci/vllm_apertus_1.5_release-arm64.sqsh`,
  with `patches/vllm-apertus-image-token.patch` bind-mounted over
  `/workspace/vllm/vllm/v1/spec_decode/llm_base_proposer.py`.
- `replica_health.json` never left `NOT_DEPLOYED` with a null `peer_id`.
