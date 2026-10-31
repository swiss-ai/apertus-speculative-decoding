# Targeted numerical investigation, 31 October 2026

No further language-model training is authorized by this investigation. The
user requested a cause rather than another training retry, and specifically
asked that CSCS's ALPS vLLM image be considered. Production and Clariden jobs
remain outside the scope of any modifications.

## What the original evidence can establish

The original first **logged** NaN is counter 519. Metrics come from the forward
pass, but are reduced across ranks and logged **after** backward, clipping and
optimizer updates. Finite forward metrics at 518 therefore do not establish
that backward/update 518 was healthy. A localized bad update at 518 and a new
forward failure at 519 are both compatible with the observations. A defective
metrics collective would also invalidate inferences from individual reduced
fields. There is no original per-rank activation/gradient dump or retained
checkpoint immediately before the failure, so the historical trigger cannot
be uniquely reconstructed from those logs alone.

The 640-step diagnostic does not isolate a numerical fix: hooks alter graph
breaks, compilation and synchronization; the trajectory differs at counter 2.
It established a finite bounded run and functioning failure checks, not the
cause of the previous run's NaNs.

## Loss arithmetic controls

`check_loss_math.py` loads the captured loss/DSpark metric files unchanged.
All **18 finite-input cases** passed finite forward and backward checks in
FP32 and CPU BF16 autocast: full, partial and empty masks; logit scales 1,
20 and 1000; saturated confidence logits. Empty masks return zero loss and
zero logit gradients. This does not test the ROCm compiler or missing batch.

Four deliberately poisoned controls establish useful distinctions:

- A NaN in target logits spoils TV/BCE but leaves CE and the forward confidence
  prediction statistic finite. It does not by itself explain the original
  combination of NaN CE and NaN predicted confidence.
- A NaN in a masked draft-logit/confidence row spoils CE, confidence prediction
  and loss despite otherwise finite predictions. Multiplying by a zero mask
  does not sanitize a NaN. This recreates the metric pattern by injecting the
  fault; it does **not** identify what created that fault in the real run.

Exact source hashes, host torch version and fixture limitations are retained
in `loss-math.json`. The helper accuracy functions are fixtures; the tested
loss and confidence code is the captured source.

`check_muon_alias.py` separately executes the exact image Muon function bodies.
The BF16 workspace alias found in the older Muon code mutates its persistent
momentum buffer with **nesterov=False**. Under the production **nesterov=True**
default, the temporary `grad.lerp(...)` output prevents that alias and the buffer
is preserved exactly in the CPU control. Both images contain this same source.
This is a demonstrated issue in an unused configuration, not a cause assigned
to this run. `muon-alias.json` preserves results and limitations.

## CSCS image comparison

The existing image campaign is on separate branch
`experiment/beverin-alps-vllm-dtensor-20261031`, worktree
`/home/projects/apertus-speculative-decoding-beverin-alps`; it was inspected
read-only. Its job 710006 demonstrates a sharded GEMM performance difference,
not a NaN/stability fix. The original failed training already forced rocBLAS,
which bypassed the measured slow default GEMM path.

The tested CSCS image is pinned to OCI digest
`7d95232cbc7511be5672d272f7a567ea52a3c5fbd5310ce0903956f0f37b5cc7`.
Its stack differs in PyTorch git revision, Triton and bundled communication
libraries. Its Muon source and FlexAttention Python lowering are byte-identical
to the original image, per the companion source audit; that does not establish
equivalence of their generated GPU kernels.

A relevant CSCS build hook explicitly disables P2P: it documents unreliable
RCCL IPC behavior under enroot on Beverin's kernel 6.4, with invalid-device-
pointer errors and hangs. The original run did not explicitly disable P2P.
Those documented symptoms are **not evidence of silent NaN corruption** in
the old stack. This is a concrete transport candidate to test, not a cause
assigned to the failed run. `alps-image-comparison/` preserves the image
identity, build hook, runtime environment and original comparison summary.

## Bounded correctness test

`comm-correctness/` contains a non-training, two-node/20-minute test of three
cells on the same nodes: original image/default P2P policy, original image with
P2P disabled, and CSCS ALPS image with its bundled stack/default P2P workaround.
Both BF16 and FP32 fresh/reused tensors are checked against bounded analytical
values for gather, reduce-scatter, shard redistribution, packed metric reduction
and structured DTensor GEMM. Every output element is checked; inputs do not grow
through repeated reductions. Runtime logs must establish actual transport use.
Each cell has a hard timeout and there is no automatic retry.

This tests a specific backend/transport candidate at low cost. Passing it cannot
exclude every intermittent kernel fault or establish full-model training
stability. A reproduced mismatch would be preserved with its exact operation,
rank, expected/observed value, image hash and transport logs before any fix is
claimed.

## Upstream reports checked for applicability

- [FlexAttention issue 153799](https://github.com/pytorch/pytorch/issues/153799)
  requires non-128-aligned sequence lengths and a score modifier. This run has
  Q=4096/KV=12288 and `score_mod=None`, so that reported trigger does not match.
- [Current Muon source](https://github.com/pytorch/pytorch/blob/main/torch/optim/_muon.py)
  uses a separate BF16 workspace. The local alias control above demonstrates
  why the older workspace alias is not active in this run's Nesterov path.
- [RCCL corruption report 6565](https://github.com/ROCm/legacy-rocm-build/issues/6565)
  reports incorrect gather/scatter results with RCCL 2.27.7, but on gfx1100
  hardware. It is not evidence that the same defect affects gfx942 MI300A.
  The pending microtest checks the relevant operations on the actual hardware.
