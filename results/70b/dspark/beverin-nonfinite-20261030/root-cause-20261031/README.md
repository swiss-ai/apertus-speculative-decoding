# Targeted numerical investigation, 31 October 2026

No further language-model training is authorized by this investigation. The
user requested a cause rather than another training retry, and specifically
asked that CSCS's ALPS vLLM image be considered. Production and Clariden jobs
remain outside the scope of any modifications.

## Latest finding: native abort in the original stack

Non-training job **710518** distinguishes the two image configurations on
the same Beverin node and physical GPUs 2/3. The original image aborts both
ranks with `HSA_STATUS_ERROR_ILLEGAL_INSTRUCTION` (code `0x2a`, SIGABRT)
during the first production-size **43008 × 8192** zero-input Muon case.
The six preceding small tall-matrix cases passed. The CSCS ALPS image then
passes all twelve tall-matrix cases per rank, including exactly zero output
for the large zero input. All its checked outputs are finite; the largest
raw Newton–Schulz relative L2 difference from its dense reference is 3.774%
(explicit BF16 tolerance 5%).

This is a native execution failure without a model, corpus, training loop
or learning rate. No vLLM server is involved in this test: the image also
provides the trainer's PyTorch, ROCm and communication libraries. The result
gives an evidence-backed reason to prefer the CSCS
stack for further controlled validation. It is **one observed paired failure**,
not reproduction of the original step-519 NaN, and does not yet identify
the failing kernel or prove a fix for full-model training. Python Muon code
is identical in both images. Native libraries, PyTorch revisions and the
transport configuration differ: original P2P/IPC versus ALPS SHM.

The test calls the installed private Muon Newton–Schulz function unchanged,
with the production shard-redistribution workaround and forced rocBLAS.
The tall shape matches each production MLP gate/up matrix and exercises
the transpose/column-shard path omitted by the earlier wide-matrix test.
The oracle additionally gathers the full sharded input and output for
comparison; production Muon does not necessarily materialize that full
update. The last visible large gather could be one of these oracle operations.
The failure is therefore in the **tall-Muon oracle workload**, not localized
to a Newton–Schulz kernel or a particular production operation.
The native failure is inferred to occur in this large zero-input case
from the saved successful cases and source order; it is not an assertion
reporting nonfinite output. Asynchronous errors prevent assigning the fault
to the last collective printed in the log.

Exact source, image hashes, environment, rank reports, logs and an independent
artifact verifier are retained in
[`muon-correctness/tall-only/results-710518/`](muon-correctness/tall-only/results-710518/).
The four large CPU/GPU core files remain intact remotely; their identities
are recorded without copying or hashing their full contents. Bounded fault
metadata is under `muon-correctness/tall-only/core-metadata/`.
Both saved GPU notes independently record an illegal-instruction exception
(`0x8`) on one AQL compute queue per process; other queue exception masks
are zero. Saved queue indices do not establish a unique active dispatch,
so pending dispatch packets must not be called the faulting kernel.
No language-model training has been restarted.

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
the old stack. They supplied a transport hypothesis for the tests below;
they do not establish the cause of the failed training run.
`alps-image-comparison/` preserves the image
identity, build hook, runtime environment and original comparison summary.

## Bounded correctness tests completed

`comm-correctness/` contains a non-training, two-node/20-minute test of three
cells on the same nodes: original image/default P2P policy, original image with
P2P disabled, and CSCS ALPS image with its bundled stack/default P2P workaround.
Both BF16 and FP32 fresh/reused tensors are checked against bounded analytical
values for gather, reduce-scatter, shard redistribution, packed metric reduction
and structured DTensor GEMM. Every output element is checked; inputs do not grow
through repeated reductions. Runtime logs must establish actual transport use.
Each cell has a hard timeout and there is no automatic retry. Job **710311**
passed all three configurations: 18,432 checks and 57,982,255,104 element
comparisons, including repeated buffers and values. This did not reproduce
silent corruption in those operations. Four ranks over a short interval do
not cover every 64-rank topology, overlap or lifetime-dependent fault.

| Test | Job | Original stack | CSCS ALPS stack |
| --- | --- | --- | --- |
| Known-value communication and structured GEMM, four ranks/two nodes | 710311 | Passed, both default P2P and P2P disabled | Passed |
| Causal sliding FlexAttention, padded/unpadded anchors, output and backward | 710467 | 8/8 cases passed | 8/8 cases passed |
| Exact Muon function, wide matrices through 8192 × 40960, two ranks | 710498 | 12/12 cases per rank passed | 12/12 cases per rank passed |
| Exact Muon function, tall matrices through 43008 × 8192, two ranks | 710518 | Native illegal-instruction abort in first large case | 12/12 cases per rank passed |

Attention job 710459 was a harness import error before any numerical case;
it is preserved separately and is not counted as a numerical failure. The
corrected attention test's worst small random-reference relative L2 error
was 0.305% (3% tolerance), with no nonfinite output or gradients. Its six
large analytical cases use production Q=4096/KV=12288, head dimension 128,
GQA 64:8 and causal sliding window 2048. Wide Muon's largest relative L2
errors were 0.216% (original) and 0.236% (ALPS); all outputs were finite.

Each test retains raw evidence and an executable verifier in its result
directory. These tests narrow specific hypotheses; they do not establish
held-out drafter quality or a healthy full epoch. In particular, the CSCS
pass and original native abort change multiple native-stack components
together and cannot isolate a particular library or transport defect.

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
  The completed communication test above checks the relevant operations on
  the actual hardware without reproducing that reported corruption.
