# Independent pinned-code audit, 31 October 2026

The historical first NaN remains unproven. A short paired test did expose a
concrete native failure in the original stack: both ranks aborted during a
tall-Muon oracle, while ALPS passed on the same node and GPUs. This is one
observed paired result, not proof of the historical cause or a specific
production optimizer fault. No model training was restarted.

## Demonstrated corruption cascade

A CPU test executed the exact image 2.13 `clip_grad.py` body using host torch
2.15 and AdamW. BF16 parameters were a 32×2 embedding and a 2×2 second tensor.
With one Inf gradient at embedding[7,0], clipping returned Inf, created one
NaN through Inf×0 and zeroed otherwise finite gradients. AdamW made exactly
that embedding element NaN. Tokens `[0,1,7,2,3]` then had one bad shared-feature
position and four finite positions. On the next update, one NaN gradient
made the common norm NaN; clipping and AdamW corrupted all 64 embedding and
all four other parameter values.

This demonstrates a possible bad-backward/update at 518 → mixed forward at
519 → complete collapse at 520. It does not establish that an Inf gradient
actually appeared at 518 or what generated it. Finite forward metrics at
518 do not clear that step's backward/update. Metrics are also all-reduced
and logged after the optimizer, so even the inference from NaN confidence
prediction to local NaN logits assumes a correct metric collective.

Pinned `train/trainer.py:528–547` performs forward, backward, unchecked norm
clipping and both optimizer steps. Image `clip_grad.py:164–181` applies the
common coefficient to every gradient. DSpark's sparse-use Markov embedding
feeds both vocabulary bias and confidence, allowing one bad row to affect
only some positions initially. Root's independent loss controls found finite
CE/TV/BCE math for zero masks and large finite inputs; target-only NaNs did
not reproduce the complete confidence/CE pattern. These controls do not
exonerate native collectives or large finite input corruption.

## NormPartial and Muon alias controls

A two-rank Gloo test ran the pinned 2.13 clipping body on host torch 2.15.
Direct DTensor norms had `_NormPartial(2)` with local values 5 and 12 and
materialized norm 13. The clip function returned Replicate with norm 13 on
both ranks and the expected clipped values. NaN and overflow controls
propagated as expected. The `_NormPartial` class AST matched, but norm strategy
registration differed between versions. This did not demonstrate a hidden
local-norm bug in the clipping path.

Both installed images use `grad.bfloat16()` in Muon. The known alias bug
requires the non-Nesterov branch; the configured true branch uses the
out-of-place `grad.lerp(buf, momentum)`. Reading only ZIP/pickle metadata from
the original optimizer established the actual saved setting: `data.pkl` was
14,449 bytes, `nesterov` appeared at offset 12,649, and NEWTRUE at 12,667.
No optimizer tensor storage was loaded. Root's CPU control reproduced the
false-branch alias issue and preserved momentum for the configured true
branch. That known bug is not a supported explanation for this run.

## Exact image/source comparison

| Item | Original | ALPS |
|---|---|---|
| torch Git revision | `733fca1901cfa5094481cfa9dfc8f8483e18f491` | `6596a0dea537c1c78520c73cc6678214781a9d06` |
| torch label | `2.13.0+git733fca1` | `2.13.0+rocm10.0.0` |
| ROCm / HIP | 7.2.3 / 7.2.53211 | 10.0.0 / 7.15.26333 |
| RCCL in tested processes | 2.27.7 | 2.30.4 |

Installed Muon, FlexAttention lowering/templates, clipping, norm-strategy,
pointwise-strategy, matrix-strategy, placement and redistribution files were
byte-identical between the images. Exact copies and hashes are retained in
the source audits. Native libraries and compilers differ. The common Flex
lowering defaults are `PRESCALE_QK=False` and `ROWS_GUARANTEED_SAFE=False`;
the forward template includes a zero-normalizer safety path. No source-level
NaN trigger was proven there.

The ALPS hook documents an RCCL P2P workaround on Beverin and disables P2P.
The original environment allowed P2P (`NCCL_P2P_LEVEL=NVL`, disable unset).
That documentation described crashes/hangs, not the historical silent NaN.
Earlier job 710006 showed a default BLAS performance difference; the failed
training already forced rocBLAS, so that speed difference is not a NaN fix.

## Completed bounded correctness checks

| Test | Verified outcome |
|---|---|
| Communication/GEMM, 710311 | Three configurations passed all 18,432 exact rank checks; no nonfinite values or mismatches |
| Attention harness, 710459 | Failed before any numerical case: dynamically loaded mask missing from `sys.modules` |
| Corrected attention, 710467 | Both images passed eight forward/backward cases each; no nonfinite values |
| Wide native Muon, 710498 | Both images passed 12 cases per rank; maximum raw-update L2 error 0.236% |
| Tall native Muon, 710518 | Original aborted on both ranks; ALPS passed all 12 cases per rank on the same GPUs |

Communication checks reset bounded BF16/FP32 rank/index/iteration inputs and
compare every output for all-gather, reduce-scatter, shard flip/round-trip,
packed metric all-reduce and structured DTensor GEMM. Original P2P/IPC and
P2P-disabled SHM were both actually exercised. There were 57,982,255,104
comparisons including repeated buffers/values, not independent fault trials.
Four ranks for seconds do not clear the original topology or lifetime.

The corrected attention oracle used the exact causal 2048-token sliding
window, Q4096/KV12288, GQA64:8, head dimension 128 and padded anchors. It
checked production-size zero/constant-value invariants and smaller random
FP32 math references. Maximum reference L2 error was 0.305%; maximum dV mass
error was 0.07185%. The two images used separate compiler caches. This did
not reproduce a deterministic padded-anchor fault, but does not recreate
the original fused model graph, input values or FSDP overlap.

The wide Muon test used shapes 16×80 and 8192×40960, cloned BF16 inputs,
forced rocBLAS and the production redistribution workaround. Rank-one
updates were bitwise equal; norm rounding differed by at most 0.637%, and
raw-update error was 0.216% original / 0.236% ALPS. Both shapes avoid the
initial transpose, leaving a real branch gap.

## Tall branch and native failure

The checkpoint header confirms gate/up weights 43008×8192 in all five
layers. A tiny CPU test and manual trace observed row shard → transpose to
column shard → Gram Partial(sum) → nonunit-beta Gram-update Partial(sum) →
column-sharded update. CPU results passed, including rank-one controls;
random final L2 error was 3.309%. CPU torch 2.15 is not the full image 2.13
runtime.

Job 710518 therefore tested 80×16 and 43008×8192 using the unchanged native
private-function probe. Original and ALPS ran sequentially on nid002488,
physical GPUs 2/3, PCI IDs `202000`/`302000`. The source revision was
`0030d3123e3bf7b8fa4f6c41e3615ab57b45203c`, with one attempt and no requeue.
Both complete image hashes were asserted. The private function hash was
`99e2359a95e1a914345207e49f072a658866333f4458b7b45faafa35c628f7c4`.

The original passed six small cases per rank, then aborted during the first
large zero-gradient case: both ranks received SIGABRT and ROCm reported
`HSA_STATUS_ERROR_ILLEGAL_INSTRUCTION`, code `0x2a`. No large-case numerical
comparison completed. The ALPS cell passed every case with finite outputs;
its maximum raw-update L2 error was 3.774%, below the explicit 5% BF16 bound.
The large zero case matched the dense reference exactly.

This observed abort does not require a learning rate or training instability:
there was no optimizer training, and the first large input was zero. However,
the oracle adds input reconstruction and final-update `full_tensor()` gathers,
which are not guaranteed in production Muon. Its last visible 672 MiB
collective does not identify the faulting operation. Original P2P/IPC versus
ALPS SHM is also confounded with native library/compiler changes. The local
shape with two ranks differs from the original 64-rank geometry.

## Bounded core conclusion

Four CPU/GPU cores remain remote. Only 355,356 bytes of headers, notes,
queue metadata and loader metadata were read, below the 4 MiB cap. Each GPU
note identifies exactly one COMPUTE_AQL queue with illegal-instruction mask
`0x8`; the other queues have no exception. Read/write pointers, retained or
pending dispatch metadata and code-object URIs were recovered, but there
is no uniquely established active dispatch, instruction PC or kernel symbol.
Provisional offset-16 entry addresses are unvalidated and explicitly rejected.
There were no wave/register reads, debugger core sessions or full-core scans.

The paired result establishes one native execution failure in the original
tall-Muon oracle workload and a same-GPU ALPS pass. It does not establish
the historical 519 trigger, a specific kernel, or ALPS training stability.
The original pre-failure inputs, RNG, parameter state and checkpoint were
not retained. No further job or experiment was launched, no external message
was sent, and one-node production and Clariden were untouched. Full evidence
and source/launch provenance are linked from the individual test directories.
