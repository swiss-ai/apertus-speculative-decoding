# Independent pinned-code audit, 31 October 2026

The original initiating fault is **not yet established**. The experiments
below demonstrate or exclude specific mechanisms; they do not turn a plausible
path into a proven cause. No further training run was launched. The separately
authorized communication/GEMM probe loads no model and has no optimizer.

## A concrete two-update corruption cascade

An inline CPU test used the exact image2.13 `torch/nn/utils/clip_grad.py` source
retained in `diagnostic/v2/source-audit/image/`, with host torch2.15.0+cpu
and AdamW. It used BF16 parameters: a32x2 embedding and a2x2 second parameter.
With all gradients finite except one `Inf` at embedding[7,0]:

1. The pinned clip returned `Inf`, leaving exactly one gradient NaN
   (`Inf*0`) and zeroing the otherwise finite gradients.
2. AdamW updated exactly that one embedding element to NaN; all four values
   in the other parameter stayed finite. A token sequence[0,1,7,2,3] then
   had one bad shared-feature position and four finite positions.
3. On the next update, a single NaN gradient made the common norm NaN.
   Clipping and AdamW then made all64 embedding elements and all4 other
   elements nonfinite.

This demonstrates the518 bad-backward/update →519 mixed forward →520 total
collapse mechanism. A finite forward metric at518 does **not** prove that
backward or its optimizer update remained finite. It does not establish
whether518 actually had an Inf gradient, nor what generated one. The first
nonfinite could instead have appeared in519 forward or a collective.

Source locations: pinned `train/trainer.py:528-547` performs forward,
backward, unchecked norm clipping and both optimizer steps; image
`clip_grad.py:164-181` computes the shared coefficient and multiplies every
gradient. DSpark `model_definitions.py:35-46` defines the sparse-use Markov
embedding; `dspark/core.py` feeds it to both vocabulary bias and confidence.
An affected embedding row can localize the first damage to a few token
positions while most greedy accuracy remains normal.

The recorded metrics are themselves all-reduced and logged after the update.
A communication fault could corrupt the metric vector as well as gradients
or parameters. Therefore even the inference from NaN confidence prediction
to local NaN confidence logits requires a correct metric all-reduce. Root's
pure loss controls test formulas, not communication correctness.

## Earlier NormPartial concern is not demonstrated for clipping

A four-value, two-rank Gloo CPU test executed the exact image2.13 clipping
module on host torch2.15.0+cpu. Direct DTensor.norm() had `_NormPartial(2)`:
local norms5 and12, materialized global norm13. The clip function's returned
norm was **Replicate**, with local/global13 on both ranks; it clipped to the
expected3/13,4/13,0,12/13. A NaN gradient on rank1 yielded replicated NaN and
all post-clip gradients NaN. A large finite-gradient overflow yielded Inf and
zero gradients. Thus `.to_local()` of the returned norm did not hide the
global value in this tested path. The `_NormPartial` class AST is identical
between the pinned2.13 image and host2.15. Their norm strategy registration
has changed, so this is not an execution of the complete2.13 dispatch stack.

Pinned2.13 `_math_ops.py:279-291` also forces partial input to replicated
placements before a nonlinear norm reduction when reduction operators
mismatch. The norm-of-stacked-individual-norms in `clip_grad.py:105-107` is
therefore distinct from checking a direct unresolved partial norm. Do not
present the earlier guard.to_local concern as the original failure cause.

## Muon alias fix does not match the saved production setting

Both installed images use `grad.bfloat16()` rather than the newer copy=True
variant. But pinned `_muon.py:333-336` uses out-of-place `grad.lerp(buf,momentum)`
when Nesterov is true. Reading only the ZIP metadata of the original saved
13GB optimizer file establishes the actual setting without loading tensors:

- member `optimizer_state_dict/data.pkl`:14,449bytes;
- `nesterov` key at pickle offset12649;
- opcode `NEWTRUE` at12667.

This agrees with the builder/default. Root's separate CPU alias test confirms
buffer aliasing for nesterov=False and preserved momentum for the configured
True case. That test uses host CPU kernels, but the known false-case alias
bug is not a supported explanation for this run.

## Exact original versus ALPS image evidence

Read-only `unsquashfs -cat` of installed Python source gave:

| Item | Original | ALPS |
|---|---|---|
| torch git |733fca1901cfa5094481cfa9dfc8f8483e18f491|6596a0dea537c1c78520c73cc6678214781a9d06|
| torch label |2.13.0+git733fca1|2.13.0+rocm10.0.0|
| ROCm /HIP |7.2.3 /7.2.53211|10.0.0 /7.15.26333|
| Muon source SHA256 |4d003aba2d0c7fcc24875845802e45edb4475a51a037e2caaa7b46c3944c0dae|identical|
| flex/flex_attention.py SHA256 |0e849dfeb3488b6838323285c7d23ef9c9bb2191ac5ce94893696bd7b117db21|identical|

The common lowering defaults `PRESCALE_QK=False`,
`ROWS_GUARANTEED_SAFE=False` at lines754-757. This excludes a claim that ALPS
changed this Python lowering or the Muon algorithm. It does not establish
identity of every Triton template or native compiler/library. Existing job
710006 establishes a default-backend speed improvement, not a NaN fix;
failed production already forced rocBLAS.

The ALPS source archive at source revisionbf992658020317ca9c4564aa6593ddad7dc42240
contains `10-rccl-p2p-workaround.sh`. Its comment documents failures with
bundled RCCL2.30.4 under enroot/Beverin kernel6.4, worked around by
NCCL_P2P_DISABLE=1. The documented outcomes are invalid-device-pointer
failures/hangs, not a demonstration of silent NaNs on this workload.
The original run's saved node0 environment has `NCCL_P2P_LEVEL=NVL` and
no `NCCL_P2P_DISABLE`; original logs identify RCCL2.27.7-HEAD:96a25b5 and
HIP7.2.53211. Thus the ALPS workaround is a credible communication lead,
but is not direct evidence that the older injected RCCL silently corrupted
this run. P2P allowed is not proof which transport a channel selected.

## Bounded discriminating correctness test

`comm-correctness/` contains the reviewed non-training probe. It resets
rank/iteration/index-encoded buffers and checks **every output value** for
all-gather, reduce-scatter, production functional shard flip and reverse,
a packed FP32 metric-vector all-reduce, and structured DTensor GEMM. Values
are bounded<=63 and sums<=156, exactly representable BF16. GEMM has +/-1
inputs and analytical +/-K outputs. No cumulative reduction can overflow.
Local mismatch flags remain authoritative even if the diagnostic flag
all-reduce is itself corrupt. First bad rank/index/expected/observed and
finite/nonfinite counts are retained; cell timeout is failure, never success.

CPU/Gloo4-rank tiny-shape preflight passed192 checks with exact references;
the source hash in cpu-preflight-summary.json matches the committed probe.
The approved GPU allocation is limited to two nodes/20minutes with three
240-second cells: original default P2P, original P2P disabled, ALPS bundled
default. All use GPUs2,3 and rocBLAS; both complete image hashes are checked
before cells. Passing cannot rule out64-rank, larger-buffer or longer-lived
faults, but a mismatch would directly identify incorrect data independently
of DSpark loss and LR. This is a correctness discriminator, not another
attempt to train through the failure.

## Additional exact template comparisons

Both images also have byte-identical installed sources:

- flex/templates/flex_attention.py.jinja:
  ec886f040afa1d8b9a12281d443851382899f94eb5624daa2e657821b6249bed;
- flex/templates/flex_backwards.py.jinja:
  c40d87e7c6334cbb73aba4d388bef1195619c648ef438718875440e8c54f10b6;
- torch/nn/utils/clip_grad.py:
  5e30dac17eb0311ab1a12e46ed2484decb66165427644d2601d13570ab0ee4d7;
- torch/distributed/tensor/_ops/_math_ops.py:
  3f6ad29c5164e418ee6c6f7f5ca7eab6d82214abc7697dc3edf373d496a3eb2d.

The forward template initializes m_i=-Inf/l_i=0 at112-114 and substitutes
l_i=1 for a zero normalizer at193-198. This is an explicit fully-masked-row
safety path; it does not validate every internal Triton branch or backward
write. Production Q/KV lengths4096/12288 are multiples128 and score_mod=None,
so the externally identified nonmultiple-length + score-modifier failure
configuration does not directly match. A padded-anchor forward/backward
oracle can test the actual configuration without loading the drafter.

## Verified communication/GEMM outcome

Job **710311** completed all three configurations in 229 seconds on
nid002484/nid002486. All 12 rank reports have exactly 1,536 successful
checks, and every check has zero nonfinite values and zero mismatches:
18,432 total checks / 57,982,255,104 element comparisons, including repeated
buffers/values. Original default actually used P2P/IPC (144 channel log
lines); original P2P-disabled and ALPS default used SHM. Both complete image
hashes were checked before running. RCCL versions were 2.27.7 and 2.30.4.
The raw 59-file collection, exact comparison counts and independently
executable verifier are in comm-correctness/results-710311/.

This narrows the evidence: no deterministic error was found in these small
fresh/reused BF16/FP32 collectives, packed metrics reductions, shard
redistribution, or structured rocBLAS GEMM. Four ranks for seconds do not
rule out rare or 64-rank/large-model/overlap-dependent corruption. The test
does not justify claiming that P2P or ALPS is the cause/fix of the old NaN.

## Attention oracle harness outcome

Job **710459** ran one node for 199 seconds, two images sequentially.
Both exited before any attention case because Dynamo could not import the
dynamically loaded module `captured_mask`. The harness had not registered
it in sys.modules. This is setup failure, not nonfinite output, a gradient
reference mismatch, or evidence that the production attention kernel is
unsupported. All 27 raw files, image checks, both tracebacks and zero-case
result records are preserved in flex-correctness/results-710459/.
No automatic retry was submitted.

## Remaining causal limits

The first bad training value is unobserved: finite logged forward metrics
at 518 do not clear backward/update 518. Metrics are logged after the
optimizer and metric all-reduce, so local feature corruption cannot be
inferred independently of communication correctness. Even finite extraction
checks would not exclude a huge finite input (for example ~1e38) that
overflows the FC dot product, then produces Inf*0 in normalization. The
original input ranges, batch, RNG and pre-failure checkpoint were not retained.
Neither the demonstrated clipping cascade nor passing short tests establishes
the initiating fault. No further training was launched.

## Verified attention oracle outcome

After the import-registration fix passed default-Inductor CPU mask preflight,
fresh job **710467** ran on nid002410 for 271 seconds. Both original and
ALPS images passed all eight forward/backward cases (16 total), with zero
nonfinite output/dQ/dK/dV values. The six production-size causal/sliding
padded/unpadded fixtures had zero output error, maximum dV mass relative
error 0.00071847. The two smaller random-QKV math-reference comparisons
had maximum relative L2 error 0.00304610 (0.305%, tolerance3%). Both complete
image hashes were checked; separate per-image compiler caches were used.
All 27 raw files/40,037 bytes are retained and verified in
flex-correctness/results-710467/. No further job was submitted.

This did not reproduce a deterministic padded-anchor/masked-tile fault in
either image. It does not validate the exact fused model graph, original
input values, FSDP overlap or optimizer; original first-trigger remains
unproven. An untested specific math path is image2.13 DTensor Muon: BF16
global norm/clamp, in-place normalization and five Newton-Schulz
GEMM/addmm iterations. Our CPU2.15 dispatch and GPU linear-GEMM checks do
not execute that full native2.13 path. This is a coverage limit, not evidence
that Muon caused the original NaN. No additional test has been submitted.

## Native wide Muon and tall-branch distinction

Job710498 completed225seconds: bothimages2ranks×12widecases passed.
Global shapes16x80 and8192x40960,6fixtures including two rank-one controls.
AllrawNSoutputsfinite, rankoneoutputsbitwiseequal; maximumrelativeL2error
0.216%original/0.236%ALPS. BF16normroundingdiffersfromdense byatmost0.637%,
within2%bound; allindependentrow-shardclampsyieldReplicate. Exactprivate
MuonfunctionSHA99e2359a95e1a914345207e49f072a658866333f4458b7b45faafa35c628f7c4
matchedbothimages.43rawfiles/719,865bytes retainedinMuonresults-710498.

The native test did NOT execute M>N transpose: captured gate/up matrices
are43008x8192. CPUtiny80x16 exactfunction plusproductionredistribution
passedall6fixtures/2ranks; randomupdateL2error3.309%, rankone0. Separate
CPUmanualtracing observesShard0→transposeShard1→GramPartial(sum)→
nonunit-betaGramUpdatePartial(sum)→orthoShard1. This is distinct native
coverage. source-audit shows bothimagesbyteidentical _matrix_ops.py,
placement_types.py,_redistribute.py; host2.15versionsdiffer. No causal
source mismatch is proven. A separately approved tall-only native job710518
usesfresh0030d312runtime andexplicit80x16/43008x8192; its results are not
inferred from these wide/CPU passes. No model training was launched.
