# Read-only pinned-source audit

The audit narrows candidates; it does not establish the first numerical trigger.
The exact nine source files, source revision, image path, extraction commands and
SHA256 hashes are retained here. The image digest was independently checked in
both actual-image preflights.

1. Generated hidden states are fully checked for NaN/Inf in
   speculators/train/data.py:255-261 and data_generation/offline.py:28-53.
   The cached branch at train/data.py:291,319-320 bypasses that check. The run
   uses on-generate=delete. Collation allocates zero-filled buffers before
   copying/casting at train/data.py:419-433. Existing input validation will
   now detect any nonfinite value reaching the trainer through either route.
2. Every query has an allowed key in the intended attention predicate:
   kv_idx=total_seq_len+q_idx passes own-block membership and the causal
   <= condition. See the original pinned models/dflash/attention.py:85-98.
   A fully masked row is not explained by that predicate. This does not prove
   that the compiled mask builder or ROCm FlexAttention kernels are correct.
   The adapter in speculators/models/attention.py:47-63 uses contiguous Q/K/V,
   GQA and the default FlexAttention kernel options.
3. First-failure confidence_pred_mean=NaN is not explained by bad target logits
   alone: dspark/metrics.py forms it from sigmoid(confidence_logits.float())
   times a finite Boolean loss mask. A NaN target acceptance can spoil BCE,
   but does not feed back into this prediction statistic during forward.
   The backbone hidden vectors and Markov previous-token embedding feed both
   confidence and draft logits. The sparse-use Markov embedding is the only
   learned shared path that could localize a bad parameter row to rare tokens.
   This is a candidate, not evidence that AdamW created a bad row.
4. The image Muon implementation casts the update to BF16, norm-clamps at1e-7,
   then applies five Newton-Schulz iterations (image/torch/optim/_muon.py:55-66).
   Its step uses momentum and a separate shape-adjusted learning rate
   (:333-341). DTensor _NormPartial implements norm-squared/all-reduce/sqrt
   in image/torch/distributed/tensor/_ops/_math_ops.py:67-116, with all matrix
   dimensions reduced at:600-622. No obvious zero-gradient division or local
   shard-only normalization defect was found by source inspection.
5. The image gradient clip computes a common coefficient and multiplies every
   gradient by it (image/torch/nn/utils/clip_grad.py:164-181). A NaN total norm
   therefore poisons previously finite gradients. The original trainer did
   not reject that norm. This explains propagation, not its initial cause.

The bounded diagnostic preserves the original world size, batches, scheduler
and LR. Per-rank input and stage checks separate bad inputs from FC/norm,
decoder-layer, Markov and confidence outputs; gradient checks separate backward
from forward failure; separate optimizer post-hooks identify the first optimizer
producing bad parameters. Capture every rank because distributed gradient
reduction can make the detecting rank differ from the originating rank.

If it fails, the smallest follow-up is to replay the captured failing batch with
the retained last good checkpoint and add hooks inside the identified stage,
then compare only that stage's eager/FP32 and original ROCm implementation.
Changing LR simultaneously would confound that comparison. The periodic
checkpoint precedes the exact failing update, so such replay may require the
intervening batches; it must not be described as exact without restoring those
updates and RNG. If no failure occurs by640, report that bound without declaring
stability or blindly retrying the recipe.
