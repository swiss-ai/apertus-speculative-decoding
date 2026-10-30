# Beverin non-finite training incident, 30 October 2026

The 32-node run `opb-thinkoff-32n-1epoch-beverin-20261027` is **failed**.
Reaching the step counter 4,077 did not produce a usable trained checkpoint.
Only job **698073** was cancelled at **2026-10-30 10:35:46 UTC**, ending its
repeated validation of the same corrupted weights. The separate one-node
production and Clariden jobs were not changed. Original run artifacts and
runtime snapshots are preserved.

## Established evidence

- Last finite logged counter: **global_step=518**, loss 0.559, CE 1.206,
  TV 0.234, greedy accuracy 0.489, analytical accepted length 3.040.
- First non-finite logged counter: **519**. CE, TV, confidence loss,
  confidence predicted mean and analytical acceptance are all NaN; greedy
  accuracy is still 0.484. At counter 520, greedy accuracy becomes zero.
  These counters are logged before the trainer increments `global_step`.
- Both Muon and AdamW LRs near the onset were **7.48e-4**, following the
  configured peak **8.48528137423857e-4**, 40-step warmup and 4,077-step
  linear schedule. There is no evidence yet establishing LR as the trigger.
- An independent, deduplicated probe of **11,073 BF16 values across all 62
  saved tensors found every sampled value non-finite**. It reads up to 64
  values at each tensor's start, middle and end; 128-element norm tensors
  have overlapping regions. This is a sample, not a full weight scan.
- The sole epoch checkpoint directory is `checkpoints/0`; mid-epoch saves
  overwrite it. There is no retained step-512 checkpoint in the failed run.
- Validation repeatedly returns non-finite metrics and the final-validation
  guard correctly rejects them. No authenticated completion marker exists.

The reduced logs cannot identify the first bad rank, tensor or operation.
Near-normal greedy accuracy at the onset is consistent with a localized
failure before the distributed backward/update spreads it. It does not
prove which of those operations failed first.

## Confirmed reliability defects

Pinned `trainer.py:528–547` neither rejects a non-finite forward loss nor
checks the gradient norm before updating parameters. It calls
`clip_grad_norm_(..., 1.0)` with the default non-finite error check disabled.
A non-finite gradient norm can contaminate otherwise finite gradients.
Training continued and later saves overwrote earlier weights.

The launcher's retry loop treats arbitrary trainer failures as recoverable.
The continuation consequently reran validation of a numerically invalid
checkpoint up to attempt 10. Identical validation cannot repair weights.

Earlier monitoring checked job state, counters and exception strings, but
did not check training loss health. Reporting counter completion as healthy
training was incorrect.

## Narrowing the initial numerical trigger

Read-only inspection of the pinned Speculators code found:

- Generated hidden states are checked with `torch.isfinite` before normal
  collation (`train/data.py:255–261`, `data_generation/offline.py:28–53`).
  Collation pads with zeros (`train/data.py:419–433`). `error_records=0`
  at the onset makes a normally detected extraction failure less likely.
  The cached-state path does not use that same finite check.
- The attention mask permits every query to see its own synthetic block
  (`models/dflash/attention.py`), so a wholly masked row is not apparent
  from the mask's intended logic.
- The image's Muon implementation casts its update to BF16, clamps its
  norm at 1e-7 and performs five Newton–Schulz iterations. Inspection alone
  did not establish a Muon zero-norm defect.
- A localized draft activation/kernel failure or a sparsely used Markov
  embedding row could fit the onset. The Markov embedding affects both
  vocabulary logits and confidence features. These remain hypotheses.
- All 32 original trainer logs and extraction logs were checked for an
  onset warning. None identifies the offending computation around the
  first NaN. A more precise claim requires instrumented evidence.

## Changes and diagnostic plan

Branch: `fix/beverin-nonfinite-diagnosis-20261030`, isolated worktree
`/home/projects/apertus-speculative-decoding-beverin-nan`.
Runtime commits: `a81d1626`, `e6c84030`, then `556a464d` for durable all-rank
capture before worker termination. Existing runtimes remain immutable.

The Beverin-only guard now checks loaded trainable weights, input tensors,
forward loss/metrics, gradients before clipping, the returned gradient norm
and weights after updates. All ranks agree on a failure before proceeding.
A terminal JSON report prohibits automatic retry and captures every rank's
batch and RNG state before forward. This matters when FSDP has spread a bad
gradient before detection; peers synchronize after capture so torchrun cannot
kill a rank mid-dump. Existing checkpoints remain
untouched on the failure path. Diagnostic mode additionally checks backbone
layers, norms, vocabulary heads, Markov embeddings/projection, confidence
head and the result of each optimizer separately.

Nine CPU guard tests passed with torch 2.15.0+cpu, including one bad rank in
a two-rank Gloo group, invalid backward despite finite forward loss, and
overflow of the gradient norm with finite gradients. Nine existing epoch
lifecycle checks, eight monitor checks, shell syntax and diff checks passed.
These CPU tests do not establish compatibility with the ROCm image; the
two-rank, one-node FSDP preflight checks that separately.

Actual-image preflight **707137** passed all three cases. The corrected
capture/barrier preflight **707139** also completed successfully: finite
update, one-rank invalid input, and invalid backward gradient. Both rank
capture files for each failure were loaded and checked for original batch
and RNG content. It ran on `nid002778` for 97 seconds with torch
`2.13.0+git733fca1`, HIP `7.2.53211`; the compute-side image hash matched
`b58797bb6a430d7587f1c10ec24ac71f198bd8d590bde98dbbe68d7744a9816f`.
Raw preflight artifacts and checksums are under `preflight/` and
`preflight/v2/`.

Preliminary diagnostic **707138** was cancelled while pending, with zero
allocation runtime, to use the capture fix. The replacement **707140** was
submitted **2026-10-30 10:54:54 UTC** and is **pending priority** at the
11:59 UTC check, with no allocation or training steps. Slurm’s tentative
start moved from 13:30 UTC to **21:45 UTC**; this can change. The foreground
observer remains active through queue delays and the eventual job end.

The fresh run is `opb-thinkoff-32n-nan-debug-20261030-v2`, using immutable
`repo-556a464d/` (full runtime and launcher commit
`556a464dfbfe7ba56699ad037bf952b15c4e6b4e`). It requests 32 nodes / 64 trainer
ranks for six hours, one allocation and **MAX_ATTEMPTS=1**. The original
recipe, LR **8.48528137423857e-4**, SEQ 8,192, 512 anchors, effective batch
524,288, true one-epoch 4,077-step linear scheduler and 40 warmup steps are
preserved. `BEVERIN_SEGMENT_STEPS=640` bounds the diagnostic while retaining
that schedule. There is no MAX_STEPS override, validation replay or reuse
of the invalid checkpoint. Per-rank activation checks and synchronized
all-rank bad-batch/RNG capture are enabled. Last good periodic checkpoints
remain intact when numerical failure stops the run.

Launch provenance and exact generated Slurm script are in `diagnostic/v2/`;
remote output is
`/iopsstor/scratch/cscs/faruk_zahiragic/beverin-feasibility/runs/opb-thinkoff-32n-nan-debug-20261030-v2/`.
Keeping the original settings first allows a failure to be attributed to a
stage before changing LR or math kernels. If it remains finite through the
bound, that is not proof the original problem has been solved. The initial
numerical trigger remains under investigation.

## Evidence

`first-nan-training-excerpt.txt`, `checkpoint-sample.json`,
`validation-attempt9-tail.txt` and `job-state-after-cancel.txt` are bounded
observations from the original run. `pinned-source/` preserves the exact
relevant remote Speculators files; `checkpoint-metadata/` preserves config,
state and the latest command record (the trainer rewrites that command
record on restart). Original launch provenance remains in the original
campaign directory. `SHA256SUMS` covers the collected original evidence.
