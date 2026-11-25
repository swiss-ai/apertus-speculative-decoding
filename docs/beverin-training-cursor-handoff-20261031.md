# Cursor handoff: restart DSpark 70B training on Beverin with the CSCS image

Prepared 31 October 2026. This is the current execution handoff. It supersedes
the older launch/retry instructions in `beverin-handoff.md` where they conflict.
**No training was launched while preparing this document.**

## Objective and scope

Produce a **healthy one-epoch DSpark drafter for Apertus-v1.5-70B on 32 Beverin
nodes**, including finite held-out validation and reproducible artifacts.
Use the digest-pinned CSCS ALPS image, retain the numerical guards, and start a
new campaign from fresh initialization. First pass the small integration checks
below, then use the first 640 updates of the new campaign as a bounded stage.
Continue those same weights to the end of the epoch after that stage passes;
do not throw away the 640 updates and train them again.

The longer-term discussion was ten epochs. **The immediate target is one good
epoch and analysis of its results, not an automatic ten-epoch launch.** The
user specifically objected to repeatedly training or retrying validation without
understanding failures. Do not describe the CSCS image as a proven cure for the
historical NaNs, and do not silently change LR, batch, data or optimizer to make
a test pass. Successful bounded gates can lead directly to the next stage;
they do not require another generic permission question.

**Protected work:** leave the one-node DSpark campaign, other users/campaigns,
all Clariden jobs, and their checkouts/checkpoints unchanged. Do not cancel,
extend, repoint or reuse them. At the read-only Beverin check at
**2026-10-31 17:19:25 UTC**, job **694478** was RUNNING for
`opb-thinkoff-1n-beverin` on `nid002504`. The unrelated EAGLE campaign had
**707003** RUNNING and **707004–707011** pending dependencies. These are a
timestamped observation, not a statement about their state when you read this.

## What happened and what is actually known

| Run/test | Established result | Consequence |
| --- | --- | --- |
| Original 32-node epoch, `opb-thinkoff-32n-1epoch-beverin-20261027`, jobs 698072/698073 | First logged NaN at counter 519; counter 520 accuracy collapses. Counter reached 4077, but every sampled value across all 62 saved tensors was nonfinite. 698072 timed out; 698073 was deliberately cancelled to stop repeated validation. | **Never resume or validate this checkpoint again as a recovery strategy.** |
| Instrumented 32-node diagnostic, `opb-thinkoff-32n-nan-debug-20261030-v2`, job 707140 | One attempt, 640 finite updates, saves successful, all 12,800 logged training metric values finite. Only a bounded saved-weight sample was checked. No held-out validation or resume was performed. | Demonstrates a bounded healthy trajectory and functioning checks, not the original cause or a full-epoch fix. Do not use its old-image weights as the new campaign's initialization. |
| Communication/GEMM correctness, 710311 | All three cells passed: original/default P2P, original/P2P disabled, ALPS/default SHM. Four ranks/two nodes; fresh/reused BF16/FP32 buffers, known-value collectives and GEMM. | No corruption reproduced under that coverage; does not clear every large-buffer or 64-rank case. |
| FlexAttention correctness, 710467 | Both images passed 8/8 forward/backward cases, including production-size padded/unpadded causal sliding attention. | No deterministic attention fault found in these cases. Earlier 710459 was a harness import failure before numerical cases, not a kernel result. |
| Wide Muon oracle, 710498 | Both images passed all 12 cases per rank through global shape 8192 × 40960. | Does not exercise the initial transpose for tall MLP matrices. |
| Tall Muon oracle, 710518 | Original image aborted both ranks with `HSA_STATUS_ERROR_ILLEGAL_INSTRUCTION`, code `0x2a`, in the first large zero-input case, shape 43008 × 8192. ALPS passed all 12 cases per rank on the **same physical GPUs 2/3**. | A concrete native-runtime failure in one paired test supports using the CSCS stack as the next candidate. It does not prove historical causation. |

Important limits of 710518: the oracle also performs full input/output gathers
for comparison, which production Muon does not necessarily perform. The native
failure is **not localized to Newton–Schulz, rocBLAS, RCCL, or a particular
production operation**. Native libraries and P2P policy changed together. Saved
GPU core notes confirm an illegal-instruction queue exception, but no unique
faulting dispatch, instruction PC or kernel symbol was established. The test
used two trainer ranks, not 64. No language model, corpus, LR or vLLM server was
involved; the vLLM image also supplies the trainer's PyTorch/ROCm libraries.

The historical trigger cannot be reconstructed uniquely: the original bad
batch/RNG/gradient state and a checkpoint just before onset were not retained.
Finite logged forward metrics at 518 do not clear backward/update 518, because
logging occurs after the optimizer and metric collective. The prior code also
failed to reject nonfinite gradients/norms, allowing corruption to spread and
later saves to overwrite good weights. Retrying those weights could not repair
them. New guards address that propagation/retry defect; they do not establish
what created the first bad value.

Two source-level suspects were checked and are not demonstrated causes here:
finite-input loss arithmetic passed the controls; the older Muon BF16 workspace
alias affects `nesterov=False`, while this run uses **Nesterov=True**. Do not
claim either as the fix. Guard hooks change compilation/synchronization and
the healthy 640-step trajectory already differs from the failed run at counter 2.

## Source branches, worktrees and evidence

The handoff branch is **`docs/beverin-training-handoff-20261031`**, based on
diagnosis commit **`3cd76413`**. It contains the diagnosis code and evidence.
Use a **new training branch and isolated worktree** based on this handoff branch
before editing or launching. Check status, other worktrees and submitted jobs
first. Never switch another session's checkout or overwrite an immutable runtime.

On the current machine:

- Handoff: `/home/projects/apertus-speculative-decoding-beverin-training-handoff`.
- Diagnosis: `/home/projects/apertus-speculative-decoding-beverin-nan`, branch
  `fix/beverin-nonfinite-diagnosis-20261030`.
- Historical training: `/home/projects/apertus-speculative-decoding-beverin`,
  branch `dspark-70b-beverin`.
- Separate image comparison: `/home/projects/apertus-speculative-decoding-beverin-alps`,
  branch `experiment/beverin-alps-vllm-dtensor-20261031`, commit `ec83303a`.
  Inspect this checkout read-only; its ALPS comparison scripts are examples,
  not a completed ALPS production-training launcher.

Read these for supporting evidence rather than repeating completed experiments:

- [Investigation overview](../results/70b/dspark/beverin-nonfinite-20261030/root-cause-20261031/README.md).
- [Tall oracle, exact image/source identities, raw logs and retained core paths](../results/70b/dspark/beverin-nonfinite-20261030/root-cause-20261031/muon-correctness/tall-only/README.md).
- [Verified 640-step diagnostic](../results/70b/dspark/beverin-nonfinite-20261030/diagnostic/v2/completed-707140/README.md).
- [Original one-epoch launch configuration](../results/70b/dspark/beverin-32n-one-epoch-20261027/launch.sh).
- [ALPS source/runtime comparison](../results/70b/dspark/beverin-nonfinite-20261030/root-cause-20261031/alps-image-comparison/).
- [Historical feasibility/scaling record](beverin-feasibility.md), especially
  the 25 October rocBLAS fix and 27 October scaling results. Its early status
  statements, pending jobs, retry advice and image defaults are historical.
- [CSCS inquiry draft](beverin-support-ticket.md). It concerns the earlier
  performance investigation and was not sent by this agent. Do not send it
  or the new evidence externally without the user's instruction.

## Access and existing cluster assets

Use **Beverin explicitly**:

```bash
ssh -i ~/.ssh/cscs-key -l faruk_zahiragic -J ela beverin.cscs.ch
```

Do not use a helper that defaults to Clariden. CSCS certificates expire; an
actual authentication failure requires renewal, not an SSH retry loop or changed
keys. A transport timeout alone does not establish a job failure. Login-node
Python is 3.6; container Python is 3.12. Use stdlib-compatible lightweight control
scripts on the login node. GPU work, full checkpoint scans/hashes, image imports
and substantial CPU work belong in Slurm allocations, not on the login node.

| Asset | Path/pin |
| --- | --- |
| Slurm | account `infra01`, partition `mi300`, 4 MI300A GPUs per node, 24 h maximum per allocation |
| Work root (`WORK`) | `/iopsstor/scratch/cscs/faruk_zahiragic/beverin-feasibility` |
| Target | `/capstor/store/cscs/swissai/infra01/hf_models/models/swiss-ai/Apertus-v1.5-70B` |
| Prepared data | `/iopsstor/scratch/cscs/faruk_zahiragic/apertus-dspark70b/data` |
| Dataset provenance | `${WORK}/provenance/data-20261025` |
| Speculators | `${WORK}/speculators-7046ac6-provenance`, base Git SHA `7046ac63fac4eeb218dc2e92a385c7d1ec8c4e00`, plus the recorded patches |
| Existing side dependencies | `${WORK}/py312/pydeps` (verify compatibility before reuse) |
| Hidden-state smoke rows | `/iopsstor/scratch/cscs/faruk_zahiragic/apertus-dspark70b/pilot/tp4-seqs512.jsonl` |

The corpus is open-perfectblend revision
`af60f3c18201652a83a93f46fcfee1b646ba3df7`, with answers regenerated by the
70B target, thinking off. Historical prepared corpus: 1,751,728 rows and
175,173 held-out rows, using the preserved conversation split/validation IDs.
Reuse that prepared dataset and provenance; do not regenerate, reshuffle the
held-out split or substitute the 8B-generated responses.

Use a new unique `RUN_ID`, for example
`opb-thinkoff-32n-alps-1epoch-<UTC-date-and-time>`, and new output directory.
Keep the Speculators checkout read-only. If changes there become necessary,
make an isolated copy with its Git identity and patch preserved; running jobs
may use the existing copy. Package the committed training runtime into a new
`${WORK}/repo-<new-sha-prefix>/` with `REVISION`, `SOURCE_BRANCH` and a source
manifest. Do not edit `repo-ada15200`, `repo-dd96fe2a`, `repo-556a464d` or any
other existing runtime. Transfer committed code through an archive/bundle to
the new directory and check the remote revision before submission. Do not
rsync with `--delete` over existing assets.

## Pin and integrate the CSCS ALPS image

Use the existing image file; do not select a moving tag or reinstall its stack:

```text
OCI: ghcr.io/eth-cscs/alps-extended-images/vllm-rocm@sha256:7d95232cbc7511be5672d272f7a567ea52a3c5fbd5310ce0903956f0f37b5cc7
SquashFS: /capstor/scratch/cscs/faruk_zahiragic/container-images/alps-vllm-rocm-7d95232cbc75.sqsh
SquashFS SHA256: 416216c9de21d110fbb7cb12cef90923af51e04c955d7d7dffa9786f125a4928
```

Observed stack: torch `2.13.0+rocm10.0.0`, torch Git
`6596a0dea537c1c78520c73cc6678214781a9d06`, HIP `7.15.26333`, vLLM `0.31.0`,
Triton `3.8.0+git4cff872c.rocm10.0.0`, actual RCCL **2.30.4**. The image's
`RCCL_VERSION=10.0.0` environment label is not the library's actual API version.
Record the runtime versions and loaded library paths, not just that label.

**Integration work remains.** `methods/dspark/beverin/submit-train.sh` still
defaults to the old image and omits the ALPS network option. Merely changing
`EDF` is insufficient. Implement and commit the following in your new branch
before using the configuration below:

1. Add a separate ALPS EDF, suggested path
   `methods/dspark/beverin/env-vllm-alps.toml`, with this content:

   ```toml
   image = "/capstor/scratch/cscs/faruk_zahiragic/container-images/alps-vllm-rocm-7d95232cbc75.sqsh"
   mounts = ["/capstor", "/iopsstor", "/users"]
   workdir = "/iopsstor/scratch/cscs/faruk_zahiragic/beverin-feasibility"
   entrypoint = false

   [env]
   PYTHONNOUSERSITE = "1"
   HWLOC_COMPONENTS = "-gl"

   [annotations]
   "com.hooks.netstack.source" = "host"
   "com.hooks.cxi.enabled" = "false"
   "com.hooks.aws_ofi_nccl.enabled" = "false"
   ```

2. Add **`--network=disable_rdzv_get` to the container `srun`** for the ALPS
   campaign, retaining one task/four GPUs per node and `--environment`.
   Prefer a separate ALPS submitter or an explicit opt-in; preserve historical
   launch behavior. Do not combine the old external OFI/RCCL injection hooks
   with the image's bundled stack. The successful two-node ALPS communication
   cell is a reference for this exact launch boundary.

3. At the Beverin container entry, before the training/extraction children,
   unset **all three** competing masks:
   `ROCR_VISIBLE_DEVICES`, `HIP_VISIBLE_DEVICES`, `GPU_DEVICE_ORDINAL`.
   The existing `env.sh` only clears the first. Select each child with
   `CUDA_VISIBLE_DEVICES=0,1` for extraction or `2,3` for training; do not
   globally pin the whole container to the training pair.

4. Preserve normal Bash startup and the image's
   `BASH_ENV=/opt/alps/env/source-alps-env.sh`. Verify
   `ALPS_EXTENDED_IMAGES_ENV_SOURCED=1` inside the actual launched shell.
   Retain the ALPS default **`NCCL_P2P_DISABLE=1`** and its other bundled
   transport defaults (observed `NCCL_PROTO=^LL128`, `FI_PROVIDER=cxi`).
   Do not reapply old `NCCL_P2P_LEVEL=NVL`/custom channel/protocol overrides.
   The image documents a P2P workaround for enroot/Beverin kernel 6.4;
   that documentation alone is not evidence of the old NaN cause.

5. Keep `BEVERIN_BLAS_BACKEND=cublas`: this selects **rocBLAS** on ROCm,
   despite the CUDA API name. It removed the old sharded GEMM slowdown and
   was enabled in the numerical comparison. Do not change it while testing
   the image change. Keep FP32 gradient reduction, not the experimental BF16
   override, and keep `DSPARK_DTENSOR_ALLGATHER=1`.

6. Preserve the `apertus15_text` plugin, its entry-point metadata,
   `apertus15_run.py` and `train_hooks.py`. They register the flattened target
   config/pruned language head and install the training guards. Use the pinned
   Speculators source plus side dependencies without shadowing image torch,
   Triton, transformers or vLLM. Check module paths as well as imports. If side
   dependencies need adjustment, build a new campaign-specific directory with
   `--no-deps` in a compute allocation, never mutate the shared image/deps.

7. Preserve numerical guards and failure capture. Use `MAX_ATTEMPTS=1`,
   `EXTRACT_START_ATTEMPTS=1`, `CHAIN=1`, and `SBATCH_EXTRA=--no-requeue`.
   Do not use the old ten-attempt policy or prequeue an `afterany` chain that
   retries arbitrary failures. An orderly deadline save can be continued only
   after checking its checkpoint/logs. A numerical failure is terminal.

Before GPU submission, run shell syntax checks and the existing relevant CPU
tests (`tests/test_beverin_numerical_guard.py`, `test_beverin_single_epoch.py`,
`test_beverin_monitor.py`) in a suitable torch environment. Render the generated
batch script with a local stub `sbatch` and inspect its actual exports, image,
network flag, retry limits, runtime paths and scheduler settings. Shell-check
any new launcher. Do not treat successful parsing as GPU compatibility.

## Training recipe to preserve

| Setting | Value |
| --- | --- |
| Layout | 32 nodes × 4 MI300A; per node TP2 target extraction on physical GPUs 0/1 and two FSDP trainer ranks on 2/3; 128 GPUs allocated, 64 training ranks |
| Drafter | DSpark, 5 layers, block size 8, Markov rank 256, confidence head with Markov input; approximately 6B trainable parameters |
| Target feature layers | `1 20 39 58 77` |
| Sequence/anchors | `SEQ=8192`, `ANCHORS=512` per training rank; nominal context-token budget `64 × 8192 = 524288` per optimizer step |
| Optimization | Muon plus the trainer's AdamW parameter groups, Nesterov=True, existing weight-decay/group defaults, gradient norm clipping at 1.0; no gradient accumulation |
| Peak LR | **`8.48528137423857e-4`**, the prior 32-node value; square-root batch heuristic from Yu's `3e-4` at 65,536 tokens, not a quality guarantee |
| Schedule | `EPOCHS=1`, linear over expected **4077 updates**, **40 warmup steps**; assert actual loader length, do not silently replace the expectation |
| Loss | Existing eager CE 0.1 + TV 0.9 plus configured confidence BCE; keep the pinned implementation/default confidence behavior |
| Precision/distribution | Existing BF16 parameter/compute policy, FP32 reduction, `--fsdp-shard`, all-gather/chunk Muon redistribution |
| Data/extraction | Existing thinking-off 70B-regenerated OPB dataset, online file-backed features, `--on-missing generate --on-generate delete`; per-node local `/tmp/hidden_states` |
| Extraction memory/workers | `EXTRACT_MEM=0.62`, `--api-server-count 1 --renderer-num-workers 1`, `WORKERS=1`, `PREFETCH=1`, eager extraction |
| Saving | `CKPT_FREQ=0.015625` (about every 64 updates at this loader length), `BEVERIN_EMPTY_CACHE_ON_SAVE=1`; deadline-save margin 2400 s |
| Reproducibility | Preserve recorded seed 42 and existing determinism setting; record actual resolved values, with no claim of bitwise replay |

The user-confirmed **1.5e-4 LR belongs to the separate one-node run**, with
16,384 context tokens per step. It is not the LR used for the 32-node campaign.
Do not change node count while keeping the 32-node batch/scheduler claims.

## Execute in stages, with no blind retry

### A. Small actual-image integration gate

Use one bounded **one-node allocation, at most 90 minutes**, new output ID,
no retry. Reuse the existing small GPU preflight rather than starting full
training to test basic compatibility:

- Run `methods/dspark/beverin/check_numerical_guard.py` with two trainer GPUs
  in the ALPS image. Its finite, one-rank bad-input and bad-backward cases must
  all pass their assertions; each injected-failure case must produce both
  rank capture files containing batch and pre-forward RNG state. Injected
  test NaNs belong only in this preflight's isolated directory.
- Run the TP2 70B extraction/check using the existing 16 smoke rows and
  `methods/dspark/beverin/extract-check.sh` as the implementation reference.
  Pass the production extraction flags explicitly (`EXTRACT_ARGS` for this
  helper, whereas training uses `EXTRACT_FLAGS`). Require all expected rows,
  token/shape alignment, finite hidden states and successful `HS_CHECK_OK`.
  Historical top-1 agreement was about 0.978, with 0.983 as the Clariden
  reference; investigate a material regression, do not merely test `/health`.
- Record actual image hash/versions, GPU visibility, loaded library paths,
  sourced ALPS environment, extraction provenance and peak shared memory.
  Run guard and extraction checks sequentially so they do not compete for
  GPU 2 or leave an old server alive.

The existing tall ALPS oracle already passed; do not repeat its known-crashing
old-image cell just to obtain another failure. If this gate fails, diagnose
the specific import/API/extraction error before requesting 32 nodes.

### B. First 640 updates of the new 32-node campaign

Use a fresh unique campaign ID and the exact recipe above, a six-hour cap,
one allocation, and `BEVERIN_SEGMENT_STEPS=640`. Keep **`EPOCHS=1` and the
4077-step schedule**. Do **not** use `MAX_STEPS=640`, which changes the scheduler
horizon. Disable the final one-epoch completion wrapper for this bounded stage:
`BEVERIN_SINGLE_EPOCH=0`; leave `BEVERIN_PILOT_RESUME` and profiling unset.

Enable `BEVERIN_CHECK_ACTIVATIONS=1` and `BEVERIN_CAPTURE_BAD_BATCH=1`.
The Beverin wrapper must export `BEVERIN_NUMERICAL_GUARD=1`; confirm the
`BEVERIN_HOOKS` record on every trainer rank. Guard checks cover loaded
parameters, inputs, loss/metrics, backward gradients **before clipping**, the
returned clipping norm and updated parameters. Activation mode adds backbone,
heads and separate optimizer-stage checks. Never mask/skip a bad batch, use
`nan_to_num`, remove a failure marker or disable the guard to continue.

Pass this stage only if:

- All 32 extraction servers and 64 trainer ranks are accounted for, with the
  intended runtime/image/ALPS settings and no unexpected trainer attempt.
- Counters 0–639 are contiguous, all logged numerical training metrics/LRs are
  finite, no numerical failure marker exists, and saves complete.
- Saved state is `epoch=0, local_step=640, global_step=640`. Both scheduler
  states retain the 4077 horizon/40 warmup; the previous matching diagnostic
  recorded next LR `0.0007224154590849137` at 640.
- On compute, check **all saved model tensors and floating optimizer-state
  tensors for finiteness**, with a version-compatible loader and bounded
  memory/chunks. Verify the optimizer/scheduler files are readable. The earlier
  11,073-value samples were not a complete scan and are not sufficient for
  this claim. Do not scan/load the 13 GB files on the login node.

Keep a stable, separately named copy of the accepted step-640 checkpoint and
its audit/provenance **while no writer is active**, before the next stage.
Mid-epoch saves reuse `checkpoints/0` and otherwise overwrite that evidence.
Do not run multiple writers under this ID or copy a partially written save.

### C. Resume the same campaign to finish the epoch and validate

Check live jobs and the accepted state before submission. Reuse **the same
RUN_ID, runtime/image, data, 32 nodes, LR and one-epoch schedule**. Remove
`BEVERIN_SEGMENT_STEPS`, `MAX_STEPS`, `BEVERIN_PILOT_RESUME` and profiling;
set `BEVERIN_SINGLE_EPOCH=1`, `BEVERIN_EXPECTED_EPOCH_STEPS=4077`.
Use one 24-hour allocation initially, `CHAIN=1`, `MAX_ATTEMPTS=1`.

Require an actual resume from 640 with restored optimizer/scheduler state and
continuous LR/counters, not an initialization or cursor reset. Audit startup
and the first resumed updates promptly. Keep the guards/activation checks on
for this first ALPS campaign; disabling them changes the tested trajectory and
requires a separate, documented performance/stability decision.

If an orderly deadline save requires another allocation, inspect its exit,
last finite metrics and saved state, confirm no concurrent writer, and submit
one continuation of the same campaign. Do not treat every Slurm TIMEOUT as a
safe resume or every `ExitCode=0:0`/COMPLETED as authenticated epoch completion.
An exception, GPU fault, nonfinite value or stall needs diagnosis before any
replacement. Preserve the last good checkpoint and raw failure evidence.

### D. Define completion and validation correctly

The trainer's held-out validation is an **inference/forward pass over the
prepared OPB validation split using target hidden states**, without optimizer
updates. It aggregates the existing loss components, accuracy, confidence and
analytical/greedy accepted-length metrics. This is separate from serving-based
speculative decoding evaluation and wall-clock inference speedup.

At the end require state `{"epoch":0,"local_step":0,"global_step":4077}`,
finite saved `val_metrics.json`, nonempty/readable model, optimizer, scheduler
and config files, and an authenticated **`single-epoch-complete.json`** via
`methods/dspark/beverin/single_epoch.py`. The completion wrapper must be active
for this unbounded final stage. The marker ties metadata/metrics to artifacts,
but does not replace an independent full finiteness scan or full weight hashes.

If training completes but validation hits the allocation limit, a checked
finite final checkpoint may be resumed **for validation only** using the
existing one-epoch recovery hook; it must not train another epoch or rewrite
the completed training checkpoint. Numerical validation failure is terminal,
not a reason to run identical validation ten times. Do not claim quality or
successful training until validation finishes and the guard authenticates it.

## Environment template for stages B/C

This is configuration for the **new, reviewed ALPS-capable launcher**, not a
claim that the current submitter already supports ALPS correctly. Apply the
integration changes above and use its actual path in place of
`ALPS_SUBMITTER`. The proposed EDF does not exist in the source baseline yet.
Resolve/record `RUNTIME` and `RUN_ID` once and keep them unchanged between B/C.

```bash
export WORK=/iopsstor/scratch/cscs/faruk_zahiragic/beverin-feasibility
export RUNTIME="${WORK}/repo-<your-committed-training-sha-prefix>"
export RUN_ID=opb-thinkoff-32n-alps-1epoch-<unique-UTC-stamp>
export EDF="${RUNTIME}/methods/dspark/beverin/env-vllm-alps.toml"
export SPECULATORS="${WORK}/speculators-7046ac6-provenance"
export SIDE_DEPS="${WORK}/py312/pydeps"
export DATA=/iopsstor/scratch/cscs/faruk_zahiragic/apertus-dspark70b/data
export DATA_PROVENANCE="${WORK}/provenance/data-20261025"
export NODES=32 CHAIN=1 MAX_ATTEMPTS=1 EXTRACT_START_ATTEMPTS=1
export STOP_MARGIN=2400 SBATCH_EXTRA=--no-requeue
export SEQ=8192 ANCHORS=512 EPOCHS=1 LR=8.48528137423857e-4
export CKPT_FREQ=0.015625 DSPARK_DTENSOR_ALLGATHER=1
export LAYERS='1 20 39 58 77'
export EXTRACT_MEM=0.62 EXTRACT_FLAGS='--api-server-count 1 --renderer-num-workers 1'
export WORKERS=1 PREFETCH=1 BEVERIN_EMPTY_CACHE_ON_SAVE=1
export BEVERIN_BLAS_BACKEND=cublas BEVERIN_CHECK_ACTIVATIONS=1
export BEVERIN_CAPTURE_BAD_BATCH=1 NCCL_DEBUG=WARN
unset MAX_STEPS BEVERIN_PILOT_RESUME BEVERIN_PROFILE_STEPS DEPENDENCY
unset BEVERIN_REDUCE_DTYPE BEVERIN_NCCL_PROTO BEVERIN_NCCL_MIN_NCHANNELS
unset BEVERIN_NCCL_NCHANNELS_PER_NET_PEER BEVERIN_FI_MR_CACHE_MONITOR

# Stage B only: refuse an existing campaign directory, after checking jobs.
test ! -e "${WORK}/runs/${RUN_ID}" || exit 1
export TIME=06:00:00 BEVERIN_SEGMENT_STEPS=640 BEVERIN_SINGLE_EPOCH=0
unset BEVERIN_EXPECTED_EPOCH_STEPS
# bash "${ALPS_SUBMITTER}"  # submit only after integration review / stage A

# Stage C is a later, separate submission after the stage-B audit passes.
# Re-export the SAME common configuration / runtime / run ID in a fresh shell.
# Do not run the fresh-directory refusal above when resuming.
# unset BEVERIN_SEGMENT_STEPS
# export TIME=24:00:00 BEVERIN_SINGLE_EPOCH=1 BEVERIN_EXPECTED_EPOCH_STEPS=4077
# bash "${ALPS_SUBMITTER}"
```

Replace placeholder values before running; do not execute this template with
literal angle-bracket placeholders. Capture the fully resolved command and
generated batch file with each deployment. Check `pass_env()` in the submitter
if introducing any new flag: an environment variable not forwarded to the
batch/container can silently leave old behavior active.

## Monitoring, ETA, failure response and provenance

Check the new campaign read-only about every five minutes. Report material
changes immediately and a concise heartbeat every three hours: UTC timestamp,
Slurm state, active attempt, phase, progress /4077, last checkpoint, numerical
health, recent steady step rate, training ETA, and separate validation/queue/
restart overhead. Include validation progress as batches/total where available.
An active Slurm allocation is not proof a trainer is progressing.

The previous monitor inferred phase from checkpoint counters and missed NaN
losses while counters advanced. Reuse parser/state logic only after adding
explicit finite-metric checks, guard/failure markers, repeated-attempt alerts
and stale-data labels. Do not repoint/revive the frozen old-run cron or create
duplicate monitors. Use Cursor's durable mechanism/new campaign-specific state
and the user's available notification channel; this handoff does not authorize
unsolicited Slack/email. Stop monitoring when the campaign completes or stops.

Historical timings, **not measurements of healthy ALPS end-to-end training**:

- 32-node pilot: 19.71 s/update, 4077 updates, 22.32 training hours plus about
  1.51 hours of regular saves. Initial target load was about 31 minutes.
- Instrumented old-image 640-step diagnostic: about 16.78 s/update; not a
  controlled comparison with uninstrumented production.
- 16-node pilot: about 42.29 h/epoch including regular saves.
- One-node production windows: approximately 20–24 days/epoch. Counts depend
  on packing; the logged one-node loader had 141,472 updates, not 4077 × 32.

32 nodes is the fastest **tested** layout, not a measured optimum beyond 32.
Re-estimate from the new campaign's first 50+ steady steps, excluding startup,
first compiling/resumed steps and save latency; charge saves separately. Do
not promise a finish time from training ETA alone. MI300A GPUs and host share
roughly 501 GB/node, and `/tmp` is RAM-backed. Historical 32-node peak was
342 GiB used with 142 GiB minimum available (10-second samples); remeasure
ALPS and saves, avoid unbounded caches/hidden-state files.

On any nonfinite/GPU/stall failure, stop this campaign's automatic progression,
retain logs, checkpoint, failure JSON, all rank batch/RNG captures and any
cores, and identify the first observed phase/tensor/rank before proposing a
specific change. The first rank observing bad gradients may not originate
them because FSDP can spread a fault. For huge cores retain paths/stat identity
and small relevant metadata; do not copy or scan tens of GB on the login node.

For every deployment retain: branch/commit and source archive/manifest; exact
launch argv/generated sbatch and EDF; resolved nonsecret environment and actual
library versions/paths; image digest/file SHA256; Speculators Git SHA and dirty
patch; dataset/target identities and metadata hashes; Slurm/node IDs; trainer,
extraction and memory logs; checkpoint cursor, configs, schedulers, finite-scan
reports and validation. Always pass **`--provenance-dir`** to Speculators'
`launch_vllm.py` and retain its `vllm_command.txt`, patches and checkpoint hashes,
plus trainer `train_command.txt`, `speculators.patch` and `run.yaml`.
Do not overwrite previous result cells or publish artifacts externally unless
asked. Large checkpoint hashes and verification belong on compute nodes.

## Analysis after the healthy epoch

Report held-out loss/CE/TV, accuracy, confidence behavior and analytical accepted
length, plus serving-based acceptance and latency/throughput if subsequently
evaluated. Compare like-for-like target, thinking mode, split and inference
settings; training acceptance estimates are not a serving speedup measurement.
Inspect training versus validation behavior and retain the best usable state
before deciding on more epochs, a different LR, or a data change.

The user asked about Chinchilla/data efficiency. A feature-conditioned
speculative drafter is not a from-scratch language-model pretraining run, so
the usual rough token/parameter heuristic is not a mandatory training target.
More epochs repeat the same data; they do not create additional unique tokens.
Do not expand the corpus or run ten epochs simply to reach that ratio. Use
held-out generalization, acceptance and actual inference benefit to decide
whether more training is useful. The healthy one-epoch results are the next
decision point.
