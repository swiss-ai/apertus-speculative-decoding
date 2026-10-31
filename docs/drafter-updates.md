# Drafter updates: workload-specific adaptation and live swap (8B)

Branch `drafter-updates` (T5). Question from Yu (15 Oct): the drafter should keep
updating for a specific workload, together with the inference service. When and
where to update it is a scheduling question. This note measures three things:
how much served-workload data a drafter needs to recover acceptance on a weak
domain, how quickly it gets there, and whether a running server can take
updated drafter weights without a restart.

## Summary

Recovery status, 31 October 2026: **the original bounded EAGLE campaign is
complete: all 19 curves and all three paired serving confirmations**. Keep
adaptation in mind for persistent multilingual traffic and low-load
summarization. Do not implement chat adaptation on this evidence; no blanket
replacement of the default drafter is justified.

At draft k3/C1, multilingual throughput improves **267→312–313 tok/s (+17%)**,
median decode time **3.41→2.78–2.79 ms/token**, and p95 request latency
**1.24→1.11–1.15 s**. Summarization improves **221→251 tok/s (+14%)**,
**4.13→3.55 ms/token**, and **1.58–1.59→1.414 s p95**. Its C64/C256 cells fail
the full benefit rule. Chat gains only about 3–8% throughput with insufficient
repeatable decode benefit and qualifies nowhere. Higher acceptance alone was
not a serving benefit.

Multilingual also has substantial matched-k gains under load, with TTFT
tradeoffs. Compare competitive settings: unadapted k3 already outperforms
adapted k7 at C64/C256; adapted k3 also has higher throughput and lower median decode latency than adapted k7 for summarization.
Larger improvements against weaker k7 baselines are retained in the raw report
but do not justify promoting dominated configurations.

The selected heads use 8k traces and no replay. Development simulated accepted
length gains are +0.757/+0.517/+0.223 for multilingual/summarization/chat; OPB
falls 4.92%/2.48%/1.93%. These are development quantities. The selected useful
heads take 29.7/40.8 minutes on four GPUs (training-loop lower bounds
1.98/2.72 GPUh). Charging all recovery traces and all curves of the respective
domain costs 15.77/23.93 GPUh, amortized at approximately 104M/160M output tokens
at matched k3/C1. The entire recovery, including all domains, serving benchmarks
and failed attempts, used **55.47 GPUh**. These are experimental recovery costs,
not a measured recurring production-update cost; volume and cadence remain
unmeasured.

| Part | Evidence | Verdict |
| --- | --- | --- |
| Original four-step smoke 3737459 | Real updates and bit-exact export/reload pass; training loss 1.643→1.960; simulated acceptance 1.585→1.602 | Infrastructure passes; original loss-improvement gate fails and remains unchanged |
| Pools/traces | Completed chat 3804324 and validated reused summarization/multilingual traces, exact hashes/renderer parity and all overlap guards | Usable original data; no regeneration of completed domains |
| Original 19 adaptation curves | 384 steps each, finite real updates, exports/reloads and all original gates recorded | Three final heads selected by the fixed development-only rule; 500-trace chat ends below initial acceptance despite its earlier best-eval gate passing |
| Paired serving | Jobs 3814370, 3814755, 3815183: 336 raw cells, 153,600 successful measured requests, zero failed warmups, 56 calibrated correctness gates pass | Multilingual specialist benefit; summarization k3/C1 benefit; chat rejected by the original substantial-benefit rule |
| Existing live same-architecture swap 3722022 | 0.33–0.56 s pause versus 207 s restart | Operational prototype already completed; not rerun |
| Scope | Original bounded EAGLE campaign complete; raw results and failures retained | Original conditional DSpark best-setting repeat still needs verified compatible data/port/provenance; no DSpark result, deployment or routing implementation claimed; broad #10/A3 remains open |

The [full verdict, absolute per-cell tradeoffs and all raw/provenance capsules](../results/8b/drafter-updates/recovery-20261030-cache/README.md)
retain all original positive, negative and failed attempts. Both same-node
GPU-swapped repeats must improve throughput and median decode latency by at
least 10%; sampled calibrated correctness is not a proof of losslessness or
full-answer quality. Fixed-concurrency seeded stand-in workloads and the original
completion caps limit the claims.

The original smoke's [unaltered artifacts and separate infrastructure assessment](../results/8b/drafter-updates/recovery-20261027/original-smoke-infrastructure.json)
are retained. Recovery uses a lightweight detached controller on the cluster,
not local Claude watchers. Its receipts and next action are at
`spec-exp-data/drafter-updates/recovery-20261030-cache/receipts/campaign.json`.
It submits one owned training job at a time and cannot create new variants.
The [initial controller staging receipt](../results/8b/drafter-updates/recovery-20261027/controller-staging-621b2050.json)
records the frozen code revision `621b2050`, all 162 verified source files,
pre-registration before trace job 3769227 started, and the successful serving
launcher preflight. The [replacement staging receipt](../results/8b/drafter-updates/recovery-20261029/controller-staging-b58ee54a.json)
records source `b58ee54a` on task branch `drafter-updates`, all 162 verified source
files, the unchanged original policy hash, and job 3790165 pending at 10:04 UTC
on 29 October. The [unaltered failed-attempt capsule](../results/8b/drafter-updates/recovery-20261029/failed-trace-3769227/artifact-manifest.json)
retains logs, exact launch/resolved environment, image digest and target contract.
The original bounded EAGLE workflow is complete; its conditional DSpark follow-up remains dependent on compatible data/port/provenance.
The [30 October staging receipt](../results/8b/drafter-updates/recovery-20261030/controller-staging-e9cc4842.json)
pins source `e9cc4842` and the unchanged original scientific-policy hash. It
records the completed-domain reuse audit, successful physical-line parsing of
all 8,448 chat inputs on the cluster, and job 3802584 pending at 12:02 UTC.
The [cache-recovery staging receipt](../results/8b/drafter-updates/recovery-20261030-cache/controller-staging-589ddd59.json)
records source `589ddd59`, 163 verified source files and the unchanged scientific
policy hash. Job 3804324 was pending at 13:33 UTC; its actual-image cache preflight
is staged before target loading. The [preserved quota-failure capsule](../results/8b/drafter-updates/recovery-20261030-cache/failed-trace-3802584/artifact-manifest.json)
retains the failed launch, logs, image/source hashes and environment.

The [preserved 3790165 failure and reuse capsule](../results/8b/drafter-updates/recovery-20261030/failed-trace-3790165/artifact-manifest.json)
contains losslessly compressed generation logs, original summaries and source/
launch provenance. Each completed domain has 256 dev and 8,192 pool outputs,
zero skips, matching input/output SHA-256, the expected target/sampling policy,
and 32 successful renderer-parity checks. Those are usable training artifacts,
not evidence of an adaptation benefit. Chat's source JSONL remains byte-identical:
one NEL in dev and nine in the pool are legal string content, not record breaks.

## Method

### Stand-in served traces (step A)

There is no captured production traffic yet. Each weak domain gets a prompt
pool the e10 drafters never saw, disjoint from the 8B test workloads, and the
target answers it as a deployment would: plain vLLM, thinking off, Apertus
sampling (T 0.7, top-p 0.8, top-k 20). Each row is seeded from its id
(`generate_targets --temperature`). Builder:
`methods/eagle/apertus_eagle/build_adaptation_pools.py`. Job script:
`methods/eagle/launch/drafter-traces.sh`.

| domain | source | input tokens | pool | dev | test |
| --- | --- | --- | --- | --- | --- |
| chat | SFT-mix `train` rows, chat_qa / multilingual / instruction_following (the chat test workload uses the `heldout_test` rows of the same domains) | 128-2048 | 8,192 | 256 | existing `eagle-8b-test` chat |
| summarization | finepdfs-edu eng/deu/fra/ita, files 4-7 (the workloads read files 0-1), same instruction and document window as the test workload, EN/DE/FR/IT 40/20/20/20 | 2048-4096 | 8,192 | 256 | existing `eagle-8b-test` summarization |
| multilingual | Infomaniak-AI/speculators-multilingual-en-fr-de-it-es (revision b1d6938), fr/de/it/es rows only: translated open-perfectblend prompts. Pool from `train_final.json`; dev and a new 128-prompt test workload from `test_final.json`; source rows used in its test file are excluded from the pool in every language | 16-2048 | 8,192 | 256 | 128 (new) |

Every pool, dev and test row is checked against the committed test and
validation workloads by rendered prompt-token digest, by normalized text and
by 13-gram near-duplicates. The multilingual test and dev rows are also
protected against the pool. Pools are in hash order, so the first N rows nest
for every N. The text stays on the cluster; the manifest is in
`results/8b/drafter-updates/data/`.

Licences: SFT-mix components and finepdfs-edu (ODC-By) as for the existing
workloads. The Infomaniak set is "inherited from source dataset", the
open-perfectblend regeneration. It is used here for a measurement and is not
redistributed.

Independent multilingual check. T6's coverage slices (workload-coverage
branch, ba1533b; `docs/workload-coverage.md`) are the weakest serving
workloads at DSpark k=7:

- 1.55-1.73 accepted tokens per step: Arabic, Swahili, German, Romansh,
  Hindi, Italian, Japanese;
- 1.87-2.23: translation from English into de/fr/it/es, and French,
  Chinese, Spanish, Yoruba, Amharic;
- EAGLE on prompts over 8k tokens: 1.81, against 2.71 for DSpark.

Their 1,344 prompts (restricted text, cluster only) are protected in the pool
build (278078e), so no pool or dev row duplicates one. The multilingual head
will be confirmed on `lang-deu`, `lang-fra`, `lang-ita`, `lang-spa` and
`translate-from-en`. These come from a different source than its
Infomaniak training traces, which are translated open-perfectblend prompts.
`multi-turn`, `code-multi` and `math-gsm8k` check for forgetting. The pools
cover only fr/de/it/es. Arabic, Swahili, Romansh, Hindi and Japanese, and
>8k-token prompts, would each need their own Apertus-answered pool: a
follow-up, not part of this curve.

### Adaptation curve (step B)

`train_rollout` gained `model.init_from`: it loads a trained draft's weights
before the optimizer is built, so the optimizer state, schedule and step
count start fresh. The frozen embedding and lm_head must equal the current
target's. A run that already has `checkpoints/resume` resumes as before. It
also gained `dataset.extra_eval`, which evaluates further held-out sets at
every eval. Every fine-tune starts from e10's served checkpoint
(`e31-opb-thinkoff-stage1/checkpoints/best`, step 251,955). The settings:

- online target features, as e10;
- 32 rows per step on 1 node (4 ranks);
- constant learning rate 3e-5 after 19 warmup steps (e10 peaked at 1e-4 and
  decayed to 0); one arm uses 1e-4;
- 384 steps (12,288 rows: 24.6 epochs of 500 traces, 6.1 of 2k, 1.5 of 8k);
- evaluation every 32 steps on the domain dev traces (`eval_data`) and on the
  first 512 of e10's 2,000 open-perfectblend held-out rows (`extra_eval`
  "opb", the forgetting check);
- replay arms add N/4 open-perfectblend training rows (20% of the mix);
- the final checkpoint is exported.

With a constant learning rate, the eval trace of one run approximates the
curve over step budgets. Simulated acceptance (`simulated_acc_len`, 7 rollout
steps) is the trainer's cheap metric. It is not serving acceptance; the node
sweeps below measure that.

### Live swap (step D)

The fork's `update_weights` and `reload_weights` reload the target.
`src/apertus_bench/drafter_reload.py` is a vLLM worker extension
(`--worker-extension-cls apertus_bench.drafter_reload.DrafterReloadExtension`)
with `reload_drafter(path)`. Each worker reads the new head's safetensors and
passes them to the draft model's own `load_weights`, which applies the same
name mapping and sharding as at startup and copies into the existing
parameters. Parameter addresses must not move: CUDA graphs and compiled code
stay valid, and no KV cache or request state is touched. The engine core runs
the RPC between two steps.

The new head must meet two conditions, or the reload is refused before any
copy:

- tensors it shares with the target (the embedding, and the lm_head when the
  head's copy is frozen) must equal the target's; they are never written;
- its architecture keys must equal those of the served head.

The RPC is reached through the dev endpoint `/collective_rpc`
(`VLLM_SERVER_DEV_MODE=1`). In production, that endpoint must sit behind the
gateway's admin interface.

`src/apertus_bench/drafter_swap.py session` runs the measurement:

1. Serve the target with head A at k=7.
2. Run a greedy pass on 32 fixed prompts.
3. Run a closed-loop sampled load at fixed concurrency. Halfway, it swaps to
   head B and the load keeps running.
4. Run a greedy pass with B.
5. Swap back to A and run a greedy pass with A again. This last pass is the
   control: the same drafter, so its differences are numerical noise.

It records:

- each stream's event times (the stall across the swap);
- spec-decode counters every second (acceptance before and after);
- parameter digests of the drafter and of a target sample.

## Results

### Live swap smoke (job 3722022)

The served e10 head (`e31-opb-thinkoff-stage1-se`) was swapped under load to
the epoch-3 head of the same run (`-ep3-se`), which has the same architecture
and lower acceptance, and later swapped back. k=7, one GPU per arm, 360 s of
closed-loop sampled load, swap at 180 s. Record:
`results/8b/sweeps/drafter-swap-smoke-20261022T142432Z/`.

| | chat, C=32 | summarization, C=64 |
| --- | --- | --- |
| requests (errors) | 5,691 (0) | 2,631 (0) |
| swap RPC, under load | 0.56 s (read 0.49 s, copy 0.06 s) | 0.33 s (read 0.23 s, copy 0.06 s) |
| swap back (warm page cache) | 0.14 s | 0.14 s |
| longest stream gap across the swap | 0.56 s | 0.30 s |
| stream gap before the swap, p50 / p99 / max | 0.019 / 0.070 / 0.10 s | 0.037 / 0.26 / 0.41 s |
| acceptance length before → after | 2.47 → 2.42 | 2.18 → 2.14 |
| greedy exact match, post vs pre / back vs pre | 32/32 / 32/32 | 32/32 / 32/32 |
| drafter digest changed, then restored; target digest unchanged | yes | yes |

- **Pause:** the swap stalls every stream for about the RPC time, at most
  half a second, and no request is lost. A restart costs far more: this
  server took 207 s from launch to its first completion (load, compile,
  CUDA graph capture), and in-flight requests would be dropped or drained.
- **Where the time goes:** most of it is reading 0.5 GB of safetensors from
  /iopsstor with a cold page cache. Reading the file into host memory before
  the RPC would cut the stall to the copy (about 60 ms).
- **Shared tensors:** 15 tensors were loaded. The frozen lm_head in the new
  head was checked equal to the target's and not written.
- **Acceptance:** the drop after swapping to e3 is small but has the
  expected sign, so the new weights are live. Each window is about 150 s
  after a 30 s settle.
- **Shutdown error:** the `AsyncLLM output_handler failed` error in
  replica.out comes from teardown after SIGINT, once the session had
  finished.

## Design note: when and where to update

A production update cadence is not established yet. It must follow the measured
adaptation curve, forgetting on OPB and held-out serving confirmation; a four-step
smoke or the successful weight-swap prototype cannot determine it. Train on a
separate node from serving, then canary a same-architecture head and retain the
previous head for rollback. Use per-domain acceptance measurements and explicit
task labels to identify a weak domain. This study uses existing dataset prompts
and target-generated stand-in traces; live production prompt/completion capture
is outside its scope.

The recovery controller keeps the original 19 configurations (500/2k/8k traces,
with/without 20% OPB replay, plus the existing high-LR summarization arm), with
384 optimizer steps each. It records every scientific gate even when an
infrastructure-complete run fails one. The preregistered rule selects at most
one final head per domain (three total),
requiring all unchanged original scientific gates and at least +0.20 development
simulated accepted tokens. Ties favor fewer traces, replay, lower learning rate,
then run name. OPB forgetting is reported without excluding specialist selection.

Selected heads proceed automatically to original within-job, GPU-swapped serving
comparisons at k=3/7 and C=1/64/256. Each head gets two sequential four-GPU rounds
in one normal allocation (24-hour limit): base/adapted on physical GPUs 0/1 for
k3 and 2/3 for k7, then reversed on the same GPUs. Original full held-out prompt
passes use natural EOS and Apertus sampling. First 32 prompts per workload, greedy
and capped at 256 output tokens, supply base/base and adapted/adapted calibration
controls using the existing correctness gate and default tolerances.

Only at least 10% throughput **and** 10% median decode-latency improvement in both
GPU-swapped repeats counts as substantial. Report absolute tok/s, TPOT, TTFT and
tail latencies, acceptance, GPU-seconds per million output tokens and compute
break-even. Material TTFT/tail/OPB regression makes the result a conditional
specialist tradeoff; it cannot justify broad replacement. No minimum arbitrary
absolute per-token saving is imposed. Simulated acceptance never implies serving
speedup. The full rule is committed in
`methods/eagle/configs/8b/adapt-recovery-policy.json` and frozen before new results.

No qualifying development head, or no substantial repeatable serving cell, closes
as `negative_complete`. Successful bounded EAGLE recovery writes `verdict.json`
and `verdict.md` automatically; no manual confirmation-selection checkpoint remains.
Infrastructure/nonfinite-update/export/correctness failures stop as `needs_repair`.
Valid negative independent scientific arms continue. The original conditional
DSpark best-setting repeat remains separate: only a qualifying EAGLE serving
result triggers a recorded dependency to verify compatible data/port/provenance;
this campaign does not submit that extension. No additional architecture,
language, context-length or adapter experiments are authorized by this recovery.

Runtime isolation matters: the original remote `scratch` symlink pointed into
the ongoing EAGLE checkout. Recovery clones the TorchSpec code and copies the
mutable Python dependencies into its own runtime directory; remaining input
symlinks are read-only. This prevents the training bootstrap from changing
another run's patched source or wandb stub.

The home-quota recovery changes cache locations only. The pinned image's
FlashInfer source uses `FLASHINFER_WORKSPACE_BASE` independently of vLLM's cache;
its usage-stat writer uses `VLLM_CONFIG_ROOT`. Both now point into owned scratch,
together with CUDA/Triton/Inductor/Hugging Face caches. `HOME`, target/drafter
settings and scientific gates stay unchanged. The wrapper tests directory writes
and the actual image's FlashInfer sampling lock/vLLM config path before loading a
target. It saves the effective paths with each deployment. Training uses a
job/node cache directory; paired serving uses a fresh directory for every arm
and repeat. No shared home files or unrelated jobs are cleaned or modified.

## Runs

| job | what | outcome |
| --- | --- | --- |
| 3721346 | trace smoke (debug) | failed: no pyarrow in the image (fixed in a33ee4f) |
| 3721372 | trace smoke (debug): 64/16/16 rows per domain, 16 generations per file | ok |
| 3722013 | pools (8,192 + 256 per domain, 128 multilingual test) and traces, Apertus sampling | failed before generation: equal source-ID/rank tuples compared dictionary payloads in the summarization selector |
| 3722022 | live swap smoke, e10 → e3 → e10, chat C=32 and summarization C=64 | ok (above) |
| 3725574 | fine-tune smoke: 4 steps from e10 on the smoke traces (`adapt-smoke.yaml`) | failed after 50 s: submitted without `EAGLE_CUDA_DEVICES=0,1,2,3`, so ranks 1-3 had no GPU (the submitter now refuses this, a2dbf52) |
| 3737459 | fine-tune smoke, resubmitted with 4 GPUs | 4 steps and export/reload pass; exit 3 solely because the four-step training-loss gate failed; original scientific result preserved |
| 3769227 | fresh full pools/traces after deterministic heap fix | failed after 409 s: distinct FinePDFs chunks shared source-only prompt IDs; no target generation/training began |
| 3790165 | content-aware summarization prompt IDs; full original pools/traces, frozen source b58ee54a | failed after 1,636 s: summarization/multilingual complete (8,192+256 each, no skips, 32 renderer checks); chat JSONL reader incorrectly split U+0085 inside prompt text |
| 3802584 | physical-line JSONL reader; chat only; source e9cc4842 | failed after 150 s during engine startup: FlashInfer sampling.lock and vLLM usage-stat writes hit home quota (errno 122); no chat output |
| 3804324 | scratch-cache wrapper and actual image lock/config probes; chat only; source 589ddd59 | completed in 650 s; validated completed domains reused |
| 3804872 / 3806293 / 3810109 | original chat / summarization / multilingual training batches | completed all 19 arms, 384 steps each; frozen development-only selection retained |
| 3814329 | first allocated confirmation, source b812627f | failed in 156 s before requests: inherited TMPDIR made vLLM AF_UNIX socket names exceed 107 bytes |
| 3814370 | repaired multilingual confirmation, source 7f0a41dc | completed in 4,264 s; 108,544 successful measured requests, zero failed warmups, all 40 correctness gates pass |
| 3814755 | original summarization confirmation | completed in 1,926 s; all eight correctness gates and 22,528 measured requests pass; actual 2 h scheduling cap retained |
| 3815183 | original final chat confirmation | completed in 1,603 s; all eight correctness gates and 22,528 measured requests pass; no primary cell qualifies; original 12 h request retained |

The first serving submission failed before Slurm at a login-node heredoc ENOSPC;
its raw setup is preserved. Fresh attempt `repair-20261031` used a writable owned
scratch TMPDIR, then sbatch rejected the frozen 24-hour request because `normal`
currently allows 12 hours. A scheduling-only cap of `12:00:00` now preserves the
original 24-hour maximum in the immutable policy while recording the actual
request. New attempts use fresh directories. All original prompts, concurrency
levels, two swapped rounds, correctness checks and selection remain unchanged;
completed training is never resubmitted by confirmation-only recovery.

Serving attempt `repair-20261031b` submitted job 3814329 (156 s, failed before
requests): vLLM inherited the long submission TMPDIR and its ZeroMQ socket name
exceeded Linux's 107-byte AF_UNIX limit. Serving now uses a short private
`/iopsstor/scratch/cscs/$USER/t5-$SLURM_JOB_ID/<round>-<arm>` directory for TMPDIR
and the pinned image's explicit `VLLM_RPC_BASE_PATH`. A real socket-bind probe
checks the UUID-shaped path before model startup. Its paths and probe are saved
with runtime provenance. Inspected failed jobs remain in raw artifacts, receipt
history and allocation accounting; an explicit terminal-job check is required
before a fresh attempt. Completed confirmation reports are hash-verified and
reused if a later independent head needs infrastructure repair.

Recovery code: `apertus_eagle.adaptation_smoke` assesses infrastructure without
rewriting gates. `apertus_eagle.adaptation_recovery` waits for the trace job,
prepares nested input subsets, renders immutable copies of the original configs
under the recovery root, then runs chat/summarization/multilingual batches through
the existing EAGLE submitter. It stops on infrastructure failure and after at most
three attempts per batch, records full original scientific gates, and finishes
with `adaptation-matrix-v1.json`, `selection.json`, paired confirmation records,
and a final machine-readable `verdict.json`. A detached sparse worktree pins
controller/launcher sources and hashes; private short overlay roots mitigate historical Pyxis
mount-list failures. Exact resolved list bytes are recorded; no measured byte
limit is assumed. An exclusive file lock and durable receipts prevent duplicate
controllers; restarting resumes owned work. `freeze_adaptation` checks the old
process identity and replaces only the trace-waiting controller under a separate
replacement lock. All historical raw scientific artifacts remain unchanged.

Results/root:
`/iopsstor/scratch/cscs/faruk_zahiragic/spec-exp-data/drafter-updates/recovery-20261030-cache`.
Controller: `receipts/start-controller-7f0a41dcc0a3.sh`; log: `receipts/controller.log`.
Previous recovery trees and frozen sources remain intact. The current
receipt retains all trace job IDs and their compute cost. Completed traces are
linked for read-only reuse; only chat has a fresh generation output directory.
No previous raw run, trained head, generated corpus or comparison is overwritten.
