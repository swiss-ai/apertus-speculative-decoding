# gpt-oss-20b effort drafters from scratch (canvas #18), launched 2026-11-20

Question (30 Oct huddle, canvas #18): fine-tuning the released DFlash head per
reasoning effort gave no consistent gain over the mixed head
([campaign-20261029-784a951f-adamw](../campaign-20261029-784a951f-adamw)). Do
effort-specific heads trained **from scratch** for one epoch do better, and how far
apart are the fine-tuned heads in weight space?

## Design

| | Scratch campaign | Matched fine-tune control |
| --- | --- | --- |
| Config | [phase2-scratch.yaml](../../../../experiments/thinking-effort/phase2-scratch.yaml) | [phase2-scratch-finetune.yaml](../../../../experiments/thinking-effort/phase2-scratch-finetune.yaml) |
| Initialization | fresh draft weights from the released DFlash config (upstream config-only path); verifier embeddings/head/norm frozen | released `z-lab/gpt-oss-20b-DFlash@d53f655` |
| Optimizer | AdamW 6e-4, 4% warmup, cosine, wd 0.01 | AdamW 1e-5 constant, wd 0.01 (Oct 29 recipe) |
| Corpus | full GSM8K train (`740312ad`), minus 64 development prompts and the 64 evaluation prompts; gpt-oss-20b responses at low/medium/high | identical files (`--data-reference` the scratch campaign) |
| Budget | per head = the smallest effort pool's assistant tokens (low), one epoch; mixed = one third per effort | same |
| Heads | low, medium, high, mixed | low, medium, high, mixed |
| Evaluation | existing Phase 2 protocol: 3 paired blocks, plain vs 4 heads at each effort, 64 math/code prompts, 1 GPU, C=1 | same |

The vanilla head is the reference arm through the Phase 1 campaign
`campaign-20261027-f409b9da`. After both campaigns finish,
[`tools/effort-weight-deltas.py`](../../../../tools/effort-weight-deltas.py)
reports relative Frobenius norms and pairwise delta cosines of the fine-tuned heads
against the released head (and of the scratch heads against each other).

## Runs (Clariden)

Source: branch `experiment/thinking-effort-scratch`, commit `f7898022`, cluster
checkout `/iopsstor/scratch/cscs/faruk_zahiragic/apertus-effort-scratch`.
Speculators `e3041ddb` restored at
`/iopsstor/scratch/cscs/faruk_zahiragic/thinking-effort/upstream/speculators-e3041ddb`
(clean, hash-pinned); drafter and Harmony vocabulary re-downloaded at the pinned
revision with all SHA-256s matching the Phase 1 lock, and backed up under
`/capstor/scratch/cscs/faruk_zahiragic/thinking-effort-backup/models`.
Campaigns live under `/iopsstor/scratch/cscs/faruk_zahiragic/thinking-effort/scratch18/`.

| Campaign | Purpose | Partition | Controller log | First job |
| --- | --- | --- | --- | --- |
| `smoke-20261120-f7898022` | 96 train / 16 dev prompts, scratch init, end-to-end check of data + training | debug 1.5 h | `smoke-f7898022-controller.log` | data 3987767 |
| `scratch-20261120-f7898022` | full scratch campaign: data → train → 3 evaluation blocks | normal 4 h | `scratch-f7898022-controller.log` | data 3987766 |
| `finetune-20261120-f7898022` | fine-tune control; started by `launch-finetune-after-data.sh` once the scratch data is complete | normal 4 h | `finetune-f7898022-controller.log` | pending |

Failed first attempt (preserved): `*-46451ee4`, data job 3987731 failed in 38 s while
pip built Speculators into the over-quota `$HOME` cache; fixed in `f7898022`.

## Smoke evaluation block-0 replaced (2026-11-20 13:46 CET)

The smoke campaign completed data and training (all four scratch heads exported;
train job 3987879). Faruk gave the debug partition to the domain experiments
first, so the smoke controller (pid 27169 on ln001) was stopped and its pending
evaluation block-0 **3988022 was cancelled** before blocks 1/2 were accepted. The
old record is kept as `submissions/phase2-block-0/job.cancelled-3988022.json`,
with `replacement.json` beside it.
[`resubmit-smoke-block0-after-dspark-smoke.sh`](resubmit-smoke-block0-after-dspark-smoke.sh)
runs on the workstation (pid 749870, log
`~/.local/state/effort-scratch18/resubmit-block0.log`). It waits for the DSpark
domain-grid watcher to write its smoke job ID, then resubmits block-0 with the
unchanged campaign argv plus `--dependency=afterany:<DSpark smoke>`, retrying
the debug submit limit. It writes the new `job.json` and `replacement.json`
and restarts the identical controller. The controller reuses any existing
`job.json`, so it waits on the new block-0 and submits blocks 1 and 2 itself.
The full scratch and fine-tune campaigns run on the normal partition and are
unaffected.

## Smoke evaluation blocks yielded again (2026-11-20 14:35 CET)

The first waiter submitted block-0 as **3988416** (afterany DSpark smoke 3988408)
and restarted the controller. The controller submitted block-1 as **3988706** with
no dependency. DSpark smoke 3988408 ended without training (time bound), and the
two blocks then held both debug submit slots, so the DSpark resubmission was
rejected. Per Faruk's standing rule (domain jobs first on debug):

- Stopped the controller (pid 216086 on ln001) and cancelled **3988416** (pending)
  and **3988706**, which had run for 2 min 35 s. The records are kept as
  `job.cancelled-<id>.json`, with a `replacement.json` history in each block directory.
- Moved block-1's partial measurement directory to
  `cancelled-attempts/phase2-block-1-job3988706`. The rerun refuses an existing
  `phase2/block-1`.
- Wrote the block-2 submission files with the controller's own `submit` in
  validate-only mode. Nothing was submitted.
- [`resubmit-smoke-blocks-after-dspark-debug-smoke.sh`](resubmit-smoke-blocks-after-dspark-debug-smoke.sh)
  (workstation, log `~/.local/state/effort-scratch18/resubmit-blocks.log`)
  replaces the first waiter. It waits for a new **debug** DSpark smoke ID in
  `~/.local/state/domain-drafters-dspark/smoke` (not 3988408 or 3988723) and waits
  until that job has **finished**. It then submits blocks 0 → 1 → 2 as an afterany
  chain starting from the smoke, so at most one block uses debug at a time, and
  restarts the identical controller, which reuses all three `job.json` records.

## Smoke evaluation blocks yielded a third time (2026-11-20 14:49 CET)

The second waiter treated "DSpark smoke 3988743 ended" as its go signal. That
smoke had **FAILED** (a DSpark backup bug, since fixed), so the waiter submitted
block-0 **3988916**, which ran for 1 min 36 s, and block-1 **3988925**
(pending, afterany 3988916). Together they blocked the domain smoke's roughly
15-minute resume. Per Faruk's domain-first decision:

- Killed the waiter (pid 948724). No controller was running yet.
- Cancelled 3988916 and 3988925. The records are kept as
  `job.cancelled-<id>.json` with `replacement.json` histories.
- Moved block-0's partial measurement directory to
  `cancelled-attempts/phase2-block-0-job3988916`. `phase2/` is empty again.
- The waiter script now proceeds **only** once
  `~/.local/state/domain-drafters-dspark/smoke-done` exists. The DSpark agent
  creates that marker after its smoke succeeds. Smoke IDs and job end states are
  ignored. If the marker is absent for 12 h, the waiter logs and exits.
  Block-0 has no dependency and blocks 1 and 2 chain afterany. The restarted
  waiter is pid 1006027 with the same log.

## Risks

- Budget: the low-effort pool is about 7.4k × ~200 tokens ≈ 1.5M assistant tokens per
  head. That is small for a 0.8B from-scratch drafter, so a scratch head that is
  much worse than the released one is the expected outcome. The comparison is still
  fair because every head gets the same tokens.
- Scratch storage: the teacher features take about 0.5 TB on iopsstor, and the
  cleaner wiped the Oct 29 campaign. Copy exports to capstor as soon as training ends.
