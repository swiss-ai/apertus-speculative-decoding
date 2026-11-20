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

## Risks

- Budget: the low-effort pool is about 7.4k × ~200 tokens ≈ 1.5M assistant tokens per
  head. That is small for a 0.8B from-scratch drafter, so a scratch head that is
  much worse than the released one is the expected outcome. The comparison is still
  fair because every head gets the same tokens.
- Scratch storage: the teacher features take about 0.5 TB on iopsstor, and the
  cleaner wiped the Oct 29 campaign. Copy exports to capstor as soon as training ends.
