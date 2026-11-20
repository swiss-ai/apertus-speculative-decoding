# Per-domain DSpark drafters: status

Design: [docs/domain-drafters-dspark.md](../../../docs/domain-drafters-dspark.md).
Cluster checkout `/iopsstor/scratch/cscs/faruk_zahiragic/spec-exp/domain-dspark`
(a `git archive` of the revision in its `.source-revision`; speculators
7046ac6 + repository patches in `scratch/speculators`). Run roots under
`/iopsstor/scratch/cscs/faruk_zahiragic/spec-exp-data/domain-drafters/`;
every root keeps `submissions/submissions.jsonl` (job, dependency, source
revision, plan hash) and `logs/<job>.out`. Heads are backed up to
`/capstor/scratch/cscs/faruk_zahiragic/domain-drafters-backup/<root>/`.

| when (UTC) | job | what | state |
| --- | --- | --- | --- |
| 2026-11-20 12:21 | 3987876 (debug, submitted at 184ec8c8; the checkout moved to 225e745e at 12:22, which changed only `domain_drafters.py` marker handling and plans, unused by this no-domain run) | runtime check, root `dspark-runtime-20261120T122122Z`: side packages into the shared `dspark-runtime/pydeps`, e10 rebind, OPB held-out (first 512 val_ids rows), e10 lr-0 OPB evaluation (expect ~6.269) | **done 12:33, 12 min, DOMAIN_OK.** Rebind verified the e10 hash; OPB eval file 576 rows (64 placeholder + 512), sha256 bda54c1a…, row digests 6c2a6983…; e10 OPB eal **6.225** (accept len 6.096, loss 0.240), weights bitwise unchanged. 0.7% below the repeat's 6.269: the old heldout.jsonl is gone, so row identity cannot be compared; every head is compared with e10 on this same file, so the grid is internally controlled |
| 2026-11-20 13:03/13:08 | none | watcher smoke submissions (afterok 3987876) | **rejected**: sbatch refuses afterok on a finished job; domain named `tool`, not `structured`. Fixed in 4864d7ff |
| 2026-11-20 13:14 | 3988408 (debug, 4864d7ff) | smoke, root `dspark-pilot-20261120T131405Z` (pilot: 512 train / 64 dev per domain, 6 domains incl. `tool`) | COMPLETED 17:51 but **paused before train** (90-min step bound > debug time left). Done: lock, eval sets, e10 lr-0 evals (bitwise unchanged): pilot math dev eal 7.855 (accept len 7.646), OPB 32 rows eal 6.358; mixed head convert + prepare |
| 2026-11-20 13:33 | none (debug) | smoke resubmit, same root | rejected: debug QOSMaxSubmitJobPerUserLimit (gpt-oss jobs 3988416/3988706 hold both slots); fixed in 4092e4aa/next commit (fail loudly, TRAIN_MINUTES=15) |
| 2026-11-20 13:33 | 3988723 (normal, 1:30) | smoke resumed on the same root | cancelled 13:35 in favour of debug (coordinator: domain jobs first on debug) |
| 2026-11-20 13:35 | 3988743 (debug, fce616d2) | smoke resumed: mixed head (512 rows, 85-86 per domain, 520k tokens) 6 steps, export, backup, evals | FAILED 9:57 at **backup**: training ok (global step 6), export ok (42 tensors changed vs e10, reload bitwise equal); `cp -a` cannot preserve ACL permissions on capstor. Fixed in the next commit (`cp -r --preserve=timestamps`) |
| 2026-11-20 13:47 | (detached `~/.local/state/domain-drafters-dspark/resubmit-smoke.sh`) | smoke resumed again (backup + mixed math/OPB evals, ~15 min) | job **3988996** (debug, 7f25b08b), COMPLETED 15:48, DOMAIN_OK. Backup verified (sha256 433ff465…). **Smoke (pilot, 6 steps, not a result):** e10 → mixed eal: pilot math dev 7.855 → 7.859, OPB 32 rows 6.358 → 6.356. All 4 lr-0 evaluations bitwise unchanged. Pipeline end to end verified; `smoke-done` written 14:04 UTC |
| pending | (watcher, restarted 13:14) | full grid: baseline + 7 heads (afterok baseline), afterok on the smoke id in `~/.local/state/domain-drafters-dspark/smoke`, read at submit time | waits for all six `READY-<domain>` (math code chat summarization multilingual tool) |

The watcher `methods/dspark/train/watch-domain-data.sh` runs on the
workstation (state and log in `~/.local/state/domain-drafters-dspark/`) and
exits after the full submission.

Known constraint (2026-11-20 13:20 CET; the data campaign's generation jobs domain-gen0/1 wait on it too): account infra01 is at its QOS group
node limit on `normal` (752 nodes running, 23 jobs waiting on
QOSGrpNodeLimit), so the 8 one-node grid jobs may queue.
