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
| pending | (watcher) | smoke on the pilot slice (`READY-pilot`), afterok 3987876 | waits for traces |
| pending | (watcher) | full grid: baseline + 7 heads (afterok baseline), afterok smoke | waits for all six `READY-<domain>` |

The watcher `methods/dspark/train/watch-domain-data.sh` runs on the
workstation (state and log in `~/.local/state/domain-drafters-dspark/`) and
exits after the full submission.

Known constraint (2026-11-20 13:20 CET; the data campaign's generation jobs domain-gen0/1 wait on it too): account infra01 is at its QOS group
node limit on `normal` (752 nodes running, 23 jobs waiting on
QOSGrpNodeLimit), so the 8 one-node grid jobs may queue.
