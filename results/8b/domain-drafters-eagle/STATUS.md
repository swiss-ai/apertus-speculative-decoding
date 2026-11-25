# Domain drafters, EAGLE grid: status log

Design: [docs/domain-drafters-eagle.md](../../../docs/domain-drafters-eagle.md).
All times CET.

| time | event |
| --- | --- |
| 2026-11-20 13:05 | Scratch cleaner had emptied e10's `checkpoints/best` (dir mtime 11-20 01:45) and `apertus-eagle/data/dspark-opb-thinking-off/`. |
| 2026-11-20 13:09 | OPB held-out rebuilt: `spec-exp-data/domain-drafters/opb/heldout.jsonl`, 2,000 rows from the first 19,314 corpus rows, sha256 4293a4204ab806110ea71bdd0f034cba94dcf6a57f739dc28db8d9467a809c7a. |
| 2026-11-20 13:12 | e10 export downloaded from HF `faruk-zahiragic/Apertus-v1.5-8B-EAGLE3.1-e10-thinking-off` @ 4c9d4c9e250a83ebeee43563ecf71c1118b735cd into `domain-drafters/init/eagle-e10-thinkoff-hf`. All 15 SHA256SUMS entries OK (model.safetensors 42b0549629bed5366be812ea0a08897d3631bff2329bb7294adddd8ab100d6ff). Copy at `/capstor/scratch/cscs/faruk_zahiragic/domain-drafters-backup/init/`. |
| 2026-11-20 13:25 | Cluster snapshot `spec-exp/domain-eagle/repo-ada3eb05` (code-only `git archive` of ada3eb05; `REVISION` file holds the branch SHA; snapshot-local git commit for `harness_revision`). TorchSpec 6c042a87 + pydeps copied from `~/apertus-spec-decoding/scratch`. |
| 2026-11-20 13:25 | Export smoke submitted: job **3987943** (normal, 1 h; 4 steps on OPB held-out rows from the e10 export). Pending on `QOSGrpNodeLimit`: infra01 had 763 nodes running. The sibling's trace generation (3987893/3987894) is pending on the same limit. |
| 2026-11-20 13:25 | Workstation watcher started (`watch-and-submit.sh`, 10-min poll, 36 h cap; log `~/.local/state/domain-eagle-watch/watch.log`). It submits the seven arms once all six READY markers exist, then copies runs and heads to `/capstor/scratch/cscs/faruk_zahiragic/domain-drafters-backup/eagle/`. |
| 2026-11-20 14:20 | Sixth domain renamed `structured` → `tool` to match the data branch (a1b9c5ae). New snapshot `repo-a1b9c5ae`. The pilot slice is READY, so the pilot smoke was submitted as job **3988406** (normal, 1 h, 4 steps from the e10 export on pilot math, all six pilot dev sets + OPB). The watcher was restarted on `repo-a1b9c5ae`. No submission uses Slurm dependencies; the arms are submitted only after the READY markers exist. |
