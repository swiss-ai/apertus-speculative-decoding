# DSpark adaptation repeat: status

Branch `experiment/dspark-adaptation`. Cluster checkout
`/iopsstor/scratch/cscs/faruk_zahiragic/spec-exp/dspark-adaptation`; run root
`ROOT=/iopsstor/scratch/cscs/faruk_zahiragic/spec-exp-data/dspark-adaptation/adapt-mult-n8k-r0-20261104`.

| when (UTC) | job | state | next |
| --- | --- | --- | --- |
| 2026-11-04 18:52 | 3856327 (`dspark-8b-adapt`, normal, 6 h, source 01114fc6) | cancelled while pending (no start estimate on normal); no step ran | moved to debug chain |
| 2026-11-04 19:34 | 3856449 (`dspark-8b-adapt`, debug, 1.5 h, source 720d3624) | pending | chain: after it ends, resubmit the same command until `ROOT/stages/report.done`; one debug job at a time, none while a `task-speed-*` job is on debug |

Resume: `adapt-job.sh` skips steps with markers in `ROOT/stages/` and stops
before a step that does not fit the job's deadline; a cut step's outputs go to
`ROOT/incomplete/` and it reruns. Submit (cluster checkout):
`STAGE=8b MODE=adapt ADAPT_ROOT=ROOT TIME=01:30:00 DSPARK_PARTITION=debug SML_ENV_DIR=ROOT/env DSPARK_TRAIN_PATCH_DIR=/iopsstor/scratch/cscs/faruk_zahiragic/da-vp-train DSPARK_LOG_ROOT=ROOT/logs ./methods/dspark/train/submit-train.sh`.
After `report.done`: if `ROOT/development-report.json` says `selected`, run
`python3 methods/dspark/train/submit_adapt_confirmation.py --root ROOT`. Job log:
`ROOT/logs/<job>.out`; per-step trainer and extraction logs in `ROOT/runs/*/logs/`.

History note: the branch was rebased onto the rewritten `drafter-updates`
(737213c7) on 2026-11-04. Job 3856327 was submitted from the pre-rewrite
commit 01114fc6, whose rewritten equivalent is 4f43d77a; `methods/dspark/` is
identical between the two (the rewrite only removed restricted capture archives).
