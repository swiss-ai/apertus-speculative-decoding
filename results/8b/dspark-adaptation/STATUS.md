# DSpark adaptation repeat: status

Branch `experiment/dspark-adaptation`. Cluster checkout
`/iopsstor/scratch/cscs/faruk_zahiragic/spec-exp/dspark-adaptation`; run root
`ROOT=/iopsstor/scratch/cscs/faruk_zahiragic/spec-exp-data/dspark-adaptation/adapt-mult-n8k-r0-20261104`.

| when (UTC) | job | state | next |
| --- | --- | --- | --- |
| 2026-11-04 18:52 | 3856327 (`dspark-8b-adapt`, normal, 1 node, 6 h, source 01114fc6) | pending (QOSGrpNodeLimit), estimated start 2026-11-05 ~00:15 | read `ROOT/development-report.json`; if `selected`, run `python3 methods/dspark/train/submit_adapt_confirmation.py --root ROOT` from the cluster checkout |

Resume: `adapt-job.sh` skips steps with markers in `ROOT/stages/`; a step that
left a run directory without a marker stops the job for inspection. Job log:
`ROOT/logs/<job>.out`; per-step trainer and extraction logs in `ROOT/runs/*/logs/`.
