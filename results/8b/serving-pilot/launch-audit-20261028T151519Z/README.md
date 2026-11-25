# DSpark research pilot launch audit

Launch state recorded on 28 October 2026 at 15:15 UTC. The implementation is
on `feat/dspark-research-serving-pilot`; the checked-out implementation revision
at collection was `5e1de333`. The [operator runbook](../../../../experiments/serving-pilot/README.md)
describes preparation, validation, qualification and rollout.

| Deployment | Slurm job | Partition | State at collection | Launch artifacts |
| --- | --- | --- | --- | --- |
| DSpark e10, k=3 | 3783300 | debug | PENDING, Priority | [Deployment](dev-dspark-20261028-a/deployment.json), [profile](dev-dspark-20261028-a/profile.json) |
| Matched plain target | 3783531 | normal | PENDING, Priority | [Deployment](dev-plain-20261028-b/deployment.json), [profile](dev-plain-20261028-b/profile.json) |
| Independent plain control | 3783544 | normal | PENDING, Priority | [Deployment](dev-plain-control-20261028-a/deployment.json), [profile](dev-plain-control-20261028-a/profile.json) |

All three serving jobs were retained across the interruption. Their validation
controllers were checked before continuing. The two plain controllers were
replaced while their jobs were still pending to extend their allocation wait
to 36 hours; the control retains `--smoke-only --stop-after`. The DSpark
controller continues its original wait. Each controller runs direct checks
with `--no-gateway` while development-gateway authentication remains pending.

The earlier [debug plain submission](dev-plain-20261028-a/submission-result.json)
was rejected by `QOSMaxSubmitJobPerUserLimit` and has no job ID. Its preparation
and rejection records are retained. The admitted plain deployment uses a new
directory on the normal partition.

The full run roots remain on Clariden under:

```text
/iopsstor/scratch/cscs/faruk_zahiragic/apertus-pilot-code/results/8b/serving-pilot/
```

This audit contains compact copies of configurations, source manifests,
rendered scripts, resolved environments, submission receipts and controller
records. Exact launch revisions are in each `repository.json`; the frozen
run sources and overlays remain with each deployment on the cluster.
[snapshot.json](snapshot.json) records the collected scheduler state and remote
paths. [snapshot-source.py](snapshot-source.py) records the collection method.

Runtime preflight, correctness verdicts and performance measurements were
pending at collection. The development gateway returned HTTP 401 with the
stored SML key. Gateway checks and research qualification must pass before
the research rollout. Use a new status snapshot when collecting later evidence;
keep this launch audit unchanged.
