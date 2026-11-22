# Pilot launch recovery

Snapshot at 2026-10-29T09:54:45.943223+00:00.

The original DSpark job 3783300 passed runtime preflight on Clariden node `nid007655`, then failed before serving: the pinned vLLM image rejected `--disable-log-requests`. Commit `c18ea9c0` uses its supported `--no-enable-log-requests` flag. All 244 repository tests and Ruff passed.

The two queued plain jobs contained the same obsolete flag in Slurm's frozen batch scripts. They were cancelled only after replacements had been rendered and checked. Original run artifacts are preserved. New independent run directories use the existing final DSpark e10 head and unchanged target/image/engine settings.

Replacements: DSpark **3790128** (`debug`), baseline **3790129** (`normal`), control **3790130** (`normal`). All were pending for priority at this snapshot. Each requests one exclusive four-GPU node for 90 minutes; one engine uses one GPU. Gateway-enabled validation controllers wait up to 36 hours; the control stops after smoke checks. No live completion, correctness gate, performance comparison or qualification has passed yet. The dev key was previously accepted by the gateway.

Compute and results are on CSCS Clariden under `faruk_zahiragic`, in `/iopsstor/scratch/cscs/faruk_zahiragic/apertus-pilot-code/results/8b/serving-pilot/`. Code is on `feat/dspark-research-serving-pilot`. Controllers run on the login node; GPU serving and benchmark load run through Slurm.

See `snapshot.json` for Slurm states and paths, and `operation/` for the exact recovery script and replacement journal. This is a historical snapshot; check live Slurm and controller reports before restarting any run.
