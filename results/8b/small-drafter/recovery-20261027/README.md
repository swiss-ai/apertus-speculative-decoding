# Shallow-drafter recovery, 27 October 2026

The experiment is not concluded. Five-layer control **3722176** completed
6,200 steps and saved validation EAL **4.63015594** (training accepted length
4.46567829). One-layer warm-start **3722177** never trained: Speculators
rejects `--from-pretrained` together with `--num-layers`. The launcher fix is
`70710111`; three focused regression cases, Ruff and shell syntax pass.

Replacement **3769226** is queued on normal: two nodes, six-hour allocation,
fresh run ID `depth-l1-warm-recovery-20261027`. It retains the old recipe:
6,200 optimizer steps, one-epoch horizon, 16,384 packed tokens and 1,024
anchors per rank, four trainer ranks, Muon learning rate 3e-4, original
prepared dataset and one-layer truncated-e10 initialization. The completed
control is not retrained, and the failed run is preserved.

Remote repository:
`/iopsstor/scratch/cscs/faruk_zahiragic/spec-exp/small-drafter`.
Data/checkpoint root:
`/iopsstor/scratch/cscs/faruk_zahiragic/spec-exp-data/small-drafter`.

## Recovered evidence

`evidence/recovery-20261027/collection-manifest.json` records 65 original
source-file SHA-256 hashes; every collected file was verified, decompressing
gzip logs before comparison. It retains control training/validation metadata,
full compressed trainer/extraction logs, original submit scripts, failed
one-layer logs, target contract, dataset metadata, warm-start lineage and
Speculators training/vLLM provenance. Large weights, optimizer state and
corpus remain on the cluster. Original epoch0_end/checkpoint_best are
symlinks to the retained checkpoint directory `0`.

The transport archive SHA-256 is
`752284a8e1eebb834d7fddb690ac67ff185a3509242178f5090fa6f0bd85e11f`.
`collect-evidence.py` is the exact collection script. Do not rerun it over
the completed collection; choose another version for later artifacts.

## Executable continuation

1. Check `clariden status 3769226 --json`, then the dedicated run's trainer
   log and checkpoint metadata. Slurm success alone is insufficient: require
   `global_step=6200`, completed validation and one-layer configuration.
2. Sync this branch to its dedicated remote checkout. The committed
   `evaluation-specs/depth-recovery-a.yaml` and `-b.yaml` use the trained
   one- and five-layer checkpoints, existing DSpark e10 at k=1..7, EAGLE e10
   at k=3/7 and plain. Both trained variants also cover k=1..7. Concurrency
   is 1/8/32/64/128/256 on the existing 128 held-out OPB prompts.
3. Once training is verified, submit these two specs through the existing
   launcher, with **normal** partition, a four-hour limit and
   `SWEEP_PATCHES=serving/patches/vllm-apertus-dspark-num-layers.patch`.
   Use a short unique `VLLM_PATCH_DIR` under the data root for each launch.
   Set `SML_ENV_DIR` under the data root as well. The first command arm in
   each allocation runs `depth_recovery.py preflight` on the compute node:
   verifies both completed checkpoints and captures their hashes and reference
   weights, workload, source revision and Slurm/node identity. A failed
   preflight invalidates the comparison. The two specs counterbalance arm
   order; actual GPU assignments are retained, not assumed to be swapped.
4. After both jobs finish, inspect all arm statuses and collect raw
   metadata, requests, summaries, metric snapshots, command/preflight logs,
   resolved environments, patch sources and hashes. Keep the training
   configuration/provenance linked to both checkpoints. Keep each new
   deployment in its own run directory and preserve failed attempts.
5. Run the deterministic collector on the completed sweep directories:

```bash
python3 methods/dspark/train/depth_recovery.py decide RUN_A RUN_B \
  --output NEW_ANALYSIS_DIRECTORY/decision.json
```

It refuses incomplete arms, any cell below 99% success, or nonpositive
throughput. G3 requires the one-layer candidate's best k to exceed the
**same-job** DSpark e10 best k by at least 5% at C=128 or 256 in both
counterbalanced repeats. The output retains the selected k, absolute
throughput, GPU-seconds/million tokens, acceptance and source hashes.
It does not turn a simulator ceiling or training EAL into a serving result.
Before recommending a positive candidate, finish the repository's calibrated
greedy-output checks with same-configuration controls. A training extension
is a new decision under the user's two-run reduction; do not start one
automatically.

Example submission from the dedicated remote repository, after step 1:

```bash
SWEEP_PARTITION=normal SWEEP_TIME=04:00:00 \
  SML_ENV_DIR=/iopsstor/scratch/cscs/faruk_zahiragic/spec-exp-data/small-drafter/env \
  VLLM_PATCH_DIR=/iopsstor/scratch/cscs/faruk_zahiragic/spec-exp-data/small-drafter/patch-depth-a \
  SWEEP_PATCHES=serving/patches/vllm-apertus-dspark-num-layers.patch \
  serving/node-sweep.sh results/8b/small-drafter/recovery-20261027/evaluation-specs/depth-recovery-a.yaml
```

Use a different patch directory and the `-b.yaml` spec for the second job.
Both completed comparisons, their analysis and a concise verdict remain to
be produced. No additional depth grid, EAGLE training or ongoing-task changes
are authorized by this recovery.
