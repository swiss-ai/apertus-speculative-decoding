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

## Automatic continuation

`methods/dspark/train/submit-depth-recovery.sh` queues an afterok chain from
training **3769226**: a two-hour validation/correctness allocation, then the
two original four-hour counterbalanced serving allocations. The downstream
jobs start only after their predecessor succeeds. `campaign.json` retains
the flat job list, dependency graph, current state and final decision path;
`events.jsonl` preserves state transitions. Job IDs are also recorded in the
coordinator's shallow-drafter receipt.

The campaign freezes executable sources at the submitted Git revision under
its own `source/` directory; weights, optimizer state and corpus stay in their
existing cluster locations. The source manifest verifies every frozen file
and the training launcher files against `70710111`. The original submitted
training script prints `70710111`; later documentation commits do not change
its executable training sources. The manifest distinguishes those before/after
source observations from the trainer's own captured argv and Speculators diff.
It preserves the original training submission and resolved environment.

Inside the allocated compute node, validation requires exactly 6,200 steps,
a one-layer config with the original five target auxiliary layers, finite
completed validation, and the original packed-token, anchor, optimizer, loss,
data and four-rank training recipe. It verifies the trainer's saved command
and exact Speculators revision/diff. Preflight hashes both trained checkpoints,
reference weights, workload, container and overlays, and checks the target's
weights, tokenizer, config and template against the recorded target contract.
Compact training metadata and compressed relevant logs are collected without
copying weights. A failed preflight starts no serving arms.

Before benchmarking, k7 of each trained variant receives two independent
deployments and two greedy captures per deployment. The smoke corpus is the
fixed first six original held-out OPB prompts, with a declared 384-token cap.
The repository's calibrated correctness gate uses both within-deployment and
cross-deployment controls of plain and the matching speculative configuration.
Its existing margins stay unchanged (prefix/edit 0.05; token difference 2).
The controls' worst cases remain visible; a pass is limited to this smoke set.

Both benchmark specs retain the full original 128-prompt held-out passes,
k=1..7 for trained one-/five-layer DSpark and existing DSpark e10, existing
EAGLE e10 k=3/7, plain, and C=1/8/32/64/128/256. They counterbalance arm order;
actual GPU assignments are retained. Every measured deployment also receives
two smoke captures. Final analysis checks every configuration against its own
same-configuration controls across the two benchmark allocations, including
the exact k values used by the throughput decision. Explicit deployment IDs
prevent a reused localhost URL from masquerading as a same-deployment control.

Final collection rejects incomplete arms, cells below 99% success and invalid
throughput. G3 requires the one-layer candidate's best k to exceed the same-job
DSpark e10 best k by at least 5% at C=128 or 256 in both repeats. The decision
retains absolute throughput, GPU-seconds/million tokens, selected k, acceptance
and source hashes. With valid correctness and throughput measurements, a miss
closes as `negative_complete`; a positive is limited to the declared workload.
Infrastructure or correctness failures stop the dependency chain and remain
unresolved, without a scientific verdict. Earlier attempts and raw cells are
preserved. No training extension or additional depth grid is launched.

Campaign root:
`/iopsstor/scratch/cscs/faruk_zahiragic/spec-exp-data/small-drafter/campaigns/depth-recovery-20261027-auto-v1`.
The final artifact will be `b/decision.json`; the precise deployment roots are
`validate/repeat-a`, `validate/repeat-b`, `a/sweep` and `b/sweep`.
