# Shallow-drafter recovery, 29 October 2026

Job **3769226** failed before its first optimizer step. Four extraction servers
became healthy, then all trainer ranks raised `PermissionError` while loading
frozen verifier embedding/head weights. The one-layer initializer's complete
config embeds Yu's private target path ending in snapshot
`a411d838600baf0e3635a3daf66fb7c55fc97bb6`; this Speculators revision does not
apply the CLI verifier path when loading a full pretrained drafter. No training
checkpoint exists. This is an infrastructure failure, not a negative result.
The completed five-layer control is retained without rerunning it.

`failure-3769226.tar.gz` retains all 29 compact source artifacts, including full
compressed trainer/extraction logs, saved training command/config/Speculators
diff, extraction provenance and submission. Every source SHA-256 was verified.
Archive SHA-256:
`578dbc03793c40e00451b29e433de3dc8536330d716869e3adce134b7750cd6b`.
The original files remain on the cluster. Downstream jobs 3769355/56/57 were
canceled after their training dependency failed; the prior frozen campaign
and strict-review helper stay intact.

The repair creates a fresh initializer that links the **same physical drafter
weight file** and changes only `speculators_config.verifier.name_or_path` to
the shared snapshot already recorded by the target contract. The original
private path's revision and architecture must match that contract. Before
extraction or training starts, an allocated pre-training gate verifies the
one-field config change, the original truncation's drafter hash, target config,
tokenizer, template and all target weight hashes, plus the container digest.
Full hashes are computed in the allocation. The expected drafter hash is
`ea50bb1c33864929c79e2384cadaaf349a1c8c05473fc7fce5ac4477ddc72bb7`.

The replacement retains exactly 6200 steps, the original prepared data, packed
16,384 tokens and 1,024 anchors per rank, four ranks over two nodes, Muon 3e-4,
CE 0.1/TV 0.9 and the one-epoch horizon. It uses fresh training/campaign directories,
a private extraction overlay and resolved environment. The continuation now
binds paths to its exact submitted training script instead of a historical
hardcoded run. Original G3 and calibrated correctness gates remain unchanged.
After a/b, the fixed-k strict review runs automatically on CPU: repeatable >=10%
throughput plus >=10% mean TPOT or request-latency benefit, with the same metric
in both repeats and explicit TTFT/tail tradeoffs. A 5% G3 pass alone is not an
implementation endorsement. No depth grid or training extension is added.

Active chain: **3790212 → 3790216 → 3790218 → 3790219**, submitted from
branch `small-drafter`, revision `fb8f5c79`. At the submission snapshot, training
is pending Priority and the three downstream allocations are pending Dependency.
The 28 focused recovery tests, Ruff and generated-script syntax checks pass.
All 66 frozen source files and seven training source files match their recorded
hashes. The strict review helper retains SHA-256
`7d9157c1ddb8d2fbeedcd3b0cc649df8ac4bc2882f7882f20763c11feae85ac2`.

The private snapshot suffix and target contract both identify revision
`a411d838600baf0e3635a3daf66fb7c55fc97bb6`. The shared target config hash was
checked during preparation and equals the contract:
`391963f8b992c8445f68123306c8fb4aff1df62ddb5477989efd5cd2d491b686`.
Actual full target/tokenizer/drafter/container hashes remain pending the
allocated gate; this is not a claim that they have already passed.

Cluster root:
`/iopsstor/scratch/cscs/faruk_zahiragic/spec-exp-data/small-drafter`.
Initializer: `drafters/dspark-e10-trunc-l1-shared-verifier-20261029/`.
Training: `runs/depth-l1-warm-recovery-20261029/`.
Campaign: `campaigns/depth-recovery-20261029-auto-v1/`.
The pre-training gate writes
`runs/depth-l1-warm-recovery-20261029/provenance/warm-start-3790212.json`.
Final raw science: `b/decision.json` under the campaign; final implementation
review: `strict-review/final-review.json` and `.md`. Both are automatic and
write-once. Infrastructure or correctness failures stop the chain without
becoming a negative scientific result.

`submission-3790212.tar.gz` preserves exact scripts, environments, source manifest,
initializer configs/rebinding report, target contract and queue/identity proof.
Archive SHA-256:
`f13e1b3258595c09917ee14fffdeddf409d80be2b9fc5a9e8eb7e4419f117e59`.
