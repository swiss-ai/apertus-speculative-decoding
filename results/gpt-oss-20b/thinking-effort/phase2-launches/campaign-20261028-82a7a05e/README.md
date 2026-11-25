# Phase 2 native effort fine-tuning launch

The [launch record](launch-record.json) pins source, upstream, paths and checks.
The implementation is frozen at `82a7a05ece452bb6897ed09bce9caca736c3f54a` in an
isolated Clariden worktree. The Phase 1 checkout stays at `f409b9da`.

Remote campaign:
`/iopsstor/scratch/cscs/faruk_zahiragic/thinking-effort/phase2/campaign-20261028-82a7a05e`

Controller PID `129861` is running independently of this session. Its launch
snapshot is [controller-status-at-launch.json](controller-status-at-launch.json).
It is waiting for qualification job **3771335** on debug (30-minute slot).
That job installs the pinned upstream package in an isolated environment,
checks conversion preserves the released tensors, runs a synthetic backward
update and native export, then tests serving the exported model. The synthetic
head is a compatibility artifact and is never a research head.

After that gate succeeds, the controller submits data generation/feature checks,
then trains four matched heads concurrently, then submits three paired evaluation
blocks. It retries debug submission limits and verifies predecessor artifacts
before advancing. Failed stages stop with their logs intact. Identical controller
restarts reuse accepted jobs, and completed raw cells are never overwritten.

Training is frozen to 512 train prompts and 64 development prompts, half GSM8K
training and half MBPP training, all regenerated at native low/medium/high.
Normalized evaluation duplicates are excluded before generation. All four heads
start from the released DFlash initialization, keep the target/architecture
fixed, and each receive 1,572,864 assistant tokens over four epochs. Mixed data
has equal tokens from each effort. The upstream trainer selects checkpoints by
development loss, and the wrapper verifies the actual token budget.

Validation completed locally: 205 main tests passed (one optional integration
skipped in the lightweight environment), 18 upstream converter/target tests
passed in an isolated Torch 2.13 CPU / Transformers 5.17 environment, and the
optional dataset integration test passed there. Real prompt selection and the
upstream training CLI were also checked. Slurm job scripts passed bash syntax
checks. The GPU qualification, four research fine-tunes and Phase 2 measurements
were still pending at this launch snapshot; no training result is claimed.

Large feature caches, optimizer states and checkpoint exports remain on scratch
alongside the complete run provenance. Follow
[the pipeline documentation](../../../../../experiments/thinking-effort/README.md)
for stage behavior and artifact formats.

Recovery update: qualification job `3771335` failed after 18 seconds because its
Slurm steps inherited multiple tasks and collided while creating the environment.
No research fine-tuning or Phase 2 measurements started. The corrected attempt
uses explicit single-task allocations and steps. Replacement qualification job
`3782362` and a fresh controller/campaign are recorded in
[the recovery record](../campaign-20261028-d0862bd6/recovery-record.json).
