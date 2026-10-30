Job **707140** is a fresh bounded 32-node numerical diagnostic using committed
runtime and launcher `556a464d`. It was submitted at 2026-10-30 10:54:54 UTC.
At 10:56 UTC it was pending priority, with a tentative 13:30:51 UTC start.

The original LR 8.48528137423857e-4, one-epoch 4,077-step horizon / 40 warmup
steps, 64 ranks, SEQ 8,192 and 512 anchors are preserved. The segment stops
at 640 steps, the allocation is six hours, and MAX_ATTEMPTS=1. Activation
checks, synchronized all-rank batch/RNG capture and a capture barrier are
enabled. This replaces job 707138, cancelled before allocation. No corrupted
weights are reused. Exact commands, source/configuration and submission
state are retained here. Compute provenance populates in the separate run.
Results remain pending.

`watch_diagnostic.py` runs in the foreground for job 707140 only. Its live
`--once` check passed. It uses the existing explicit chat queue for allocation
start, the first training step, numerical failure, presence of all 64 capture
files, terminal state and a three-hour heartbeat. It installs no cron entry
and changes no jobs. Authentication failure suspends SSH; transient transport
failures do not imply training failure. `observations.jsonl` and
`notifications.jsonl` remain live until it stops; final bytes/hashes will be
retained then.

`source-audit/` preserves the exact relevant Speculators and image math code
plus bounded conclusions. `one-node-reference-20261030T113306Z.json` records
an independent read-only health comparison: the untouched one-node run was
finite at step 29,637, with loss 0.441 and latest save 28,730. Its different
batch, LR and schedule prevent a causal comparison with the failing run.
