# GPT-OSS native reasoning effort campaign

The implementation is at source revision `f409b9dab1d776817dd283488db8c73e8927e690`.
The [launch record](launch-record.json) preserves earlier integration failures and
submission commands. The [post-preflight snapshot](post-preflight-status.json)
records the successful preflight and subsequent queue state; it is a snapshot,
not a live status page.

Both plain and fixed-head DFlash deployments passed all 64 prompts at native
low/medium/high effort: 192 render checks per deployment. Each also completed 12
extra generation checks with analysis, a nonempty final answer and natural
termination. DFlash verification counters were observed. This is compatibility
validation, not an effort-comparison result. Local validation: 193 tests passed,
Ruff and whitespace checks passed.

The full remote campaign is at:

`/users/faruk_zahiragic/apertus-spec-decoding/results/gpt-oss-20b/thinking-effort/campaign-20261027-f409b9da`

Preparation job `3768739` completed successfully. Phase 1 blocks 0 and 1 were
submitted as `3768740` and `3768791` on debug with a 90-minute allocation each.
Controller PID `133383` retries admission for block 2, preserving each submission
attempt and job ID in the remote campaign. All blocks require a successful
preparation dependency. A failed block prevents a proceed decision.

Keep the remote source checkout at the recorded revision while these jobs are
queued/running. Do not sync a new HEAD into this live campaign. Each block checks
its immutable source/model/container/corpus identities.

The remote campaign retains complete inputs (including the offline vocabulary),
rendered token IDs, logs, source overlay, environment files and submission records.
This local directory contains the input lock, frozen spec, runtime manifests,
preflight outputs and serving provenance; the raw preflight logs are also copied
locally. Model weights and compile caches remain on Clariden.

Specialist training is conditional on a complete Phase 1 proceed decision and a
compatible upstream GPT-OSS DFlash training/export recipe. The implementation
provides the gated training handoff and validates/evaluates the four exported
heads; it does not introduce a new training framework.

Update at 2026-10-27 18:00 UTC: blocks 0 and 1 completed with all 12 cells valid
(768 measured requests, no output-cap truncation). The original controller stopped
on a Slurm `Job dependency problem` before submitting block 2. After confirming
successful preflight, unchanged source and no existing block 2 deployment, only
that missing block was resubmitted as job `3769167` on debug; its recovery argv
and checks are preserved under `submissions/phase1-block-2/`.
The [interim report](reports/phase1-20261027T172238778034Z/comparison.md) remains
pending until the third repeat completes. Specialist training has not started.

Phase 1 final update: recovery job `3769167` completed successfully in 25 minutes.
All three repeats and all 18 cells are complete and valid (1,152 measured requests,
no output-cap truncation). The [final Phase 1 report](reports/phase1-20261027T185227817834Z/comparison.md)
passes the proceed gate with no identity/validity problems. Relative to medium,
accepted draft tokens per round increased 28–35% at low effort and decreased
15–18% at high effort across the three repeats. The shared drafter improved
latency over plain serving at every effort in every repeat. This supports running
the conditional specialist-versus-mixed comparison; it does not establish that
specialist heads are required. Phase 2 training has not started.
