# Shallow DSpark recovery: correctness blocked

The bounded recovery is **closed as correctness blocked**, with no implementation
or serving-speed endorsement. Evidence retrieved on 2026-11-04 shows the same
gate also rejects the deployed DSpark e10 reference, so the block is shared and
not specific to shallow drafting (see below). Cache-safe validation **3804151** failed after
496 seconds on `nid007156` with `CorrectnessGateFailed: calibrated correctness
exceeded the existing protocol margins`. This attempt reached the scientific
gate; the earlier FlashInfer home-quota startup failure is a separate attempt.
Only its unallocated, dead-dependent benchmark jobs **3804152/3804153** were
canceled. No further training, validation or benchmark runs are being launched.

The motivation was to reduce draft cost under load by training one layer instead
of five. The one-layer warm-start run **3790212** completed all **6,200 steps**
in 8,703 seconds; its held-out trainer EAL was **3.520**, versus **4.630** for
the completed five-layer control **3722176**. Their trainer accepted-length
metrics were **3.352 versus 4.466**. These training quantities do not measure
served acceptance, throughput, latency or memory savings. Both checkpoints,
the original recipe and all earlier attempts remain preserved.

The final diagnostic kept the same six held-out prompts, 384-token cap, pinned
image, checkpoints, engine flags and original prefix/edit/token margins. Frozen
source `296ed4f0` requested GPU roles plain/e10/L1/L5 = 0/1/2/3 in A and
3/2/1/0 in B. It included existing DSpark e10 k7 as a diagnostic reference;
each treatment's calibration used only plain and its own configuration.

**Retrieved gate evidence (2026-11-04).** Cluster storage recovered and the
saved reports are now in [evidence-3804151](evidence-3804151/) (hash-verified
against the remote copies; see its `retrieval-manifest.json`; served captures
and prompt text stay on the cluster). Actual GPU roles match the configured
counterbalance: plain/e10/L1/L5 = 0/1/2/3 in A and 3/2/1/0 in B
(`validate/repeat-*/*/status.json`). All four plain-vs-plain controls (same and
cross deployment) were exact 6/6. **All 12 treatment reports failed**, including
the existing deployed DSpark e10 k7 reference:

| arm (vs plain) | repeat | exact | min prefix frac | max edit dist | max token diff | failed checks | same-arm A-vs-B exact |
|---|---|---|---|---|---|---|---|
| DSpark e10 k7 (reference) | A / B | 4/6 / 4/6 | 0.183 / 0.183 | 0.502 / 0.506 | 0 / 0 | prefix, edit | 4/6 |
| depth L1 k7 | A / B | 2/6 / 4/6 | 0.183 / 0.183 | 0.506 / 0.502 | 2 / 0 | prefix, edit | 2/6 |
| depth L5 k7 (control) | A / B | 2/6 / 2/6 | 0.036 / 0.183 | 0.514 / 0.466 | 27 / 22 | edit, token diff | 2/6 |

Both captures within each deployment were identical, so each cell is one value
per repeat. Allowances (control envelope plus the unchanged 0.05/0.05/2
margins) were prefix >= 0.350/0.348/0.000 and edit <= 0.353/0.465/0.294 for
e10/L1/L5. Every speculative arm was exact within a deployment but not across
the two GPU-reversed deployments; plain was exact everywhere. e10 and L1 share
the same worst divergence point (137 common characters, fraction 0.183).

**What this changes.** The 2026-10-31 closure could not say whether the gate
failure was specific to the shallow drafter. It is not: the calibrated gate
rejects the already-deployed e10 reference with the same worst-case prefix and
edit distance as the one-layer drafter. The correctness block is therefore a
shared speculative-serving or gate-calibration issue, not evidence against
shallow drafting. The five-layer control diverged more (27-token length
difference), and nothing here separates L1 from e10. The closure verdict still
stands (no implementation endorsement, no further runs). The reason is now the
trainer acceptance loss plus the absence of any throughput measurement, not a
drafter-specific correctness defect. A shared-path investigation should start
from e10 under this harness, not from the shallow drafter.

No valid trained-model throughput grid exists, so neither the original G3
5% threshold nor the stricter repeatable fixed-k 10% throughput-plus-latency
implementation screen was evaluated. There are no defensible absolute
time/memory/cost savings to recommend. Keep shallow drafting as a profiled cost
idea, with the acceptance loss attached and the shared serving correctness block noted; do not
implement it based on this experiment. This result does not prove a speed loss.

The machine-readable [closure](closure.json) and [job states](job-final-states.txt)
record the distinction and preserved identities. Prior compact training and
validation evidence is in [recovery-20261030](../recovery-20261030/README.md).
Compact gate reports, statuses, logs and runtime provenance are in
[evidence-3804151](evidence-3804151/), with a per-arm digest in
[correctness-summary.json](evidence-3804151/correctness-summary.json). The 16 raw
captures remain under the remote campaign below. Its original failed operational
state is preserved in [campaign-before-closure.json](campaign-before-closure.json).

`/iopsstor/scratch/cscs/faruk_zahiragic/spec-exp-data/small-drafter/campaigns/depth-recovery-20261030-gpu-calibration-v2`
