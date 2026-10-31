# Shallow DSpark recovery: correctness blocked

The bounded recovery is **closed as correctness blocked**, with no implementation
or serving-speed endorsement. Cache-safe validation **3804151** failed after
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

**Evidence retrieval limitation:** all 12 treatment-report filenames were
listed, but individual small JSON reads stalled in Lustre `cl_sync_io_wait`.
Exact final per-model metrics, the e10 reference outcome, plain calibration and
actual GPU status records could not yet be retrieved. SIGKILL was issued to the
two own readers; they were waiting in kernel I/O. Do not attribute the original
3790216 metrics or its exact plain controls to this counterbalanced attempt.
The top-level calibrated failure is confirmed independently by the campaign
state and Slurm accounting. A common serving-path cause remains unresolved.

No valid trained-model throughput grid exists, so neither the original G3
5% threshold nor the stricter repeatable fixed-k 10% throughput-plus-latency
implementation screen was evaluated. There are no defensible absolute
time/memory/cost savings to recommend. Keep shallow drafting as a profiled cost
idea, with the acceptance loss and serving correctness block attached; do not
implement it based on this experiment. This result does not prove a speed loss.

The machine-readable [closure](closure.json) and [job states](job-final-states.txt)
record the distinction and preserved identities. Prior compact training and
validation evidence is in [recovery-20261030](../recovery-20261030/README.md).
Latest raw gate reports, captures, statuses, logs and runtime provenance remain
under the remote campaign below. Its original failed operational state is
preserved in [campaign-before-closure.json](campaign-before-closure.json). A new
remote closure file and terminal campaign annotation are written; only final
compact gate/status/log collection awaits storage recovery. No new scientific
run is needed to retrieve those existing artifacts.

`/iopsstor/scratch/cscs/faruk_zahiragic/spec-exp-data/small-drafter/campaigns/depth-recovery-20261030-gpu-calibration-v2`
