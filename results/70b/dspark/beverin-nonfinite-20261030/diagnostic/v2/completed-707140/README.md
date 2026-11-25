# Verified bounded diagnostic 707140

The fresh instrumented 32-node run **passed its 640-step bound**. It did not
reproduce the original NaN trigger, and does not establish a fix or full-epoch
stability. No further run, retry, cancellation or monitor was submitted.

The allocation ran **2026-10-30 16:42:17–20:31:35 UTC**, elapsed **3 h 49 min 18 s**,
and exited 0. There was one trainer attempt. The 640 metric records have
contiguous counters **0–639** and **12,800 finite training metric values**;
all recorded profile/LR values are also finite. Ten regular saves completed,
ending at state **epoch0/local640/global640**. Both saved schedulers have
last_epoch 640 and next LR **0.0007224154590849137**, matching the original
4,077-step horizon / 40-step warmup. No validation or resume was attempted.

| Logged counter | Original loss / CE / TV / accuracy | Diagnostic loss / CE / TV / accuracy |
|---|---|---|
| 519 | NaN / NaN / NaN / 0.484 | 0.559 / 1.201 / 0.233 / 0.493 |
| 639 | NaN / NaN / NaN / 0 | 0.539 / 1.128 / 0.222 / 0.510 |

All 32 trainer logs have two guard hook records: **64 total**, each enabled
with rocBLAS and FP32 reduction. Every node's resolved environment enables
activation checks and bad-batch capture. Installation flags, immutable code
and actual-image preflight evidence agree. Individual stage invocation
counts are not logged. No failure marker or bad-batch capture exists.

The final checkpoint sample read **11,073 deduplicated BF16 values across
all 62 saved tensors**, with **zero non-finite samples**. Only 22,146 tensor
data bytes were requested in contiguous windows. This is **not a full
saved-weight scan or optimizer-state scan**, and no full weight hash was
computed on the login host. The exact sampled indices/bits are retained.

Container and dataset metadata hashes, Speculators revision, model config
and normalized Speculators patch match the original. Trainer YAML differs
only in the output path. Both runs log **seed 42, deterministic_cuda=0**.
Only six of their 32 physical nodes overlap. The first logged training
metric difference is already counter 2: loss 1.376 versus 1.375. The new run
adds compiler-disabled hooks and synchronous numerical checks to a
compile-decorated forward path; online hidden-state extraction also remains
in use. These differences prevent a claim of exact trajectory reproduction
or an attribution of the passing result to a particular intervention.

There are **no numerical warnings**. Four hidden-state file-lock timeouts
and six connection aborts were retried successfully; every logged
error_records value is zero. All 32 extraction servers logged telemetry
thread home-quota errors at startup. Twenty-nine EngineDeadError lines
occur during explicit target-server SIGTERM cleanup at 21:31 local time,
after the final save. These are preserved and classified, rather than
being mistaken for a training numerical failure.

Diagnostic timing: **16.78 s mean / 16.5 s median** over 630 samples excluding
first-step compilation and post-save timing, with the last ten averaging
**16.23s**. Activation checks are enabled, so this is not a controlled
production throughput comparison.

The observer history has a **14h12m gap**, and completion was observed
**13h34m after the allocation ended**. Start and first-step notifications
were queued promptly. The gap's cause is not established; verification
made no monitoring changes. See monitoring-audit.json and the retained
observations/notification receipts in the parent directory.

`verification.json` records sixteen passing checks and comparisons.
`warning-audit.json` retains matched operational warnings. `raw/` contains
556 files / 58.25 MB of completed logs, deployment/checkpoint metadata,
immutable runtime source and original-run comparison artifacts, with every
source byte hash verified. Large model and optimizer files remain on
Beverin at the recorded paths. `quick-evidence/` preserves the independently
collected early evidence. Collection attempts are retained; the final
file collector and bounded checkpoint sampler are separate read-only
programs. SHA256SUMS covers this completed cell.
