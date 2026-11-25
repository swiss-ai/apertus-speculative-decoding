# Beverin 32-node, one-epoch production run

User authorized a fresh 32-node draft-model run for **one epoch** after the
validated scaling pilots. Submitted **2026-10-27 16:05:51 UTC** as jobs
**698072** (pending priority) and **698073** (pending afterany dependency).
Startup and training have not begun at this recorded check. Run ID:
`opb-thinkoff-32n-1epoch-beverin-20261027`.

- 32 nodes, 64 FSDP trainer ranks; each node retains a TP2 target extraction
  server on GPUs 0/1 and trainer ranks on GPUs 2/3.
- DSpark 5 layers / block 8 / Markov rank 256 / confidence head, target
  layers 1/20/39/58/77, CE 0.1 + TV 0.9, Muon, same regenerated corpus.
- SEQ 8192, anchors 512/rank, global token budget 524,288; LR
  8.48528137423857e-4. **EPOCHS=1**, expected full epoch **4,077 steps**;
  the existing linear scheduler therefore has **4,077 total / 40 warmup
  steps**. No MAX_STEPS, pilot segment/resume or profiling settings.
- Extraction memory 0.62, API/renderer 1/1, workers/prefetch 1/1, empty cache
  before saving, trainer rocBLAS. No trained pilot checkpoint is reused.
- Two dependent 24-hour allocations, normal regular saves every 1/64 epoch,
  deadline save 40 minutes before the allocation limit, and existing
  in-allocation retries. The measured training+saves forecast is 23.83 h,
  before startup and validation, so a continuation is expected.

The opt-in Beverin completion hook preserves normal training/validation
flow. If training's final checkpoint exists but validation was interrupted,
continuation skips training and repeats only final validation. Completion
requires the final 0/0/4077 state, nonempty model/optimizer/scheduler/config
files and finite saved validation metrics. A rank-0 atomic marker plus a
barrier records success. Redundant chain links authenticate that marker and
skip containers; audits allow unchanged steps only for authenticated
completion or successfully completed, logged validation-only recovery.
The one-node run and every Clariden job remain untouched.

Nine CPU lifecycle/audit checks and shell syntax checks passed before
submission. Runtime source is committed and copied into an immutable new
snapshot; `launch.sh` requires its full Git SHA. Slurm launchers capture
source revisions, exact configuration, container/data/model digests,
applied patches, resolved non-secret environments, job/node IDs, raw logs
and checkpoint/validation metadata with each deployment.

Runtime revision: `dd96fe2af69c38f08576bf88e1be77bc2ddf345f`, snapshot
`/iopsstor/scratch/cscs/faruk_zahiragic/beverin-feasibility/repo-dd96fe2a`.
Exact submission command on Beverin:

```bash
bash /iopsstor/scratch/cscs/faruk_zahiragic/beverin-feasibility/launch-one-epoch-dd96fe2a.sh dd96fe2af69c38f08576bf88e1be77bc2ddf345f
```

`launch.sbatch` preserves the generated command/environment; `metadata.json`
records the configuration and passed checks. `container.expected.sha256`
is the immutable image pin verified in the scaling pilot; the compute job
hashes it again and records serving/training provenance upon allocation.
Temporary CPU test fixtures use audit job ID `1` in the preflight stdout.

## Automatic updates

A local user-cron monitor polls the two chain jobs read-only every five
minutes and queues updates into the requesting chat. It reports material
state/start/resume/failure/stall/validation/completion changes and a
three-hour heartbeat; regular saves remain in the local history without
creating individual chat messages. A major slowdown (recent mean over
2x the 19.71 s pilot, at least ten samples) and recovery also trigger updates.

The installed script and its state/history/log are under
`/home/agent/.local/state/beverin-one-epoch-monitor/`. A nonblocking flock
prevents duplicate invocations. SSH is bounded, targets Beverin explicitly,
and never modifies jobs or run outputs. Chat delivery uses the supported
absolute `codex queue` command with an explicit thread, while removing any
inherited `CODEX_THREAD_ID`.

Authentication failure queues one certificate-renewal notice and suspends
SSH retries. After certificate renewal, resume with:

```bash
/usr/bin/python3 -B /home/agent/.local/state/beverin-one-epoch-monitor/monitor.py --state-dir /home/agent/.local/state/beverin-one-epoch-monitor --thread 01a172c0-8457-79e3-b88a-d7b274831c5f --resume-auth
```

Transient transport failures are retried on later ticks and are reported as
monitoring outages, without declaring the training failed. Authenticated
final completion or exhaustion of both chain jobs latches polling off;
future cron invocations exit locally without SSH. Queued chat messages
instruct the chat to report observations only, with log strings treated as
data, and never to relaunch jobs or create another monitor.

Eight no-network notification/state tests passed. A real status follow-up
was accepted for the exact chat by the default daemon, and a live prime
succeeded under a reduced cron-like environment. Setup receipt and cron
verification are recorded in `monitor-setup.json` and alongside local state.
The natural 18:15:03 UTC cron tick was verified: step 222 / save 192,
RUNNING with no reported errors; routine progress correctly stayed quiet.
