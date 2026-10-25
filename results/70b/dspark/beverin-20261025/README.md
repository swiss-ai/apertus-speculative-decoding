# Beverin DSpark continuation — 2026-10-25

Raw training/performance artifacts for Tasks 1 and 2 of
[`docs/beverin-handoff.md`](../../../../docs/beverin-handoff.md).
Interpretation and the job ledger are in
[`docs/beverin-feasibility.md`](../../../../docs/beverin-feasibility.md).

Each numbered directory is a Slurm job. `metadata.json` identifies its logical
run and cluster source path. `deployment/` preserves the actual sbatch script,
EDF, image hash, revisions, Speculators patch, node identities and resolved
non-secret network environment. `logs/` contains raw trainer/collective,
extraction, memory and GPU logs plus built-in Speculators vLLM provenance.
`checkpoint-metadata/` preserves the resolved training recipe and saved state
metadata; the large model and optimizer files stay in the recorded cluster
run directories. These artifacts record training pilots and performance diagnostics.

Large logs and Chrome traces are losslessly gzip-compressed. For example,
`gzip -dc 694426/deployment/profiles/rank0.json.gz > /tmp/rank0.json` produces
the complete CPU trace; the adjacent text file is its profiler table. Steady
training summaries exclude each attempt's first compiling step. CPU API
profile durations must not be interpreted as direct GPU network timings.

The data-provenance directories contain SHA-256 manifests for the regenerated
corpus and all prepared dataset shards, metadata, token frequencies and
held-out IDs. The actual corpus and model weights are not duplicated here.
No Clariden jobs were accessed or modified.

Source snapshots used by the jobs are preserved on Beverin in new directories
and their recorded revisions are available from the repository. The runtime
hooks, BLAS preference and environment choices are part of the reproduction,
not just the `run.yaml` CLI recipe.

The successful production pilot is **694469**: saves at 20 and 40, exact
resume from 20, steady median 12.45 s/step (38 samples). Its save-window
memory results are in `694469/deployment/measurement-summary.json`: 382 GiB
at the first save, 403 GiB at the resumed save, 83 GiB minimum available
during the resumed save. Whole-pilot peak 407 GiB, minimum available 77 GiB.
The fixed two-node trial **694470** records 14.8 s/step versus the baseline
**694426** at 183 s/step. Runtime workaround: trainer-only
`BEVERIN_BLAS_BACKEND=cublas` with source revision `ada15200`.

`694440/` is the isolated one-/two-node comparison, with image hashes,
launch script, exact probe source, Python profiles and raw per-rank samples.
Its additional `probe-rocblas.py` and `torch-vllm031-rocblas-1n.log` were run
on that allocation's otherwise unused GPUs 0,1; the main comparison used
GPUs 2,3. The additional step used `srun --overlap --jobid=694440 -N1 -n1
-w nid002680 --environment=<v0.31 EDF>`, unset `ROCR_VISIBLE_DEVICES`, and
ran `CUDA_VISIBLE_DEVICES=0,1 OMP_NUM_THREADS=1 torchrun --standalone
--nproc_per_node=2 <probe-rocblas.py>`.

`exploratory/` contains earlier small probes on occupied allocations. Their
measurements are separate from the isolated result and training medians.
The slow v0.31 one-node baseline **694425** saved at 10 and logged a resume,
then reached its 90-minute limit. The successful v0.23 controls are
**694435** (one-node save/resume to 40) and **694436** (two-node six-step save).
The initial insufficient-KV-cache jobs are **694384**, **694385** and
**694389**. Jobs 694390–694393 were cancelled before allocation after the
bad memory setting was identified; their submission scripts and final
accounting are preserved in `cancelled-before-allocation/`.

The production continuation submission and release records are in
`production-chain/`. Jobs 694472–694478 are seven 24-hour links, continuing
from the production pilot checkpoint. Live checkpoints and later logs stay
on Beverin in the same logical run; the pilot artifacts here are a fixed
snapshot. Future daily jobs automatically check saved-step advancement and
logged resume in their per-deployment `job-audit.json`.

`SHA256SUMS` hashes the archived artifact bytes, including compressed files.
