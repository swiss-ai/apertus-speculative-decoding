# Native reasoning effort experiment

This implements the [execution plan](../../docs/thinking-effort-execution-plan.md):
GPT-OSS-20B, native low/medium/high effort, one fixed DFlash head, and a plain
baseline at each effort. Three independent paired deployment blocks run one GPU
at concurrency 1. The 64-prompt probe produces 18 cells and 1,152 measured requests.

Preview the schedule locally without loading models or submitting jobs:

```bash
PYTHONPATH=src python3 -m apertus_bench.effort plan
```

## Run on Clariden

The [spec](gpt-oss-20b.yaml) pins the public drafter revision and references the
existing target and source probe on the cluster. The [EDF environment](env.toml)
uses vLLM 0.30.0 ARM64. The target stays in its released MXFP4 format. The preparation
job downloads the head, checks runtime imports, hashes the image/model artifacts,
and tests both servers on four extra prompts at each effort. It also stages the
Harmony vocabulary in a persistent offline cache, verifies its published SHA-256,
and retains a copy with the campaign inputs. Native rendering loads this cache
before model initialization, so a missing vocabulary fails immediately.

After syncing committed source to the cluster, choose a new absolute campaign
directory for each independent attempt:

```bash
RUN=/users/faruk_zahiragic/apertus-spec-decoding/results/gpt-oss-20b/thinking-effort/campaign-UNIQUE
bash serving/thinking-effort.sh --output "$RUN" --prepare --partition debug --time 01:30:00
# Use the returned preparation job ID. These jobs run only after it succeeds.
bash serving/thinking-effort.sh --output "$RUN" --dependency PREPARE_JOB_ID
```

If normal nodes are busy, `--partition debug --time 01:30:00` also applies to
measurement blocks that fit the debug time limit. The debug QoS admits only two
submitted jobs per user. Add `--supervise` to retry that admission limit; run the
submitter under `nohup` for unattended submission. Accepted job IDs and every
submission attempt are saved, and restarting the identical command reuses job
IDs instead of submitting duplicates. A failed preparation dependency stops the
submitter. The launcher selects the repository or existing cluster benchmark
virtualenv; `EFFORT_PYTHON` can override the login-node Python explicitly.

Without `--dependency`, the launcher requires a completed preparation/preflight
in that campaign. `--blocks 0` selects one block; the default submits all three.
`--validate-only` writes inspectable job/environment files without calling sbatch;
use a separate output directory for that dry run. It does not validate GPU support.
Failed attempts stay intact. A new attempt gets a new campaign directory.

GPT-OSS uses vLLM's native Harmony renderer. The recorded
`apertus_bench.effort_server` overlay pins its system-message date and adds a
read-only validation endpoint through `NativeRenderMiddleware`. Prompt validation
calls `/v1/chat/completions/render`, which invokes the running chat handler's
native renderer, and verifies the effort and context length for every prompt.
The normal `/tokenize` route is not used to infer GPT-OSS chat rendering.

Both arms receive the same explicit sampling parameters, disabled prefix caching
and disabled asynchronous scheduling. Each deployment records its launch command,
model/image hashes, package versions, source revision/patch, GPU/node/job identity,
and the Harmony overlay. Raw streams and per-request before/after counters are
saved alongside the canonical metadata, requests and summaries.

Analysis/final text is retained separately. Per-channel token counts are tokenizer
re-encodings, explicitly labelled as such; they exclude Harmony control tokens.
Server-reported reasoning usage is also retained. Acceptance is measured from
counters, without pretending that stream chunks identify accepted tokens by phase.
Request timing excludes counter scrapes. Errors, missing counters, missing final
answers, counter resets or excessive output truncation invalidate a comparison.

## Inspect results and the conditional gate

Each successful block writes a new report automatically. Reports can also be
generated locally from collected artifacts:

```bash
PYTHONPATH=src python3 -m apertus_bench.effort report "$RUN"
```

The JSON and Markdown report preserves every block and math/code stratum. Every
speedup uses the plain deployment in the same block and effort. A missing,
invalid or unmatched cell prevents a proceed decision. Three consistent blocks
with at least 10% relative acceptance sensitivity, or a consistent win/loss
crossover between efforts, enable the conditional specialization comparison.
Otherwise the bounded experiment stops with the shared head.

## Conditional training and Phase 2

The implementation does not manufacture an upstream DFlash training interface.
If Phase 1 proceeds, use an existing GPT-OSS-compatible DFlash training/export
recipe. Missing training/export support is the prerequisite failure specified in
the execution plan. This is separate from serving support.

Prepare a training handoff with explicit, disjoint train/development prompt files,
a fixed per-head supervised-token budget, and a JSON optimizer schedule:

```bash
PYTHONPATH=src python3 -m apertus_bench.effort training-plan "$RUN" \
  --output "$RUN/training-handoff" --train-prompts TRAIN.jsonl \
  --development-prompts DEV.jsonl --supervised-tokens TOKEN_BUDGET \
  --optimizer OPTIMIZER.json
```

This command refuses to run unless the complete Phase 1 decision is `proceed`.
It checks prompt-ID/text disjointness and emits the four matched fine-tuning
specifications. Regenerate responses from the frozen target at each native effort,
preserving analysis, final output and native formatting. The mixed head balances
supervised tokens across all three efforts. All four heads start from the same
released weights with identical per-head budgets and checkpoint-selection rules.

Each trainer/exporter must write the `required_export_provenance` schema listed
in `training-plan.json`, including the actual export's model-manifest hash, the
matched budget/split/initialization identities, and its effort role. The manifest
hash is computed by `apertus_bench.effort.model_manifest(Path(CHECKPOINT))`.
Supply a `heads.json` mapping `low`, `medium`, `high` and `mixed` to
`{"path": "/checkpoint", "provenance": "/training/provenance.json"}`. Keep all
training commands/configurations, histories, corpus hashes, reports and patches.
For Speculators launches always pass `--provenance-dir`.

```bash
bash serving/thinking-effort.sh --output "$RUN" --phase phase2 --heads HEADS.json
PYTHONPATH=src python3 -m apertus_bench.effort report "$RUN" --phase phase2
```

Phase 2 refuses mismatched or duplicate checkpoints and cannot start before the
Phase 1 gate. It adds 15 serving deployments and 2,880 measured requests. The
matrix compares all four heads at all three efforts, with fresh paired baselines.
The specialist decision requires a consistent 10% latency reduction versus the
mixed head and lower decode cost per token. Adapter serving, routing, offloading,
and Apertus integration are not part of this implementation.

## Executable Phase 2 pipeline

The [Phase 2 configuration](phase2.yaml) now freezes four matched fine-tunes of
the released head through the existing Speculators trainer at commit
`e3041ddb4e58d8d0fad86f938f44423bdf424ae9`. Its converter, DFlash loss, backward
pass, optimizer, validation and checkpoint selector are reused. Native export
keeps the released configuration and tensor names for the already-validated vLLM
runtime. The wrapper records any source/conversion checks alongside the run.

The training set has 512 task prompts and development has 64, split equally
between pinned GSM8K **training** and MBPP **training** subsets. IDs are split
before generation, with normalized task duplicates excluded against the 64
evaluation prompts. Every task is regenerated at all three native efforts, with
the Phase 1 target, date, sampling and termination settings. The run retains raw
responses, analysis/final text and actual native prompt/completion token IDs.
Tasks are retained at all three efforts together, requiring natural completion
and at most 8,192 tokens for training. All exclusions are reported.

Each head sees exactly **1,572,864 supervised assistant tokens** across four
epochs; prompt and padding tokens are excluded. A bounded corpus can recur to
fill that quota. The mixed head receives exactly one third from each effort.
The actual token count is measured in upstream training calls and must match
the declared budget. AdamW uses constant LR `1e-5`, weight decay `0.01`, seed 0,
FP32 weights with BF16 compute, CE against the upstream teacher's argmax labels,
and the original fixed exponential position weighting (gamma 4). This is the
existing Speculators DFlash recipe; no new loss or architecture is introduced.
Each head is selected by its lowest development loss, never by evaluation data.

Before large data generation, three real native prompts test the hidden-state
extraction path. Extracted last-layer features plus the frozen target norm/head
must reproduce vLLM's greedy next token. The five draft feature layers, tensor
shape, native IDs and finite values are also checked. Feature caches and all
corpus files are hashed before training. Shared embeddings, verifier head and
verifier norm stay frozen.

Use a **new Phase 2 campaign and isolated source worktree**. Keep large feature
caches, optimizer states and checkpoints on scratch storage. Reference the old
campaign with `--phase1-reference`; the GPU jobs revalidate its completed proceed
decision and enforce identical target, initialization, corpus and serving settings.
Code revisions may differ between phases, and remain frozen within each phase.

```bash
PYTHONPATH=src python3 -m apertus_bench.effort_training_submit \
  --output /iopsstor/scratch/cscs/USER/thinking-effort/phase2/UNIQUE \
  --phase1-reference /absolute/phase1/campaign \
  --upstream /absolute/pinned/speculators \
  --qualification /absolute/completed/prerequisite/attempt \
  --qualification-job SLURM_JOB_ID --partition debug --time 01:30:00
```

Run this lightweight controller under `nohup` for unattended execution. It waits
for the GPU training/export/serving qualification, submits data generation and
feature extraction, then trains the four heads concurrently on the four GPUs
of one node. After validated exports it submits all three paired evaluation
blocks, retrying debug admission limits. Stages are submitted only after a
confirmed predecessor completion; expired Slurm dependency IDs are not reused.
Accepted job IDs are preserved and reused on an identical controller restart.
Failures retain artifacts and stop the pipeline. A completed measurement or
training directory is never overwritten.

`bootstrap-phase2.py` creates an isolated training environment over the serving
image, preserving its Torch/Transformers/vLLM versions and installing pinned
Speculators package metadata/dependencies. It records install commands, logs and
resolved packages. Evaluation uses the original serving image environment.
Serving, training and exported checkpoints retain the required provenance.
The final comparison includes each head's training-token cost, stored bytes and
projected BF16 tensor-weight memory; it does not claim to measure simultaneous
multi-head residence or KV-cache memory.
