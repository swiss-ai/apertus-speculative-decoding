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
