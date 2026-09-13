# Apertus speculative decoding

This repository studies whether speculative decoding should be enabled for
`swiss-ai/Apertus-v1.5-70B` on a four-GH200 Clariden node. It compares the plain 70B server,
8B draft-model speculation, and model-free n-gram speculation under controlled, repeated,
production-facing workloads.

The project is scoped as an 8 ECTS semester research project. It goes beyond a one-off speed
test: it measures latency, throughput, acceptance, serving capacity, memory trade-offs, and the
conditions under which a configuration wins or loses.

## Research questions

1. How does speedup vary across open chat, code, and long-context summarization at concurrency
   1, 8, and 32?
2. How strongly do draft acceptance rate and mean accepted length predict TTFT, TPOT, and output
   throughput?
3. Which combination of speculative depth `{2,3,5,8}` and draft TP is best, and when does draft
   overhead outweigh accepted work? Draft TP is pinned to 4 for now: the pinned vLLM refuses draft
   TP != target TP, so only a depth sweep is answerable
   (`results/deployment-failures/draft-n3-tp1-3392110/`).
4. When is n-gram speculation a better operational choice than the 8B draft model?
5. Does the winning setting increase sustainable request rate without unacceptable tail latency
   or loss of KV-cache capacity?

The full experimental contract is in [docs/protocol.md](docs/protocol.md), the initial reading map
is in [docs/literature.md](docs/literature.md), and the staged matrix is machine-readable in
[configs/experiment.yaml](configs/experiment.yaml).

## What is implemented

- parameterized baseline, draft-model, and n-gram `sml` launchers using Capstor model paths;
- a streaming OpenAI-compatible client measuring per-request E2E, TTFT, conventional TPOT, and
  stream-event gaps;
- before/after snapshots of vLLM's exact speculative-decoding Prometheus counters;
- closed-loop fixed-concurrency matrix execution with warmup and deployment-level repeats;
- sequential greedy-output capture, so correctness can be checked while using only one model node;
- immutable raw cell artifacts plus a CSV analysis step;
- a small smoke corpus and a specified schema for the real benchmark corpus.

The current `model-launch` k6 path is useful for serving-capacity sweeps, but it sends non-streaming
requests and therefore cannot produce true client-observed TTFT. This repository uses its own
streaming client for latency experiments and reuses `swiss-ai/bench-agent` for the final open-loop
capacity experiment.

## Setup

Keep this repository next to a `model-launch` checkout at the pinned revision (or a reviewed newer
revision containing the Apertus 1.5 release environment):

```text
Projects/
├── apertus-speculative-decoding/
└── model-launch/
```

Create the local environment:

```bash
cd apertus-speculative-decoding
python -m venv .venv
. .venv/bin/activate
python -m pip install -e '.[dev]'
apertus-bench validate-workloads workloads/smoke.jsonl
```

If another branch currently occupies the main `model-launch` checkout, a separate Git worktree
avoids disturbing it:

```bash
git -C ../model-launch fetch origin
git -C ../model-launch worktree add ../model-launch-apertus \
  909026a990454557f1b54d26f24ec3ad92e51e35
export MODEL_LAUNCH_ROOT=../model-launch-apertus
```

The launchers default to the cached Capstor paths. Override `TARGET_MODEL`, `DRAFT_MODEL`, or
`MODEL_LAUNCH_ROOT` when the site layout differs.

## Launch a variant

From this repository on the Clariden login node:

```bash
# Plain target
./launch/baseline.sh

# 8B draft, depth 5, draft TP=1
NUM_SPECULATIVE_TOKENS=5 DRAFT_TP=1 ./launch/draft-model.sh

# N-gram, depth 5
NUM_SPECULATIVE_TOKENS=5 PROMPT_LOOKUP_MAX=4 ./launch/ngram.sh
```

Each launcher prints its unique served model name. Do not begin measurement merely because the
name appears in `/v1/models`; require one successful short chat completion first.

The launchers speak the pinned `model-launch` CLI (`--system`, `--framework`, `--environment`,
`--nodes-per-replica`, `--time`). Later revisions rename those to `--firecrest-system`,
`--serving-framework`, `--slurm-environment`, and also rename the OpenTela share mount from
`/ocfbin` to `/opentelabin`, so a newer or older `sml` paired with the pinned environment toml
fails on the node rather than at submission. Install the pin and put it first on `PATH`:

```bash
python3 -m venv ~/venvs/sml-apertus
~/venvs/sml-apertus/bin/pip install -e "$MODEL_LAUNCH_ROOT"
export PATH="$HOME/venvs/sml-apertus/bin:$PATH"
```

The packaged environment toml also carries a literal `{arch}` in its image path, which only some
`sml` builds substitute on the node; one that does not makes pyxis reject the placeholder and the
job dies seconds after it starts. `launch/resolve-env.sh` writes a resolved copy (`arm64` for
GH200) under `~/.sml` and the launchers pass that. Override the source toml with `ENV_SOURCE`, the
architecture with `SML_ARCH`, and the partition with `SML_PARTITION`; any extra arguments are
forwarded to `sml advanced`, so `--no-tui` works for non-interactive launches.

`draft_model` speculation additionally needs a source fix that the pinned image predates: vLLM
reads `image_token_index` off the target config, which Apertus 1.5 does not define (it uses
`image_token_id`), so the drafter fails to load and the engine never starts. `patches/` holds the
one-line fix and `launch/patch-vllm.sh` binds a corrected copy over the read-only image:

```bash
IMAGE=/capstor/store/cscs/swissai/infra01/container-images/ci/vllm_apertus_1.5_release-arm64.sqsh
EXTRA_MOUNTS="$(./launch/patch-vllm.sh "$IMAGE" patches/vllm-apertus-image-token.patch)" \
  ./launch/draft-model.sh --no-tui
```

Record the overlay alongside any measurement taken with it, and drop `EXTRA_MOUNTS` once a
rebuilt image carries the fix. The baseline and n-gram variants never load a drafter, so they do
not need it.

## Greedy correctness sanity check on one node

Capture the baseline outputs, stop the baseline job, launch the speculative variant, and capture
again with the same corpus and sampling settings:

```bash
export CSCS_SERVING_API='...'
API=https://api.swissai.svc.cscs.ch

apertus-bench capture \
  --base-url "$API" \
  --model "$BASELINE_SERVED_MODEL" \
  --workloads workloads/smoke.jsonl \
  --output results/correctness/baseline.json

apertus-bench capture \
  --base-url "$API" \
  --model "$SPEC_SERVED_MODEL" \
  --workloads workloads/smoke.jsonl \
  --output results/correctness/draft-n3-tp4.json

apertus-bench compare-captures \
  results/correctness/baseline.json \
  results/correctness/draft-n3-tp4.json \
  --output results/correctness/comparison.json
```

Exact greedy agreement is a useful integration test, not a proof of distributional equivalence.

## Run one deployed variant

The example below runs all three smoke strata at concurrency 1 and 8. The metrics URL must resolve
to that vLLM server's `/metrics` endpoint; use `--no-metrics` only for harness debugging.

```bash
apertus-bench matrix \
  --base-url "$API" \
  --metrics-url "$VLLM_METRICS_URL" \
  --model "$SPEC_SERVED_MODEL" \
  --variant draft-n3-tp4 \
  --method draft_model \
  --num-speculative-tokens 3 \
  --draft-tensor-parallel-size 4 \
  --workloads workloads/smoke.jsonl \
  --concurrencies 1 8 \
  --requests 32 \
  --repeats 2 \
  --metadata slurm_job_id="$SLURM_JOB_ID" \
  --output results/smoke/draft-n3-tp4
```

For the real experiment, replace the smoke corpus, use the matrix and repeat counts in the
protocol, and put each independently launched deployment in its own result directory.

After baseline and candidate runs exist:

```bash
apertus-bench analyze results/experiment-001 --output analysis/results.csv
```

## Metric definitions

- **TTFT:** client request start to the first non-empty streamed content/reasoning/tool payload.
- **TPOT:** `(E2E - TTFT) / (completion_tokens - 1)` per request; this is the conventional average
  inter-token latency measure used for the headline result.
- **Stream-event gap:** time between non-empty SSE payloads. Speculation can deliver several
  accepted tokens in one event, so this diagnostic is not treated as token-level ITL.
- **Output throughput:** successful completion tokens divided by whole-cell wall time.
- **Acceptance rate:** delta accepted draft tokens / delta proposed draft tokens.
- **Mean acceptance length:** `1 + accepted_tokens / drafts`, including the verified bonus token,
  matching pinned vLLM semantics.

Requests within one deployment are useful for latency distributions but are not independent
experimental replicates. Confidence intervals and configuration comparisons use independently
launched deployment repeats.

## Upstream pins

- `swiss-ai/model-launch`: `909026a990454557f1b54d26f24ec3ad92e51e35`
- `swiss-ai/vllm` branch `apertus-1-5`: `a601a9d998ddeb488f0c17e8512874b116aa7658`
- vLLM base: `v0.23.1rc1`
