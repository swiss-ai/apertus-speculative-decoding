# Smoke screening, 2026-09-13, `debug` partition

First paired measurement of the project: plain Apertus-v1.5-70B against 8B `draft_model`
speculation at depth 3 with draft TP=4, same corpus, same `max_model_len`, same client.

**Headline: acceptance is high and the arm is still much slower.** The drafter's tokens are
accepted 62–79% of the time (mean acceptance length 2.86–3.36 of a possible 4), yet output
throughput falls to 0.42–0.47× of baseline, TPOT roughly doubles, and TTFT rises 3–5×. Every cell
reproduces the same direction, so this is not noise; see [Interpretation](#interpretation) for the
two shipped-configuration effects that are confounded with `method` here.

## Cells

| Variant | Job | Node | Cells |
|---|---|---|---|
| `baseline` (`method=none`) | 3391425 | nid007645 | `baseline-3391425/<workload>/c{1,8}/repeat-01` |
| `draft-n3-tp4` (`method=draft_model`) | 3391426 | nid006633 | `draft-n3-tp4-3391426/<workload>/c{1,8}/repeat-01` |

`analysis.csv` is `apertus-bench analyze` over both variants. `provenance/` holds each arm's
`/v1/models` response and the required pre-measurement chat completion. Correctness captures are in
`../correctness/smoke-20260913-debug/`.

`../../canvases/apertus-speculative-decoding-smoke-screening.canvas.tsx` renders this run as a
Cursor canvas: the paired per-cell comparison, the acceptance breakdown, the KV-cache cost, and the
caveats. Copy it into the workspace's managed `canvases/` directory to open it beside a chat.

## Results

All 12 cells had a 1.000 success rate with exact token usage on every request. `b` is baseline,
`d` is `draft-n3-tp4`.

| Workload | Concurrency | TTFT p50 (ms) b→d | TTFT p95 (ms) b→d | TPOT p50 (ms) b→d | Output tok/s b→d | Throughput ratio |
|---|---:|---|---|---|---|---:|
| open_chat | 1 | 31.0 → 111.6 | 35.5 → 113.5 | 14.13 → 31.86 | 70.1 → 31.0 | 0.442 |
| open_chat | 8 | 44.4 → 197.1 | 64.9 → 221.9 | 14.69 → 32.54 | 432.3 → 190.6 | 0.441 |
| code | 1 | 27.8 → 131.1 | 30.5 → 133.5 | 14.15 → 30.16 | 70.5 → 33.0 | 0.469 |
| code | 8 | 44.8 → 218.0 | 66.7 → 251.1 | 14.70 → 30.07 | 540.6 → 255.2 | 0.472 |
| long_context_summarization | 1 | 33.6 → 120.3 | 36.5 → 132.5 | 14.16 → 32.72 | 70.2 → 29.4 | 0.419 |
| long_context_summarization | 8 | 45.7 → 198.4 | 70.5 → 234.5 | 14.72 → 31.36 | 493.7 → 212.0 | 0.429 |

### Speculative-decoding counters (draft arm)

Deltas across each measured window, from `/metrics` snapshots taken before and after the cell.
Acceptance rate is accepted / drafted tokens; mean acceptance length includes the bonus token.

| Workload | Concurrency | Drafts | Draft tokens | Accepted | Acceptance rate | Mean acceptance length | Per-position acceptance (0/1/2) |
|---|---:|---:|---:|---:|---:|---:|---|
| open_chat | 1 | 1128 | 3384 | 2124 | 0.628 | 2.88 | 0.766 / 0.638 / 0.479 |
| open_chat | 8 | 1122 | 3366 | 2131 | 0.633 | 2.90 | 0.798 / 0.655 / 0.447 |
| code | 1 | 2748 | 8244 | 6492 | 0.787 | 3.36 | 0.887 / 0.773 / 0.703 |
| code | 8 | 2753 | 8259 | 6490 | 0.786 | 3.36 | 0.892 / 0.769 / 0.697 |
| long_context_summarization | 1 | 1932 | 5796 | 3660 | 0.631 | 2.89 | 0.764 / 0.621 / 0.509 |
| long_context_summarization | 8 | 1967 | 5901 | 3660 | 0.620 | 2.86 | 0.766 / 0.601 / 0.493 |

Code accepts best and also loses least, which is the direction H2 predicts. No counter decreased in
any window, so no window was rejected for a server restart.

### KV-cache cost

The drafter's weights cost more than half the KV cache:

| Arm | KV cache | Max concurrency at 131072 tokens/request | GPU blocks |
|---|---:|---:|---:|
| baseline | 513,696 tokens | 3.92× | — |
| draft-n3-tp4 | 251,472 tokens | 1.92× | 15,717 |

## Interpretation

Acceptance is not the binding constraint here, so H1's "acceptance alone is insufficient" case is
what this run demonstrates. Two properties of the pinned vLLM revision change with `method` and
cannot be held constant, and both penalise the speculative arm independently of acceptance:

- `Async scheduling not supported with draft_model-based speculative decoding and will be
  disabled.` The baseline logs `Asynchronous scheduling is enabled`; the draft arm logs
  `Asynchronous scheduling is disabled`. The baseline therefore overlaps CPU scheduling with GPU
  execution and the draft arm does not.
- `max_num_scheduled_tokens is set to 7168 based on the speculative decoding settings. This may
  lead to suboptimal performance.` The baseline ran with `max_num_batched_tokens=8192`.

A ~2.2× TPOT regression is far larger than either effect plausibly explains on its own, so the
draft forward passes and verification at TP=4 are the likely dominant cost, but this run cannot
separate the three. Treat the numbers as a measurement of the shipped configuration, not as a
measurement of speculation in isolation. `docs/protocol.md` now records the asymmetry.

## Provenance

- Cluster: Clariden (CSCS), partition `debug`, `--time 01:00:00`, 1 node × 4 GH200 per arm.
- Baseline job 3391425 on nid007645, replica head IP 172.28.51.237, started 17:23:47 CEST,
  cancelled 18:07 CEST after measurement.
- Draft job 3391426 on nid006633, replica head IP 172.28.32.244, started 18:07:36 CEST,
  cancelled 18:51 CEST after measurement. It was queued behind the baseline with reason
  `QOSMaxJobs`, so the two arms ran sequentially rather than concurrently.
- Served model names:
  `faruk_zahiragic/swiss-ai/Apertus-v1.5-70B-baseline-faruk_zahiragic-20260913T152337Z` and
  `faruk_zahiragic/swiss-ai/Apertus-v1.5-70B-draft-n3-tp4-faruk_zahiragic-20260913T152358Z`.
- Target `swiss-ai/Apertus-v1.5-70B`, draft `swiss-ai/Apertus-v1.5-8B`, both from the Capstor
  cache; target TP=4, draft TP=4, `num_speculative_tokens=3`.
- `max_model_len=131072` and `gpu_memory_utilization=0.8` on both arms.
- `model-launch` revision `909026a990454557f1b54d26f24ec3ad92e51e35`.
- vLLM revision `a601a9d998ddeb488f0c17e8512874b116aa7658`, build
  `0.23.1rc1.dev1029+ga601a9d99` (confirmed from the server's `/version`), from image
  `/capstor/store/cscs/swissai/infra01/container-images/ci/vllm_apertus_1.5_release-arm64.sqsh`.
- **The draft arm is not the stock pinned image.** `patches/vllm-apertus-image-token.patch` was
  bind-mounted over `/workspace/vllm/vllm/v1/spec_decode/llm_base_proposer.py` via
  `launch/patch-vllm.sh`, because the pinned image reads `image_token_index` off the target config
  and Apertus 1.5 only defines `image_token_id`, so the drafter cannot load without it. The fix is
  open upstream as swiss-ai/vllm#20 and is unmerged. The baseline arm used the stock image. Anyone
  reproducing the draft arm needs the same overlay, and any rebuilt image carrying the fix should
  be treated as a different build.
- Harness revision `270413f22f41ad008a72fffae5359cb3bd03785f`, `apertus-bench` 0.1.0, Python
  3.13.9, run from a venv on login node clariden-ln004.
- Load generator: on-cluster, hitting the replica node IP directly on port 8080. The public
  gateway was not used, so gateway latency and OpenTela routing are excluded. Both arms did
  register with the gateway (`replica_health.json` reached `HEALTHY` with a non-null `peer_id`).
- Corpus `workloads/smoke.jsonl`, SHA-256
  `316fb566a18e32305b98929e34fdac515b9a43c6c7329994b319870f879952af`, 6 prompts across 3 strata,
  identical ordered prompt IDs and per-prompt output caps on both arms.
- Sampling: temperature 0.0, top_p 1.0, per-request seed `1 + request_index`, natural EOS.
- Per cell: 8 warmup requests at the cell's concurrency, then 24 measured requests in closed loop,
  with a `/metrics` snapshot immediately before and after the measured window.
- Measurement window 2026-09-13T15:35:52Z to 2026-09-13T16:35:24Z.

Each arm answered `/v1/models` and then completed a real short chat completion before any
measurement, as the protocol requires. Both returned the same text for the verification prompt:

> Speculative decoding is an optimization technique in large language models that predicts and
> generates multiple future tokens in parallel before committing to the final output, allowing the
> model to skip over incorrect guesses and accelerate inference.

Full responses with usage are in `provenance/*-first-chat-completion.json`.

## Deviations and caveats

- **One deployment repeat per arm, not two.** `configs/experiment.yaml` screening asks for two
  independent deployments; a single `debug` hour per arm allowed one. No confidence interval over
  deployment-level effects can be computed from this run, and the analysis unit the protocol
  requires is the deployment. The consistency of the effect across six cells is suggestive, not a
  substitute.
- **The arms ran on different nodes** (nid007645 vs nid006633) and at different times, so node and
  time effects are not separated from `method`. The protocol treats node/date as a nuisance block;
  this run cannot.
- **The smoke corpus is not the research corpus.** Its "long context" prompts are ~350 tokens, far
  below the 16k–65k target in `workloads/README.md`, so the
  `long_context_summarization` rows here measure almost no prefill. Prefill-heavy behaviour, where
  speculation has the most room to help TTFT, is untested.
- **`require_exact_greedy_smoke_match` failed: 0 of 6 exact matches.** See
  `../correctness/smoke-20260913-debug/comparison.json`. Each pair agrees for the first 297–564
  characters and then diverges into a different but equally fluent continuation, with completion
  token counts within 3 of each other. That is the signature of floating-point nondeterminism in
  the target forward pass — the arms ran on different nodes, under different batch shapes and token
  budgets, with prefix caching enabled — rather than of a rejection-sampling bug. It is not
  evidence for or against the lossless theorem, and the protocol already says exact agreement is an
  integration check rather than a proof. It does mean this run does not clear that gate, and a
  same-node, same-batch-shape re-check is needed before any correctness claim.
- Concurrency 32 and the confirmation matrix were not run; there is no `ignore_eos` control, no
  capacity study, and no n-gram arm.
- The 24 requests per cell cycle through only 2 prompts per stratum, so latency distributions are
  narrow by construction and request-level percentiles are not independent samples.
