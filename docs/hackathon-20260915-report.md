# Speculative decoding on Apertus-v1.5-70B: the 8B drafter against n-gram

CSCS / Swiss AI gathering · OAT Lugano · 2026-09-15 · Faruk Zahiragić (EPFL)

Issue: [swiss-ai/apertus-program#1057](https://github.com/swiss-ai/apertus-program/issues/1057) ·
code: [swiss-ai/apertus-speculative-decoding](https://github.com/swiss-ai/apertus-speculative-decoding)

Six independently launched deployments on 4×GH200 Clariden nodes. Same corpus, same client, same
`max_model_len`. Three workloads × two concurrencies × six deployments = **36 cells**, all at a
1.000 success rate.

| | |
| --- | --- |
| Jobs | 3391425 / 3405008 (baseline) · 3391426 / 3392153 (draft) · 3392370 / 3403354 (n-gram) |
| Speculative config | depth 3 · draft TP=4 · n-gram lookup 1–4 |
| Repeats | 2 independent deployments per arm |
| Corpus | smoke, 6 prompts |

Numbers below are from
[`results/smoke-screening-20260913-debug/`](../results/smoke-screening-20260913-debug/)
(`README.md`, `analysis.csv`, 36 rows). Prefer that run over
[`docs/hackathon-20260915.md`](hackathon-20260915.md) where they disagree: this report uses six
deployments / 36 cells, the baseline repeat, and TTFT in 22 of 24 comparisons.

Companion slides: [HTML](hackathon-20260915-slides.html) ·
[PDF](hackathon-20260915-slides.pdf) (GitHub PDF preview).

---

## Verdict

**Do not enable `draft_model` speculation for Apertus 1.5 70B on this stack.**

The 8B drafter accepts 60–80% of its proposed tokens and still returns **0.375–0.472×** baseline
output throughput in all twelve of its cells, across two independent deployments. The cost is not
acceptance and not verification. A speculative step with the drafter costs **6.2–7.5×** a baseline
decode step, of which the drafter alone is **4.9–6.2×**. Three 8B forward passes at TP=4 cannot be
five to six whole 70B steps, so that is per-step overhead in the draft path.

**N-gram is the only positive result, and it survives every denominator.**

It accepts only 13–29%. On long-context summarization it beats baseline in both of its deployments
and against both baselines: **1.127–1.204×** at concurrency 1 and **1.071–1.098×** at concurrency 8.
No ratio in the study moves by more than 0.005 when the denominator is swapped. All six n-gram cells
clear the 10% spread gate, worst 6.6%. Faster to first token in **22 of 24** comparisons.

**Acceptance is not the binding constraint. The arm that accepts worst is the only arm that wins.**

---

## Headline numbers

| Metric | Value |
| --- | --- |
| Draft output throughput vs baseline | **0.375–0.472×** (12 cells) |
| N-gram output throughput vs baseline | **0.848–1.204×** (12 cells) |
| N-gram over draft repeat 1 | **1.8–2.7×** in every cell |
| Draft engine step vs baseline decode step | **6.2–7.5×** (derived) |
| Verification + scheduling only (n-gram) | **1.23–1.36×** |
| KV-cache tokens, n-gram vs baseline | **−19%** — a capacity **cost**, no drafter resident |
| KV-cache tokens, draft vs baseline | **−51%** — a capacity **cost** after hosting the 8B drafter |

Baseline itself reproduces to **0.41%** on output throughput across two nodes. Speculative arms
spread 1.7–11.1% (draft) and 1.7–6.6% (n-gram). Speculation costs predictability as well as
capacity.

---

## What was run

| Deployment | Method | Job | Node | Image |
| --- | --- | ---: | --- | --- |
| `baseline` repeat 1 | `none` | 3391425 | nid007645 | stock pinned |
| `baseline` repeat 2 | `none` | 3405008 | nid006653 | stock pinned |
| `draft-n3-tp4` repeat 1 | `draft_model` | 3391426 | nid006633 | pinned + overlay |
| `draft-n3-tp4` repeat 2 | `draft_model` | 3392153 | nid006687 | pinned + overlay |
| `ngram-n3` repeat 1 | `ngram` | 3392370 | nid006687 | stock pinned |
| `ngram-n3` repeat 2 | `ngram` | 3403354 | nid007500 | stock pinned |
| `draft-n3-tp1` | `draft_model` | 3392110 | nid006593 | never served |

Target `swiss-ai/Apertus-v1.5-70B` TP=4, drafter `swiss-ai/Apertus-v1.5-8B` TP=4,
`num_speculative_tokens=3`, `max_model_len=131072`, `gpu_memory_utilization=0.8`, temperature 0.0,
natural EOS. vLLM `a601a9d` (`0.23.1rc1.dev1029+ga601a9d99`), `model-launch` `909026a`, `debug`
partition.

Per cell: 8 warmup then 24 measured requests in closed loop, with a `/metrics` snapshot immediately
before and after. The load generator ran on-cluster against the replica node IP (no gateway
latency). Total completion tokens agree within 0.2% across the six deployments (36,148 / 36,164
baseline; 36,160 / 36,164 draft; 36,184 / 36,115 n-gram).

---

## N-gram vs draft vs baseline

Output tokens per second. Ratios divide by **baseline repeat 1**. Swapping to repeat 2 or the mean
moves no ratio by more than 0.005. Bold cells beat baseline.

| Workload | Conc. | Baseline tok/s | Draft r1 → r2 | N-gram r1 → r2 | d÷b r1 / r2 | n÷b r1 / r2 |
| --- | ---: | ---: | --- | --- | --- | --- |
| open_chat | 1 | 70.1 | 31.0 → 29.1 | 63.3 → 64.3 | 0.442 / 0.415 | 0.902 / 0.917 |
| open_chat | 8 | 432.3 | 190.6 → 173.2 | 366.5 → 372.9 | 0.441 / 0.401 | 0.848 / 0.863 |
| code | 1 | 70.5 | 33.0 → 32.5 | 66.5 → 69.1 | 0.469 / 0.461 | 0.943 / 0.980 |
| code | 8 | 540.6 | 255.2 → 238.5 | 464.3 → 480.3 | 0.472 / 0.441 | 0.859 / 0.888 |
| long_context_summarization | 1 | 70.2 | 29.4 → 26.3 | 79.2 → 84.5 | 0.419 / 0.375 | **1.127 / 1.204** |
| long_context_summarization | 8 | 493.7 | 212.0 → 200.8 | 528.7 → 539.8 | 0.429 / 0.407 | **1.071 / 1.093** |

Against both baselines, the n-gram summarization win is **1.126–1.204×** at concurrency 1 and
**1.071–1.098×** at concurrency 8. The draft arm is **0.375–0.474×** across every
numerator–denominator pairing.

### Acceptance

| Arm | Acceptance rate | Mean acceptance length (max 4) |
| --- | --- | --- |
| `draft-n3-tp4`, both repeats | 0.605–0.797 (code highest) | 2.81–3.39 |
| `ngram-n3`, both repeats | 0.127–0.285 (summarization highest) | 1.38–1.85 |

Draft acceptance is a near-invariant between deployments (0.5–4.3%). Quality is a property of the
model pair, not the node.

N-gram acceptance is workload-dependent in the prompt-lookup direction: summarization accepts best
because a summary quotes the document. Code generation has little to copy forward. It moves
2.0–13.9% between deployments, because prompt lookup matches against the prompt plus tokens
generated so far, and greedy outputs diverge across deployments.

---

## Where the speculative step cost goes

Per-token latency understates per-step cost. Mean step time is derived as
`mean TPOT × T ÷ (T − accepted)`, where `T` is the cell's completion tokens. The n-gram arm
proposes with no forward pass, so it isolates verification. The baseline emits one token per step,
so its TPOT is its step time.

| | |
| --- | --- |
| Baseline decode step | **~14.2 ms** |
| N-gram step (verify + scheduling) | **17.4–20.0 ms · 1.23–1.36×** |
| Draft step | **91–106 ms · extra 72–88 ms · 6.2–7.5× total** |

| Workload | Conc. | Baseline step | N-gram r1 / r2 | Draft r1 / r2 | Verify × | Draft total × |
| --- | ---: | --- | --- | --- | --- | --- |
| open_chat | 1 | 14.13 ms | 18.64 / 18.32 | 91.15 / 100.41 | 1.32 / 1.30× | 6.45 / 7.11× |
| open_chat | 8 | 14.67 ms | 20.00 / 19.61 | 94.24 / 101.27 | 1.36 / 1.34× | 6.42 / 6.90× |
| code | 1 | 14.15 ms | 18.13 / 17.45 | 101.62 / 104.25 | 1.28 / 1.23× | 7.18 / 7.37× |
| code | 8 | 14.70 ms | 19.54 / 18.88 | 101.34 / 106.45 | 1.33 / 1.28× | 6.89 / 7.24× |
| long_context_summarization | 1 | 14.16 ms | 17.79 / 17.36 | 96.63 / 105.52 | 1.26 / 1.23× | 6.82 / 7.45× |
| long_context_summarization | 8 | 14.71 ms | 19.42 / 18.98 | 91.77 / 98.56 | 1.32 / 1.29× | 6.24 / 6.70× |

Two n-gram deployments on different nodes put step cost within **1.7–3.9%** of each other. A node
effect cannot absorb a drafter cost of five to six baseline steps.

This is an inference from committed counters, not a kernel profile. One assumption is known to be
violated in a direction that inflates the drafter share: `draft_model` runs with a 7168-token
per-step budget while n-gram keeps the baseline's 8192. Client-side median stream-event gaps
corroborate the same split (~91–101 ms draft vs ~16–20 ms n-gram vs ~14 ms baseline).

---

## KV cache: speculation is a capacity cost

These are the most reproducible numbers in the study. Enabling speculation costs cache even when it
is free of model weights. **−19% and −51% are costs, not savings.**

| Arm | KV cache (tokens) | Max concurrency at 131072 tok/request | Loss vs baseline |
| --- | ---: | ---: | ---: |
| baseline (r1 / r2) | 513,696 / 513,696 | 3.92× | — |
| `ngram-n3` (r1 / r2) | 416,640 / 416,640 | 3.18× | **−18.9%** |
| `draft-n3-tp4` (r1 / r2) | 251,472 / 251,488 | 1.92× | **−51.0%** |

The n-gram arm loads no drafter and runs the same image as the baseline, yet gives up 97,056 tokens
of cache — presumably the larger per-step activation footprint of verifying four positions at once.
The drafter's weights then cost a further 165,168 tokens. The draft arm's halved capacity is
roughly 37% speculation machinery and 63% drafter weights.

Both baseline deployments and both n-gram deployments sized their cache to the same token count on
different nodes. The two draft deployments came within 16 tokens of each other. Unlike throughput,
these capacity costs are deterministic properties of the configuration.

`max_model_len` had to come down from 262144 to 131072: with the drafter resident, a single
full-context request would not fit. Anyone copying the published example at its default 262144
hits this the moment they add a drafter.

---

## What this run cannot claim

- **Two serving properties are confounded with method.** Async scheduling is auto-disabled for both
  speculative arms, not the baseline. The per-step token budget shrinks to 7168 for `draft_model`
  only (baseline and n-gram stay at 8192), so part of the 72–88 ms is budget, not drafter.
- **The smoke “long context” is ~283 prompt tokens**, not 16k–64k. These rows measure almost no
  prefill. N-gram’s best regime is untested — this probably understates its advantage.
- **The draft arm ran a patched image.** Stock vLLM crashes at drafter load (`image_token_index`).
  Overlay is [swiss-ai/vllm#20](https://github.com/swiss-ai/vllm/pull/20), unmerged. Baseline and
  n-gram used the stock image.
- **Do not present “the gate passes for n-gram”.** Six greedy prompts. Two n-gram deployments
  return opposite calibrated-gate verdicts. Divergence between arms is the same size as divergence
  between two deployments of one arm — including two baselines (1 of 6 exact).
- **One draft cell breaches the 10% spread gate.** `long_context_summarization` at concurrency 1
  moves 11.1% between draft deployments. Direction is unaffected. All six n-gram cells pass, worst
  6.6%.
- **`draft_tensor_parallel_size=1` is unservable** at this vLLM revision (target TP and draft TP
  must match). Hypothesis H4 is closed for now; the step-cost split says TP=1 would not have
  rescued five to six baseline steps anyway.
- **Not measured:** concurrency 32, depths other than 3, confirmation-matrix bootstrap CIs,
  GPU / power telemetry, a within-deployment correctness capture.

---

## The ask

**Profile the draft path.** Find where 72–88 ms per step goes when three 8B forward passes should
cost a fraction of one 70B step.

That is the highest-value next experiment, and it is a vLLM question: captured graphs, model-switch
/ sync overhead, 7168-token budget. If that overhead is fixable, the whole conclusion changes.

Also worth doing:

1. A genuinely long-context summarization workload (16k–64k input tokens) — the regime n-gram
   should be best in.
2. A baseline with async scheduling off, to test the leading hypothesis for n-gram's TTFT
   advantage.
3. A path for [swiss-ai/vllm#20](https://github.com/swiss-ai/vllm/pull/20) into a built image.
   Today nobody can run draft-model speculation with Apertus 1.5 without a bind-mount overlay, and
   the upstreaming PRs do not carry the fix.

---

## Artifacts

| What | Where |
| --- | --- |
| 36 measured cells | [`results/smoke-screening-20260913-debug/`](../results/smoke-screening-20260913-debug/) |
| Run narrative and tables | [`results/smoke-screening-20260913-debug/README.md`](../results/smoke-screening-20260913-debug/README.md) |
| Flat analysis (36 rows) | [`results/smoke-screening-20260913-debug/analysis.csv`](../results/smoke-screening-20260913-debug/analysis.csv) |
| Greedy captures and gates | [`results/correctness/smoke-20260913-debug/`](../results/correctness/smoke-20260913-debug/) |
| Unservable draft TP=1 | [`results/deployment-failures/draft-n3-tp1-3392110/`](../results/deployment-failures/draft-n3-tp1-3392110/) |
| vLLM one-line fix | [`patches/vllm-apertus-image-token.patch`](../patches/vllm-apertus-image-token.patch) |
| Slides | [HTML](hackathon-20260915-slides.html) · [PDF](hackathon-20260915-slides.pdf) |
| Longer narrative (5 deployments; stale vs this report) | [`hackathon-20260915.md`](hackathon-20260915.md) |
