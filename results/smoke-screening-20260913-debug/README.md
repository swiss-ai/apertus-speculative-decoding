# Smoke screening, 2026-09-13, `debug` partition

Screening of Apertus-v1.5-70B at depth 3 on one Clariden GH200 node: plain serving, 8B
`draft_model` speculation at draft TP=4 (two deployments), and model-free n-gram speculation (two
deployments). Same corpus, same `max_model_len`, same client, same on-cluster load path.

**Headline: the 8B drafter accepts well and loses badly; n-gram accepts poorly and is the only arm
that wins anything, and its win reproduces across two independent deployments.** The drafter's tokens
are accepted 60–80% of the time (mean acceptance length 2.81–3.39 of a possible 4) yet output
throughput falls to 0.375–0.472× of baseline. N-gram accepts only 13–29% and reaches 0.85–1.20× of
baseline, beating the draft arm by 1.8–2.7× everywhere and beating the **baseline** on long-context
summarization in both of its deployments (1.127× and 1.204× at concurrency 1, 1.071× and 1.093× at
concurrency 8). All six n-gram cells reproduce within the protocol's 10% spread gate, worst 6.6%.

Splitting the two arms locates the cost. Verifying a depth-3 proposal costs **1.23–1.36×** a plain
decode step; proposing with the 8B drafter costs a further **4.9–6.2×** a decode step. Draft cost
dominates verification cost by an order of magnitude — but it is also about an order of magnitude
larger than an 8B forward pass can account for, so it is draft-path overhead rather than drafter
arithmetic. See [the decomposition](#step-cost-decomposition-draft-cost-versus-verification-cost).

## Cells

| Variant | Deployment repeat | Job | Node | Cells |
|---|---:|---|---|---|
| `baseline` (`method=none`) | 1 | 3391425 | nid007645 | `baseline-3391425/<workload>/c{1,8}/repeat-01` |
| `draft-n3-tp4` (`method=draft_model`) | 1 | 3391426 | nid006633 | `draft-n3-tp4-3391426/<workload>/c{1,8}/repeat-01` |
| `draft-n3-tp4` (`method=draft_model`) | 2 | 3392153 | nid006687 | `draft-n3-tp4-3392153/<workload>/c{1,8}/repeat-02` |
| `ngram-n3` (`method=ngram`) | 1 | 3392370 | nid006687 | `ngram-n3-3392370/<workload>/c{1,8}/repeat-01` |
| `ngram-n3` (`method=ngram`) | 2 | 3403354 | nid007500 | `ngram-n3-3403354/<workload>/c{1,8}/repeat-02` |

A `draft_tensor_parallel_size=1` cell was attempted between the draft repeats and the n-gram arm, and
is not servable at the pinned vLLM revision; see `../deployment-failures/draft-n3-tp1-3392110/`. The
draft and n-gram arms therefore have two deployment repeats each, and **the baseline has one**, which
is now the design's weakest point: every ratio quoted here still divides by a single baseline
deployment.

`analysis.csv` is `apertus-bench analyze` over all five deployments (30 rows). `provenance/` holds
each deployment's `/v1/models` response and the required pre-measurement chat completion.
Correctness captures are in `../correctness/smoke-20260913-debug/`.

`../../canvases/apertus-speculative-decoding-smoke-screening.canvas.tsx` renders this run as a
Cursor canvas: the paired per-cell comparison, the acceptance breakdown, the KV-cache cost, and the
caveats. Copy it into the workspace's managed `canvases/` directory to open it beside a chat. It
covers all four deployments: the between-deployment spread of the two draft repeats, the n-gram arm,
the derived step-cost decomposition, the corrected correctness gate, and the unservable draft TP=1
level. It predates n-gram deployment repeat 2 (job 3403354) and so shows the n-gram arm as a single
deployment.

## Results

All 30 cells had a 1.000 success rate with exact token usage on every request, and total completion
tokens agree within 0.2% across all five deployments (36,148 baseline; 36,160 and 36,164 draft;
36,184 and 36,115 n-gram), so the arms really did do the same work. The table below pairs the baseline against draft repeat 1; repeat 2 and the n-gram arm have
their own sections. `b` is baseline, `d` is `draft-n3-tp4`.

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

### N-gram arm

Job 3392370 on nid006687, depth 3, `prompt_lookup_min=1`, `prompt_lookup_max=4`, **stock pinned
image with no overlay**, same corpus and client. This arm loads no drafter, so its proposals cost no
GPU forward pass.

| Workload | Conc. | TTFT p50 (ms) b→n | TTFT p95 (ms) b→n | TPOT p50 (ms) b→n | Output tok/s b→n | Ratio to baseline | Ratio to draft r1 |
|---|---:|---|---|---|---|---:|---:|
| open_chat | 1 | 31.0 → 24.1 | 35.5 → 24.6 | 14.13 → 16.03 | 70.1 → 63.3 | 0.902 | 2.04× |
| open_chat | 8 | 44.4 → 39.5 | 64.9 → 48.8 | 14.69 → 17.36 | 432.3 → 366.5 | 0.848 | 1.92× |
| code | 1 | 27.8 → 26.5 | 30.5 → 27.5 | 14.15 → 14.91 | 70.5 → 66.5 | 0.943 | 2.01× |
| code | 8 | 44.8 → 39.4 | 66.7 → 55.2 | 14.70 → 16.04 | 540.6 → 464.3 | 0.859 | 1.82× |
| long_context_summarization | 1 | 33.6 → 27.7 | 36.5 → 28.4 | 14.16 → 12.51 | 70.2 → 79.2 | **1.127** | 2.69× |
| long_context_summarization | 8 | 45.7 → 41.0 | 70.5 → 57.1 | 14.72 → 13.36 | 493.7 → 528.7 | **1.071** | 2.49× |

N-gram beats the draft-model arm by 1.8–2.7× everywhere, and it is the **only configuration measured
so far that beats the baseline on anything**: long-context summarization gains 7–13% output
throughput and 12% lower TPOT. It loses 6–15% on open chat and code. It also has **lower TTFT than
baseline in every cell**, p50 and p95 alike, which no other arm managed.

| Workload | Conc. | Drafts | Draft tokens | Accepted | Acceptance rate | Mean acceptance length | Per-position acceptance (0/1/2) |
|---|---:|---:|---:|---:|---:|---:|---|
| open_chat | 1 | 1092 | 3276 | 456 | 0.139 | 1.42 | 0.209 / 0.132 / 0.077 |
| open_chat | 8 | 1075 | 3225 | 452 | 0.140 | 1.42 | 0.214 / 0.131 / 0.075 |
| code | 1 | 4164 | 12444 | 1584 | 0.127 | 1.38 | 0.216 / 0.098 / 0.066 |
| code | 8 | 4119 | 12305 | 1623 | 0.132 | 1.39 | 0.223 / 0.104 / 0.067 |
| long_context_summarization | 1 | 2244 | 6732 | 1668 | 0.248 | 1.74 | 0.310 / 0.251 / 0.182 |
| long_context_summarization | 8 | 2216 | 6648 | 1763 | 0.265 | 1.80 | 0.342 / 0.264 / 0.190 |

Acceptance is far below the model-based drafter's 0.60–0.80, as expected, and it is strongly
workload-dependent — but **not in the way the code stratum predicted**. Summarization accepts
0.248–0.265 and is the only stratum that wins, which fits prompt lookup: a summary reuses spans from
the document. Code accepts 0.127–0.132, no better than open chat, because these are code
*generation* prompts with short instructions and little text to copy forward. Prompt lookup rewards
prompts whose output quotes the prompt, not prompts about code.

#### N-gram deployment repeat 2: the summarization win reproduces

Job 3403354 on nid007500 re-ran the identical configuration — depth 3, `prompt_lookup_min=1`,
`prompt_lookup_max=4`, `max_model_len=131072`, stock image, same corpus, same 8 warmup and 24 measured
requests per cell, same direct-node-IP load path. Spread is the absolute difference over the mean of
the two repeats, judged against `maximum_repeat_spread_fraction: 0.10`.

| Workload | Conc. | Output tok/s r1→r2 | Ratio to baseline r1→r2 | Ratio spread | Gate |
|---|---:|---|---|---:|---|
| open_chat | 1 | 63.3 → 64.3 | 0.902 → 0.917 | 1.7% | pass |
| open_chat | 8 | 366.5 → 372.9 | 0.848 → 0.863 | 1.7% | pass |
| code | 1 | 66.5 → 69.1 | 0.943 → 0.980 | 3.9% | pass |
| code | 8 | 464.3 → 480.3 | 0.859 → 0.888 | 3.4% | pass |
| long_context_summarization | 1 | 79.2 → 84.5 | **1.127 → 1.204** | 6.6% | pass |
| long_context_summarization | 8 | 528.7 → 539.8 | **1.071 → 1.093** | 2.1% | pass |

**The win reproduces.** `long_context_summarization` exceeds 1.0× baseline in both deployments at both
concurrencies, and repeat 2 is the larger win. Every cell clears the 10% spread gate, worst case 6.6%,
so the n-gram arm is markedly more reproducible than the draft arm, which spread 1.7–11.1% and
breached the gate in one cell. Repeat 2 is slightly *faster* than repeat 1 in all six cells, the
opposite direction from the draft repeats, so the uniform shift between the two draft deployments was
not a systematic "later run is slower" artefact of the harness or the partition.

The TTFT advantage also reproduces: n-gram is below baseline in all six cells at both p50 and p95 in
repeat 2 as well, so it holds in 12 of 12 measured cells. TTFT spread between the two n-gram
deployments is 0.7–6.4%.

| Workload | Conc. | TTFT p50 (ms) base → r1 → r2 | TTFT p95 (ms) base → r1 → r2 | TPOT p50 (ms) base → r1 → r2 |
|---|---:|---|---|---|
| open_chat | 1 | 31.0 → 24.1 → 25.3 | 35.5 → 24.6 → 26.0 | 14.13 → 16.03 → 15.69 |
| open_chat | 8 | 44.4 → 39.5 → 38.1 | 64.9 → 48.8 → 51.6 | 14.69 → 17.36 → 17.02 |
| code | 1 | 27.8 → 26.5 → 25.0 | 30.5 → 27.5 → 25.8 | 14.15 → 14.91 → 14.48 |
| code | 8 | 44.8 → 39.4 → 37.6 | 66.7 → 55.2 → 53.6 | 14.70 → 16.04 → 15.44 |
| long_context_summarization | 1 | 33.6 → 27.7 → 26.3 | 36.5 → 28.4 → 27.5 | 14.16 → 12.51 → 11.69 |
| long_context_summarization | 8 | 45.7 → 41.0 → 39.2 | 70.5 → 57.1 → 56.8 | 14.72 → 13.36 → 12.85 |

Repeat 2's counters, and the acceptance stability question:

| Workload | Conc. | Drafts | Draft tokens | Accepted | Acceptance rate r1→r2 | Spread | Mean acceptance length r1→r2 | Spread | Per-position (0/1/2) |
|---|---:|---:|---:|---:|---|---:|---|---:|---|
| open_chat | 1 | 1068 | 3204 | 456 | 0.139 → 0.142 | 2.2% | 1.42 → 1.43 | 0.7% | 0.213 / 0.135 / 0.079 |
| open_chat | 8 | 1100 | 3300 | 431 | 0.140 → 0.131 | 7.1% | 1.42 → 1.39 | 2.0% | 0.201 / 0.123 / 0.068 |
| code | 1 | 4032 | 12048 | 1584 | 0.127 → 0.131 | 3.2% | 1.38 → 1.39 | 0.9% | 0.223 / 0.104 / 0.065 |
| code | 8 | 4102 | 12252 | 1649 | 0.132 → 0.135 | 2.0% | 1.39 → 1.40 | 0.6% | 0.227 / 0.107 / 0.068 |
| long_context_summarization | 1 | 2136 | 6408 | 1824 | 0.248 → 0.285 | 13.9% | 1.74 → 1.85 | 6.2% | 0.365 / 0.275 / 0.213 |
| long_context_summarization | 8 | 2172 | 6516 | 1801 | 0.265 → 0.276 | 4.1% | 1.80 → 1.83 | 1.9% | 0.358 / 0.271 / 0.200 |

**Acceptance is not the near-invariant for n-gram that it was for the draft arm.** The drafter's
acceptance rate moved 0.5–4.3% between deployments; n-gram's moves 2.0–13.9%, and the largest move is
in the cell that matters most (`long_context_summarization` at concurrency 1, 0.248 → 0.285). The
mechanism is visible in the design: the drafter's acceptance is a property of a fixed model pair
scoring fixed text, whereas prompt lookup matches against the prompt *plus the tokens generated so
far*. Because the two deployments' greedy outputs diverge numerically (see the correctness note
below), they generate different text, and different text offers different n-grams to match. N-gram
acceptance therefore inherits output nondeterminism in a way model-based acceptance does not. Mean
acceptance length is the steadier of the two quantities (0.6–6.2%).

Both are reported because they are not interconvertible in general (`docs/protocol.md`, "Acceptance
quantities"). On this corpus they nearly are, and the residual is measurable rather than hypothetical:
`draft_tokens / drafts` is exactly 3.000 in the open-chat and summarization cells and 2.987–2.988 in
the two code cells, so prompt lookup proposed short only on code, in both deployments. Where it is
3.000 the two quantities coincide exactly; on the code cells `(mean_acceptance_length − 1) / 3`
understates `acceptance_rate` by 0.4% relative (0.1313 against 0.1319 at concurrency 8). Small here,
but it is the wrong conversion, and on the research corpus — longer prompts, more short partial
matches — there is no reason to expect the gap to stay this small.

### Step-cost decomposition: draft cost versus verification cost

Speculation changes tokens per engine step, so per-token latency understates per-step cost. Mean
step time is recovered from the counters as `mean TPOT × T / (T − accepted)`, where `T` is the cell's
completion tokens: total steps are `T − accepted`, because every accepted draft token is a token that
did not need its own step.

Both speculative arms now have two deployments, so every figure below is a range over two independent
deployments rather than a single measurement.

| Workload | Conc. | Baseline step | N-gram step r1/r2 | Draft step r1/r2 | N-gram × r1/r2 | Draft × r1/r2 |
|---|---:|---:|---:|---:|---:|---:|
| open_chat | 1 | 14.13 ms | 18.64 / 18.32 ms | 91.15 / 100.41 ms | 1.32 / 1.30× | 6.45 / 7.11× |
| open_chat | 8 | 14.67 ms | 20.00 / 19.61 ms | 94.24 / 101.27 ms | 1.36 / 1.34× | 6.42 / 6.90× |
| code | 1 | 14.15 ms | 18.13 / 17.45 ms | 101.62 / 104.25 ms | 1.28 / 1.23× | 7.18 / 7.37× |
| code | 8 | 14.70 ms | 19.54 / 18.88 ms | 101.34 / 106.45 ms | 1.33 / 1.28× | 6.89 / 7.24× |
| long_context_summarization | 1 | 14.16 ms | 17.79 / 17.36 ms | 96.63 / 105.52 ms | 1.26 / 1.23× | 6.82 / 7.45× |
| long_context_summarization | 8 | 14.71 ms | 19.42 / 18.98 ms | 91.77 / 98.56 ms | 1.32 / 1.29× | 6.24 / 6.70× |

Verifying a depth-3 proposal costs **1.23–1.36×** a plain decode step, over 12 cells from two
deployments; the second n-gram deployment lands slightly cheaper than the first in all six cells, so
the range widened downwards rather than moving. Proposing with the 8B drafter costs **another
72.3–88.2 ms**, or 4.9–6.2× a full 70B target step, taking the total to 6.2–7.5×. So of the
speculative step cost, **verification and the scheduling change account for roughly a quarter to a
third of one baseline step, and the draft model accounts for about five to six whole baseline steps.**
Draft cost dominates by an order of magnitude over verification cost.

That confirms the draft-plus-verify cost is located in the drafter, and it simultaneously rules out
the intuitive version of the explanation. Three sequential 8B forward passes at TP=4 should cost a
*fraction* of one 70B forward pass, not five to six of them. The measured cost is roughly an order of
magnitude larger than the drafter's arithmetic warrants, so it is dominated by per-step overhead in
the draft path — separate model execution, launch and synchronisation cost, and drafter steps that
appear not to benefit from the target's captured graphs — rather than by drafter FLOPs or
tensor-parallel collectives. Eliminating draft communication entirely (the draft TP=1 cell, which the
pinned vLLM refuses) could therefore not have recovered more than a slice of a cost this large.

The decomposition assumes: the baseline emits exactly one token per step, so its TPOT is its step
time; mean TPOT over a cell is a fair average step cost despite mixing prefill-adjacent and
steady-state steps; verification cost is the same in both speculative arms; and the arms are
comparable across nodes. The third assumption is imperfect and the direction is known —
`draft_model` runs with a 7168-token per-step budget while n-gram runs with the baseline's 8192, so
part of the 72–88 ms gap is budget, not drafter. The fourth is also imperfect, though the n-gram arm's
repeat 1 and draft repeat 2 share node nid006687, which makes that pairing node-matched: against
repeat 2 the gap is 79.1–87.7 ms, wider than against repeat 1, so node effects do not explain it away.
The n-gram arm's own two deployments, on nid006687 and nid007500, put step cost within 1.7–3.9% of
each other in every cell, which bounds how much of the drafter's five-to-six-step cost could be a node
effect: not much.

Job 3392153 on nid006687 re-ran the identical configuration and workload. It reproduces the
regression in every cell and shifts slightly further from baseline. `r1→r2` is repeat 1 → repeat 2;
spread is the absolute difference over the mean of the two repeats.

| Workload | Conc. | Acceptance r1→r2 | TPOT p50 (ms) r1→r2 | Output tok/s r1→r2 | Ratio to baseline r1→r2 | Ratio spread |
|---|---:|---|---|---|---|---:|
| open_chat | 1 | 0.628 → 0.638 | 31.86 → 34.17 | 31.0 → 29.1 | 0.442 → 0.415 | 6.4% |
| open_chat | 8 | 0.633 → 0.628 | 32.54 → 35.54 | 190.6 → 173.2 | 0.441 → 0.401 | 9.6% |
| code | 1 | 0.787 → 0.797 | 30.16 → 30.54 | 33.0 → 32.5 | 0.469 → 0.461 | 1.7% |
| code | 8 | 0.786 → 0.781 | 30.07 → 31.75 | 255.2 → 238.5 | 0.472 → 0.441 | 6.8% |
| long_context_summarization | 1 | 0.631 → 0.605 | 32.72 → 37.39 | 29.4 → 26.3 | 0.419 → 0.375 | 11.1% |
| long_context_summarization | 8 | 0.620 → 0.633 | 31.36 → 33.10 | 212.0 → 200.8 | 0.429 → 0.407 | 5.4% |

Two things matter here. **Acceptance is the most stable quantity in the experiment** — it moves by
0.5–4.3% between independent deployments, so the drafter's quality is a property of the model pair
rather than of a deployment. **Latency and throughput are not** — the ratio to baseline moves by
1.7–11.1%, and repeat 2 is slower in all six cells, which is the signature of a node or drift effect
rather than symmetric noise. `configs/experiment.yaml` sets
`maximum_repeat_spread_fraction: 0.10`; `long_context_summarization` at concurrency 1 breaches it at
11.1% and `open_chat` at concurrency 8 sits just under at 9.6%.

The conclusion is unaffected in direction and should be read with that spread attached: the
speculative arm delivers **0.375–0.472×** baseline output throughput across both deployments, so the
regression is far larger than deployment-level variability. Two repeats give the screening design
the repeats it asks for, but two is the minimum for a spread and too few for a bootstrap confidence
interval, and the baseline still has only one deployment.

### KV-cache cost

| Arm | KV cache | Max concurrency at 131072 tokens/request | Loss vs baseline |
|---|---:|---:|---:|
| baseline | 513,696 tokens | 3.92× | — |
| ngram-n3 (repeat 1 / 2) | 416,640 / 416,640 tokens | 3.18× | −18.9% |
| draft-n3-tp4 (repeat 1 / 2) | 251,472 / 251,488 tokens | 1.92× | −51.0% |

The n-gram arm splits the draft arm's capacity cost in two, and the answer is not the expected one.
**Speculation costs KV cache even with no drafter resident**: n-gram gives up 97,056 tokens of cache
against an identical model and image, presumably to the larger per-step activation footprint of
verifying four positions at once. The drafter's weights then cost a further 165,168 tokens. So the
draft arm's halved capacity is roughly 37% speculation machinery and 63% drafter weights, not purely
the weights.

These are the most reproducible numbers in the study. The two n-gram deployments sized their cache to
**the same 416,640 tokens on different nodes**, byte-for-byte, and the two draft deployments came
within 16 tokens of each other, so the −18.9% and −51.0% capacity costs are deterministic properties
of the configuration rather than run-to-run noise.

## Interpretation

Acceptance is not the binding constraint here, so H1's "acceptance alone is insufficient" case is
what this run demonstrates. Two properties of the pinned vLLM revision change with `method` and
cannot be held constant. The n-gram arm shows they do not apply to the same set of arms, which
matters for reading the earlier draft-versus-baseline comparison:

- **Async scheduling is disabled for both speculative arms, not just `draft_model`.** The baseline
  logs `Asynchronous scheduling is enabled`; the draft arm logs `Async scheduling not supported with
  draft_model-based speculative decoding and will be disabled` and the n-gram arm logs the same
  sentence with `ngram`. So this confound separates speculation from the baseline but is common to
  the two speculative arms and cannot explain any difference between them.
- **The per-step token budget shrink is specific to `draft_model`.** The draft arm logs
  `max_num_scheduled_tokens is set to 7168 based on the speculative decoding settings`; the n-gram
  arm never logs it and runs at the baseline's `max_num_batched_tokens=8192`. This one is asymmetric
  between the speculative arms and inflates part of the draft-versus-n-gram step-cost gap.

Both were re-verified from n-gram repeat 2's own server log rather than assumed to carry over, and
both hold identically: `Async scheduling not supported with ngram-based speculative decoding and will
be disabled` followed by `synchronous scheduling is disabled`, `Chunked prefill is enabled with
max_num_batched_tokens=8192`, and no `max_num_scheduled_tokens` line anywhere in the log. So the
n-gram arm's win over baseline is achieved *while* giving up async scheduling, in both deployments.

With the n-gram arm in hand, the draft arm's regression is no longer attributable to speculation as
such. The verification-plus-scheduling package costs 1.23–1.36× a decode step, which n-gram converts
into a net win where acceptance is high enough, reproducibly. The drafter adds roughly five more baseline steps per
step, and that is what turns speculation into a 2.1–2.7× regression. Async scheduling being disabled
in both arms while n-gram still beats baseline on summarization also bounds that confound: it cannot
be worth more than the margin n-gram wins by.

One unexplained observation: **n-gram has lower TTFT than the baseline in all twelve cells**, p50 and
p95, across both deployments and both nodes. One plausible mechanism is that async scheduling, which
the baseline has and n-gram does not, prepares the next step before the current step's output is
handled and so adds latency to the first token. That is a hypothesis, not a measurement; the clean
test is a baseline relaunched with async scheduling explicitly disabled, which the pinned vLLM
supports through the same config field. Node effects are not separated either, but the effect now
reproduces on two nodes with a between-deployment spread of 0.7–6.4%, against a TTFT gap to baseline
of 4.6–31%, so it is not a single node's quirk.

## Provenance

- Cluster: Clariden (CSCS), partition `debug`, `--time 01:00:00`, 1 node × 4 GH200 per arm.
- Baseline job 3391425 on nid007645, replica head IP 172.28.51.237, started 17:23:47 CEST,
  cancelled 18:07 CEST after measurement.
- Draft job 3391426 (repeat 1) on nid006633, replica head IP 172.28.32.244, started 18:07:36 CEST,
  cancelled 18:51 CEST after measurement. It was queued behind the baseline with reason
  `QOSMaxJobs`, so the arms ran sequentially rather than concurrently.
- Draft job 3392153 (repeat 2) on nid006687, replica head IP 172.28.33.172, started 19:03:40 CEST,
  ready 19:18 CEST, cancelled 19:36 CEST after measurement.
- N-gram job 3392370 (repeat 1) on nid006687, replica head IP 172.28.33.172, started 19:48:00 CEST,
  ready 20:01 CEST, cancelled 20:09 CEST after measurement. Same physical node as draft repeat 2, so
  that one pairing is node-matched; `prompt_lookup_min=1`, `prompt_lookup_max=4`,
  `num_speculative_tokens=3`.
- N-gram job 3403354 (repeat 2) on **nid007500**, replica head IP 172.28.44.144, submitted
  2026-09-14 23:58:13 CEST, ready 2026-09-15 00:10:56 CEST, cancelled 00:18:59 CEST after
  measurement. A fifth distinct node, so this repeat is *not* node-matched to anything: it samples
  node effect and deployment effect together, which is what a deployment repeat is for.
  `prompt_lookup_min=1`, `prompt_lookup_max=4`, `num_speculative_tokens=3`,
  `max_model_len=131072`, `gpu_memory_utilization=0.8`, target TP=4 — identical to repeat 1.
- Served model names:
  `faruk_zahiragic/swiss-ai/Apertus-v1.5-70B-baseline-faruk_zahiragic-20260913T152337Z`,
  `faruk_zahiragic/swiss-ai/Apertus-v1.5-70B-draft-n3-tp4-faruk_zahiragic-20260913T152358Z`,
  `faruk_zahiragic/swiss-ai/Apertus-v1.5-70B-draft-n3-tp4-faruk_zahiragic-20260913T170332Z`,
  `faruk_zahiragic/swiss-ai/Apertus-v1.5-70B-ngram-n3-faruk_zahiragic-20260913T174747Z`, and
  `faruk_zahiragic/swiss-ai/Apertus-v1.5-70B-ngram-n3-faruk_zahiragic-20260914T215813Z`.
- KV cache: 513,696 tokens on the baseline (3.92×), 416,640 on both n-gram deployments (3.18×),
  251,472 and 251,488 on the draft repeats (1.92×), all at `max_model_len=131072`.
- **Both n-gram deployments ran on the stock pinned image with no overlay**, because n-gram loads no
  drafter and the `image_token_index` bug is in the draft-model proposer path. Their image provenance
  therefore matches the baseline exactly, unlike the draft arm.
- Target `swiss-ai/Apertus-v1.5-70B`, draft `swiss-ai/Apertus-v1.5-8B`, both from the Capstor
  cache; target TP=4, draft TP=4, `num_speculative_tokens=3` on every speculative arm.
- `max_model_len=131072` and `gpu_memory_utilization=0.8` on all arms.
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
- Harness revision `270413f22f41ad008a72fffae5359cb3bd03785f` for the baseline and both draft
  deployments, `a32a5e0c1c8fcb53aacf3bb305e2ad44acf5692a` for n-gram repeat 1, and
  `85ccf4dcf65366cd3927f80eebc2fcf046fa1f4d` for n-gram repeat 2. The first two are byte-identical in
  `src/`, `tests/` and `pyproject.toml`. The third adds correctness-gate code and a CLI subcommand and
  leaves the measurement path — `runner.py`, `client.py`, `prometheus.py`, `workloads.py`, `stats.py` —
  byte-identical to the other two, verified by `git diff` over those files, so the n-gram repeat pair
  measured with the same instrument. `apertus-bench` 0.1.0, Python 3.13.9, run from a venv on login
  node clariden-ln004.
- Load generator: on-cluster, hitting the replica node IP directly on port 8080. The public
  gateway was not used, so gateway latency and OpenTela routing are excluded. Every deployment did
  register with the gateway (`replica_health.json` reached `HEALTHY` with a non-null `peer_id`).
- Corpus `workloads/smoke.jsonl`, SHA-256
  `316fb566a18e32305b98929e34fdac515b9a43c6c7329994b319870f879952af`, 6 prompts across 3 strata,
  identical ordered prompt IDs and per-prompt output caps on every arm.
- Sampling: temperature 0.0, top_p 1.0, per-request seed `1 + request_index`, natural EOS.
- Per cell: 8 warmup requests at the cell's concurrency, then 24 measured requests in closed loop,
  with a `/metrics` snapshot immediately before and after the measured window. Repeat 1 was driven by
  `apertus-bench matrix`; draft repeat 2 by `apertus-bench run` per cell so each cell carries its
  deployment repeat index; n-gram repeat 1 by `apertus-bench matrix` again, and n-gram repeat 2 by
  `apertus-bench run` per cell for the same reason as draft repeat 2. The workload, request
  counts, warmup, sampling and load path are identical throughout.
- Measurement windows 2026-09-13T15:35:52Z to 16:35:24Z (baseline and draft repeat 1), 17:19Z to
  17:35Z (draft repeat 2), 18:02Z to 18:09Z (n-gram repeat 1), and 2026-09-14T22:12:26Z to
  22:18:11Z (n-gram repeat 2).

Each deployment answered `/v1/models` and then completed a real short chat completion before any
measurement, as the protocol requires. All five returned the same text for the verification prompt:

> Speculative decoding is an optimization technique in large language models that predicts and
> generates multiple future tokens in parallel before committing to the final output, allowing the
> model to skip over incorrect guesses and accelerate inference.

Full responses with usage are in `provenance/*-first-chat-completion.json`.

## Deviations and caveats

- **Only the baseline has a single deployment.** `configs/experiment.yaml` screening asks for two
  independent deployments per configuration; both speculative arms now have two, and the baseline has
  one. So the baseline contributes no deployment-level spread and **every ratio quoted in this run
  divides by one deployment**. The spread figures here are between-deployment spreads of the
  numerator only, and a baseline repeat would widen them. This is now the single highest-value missing
  cell: it is the cheapest measurement that would firm up the summarization win, more so than a third
  n-gram repeat. Two repeats also support a spread but not the bootstrap confidence interval over
  deployment effects the protocol's analysis section asks for.
- **`maximum_repeat_spread_fraction: 0.10` is breached in one cell of the draft arm**,
  `long_context_summarization` at concurrency 1 (11.1% on the throughput ratio). Treat that cell's
  effect size as the least settled of the draft arm's six; its direction matches the others. All six
  n-gram cells clear the gate (worst 6.6%).
- **Node effects are not separated from `method`.** The five deployments used nid007645, nid006633,
  nid006687, nid006687 and nid007500, at five different times. Draft repeat 2 and n-gram repeat 1
  share a node, so that one pairing is clean; every comparison involving the baseline crosses nodes.
  The protocol treats node/date as a nuisance block; this run cannot. Draft repeat 2 being uniformly
  slower than repeat 1 is consistent with a node effect and is a reason not to read small differences
  as anything else — though n-gram repeat 2 being uniformly *faster* than n-gram repeat 1, on a
  different node again, shows the shift is not a fixed direction and so behaves like a node/deployment
  effect rather than drift in the harness or the cluster.
- **The smoke corpus is not the research corpus.** Its "long context" prompts are 270 and 295
  Apertus tokens (6,780 prompt tokens over the 24 requests of a cell, ~283 per request), far
  below the 16k–65k target in `workloads/README.md`, so the `long_context_summarization` rows here
  measure almost no prefill. That matters most for the n-gram result: prompt lookup is expected to do
  best exactly where there is a long document to copy spans from, so **this corpus probably
  understates n-gram's advantage on real summarization traffic** rather than overstating it. It also
  leaves prefill-heavy behaviour, where speculation has the most room to help TTFT, untested.
- **Exact greedy agreement fails and was the wrong criterion; under the calibrated gate this run
  shows no correctness violation.** Baseline against draft repeat 1 gives 0 of 6 exact matches
  (`../correctness/smoke-20260913-debug/comparison.json`); baseline against draft repeat 2 gives 3 of
  6 (`comparison-repeat2.json`); the n-gram deployments give 2 of 6 and 1 of 6
  (`comparison-ngram-n3.json`, `comparison-ngram-repeat2.json`); and **the
  two identically configured draft deployments agree with each other on 0 of 6**
  (`comparison-draft-repeat1-vs-repeat2.json`). Two deployments of the same speculative configuration
  disagreeing as much as speculation disagrees with the baseline rules out a rejection-sampling
  explanation. Mismatching pairs share a long prefix — 297–564 characters in the repeat-1 comparison —
  and then diverge into a different but equally fluent continuation with completion token counts
  within 3, which is what floating-point nondeterminism in the target forward pass looks like across
  nodes and batch shapes with prefix caching enabled. The gate was reformulated accordingly
  (`docs/protocol.md`, "Greedy correctness gate"): the same captures were re-analysed against the
  draft-repeat pair as a calibration control, and both baseline-versus-draft comparisons land inside
  that control's divergence envelope (`gate-baseline-vs-draft-repeat1.json`,
  `gate-baseline-vs-draft-repeat2.json`). That is consistent with losslessness and is not a proof of
  it; a same-deployment paired check is impossible here because one deployment serves exactly one
  speculative configuration, and no baseline-vs-baseline control exists yet.
- **The n-gram arm now has its own calibration control, and the gate splits on it — which says the
  gate is under-powered at six prompts, not that n-gram is unsound.** The second n-gram deployment
  supplies the same-configuration cross-deployment control the n-gram arm previously lacked
  (`comparison-ngram-repeat1-vs-repeat2.json`: 1 of 6 exact, worst-case normalized edit distance
  0.264). Judged against it, **baseline-versus-n-gram-repeat-1 passes** (worst-case edit distance
  0.308 against an allowance of 0.314) and **baseline-versus-n-gram-repeat-2 fails that one check of
  four** (0.392 against 0.314), while both pass on comparable cases, common prefix and completion-token
  difference: `gate-baseline-vs-ngram-repeat1.json`, `gate-baseline-vs-ngram-repeat2.json`. Two
  deployments of one configuration returning opposite verdicts against an identical control is the
  clearest possible sign that the instrument's resolution is coarser than the effect it is asked to
  detect: the allowance is a worst-of-six order statistic from a single control pair, so it is itself a
  noisy quantity, and adding the draft-repeat pair as a second control does not loosen it because the
  gate takes the tightest control's envelope. **Do not present "the gate passes for n-gram" as a
  result.** What is presentable: the divergence between arms is of the same character and roughly the
  same size as the divergence between two deployments of one arm, and deciding losslessness at this
  resolution needs many more prompts and a baseline-versus-baseline control.
- **The step-cost decomposition is an inference, not a direct measurement.** It rests on the
  assumptions listed with it, and one is known to be violated in a direction that inflates the
  drafter's share: the draft arm's per-step token budget is 7168 against n-gram's 8192. A kernel-level
  profile of the two arms, which the protocol lists as an extension, is what would settle the split.
- Concurrency 32 and the confirmation matrix were not run; there is no `ignore_eos` control and no
  capacity study. The n-gram depth sweep (2, 5, 8) and `prompt_lookup_max` variation are untouched —
  only depth 3 with lookup 1–4 was measured.
- The 24 requests per cell cycle through only 2 prompts per stratum, so latency distributions are
  narrow by construction and request-level percentiles are not independent samples.
