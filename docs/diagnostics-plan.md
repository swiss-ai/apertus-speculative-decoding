**Apertus 1.5 speculative decoding: diagnostic experiments and improvement decisions**

Prepared 2026-09-19. This is a plan for new experiments; no new GPU measurements are claimed.
It supplements [the protocol](protocol.md) and changes the order of work: establish where time
and memory go before running the full depth sweep or training another drafter.

The immediate objective is to determine whether the 8B/70B regression is dominated by a fixable
execution path, unavoidable draft-model cost on this hardware, or both. In parallel, determine
whether n-gram's summarization gain follows reusable text spans and survives realistic context
lengths and serving load. The outcome should be a measured improvement or a defensible stopping
decision for each approach.

Team feedback identifies two specific hypotheses: the 8B model may be too expensive relative
to the 70B target, and separate training may leave their predictions insufficiently aligned
despite a shared vocabulary. Neither is an established cause yet. The experiments below test
draft cost and prediction alignment separately before assessing their combined effect.

**1. Starting evidence and what remains uncertain**

Use the six-deployment, 36-cell
[report](hackathon-20260915-report.md) and
[raw results](../results/70b/smoke-screening-20260913-debug/) as the historical reference. The older
five-deployment narrative is incomplete. Ratios below use the mean of the two baselines where
computed from [analysis.csv](../results/70b/smoke-screening-20260913-debug/analysis.csv).

| Recorded observation | Implication for the next experiment |
| --- | --- |
| Depth-3, TP=4 drafting delivers 0.375–0.473× baseline throughput despite 60.5–79.7% acceptance and mean acceptance length 2.81–3.39. | Prioritize round cost over acceptance training. Acceptance still matters, but improving it alone has limited headroom at depth 3. |
| The report estimates 91–106 ms per draft round versus roughly 14–15 ms per baseline token. | Measure actual engine rounds and their critical path. These are inferred timings, not kernel measurements. |
| Baseline uses async scheduling; both speculative arms disable it. Draft's effective scheduled-token budget is 7168, versus 8192 for baseline/n-gram. | Introduce baselines that match these settings. Record actual active batch shapes: a budget difference does not establish that the budget bound the measured run. |
| N-gram improves summarization throughput by roughly 7–20%, while generally slowing chat/code. | Test prompt/output overlap as the mechanism; task labels alone are insufficient. |
| The two summarization prompts contain 270 and 295 input tokens. | The current result does not establish behavior at 16k–64k context. |
| Advertised KV capacity falls from 513,696 tokens to 416,640 with n-gram and approximately 251,480 with the drafter. | Attribute memory allocations and measure usable serving capacity; n-gram also has a substantial memory cost. |
| Draft TP=1 fails before serving; greedy outputs vary even across baseline deployments. | Keep unsupported TP layouts out of the initial sweep; strengthen correctness controls before judging new implementations. |

Treat the report's allocation of the draft-minus-n-gram difference to drafting as a hypothesis.
N-gram also incurs lookup, verification, sampling, cache bookkeeping, and scheduler costs;
its proposal coverage and verification shapes can differ from the neural drafter. Subtracting
the arms does not isolate the 8B forward pass. Parameter counts alone cannot rule out slow
kernels, communication, or poor utilization.

The pinned proposer already contains CUDA-graph dispatch logic, so “drafting has no graphs”
also needs measurement. Inspect the executed branches and graph coverage in
[the pinned proposer](https://github.com/swiss-ai/vllm/blob/a601a9d998ddeb488f0c17e8512874b116aa7658/vllm/v1/spec_decode/llm_base_proposer.py).

The team's size concern is about **measured latency and memory**, not a universal parameter-ratio
cutoff. Although 8B is approximately 11% of 70B by parameter count, a depth-3 round runs multiple
sequential draft steps, pays verification and coordination costs, and reserves another model's
weights and KV cache. Its latency need not scale with its parameter count. P1–P3 below establish
whether the cost comes from the embedded execution path or persists in an efficient 8B execution.

The separate-training concern is about **conditional prediction agreement**. A shared vocabulary
allows direct proposal verification once token IDs and special-token behavior are confirmed;
it does not make the models assign the same probabilities or choose the same continuations.
Separate training is compatible with speculative decoding: the core algorithm supports
off-the-shelf models without retraining. Target-aligned training is therefore a possible
performance improvement, not a prerequisite for correctness.
[Leviathan et al.](https://proceedings.mlr.press/v202/leviathan23a.html)

**2. Establish the break-even target before optimizing**

For a single active request in steady-state decoding, define:

- `t0`: baseline time per committed output token, with the same context and controlled settings.
- `g(k)`: mean committed output tokens per complete speculative round at maximum depth `k`.
- `t_round(k)`: wall time of that round, including drafting, verification, sampling, cache
  reconciliation, and exposed CPU/communication time. Count overlapping operations once.

Then `decode_speedup ≈ g(k) × t0 / t_round(k)`. A round pays off when
`t_round(k) < g(k) × t0`. This applies the cost/acceptance reasoning of
[Leviathan et al.](https://proceedings.mlr.press/v202/leviathan23a.html) to the measured system.

At the observed depth-3 acceptance, the approximate break-even round cost is **40–50 ms**.
The inferred 91–106 ms therefore needs roughly a halving, subject to direct measurement.
Even perfect acceptance yields at most four tokens per round: if the reported 6.2–7.5× round
cost is confirmed, its optimistic speed ceiling at unchanged cost is only **0.53–0.65×**.
This is the reason to profile before investing in acceptance training.

At unchanged depth and round cost, increasing the observed mean acceptance length from
2.81–3.39 to its maximum of 4 offers only **18–42% more tokens per round**. That cannot by itself
close the recorded throughput gap. It does not rule out alignment helping after execution costs
fall, or enabling a better depth; those possibilities require remeasuring both cost and acceptance.

Record real committed tokens and proposal coverage. `1 + accepted_tokens / drafts` describes
the usual speculative-round convention, but omits rounds with no proposal and can differ from
delivered tokens at EOS/output limits. Neither `mean TPOT × T/(T−accepted)` nor SSE chunk gaps
are direct engine-round measurements. At concurrency above one, trace batch membership and
committed tokens per engine iteration; do not equate request-round counts with GPU iterations.

For longer inputs, separate prefill from decode. If only decoding improves by `S_decode`, an
illustrative service-time model is `T_new = T_prefill + T_decode/S_decode + T_extra`, including
any draft prefill. Long input does not automatically imply a larger end-to-end win. TTFT
improvements need their own scheduler/cache/prefill explanation.

**3. Common controls and instrumentation**

Keep target weights, tokenizer/chat template, target precision and TP=4, GPU-memory utilization
0.8, and `max_model_len=131072` fixed initially. Capture immutable checkpoint revisions and
tokenizer hashes, not only mutable model paths. Use the pinned image plus a recorded Apertus
compatibility patch consistently across compared arms after checking baseline behavior.
Save the resolved engine configuration, container digest, source diff, CUDA/PyTorch/NCCL versions,
attention backend, compile/graph settings, GPU topology, node ID, and power/clock policy.

Use two distinct workloads for distinct questions:

- **Mechanistic measurements:** greedy decoding, fixed 256 generated tokens using `ignore_eos`,
  prefix caching explicitly off, short prompts, no unrelated traffic, and stable warmed shapes.
  Use teacher-forced prefix replay for exact shape matching where needed. Repeat on natural EOS
  before interpreting a performance gain as useful to users.
- **Serving measurements:** representative prompts, natural EOS, production sampling parameters,
  and separately declared cold-cache and realistic warm-cache regimes. Do not estimate cold
  prefill by cycling two already cached prompts. Keep source documents disjoint across tuning
  and confirmation sets.

Start timing after model readiness, compilation, and graph warmup. Drain queues before taking
counter snapshots and between cells. Counter resets, missing token usage, and success below
99% invalidate a comparison. Record runtime variability rather than removing slow repeats.

| Measurement | Required detail |
| --- | --- |
| Client metrics | TTFT, E2E, TPOT p50/p95, completion lengths, successful output tokens/s, errors and stream-event gaps; gaps remain event-level observations. |
| Round trace | Request/iteration IDs, context length, active/padded batch sizes, prefill/decode token counts, attempted/proposed/accepted/committed tokens, no-proposal rounds, fallback/EOS reasons. |
| CPU/GPU timeline | Draft prefill and each draft step; target verification; logits/sampling; KV rollback/catch-up; scheduling; host/device copies; synchronization and collective waits. |
| Execution efficiency | Graph replay versus eager execution, graph breaks/recompilation, kernel count and duration, attention/GEMM/logits time, GPU idle gaps, HBM activity, SM activity and NCCL time across all four ranks. |
| Memory/load | Per-rank weights, target/draft KV, activation/workspace peaks, graph pools, reserved/allocated memory; cache occupancy, preemptions, waiting/running requests, queue growth and power. |

Use short Nsight Systems captures for the execution timeline and targeted PyTorch profiles for
operator shapes and allocation attribution. Avoid inserting a device synchronization around
every timed operation. Capture all ranks, inspect the slowest-rank dependency, and do not sum
overlapping kernel durations into wall time. Compare instrumented and uninstrumented runs;
all headline performance numbers come from unprofiled runs. Verify profiler options in the
pinned build; current [vLLM profiling documentation](https://docs.vllm.ai/en/latest/contributing/profiling/)
is guidance, not a guarantee that every current option exists in that image.

**4. First experiment block: reproduce and isolate configuration effects**

First reproduce the three historical arms on the smoke set. Then run the following controlled
six-configuration block with prefix caching off and the fixed-output workload. Use concurrency
1 and 8, one 512–1024-token prompt-length bucket, 32 distinct prompts per cell, and two independent
deployment blocks. This is **24 screening cells**, excluding historical reproduction and profiles.
Randomize variant order within node/date blocks and bracket long blocks with a baseline.

| ID | Configuration | Comparison and question |
| --- | --- | --- |
| B0 | Plain 70B, async on, effective token budget 8192 | Operational reference. |
| B1 | Plain 70B, async off, effective budget 8192 | B1 vs B0: how much do scheduling and output delivery explain? |
| B2 | Plain 70B, async off, effective budget 7168 | B2 vs B1: does the reduced budget matter at the actual batch shape? |
| N3 | N-gram, depth 3, lookup 1–4, async off, effective budget 8192 | Reproduces the n-gram execution configuration. |
| N3m | N-gram, depth 3, lookup 1–4, async off, effective budget 7168 | N3m vs N3: budget sensitivity; N3m vs B2: total n-gram-path cost/benefit. |
| D3 | 8B draft, depth 3, TP=4, async off, effective budget 7168 | D3 vs B2: total draft-path effect after matching these controls. |

“Effective budget” means the value after vLLM's reservation rules, confirmed from logs and
observed scheduling. Do not assume passing `max_num_batched_tokens=7168` makes every method
equivalent. Record reserved lookahead slots and `max_num_seqs` as well. If exact matching is
unsupported, document it and compare fixed active token/batch shapes in a microbenchmark;
do not silently bypass a scheduler invariant. Add a mixed prefill/decode workload if the small
decode-only cells never approach either budget.

This block estimates configuration effects; it does not force all memory allocations or internal
batch shapes to match. Keep B0 in every final comparison so a fix must beat the best plain
target configuration, even if it beats B2 more easily.

**5. Second block: locate the draft cost and test whether it is recoverable**

Run these diagnostics in priority order. Each should produce a time breakdown, a controlled
intervention, and a predicted versus observed change in round cost.

| ID | Experiment | Evidence to look for | Improvement to test if supported |
| --- | --- | --- | --- |
| P1 | Profile B2, N3m and D3 at concurrency 1, then 8, using warmed short-context requests. | Exposed CPU gaps; eager draft kernels; padded work; NCCL waits; draft attention/logits; target verification; KV catch-up. | Change the dominant component first; attach traces to an engine issue or patch. |
| P2 | Run the 8B model alone at TP=4 with matched context, batch, precision and draft execution settings. Compare one-token latency with each embedded draft step. Test graph/eager controls separately where supported. | A fast standalone 8B but slow embedded path indicates integration or different execution settings. Both slow points to the model/kernel/layout floor. | Restore effective graph capture, eliminate host synchronization/copies, reduce padding, or optimize the slow operator. Preserve output semantics. |
| P3 | Replay precomputed proposals through the real verifier at fixed prefixes and shapes: target-correct proposals, deliberately first-token-wrong proposals, and recorded 8B proposals. | Verification floor, rejection/rollback cost and achievable gain with negligible proposal generation. | Optimize verification/cache reconciliation if it already consumes the available budget; otherwise prioritize the drafter. |
| P4 | Sweep draft depth `1,2,3,5,8` on the same prefixes and active batches; start with `1,3,5`, add the rest only if useful. | Round-cost slope with depth versus per-position accepted-prefix survival; fixed setup cost versus sequential draft cost. | Select a shallower depth, or adaptive depth if gains differ by load/context. |
| P5 | Only if traces show exposed collective cost: benchmark 8B TP=1/2/4 standalone, then a supported integrated layout. | A measured communication saving large enough to meet the break-even budget, including transfer and synchronization overhead. | Use a correctly implemented smaller draft TP or alternate placement; account for GPU imbalance, added hardware and KV memory. |

Precomputed-proposal replay is a diagnostic bound, not a deployable result: count its materialization
cost, keep the target verifier unchanged, and label its privileged knowledge. If a “draft loaded
but bypassed” control is implemented, retain the same target execution path and record its memory
reservation; it helps distinguish residency cost from execution cost. Never interpret skipping
verification as an optimization.

The current launcher rejects depth 1; enable it in a diagnostic launcher only after checking
engine support. Integrated draft TP=1 is explicitly unsupported at the pin, as confirmed by
[the failure artifact](../results/70b/deployment-failures/draft-n3-tp1-3392110/) and
[the source guard](https://github.com/swiss-ai/vllm/blob/a601a9d998ddeb488f0c17e8512874b116aa7658/vllm/v1/spec_decode/draft_model.py).
Deleting that guard is not a valid TP experiment. A newer engine or a proper implementation
must pass startup, cache isolation and correctness checks first. Standalone TP=1 latency is
only a bound on potential savings, not evidence that the integrated setting works.

For each depth, compute `g/t_round`, plus the hypothetical ceiling `(k+1)/t_round`. Increasing
depth helps only if the added committed tokens per added time exceed the current `g/t_round`.
Use the empirical accepted-prefix distribution, not an independent-token assumption such as
`acceptance_rate**position`. If the measured cost floor cannot beat `g × t0`, stop tuning that
layout and carry forward the simpler baseline or a cheaper proposer.

**6. Third block: determine whether acceptance warrants model work**

Test the separate-training hypothesis at the same prefixes, rather than comparing independently
generated answers. Two models can give equally good answers while choosing different words,
which is enough to reduce speculative acceptance.

Use at least 100 distinct development prompts per main workload, including relevant production
languages. On target-generated prefixes, measure draft/target top-1 agreement, first rejection
position, accepted-prefix survival, EOS behavior, and acceptance by context/output position.
Verify identical token IDs and special-token/template handling, not just equal vocabulary size.
Replay a subset through the online path to check that offline estimates predict online acceptance.

For sampled decoding, include the intended production temperature and top-p (0.8 is a useful
additional condition if relevant). Estimate distribution overlap `sum_v min(p(v), q(v))` at the
same prefix using the actual proposal and target distributions after sampling transforms; greedy
top-1 agreement and KL alone do not determine sampled acceptance. Inspect whether this engine
uses greedy or probabilistic proposals before defining `q`. Do not assume the same random seed
produces identical sampled outputs across algorithms.

Only commission a smaller or distilled drafter if the measured cost/acceptance frontier suggests
it can win. Compare proposed alternatives by committed tokens per round divided by round time,
memory, and training cost. A faster sub-8B drafter with somewhat lower acceptance may dominate
8B. Distillation should target 70B continuations on disjoint training data and be evaluated on
held-out workloads, including languages that would be easy to miss in an English-only corpus.

Use the following comparisons to separate size and target alignment. Begin with existing
compatible checkpoints and offline measurements. Commission training pilots only after P1–P3
and the break-even analysis identify a plausible winning configuration; the table is not a
requirement to train all four candidates.

| Candidate | Size and training | What its comparison establishes |
| --- | --- | --- |
| A | Existing Apertus 1.5 8B, unchanged | Reference draft latency, memory and accepted-prefix distribution. |
| B | Same 8B architecture/tokenizer, trained to match the 70B target | B vs A tests whether target alignment increases acceptance at essentially the same model size. Measure execution time rather than assuming it is unchanged. |
| C | Substantially smaller compatible drafter, without target-specific distillation | C vs A measures the speed/memory versus acceptance trade-off of the smaller candidate. A different architecture/backend makes this a package comparison, not a pure size effect. |
| D | Same smaller architecture/tokenizer as C, with target-aligned training | D vs C tests alignment at fixed size; D vs B tests whether a cheaper aligned drafter produces better end-to-end results. |

A range such as 0.5–3B is a candidate design space, not a claim that suitable Apertus-compatible
weights exist. Verify availability, token mapping and integrated runtime support before selecting
a checkpoint. Hold target weights, prompts, sampling, depth, TP/layout and serving settings fixed
for initial comparisons; report and justify unavoidable changes. Plot all candidates by measured
round time, accepted length and KV cost, then compare their best validated serving settings.

For alignment pilots, use disjoint 70B-generated training continuations or teacher distributions
and evaluate on held-out target prefixes and online rollouts. Keep training data and budgets
comparable across pilots. If claiming a causal benefit from target alignment specifically, add a
matched fine-tuning control without target supervision: A versus B alone also changes training
exposure. An improvement would support the proposed remedy; it would not by itself prove that
separate pretraining caused the original regression.

Draft-only quantization is a separate candidate when kernels and the integrated draft loader
support it: measure both latency and acceptance, and keep the target precision fixed. EAGLE-like
or other auxiliary-head proposers are a later research branch requiring Apertus-compatible
training and runtime support; method availability in
[vLLM's overview](https://docs.vllm.ai/en/latest/features/spec_decode/) does not establish that
an Apertus checkpoint or a compatible implementation is available.

**7. Fourth block: explain and extend the n-gram result**

The working hypothesis is reusable input/output spans. This follows the mechanism of
[prompt lookup decoding](https://github.com/apoorvumang/prompt-lookup-decoding), but it has not
been isolated by the two smoke summarization prompts. Longer documents may offer more useful
matches and may also increase lookup/attention cost and prefill's share of latency.

Build paired conditions using the same underlying documents or code and matched output caps:

| Workload condition | Mechanistic contrast |
| --- | --- |
| Extractive summary versus paraphrased/abstractive summary | Similar task and source; different expected span reuse. Measure realized overlap rather than trusting the instruction label. |
| Grounded answer/entity extraction versus open-ended response | Input-supported copying versus novel generation. |
| Edit/complete supplied code versus generate code from a short instruction | Tests whether the observed code loss comes from lack of reusable source code. |
| Summarize in the source language versus translate the summary | Tests whether a workload label predicts benefit when copying becomes less useful. |

Start with 1k and 16k inputs, concurrency 1 and 8, at least 30 distinct sources per selected
condition. Extend promising settings to 32k/64k inputs, output caps 256/512, and concurrency 32.
Use actual templated Apertus token counts. For length scaling, include relevant source material
and a separately labeled distractor-length control; merely repeating filler changes overlap.
Do not run a full product of every factor initially.

Instrument lookup attempts, nonempty proposal rate, actual proposed length, matched span length,
match location (prompt or generated history), lookup time and accepted-prefix length. Measure
token-level output span coverage by prompt versus generated history separately. Break lookup
time down by context length and CPU/GPU transfers; measure verification even after unhelpful
matches. Report no-match rounds in the denominator of committed tokens per engine iteration.

At depth 3, compare lookup ranges `1–4`, `2–4`, and `3–8`; then sweep depth `2,3,5,8` for the best
range. The current lookup range `1–4` remains the control. Larger minimum matches may improve
precision while reducing proposal coverage; select on net time saved. Repeat the winning
configuration on held-out sources.

If copying predicts the win, test routing n-gram only to supported workloads. If lookup cost
dominates at long context, test bounded/indexed lookup or an alternative supported proposer.
If benefits change with batch load, test a load-aware fallback to ordinary decoding. Any router
must use information available before or during generation, include its own overhead, and be
evaluated on the real traffic mixture. Retrospective output overlap can explain a result but
cannot itself be a pre-request routing feature. Per-request method switching may require engine
work or separate pools; do not assume it is currently supported.

**8. Memory, correctness and final serving confirmation**

Attribute the 19% and 51% KV-capacity losses before calling either inherent. Capture memory after
weights load, memory profiling, graph capture and steady-state execution. Separate target/draft
KV bytes and cache-group allocation rules, graph pools, activation/workspace reservations,
allocator fragmentation and the most constrained rank. For an equal-capacity diagnostic, cap
baseline KV capacity to the candidate's value; separately retain each arm's native allocation
for the operational comparison. A loaded drafter continues to consume memory even when drafting
is disabled dynamically.

Probe real 16k/32k/64k requests until preemptions or SLO failures occur. Advertised maximum
concurrency at 131072 tokens is not measured serving capacity. Smaller graph-capture ranges or
token budgets may recover KV space while reducing batching performance, so retest both latency
and capacity. Change maximum context or target/KV precision only as explicit additional arms.

Correctness work accompanies every execution change:

- Capture at least 100 diverse greedy prompts twice within each deployment and across baseline
  and candidate deployments, with identical input tokenization. Measure token-level first
  divergence and use shared-prefix target-logit checks to investigate early disagreements.
- Include no-match, immediate rejection, full acceptance, EOS/stop boundaries, output limits,
  long context, mixed batches and cache rollback cases. Validate exact verifier/sampler behavior
  on controlled small distributions where expected results are known.
- Keep the existing calibrated text-divergence gate as a smoke diagnostic. Its six-prompt,
  loosest-control envelope does not prove distribution preservation or establish that a failing
  n-gram case is harmless. Investigate unresolved failures before recommending deployment.
- For stochastic sampling, compare repeated-sample distributions at fixed prefixes and task
  quality with predeclared tolerances and sufficient samples to resolve them. Neither seed
  equality nor fluent text is a correctness test.

Confirm only baseline plus at most one surviving draft setting and one n-gram setting. Use five
independently launched, counterbalanced deployment blocks and disjoint held-out data, concurrency
1/8/32, natural EOS, and relevant sampling settings. Start with 100 unique prompts per reported
stratum and extend if tail metrics or effect intervals are too imprecise. Cluster comparisons
by deployment/node block and account for source-document clustering in paired text conditions;
do not treat cycled requests as independent replicates. Report absolute metrics and paired
ratios with 95% intervals. Five repeats are a starting point, not guaranteed statistical power.

Then use the planned open-loop Poisson load sweep in [the protocol](protocol.md): locate the
maximum request rate satisfying TTFT/TPOT p95, errors, and bounded queue growth, then confirm
near the boundary with repeated soaks of at least 10 minutes. Verify that the load generator
is not limiting arrivals. Compare the same hardware budget, include any extra draft devices,
and report GPU-seconds and, where available, energy per completed request/output token.

Predeclare service-specific absolute SLOs before the capacity sweep. Until those are supplied,
use the following proposed research decision rule: a candidate should deliver at least a 10%
point-estimate improvement in its declared primary metric (TPOT or sustainable request rate),
with the 95% interval excluding no improvement, at least 99% request success, and no more than
5% regression in the other declared latency/capacity guardrails. These are proposed thresholds,
not existing production requirements. A workload-specific win is acceptable if routing and its
capacity cost are measured; otherwise the candidate must win on the declared traffic mixture.

**9. How results turn into improvements**

| Diagnostic outcome | Next change | Acceptance criterion for the change |
| --- | --- | --- |
| Draft integration, graph breaks or CPU synchronization dominates | Focused engine patch or compatible engine upgrade, keeping weights fixed | The targeted trace component shrinks, total round cost falls as predicted, correctness holds, and the candidate beats B0 in unprofiled confirmation. |
| Draft collectives dominate | Supported smaller draft TP or revised placement | Exposed collective savings exceed transfer/imbalance costs; comparison includes memory and total hardware. |
| The 8B compute/logits floor is too high | Cheaper drafter, supported draft quantization, or a trained auxiliary proposer | Its measured `g/t_round` and serving capacity improve despite any acceptance reduction. |
| Acceptance is poor only in specific languages/tasks/positions | Target-aligned training or adaptive depth | Held-out accepted length improves enough to cover measured drafting cost without hiding losing strata. |
| Verification or KV reconciliation dominates | Fix verification kernels, padding or cache handling; reduce depth | Oracle/replay and ordinary runs both show the predicted reduction, with rejection/EOS correctness intact. |
| N-gram wins only with high span reuse | Tune lookup/depth and use a validated workload route | Gains reproduce on new documents and the routed traffic mix; router overhead and fragmentation are included. |
| Gains disappear under load or KV pressure | Disable speculation above a measured load threshold; reduce reservations if useful | Sustainable request rate and tail latency meet the declared gates. |
| No supported change crosses break-even economically | Retain plain 70B; keep n-gram only where confirmed useful | Publish the measured floor, confidence bounds and conditions that would justify revisiting the decision. |

An engine upgrade needs a fresh plain-70B baseline on that same revision. Attribute individual
fixes when possible; do not credit speculation for an unrelated target-kernel improvement.

**10. Execution order, effort and deliverables**

| Order | Work package and responsible role | Expected deliverable / stop point |
| --- | --- | --- |
| 1 | Benchmark owner + serving engineer: provenance, controls and instrumentation; roughly 1–2 working days | Reproducible six-configuration block, trace hooks and valid effective configurations. |
| 2 | Serving engineer: round profiles, standalone 8B and proposal replay; roughly 2–3 days | Per-component timings and a break-even assessment. Stop broad draft sweeps if no plausible fix closes the gap. |
| 3 | Benchmark/model owners: targeted depth/acceptance and n-gram contrasts; roughly 3–5 days | One or two supported fixes and a workload-specific n-gram hypothesis tested on real contexts. |
| 4 | Owners of the identified bottleneck: implement and remeasure the smallest promising change | Before/after traces, correctness evidence, and an unprofiled performance result. Training or substantial engine development is separately estimated. |
| 5 | Benchmark owner: independent confirmation and capacity study; roughly 2–4 days plus queue time | Deployment recommendation with latency, throughput, memory/cost, uncertainty and limitations. |

These are planning estimates, not a booked allocation. Measure startup and cell duration in the
pilot before reserving the larger matrix. Budget node-hours as the sum of startup/warmup, cell,
profiling and soak durations for each deployment; one Clariden node-hour here uses four GPU-hours.
Keep an experiment ledger with hypothesis, controlled variables, run IDs, result, uncertainty,
next action and a predeclared stopping rule.

Save each new deployment under `results/diagnostics-<date>/<experiment-id>/<deployment-id>/`, with
the resolved configuration, logs, request rows, metric snapshots, traces and memory accounting.
The final report should include a round-time breakdown, speedup versus measured `g/t_round`,
acceptance by position, n-gram gain versus overlap/context length, and SLO capacity versus KV cost.

Implementation needed before running this plan is explicit:

- Extend diagnostic launch configuration to control scheduling, effective budgets, prefix cache,
  graph mode and profiling; current launchers do not expose all of these factors. Verify options
  against the pinned binary and save the resolved configuration. Do not overload unrelated
  `sml` arguments with engine flags.
- Add per-round/lookup instrumentation and a verifier replay harness; existing Prometheus counters
  cannot provide the required causal timing breakdown.
- Prepare versioned development/confirmation corpora and source manifests. Store extra corpus
  attributes in a sidecar keyed by prompt ID, or extend the current workload schema deliberately.
- Add block-aware analysis. The current `analyze` command pools baseline cells by workload and
  concurrency, which would mix B0/B1/B2 controls; analyze these in separate roots until comparison
  grouping includes the diagnostic configuration.
- Relaunch between independent deployment repeats. `apertus-bench matrix --repeats N` currently
  repeats against the same live server; it does not create N independent deployments.

The first actionable deliverable is the controlled round profile and break-even decision. It
determines whether to spend the next allocation on an engine fix, a cheaper drafter, or n-gram
workload selection.
