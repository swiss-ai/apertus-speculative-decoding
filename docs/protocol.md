# Experimental protocol

## Objective and decision rule

The operational question is not whether speculative decoding can be faster in one favorable
prompt. It is whether one configuration improves the production-relevant performance envelope of
Apertus-v1.5-70B enough to justify its compute, memory, and operational complexity.

Recommend enabling a setting only if confirmation runs show a repeatable gain on at least one
declared production workload without a material regression on the others, and the capacity study
shows no unacceptable tail-latency or KV-cache penalty. A workload-specific recommendation is
valid when traffic can be routed by workload; otherwise evaluate the declared traffic mixture.

## Hypotheses

- H1: speedup is positively associated with acceptance rate, but acceptance alone is insufficient
  because draft forward passes and synchronization have nonzero cost.
- H2: code and summarization have higher acceptance and larger gains than open chat.
- H3: optimal speculative depth decreases as concurrency rises because target-model batching
  becomes more efficient and draft overhead competes for GPU time.
- H4: TP=1 reduces draft communication but introduces memory/compute imbalance; TP=4 can win at
  higher concurrency despite collective overhead.
- H5: n-gram speculation is competitive on prompts whose outputs reuse prompt spans, especially
  summarization, with much smaller memory cost.

## Experimental factors

The target model, tokenizer, vLLM image, target TP=4, maximum context, GPU-memory utilization,
chat template, request corpus, request order, and generation parameters are controlled.

Two serving properties cannot be held constant across methods in the pinned vLLM revision, so they
are confounded with `method` and must be reported with every comparison:

- asynchronous scheduling is enabled by default but is switched off automatically for
  `draft_model` speculation (`Async scheduling not supported with draft_model-based speculative
  decoding and will be disabled`), so the baseline overlaps CPU scheduling with GPU execution and
  the draft-model arm does not;
- the scheduler shrinks its per-step token budget for speculation
  (`max_num_scheduled_tokens is set to 7168 based on the speculative decoding settings`, against
  `max_num_batched_tokens=8192` on the baseline), and vLLM itself warns this may be suboptimal.

Both effects work against the speculative arm independently of acceptance, so a measured
regression is a regression of the shipped configuration, not evidence that speculation itself is
slower. Do not describe such a result as an acceptance-driven effect without separating them.

The manipulated serving factors are:

| Factor | Levels |
|---|---|
| method | baseline, Apertus-8B draft, n-gram |
| speculative tokens | 2, 3, 5, 8 |
| draft TP | 1, 4 |
| workload | open chat, code, long-context summarization |
| concurrency | 1, 8, 32 |

Greedy decoding is the core experiment. Temperature 0.8, the thinking variant, kernel profiling,
and detailed memory-capacity experiments are extensions after the core conclusion is stable.

## Staged design

### Screening

Run all draft depths × draft TP settings and all n-gram depths at concurrency 1 and 8. Use two
independently launched deployment repeats. The purpose is to eliminate settings that are
consistently dominated, not to make final claims.

### Confirmation

Select candidates using a rule fixed before inspecting confirmation results: retain the best
draft configuration per workload plus any setting within 5% of it that has lower memory cost or
better worst-workload behavior. Compare those candidates, baseline, and the best n-gram setting at
concurrency 1, 8, and 32 over five independent deployment repeats.

Randomize or counterbalance variant order. Record physical node IDs and treat node/date blocks as
nuisance variables. Re-run an interleaved baseline often enough to detect cluster/runtime drift.

### Serving capacity

Use `swiss-ai/bench-agent` for an open-loop Poisson arrival-rate search on baseline and the final
candidate. Declare TTFT p95, TPOT p95, error-rate, backlog, and load-generator-capacity gates before
the search. Confirm the knee with repeated soaks. Keep this result separate from fixed-concurrency
latency/throughput results: the two designs answer different questions.

## Workloads

Use versioned, deterministic prompt strata. The same ordered prompt IDs and per-prompt output caps
must be used for every configuration. Store the corpus digest with each result.

- Open chat samples varied natural instructions and conversation turns.
- Code uses generation tasks rather than only explanation questions.
- Long-context summarization uses 16k–64k token documents with a bounded output.

Measure the actual Apertus-tokenizer input length distribution and report min/median/p95/max.
Natural EOS is primary. A separate `ignore_eos` experiment may fix output lengths to isolate
serving mechanics, but its results must not be pooled with natural generation.

## Run procedure

1. Record Git revisions, container path/digest, any source file bind-mounted over the image, launch
   command, Slurm job and node IDs, model paths, GPU clocks/power policy when available, and the
   corpus digest.
2. Wait for model registration, then require a real short chat completion.
3. Capture greedy outputs for correctness on the smoke set.
4. Warm up the same workload and concurrency shape before collecting counters or timing.
5. Snapshot `/metrics`, run a fixed request count in closed loop, then snapshot again.
6. Save request rows, metric snapshots, and logs. Reject runs with server restarts/counter resets.
7. Drain outstanding work before the next cell. Stop idle jobs promptly.

The load generator should run on-cluster to minimize uncontrolled network variance. One client
process must be shown capable of driving the server at the largest tested load without CPU,
socket, or scheduling saturation.

## Outcomes

Primary outcomes:

- client-observed TTFT p50/p95;
- client-observed TPOT p50/p95;
- aggregate output tokens/s;
- sustainable request rate under declared SLO gates.

Mechanistic outcomes:

- draft acceptance rate and mean acceptance length;
- per-position acceptance vector;
- GPU memory used and advertised KV-cache capacity;
- preemptions, queue depth, GPU utilization, SM/tensor activity, power, and communication metrics;
- initialization time and failure modes.

## Analysis

The independent unit is a deployment repeat, not an individual request. Report every repeat,
median effect, spread, and a 95% bootstrap confidence interval over deployment-level effects.
Present absolute metrics alongside speedup ratios.

Plot speedup and TPOT change by workload/concurrency, acceptance versus speedup, and memory/capacity
changes. Model the interaction between acceptance, speculative depth, draft TP, workload, and
concurrency only as an explanatory analysis; do not let a fitted aggregate hide losing strata.

The final recommendation must state where the setting wins, where it loses, its memory cost, and
whether configuration selection was performed on the same samples used for final confirmation.

## Validity and failure handling

- Reject cells below 99% success rather than averaging only fast successful requests.
- Distinguish deployment unavailability and load-generator exhaustion from model saturation.
- Reject a metric window if cumulative speculative counters decrease.
- Disclose missing usage records; token throughput is invalid without exact token counts.
- Do not call individual streamed chunks tokens. vLLM may emit several accepted tokens together.
- Exact greedy agreement is an integration sanity check, not proof of the lossless theorem.
- Sampling comparisons require identical seeds and parameters but may not be bitwise identical
  across separate distributed executions; compare distributions when studying temperature.
- Report selection bias, workload licensing, gateway/network effects, and node-to-node variability.

## Deliverables

- reproducible launch, workload-preparation, benchmark, and analysis code;
- raw request-level results, metric snapshots, launch logs, and provenance;
- a report explaining both performance results and mechanisms;
- a recommendation for production serving;
- if positive, a `model-launch` pull request updating the example's tuned defaults.
