# gpt-oss-20b reasoning effort experiment plan

Use `openai/gpt-oss-20b` to test whether native **low, medium and high reasoning
effort** changes draft acceptance and speedup, then conditionally test whether
effort-specific draft weights improve on a shared drafter.
The model is open-weight under Apache 2.0 and supports these three effort levels.
[Official OpenAI model documentation](https://developers.openai.com/api/docs/models/gpt-oss-20b)

This addresses the initial open-model investigation in Yu's
[26 October 2026 DM](https://swissai-initiative.slack.com/archives/D0C2VQJ46G1/p1793011705342819),
to inform how to provide drafters for Apertus 2.0. The native effort requirement
determines the model choice. Yu's confirmed previous testing in
[#swiss-ai-spec-decode](https://swissai-initiative.slack.com/archives/C0C2QMYCQ5B/p1790155748119649)
concerns Qwen3-8B; those messages do not establish prior gpt-oss-20b testing by Yu.

**1. Question and scope**

Working hypothesis: changing the target's reasoning effort changes its generated
distribution enough that a fixed drafter's acceptance, and consequently its
speedup, may depend on effort. Measure that sensitivity before choosing how to
provide drafters.

Set `reasoning_effort` to `low`, `medium` or `high` in a single chat-completion
request. The target's Harmony prompt carries the corresponding reasoning setting,
and the target generates its analysis and final answer naturally. Use the same
task messages at every effort. A common output cap is a safety bound and does not
define the effort levels.
[OpenAI Harmony format](https://developers.openai.com/cookbook/articles/openai-harmony)

Phase 1 measures one released drafter across native effort levels. Phase 2 runs
only if Phase 1 shows material effort sensitivity, and compares effort-specific
fine-tuning with mixed-effort fine-tuning. Delta-weight serving, offloading and
Apertus 2.0 integration remain outside this experiment. Stable Phase 1 results
can justify stopping this bounded investigation; they cannot establish that
specialists would never improve it.

**Prior work and its effect on the experiment**

- **DFlash:** the authors report GPT-OSS-20B measurements at medium effort and
  state that low effort gives higher acceptance length. Their released head is
  therefore the Phase 1 baseline. Measure low/medium/high under one local protocol;
  effort-dependent acceptance is an existing observation, not a novelty claim.
  The model card does not provide the specialist-versus-mixed-effort comparison.
  Its hardware, sampling and task sets differ from this protocol, so numerical
  agreement with its speedups is not a pass criterion.
  [Paper](https://arxiv.org/abs/2602.06036),
  [GPT-OSS-20B results](https://huggingface.co/z-lab/gpt-oss-20b-DFlash#evaluation)
- **TAPS:** this preprint compares task-specialist and mixed-data drafters,
  checkpoint averaging, routing and merged-tree verification. It finds domain
  specialization and improved robustness from mixed training; inference-time
  composition outperforms naive averaging by acceptance length in its experiments.
  Use its controlled specialist/mixed comparison as a design precedent for Phase
  2. Its domains are math and conversation, not native effort levels; those results
  do not settle this experiment or establish that LoRA adapters would fail.
  [TAPS](https://arxiv.org/abs/2603.27027)
- **Shared base plus adapters:** a vLLM RFC reports domain-specific LoRA adapters
  on a shared DFlash drafter with a GPT-OSS-120B target. This supports the
  feasibility question behind Yu's base-plus-delta option, but is not a
  peer-reviewed effort comparison and some experimental details are private.
  Defer adapter implementation until full-head specialization demonstrates value.
  [RFC](https://github.com/vllm-project/vllm/issues/52038)

The cited sources leave the exact mixed-effort versus low/medium/high-specialist
comparison unresolved. That defines the question here; it is not a claim that
no other work exists.

**2. Phase 1 configuration**

These are proposed settings to freeze before measurement.

| Native reasoning effort | Plain gpt-oss-20b | Same target with fixed DFlash head |
| --- | --- | --- |
| `low` | Measure | Measure |
| `medium` | Measure | Measure |
| `high` | Measure | Measure |

- **Drafter:** use `z-lab/gpt-oss-20b-DFlash`, the released 0.8B DFlash checkpoint
  for `openai/gpt-oss-20b`. Keep its weights fixed and use `k=7` (block size 8),
  matching its documented evaluation. Use this one model-method pairing throughout,
  with no algorithm comparison or Phase 1 training. Pin both model revisions, verify
  compatibility, and save weight hashes and the head's configuration. Record its
  documented training distribution and whether its training effort is known.
  [Drafter model card](https://huggingface.co/z-lab/gpt-oss-20b-DFlash)
- **Serving:** one Daint GH200 GPU, target and drafter TP=1, concurrency 1. Use a
  pinned vLLM runtime that supports gpt-oss, this DFlash head, Harmony and native
  effort requests on GH200. Keep the target's released MXFP4 weight format in both
  arms; record actual compute, head and KV-cache dtypes. Confirm kernel and memory
  compatibility during preflight. Hold prefix caching off, context length at
  65,536, and the combined analysis-plus-final output cap at 32,768. Match effective
  scheduler settings where supported and record differences.
- **Sampling:** explicitly set temperature 1.0, top-p 1.0, top-k disabled and
  min-p 0, with default penalties and natural completion termination. These are
  experimental settings held constant across effort levels. Use repeat seeds
  0, 1 and 2 with the same prompt/seed schedule in both arms; sampled outputs need
  not match exactly. Record the drafter's proposal sampler and the target's
  rejection-sampling configuration.
- **Prompts:** reuse the existing 64-prompt probe selection, 32 math and 32
  HumanEval, identified by [its manifest](../workloads/8b/probe-speculator-benchmarks.manifest.json).
  Keep task texts and IDs, replace Apertus-specific framing with the target's
  Harmony template, and save a new manifest. Override the old short output cap.
  Check prompt plus output fits the context; apply any filtering identically to
  all cells and update the request count if the selection changes.

The released target uses MXFP4 weights; the runtime's actual precision and
hardware support must be recorded rather than inferred from a generic dtype flag.
[OpenAI serving guide](https://developers.openai.com/cookbook/articles/gpt-oss/run-transformers)

**3. Native effort control and measurement preparation**

Send each prompt once to `/v1/chat/completions`, with the effort explicitly set:

```json
{
  "model": "gpt-oss-20b",
  "messages": [{"role": "user", "content": "<unchanged task prompt>"}],
  "reasoning_effort": "high",
  "temperature": 1.0,
  "top_p": 1.0,
  "top_k": -1,
  "min_p": 0.0,
  "max_tokens": 32768,
  "seed": 0,
  "stream": true,
  "stream_options": {"include_usage": true}
}
```

Use `gpt-oss-20b` as the served alias in both arms. Confirm that the pinned server
passes the top-level effort into Harmony rendering. Save a rendered prompt and
token IDs for each level, verifying the corresponding `Reasoning: low`,
`Reasoning: medium` or `Reasoning: high` setting. Fix rendered dates and other
template defaults across arms and blocks so only effort changes the prompt.
The [target template](https://huggingface.co/openai/gpt-oss-20b/blob/main/chat_template.jinja)
exposes `reasoning_effort` and defaults to medium, making this check necessary.

Preserve one continuous generation through analysis and final output, with no
forced transition or second call. Count analysis and final tokens separately,
and include the entire request in end-to-end timing.

The current [benchmark client](../src/apertus_bench/client.py) needs a narrow
extension to send the native effort and preserve separate analysis and final
events, including the pinned runtime's actual reasoning field. Add the effort to
generation settings and cell metadata. Add the `dflash` method label to the
benchmark CLI, which currently accepts EAGLE and DSpark but not DFlash. Send
min-p explicitly so the recorded sampling configuration matches the request. Reuse
[counter parsing](../src/apertus_bench/prometheus.py) and the existing artifact
formats. No new budget-control mechanism is required.

Before timed cells, use four extra prompts at all three efforts in both arms to
check rendering, response parsing, natural final answers, context/memory limits,
and speculative counters. Verify standard target verification and rejection
sampling. Unsupported effort handling, head loading, sampling or malformed
continuations block measurement. Fixed-seed text equality is not a correctness
criterion for sampled decoding.

**4. Phase 1 execution**

1. Pin and record target, tokenizer, drafter, runtime and prompt identities, then
   perform the preparation above. Use the head's published serving configuration
   as the starting point: `method=dflash`, `num_speculative_tokens=7`.
2. Run three deployment blocks. Each block independently launches plain and
   DFlash sequentially on the same GPU. Alternate arm order between blocks and
   rotate effort order. Each deployment measures all three efforts.
3. Warm each cell with two extra prompts, excluded from measurements, then run
   all 64 prompts sequentially. Keep unrelated traffic off the endpoint.
4. Snapshot counters before and after each request and the whole cell, outside
   request timers. Save separate analysis/final token counts and full stream
   events. Reject cells with server restarts or counter resets. Missing counters
   mean acceptance is unavailable, not zero.
5. Stop each deployment after its cells finish. Preserve failed attempts and give
   any replacement deployment a new directory.

Phase 1 total: **6 independently launched deployments, 18 cells, 1,152 measured requests**,
plus preflight and warmup. Each measured request generates analysis and a final
answer in one call. There is no concurrency, draft-depth, algorithm or hardware
sweep.

**5. Phase 1 measurements and decision**

For each effort, show the three repeats and their median and range:

| Measurement | Purpose |
| --- | --- |
| Accepted draft tokens / proposed draft tokens; accepted draft tokens / round | Measure the fixed drafter's fit at each effort |
| Generated tokens/s and end-to-end latency | Measure practical acceleration; compare DFlash with plain at the **same effort** |
| Analysis and final token counts, output-cap hits, finish reasons, empty answers and errors | Establish the effort response and identify truncation or failures |

Calculate latency speedup within each deployment block as plain latency divided
by DFlash latency for the same effort. Show absolute timings alongside ratios.
Use deployment blocks as repeats; requests do not supply independent deployment
replication. Keep math and code summaries visible so their mixture cannot hide a
different response to effort. Check generated-length distributions and decode
time per generated token: shorter sampled outputs alone do not establish faster
decoding.

Predeclare a 10% relative change in accepted draft tokens per round as the
descriptive threshold for material sensitivity: compare low and high separately
with medium within each block, requiring the same direction in all three blocks.
Report the actual spread rather than treating this small probe as a precise
significance test. A speedup consistently crossing below 1× is also operationally
relevant.

Use natural completions for interpretation. If more than 5% of requests in any
cell hit the common output cap, mark that effort comparison as limited by
truncation; do not interpret the cap as native effort behavior or silently drop
those requests. Identical output-length summaries alone do not establish that
the server ignored effort: check the saved rendered prompts first.

Aggregate acceptance can change because the analysis/final mixture changes.
Use phase-specific acceptance only if the runtime already exposes reliable
token/round attribution; streamed text chunks and cell counters cannot supply
that attribution. Otherwise report the aggregate effect without claiming to
separate composition from a change within a phase. Stable results support this
fixed head across these efforts on this probe; they do not establish that
specialist heads cannot improve it.

Proceed to Phase 2 if the declared acceptance-sensitivity threshold is met, or
if DFlash consistently wins over plain at one effort but loses at another across
all three blocks. First rule out ignored effort settings, invalid cells,
truncation and configuration drift. This gate motivates a specialization test;
it does not prove that different weights are required. Otherwise stop with the
Phase 1 table and a bounded recommendation to retain one head on this probe.

**6. Conditional specialist versus shared comparison**

Create four fine-tuned copies of the same released DFlash head. Each uses the
same target revision, architecture, initialization, optimizer schedule and total
supervised-token budget. Freeze the budget, training seed and checkpoint-selection
rule before training. Evaluate every head at every native effort:

| Fine-tuning data | Target low | Target medium | Target high |
| --- | --- | --- | --- |
| Low effort only | Measure | Measure | Measure |
| Medium effort only | Measure | Measure | Measure |
| High effort only | Measure | Measure | Measure |
| Balanced low/medium/high mixture | Measure | Measure | Measure |

Use the same task-prompt IDs to generate training responses with the frozen
target at all three efforts, preserving the actual Harmony setting and target
features. Separate train and development prompt IDs before generation. Keep the
64 evaluation prompts out of both sets, including duplicates. Balance the mixed
head by supervised tokens across efforts. Select checkpoints on development
data, never on the evaluation matrix. Use an existing compatible DFlash training
recipe; missing training or export support is a prerequisite failure, not a new
framework-development task.

Equal per-head budgets make the serving comparison controlled; three specialist
heads still cost three times the aggregate fine-tuning tokens of one mixed head.
Report that cost and stored/resident draft weights alongside performance. The
released Phase 1 head is a reference, not a substitute for the matched mixed head.

Use the Phase 1 serving and measurement protocol. Each of three fresh deployment
blocks launches plain and the four trained heads sequentially on the same GPU,
with balanced variant order. Each deployment measures all three efforts. This
conditional stage adds **4 fine-tuning runs, 15 serving deployments, 45 cells and
2,880 measured requests**, plus data generation, preflight and warmup. Its plain
deployments provide contemporaneous baselines. Do not pool the earlier baseline
cells with these comparisons.

Compare heads within each target-effort column. The main result is each matched
specialist versus the mixed head; the off-diagonal results show transfer across
efforts. Declare useful specialization only if a matched specialist reduces
median end-to-end latency by at least 10% relative to the mixed head in all three
deployment blocks, corroborated by lower decode time per generated token and
without a comparison invalidated by errors or truncation.
Report acceptance, absolute latency, speedup against plain and memory costs even
when that descriptive decision threshold is not met.

If the mixed head matches the specialists, retain one shared head. If specialists
show useful gains, identify the efforts where they help and quantify their cost.
That result motivates a later shared-base-plus-delta test; this experiment stops
before adapter serving, routing, offloading or checkpoint-averaging work.

**7. Reproducibility and handoff**

Follow [AGENTS.md](../AGENTS.md), [the result format](../results/README.md), and
the provenance and comparison principles in [the protocol](protocol.md). The
native effort and sampling settings above are specific to this experiment.

Use `results/gpt-oss-20b/thinking-effort/<phase>/<deployment-id>/`, with a fresh
directory per launch and separate effort cells. Preserve `metadata.json`, `requests.jsonl`,
`summary.json`, raw metric snapshots, full outputs and stream events, per-channel
token counts, and rendered Harmony prompts. Include Git revisions and patches,
launch commands, resolved environments, container path/digest, target and drafter
revisions/hashes, tokenizer/template hashes, corpus digest, effort, sampling
settings, seeds, job/node IDs, launch logs and checkpoint validation reports.

Pass non-secret provenance through `--metadata KEY=VALUE`. Record
`deployment_id`, `checkpoint_sha256`, `target_model`,
`target_revision` and `target_tensor_parallel_size`. Any use of Speculators'
`launch_vllm.py` must include `--provenance-dir` pointing into that deployment's
artifacts. Published checkpoint configuration and serving support still require
validation in the pinned runtime; Phase 1 has no training fallback.

If Phase 2 runs, retain each training configuration, target contract, prompt/split
manifests, generated corpus and hashes, full commands, source revisions/patches,
training history, validation results, initialization hash and exported head
configuration/validation reports. Preserve the provenance artifacts generated
by the selected trainer. Speculators runs must retain `train_command.txt`,
`speculators.patch`, `eval_command.txt` and all applicable vLLM provenance files
listed in AGENTS.md. Publish those artifacts with any model or result release.

Finish with the Phase 1 effort table and its stop/proceed decision. If Phase 2
runs, add the specialist/mixed matrix and the latency/memory tradeoff. Distinguish
confirmation of DFlash's effort observation from evidence about the organization
of draft weights, and keep all conclusions within the measured efforts and probe.

**Implementation entry points**

The runnable configuration, Slurm preparation/launch commands, artifact layout
and conditional training/export contract are in
[experiments/thinking-effort](../experiments/thinking-effort/README.md).
`python3 -m apertus_bench.effort plan` previews the schedule without GPU work.
The server uses a recorded date override in vLLM's native Harmony renderer;
render checks use `/v1/chat/completions/render`, not the Hugging Face template
or the generic `/tokenize` path. Channel token counts from text re-encoding
are labelled separately from server-reported usage and total completion tokens.
