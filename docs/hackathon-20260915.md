# Speculative decoding on Apertus-v1.5-70B — what we measured, and what it costs

CSCS / Swiss AI gathering, OAT Lugano, 2026-09-15. Faruk Zahiragić (EPFL), issue
[`swiss-ai/apertus-program#1057`](https://github.com/swiss-ai/apertus-program/issues/1057),
code in [`swiss-ai/apertus-speculative-decoding`](https://github.com/swiss-ai/apertus-speculative-decoding).

Every number below is read from artifacts committed in this repository. The run directory is
`results/smoke-screening-20260913-debug/` (30 cells, `analysis.csv`, per-cell Prometheus snapshots),
with correctness captures in `results/correctness/smoke-20260913-debug/` and one recorded launch
failure in `results/deployment-failures/`.

---

## 1. The one-slide version

Five independently launched deployments on one 4×GH200 Clariden node each, same corpus, same client,
same `max_model_len`, three workloads × two concurrencies = 30 cells, all at a 1.000 success rate.
Both speculative arms now have two independent deployments; the baseline has one.

- **The 8B drafter at depth 3, draft TP=4 accepts well and is roughly 2× slower than baseline.**
  Acceptance rate 0.605–0.797, mean acceptance length 2.814–3.392 out of a possible 4 — and output
  throughput 0.375–0.472× of baseline in all six paired cells, in both draft deployments. TPOT p50
  goes from ~14 ms to 30–37 ms; TTFT p50 rises 3.6–5.5×.
- **Acceptance is not the binding constraint.** The arm that accepts *worst* is the only arm that
  wins anything.
- **Model-free n-gram speculation (depth 3, prompt lookup 1–4) is the only configuration that beats
  baseline on anything, and the win reproduced on a second independent deployment.** It accepts only
  0.127–0.285, and on long-context summarization it reaches **1.127× then 1.204×** baseline
  throughput at concurrency 1 and **1.071× then 1.093×** at concurrency 8. All six cells clear the
  protocol's 10% between-deployment spread gate, worst case 6.6%, against the draft arm's 11.1%
  breach. Lower TTFT than baseline now holds in **12 of 12** cells, p50 and p95 alike. It loses
  2–15% on open chat and code.
- **Why**, recovered from the vLLM speculative counters: verification plus the scheduling change
  costs **1.23–1.36×** a baseline decode step. The 8B drafter adds a further **4.92–6.22×**, for
  **6.24–7.45×** total. Three sequential 8B forward passes at TP=4 cannot be five to six whole 70B
  target steps, so this is *per-step overhead in the draft path*, not drafter arithmetic and not
  tensor-parallel collectives. The two n-gram deployments, on different nodes, put step cost within
  **1.7–3.9%** of each other in every cell, so a node effect cannot absorb a cost of that size.
  **That is the profilable item, and it is the ask for this group.**

**Serving recommendation today: do not enable `draft_model` speculation for Apertus 1.5 70B on this
stack.** N-gram is the only positive result in the study and it is now a repeated one, so it is the
configuration worth carrying forward. What it still needs before a production claim is a **baseline
repeat** — the baseline is the denominator of every ratio here and the only arm with one deployment
— and a genuinely long-context workload, which is the regime it should be best in and where this
corpus tests it least.

## 2. What was run

| Deployment | Method | Job | Node | Image |
|---|---|---|---|---|
| `baseline` | `none` | 3391425 | nid007645 | stock pinned |
| `draft-n3-tp4` repeat 1 | `draft_model` | 3391426 | nid006633 | pinned **+ overlay** |
| `draft-n3-tp4` repeat 2 | `draft_model` | 3392153 | nid006687 | pinned **+ overlay** |
| `ngram-n3` repeat 1 | `ngram` | 3392370 | nid006687 | stock pinned |
| `ngram-n3` repeat 2 | `ngram` | 3403354 | nid007500 | stock pinned |
| `draft-n3-tp1` | `draft_model` | 3392110 | nid006593 | never served — see §5.2 |

Target `swiss-ai/Apertus-v1.5-70B` TP=4, drafter `swiss-ai/Apertus-v1.5-8B` TP=4,
`num_speculative_tokens=3`, `max_model_len=131072`, `gpu_memory_utilization=0.8`, temperature 0.0,
natural EOS. vLLM `a601a9d` (`0.23.1rc1.dev1029+ga601a9d99`), `model-launch` `909026a`,
`debug` partition; the first four deployments on 2026-09-13, n-gram repeat 2 late on 2026-09-14. Per
cell: 8 warmup then 24 measured requests in closed loop, with a `/metrics` snapshot immediately
before and after. The load generator ran on-cluster against the replica node IP, so no gateway
latency is included. Total completion tokens agree within 0.2% across the five deployments
(36,148 / 36,160 / 36,164 / 36,184 / 36,115), so the arms really did the same work.

## 3. The headline numbers

Baseline (`b`) against draft repeat 1 (`d`) and the two n-gram deployments (`n₁`, `n₂`). Full tables
for all 30 cells are in `results/smoke-screening-20260913-debug/README.md` and `analysis.csv`.

| Workload | Conc. | TPOT p50 (ms) b→d→n₁→n₂ | Output tok/s b→d→n₁→n₂ | d÷b | n₁÷b | n₂÷b | n spread |
|---|---:|---|---|---:|---:|---:|---:|
| open_chat | 1 | 14.13 → 31.86 → 16.02 → 15.69 | 70.1 → 31.0 → 63.3 → 64.3 | 0.442 | 0.902 | 0.917 | 1.7% |
| open_chat | 8 | 14.69 → 32.54 → 17.36 → 17.02 | 432.3 → 190.6 → 366.5 → 372.9 | 0.441 | 0.848 | 0.863 | 1.7% |
| code | 1 | 14.15 → 30.16 → 14.91 → 14.48 | 70.5 → 33.0 → 66.5 → 69.1 | 0.469 | 0.943 | 0.980 | 3.9% |
| code | 8 | 14.70 → 30.07 → 16.04 → 15.44 | 540.6 → 255.2 → 464.3 → 480.3 | 0.472 | 0.859 | 0.888 | 3.4% |
| long_context_summarization | 1 | 14.16 → 32.72 → 12.51 → 11.69 | 70.2 → 29.4 → 79.2 → 84.5 | 0.419 | **1.127** | **1.204** | 6.6% |
| long_context_summarization | 8 | 14.72 → 31.36 → 13.36 → 12.85 | 493.7 → 212.0 → 528.7 → 539.8 | 0.429 | **1.071** | **1.093** | 2.1% |

Two things to note in the n-gram columns. **The summarization win reproduces and is larger the second
time**, and every cell clears `maximum_repeat_spread_fraction: 0.10` — the n-gram arm is materially
more reproducible than the draft arm, which spread 1.7–11.1% and breached the gate in one cell. And
**repeat 2 is faster in all six cells, the opposite direction from the draft repeats**, where repeat
2 was slower in all six. So the uniform between-deployment shift has no fixed direction; it behaves
like a node/deployment effect rather than drift in the harness, the cluster, or the time of day.

Acceptance, from Prometheus counter deltas over each measured window:

| Arm | Acceptance rate | Mean acceptance length (max 4) |
|---|---|---|
| `draft-n3-tp4`, both repeats | 0.605–0.797 (`code` highest) | 2.814–3.392 |
| `ngram-n3`, both repeats | 0.127–0.285 (summarization highest) | 1.380–1.854 |

For n-gram, acceptance is workload-dependent in the direction prompt lookup predicts, not the
direction "code is repetitive" predicts: summarization accepts best because a summary quotes the
document, while these code *generation* prompts have short instructions and little to copy forward,
so code accepts no better than open chat.

**Acceptance is the near-invariant for the model-based drafter, and not for n-gram.** The drafter's
acceptance rate moves only 0.5–4.3% between deployments, less than its throughput ratio does
(1.7–11.1%), so drafter quality is a property of the model pair. N-gram's moves **2.0–13.9%**, *more*
than its throughput ratio (1.7–6.6%), with the largest move in the cell that matters most
(`long_context_summarization` at concurrency 1, 0.248 → 0.285). The mechanism is clean: prompt lookup
matches against the prompt *plus the tokens generated so far*, so two deployments whose greedy
outputs diverge numerically generate different text and offer different n-grams to match. N-gram
acceptance inherits output nondeterminism; model-based acceptance, which scores fixed text, does not.
Mean acceptance length is the steadier of the two (0.6–6.2%).

## 4. Why: the step-cost decomposition

Speculation changes how many tokens leave the engine per step, so per-token latency understates
per-step cost. Mean step time is recovered as `mean TPOT × T ÷ (T − accepted)`, where `T` is the
cell's completion tokens: every accepted draft token is a token that needed no step of its own, so
`T − accepted` is the number of engine steps. The baseline emits one token per step, so its TPOT is
its step time. The n-gram arm proposes without any forward pass, so it isolates verification. Both
speculative arms now have two deployments, so every figure below is a range over two independent
deployments rather than a single measurement.

| Workload | Conc. | Baseline step | N-gram step r1 / r2 | Draft step r1 / r2 | Verify × r1 / r2 | Draft total × r1 / r2 |
|---|---:|---:|---:|---:|---:|---:|
| open_chat | 1 | 14.13 ms | 18.64 / 18.32 ms | 91.15 / 100.41 ms | 1.32 / 1.30× | 6.45 / 7.11× |
| open_chat | 8 | 14.67 ms | 20.00 / 19.61 ms | 94.24 / 101.27 ms | 1.36 / 1.34× | 6.42 / 6.90× |
| code | 1 | 14.15 ms | 18.13 / 17.45 ms | 101.62 / 104.25 ms | 1.28 / 1.23× | 7.18 / 7.37× |
| code | 8 | 14.70 ms | 19.54 / 18.88 ms | 101.34 / 106.45 ms | 1.33 / 1.28× | 6.89 / 7.24× |
| long_context_summarization | 1 | 14.16 ms | 17.79 / 17.36 ms | 96.63 / 105.52 ms | 1.26 / 1.23× | 6.82 / 7.45× |
| long_context_summarization | 8 | 14.71 ms | 19.42 / 18.98 ms | 91.77 / 98.56 ms | 1.32 / 1.29× | 6.24 / 6.70× |

Verifying a depth-3 proposal plus the scheduling change costs **1.23–1.36×** a plain decode step over
12 cells from two deployments; repeat 2 lands slightly cheaper in all six cells, so the range widened
downwards rather than moving. Adding the 8B drafter costs **another 72.3–88.2 ms**, i.e.
**4.92–6.22×** a full 70B target step.

That number is the finding. An 8B model at TP=4 doing three sequential forward passes should cost a
*fraction* of one 70B forward pass, not five to six of them. Whatever is consuming that time is
per-step overhead in the draft path — separate model execution, launch and synchronisation, drafter
steps that appear not to benefit from the target's captured graphs — rather than drafter FLOPs or
draft-side collectives. A kernel-level profile of the draft path against the baseline is the single
highest-value next experiment, and it is a vLLM-side question, not a workload question.

The decomposition is an **inference from committed counters, not a direct measurement**. It assumes
mean TPOT is a fair average step cost, that verification costs the same in both speculative arms,
and that arms are comparable across nodes. The second assumption is violated in a known direction:
`draft_model` runs with a 7168-token per-step budget while n-gram keeps the baseline's 8192, so part
of the 72–88 ms is budget rather than drafter. The third is imperfect too, but two independent
measurements now bound it. N-gram repeat 1 and draft repeat 2 share node nid006687, and the gap
against draft repeat 2 is *wider* (79.1–88.2 ms), not narrower. And the two n-gram deployments, on
nid006687 and nid007500, put step cost within **1.7–3.9%** of each other in every cell — so
node-to-node variation on this measurement is a few percent, against a drafter cost of five to six
whole baseline steps. A node effect cannot absorb that.

Corroboration at the client: the median stream-event gap is ~91–101 ms in the draft deployments
against ~16–20 ms across both n-gram deployments and ~14 ms for baseline — the same per-step cost,
observed directly.

## 5. Platform findings that affect other people

These are why this belongs at a gathering rather than in a private report.

### 5.1 `draft_model` speculation with Apertus 1.5 crashes at drafter load on the pinned image

vLLM's `SpecDecodeBaseProposer.load_model()` copies the target's image token onto the draft config
and reads `target_model.config.image_token_index` for any architecture not on an allow-list.
`Apertus1p5ForConditionalGeneration` is multimodal but was not on that list, and `Apertus1p5Config`
defines `image_token_id`, not `image_token_index` — so the 70B target loads, the drafter raises
`AttributeError`, and the engine never comes up.

The one-line fix is `patches/vllm-apertus-image-token.patch`, bind-mounted over the read-only image
by `launch/patch-vllm.sh`, and is open as [`swiss-ai/vllm#20`](https://github.com/swiss-ai/vllm/pull/20)
against `apertus-1-5`. **No branch and no prebuilt image on Capstor contains it.** Critically, the
Apertus 1.5 upstreaming PRs — [`swiss-ai/vllm#16`](https://github.com/swiss-ai/vllm/pull/16) and its
upstream counterpart [`vllm-project/vllm#50496`](https://github.com/vllm-project/vllm/pull/50496) —
touch no file under `vllm/v1/spec_decode/`, so **the bug ships upstream as-is** unless something
changes. Anyone trying draft-model speculation with an Apertus 1.5 target hits it.

Where #20 currently stands is itself a point for this room: the maintainers' position is that
`apertus-1-5` is the reference branch for the upstream PR and should not take experimental changes,
so the fix should land either fully developed there or as a dedicated PR. That is a reasonable
policy and it also means **there is currently no path by which a stock image gains this one line.**
Deciding that path is a five-minute conversation for the people who own the fork and the image
build.

### 5.2 `draft_tensor_parallel_size=1` is unservable, which closes a factor of the design

Job 3392110 reached vLLM with the flag intact and all four workers raised at engine construction:

```text
ValueError: Currently, 'draft_tensor_parallel_size' and 'tensor_parallel_size' must be the same.
Got 1 and 4. Please pass 'draft_tensor_parallel_size' in the speculative_config.
```

`_raise_if_draft_tp_mismatch` (`vllm/v1/spec_decode/draft_model.py:74`) is called unconditionally
from `DraftModelProposer.__init__`. Its own comment gives the reason: with target TP>1 and draft
TP=1, every rank compiles the draft as rank 0 and corrupts the `torch.compile` cache. We did not
patch the guard out — the failure mode it prevents is silent, not clean. Evidence and traceback are
in `results/deployment-failures/draft-n3-tp1-3392110/`.

Consequence: hypothesis H4 and the draft-TP half of research question 3 are unanswerable at this
revision, and `configs/experiment.yaml` now pins `draft_tensor_parallel_size: [4]`. The step-cost
decomposition also says TP=1 would not have rescued the result: eliminating draft-side collectives
entirely cannot recover five to six baseline steps.

### 5.3 Speculation costs KV cache even with no drafter resident

From `vllm:cache_config_info` in each deployment's `metrics_after.prom`, all at
`gpu_memory_utilization=0.8` and `max_model_len=131072`:

| Arm | KV cache (tokens) | Max concurrency at 131072 tok/request | Loss vs baseline |
|---|---:|---:|---:|
| baseline | 513,696 | 3.92× | — |
| `ngram-n3` (r1 / r2) | 416,640 / 416,640 | 3.18× | −18.9% |
| `draft-n3-tp4` (r1 / r2) | 251,472 / 251,488 | 1.92× | −51.0% |

The n-gram arm loads no drafter and runs the same image as the baseline, yet gives up 97,056 tokens
of cache — presumably to the larger per-step activation footprint of verifying four positions at
once. **Enabling speculation is a capacity decision even when it is free of model weights.** The
drafter's weights then cost a further 165,168 tokens, so the draft arm's halved capacity is roughly
37% speculation machinery and 63% drafter weights.

**These are the most reproducible numbers in the study.** The two n-gram deployments sized their
cache to the same 416,640 tokens on different nodes, identically, and the two draft deployments came
within 16 tokens of each other. So unlike throughput, the −18.9% and −51.0% capacity costs are
deterministic properties of the configuration rather than sizing accidents, and they hold even if the
latency picture changes.

### 5.4 `max_model_len` had to come down from 262144 to 131072

With the drafter resident, job 3374717 could not reserve cache for a single full-context request:
*"To serve at least one request with the model's max seq len (262144), 28.0 GiB KV cache is needed,
which is larger than the available KV cache memory (26.86 GiB)"*. No protocol workload exceeds
65,536 input tokens, so 131072 was applied to **every** arm to keep them comparable. Anyone copying
the published example script at its default 262144 will hit this the moment they add a drafter.

### 5.5 Two launch-path issues worth flagging to the `model-launch` owners

Both are worked around in this repository's launchers and neither is a defect in `model-launch`
itself so much as a toolchain-pairing hazard.

- **The `{arch}` placeholder in the environment toml.** `vllm_apertus_1.5_release.toml` ships
  `...vllm_apertus_1.5_release-{arch}.sqsh`. Only some `sml` builds substitute it on the node; one
  that does not passes the literal through and pyxis rejects it, killing the job seconds after start
  (job 3234057, dead in 19 s). `launch/resolve-env.sh` writes a resolved copy under `~/.sml`.
- **A split toolchain: newer CLI flags against the pinned environment toml.** Revision `909026a`
  expects `--system`, `--framework`, `--environment`; other `sml` builds in circulation use
  `--firecrest-system`, `--serving-framework`, `--slurm-environment` and expect the OpenTela share
  at `/ocfbin`, whereas the pinned toml mounts it at `/opentelabin`. `model-launch` commit `4413441`
  ("Fix SGL Router Registration Issue…", PR #161, 2026-06-23) did both renames together. Pair an
  `sml` from the other side of that commit with this environment toml and the job submits fine,
  starts its container, and then dies on a missing binary (job 3374539) — a failure on the node
  rather than at submission, which is the expensive kind. The launchers pin the CLI explicitly; a
  version assertion between the `sml` build and the environment toml would kill this class of bug
  for everyone.

## 6. Diff against the published `Apertus-v1.5-70B-spec-decode.sh`

Rob's suggested starting point on #1057 is
`examples/clariden/cli/swiss-ai/apertus-ai-1.5-release/Apertus-v1.5-70B-spec-decode.sh` in
`swiss-ai/model-launch`. This experiment's `launch/draft-model.sh` is that script with three
substantive changes and a few operational ones. **The speculative configuration itself is
identical**, so the numbers above are numbers for the published configuration.

| Setting | Published example | `launch/draft-model.sh` | Same? |
|---|---|---|:--:|
| method | `draft_model` | `draft_model` | yes |
| draft model | `swiss-ai/Apertus-v1.5-8B` | same, Capstor cache path | yes |
| `num_speculative_tokens` | 3 | 3 (parameterized) | yes |
| `draft_tensor_parallel_size` | 4 | 4 (parameterized) | yes |
| `tensor-parallel-size` | 4 | 4 | yes |
| `gpu-memory-utilization` | 0.8 | 0.8 | yes |
| `fuse_allreduce_rms` | false | false | yes |
| tool parser / chat template | `apertus`, `string` | same | yes |
| **`max-model-len`** | **262144** | **131072** | **no** |
| **image** | stock pinned | **pinned + `image_token_index` overlay** | **no** |
| **environment toml** | packaged, `{arch}` unresolved | resolved copy under `~/.sml` | **no** |
| model paths | HF names | Capstor cache paths | operational |
| `--served-model-name` | `swiss-ai/Apertus-v1.5-70B` | unique per launch | operational |
| `--time` / `--partition` | 12:00:00 / normal | 04:00:00 default; 01:00:00 debug as run | operational |
| `--system` / `--nodes-per-replica` | prompted through the TUI | passed explicitly (`clariden`, 1) | operational |
| TUI | `--tui` | passthrough (`--no-tui` used) | operational |

Three things to say about that diff:

1. **The published example does not currently run.** With a stock image it dies at drafter load
   (§5.1), and at `max-model-len 262144` it cannot reserve KV cache for one request once the drafter
   is resident (§5.4). Both of our deviations are forced, not tuning choices.
2. **Its header comment claims "Lossless — outputs match the plain 70B".** The theorem is about
   exact arithmetic. On this stack two *identically configured* deployments agree on 0 of 6 greedy
   prompts, so "outputs match" is not a checkable property here and shouldn't be advertised as one
   (§7). The weaker, true statement is the one our gate supports.
3. `num_speculative_tokens=3` and draft TP=4 are the example's defaults and are exactly what was
   measured, so nobody needs to wonder whether we handicapped it. If the example gains tuned
   defaults later, the honest current default is "off".

## 7. Method note: the correctness gate was measuring the wrong thing

Relevant to the "LLM model evaluation methods" thread as much as to this experiment.

The original gate required exact greedy token equality between baseline and speculative arms. It
failed — and then the control showed why the failure meant nothing:

| Pair | Exact matches | Worst norm. edit distance | Role |
|---|---:|---:|---|
| baseline vs. draft repeat 1 | 0 / 6 | 0.227 | compared |
| baseline vs. draft repeat 2 | 3 / 6 | 0.262 | compared |
| **draft repeat 1 vs. repeat 2** (identical config) | **0 / 6** | **0.256** | **control** |

Two deployments running the *same* configuration agree as poorly as speculation agrees with the
baseline. The gate was measuring cross-deployment floating-point nondeterminism — different nodes,
different batch shapes, a different per-step token budget, prefix caching — not correctness.
Divergent pairs share 151–564 characters of prefix and then continue differently but fluently, with
completion-token counts within 3.

It was replaced with a **calibrated divergence criterion**: per-prompt common-prefix fraction,
normalized edit distance and completion-token difference, judged against a same-configuration
control plus a declared margin. Under it both baseline-vs-draft comparisons return
`within_control_envelope` — no evidence that speculation changes the output distribution. That is
consistent with losslessness and is *not* a proof of it; the control is a single cross-deployment
pair of six prompts drawn from the speculative arm, so baseline-side variation is unmeasured.

Two structural points that generalise beyond this project:

- **A same-deployment paired baseline-vs-speculative check is impossible by construction.**
  `speculative_config` is engine-level and fixed at launch, so one deployment serves exactly one
  speculative configuration. No harness change creates that pairing.
- **The second n-gram deployment gave that arm its first same-configuration control, and the gate
  then split across the two deployments — which bounds the gate's resolution, not n-gram's
  soundness.** The control (n-gram repeat 1 vs. repeat 2) has 1 of 6 exact matches and a worst
  normalized edit distance of 0.264, giving an allowance of 0.314. Baseline-vs-repeat-1 comes in at
  0.308 and returns `within_control_envelope`; baseline-vs-repeat-2 comes in at 0.392 and returns
  `exceeds_control_envelope` on that one check of four, while passing the other three (and its
  completion-token difference sits exactly on the allowance, 40 against 40). Two deployments of one
  configuration returning opposite verdicts against an identical control is the sharpest available
  statement of the limit: the allowance is a worst-of-six order statistic from a single control pair,
  so it is itself a noisy quantity, and adding the draft-repeat pair as a second control does not
  loosen it because the gate takes the tightest envelope. **Six prompts cannot adjudicate
  losslessness at this resolution, so "the gate passes for n-gram" is not a result we have.** What is
  presentable: the divergence between the arms is of the same character and roughly the same size as
  the divergence between two deployments of one arm. Deciding losslessness needs many more prompts
  and a baseline-versus-baseline control.

## 8. What this run cannot claim

Stated plainly, because several of these push in the same direction as the headline.

- **Async scheduling is auto-disabled for both speculative arms and not for the baseline.** It
  separates speculation from baseline but is common to the two speculative arms, so it cannot
  explain any difference *between* them — and since n-gram beats baseline on summarization in both
  of its deployments while carrying it, the confound is bounded by that margin. Both this and the
  token-budget asymmetry below were re-read from n-gram repeat 2's own server log rather than assumed
  to carry over, and both hold identically.
- **The per-step token budget shrinks to 7168 for `draft_model` only** (`max_num_batched_tokens=8192`
  on the baseline and n-gram). This one is asymmetric between the speculative arms and inflates any
  draft-versus-n-gram comparison, including part of the step-cost split in §4.
- **The draft arm ran a different image** (the overlay) from the baseline and n-gram arms.
- **The baseline is now the only single-deployment arm, and it is the denominator of every ratio in
  this report.** Both speculative arms have two independent deployments; the baseline has one. So
  every spread quoted here is a spread of the numerator only, and a baseline repeat would widen them
  all. That makes a baseline repeat the cheapest measurement that would firm up the summarization
  win — more valuable now than a third n-gram repeat.
- **One cell of the draft arm breaches the declared `maximum_repeat_spread_fraction: 0.10`**,
  `long_context_summarization` at concurrency 1, at 11.1%. All six n-gram cells clear it, worst 6.6%.
  Two repeats support a spread but not the bootstrap confidence interval over deployment effects the
  protocol's analysis section asks for.
- **All five deployments ran on different nodes at different times**, so node effects are not
  separated from method. Only the n-gram-repeat-1 / draft-repeat-2 pairing is node-matched
  (nid006687). Draft repeat 2 was slower in all six of its cells and n-gram repeat 2 faster in all
  six of its own, on a fifth node — so the shift has no fixed direction, which argues for a
  node/deployment effect rather than harness or cluster drift, but does not separate it from method.
- **The smoke corpus is not the research corpus.** Its "long context" stratum averages ~283 prompt
  tokens per request (270 and 295 for the two prompts), against the 16k–64k the protocol specifies.
  So these rows measure almost no prefill, and n-gram's best regime is untested — this probably
  *understates* n-gram's advantage rather than overstating it.
- **Not measured at all:** concurrency 32, the confirmation matrix with five repeats and bootstrap
  CIs, depths other than 3 in either method, an `ignore_eos` control, the open-loop capacity search,
  and GPU utilization / power / SM-activity telemetry. The `debug` partition QOS allowed one running
  job at a time, which forced everything to run sequentially.

## 9. Proposed next steps

In priority order. The first is the one this room is best placed to help with; the second is the
cheapest thing that would firm up the only positive result in the study.

1. **Profile the draft path.** A kernel-level trace of `draft-n3-tp4` against the baseline, to find
   where 72–88 ms per step goes when three 8B forward passes should cost a fraction of one 70B step.
   This is the highest-value item and it is a vLLM question. Candidate hypotheses to discriminate:
   drafter steps not using captured/compiled graphs, per-step model-switch and synchronisation
   overhead, and the 7168-token budget. If the overhead is fixable, the entire conclusion changes.
2. **Repeat the baseline.** It is the only arm with one deployment and the denominator of every ratio
   in this report, so all quoted spreads are of the numerator alone. One extra baseline launch is now
   worth more than a third n-gram repeat, and it would also supply the baseline-versus-baseline
   control the correctness gate has never had.
3. **Run an async-scheduling-disabled baseline**, ideally back to back with (2) since both are
   baseline launches. It removes the last confound between the baseline and both speculative arms,
   and tests the leading hypothesis for why n-gram has lower TTFT than the baseline in all 12 cells.
4. **A genuinely long-context summarization workload** (16k–64k input tokens). Prompt lookup should
   do best exactly where there is a document to copy spans from, and that regime is currently
   untested — so n-gram's repeated win is probably understated rather than overstated.
5. **Decide a path for [`swiss-ai/vllm#20`](https://github.com/swiss-ai/vllm/pull/20)** and get the
   one line into a built image. Today nobody can run draft-model speculation with Apertus 1.5
   without a bind-mount overlay, and the upstreaming PRs do not carry the fix.

Two open cells that a late result would slot into, if cluster access returns: the depth sweep at
draft TP=4 (`num_speculative_tokens` 2, 5, 8) and the n-gram depth sweep. Both extend §3's tables
with rows in the same shape; neither is expected to change the sign of the draft-model result, since
depth changes how much drafting happens per step but not the per-step overhead in §4.

## 10. Reproducing this

```bash
# Environment
git clone git@github.com:swiss-ai/apertus-speculative-decoding.git && cd apertus-speculative-decoding
python -m venv .venv && . .venv/bin/activate && python -m pip install -e '.[dev]'

# Pin the model-launch CLI to the revision the environment toml expects, and put it first on PATH
git -C ../model-launch worktree add ../model-launch-apertus 909026a990454557f1b54d26f24ec3ad92e51e35
export MODEL_LAUNCH_ROOT=../model-launch-apertus
python3 -m venv ~/venvs/sml-apertus && ~/venvs/sml-apertus/bin/pip install -e "$MODEL_LAUNCH_ROOT"
export PATH="$HOME/venvs/sml-apertus/bin:$PATH"
```

```bash
# Launch, one arm at a time (debug QOS permits one running job)
SML_PARTITION=debug SML_TIME=01:00:00 ./launch/baseline.sh --no-tui

IMAGE=/capstor/store/cscs/swissai/infra01/container-images/ci/vllm_apertus_1.5_release-arm64.sqsh
EXTRA_MOUNTS="$(./launch/patch-vllm.sh "$IMAGE" patches/vllm-apertus-image-token.patch)" \
  SML_PARTITION=debug SML_TIME=01:00:00 NUM_SPECULATIVE_TOKENS=3 DRAFT_TP=4 \
  ./launch/draft-model.sh --no-tui

SML_PARTITION=debug SML_TIME=01:00:00 NUM_SPECULATIVE_TOKENS=3 PROMPT_LOOKUP_MAX=4 \
  ./launch/ngram.sh --no-tui
```

Each launcher prints its unique served model name. Wait for `/v1/models` **and** one successful
short chat completion before measuring — registration alone is not readiness.

```bash
# Measure one deployed arm (repeat per deployment)
apertus-bench matrix \
  --base-url "$API" --metrics-url "$VLLM_METRICS_URL" --model "$SERVED_MODEL" \
  --variant draft-n3-tp4 --method draft_model \
  --num-speculative-tokens 3 --draft-tensor-parallel-size 4 \
  --workloads workloads/smoke.jsonl --concurrencies 1 8 --requests 32 --repeats 2 \
  --metadata slurm_job_id="$SLURM_JOB_ID" \
  --output results/<run>/draft-n3-tp4-$SLURM_JOB_ID

# Analyse all deployments of a run into one CSV
apertus-bench analyze results/smoke-screening-20260913-debug --output /tmp/analysis.csv
```

Correctness, against the committed captures — no cluster needed:

```bash
apertus-bench correctness-gate \
  --treatment results/correctness/smoke-20260913-debug/baseline.json \
              results/correctness/smoke-20260913-debug/draft-n3-tp4.json \
  --control   results/correctness/smoke-20260913-debug/draft-n3-tp4.json \
              results/correctness/smoke-20260913-debug/draft-n3-tp4-repeat2.json \
  --output /tmp/gate.json
```

Substituting `ngram-n3.json` / `ngram-n3-repeat2.json` for both treatment candidate and control
reproduces the split verdict discussed in §7.

## 11. Artifacts

| What | Where |
|---|---|
| 30 measured cells, per-cell `summary.json` / `requests.jsonl` / `metrics_*.prom` | `results/smoke-screening-20260913-debug/` |
| Run narrative, all tables, provenance | `results/smoke-screening-20260913-debug/README.md` |
| Flat analysis over all five deployments (30 rows) | `results/smoke-screening-20260913-debug/analysis.csv` |
| `/v1/models` and pre-measurement chat completions | `results/smoke-screening-20260913-debug/provenance/` |
| Greedy captures, exact-match comparisons, calibrated gates | `results/correctness/smoke-20260913-debug/` |
| Unservable draft TP=1, with traceback | `results/deployment-failures/draft-n3-tp1-3392110/` |
| The vLLM one-line fix | `patches/vllm-apertus-image-token.patch` |
| Launchers and image-overlay helper | `launch/` |
| Experimental contract and gate definitions | `docs/protocol.md`, `configs/experiment.yaml` |
| Presentation visual (all five deployments, 30 cells) | `canvases/apertus-speculative-decoding-smoke-screening.canvas.tsx` |

Corpus `workloads/smoke.jsonl`, SHA-256
`316fb566a18e32305b98929e34fdac515b9a43c6c7329994b319870f879952af`.

Two launch failures referenced in §5.5 (jobs 3234057 and 3374539) are described in `README.md` and
`launch/resolve-env.sh` but their logs are not archived under `results/deployment-failures/`; that
gap should be closed the next time either is reproduced. Job 3374717's KV-cache error is recorded in
the commit message of `270413f`.
