**Apertus 1.5: EAGLE 3.1 on 8B first, then transfer the pipeline to 70B**

Updated 2026-09-24. This is the implementation handoff. The immediate target is
`swiss-ai/Apertus-v1.5-8B`, accelerated by a small head trained specifically for that target.
EAGLE 3.1 remains the default; EAGLE-3 is the fallback for a demonstrated 3.1-specific problem.
The eventual 70B experiment follows a bounded 8B integration pilot. This revision supersedes
the previous execution order and task gates. It does not launch or cancel GPU jobs.

**1. Why start on 8B, and why keep EAGLE 3.1?**

Starting on 8B is better for establishing the implementation. The frozen teacher has fewer
weights to load and less work per forward pass; feature extraction and debugging are cheaper.
The first serving experiment can use one GPU, subject to an explicit memory check, so tensor
parallel communication does not complicate the initial feature/verification checks. This is
an engineering inference from model size and topology, not a measured Apertus speedup.

| Consideration | 8B first | 70B first |
| --- | --- | --- |
| Iteration cost | Lower teacher memory, loading and feature-generation cost. | More expensive retries and larger feature tensors. |
| Initial serving topology | Target TP=1, draft TP=1. | Target TP=4, draft TP=4. |
| Performance interpretation | A cheaper target leaves less time for drafting to pay for itself. | More expensive target execution may offer more amortization, depending on acceptance and verification cost. |
| Reusable work | Renderer, adapters, training/export pipeline and correctness checks. | Directly creates the eventual target's head. |
| Extra work | Retargeting existing defaults; separate 70B training later. | No intermediate head training. |
| Coverage | Does not establish TP=4 behavior or 70B speedup. | Tests the intended larger-model serving configuration immediately. |

The decision is **a bounded 8B pilot, not two full research campaigns**. Do not require a 3×
8B speedup before trying 70B: a correct but slow 8B configuration may still transfer useful
infrastructure to a more favorable 70B regime. Conversely, an 8B win does not establish a 70B
win. The learned 8B head and its hidden-state caches do not transfer to the 70B checkpoint.

EAGLE 3.1's published changes normalize target features and recurrent draft feedback to address
attention drift. The authors report improved numerical robustness and provide training and
vLLM integration. That supports retaining it; it does not certify this custom Apertus stack.
This review found no basis to declare EAGLE 3.1 generally unstable.
[EAGLE 3.1 announcement](https://vllm-project.github.io/2026/05/26/eagle-3-1.html)

Keep EAGLE-3 available as the established comparison/fallback. Both algorithms use vLLM
`method=eagle3`; their trained architectures differ. EAGLE-3 uses multi-layer target features
and simulated draft rollouts during training. Its reported speedups are benchmark-specific.
[EAGLE-3 paper](https://arxiv.org/abs/2503.01840),
[official implementation](https://github.com/SafeAILab/EAGLE)

**Fallback rule:** after the shared data, feature and export checks pass, reproduce any
3.1-specific NaNs, unstable rollout behavior, unsupported normalization path or correctness
failure on a small fixed input. Train an E3 control with the same target, data, width and
budget. If E3 passes while E3.1 fails, continue with E3 and record the evidence. If both fail,
fix the shared pipeline. Import failures, wrong tensor shapes and missing dependencies alone
do not diagnose 3.1 instability. If both work but E3 is faster, select by measured validation
latency. Do not switch architectures by editing a trained head's flags.

**2. Targets, limits and artifact layout**

| Setting | Stage A: immediate 8B pilot | Stage B: subsequent 70B experiment |
| --- | --- | --- |
| Target | `swiss-ai/Apertus-v1.5-8B` | `swiss-ai/Apertus-v1.5-70B` |
| Target/draft TP | 1 / 1 | 4 / 4 |
| Precision | BF16 target and head initially; verify resolved dtype | Same starting precision, independently verified |
| Max model length | 32768 | 131072 |
| GPU memory utilization | 0.8, validated with both arms | 0.8, validated with both arms |
| Prefix caching | Off for controlled measurements | Off initially; intended policy in capacity study |
| Core concurrency | 1, 8 | 1, 8, 32 |
| Core methods | Plain target; EAGLE 3.1, or E3 fallback | Plain target; selected EAGLE architecture |
| Extra control | Baseline matched to candidate settings, only if they differ | Same rule |
| Config root | `methods/eagle/configs/8b/` | `methods/eagle/configs/70b/` |
| Result root | `results/8b/eagle/` | `results/70b/eagle/70b/` |

These are planned settings, not claims that existing launchers implement them. Their adaptation
is A1 below. If a memory or runtime limit requires a change, apply it to both arms within that
stage and record it before comparing results. Start with text inputs. Quantization, image/audio
inputs and a different chat/thinking mode each require a separate experiment.

Create a target-specific `lock.json`, `preflight/compatibility.json`, tokenizer/template hashes,
training configs, data manifests and checkpoint provenance under each stage's roots. Keep
immutable runs in unique subdirectories. Do not overwrite existing 70B records or reuse them
as an 8B contract. Compare each treatment only with the same target's baseline.

Check the internal inventory for a head trained on the exact target revision before training.
The public `thomaskiefer/EAGLE3-Apertus-8B-Instruct-2509` head targets the older 2509 checkpoint;
it is not an Apertus **1.5** 8B head. Matching nominal parameter count is insufficient.
[Model card](https://huggingface.co/thomaskiefer/EAGLE3-Apertus-8B-Instruct-2509)

**3. Reuse the scaffolding, then remove its 70B assumptions**

Training helpers, EAGLE launch/validation code and benchmark metadata support already exist.
Inspect and adapt them; do not recreate implemented features from an outdated task list.
The execution ledger is [eagle-progress.md](eagle-progress.md). Its dated status entries describe
earlier work, not current cluster state. Add new A/B task entries and preserve existing records.
If a job is recorded as running, check its actual state before submitting another job; this
plan revision itself does not authorize cancelling it.

Start with recorded serving source `a601a9d998ddeb488f0c17e8512874b116aa7658`, `model-launch`
`909026a990454557f1b54d26f24ec3ad92e51e35`, and candidate TorchSpec source
`6c042a87140a84d13839e341ece2c5c3ada918bc`. Validate installed dependencies and patches;
these pins are starting references, not proof of a working combination. Record image digests,
ARM64 build, driver/CUDA/PyTorch/Transformers versions and per-rank device placement.

The Apertus wrapper and EAGLE code expose the relevant interfaces, but vocabulary dimensions,
feature indexing and the actual execution path need testing. In particular, Apertus's input
and output vocabulary dimensions can differ. Use the checkpoint's tensor shapes and output
mask when loading the teacher head or reconstructing logits.
[Apertus wrapper](https://github.com/swiss-ai/vllm/blob/a601a9d998ddeb488f0c17e8512874b116aa7658/vllm/model_executor/models/apertus_mm.py),
[EAGLE head implementation](https://github.com/swiss-ai/vllm/blob/a601a9d998ddeb488f0c17e8512874b116aa7658/vllm/model_executor/models/llama_eagle3.py)

**4. Stage A task order: obtain a working 8B implementation**

| Task | Dependencies | Deliverables and completion gate |
| --- | --- | --- |
| A0: inspect target/environment | — | 8B lock and contract; import/config probe passes before loading teacher weights. |
| A1: retarget scaffolding | A0 | Explicit target/contract/TP/config parameters; target-isolation and launcher tests pass. |
| A2: data and parity | A1 | Disjoint splits, target-generated responses, matching renderer/features/logits. |
| A3: train and export pilot | A2 | Reloadable 8B E3.1 head with finite loss and improving held-out rollout acceptance; E3 fallback if justified. |
| A4: serving correctness | A3 | No unexplained systematic verification discrepancy; restart/load smoke passes twice. |
| A5: profile and screen | A4 | Direct EAGLE cost measurements and depth selection on validation prompts. |
| A6: confirm and hand off | A5 | Three independent baseline/candidate deployments on untouched 8B test prompts; explicit 70B transfer decision. |

Use A0–A6 and B0–B3 in the ledger so older task statuses cannot be mistaken for completion
of this revised scope. A failed prerequisite stops its dependent task, not independent work.
Attach the error, minimal reproducer and next action. Inspect repeated identical failures
before resubmitting; infrastructure failures are not negative algorithm results.

**A0 — Establish the 8B contract**

1. Read the actual Capstor checkpoint at
   `/capstor/store/cscs/swissai/infra01/hf_models/models/swiss-ai/Apertus-v1.5-8B`.
   Save model revision/weight manifest, input/output vocabulary, hidden size, layer count,
   head dimensions, RoPE, norm epsilon, tensor keys/shapes, precision, BOS/EOS and chat template.
2. Derive auxiliary layer indices for this target. Record separately the decoder-layer
   convention used by the trainer and the convention used by the serving model. Do not copy
   the 70B tuples `(2,40,77)` or `(1,39,76)` into the 8B config.
3. Verify draft imports, optional dependencies, config parsing and tensor-name discovery
   before the teacher load. Record every patch and test its applicability to the installed
   source. Keep training and serving environments explicit.
4. Measure one-GPU memory requirements using a short input, then the configured 4k training
   and serving shapes. If a single GPU does not fit, first use sequential feature extraction
   and head training. If serving needs TP=2, change both baseline and candidate to TP=2 and
   save a revised layout; never silently spill the timed serving path to CPU.
5. Record allocated GPU-hours as well as utilized GPU-hours. A one-GPU process on an exclusive
   four-GPU node is not a one-GPU allocation saving.

**A1 — Adapt existing code deliberately**

The following were present on 2026-09-24; verify again before editing:

| Location | Required adaptation |
| --- | --- |
| `methods/eagle/launch/eagle.sh`, `serving/baseline.sh` | Parameterize target TP, target contract, served name and stage output roots. They currently hardcode target TP=4; the EAGLE launcher only accepts draft TP=4. Add matched TP=1/1 for Stage A, retaining TP=4/4 for Stage B. |
| `methods/eagle/launch/train-eagle*.sh`, submit wrappers | Accept an explicit config instead of selecting a 70B config from a stage label; parameterize teacher/trainer GPU counts, target path, layer IDs and output paths. Print resolved settings before submission. |
| `methods/eagle/apertus_eagle/contract.py`, `target_adapter.py` | Read the selected contract throughout. Replace fixed layer tuples, default contract path and the assertion of 80 layers with contract-derived values. |
| `src/apertus_bench/eagle.py` | Pass the existing `--contract` option from launchers; validate head provenance against the selected target revision and tokenizer, not only shapes or directory names. |
| `methods/eagle/apertus_eagle/overfit_direct.py` | If used, parameterize its target/layer/config defaults and prove token alignment. Treat its current one-step loss as an import/overfit debug tool, not a substitute for full rollout training. |
| `methods/eagle/configs/train-*.yaml`, `draft-*-config.json` | Generate new files under `methods/eagle/configs/8b/`; existing root-level templates are 70B-specific even when named `e3`. Keep their recorded 70B values intact. |
| `src/apertus_bench/analysis.py` | Add target identity/revision, tokenizer, target TP, draft TP and precision to exported provenance. Require target identity/revision, tokenizer, target TP and precision to match the baseline; record draft TP as a treatment factor and match it across speculative repeats. Existing keys do not yet prevent 8B/70B baseline pooling. |

The CLI already accepts `--method eagle3` and requires a logical `--algorithm` plus deployment
and checkpoint metadata. Retain those features. Retain operational versus matched baseline
separation, E2E metrics and missing-counter handling; extend them only where needed.
Use a checkpoint manifest digest that covers weights and config, not a config hash alone.

Add focused tests for target-contract selection, wrong-target head rejection, mixed-target/TP
baseline rejection, and launcher propagation of the 8B settings. Run existing relevant tests
after changes. `VALIDATE_ONLY=1` may check architecture templates, but a config-only directory
must never be accepted as a trained serving head.

`matrix --repeats N` repeats requests against one server. It does not launch independent
deployments. Launch each deployment separately, use `--repeats 1`, and record its unique ID.
Do not pool different targets, images, corpora, generation modes or cache policies.

**A2 — Data and numerical parity**

Use disjoint source-group splits, fixed before target response generation: 32 conversations
for the debug overfit check, 1,000 pilot training conversations, 128 training-validation
conversations, and separate performance-validation/test prompts. Keep shared documents and
near-duplicates within one split. Reuse eligible prompt preparation code and source manifests;
regenerate assistant responses using the **8B** target and record their generation settings.
Original dataset assistant text may test plumbing but is not the target-aligned pilot corpus.

Default coverage is chat/code/summarization at 40/30/30%, with English/German/French/Italian
at 40/20/20/20%. These are coverage defaults, not production frequencies. Prefer an available
declared traffic mix. Save licenses, dataset revisions, selection seed and corpus digests.
Use natural EOS and greedy target responses for the first experiment.

Verify at least 32 diverse single/multi-turn prompts against the serving renderer: exact token
IDs, one BOS, assistant boundaries and loss mask. Resolve template/default-system behavior
from the 8B checkpoint. Validate that every input ID is representable by the draft embedding,
including control tokens. Keep target input vocabulary, target output support and any reduced
draft output vocabulary distinct.

On fixed prefixes compare captured auxiliary features and reconstructed target logits with
the serving path. Check layer offsets, residual addition, position IDs, final normalization
exactly once, output masking and token shifts. Measure numerical error against repeated
same-path controls; report argmax agreement and margins. Systematic offsets block training.
If extraction uses Hugging Face and serving uses vLLM, explicitly test that boundary.

Start training sequences at 4,096 tokens. Cache bounded feature shards, not a complete
full-vocabulary logit corpus. Measure bytes/token over 100 samples before materializing all
pilot data. Each cache must name the target revision, feature contract and corpus digest;
refuse reuse after any of them changes.

**A3 — Train a bounded 8B head**

Use the pinned and validated TorchSpec EAGLE training implementation, retaining its simulated
multi-step draft rollouts. A wrapper that merely calls a draft backbone once does not exercise
that objective. The framework supplies EAGLE 3.1 support, but the Apertus adapter still needs
A2's checks. [TorchSpec](https://github.com/lightseekorg/TorchSpec),
[3.1 training change](https://github.com/lightseekorg/TorchSpec/pull/97)

Start with one supported Llama-style draft layer and target-aligned width, deriving dimensions
from the 8B contract. Count actual parameters and memory for the embedding, fusion, decoder,
output projection and optimizer. Do not assume a fixed parameter count from another model.

| Head | Trained configuration | When to run |
| --- | --- | --- |
| E3.1 | `fc_norm=true`, `norm_output=true` | Default. |
| E3 | `fc_norm=false`, `norm_output=false` | Controlled fallback when the rule in section 1 triggers. |

Create `methods/eagle/configs/8b/train-e31-overfit.yaml`, `train-e31.yaml` and a derived
`draft-e31-config.json`. Generate E3 equivalents only when needed. Use separate checkpoint
directories; never relabel or resume a head under the other architecture.

Initial recipe: frozen teacher; BF16 head; upstream forward-KL rollout objective; rollout
length 7; learning rate `1e-4`; warmup ratio `0.015`; gradient clipping `0.5`; microbatch 1;
gradient accumulation to 16 sequences per optimizer step. Preserve and record other upstream
optimizer defaults. Audit accumulation semantics before computing the effective batch.

Run the 32-example overfit check first, with a cap of 400 optimizer steps. Then train on the
1,000-example pilot, evaluating held-out acceptance every 100 steps, with a cap of 2,000 steps.
Require finite losses, real optimizer updates, decreasing loss and improving held-out rollout
acceptance. If there is no learning, debug before adding data. If the pipeline works but the
corpus is too small for useful acceptance, allow one expansion to **10,000 conversations**,
up to three epochs, stopping after three evaluations without improvement. Do not automatically
launch the full 50k/300k training program on 8B.

Start with full valid target output support. If the output projection dominates measured
cost, a reduced draft vocabulary is a later controlled ablation. Build mappings from training
data only, test control-token coverage and per-language held-out coverage, and preserve the
full target sampler. Update validation rules before attempting that ablation.

For the simplest resource path, extract features with the 8B teacher on one allocated GPU,
stop/unload it, then train the head on the same GPU. Use disjoint GPU placements only if
streaming extraction is needed. Keep the target embedding/norm/head weights available for
replay. Validate EAGLE replay support locally; the documented offline example is not an
Apertus certification. [Offline replay](https://github.com/lightseekorg/TorchSpec/blob/6c042a87140a84d13839e341ece2c5c3ada918bc/docs/offline_training.md)

The following are templates inside the pinned training environment on an allocated GPU,
after A1/A2. Set the variables to absolute paths and validate the resolved config first:

```bash
python3 -m torchspec.offline.generate \
  --config "$EAGLE_TRAIN_CONFIG" --output "$EAGLE_REPLAY_DIR"

python3 -m torchspec.train_entry \
  --config "$EAGLE_TRAIN_CONFIG" \
  inference.inference_engine_type=offline \
  inference.offline.data_path="$EAGLE_REPLAY_DIR" \
  inference.offline.num_engines=1 \
  training.training_num_gpus_per_node=1 \
  output_dir="$EAGLE_TRAIN_OUTPUT"

python3 tools/convert_to_hf.py --input-dir "$EAGLE_CHECKPOINT_DIR"
```

Run the conversion command from the TorchSpec checkout. Export the architecture flags,
auxiliary-layer convention, token mappings and target provenance. Compare training-time and
exported head logits, then reload the checkpoint. A saved file alone is not completion.
Record teacher/train tokens per second, peak GPU/host memory, disk use and allocated GPU-hours.
Set token, disk and walltime caps from the first 100 samples/100 warm steps and available
allocation before expanding.

**A4 — Verify correctness and operational stability**

1. Capture at least 128 prompts at C=1, prefix caching off, twice per deployment for plain 8B
   and candidate. Also collect independently launched controls for both configurations.
   Retain token IDs, completion lengths, finish reasons and errors.
2. Compare exact greedy outputs first. On divergence, find the first differing token on an
   identical forced prefix and compare target/verification logits, argmax margin and repeated
   numerical variation. Use the existing calibrated gate only as a secondary regression
   signal. Do not assume nondeterminism or loosen tolerances to obtain a pass.
3. Exercise early rejection, full acceptance, bonus token, EOS at each position, output cap,
   KV rollback and vocabulary mapping. Unexplained systematic differences block promotion.
4. Prove two independent load/warmup/restart cycles without crashes or unexplained memory
   growth. Record whether a failure is shared infrastructure or specific to E3.1.
5. Before adding temperature 0.8, test the pinned rejection sampler and fixed-prefix output
   distributions. Same-seed sampled sequence equality is not required, because RNG consumption
   can differ. A greedy test alone does not establish sampled distribution equivalence.

**A5 — Measure cost and screen depths**

Compare the plain 8B operational baseline with the trained 8B head. If speculation changes
scheduling or effective token budget, add a separate matched baseline to explain the difference.
The operational baseline remains the denominator for speedup claims. Measure actual effective
settings; do not assume a budget limit binds at the observed batch size.

Start at depth 3, C=1, with 32 distinct prompts of 512–1024 input tokens and 256 fixed output
tokens. Use `ignore_eos=true` only for this mechanics check. Capture a short warm GPU trace,
then obtain headline timings in unprofiled runs. Mark drafting, feature assembly, target
verification, sampling, cache bookkeeping and exposed CPU work. Record graphs/eager execution,
kernel shapes and idle gaps. Sum neither overlapping kernel durations nor per-request rounds
as if they were batched engine iterations.

At C=1, `decode_speedup ≈ g × t0 / t_round`, where `g` is committed tokens per round and `t0`
is plain-target time per token at the same context. Count zero-proposal and EOS/limit cases;
`1 + accepted/drafts` is useful but not always the delivered-token count. SSE event gaps are
not token latency. Use this measurement to decide whether the limit is draft execution,
acceptance, verification, or shared serving overhead.

Then screen depths `{2,3,5,8}` at C={1,8} using natural EOS and greedy sampling. Per cell use
64 distinct validation prompts, 64 measured requests, two independent deployments, and warmup
of at least `max(8,2 × concurrency)` after graph/compile activity has settled.

| Stratum | Input length | Output cap |
| --- | --- | --- |
| Chat | 128–2048 tokens | 256 |
| Code generation | 128–4096 tokens | 512 |
| Summarization | 2048–4096 tokens | 512 |

With a plain baseline, one matched baseline and four EAGLE depths, this is
6 × 3 strata × 2 concurrencies × 2 deployments = **72 cells**. Omit a redundant matched
baseline, and stop expanding clearly dominated depths. Record omissions. If depth-dependent
settings need additional matched controls, record them separately. Add a 16k-input summary
stress check only for the selected depth, at C=1/8; keep its results separate.

Choose at most two candidates by geometric-mean TPOT speedup versus the operational baseline
over declared validation strata. Use an available fixed traffic mix; otherwise equal weights.
A setting within 5% of the best with lower memory cost or better worst-stratum latency may
take the second slot. Do not select on the untouched confirmation set.

This launch template illustrates **8B target TP=1 and draft TP=1** inside an allocated job.
The target/head contract must already pass validation; add the recorded container patches
and common scheduler controls through the adapted launcher:

```bash
vllm serve "$TARGET_MODEL" \
  --served-model-name "$SERVED_MODEL" \
  --tensor-parallel-size 1 \
  --gpu-memory-utilization 0.8 --max-model-len 32768 \
  --no-enable-prefix-caching \
  --chat-template-content-format string \
  --enable-auto-tool-choice --tool-call-parser apertus \
  --compilation-config.pass_config.fuse_allreduce_rms false \
  --speculative-config.method eagle3 \
  --speculative-config.model "$EAGLE_HEAD" \
  --speculative-config.num_speculative_tokens 3 \
  --speculative-config.draft_tensor_parallel_size 1
```

Set `TARGET_MODEL` to the Capstor Apertus-v1.5-8B checkpoint; an 8B served name alone does not
change the loaded target. For the baseline, use the same target/precision/context/cache
settings and omit the speculative arguments. Require a successful short completion before
warmup; registration in `/v1/models` alone is insufficient.

After A1, one independently launched deployment can be measured as follows:

```bash
apertus-bench matrix \
  --base-url "$API" --metrics-url "$VLLM_METRICS_URL" \
  --model "$SERVED_MODEL" --variant apertus15-8b-eagle31-k3 \
  --method eagle3 --algorithm eagle31 \
  --num-speculative-tokens 3 --draft-tensor-parallel-size 1 \
  --workloads workloads/8b/eagle-8b-validation.jsonl \
  --concurrencies 1 8 --requests 64 --warmup-requests 16 --repeats 1 \
  --metadata target_model=swiss-ai/Apertus-v1.5-8B \
  --metadata target_revision="$TARGET_REVISION" \
  --metadata target_tensor_parallel_size=1 \
  --metadata deployment_id="$DEPLOYMENT_ID" \
  --metadata block_id="$BLOCK_ID" \
  --metadata checkpoint_sha256="$EAGLE_HEAD_DIGEST" \
  --output "results/8b/eagle/screening/$DEPLOYMENT_ID"
```

Supply remaining provenance fields in the planning manifest, including paired baseline ID,
image digest, tokenizer hash, precision and actual scheduling. If using the fallback, change
both the head and logical label to `eagle3`. Run the client on-cluster against the measured
replica; verify `/metrics` is that replica's endpoint with no unrelated traffic.

**A6 — Confirm the 8B pipeline and transfer decision**

Confirm plain 8B versus the selected candidate with **three independent deployments each**,
C={1,8}, and 128 untouched prompts per core stratum. For one candidate, this is
2 × 3 strata × 2 concurrencies × 3 deployments = **36 cells**. Randomize/counterbalance
deployment order within node/date blocks. Report all repeat effects; three repeats are an
engineering check, not a precise population-level speedup estimate.

Save p50/p95 TTFT, TPOT and E2E, output tokens/s, success rate, output lengths, finish reasons,
acceptance by position, committed tokens/round, peak memory and KV capacity. Keep 16k stress
results distinct. Compare effects within compatible deployment blocks; requests within one
deployment are nested observations. Never hide slow successful requests or reset counters.

Write `docs/eagle-8b-report.md` with the working commands, exported head, correctness/stability
evidence, cost breakdown and limits. Stage A succeeds when a target-aligned head trains,
exports, serves reproducibly and has interpretable acceptance/cost measurements. A performance
win is a separate outcome. Unresolved correctness or unexplained runtime overhead blocks
scaling; a small/negative 8B speedup alone does not.

Freeze the working pipeline and proceed to Stage B once these gates pass and the available
allocation covers its measured estimate. Do not spend the 8B pilot budget chasing a 3× result.

**5. Stage B: apply the validated pipeline to 70B**

| Task | Required work |
| --- | --- |
| B0: new target contract and TP=4 | Build a separate 70B config/lock, train/serve feature-parity probe and all-rank check. Reuse verified adapters, not 8B dimensions or feature caches. |
| B1: train target-specific head | Regenerate responses with 70B; train/export a new head. Reuse prompt split definitions and the selected recipe as a starting point. |
| B2: screen and confirm | Compare against a fresh plain 70B baseline on the same runtime; tune depth again, then five independent serving deployments per selected configuration. |
| B3: capacity and recommendation | Measure sustainable request rate and write `docs/eagle-70b-report.md` with deployment boundaries and costs. |

Retest the exact TP=4 path for auxiliary features, collective ordering, verification and
numerics. A TP=1 8B result does not establish any of these. Start 70B with the selected EAGLE
architecture; apply the same E3.1 fallback rule. Never attach the 8B-trained head to 70B.

Begin B1 with the same tiny/pilot gates before a substantive run. First substantive budget:
50,000 target-regenerated conversations, 2,000 validation conversations, up to three epochs;
extend toward 300,000 only if validation acceptance/latency and resource estimates justify it.
Introduce representative 16k training sequences before asserting long-context robustness.
One teacher node plus a training node can support streaming extraction if available; otherwise
use sequential extraction/replay only when measured memory, disk and I/O budgets fit.

Recheck depth `{2,3,5,8}` at C={1,8} on validation prompts; retain at most two candidates.
For confirmation use chat, code, and separate 16k/32k/64k summary strata, C={1,8,32}, 256
unique untouched prompts per stratum and **five independent deployments**. Plain 70B plus
one candidate yields 2 × 5 × 3 × 5 = **150 cells**; a second candidate adds 75. If 64k/C=32
causes queueing or preemption, report it instead of silently changing the context/load.

Report 95% bootstrap intervals over deployment-block effects and show each repeat. Distinguish
serving repeatability of one head from reproducibility across training seeds; add a second
training seed if making a method-level claim. Treat **3× decode speedup as a stretch target**.
Prefill and queueing limit E2E gains, so do not substitute TPOT speedup for request-latency gain.

Predeclare these research decision defaults, or replace them before confirmation with actual
team requirements: a useful low-load result is TPOT speedup ≥1.5× with a deployment-level 95%
lower bound >1.1× on the selected strata. Broad enablement additionally requires ≥99% success,
no declared high-load stratum with >5% throughput regression or >10% p95 TTFT/TPOT regression,
and acceptable capacity. Report smaller gains honestly; consider selective routing if only
specific workloads win.

For capacity, run an open-loop Poisson arrival sweep on the same declared traffic mix and
fixed TTFT/TPOT/error/backlog gates. Use production SLOs if available. Otherwise freeze numeric
research bounds from a low-load baseline calibration and label them experimental. Increase
load until a gate fails, refine the bracket, then confirm the sustainable rate with three
30-minute soaks per arm. Check client saturation, queue growth, preemptions, cache occupancy
and allocated GPU-hours. Retest the intended prefix-cache policy before recommending service.

**6. Scope and completion checklist**

Further methods are optional after the core experiment. If sequential head execution remains
the measured bottleneck, P-EAGLE is a possible separate trained-head comparison. Do not turn
on parallel drafting for an ordinary E3/E3.1 head or let this delay the working 8B pilot.
[P-EAGLE paper](https://arxiv.org/abs/2602.01469)

- [ ] A0/A1: establish the 8B contract and remove fixed 70B/TP=4 assumptions from the active path.
- [ ] A2: prove renderer, feature, token-shift and output-vocabulary parity.
- [ ] A3: train/export E3.1; use E3 only through the evidence-based fallback rule.
- [ ] A4: resolve verification differences and demonstrate repeatable startup/serving.
- [ ] A5/A6: profile, screen and confirm the bounded 8B pilot; record limits and cost.
- [ ] B0–B3: independently validate/train/measure 70B after the pipeline is ready.
- [ ] Preserve raw artifacts and append exact commands, failures and next actions to the ledger.

The accompanying `methods/eagle/experiments/smoke-70b/experiment.yaml` describes this revised experiment. It is a
planning manifest, not proof that all referenced per-target configs and launcher changes exist.
No new training or performance result is claimed by this plan revision.
