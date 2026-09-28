# EAGLE 3.1 for Apertus 1.5 8B: Stage A report

Status: Stage A complete, 2026-09-29 (A6 confirmation ran 2026-09-26). Full chronology, job ids
and raw numbers: [eagle-progress.md](eagle-progress.md). Plan:
[eagle-execution-plan.md](eagle-execution-plan.md).

## Summary

- A target-aligned EAGLE 3.1 head for `swiss-ai/Apertus-v1.5-8B` (revision
  `a411d838…`) trains, exports, reloads bit-exactly and serves in the pinned vLLM
  (`0.23.1rc1.dev1029+ga601a9d99`, TP=1 target, TP=1 draft) with three overlay
  patches.
- Greedy speculative output matches plain greedy output except at bf16
  near-ties (every measured divergence has a forced-prefix log-probability
  margin <= 0.25). Two independent serving load/restart cycles are clean.
- Current head: `e31-mix10k-3515343-se` (9,909 target-regenerated conversations,
  chat 40 / code 30 / summarization 30, 3 epochs). **Confirmed on the untouched
  test strata: 1.40x geometric-mean TPOT speedup at depth 2** (three independent
  deployments: 1.402 / 1.399 / 1.397), 1.39x at depth 3. Code 1.83–1.95x,
  chat 1.20x, summarization 1.16–1.26x. Every cell is faster than plain.
- A 1k-conversation head already gives 1.20x geometric-mean TPOT speedup at
  depth 2 on the validation strata (code ~1.6x, chat and summarization
  ~1.04–1.08x).

## Target contract (A0)

`targets/8b/contract.json`, `targets/8b/lock.json`.

| field | value |
| --- | --- |
| checkpoint | `/capstor/store/cscs/swissai/infra01/hf_models/models/swiss-ai/Apertus-v1.5-8B` |
| revision | `a411d838600baf0e3635a3daf66fb7c55fc97bb6` |
| weight manifest sha256 | `0a6c3ca1c1e8cf220c21971b3b96b0e8799827f7bb00ac056dfb3657a533b08a` |
| text model | hidden 4096, 32 layers, 32/8 heads, head_dim 128, xIELU MLP 21504 |
| vocabulary | input 266752, output (lm_head) 131072 |
| aux layers | vLLM `(2, 16, 29)`; HF decoder-layer outputs `[1, 15, 28]` |
| tokenizer / template | identical hashes to the 70B record |
| image | `vllm_apertus_1.5_release-arm64.sqsh`, sha256 `f5fc017f…` |
| environment | torch 2.11.0+cu130, transformers 5.14.0.dev0, triton 3.6.0, CUDA 13.0, GH200 |

The aux-layer rule was read from the image's vLLM source, not assumed, and
HF-versus-vLLM parity confirmed the offset: the claimed alignment has relative
error 0.003–0.028 against 0.059–44 for the neighbouring layers.

## Pipeline

Training reuses TorchSpec `6c042a8`'s `Eagle3Model` rollout objective (7 draft
steps, forward KL, `0.8^i` position weights) and `BF16Optimizer`. Only the
Ray/Mooncake data plane is replaced, because the serving image has neither.

```bash
# on a Clariden login node
EAGLE_TRAIN_CONFIG=methods/eagle/configs/8b/train-e31-mix10k.yaml ./methods/eagle/launch/submit-eagle-train.sh
# steps: generate -> extract -> train -> export -> verify (apertus_eagle.pipeline)
STAGE=8b EAGLE_HEAD=<head dir> ./methods/eagle/launch/eagle.sh --no-tui       # serve TP=1/1
STAGE=8b ./serving/baseline.sh --no-tui                          # matched plain target
ARM=eagle DEPTH=2 PHASE=screen BLOCK_ID=b1 EAGLE_HEAD=... ./methods/eagle/launch/eagle8b-measure.sh
```

| step | module | notes |
| --- | --- | --- |
| responses | `apertus_eagle.generate_targets` | 8B greedy, natural EOS, serving default (`Deliberation: disabled`); prompt ids checked against `LLM.chat` |
| features | `apertus_eagle.features` | HF teacher, 32.8 KB/token, resumable, provenance-checked |
| parity | `apertus_eagle.parity_vllm` | per-layer offset test against vLLM hooks |
| training | `apertus_eagle.train_rollout` | TorchSpec objective; lr 1e-4, warmup 0.015, clip 0.5, 16 sequences/step |
| export | `apertus_eagle.export_head` | serving keys, provenance, no `embed_tokens`, bit-exact reload check |
| verify | `apertus_eagle.verify_offline` | token-level greedy, cap sweep, margins, sampled TV test |

vLLM overlay patches (all regenerated from the image source and dry-run tested):
`vllm-apertus-eagle3-aux-layers.patch`, `vllm-apertus-image-token.patch`,
`vllm-apertus-eagle-mrv2-inner.patch` (new: model runner v2's
`load_eagle_model` assumed `get_language_model().model`).

## Heads

| head | corpus | held-out sim. acc. length | offline acceptance, depth 3 |
| --- | --- | --- | --- |
| e31-pilot-3508057 | 1000 DeepMath (split bug) | 2.10 on math | 51.9% math, 6.0% A5 prompts |
| e31-mix-3514907-se | 987 mixed | 0.44 | 17.0% |
| e31-mix10k-3515343-se | 9909 mixed, 3 epochs | 0.665 | 25.6% |

Draft: 1 Llama layer, hidden 4096, SwiGLU 14336, 293.6M trainable parameters;
lm_head seeded from the target and frozen; input embedding shared with the
target at serve time. Draft adds 1.55 GiB served memory; KV capacity 466,144 ->
439,072 tokens.

## Correctness and stability (A4)

- Greedy, offline engine, 240 requests per capture including an output-cap
  sweep over every draft position: EAGLE equals plain whenever both use the same
  compiled kernels; divergences appear only across compilations and are all
  near-ties (<= 0.25 logprob). 10k head: 239/240 and 240/240.
- Served: two independent EAGLE deployments ready in 285 s / 297 s, 18.78 GiB
  each; chat completion, 32 HTTP greedy captures and acceptance counters equal
  across restarts.
- Sampled (temperature 0.8, 1024 samples x 32 prompts, independent seeds): TV
  from the plain reference 0.0070–0.0257 for EAGLE against 0.0055–0.0201 for a
  second plain run; sign counts at chance (20/32 at 4 tokens, p ≈ 0.11). No
  evidence of a distribution mismatch; power limited.

## Measurements (A5/A6)

### Screening (A5, validation strata, head `e31-mix10k-3515343-se`)

Geometric-mean TPOT speedup against the same-block plain baseline, two
independent deployments per depth (blocks b5, b6):

| depth | all | C=1 | C=8 | worst cell | committed tokens/round |
| --- | --- | --- | --- | --- | --- |
| 2 | 1.373 | 1.410 | 1.336 | 1.140 | 1.72 |
| 3 | 1.366 | 1.393 | 1.340 | 1.088 | 1.84 |
| 5 | 1.260 | 1.289 | 1.233 | 0.951 | 1.96 |

Depth 8 was omitted (0.91x in the 1k-head screen). By stratum at depth 2:
code 1.76–1.90x (acceptance 61%, median E2E 0.54–0.60x), summarization
1.16–1.26x, chat 1.14–1.19x. Selected for confirmation: depths 2 and 3.

### Confirmation (A6, test strata, head `e31-mix10k-3515343-se`)

Three independent deployments each of plain, depth 2 and depth 3, blocks c1–c3,
rotated order, 128 untouched test prompts per stratum, C = 1 and 8, greedy.
36 EAGLE cells, success 100% everywhere. Raw cells: `results/8b/eagle/confirm/`;
report: `results/8b/eagle/confirm-report.json` (`apertus_eagle.a6_report`).

Geometric-mean TPOT speedup against the same-block plain deployment:

| depth | all | c1 / c2 / c3 | C=1 | C=8 | worst cell | tokens/round g |
| --- | --- | --- | --- | --- | --- | --- |
| **2** | **1.400** | 1.402 / 1.399 / 1.397 | 1.437 | 1.364 | 1.147 | 1.75 |
| 3 | 1.389 | 1.392 / 1.388 / 1.386 | 1.421 | 1.356 | 1.098 | 1.90 |

Per stratum (range over the three blocks):

| depth | stratum | C=1 | C=8 | acceptance by position | E2E p50 ratio |
| --- | --- | --- | --- | --- | --- |
| 2 | code | 1.95–1.95 | 1.81–1.84 | 0.75, 0.54 | 0.57–0.59 |
| 2 | chat | 1.20–1.22 | 1.19–1.20 | 0.34, 0.10 | 0.79–0.81 |
| 2 | summarization | 1.26–1.26 | 1.15–1.17 | 0.39, 0.14 | 0.82–0.84 |
| 3 | code | 2.09–2.11 | 1.96–1.99 | 0.74, 0.53, 0.39 | 0.54–0.56 |
| 3 | chat | 1.14–1.15 | 1.13–1.15 | 0.34, 0.10, 0.03 | 0.83–0.84 |
| 3 | summarization | 1.19–1.20 | 1.10–1.13 | 0.39, 0.14, 0.04 | 0.85–0.87 |

- Depth 2 is the recommended setting: best overall and best worst cell. Depth 3
  wins only on code.
- Repeat effects are tight (spread <= 0.01 overall), so the three deployments
  agree; this is an engineering check, not a population-level interval.
- TTFT p50 rises 0–31% at C=1 (the draft adds prefill work); TPOT p95 ratio
  0.61–1.09.
- Memory: model load 17.78 GiB vs 17.23 GiB plain; KV capacity 438,784 vs
  466,144 tokens (-5.9%).
- Output lengths and finish reasons match plain per stratum (e.g. chat mean 190
  tokens in every arm). 56% of chat responses hit the output cap in every arm.
- Greedy agreement at C=1, per prompt: EAGLE vs its same-block plain deployment
  79% chat / 91% code / 75% summarization (depth 2), against a control of plain
  vs plain across deployments of 65% / 86% / 52%. EAGLE output differs from
  plain no more than one plain relaunch differs from another, so the divergences
  are deployment numerics (fresh compile per deployment), not the draft. This
  also explains the one A5 depth-2 deployment (b6) that diverged more than its
  siblings.

**Transfer decision:** Stage A gates pass (target-aligned head trains, exports
bit-exactly, serves reproducibly, acceptance and cost are interpretable, and a
consistent win). Proceed to Stage B (70B) when allocation allows, reusing this
pipeline. Before that, the DSpark comparison retrains the 8B head on the shared
corpus (see Limits).

Profile (depth 3, C=1, 256 fixed tokens, ignore_eos): plain t0 = 5.76 ms;
EAGLE round cost ≈ 7.5 ms ≈ 1.31 t0, so break-even needs g >= 1.31.

## Incidents that changed results

1. Pilot splits were 100% DeepMath: `prepare_data.subsample_pilot` sorted by
   (source, id) before truncating. Replaced by domain-stratified sampling.
2. Literal multimodal tokens (`<|image|>` = 131079) in prompt text crashed the
   EAGLE engine (device-side assert in the draft embedding lookup). Fixed by
   exporting without `embed_tokens` so vLLM shares the target embedding;
   training skips such rows.
3. OpenCodeReasoning-2 repeats problem statements under different ids; 18 (1k)
   and 96 (10k) training rows duplicating benchmark prompts were removed.
4. A corrupted validation workload file (overlapping writers) made the first
   screening fail to load; writes are now atomic.
5. The serving environment does not mount `/users`: traces and compile caches
   now go to `/iopsstor`.

## Limits

- Text only, TP=1, BF16, thinking disabled; one training seed.
- Code prompts reach only ~1.2k input tokens; the 4k code band is not covered.
- The training corpus is 10k conversations with short (~210 token) responses;
  acceptance was still rising at the end of 3 epochs.
- Greedy equality is not bitwise between compilations (bf16 near-ties).
- Every job holds an exclusive 4-GPU node while using one GPU.
- The benchmark client and campaign driver ran on a login node against the
  replica; later campaigns should run the client inside a compute allocation.
- Not yet comparable to DSpark: that head is trained on
  `mlabonne/open-perfectblend`; an apples-to-apples run retrains EAGLE on the
  same corpus and token budget.
