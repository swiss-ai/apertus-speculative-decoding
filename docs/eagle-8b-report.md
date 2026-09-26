# EAGLE 3.1 for Apertus 1.5 8B: Stage A report

Status: draft, 2026-09-25. Sections marked **pending** are filled once the A5
screen of the 10k head and the A6 confirmation finish. Full chronology, job ids
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
  chat 40 / code 30 / summarization 30, 3 epochs). **Pending:** screening and
  confirmation speedups.
- A 1k-conversation head already gives 1.20x geometric-mean TPOT speedup at
  depth 2 on the validation strata (code ~1.6x, chat and summarization
  ~1.04–1.08x).

## Target contract (A0)

`results/eagle/8b/preflight/compatibility.json`, `configs/eagle/8b/lock.json`.

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
EAGLE_TRAIN_CONFIG=configs/eagle/8b/train-e31-mix10k.yaml ./launch/submit-eagle-train.sh
# steps: generate -> extract -> train -> export -> verify (apertus_eagle.pipeline)
STAGE=8b EAGLE_HEAD=<head dir> ./launch/eagle.sh --no-tui       # serve TP=1/1
STAGE=8b ./launch/baseline.sh --no-tui                          # matched plain target
ARM=eagle DEPTH=2 PHASE=screen BLOCK_ID=b1 EAGLE_HEAD=... ./launch/eagle8b-measure.sh
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

**Pending**: screen of the 10k head (blocks b5/b6, depths 2/3/5), depth
selection, and three-deployment confirmation on the untouched test strata.

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
