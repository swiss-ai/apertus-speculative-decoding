# Workload coverage and pipeline parallelism, Apertus 1.5 8B

Yu (DM, 15 Oct): with the 8B checkpoints in hand, test them on different
workloads and configurations first: out-of-distribution requests, datasets
that were trained on but have little data, P/D disaggregation or not, TP or
PP. TP, DP and P/D are in [system-optimizations-8b.md](system-optimizations-8b.md)
("Several GPUs and disaggregation"). This document covers pipeline
parallelism and the under-represented languages and domains.

Status 2026-10-24: done. PP is measured, plain only, since neither drafter can run with PP. Per-slice acceptance is measured for DSpark e10 and EAGLE e10 at k = 1/3/5/7, against each slice's share of the training corpus, with the best draft length per slice under load.

## Summary

| question | result |
| --- | --- |
| speculative decoding with PP | **Not supported on this image.** DSpark e10 and EAGLE e10 (model runner v2, and EAGLE on v1) are refused when the config is validated, because the draft model inherits the target's PP size and does not implement `SupportsPP`. Behind that refusal, the v2 runner refuses eagle3/dspark with PP, and so does DSpark's own setup code. Plain PP serves. |
| plain PP on a node | **Never worth it for an 8B target.** PP4 gives 0.51-0.68x of plain DP4's node throughput, and PP2xDP2 0.73-0.94x, at C = 4-512. TP4 is faster at low load (1.82x at C=4). DSpark on DP4 beats every PP layout up to C=256 (1.56x to 0.97x of DP4). |
| training-corpus share of the slices | Open-PerfectBlend (1.80M rows) is 99.55% English prompts. de/fr/it/es/zh/ja are 0.02-0.05% each, hi/ar about 0.01%. Romansh, Swiss German, Swahili and Amharic do not occur, and Yoruba has 2 rows. No prompt reaches 8k tokens. Tool calls are 0.08% and JSON schemas 0.04%. Translation requests are 0.7%, non-Python code 1.8%, math 44% and multi-turn 25%. |
| acceptance per slice | **Language decides acceptance; the corpus share within 0-0.05% does not.** Every non-English language accepts 1.55-2.23 tokens per step at k=7 (DSpark e10), against 3.24 for the same task in English. German (0.027% of the corpus) is as low as Romansh (0%): 1.66 vs 1.69. Structured output accepts as well as math whatever its share: JSON schema 6.61 at 0.04%, math 6.72 at 44%, non-Python code 5.63 at 1.8%. The two drafters accept within 7% of each other at k=7. The exceptions: long contexts (13k-token prompts), DSpark 2.71 and EAGLE 1.81; translation into English, EAGLE 15% lower; and Yoruba, EAGLE 21% higher. Acceptance does not change with load (C=128 within 2.5% of C=1). |
| best k per slice under load | Math and JSON: k=7 up to 32 running requests per GPU, k=4-5 at 128-256 (1.1-1.4x predicted). Code and tool calls: k=7 at C=1, then k=3-4. English chat, multi-turn and to-English translation: k=7 at C=1, k=4 at 32, k=1 at 128, plain at 256. Non-English languages and from-English translation: DSpark k=5 at C=1 (k=3-7 on Swahili and Arabic; 1.13-1.69x predicted), k=2 at 32, plain from about 64. EAGLE should use k=1-3 on these, and plain from 128. |
| slices that most need drafter data | All non-English output: Arabic, Swahili, German, Romansh, Hindi, Italian, Japanese (1.55-1.73 at k=7), then translation into de/fr/it/es, French, Chinese, Spanish, Yoruba, Amharic (1.87-2.23). EAGLE also needs long-context data. |

## Pipeline parallelism

### Can a drafter run with PP?

`pipeline_parallel_size` is now an engine key of the node sweeps (32d8929).
An arm takes tp x dp x pp GPUs. The launch check `pp-smoke` (job 3721313)
served Apertus 1.5 8B with `--pipeline-parallel-size 2`:

| arm | outcome |
| --- | --- |
| plain PP2 | serves; chat at C=4, 128/128 requests |
| DSpark e10 k=3, PP2 | `NotImplementedError: Pipeline parallelism is not supported for this model` (draft model, config validation) |
| EAGLE e10 k=3, PP2 | same |
| EAGLE e10 k=3, PP2, v1 model runner | same |

The source has two more refusals after the config check:
- `vllm/v1/worker/gpu/model_runner.py:197-203` refuses eagle3, dflash and dspark with PP. They need auxiliary hidden states, which the last stage alone holds.
- `spec_decode/dspark/utils.py:38-39` refuses PP for DSpark.

The v1 runner has no DSpark proposer at all. Lifting the restriction is an
upstream change: drafting on the last stage, plus the aux-hidden-state
hand-off. Recorded in `results/deployment-failures/pp-spec-decode-20261022/`.

### Plain PP against the measured layouts

`multigpu-pp` (job 3721388): summarization, whole node, C per node. Output
tok/s relative to plain DP4 measured in the same job:

| C (per node) | plain DP4, tok/s | plain PP4 | plain PP2xDP2 | plain TP4 | DSpark e10 k=7 DP4 |
| --- | --- | --- | --- | --- | --- |
| 4 | 599 | 0.68x | 0.94x | 1.82x | 1.56x |
| 32 | 2,760 | 0.59x | 0.81x | 1.22x | 1.31x |
| 128 | 5,473 | 0.52x | 0.76x | 0.89x | 1.07x |
| 256 | 6,511 | 0.53x | 0.74x | 0.81x | 0.97x |
| 512 | 7,283 | 0.51x | 0.73x | 0.72x | 0.89x |

TPOT p50 at C=4:

| layout | TPOT p50 |
| --- | --- |
| plain TP4 | 3.4 ms |
| DSpark DP4 | 3.7 ms |
| plain DP4 | 6.0 ms |
| plain PP2xDP2 | 6.5 ms |
| plain PP4 | 8.9 ms |

PP cuts neither latency nor throughput:
- A token still passes through every layer, one stage after another, plus a hand-off between stages.
- With only 2-4 microbatches in flight, PP4 keeps about half the node busy.
- The 8B model fits on one GPU, so PP does not even free memory that would matter.

The controls match the earlier runs to within the between-job noise:
- DP4 now 599 / 2,760 / 5,473 / 6,511 / 7,283 tok/s at C = 4 / 32 / 128 / 256 / 512, against 598 / 2,730 / 5,439 / 6,569 / 7,286 in job 3619861.
- TP4 now 1.82x / 1.22x / 0.89x / 0.81x / 0.72x of DP4, against 1.74x / 1.13x / 0.90x / 0.80x / 0.72x in job 3620830.
- DSpark on DP4 now 1.56x / 1.31x / 1.07x / 0.97x / 0.89x, against 1.55x / 1.34x / 1.10x / 0.98x / 0.90x.

## Coverage slices

`apertus_bench.coverage_workloads` (9eca8bf) builds 21 slices of 64 prompts
each from pinned public sources:
- Every row turns thinking off, as the drafters were trained, and caps answers at 512 tokens.
- Prompts are drawn in a fixed hash order, so a rebuild from the same revisions is identical.
- The file stays on the cluster, at `/iopsstor/scratch/cscs/faruk_zahiragic/spec-exp/workload-coverage/workloads/coverage/coverage.jsonl` (sha256 `ddb921fe…7804`).
- `workloads/coverage/manifest.json` holds the per-slice digests and the source revisions.

| slices | source (licence) | selection |
| --- | --- | --- |
| lang-eng (control), lang-deu, -fra, -ita, -spa, -roh (Romansh), -zho, -jpn, -hin, -ara, -swa, -yor, -amh | MURI-IT test split (Apache-2.0) | reverse-instruction (MRI) rows, so every language asks for the same kind of task. Yoruba has only 7 MRI rows and takes the xP3/SNI rows instead. |
| code-multi | HumanEvalPack (MIT) | instructions in C++, Go, Java, JavaScript, Rust (13/13/13/13/12) |
| math-gsm8k | GSM8K test (MIT) | questions |
| long-context | LongBench (MIT) | 8.5k-23.6k prompt tokens (mean 13.8k): 32 summarization (gov_report, qmsum), 32 QA (qasper, multifieldqa_en, narrativeqa, musique) |
| tool-call, json-schema | Hermes function-calling v1 (Apache-2.0) | single-turn, tools or schema in the system prompt |
| translate-from-en, translate-to-en | WMT24++ (Apache-2.0) | 300-2,000-character paragraphs, English to and from de/fr/it/es, 16 per language each way |
| multi-turn | MT-Bench human judgments (CC-BY-4.0) | second turns, GPT-4's first answer as the assistant turn |

Swiss German has no public prompt source with a licence that allows this
use, so it has no slice.

### Share of the training corpus

`apertus_bench.corpus_share` (3cdd43d, job 3722012) read every row of
`open-perfectblend_apertus15-8b_thinking-off_seed0.jsonl`:
- 1,803,261 rows classified, 6,423 without a parseable final answer.
- Language: fastText lid.176 on the last user turn.
- Domains: the heuristics in the module docstring. Domains overlap.
- Counts in `results/8b/coverage/corpus-share.json`.

| slice | share of rows | share of answer tokens |
| --- | --- | --- |
| English prompts | 99.55% | 99.82% |
| French / Spanish / Chinese | 0.054% / 0.040% / 0.037% | 0.030% / 0.020% / 0.023% |
| German / Italian / Japanese | 0.027% / 0.024% / 0.021% | 0.011% / 0.009% / 0.010% |
| Hindi / Arabic | 0.011% / 0.007% | 0.002% / 0.001% |
| Romansh, Swiss German, Swahili, Amharic | 0 | 0 |
| Yoruba | 2 rows | 0 |
| math | 43.8% | 46.7% |
| multi-turn | 25.0% | 29.4% |
| code (any fenced) | 13.3% | 11.2% |
| non-Python code (code-multi languages) | 1.8% | 2.2% |
| translation request | 0.66% | 0.35% |
| tool call | 0.081% | 0.075% |
| JSON schema / JSON answer | 0.040% | 0.018% |
| prompt >= 8k tokens | 0 | 0 |

## Acceptance per slice

### Method

`coverage-a` (DSpark e10, job 3722605) and `coverage-b` (EAGLE e10, job 3725070):
- Four arms each, k = 1/3/5/7, one GPU each, greedy decoding.
- Every slice runs at C=1 with 64 requests, a whole pass over its prompts.
- lang-roh, lang-deu and tool-call also run at C=128 (128 requests, two passes).
- All 13,824 requests returned 200.
- Acceptance length is accepted tokens per step, bonus token included.

There is no plain arm, so speedups come from the model below, not from measurement.

Best k under load uses `waste_report.predicted_speedup` (T1's step cost model, cherry-picked as 56534bf) with:
- each slice's measured acceptance length for k = 1/3/5/7, and the k=7 per-position rates for k = 2/4/6;
- the cost table of `kv-cost-a-20261015T230650Z/dspark-e10-k7-paper-cut-kv` (target verify time by step size, DSpark draft step, KV read per context token);
- each slice's mean context (prompt plus half the answer);
- B = 1/32/128/256 running requests per GPU; a load is skipped when B requests of the slice's mean length do not fit in the KV cache.

The cost table has no EAGLE draft cost. It is fitted from the C=1 cells as 0.75 + 0.54 (k-1) ms (first pass, then each later pass) and scaled with B like DSpark's one-token draft step.

Script: `tools/coverage-best-k.py`. Output, with the speedup of every k at every load: `results/8b/coverage/best-k.json`.

Checks of the model against the measured cells:

| check | result |
| --- | --- |
| DSpark step at C=1 | matches the measured step (TPOT x acceptance length) within 2-3% (lang-eng k=7: 7.75 ms predicted, 7.63 ms measured); tool-call reads 6-14% low |
| ordering of k at C=128, lang-deu (measured over k=1) | DSpark 1.00 / 0.82 / 0.64 / 0.52, model at B=64 1.00 / 0.82 / 0.59 / 0.46; EAGLE 1.00 / 0.76 / 0.58 / 0.46, model 1.00 / 0.75 / 0.54 / 0.43 |
| tool-call at C=128 | flatter than the model: measured 1.00 / 0.97 / 0.94 / 0.92, model 1.00 / 1.24 / 1.11 / 0.98. With 2.2k-token prompts and 64 prompts per pass, the cell is mostly prefill, which the model leaves out. |

For the same reason, the model's long-context speedups (about 2x for DSpark at C=1) are upper bounds: a 14k-token prefill per request is not counted.

### Results

Acceptance length (tokens per step) at C=1, and the best k by the cost model:
- "plain" means no draft length beats the plain target at that load.
- "—" means B requests of that slice do not fit in the KV cache (long-context fits about 24 per GPU with DSpark and 31 with EAGLE, tool-call about 150-185).
- Sorted by DSpark k=7 acceptance.

| slice | corpus share | DSpark k=3 / k=7 | EAGLE k=3 / k=7 | DSpark best k, B = 1 / 32 / 128 / 256 | EAGLE best k, B = 1 / 32 / 128 / 256 |
| --- | --- | --- | --- | --- | --- |
| math-gsm8k | 43.79% | 3.74 / 6.72 | 3.69 / 6.51 | 7 / 7 / 5 / 4 | 7 / 7 / 5 / 4 |
| json-schema | 0.040% | 3.73 / 6.61 | 3.69 / 6.47 | 7 / 7 / 5 / 5 | 7 / 7 / 5 / 3 |
| code-multi | 1.78% | 3.50 / 5.63 | 3.42 / 5.55 | 7 / 4 / 3 / 3 | 7 / 4 / 3 / 2 |
| translate-to-en | 0.662% | 3.05 / 3.99 | 2.78 / 3.39 | 7 / 4 / 1 / plain | 5 / 4 / 1 / 1 |
| multi-turn | 25.01% | 2.96 / 3.82 | 2.86 / 3.67 | 7 / 4 / 1 / 1 | 5 / 4 / 1 / 1 |
| tool-call | 0.081% | 2.90 / 3.73 | 2.73 / 3.51 | 7 / 4 / 3 / — | 5 / 3 / 3 / — |
| lang-eng | 99.55% | 2.69 / 3.24 | 2.62 / 3.24 | 7 / 4 / 1 / plain | 4 / 4 / 1 / plain |
| long-context | 0 | 2.47 / 2.71 | 1.79 / 1.81 | 5 / — / — / — | 2 / — / — / — |
| lang-amh | 0 | 2.12 / 2.23 | 2.08 / 2.22 | 5 / 2 / plain / plain | 3 / 2 / 1 / plain |
| lang-yor | 2 rows | 1.97 / 2.04 | 2.03 / 2.47 | 5 / 3 / plain / plain | 3 / 1 / 1 / plain |
| lang-spa | 0.040% | 1.94 / 2.04 | 1.83 / 1.91 | 5 / 2 / plain / plain | 3 / 1 / plain / plain |
| lang-zho | 0.037% | 1.92 / 2.03 | 1.84 / 1.97 | 5 / 2 / plain / plain | 3 / 1 / plain / plain |
| lang-fra | 0.054% | 1.82 / 1.88 | 1.72 / 1.79 | 5 / 2 / plain / plain | 2 / 1 / plain / plain |
| translate-from-en | 0.662% | 1.79 / 1.87 | 1.70 / 1.76 | 5 / 2 / plain / plain | 2 / 1 / plain / plain |
| lang-jpn | 0.021% | 1.66 / 1.73 | 1.62 / 1.71 | 6 / 2 / plain / plain | 1 / 1 / plain / plain |
| lang-ita | 0.024% | 1.65 / 1.71 | 1.61 / 1.69 | 5 / 2 / plain / plain | 2 / 1 / plain / plain |
| lang-hin | 0.011% | 1.64 / 1.70 | 1.62 / 1.69 | 6 / 2 / plain / plain | 1 / 1 / plain / plain |
| lang-roh | 0 | 1.64 / 1.69 | 1.60 / 1.67 | 5 / 2 / plain / plain | 1 / 1 / plain / plain |
| lang-deu | 0.027% | 1.61 / 1.66 | 1.57 / 1.61 | 5 / 2 / plain / plain | 1 / 1 / plain / plain |
| lang-swa | 0 | 1.55 / 1.58 | 1.56 / 1.62 | 3 / plain / plain / plain | 1 / 1 / plain / plain |
| lang-ara | 0.007% | 1.44 / 1.55 | 1.37 / 1.45 | 7 / plain / plain / plain | 1 / 1 / plain / plain |

Acceptance does not change with load. At C=128 the acceptance length is within 2.5% of C=1 for every k and both drafters on the three load-check slices. The largest gap is lang-roh, DSpark k=5: 1.73 vs 1.69.

Reading the table:
- **Language is the dominant factor.** On the same reverse-instruction task, English accepts 3.24 at k=7 and every other language 1.55-2.23. This holds for the large European languages Apertus targets (de 1.66, it 1.71, fr 1.88, es 2.04) as for languages absent from the corpus (Romansh 1.69, Swahili 1.58). The corpus shares of these languages are all at or below 0.05%, so the drafters have seen essentially no non-English answers. A share of 0.03% buys nothing over 0%.
- **Translation follows the output language.** Translating into English accepts 3.99 (k=7), out of English 1.87, the same as the target languages themselves. Having the source text in the prompt does not help when the answer is in a language the drafter has not learned.
- **Structured and formulaic output accepts well with little data.** JSON that follows a schema (0.04% of the corpus) accepts 6.61, as high as math (44%). Non-Python code (1.8%) accepts 5.63, close to the math/code probe (about 6.7). Tool calls (0.08%) accept 3.73, above English free text.
- **Long contexts hurt EAGLE much more than DSpark.** 13k-token prompts: DSpark 2.71, EAGLE 1.81, against 3.24/3.24 on short English prompts. No training row has a prompt above 8k tokens. A likely cause, not tested here: DSpark's drafter attends over a 2048-token window, while EAGLE's head attends over the whole context, at positions it never saw in training.
- **Multi-turn** (MT-Bench second turns, 25% of the corpus) accepts 3.82, above the SFT-mix chat workload (about 2.6).
- **EAGLE and DSpark accept within 7% of each other** at k=7 on every short-prompt slice but two: translation into English (EAGLE 3.39 vs 3.99) and Yoruba (EAGLE higher, 2.47 vs 2.04). The difference in best k comes from draft cost: EAGLE pays a pass per drafted token, DSpark one step for all of them. So EAGLE's best k is shorter at C=1 (1-3 on the languages, where DSpark takes 5).

### What needs drafter data (for the drafter-updates work)

From weakest up, DSpark e10 at k=7:

| priority | slices | acceptance at k=7 | corpus share |
| --- | --- | --- | --- |
| 1 | Arabic, Swahili, German, Romansh, Hindi, Italian, Japanese | 1.55-1.73 | 0-0.03% each |
| 2 | from-English translation (de/fr/it/es), French, Chinese, Spanish, Yoruba, Amharic | 1.87-2.23 | 0-0.05% each (translation requests 0.66%) |
| 3 | long context above 8k tokens (EAGLE) | EAGLE 1.81, DSpark 2.71 | 0 |
| 4 | English free-form writing (lang-eng), tool calls, multi-turn | 3.2-3.8 | |

German, French and Italian matter most for Apertus. The corpus has almost nothing in them (0.03-0.05%). The English control shows how high the same task can reach (3.24 against 1.66-1.88). The data needed is non-English prompts with the target's own answers, as the corpus was regenerated.

### Draft length per task (for the per-request-k work)

From `results/8b/coverage/best-k.json`. B is running requests per GPU (a node of four DP engines holds 4 B). Prefill is not modelled.

| task | DSpark e10: C=1 / B=32 / B=128 / B=256 | EAGLE e10: C=1 / B=32 / B=128 / B=256 |
| --- | --- | --- |
| math, JSON schema | 7 / 7 / 5 / 4-5 | 7 / 7 / 5 / 3-4 |
| code (non-Python), tool calls | 7 / 4 / 3 / 3 | 5-7 / 3-4 / 3 / 2-3 |
| English chat, multi-turn, to-English translation | 7 / 4 / 1 / plain-1 | 4-5 / 4 / 1 / plain-1 |
| long context (up to about 24 per GPU) | 5 | 2 |
| non-English languages, from-English translation | 5 / 2 / plain / plain | 1-3 / 1 / plain / plain |

## Runs

| sweep | job | what |
| --- | --- | --- |
| `pp-smoke-20261022T125029Z` | 3721313 | PP2: plain, DSpark e10 k=3, EAGLE e10 k=3 (v2 and v1 runners); drafter arms refused |
| `multigpu-pp-20261022T130434Z` | 3721388 | plain DP4, PP4, PP2xDP2, TP4, DSpark e10 k=7 DP4; summarization C=4-512 |
| (node command) | 3722012 | coverage slices built; corpus shares over the full OPB corpus |
| `coverage-a-20261022T153452Z` | 3722605 | DSpark e10 k=1/3/5/7 on the 21 slices, C=1 (C=128 on lang-roh, lang-deu, tool-call) |
| `coverage-b-20261022T205359Z` | 3725070 | EAGLE e10 k=1/3/5/7, same cells |
