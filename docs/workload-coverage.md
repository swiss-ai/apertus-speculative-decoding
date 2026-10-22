# Workload coverage and pipeline parallelism, Apertus 1.5 8B

Yu (DM, 15 Oct): with the 8B checkpoints in hand, test them on different
workloads and configurations first: out-of-distribution requests, datasets
that were trained on but have little data, P/D disaggregation or not, TP or
PP. TP, DP and P/D are in [system-optimizations-8b.md](system-optimizations-8b.md)
("Several GPUs and disaggregation"). This document covers pipeline
parallelism and the under-represented languages and domains.

Status 2026-10-22:
- PP is measured (plain only; neither drafter can run with PP).
- The coverage slices and the training-corpus shares are built.
- The per-slice acceptance sweeps (coverage-a/b, jobs 3722605 and 3725070) are queued.

## Summary

| question | result |
| --- | --- |
| speculative decoding with PP | **Not supported on this image.** DSpark e10 and EAGLE e10 (model runner v2, and EAGLE on v1) are refused when the config is validated, because the draft model inherits the target's PP size and does not implement `SupportsPP`. Behind that refusal, the v2 runner refuses eagle3/dspark with PP, and so does DSpark's own setup code. Plain PP serves. |
| plain PP on a node | **Never worth it for an 8B target.** PP4 gives 0.51-0.68x of plain DP4's node throughput, and PP2xDP2 0.73-0.94x, at C = 4-512. TP4 is faster at low load (1.82x at C=4). DSpark on DP4 beats every PP layout up to C=256 (1.56x to 0.97x of DP4). |
| training-corpus share of the slices | Open-PerfectBlend (1.80M rows) is 99.55% English prompts. de/fr/it/es/zh/ja are 0.02-0.05% each, hi/ar about 0.01%. Romansh, Swiss German, Swahili and Amharic do not occur, and Yoruba has 2 rows. No prompt reaches 8k tokens. Tool calls are 0.08% and JSON schemas 0.04%. Translation requests are 0.7%, non-Python code 1.8%, math 44% and multi-turn 25%. |
| acceptance per slice | pending (coverage-a/b) |

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

Pending: `coverage-a` (DSpark e10) and `coverage-b` (EAGLE e10) measure
k = 1/3/5/7. Each slice gets one C=1 cell; lang-roh, lang-deu and tool-call
also get a C=128 cell.

## Runs

| sweep | job | what |
| --- | --- | --- |
| `pp-smoke-20261022T125029Z` | 3721313 | PP2: plain, DSpark e10 k=3, EAGLE e10 k=3 (v2 and v1 runners); drafter arms refused |
| `multigpu-pp-20261022T130434Z` | 3721388 | plain DP4, PP4, PP2xDP2, TP4, DSpark e10 k=7 DP4; summarization C=4-512 |
| (node command) | 3722012 | coverage slices built; corpus shares over the full OPB corpus |
| `coverage-a-20261022T153452Z` | 3722605 | DSpark e10 k=1/3/5/7 on the 21 slices (queued) |
| `coverage-b-20261022T205359Z` | 3725070 | EAGLE e10 k=1/3/5/7 on the 21 slices (queued) |
