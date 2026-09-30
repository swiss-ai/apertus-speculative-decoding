# DSpark evaluation benchmarks vs the open-perfectblend training corpus

Both drafters (DSpark and EAGLE 3.1) train on `mlabonne/open-perfectblend`
(revision `af60f3c`). A benchmark prompt that is also a training prompt inflates
acceptance on that benchmark. Check: exact normalized user text, or 13-gram
Jaccard >= 0.5 on the first 2,000 characters, against every user turn of the
1,420,909 corpus rows (`apertus_eagle.benchmark_overlap`, job 3550031, report
`results/8b/eagle/data-perfectblend/benchmark-overlap.json`). Benchmarks are
the public Hugging Face copies; the DeepSpec harness may use other copies.

| benchmark | source | prompts | in corpus | share | corpus source of the matches |
| --- | --- | --- | --- | --- | --- |
| GSM8K | openai/gsm8k test | 1,319 | 7 | 0.5% | lmsys arena |
| MATH500 | HuggingFaceH4/MATH-500 | 500 | 7 | 1.4% | MetaMathQA, UltraInteract |
| AIME25 | opencompass/AIME2025 | 30 | 0 | 0% | |
| MBPP | mbpp full test | 499 | 0 | 0% | |
| HumanEval | openai_humaneval | 164 | 10 | 6.1% | evol-codealpaca, lmsys arena |
| LiveCodeBench | code_generation_lite v6 | 1,055 | 1 | 0.1% | lmsys arena |
| MT-Bench | mt_bench_prompts | 80 | 2 | 2.5% | lmsys arena |
| AlpacaEval | tatsu-lab/alpaca_eval | 805 | 48 | 6.0% | AutoIF, lmsys arena, UltraFeedback |
| **Arena-Hard v0.1** | lmarena-ai/arena-hard-auto | 500 | **129** | **25.8%** | lmsys arena (Arena-Hard's prompts come from Chatbot Arena) |

Also: 77 of the 128 code prompts of our own A6 test set have near duplicates in
the corpus (`check.json`); chat and summarization are clean.

Recommendation: report Arena-Hard, and preferably AlpacaEval and HumanEval,
both on all prompts and on the clean subset (`contaminated_ids` in the report).
