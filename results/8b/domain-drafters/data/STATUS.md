# Domain drafters (A3): trace corpus status

Branch `experiment/domain-drafters-data`. Per-domain stand-in served traces
from the Apertus 1.5 8B target, for the sibling DSpark and EAGLE domain grids.

## Contract

`/iopsstor/scratch/cscs/faruk_zahiragic/spec-exp-data/domain-drafters/traces/`

- `<domain>/{train.jsonl,dev.jsonl}`: 8,000 / 256 rows for math, code, chat,
  summarization, multilingual, tool. Row format is `generate_targets`'
  (`prompt_ids`, `generated_ids`, `finish_reason`, ...), as the T5
  `multilingual-n8000.jsonl`.
- `pilot/<domain>/{train.jsonl,dev.jsonl}`: first 512 / 64 prompts of the same
  hash-ordered splits (nested subsets of the full splits).
- `MANIFEST.json`: rows, sha256, finish reasons, generation settings, render
  parity, pool manifests (sources and revisions), job IDs.
- `READY-pilot` once all six pilot domains exist; `READY-<domain>` per full domain.
- Backup: `/capstor/scratch/cscs/faruk_zahiragic/domain-drafters-backup/`.

Generation: `generate_targets` on one GH200 GPU per domain, thinking off
(served chat template), temperature 0.7 / top-p 0.8 / top-k 20, per-row seed
from id and seed 1, natural EOS, 2,048 new-token cap, prompts up to 4,096
tokens. Same settings as the T5 traces.

## Prompt sources

| domain | source (pinned revision) |
| --- | --- |
| math | openai/gsm8k main train (740312ad), EleutherAI/hendrycks_math 7 subjects train (21a56338) |
| code | google-research-datasets/mbpp full train/validation/prompt (4bb6404f), ise-uiuc/Magicoder-OSS-Instruct-75K (5f839b1f) |
| tool | glaiveai/glaive-function-calling-v2 (e7f4b645): function definitions + first user turn as one user message |
| chat | T5 builder: SFT-mix train rows (chat_qa / multilingual / instruction_following), restricted, cluster only |
| summarization | T5 builder: finepdfs-edu eng/deu/fra/ita files 4-7, restricted, cluster only |
| multilingual | T5 builder: Infomaniak-AI/speculators-multilingual-en-fr-de-it-es (b1d69387) fr/de/it/es |

Overlap guard (prompt-token digest, normalized text, 13-gram near-duplicate)
against `workloads/8b/`: eagle-8b-test, eagle-8b-validation, eagle-8b-profile,
arena-hard, opb-heldout, opb-calib, probe-speculator-benchmarks (+thinking).
The coverage workload (`coverage.jsonl`, task-speed slices) was deleted by the
scratch cleaner and is not in the guard; its sources are disjoint by split or
dataset (GSM8K test, hermes-function-calling-v1, humanevalpack, mt_bench,
wmt24pp, LongBench, muri-it). Re-check overlap once it is rebuilt.

Public source files pre-downloaded with `huggingface_hub` at the pinned
revisions to `domain-drafters/src/<owner>__<name>/` (with `revision.txt`).
Candidate counts (login-node parse check): math 14,973, code 75,671, tool 78,362.

## Submissions (2026-11-20)

Source snapshot `/iopsstor/scratch/cscs/faruk_zahiragic/apertus-domain-data-472b7c43`
(commit 472b7c43, `git archive`), image `vllm_apertus_1.5_release-arm64.sqsh`
(digest written to `domain-drafters/image.sha256` by the first job).

| job | partition | content | state at submission |
| --- | --- | --- | --- |
| 3987893 | normal | pools if needed, then math, code, tool (pilot then full) | pending, infra01 QOSGrpNodeLimit (est. start ~17:20) |
| 3987894 | normal | pools if needed, then chat, summarization, multilingual | pending, same |
| pilot | debug | pools + pilot of all six | debug per-user submit limit full; login-node loop `submissions/retry-pilot.sh` resubmits every 3 min (job id in `submissions/pilot-job-id`) |
