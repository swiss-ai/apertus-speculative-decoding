# DSpark repeat of the best drafter-adaptation setting (8B)

Branch `experiment/dspark-adaptation`, run 2026-11-04/05. This is the
pre-registered conditional follow-up of the T5 adaptation study
([drafter-updates.md](drafter-updates.md)). The best EAGLE setting
(`adapt-mult-n8k-r0`) qualified in serving (job 3814370), which triggered one
repeat on the DSpark e10 head (Apertus 1.5 8B, thinking off, served at k=3).
There was no search over settings.

## Verdict

**No implementation case.** Two results decide it:

- **Correctness gate.** The frozen calibrated correctness gate failed in 1 of
  40 checks. Under the frozen policy that makes the run terminal
  (`needs_repair`), so no serving claim is made.
- **Benefit rule.** Even setting correctness aside, no workload, k or
  concurrency cell meets the unchanged substantial-benefit rule: at least 10%
  more throughput **and** at least 10% lower median TPOT in both GPU-swapped
  repeats.
  - The adapted DSpark head is consistently but modestly better on
    multilingual traffic. At the served k=3 the gain is +3.0–5.0%
    throughput and −5.0–6.9% median TPOT.
  - At k=7 the gain is +6.0–9.1% throughput and −11–13% TPOT. TPOT passes
    there, but throughput misses 1.10 in every cell.
  - OPB, code, math and multi-turn change by less than ±2.5% in throughput
    (no forgetting).

I did not rerun the failed gate. A passing rerun could not make any cell
qualify, so a repair would only buy a correctness label for a negative
result.

Compared with EAGLE: the same data and recipe moved the EAGLE e10 head much
further.

| | development gain | k3/C1 multilingual throughput |
| --- | --- | --- |
| EAGLE | simulated acceptance 1.66 → 2.42 (+46%) | 267 → 312 tok/s (+17%, job 3814370) |
| DSpark | trainer EAL 3.84 → 4.14 (+7.8%) | 280 → 293 tok/s (+4.6%, this run) |

DSpark e10 already starts closer to the target on these languages. The two
runs come from different jobs, so the cross-method comparison carries
between-job spread (up to about 5%).

## Prerequisites (verified before training)

- **Data.** The trace data is compatible as it stands. The 8,000 multilingual
  traces (`adapt-data/multilingual-n8000.jsonl`, sha256 2038cf24…) and 256
  dev traces (de3693e6…) already carry `prompt_ids` and `generated_ids`.
  - They become speculators rows: `input_ids` = prompt + completion,
    `loss_mask` = 0 on the prompt and 1 on the completion. This is the layout
    of Yu's DSpark corpus, with the same chat template prefix and EOS 68.
    Nothing is re-tokenized.
  - Target hidden states are not stored. The trainer's per-rank vLLM
    extraction servers generate them online, as in e10's own training, so no
    features had to be regenerated.
  - 44 traces stop at the 2,048-token generation cap. They are kept, as in
    the EAGLE run.
- **Port.** The 8B port from `small-drafter` (bf91486a) warm-starts with
  `--from-pretrained`. e10's config points its verifier at an inaccessible
  private snapshot of the same revision (a411d838). A rebound copy changes
  only that path, and the weights are symlinked and hash-checked
  (002d1167…).
- **Export.** The trained checkpoint is a speculators DSpark checkpoint that
  the vLLM fork serves directly, like e10. The export has the same tensor
  names, shapes and dtypes as e10, and the same config apart from the
  verifier path. Its reload is bitwise equal.
- **Two pitfalls found and fixed.**
  - `speculators prepare-data` shuffles rows, but the trainer holds out the
    *trailing* rows by position. The dev/OPB rows would have been trained on.
    The row check caught this in job 3861814. The dataset is now built with
    speculators' own loader and pretokenized pass-through, without the
    shuffle (4bbc042b).
  - speculators has no eval-only mode. Initial and final acceptance are
    validation epochs after one optimizer step at learning rate 0, and all
    four evaluations were checked to leave the weights bitwise unchanged.

## Frozen mapping of the EAGLE setting

The full mapping is in `methods/dspark/configs/8b/adapt-mult-n8k-r0.json`.

| | EAGLE `adapt-mult-n8k-r0` | DSpark repeat |
| --- | --- | --- |
| data | 8,000 multilingual traces, no replay | same file |
| steps | 384 | 384 (epoch = 257 steps) |
| rows per step | 32 (12,288 rows) | ~31 (16 rows per rank at the 584-token mean, `total_seq_len` 9,344, 2 ranks) |
| LR / schedule | 3e-5, constant after 19 warmup steps | 3e-5 for Muon and AdamW, cosine with 0 cycles = constant after 19 warmup steps |
| seed | 1 | 1 |
| other | EAGLE e10 recipe | e10 recipe unchanged (block 8, Markov 256, confidence head, CE 0.1 / TV 0.9, Muon, anchors 1024) |
| dev metric | simulated acceptance, 256 dev traces | trainer `eal_epoch`, same 256 dev traces |
| forgetting | first 512 OPB held-out rows | same rows |
| selection | all gates and dev gain ≥ +0.20 | same rule |

## Development results (job 3861955, 46 min on one node)

| | e10 | adapted |
| --- | --- | --- |
| multilingual dev EAL | 3.842 | **4.142 (+0.301)** |
| multilingual dev accept length | 3.678 | 3.953 |
| multilingual dev val loss | 0.554 | 0.503 |
| OPB held-out EAL (512 rows) | 6.269 | 6.249 (−0.33%) |
| OPB held-out val loss | 0.2362 | 0.2372 |

- **Gates.** All gates pass: finite losses, real updates (42 of 62 tensors
  changed, max |Δ| 0.015), loss decreased, held-out acceptance improved,
  384 steps, export reload bitwise, and lr-0 evaluations unchanged.
- **Selection.** The head is selected under the unchanged rule.
- **Exported head.** `ROOT/head`, model.safetensors sha256
  `f330419206bebb401a6d1117afde552d63968a46cec3d5b5c51aafae1de69165`.
- **Caveat.** These are trainer metrics, not serving acceptance.

## Serving confirmation (jobs 3862494 round a, 3862575 round b)

The design is unchanged from 3814370:

- base DSpark e10 vs adapted at k3 and k7, C = 1/64/256;
- ten workloads: OPB held-out, the multilingual test set, `lang-deu/fra/ita/spa`,
  `translate-from-en`, `multi-turn`, `code-multi` and `math-gsm8k`, with the
  same files and corpus hashes;
- round a on GPUs 0–3, round b with every pair swapped;
- full prompt passes, Apertus sampling, 240 cells and zero failed warmups.

The debug partition limits jobs to 1.5 h, so each round ran as its own job
into one driver output. The driver checks the frozen source manifest in both
jobs.

**Correctness.** 39 of 40 calibrated gates pass. The failure is
`lang-spa`, k7, round b, on maximum completion-token difference: 131 tokens
against an allowance of 124 (control envelope 122 + 2).

- Every other check of that gate passes: common prefix, edit distance and
  comparability.
- The treatment's exact greedy matches (6/32) equal those of the base/base
  cross-deployment control (6/32).
- DSpark greedy outputs diverge heavily between independent deployments of
  the same head anyway (6–10 of 32 exact matches). The failure is consistent
  with that noise, but the frozen rule counts it, and no limit was changed.

**Absolute numbers (base → adapted, repeat a / repeat b).** The sources are
`verdict.json` and `serving-absolute-cells.csv`.

| workload | k | C | throughput (tok/s) | median TPOT (ms) | acceptance length |
| --- | --- | --- | --- | --- | --- |
| multilingual | 3 | 1 | 280→293 / 282→294 | 3.29→3.07 / 3.26→3.08 | 2.37→2.49 / 2.38→2.49 |
| multilingual | 3 | 64 | 7122→7339 / 6997→7237 | 7.31→6.81 / 7.51→7.01 | 2.36→2.50 / 2.36→2.50 |
| multilingual | 3 | 256 | 8804→9174 / 8715→9154 | 24.67→23.43 / 25.14→23.52 | 2.36→2.51 / 2.37→2.51 |
| multilingual | 7 | 1 | 327→355 / 329→354 | 2.72→2.43 / 2.81→2.47 | 2.75→3.00 / 2.77→2.98 |
| multilingual | 7 | 64 | 5133→5581 / 5176→5487 | 10.10→8.93 / 9.90→8.65 | 2.77→3.02 / 2.78→3.02 |
| multilingual | 7 | 256 | 5995→6538 / 5958→6470 | 35.31→31.11 / 35.47→31.42 | 2.79→3.03 / 2.78→3.03 |
| lang-ita | 3 | 1 | 195→201 / 198→202 | 4.94→4.78 / 4.88→4.75 | 1.64→1.70 / 1.66→1.70 |
| lang-deu | 3 | 256 | 7042→7147 / 6976→7118 | 33.40→33.10 / 33.69→33.34 | 1.60→1.64 / 1.60→1.64 |
| opb (forgetting) | 3 | 1 | 375→374 / 379→374 | 2.45→2.43 / 2.41→2.42 | 3.18→3.17 / 3.20→3.18 |
| opb (forgetting) | 3 | 64 | 7279→7183 / 7196→7090 | 6.85→7.07 / 6.89→7.04 | 3.18→3.18 / 3.19→3.18 |
| opb (forgetting) | 3 | 256 | 8798→8776 / 8768→8700 | 24.86→24.76 / 24.87→25.06 | 3.18→3.16 / 3.18→3.16 |

Ratio ranges by workload group (adapted / base, both repeats):

- **Independent language slices** (`lang-*`, `translate-from-en`): +0–4%
  throughput at k3 and +1–6.5% at k7.
- **Forgetting workloads:**
  - OPB k3: −0.3 to −1.5% throughput, with median TPOT up to +3.1% at C64.
  - OPB k7: −0.5% to +2.5% throughput.
  - multi-turn, code and math: within −0.8% to +3.9%.
- **TTFT p95.** Ratios above 1.05 appear in many cells, forgetting cells
  included, and below 0.95 in others. TTFT is prefill-dominated and the
  drafter barely touches it, so treat this as run-to-run TTFT noise.

**Compute.** This counts every DSpark job, including the three failed
attempts and the cancelled job 3856327:

- adaptation: 11,980 GPU-s (3.33 GPUh);
- confirmation: 14,908 GPU-s;
- reused EAGLE trace allocation: 11,380 GPU-s;
- total: 38,268 GPU-s (10.6 GPUh).

The successful training job alone took 46 min on one node, all
evaluations included. The 384 steps took 7.3 min.

## Limits

- One setting, one seed, one domain. These are stand-in served traces, not
  production captures, and the conclusions are limited to the recorded
  workloads and concurrencies.
- The learning rate was copied as a value (3e-5). That is 10% of DSpark e10's
  peak LR (3e-4, Muon), against 30% for EAGLE (peak 1e-4). The pre-registered
  repeat did not tune it, so a stronger DSpark adaptation may exist; testing it
  would be a new experiment.
- The trainer EAL is not serving acceptance. The dev gain (+7.8%) shows up in
  serving as +5% (k3) to +9% (k7) acceptance length on the multilingual test
  set.
- There is no plain (non-speculative) arm. The frozen 3814370 design compares
  base and adapted drafters only.
- Calibrated correctness is consistent with losslessness, not a proof of it,
  and here it failed one check.

## Records

- `results/8b/dspark-adaptation/`: `verdict.json`, `serving-absolute-cells.csv`,
  `STATUS.md` (every job), `development/` (development report, conversion and
  row checks, rebinding and export reports, lr-0 checks, trainer configs and
  commands, speculators patch, extraction-server vLLM provenance, trainer
  logs), and `confirmation-debug-20261105/`.
  - The confirmation folder holds the spec, selection, receipts and a
    sha256 manifest of every confirmation file.
  - `confirmation-debug-20261105-cells.tgz` has each cell's `metadata.json`,
    `summary.json`, Prometheus snapshots and time series, plus the load-test
    summaries, correctness reports, provenance and replica logs.
- Kept on the cluster only:
  - `requests.jsonl` (252 MB; it holds no text, only hashes and timings);
  - greedy captures (restricted output text);
  - weights, converted rows and prepared datasets.

  They are under
  `/iopsstor/scratch/cscs/faruk_zahiragic/spec-exp-data/dspark-adaptation/adapt-mult-n8k-r0-20261104`.
  `confirmation-debug-20261105-raw.tgz` next to it (sha256 e929ca8b…) bundles
  the raw cells, including `requests.jsonl`.
