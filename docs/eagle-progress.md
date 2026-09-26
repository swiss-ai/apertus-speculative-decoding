# EAGLE 3.1 execution ledger

Layout note 2026-09-26: the repository was reorganised by method and target. Entries
below keep the paths that were current when they were written. Mapping:

| old | new |
| --- | --- |
| `training/apertus_eagle/` | `methods/eagle/apertus_eagle/` |
| `launch/{eagle,eagle8b-*,stagea-campaign,*train*,*torchspec*,prepare-eagle-splits,probe-renderer}.sh` | `methods/eagle/launch/` |
| `launch/{baseline,resolve-env,stage-defaults,patch-vllm,vllm-extra-flags,wait-replica}.sh` | `serving/` |
| `launch/{draft-model,standalone-8b}.sh`, `configs/drafter-inventory.json` | `methods/draft_model/` |
| `launch/ngram.sh` | `methods/ngram/` |
| `launch/*diagnostics*.sh`, `configs/diagnostics.yaml` | `experiments/round-cost-70b/` |
| `configs/experiment.yaml` | `experiments/smoke-70b/` |
| `patches/vllm-*`, `patches/torchspec-*` | `serving/patches/`, `methods/eagle/patches/` |
| `configs/eagle/8b/*` | `methods/eagle/configs/8b/` (`lock.json` -> `targets/8b/`) |
| `configs/eagle/{draft,train}-*` | `methods/eagle/configs/70b/` (`lock.json` -> `targets/70b/`) |
| `results/eagle/8b/preflight/compatibility.json` | `targets/8b/contract.json` |
| `results/eagle/preflight/compatibility.json`, `target-*.json` | `targets/70b/contract.json`, `targets/70b/checkpoint/` |
| `results/eagle/8b/workloads/` | `workloads/8b/` |
| `results/eagle/8b/` | `results/8b/eagle/` |
| `results/eagle/` (70B) and the other `results/*` | `results/70b/` |

Scope update 2026-09-24: [the revised execution plan](eagle-execution-plan.md) starts with
an Apertus 1.5 8B pilot (A0–A6), followed by 70B transfer (B0–B3). EAGLE 3.1 remains the
default; EAGLE-3 is a conditional fallback. The entries below are preserved snapshots of
earlier work, not the active task gates or a current cluster-state check. The revised plan
and manifest have been updated; no 8B implementation or GPU run is claimed by this notice.

## Stage A (8B) ledger

Updated 2026-09-24 10:25 UTC. Cluster access: `ssh clx` (multiplexed alias for
`clariden.alps.cscs.ch` via `ela`); the renewed CSCS cert is valid until
2026-09-25 11:35 CEST. Clariden login sessions hung for ~15 minutes at 09:35 UTC
(authentication succeeded, session channel never opened) and recovered at 09:48.

| Task | Status | Evidence / next action |
| --- | --- | --- |
| A0 target/environment | Contract recorded; environment versions pending first job print. | `results/eagle/8b/preflight/compatibility.json`. Lock file after job 3505554 prints versions. |
| A1 retarget scaffolding | Done; 84 local tests pass. | See A1 below. |
| A2 data and parity | **Pass.** Job 3505554 (debug, 52:40, FAILED 1:0 only at the pooled parity gate, corrected below). | `results/eagle/8b/data/`, `results/eagle/8b/preflight/{hf-parity,feature-parity}.json`, `results/eagle/8b/data/split-overlap.json`. |
| A3 train/export | Overfit **pass** (3508056). Pilot trained (3508057, gates pass); export check rerun with verify in **3509300**. | `results/eagle/8b/runs/`, `results/eagle/8b/heads/`. Job 3505427 was cancelled while held: its sbatch predates the vLLM overlays that `verify` needs. |
| A4–A6 | Not started. | Need a reloadable 8B head. |

Last 70B record: tiny-overfit **3492049** FAILED 1:0 after 00:37:53 on nid006905
(2026-09-23 11:41–12:19 UTC). It is `overfit_direct` exiting 1 because the one-step
loss ended at 0.150 (from 14.53, 200 steps), above its 0.05 threshold. A 6.6 GB
`results/eagle/heads/e31-overfit/model.safetensors` exists on Clariden. It is a
one-step 70B debug head, not a rollout-trained head, and is outside Stage A.

### A2 results (job 3505554)

- Regeneration (285 s, 8B TP=1 vLLM): overfit 32/32 `stop`; validation 121 `stop`, 7
  `length`; train 1000 rows. `LLM.chat` prompt ids equal ours on 32 prompts.
  Summary: `results/eagle/8b/data/generation-summary.json`.
- Overlap: no exact or 13-gram near-duplicate user prompts between
  validation/heldout_test and train/overfit.
- Caches (`/iopsstor/scratch/cscs/faruk_zahiragic/apertus-eagle/8b/features/`):
  overfit 28,725 tokens; validation 128 rows; train 1000 rows, 927,033 tokens,
  30.4 GB; 32,773 bytes/token; peak GPU 19–22 GB; teacher loader
  `AutoModelForImageTextToText` (Python xIELU fallback).
- HF self-parity (8 prefixes): `lm_head(norm output)` equals the model logits
  exactly (max abs 0, argmax 100%); repeated forward bitwise identical; model
  logits beyond id 131071 are bf16 min (masked).
- HF versus vLLM (8 prefixes, <=1024 tokens, eager TP=1): claimed alignment
  HF layer i = vLLM index i+1 has relative error 0.003–0.028; its neighbours
  0.059–44. Worst claimed/nearest-neighbour ratio 0.086. vLLM repeats bitwise
  identical. Next-token argmax agreement 98.2–99.6%; every disagreement has an
  HF top-2 margin <= 0.75 logits.
- Gate correction: the first gate pooled worst claimed error over all layers
  (0.028, layer 28) against best neighbour error over all layers (0.059, layer
  15) and failed. Error grows with depth, so that compares different layers.
  The gate is now per sample and per layer (claimed <= 0.2 x nearest
  neighbour); it passes at 0.086. Changed before any training result existed;
  the as-run summary is kept as `summary_as_run` in the report.

### A3 overfit (job 3508056, debug, COMPLETED 0:0, 31:48)

Attempt 3506531 stopped after step 1 on my own guard ("first step did not change
fc.weight"): TorchSpec's warmup starts at `init_lr=0`, so step 1 is a zero update by
design. The guard now checks the lr each step actually used. Pilot 3506532 was
killed by its `afterok` dependency, as intended.

Rerun 3508056: 293,629,952 trainable / 1,073,741,824 frozen parameters; 400 steps
in 688.8 s (≈7.3k train tokens/s after compile; first step 17 s); peak GPU
22.5 GB. Train loss 18.54 (first decile) → 0.22 (last decile). Held-out here =
the same 32 rows (memorization check):

| step | loss | sim. acc. length | acc position 0..6 |
| --- | --- | --- | --- |
| 0 | 31.47 | 0.00 | 0.00 |
| 80 | 1.37 | 2.42 | 0.86 … 0.62 |
| 160 | 0.30 | 6.06 | 0.96 all |
| 400 | 0.22 | 6.87 | 0.997, 0.996, 0.994, 0.994, 0.994, 0.994, 0.993 |

All four gates true. Export `…/8b/heads/e31-overfit-3508056` reloads bit-exactly
(first-step logits max diff 0, rollout metrics identical); checkpoint manifest
`fefdf20e40426bca9a81f76927903ad3e89297e1778c23c01e2a0fc3f9640ea2`. This is a debug
head (memorized 32 rows), not a serving candidate.

### A3 pilot (job 3508057, normal, 01:02:01)

1000 target-regenerated conversations (927k tokens), 2000 optimizer steps x 16
sequences (32 epochs), eval every 100 steps on the disjoint 128-row validation
split. All training gates true (finite, real updates, loss decreased, held-out
acceptance improved). Early stopping (3 evals without improvement) never fired.

| step | held-out loss | sim. acc. length | acc position 0..6 |
| --- | --- | --- | --- |
| 0 | 31.25 | 0.00 | 0 |
| 100 | 3.78 | 0.75 | 0.51 … 0.30 |
| 500 | 1.94 | 1.80 | 0.75 … 0.53 |
| 800 | 1.82 | 2.01 | 0.76 … 0.58 |
| 1500 | 1.87 | 2.09 | 0.77 … 0.60 |
| 2000 | 1.92 | 2.10 | 0.768, 0.682, 0.646, 0.627, 0.616, 0.606, 0.599 |

Held-out loss is lowest at step 800 and rises after; acceptance keeps a small
upward drift and plateaus at ~2.1 from step 1500. The 1000-row corpus is the
limit (32 epochs). The best-by-acceptance checkpoint is step 2000.

Export first failed its own reload check: logits bit-identical and accuracies
identical, but rollout loss differed by 1.7e-6 and the check demanded 0. The
compiled fp32 loss reduces in a nondeterministic order. The check now requires
bitwise-equal exported tensors, identical logits and accuracies, and a loss
difference within the same-weights repeat noise (floor 1e-5). The failed export
was moved to `…/heads/e31-pilot-3508057.reload-check-too-strict`.

### A4 offline verification (jobs 3509300, 3509367, 3509606)

- 3509300: export passed the new reload check; the first capture then failed
  with "Free memory on device cuda:0 (72.43/95.0 GiB) on startup is less than
  desired (0.8, 76.0 GiB)" while the export process released memory. Captures
  now wait for free memory and run at utilization 0.6 (offline C=1 only).
- 3509367: plain captures done; EAGLE engine failed in model runner v2:
  `AttributeError: 'ApertusModel' object has no attribute 'model'` at
  `vllm/v1/worker/gpu/spec_decode/eagle/utils.py:52` (`load_eagle_model`).
  New overlay `patches/vllm-apertus-eagle-mrv2-inner.patch` (same shape as the
  aux-layers patch), now in `launch/eagle.sh` and training jobs. Same pattern
  also exists at `v1/spec_decode/llm_base_proposer.py:1570` (v1 runner path,
  not exercised here) and in the dflash/dspark loaders (unused).
- 3509606 (COMPLETED 0:0, 12:01): 128 heldout_test prompts (never used for
  training, early stopping or selection) + cap sweep of 16 prompts x caps
  {1,2,3,4,5,9,17} = 240 requests per capture; TP=1, max_num_seqs=1, prefix
  caching off, greedy, natural EOS, max 512 tokens, depth 3.

| comparison | exact / 240 | finish mismatches |
| --- | --- | --- |
| plain r1 vs EAGLE r1 | 240 | 0 |
| plain r2 vs EAGLE r2 | 240 | 0 |
| plain r1 vs plain r2 | 163 | 10 |
| EAGLE r1 vs EAGLE r2 | 163 | 10 |

The r1 captures ran with a cold compile cache (load 153–156 s), the r2 captures
with a warm one (89–96 s). Plain and EAGLE differ between the two compile states
on the same 77 cases and never within one state: EAGLE adds no divergence
beyond the engine's own compile-state nondeterminism. EOS fell at every position
modulo k+1 in the EAGLE captures; the cap sweep hit the output limit at every
draft position, the bonus token and beyond.

Acceptance (depth 3, engine counters, r1 / r2): 37,980 / 75,207 and 38,030 /
74,880 draft tokens accepted (50.5% / 50.8%); per position 0.699, 0.481, 0.334;
committed tokens per round 1 + 37,980/25,069 = 2.515. Offline C=1 output rate
(includes prefill; not an A5 measurement): plain 171 tok/s, EAGLE 317 tok/s.

### A4.4 serving deployments (launch/eagle8b-smoke.sh, debug, sml TP=1/1)

| deployment | ready (s) | model load | chat + 32 HTTP captures | drafts | accepted / draft tokens | per position |
| --- | --- | --- | --- | --- | --- | --- |
| 3509770 | 285 | 18.78 GiB | ok | 3248 | 4962 / 9744 (50.9%) | 0.702, 0.488, 0.337 |
| 3509818 | 297 | 18.78 GiB | ok | 3240 | 4971 / 9720 (51.1%) | 0.708, 0.490, 0.337 |

Both logs show `LlamaForCausalLMEagle3`, "Using Eagle3 auxiliary layers from config:
(2, 16, 29)", draft max length overridden to 32768. Same load size on restart; no
growth. Served acceptance matches the offline engine (50.5–50.8%).

Baseline smoke 3509870: ready 254 s, model load 17.23 GiB (EAGLE deployments
18.78 GiB, so the draft adds 1.55 GiB). Served greedy captures (32 heldout_test
prompts, max 256 tokens, via the OpenAI server):

| pair | exact / 32 |
| --- | --- |
| baseline 3509870 vs EAGLE 3509770 | 32 |
| baseline 3509870 vs EAGLE 3509818 | 22 |
| EAGLE 3509770 vs EAGLE 3509818 | 22 |

Same structure as offline: 3509770 and the baseline started with a cold vLLM
torch.compile cache for their model, 3509818 with a warm one, and outputs differ
only across that boundary. A5/A6 must hold the compile-cache state fixed across
arms and repeats (record it per deployment).

Operational mistakes (mine), recorded so they are not read as system failures:
- A retry loop around a hanging SSH command launched the smoke four times.
  3509784 FAILED with `ModuleNotFoundError: vllm.v1.worker.gpu.spec_decode.eagle.utils`
  because a later launch's `patch-vllm.sh` wiped `~/.sml/vllm-patch` while that
  container was mounting it; 3509837 was cancelled while pending. The driver now
  takes a `flock` and uses a per-deployment `VLLM_PATCH_DIR`.
- The driver found its job by diffing `squeue`, and once picked the concurrently
  submitted training job 3509855; killing that driver ran its exit trap, which
  cancelled 3509855 after 45 s. The driver now reads "Job submitted: N" from sml.

### A4.5 sampled check (temperature 0.8)

First run (job 3509935) is invalid. Seeds 1, 2 and 3 for reference, control and
EAGLE gave identical first-token counts on 22 of 23 prompts with a varied first
token: vLLM derives the n per-sample streams of one request from consecutive
seeds, so the three runs shared almost all random streams. The control TV
(0.0001–0.0007) was therefore not sampling noise, and the EAGLE gap (0.0006–0.013)
only measured the rejection sampler consuming shared streams differently.
Kept as `sampled-v0-correlated-seeds`. Rerun (job 3512071): seeds 1, 10000019,
20000039, n=1024, horizon 4, 32 heldout prompts, batched (max_num_seqs 256);
`dist` now flags correlated streams itself.

Rerun result (3512071, COMPLETED 0:0, 13:29): no correlated streams (0 of 24
varied prompts with identical first-token counts). Mean TV of the first k sampled
tokens from the plain reference:

| k | plain control | EAGLE | prompts EAGLE > control |
| --- | --- | --- | --- |
| 1 | 0.0055 | 0.0070 | 12 / 32 |
| 2 | 0.0121 | 0.0157 | 15 / 32 |
| 3 | 0.0178 | 0.0224 | 17 / 32 |
| 4 | 0.0201 | 0.0257 | 20 / 32 |

No evidence of a distribution mismatch: sign counts are at chance (20/32 at k=4,
one-sided p≈0.11). The mean excess is small but positive and grows with k
(+0.0015 at k=1, where speculation cannot act, to +0.0057 at k=4). Power is
limited (one control pair, 32 prompts); a larger test would be needed to bound
it tighter. Temperature 0.8 is not part of the core A5/A6 cells (greedy).

### A5 stop: EAGLE engine crash on multimodal token ids (2026-09-24)

A5 campaign deployment 2 (`eagle profile p1 3`, sml job 3512315) died on its first
warmup requests: `EngineCore encountered a fatal error ... torch.AcceleratorError:
CUDA error: device-side assert triggered`; the client saw "stream completed
without usage". The campaign runner was stopped after that deployment. The
baseline profile deployment 3512135 had completed (C=1, 256 fixed tokens,
32/32 ok, 169.4 tok/s, TPOT p50 5.76 ms, ready 759 s with a cold cache).

Offline reproduction (job 3512427, `CUDA_LAUNCH_BLOCKING=1`, profile prompts,
greedy, ignore_eos): plain passes (8192 tokens, no generated id >= 131072); EAGLE
fails on the 6th prompt with an Inductor assert `index out of bounds: 0 <= tmp5 <
131072` in the fused draft `embed_tokens` lookup + input RMSNorm kernel. That
prompt contains a literal `<|image|>` in its text, which the tokenizer maps to id
131079; 5 of the 32 profile prompts do (all code prompts). The draft kept its own
131072-row embedding (text rows), so a prompt id outside it overflowed. It was
not `ignore_eos`: the heldout prompts used by A4 happened to contain no such token.

Fix: export the head without `embed_tokens` (`export_head --drop-embed-tokens`,
config `model.share_target_embedding: true`). vLLM's `process_eagle_weight` then
leaves `has_own_embed_tokens` false and `load_eagle_model` shares the target's
266752-row embedding. The draft's frozen rows were bit-identical copies of the
target's text rows, so text-id numerics are unchanged; the head also shrinks by
1 GiB. New head `e31-pilot-3508057-se`; job 3512533 re-runs export, the
ignore_eos reproduction and the A4 greedy verification on it. A4 results for the
first export stand for text-only prompts; the `-se` head is the serving candidate.

Job 3512533 (COMPLETED 0:0, 36:20) on `e31-pilot-3508057-se` (checkpoint manifest
`6fbd3f843ed4327fd72c38ff293192fdcc4e63b17da70e16c24b0789432b123d`, no
`embed_tokens`; reload bit-exact):
- ignore_eos repro: plain and EAGLE both complete 8192 tokens on the 32 profile
  prompts (no crash). Acceptance there is only 1496 / 20016 (7.5%): natural EOS
  falls at 85–175 tokens, and ignore_eos forces generation past
  `<|assistant_end|>`, which the head never saw. The fixed-256 profile is a
  round-cost mechanics check, not an acceptance estimate.
- greedy re-verification (240 requests per capture): plain r2 vs EAGLE r2 240/240
  exact; plain r1 vs plain r2 240/240; plain r1 vs EAGLE r1 146/240 and EAGLE r1
  vs EAGLE r2 146/240. EAGLE r1 was the first load of the new head config (cold
  compile), the others warm. All 94 divergences are near-ties under the plain
  target on the forced prefix: |logprob margin| <= 0.25 (median 0.125, the bf16
  step), 41 below 0.05. Acceptance on heldout, depth 3: 38115 / 74943 (50.9%).
A5 resumed with this head (`configs/eagle/8b/a5-plan-resume.txt`).

### A5 profile (depth 3, C=1, 32 prompts x 256 fixed tokens, ignore_eos)

| deployment | output tok/s | TPOT p50 (ms) | TTFT p50 (ms) | acceptance | g = 1 + acc/draft |
| --- | --- | --- | --- | --- | --- |
| baseline 3512135 | 169.4 | 5.759 (t0) | 40.5 | – | 1 |
| EAGLE k3 (…213132Z) | 157.3 | 6.141 | 47.5 | 7.5% | 1.224 |
| EAGLE k3 (…213805Z) | 158.3 | 6.180 | 41.1 | 7.7% | 1.232 |

Round cost t_round ≈ TPOT x g ≈ 7.52 ms ≈ 1.31 t0; break-even needs g ≥ 1.31.
With ignore_eos, 40–60% of each 256-token output lies past `<|assistant_end|>`
(natural EOS at 85–175 tokens), where the head is off-distribution, so g here
is far below the natural-EOS g of 2.51 (A4). Predicted C=1 TPOT at g = 2.51 with
the same round cost: 7.52 / 2.51 ≈ 3.0 ms (≈1.9x); screening measures it.

Torch trace (PROFILE_TRACE deployment) was lost: the serving environment mounts
`/capstor` and `/iopsstor` but not `/users`, so the trace directory (and the
per-deployment `VLLM_CACHE_ROOT`) existed only inside the container. Traces and
compile caches now go to `/iopsstor/scratch/cscs/$USER/apertus-eagle/8b/`; the
traced profile is re-queued after screening. Consequence for the earlier smoke
reading: serving deployments always compiled from scratch, so the 22/32 serving
divergence between two EAGLE deployments reflects per-compilation kernel
choices (autotuning), not a warm/cold cache. Offline runs that reused one set of
compiled kernels were bitwise repeatable.

First screening attempt: all 60 cells failed with `line 193: invalid JSON`.
`eagle-8b-validation.jsonl` had 196 lines for 192 rows: an earlier `finalize`
whose SSH session I had treated as dead was still running and wrote over the
later run's files. `finalize` now writes to a temp file, re-reads it and renames;
files are ASCII-escaped. The rewritten sets contain no new ids (validation 192,
test 384); the profile set is unchanged. Screening rerun started 23:56 UTC
(`configs/eagle/8b/a5-plan-screen-rerun.txt`).

### A5 screening (rerun, 2026-09-24 23:56 – 2026-09-25 02:28 UTC) — provisional

10 deployments, all `status=ok`, 60 cells, success rate 1.0 everywhere.
Summary `results/eagle/8b/screen-summary.json` (`apertus_eagle.a5_summary`,
speedup = same-block baseline TPOT p50 / EAGLE TPOT p50, geometric mean over
chat/code/summarization):

| depth | all cells | C=1 | C=8 | worst cell | g | acceptance |
| --- | --- | --- | --- | --- | --- | --- |
| 2 | 0.959 | 0.954 | 0.964 | 0.909 | 1.16 | 6–9% |
| 3 | 0.909 | 0.895 | 0.923 | 0.857 | 1.17 | 4–6% |
| 5 | 0.794 | 0.788 | 0.801 | 0.758 | 1.17 | 2.5–3.8% |
| 8 | 0.679 | 0.674 | 0.683 | 0.647 | 1.17 | 1.6–2.3% |

Baseline TPOT p50 is stable between blocks (chat C=1 5.78 / 5.81 ms). Every
depth is slower than plain because g ≈ 1.17 at every depth: only draft position
0 is ever accepted (~17%). This contradicts A4 (depth 3, 50.5–50.9%, g ≈ 2.5,
position 0 ≈ 0.70) measured with the first export (served, smoke) and with the
`-se` export (offline, heldout prompts). The low value also appeared in the
profile (7.5%), which I first attributed only to ignore_eos. Not used for
selection until explained; diagnostic job 3514782 separates head export
(`-se` vs original) from prompt set (A5 validation vs A4 heldout), offline.

### A5 root cause: the pilot corpus was 100% DeepMath

Diagnostic 3514782 (offline, 48 prompts each, depth 3, max 256 tokens):

| head export | prompts | acceptance | g | accepted per position |
| --- | --- | --- | --- | --- |
| original | A4 heldout | 51.9% | 2.556 | 3391, 2382, 1662 |
| `-se` | A4 heldout | 51.9% | 2.556 | 3391, 2382, 1663 |
| original | A5 validation | 6.0% | 1.18 | 1089, 197, 45 |
| `-se` | A5 validation | 6.0% | 1.18 | 1089, 197, 45 |

The export change is irrelevant; the prompt set decides. `results/eagle/data/`
train (1000), validation (128) and heldout_test (256) are all domain `math`,
dataset `deepmath`: `prepare_data.subsample_pilot` hash-assigned rows correctly
but then took the first N sorted by (source, id), and "DeepMath-103K" sorts
first. The head was trained and evaluated on math only. Consequences:
- A4 correctness results stand (losslessness does not depend on head quality).
- Every acceptance number so far (A3 simulated acceptance, A4 50.9%) is on
  math only. The A5 screening measured a math-only head on chat/code/summary;
  its ranking is not used.

Fix: `training/apertus_eagle/build_train_splits.py` samples by hash rank within
each domain with the plan's coverage defaults (chat 40 = chat_qa 20 +
multilingual 10 + instruction_following 10; code 30; summarization 30, finepdfs
EN/DE/FR/IT 40/20/20/20 from files after the ones the workloads read). Train from
SFT-mix rows assigned `train`, training-eval from `validation`. New config
`configs/eagle/8b/train-e31-mix.yaml` (data root `…/apertus-eagle/8b-mix`); same
recipe, 1000 conversations, 2000-step cap.

Mixed corpus (`results/eagle/8b/data-mix/`): train chat_qa 200, multilingual 100,
instruction_following 100, code 300, summarization 300 (EN 120, DE/FR/IT 60);
validation 25/12/12/38/41, no quota shortfall. Overlap check against the A5/A6
workloads found 18 exact duplicates, all OpenCodeReasoning-2 problem statements
repeated under different row ids (8 validation, 9 test, 1 profile workload
prompt). Those training rows were removed: train 987, training-eval 123; recheck
shows 0 overlap with every workload and 0 between train and training-eval.
Jobs: 3514889 (debug, generate + extract) FAILED 1:0 after 3:31: generation done
(validation 123/123 `stop`, 26,169 tokens; train 985/987, 2 prompts too long,
209,930 tokens; 32/32 render checks), then extraction stopped on a training row
with a literal `<|image|>` (131079). TorchSpec clamps draft input ids to the
text rows, so such a row would train on the wrong embedding row; the extractor
now skips and records them. Dependent 3514890 was cancelled by `afterok`.
Resubmitted 3514906 (debug, extract) -> 3514907 (normal, train + export + verify). A first all-in-one normal submission 3514884 was cancelled a
few seconds after it started (QOSGrpNodeLimit had suggested a long wait).

### A3 mixed pilot (3514906 extract, 3514907 train/export/verify)

Extraction (debug, 9:31): validation 104,754 tokens (3.43 GB), train 853,981 tokens
(27.99 GB); 3 + 27 rows skipped for literal multimodal ids; teacher 2.1–2.7k tok/s.
Training (normal, 1:11:28 incl. verify): stopped at step 1900 (3 evals without
improvement), best step 1600, all gates true. Held-out (123-row mixed split):

| step | loss | sim. acc. length | acc position 0..6 |
| --- | --- | --- | --- |
| 0 | 30.10 | 0.00 | 0 |
| 500 | 6.70 | 0.41 | 0.31 … 0.19 |
| 1600 | 6.39 | 0.44 | 0.329, 0.261, 0.238, 0.230, 0.225, 0.222, 0.218 |

Only ~210k supervised (generated) tokens: responses average ~213 tokens with
thinking disabled. Offline depth-3 acceptance on the A5 validation workload
(chat + code rows, 240 requests): 17.0%, g = 1.51, accepted per position 6036,
2438, 1283 (math-only head: 6.0%, g 1.18). Head `e31-mix-3514907-se`.

Greedy check on the same prompts: plain r1 vs r2 240/240; plain r1 vs EAGLE r1
201/240, plain r2 vs EAGLE r2 171/240, EAGLE r1 vs r2 170/240. All 39 r1
divergences have |forced-prefix logprob margin| <= 0.25 (median 0.125; none
> 0.5; EAGLE token always in the target top-20); divergences concentrate in chat
(53 of 69 in r2), median first divergence at token 65–73. Explained as bf16
near-tie flips between multi-token verification and single-token decoding,
which diverse chat text hits more often than the math prompts did.

Decision: the pipeline works and the 1k corpus is the limit, so run the plan's
one expansion to 10,000 conversations (`configs/eagle/8b/train-e31-mix10k.yaml`,
same mix and recipe, <= 3 epochs, patience 3). The mixed pilot head is screened
meanwhile (`configs/eagle/8b/a5-plan-screen-mix.txt`, blocks b3/b4).

### A5 screen of the 1k mixed head (blocks b3/b4, 2026-09-25 04:18–06:08 UTC)

10/10 deployments ok, success 1.0. `results/eagle/8b/screen-mix1k/screen-summary.json`.

| depth | all | C=1 | C=8 | worst | g |
| --- | --- | --- | --- | --- | --- |
| 2 | 1.201 | 1.212 | 1.190 | 1.033 | 1.48 |
| 3 | 1.162 | 1.163 | 1.162 | 0.968 | 1.54 |
| 5 | 1.047 | 1.052 | 1.041 | 0.852 | 1.59 |
| 8 | 0.907 | 0.914 | 0.901 | 0.716 | 1.62 |

Per stratum at depth 2 (b3 / b4): code 1.61 / 1.58 (C=1), 1.53 / 1.55 (C=8),
acceptance 45%, E2E p50 0.61–0.67x; chat 1.03–1.05 (12.5%); summarization
1.04–1.08 (15%). Blocks agree within ~0.04 on every cell. Superseded by the 10k
head before confirmation.

### A3 10k expansion (3515342 generate+extract, 3515343 train/export/verify)

Corpus `results/eagle/8b/data-mix10k/`: 9909 train (96 workload duplicates
removed, 90 of them code), 116 training-eval (7 train duplicates removed).
Extraction 36:26 on normal (9532 train rows cached, 287 GB, teacher up to 5.3k
tok/s). Training: 3 epochs, 1785 steps, 25.3M tokens, 51 min; stop `max_epochs`;
best step 1700; all gates true; still improving at the end.

| step | held-out loss | sim. acc. length | acc position 0..6 |
| --- | --- | --- | --- |
| 200 | 7.29 | 0.28 | 0.23 … 0.15 |
| 800 | 4.65 | 0.53 | 0.38 … 0.24 |
| 1600 | 4.06 | 0.665 | 0.438, 0.352, 0.326, 0.313, 0.305, 0.299, 0.294 |

Offline depth 3 on the A5 validation workload: acceptance 25.6%, g 1.768
(1k mixed head 17.0%, 1.51). Greedy: plain vs EAGLE 239/240 in both compile
states, EAGLE r1 vs r2 240/240; the single divergence has margin 0.25. Head
`e31-mix10k-3515343-se`, manifest `0fa2f62346e05916852fea88de36178cd04781f2c7abf225a907a93e5a4d8e18`,
reload bit-exact. Screening started with blocks b5/b6 (depth 8 omitted as
dominated in the 1k-mix screen).

### A5 screen of the 10k head (blocks b5/b6) and the Clariden outage

Campaign `configs/eagle/8b/a5-plan-screen-mix10k.txt` started 2026-09-25 07:11 UTC
on login node ln004. All six EAGLE deployments (k2/k3/k5 in b5 and b6) and the b5
baseline finished `status=ok`. The last deployment (b6 baseline, sml job 3516465)
started 08:28; ln004 then stopped accepting sessions (authentication succeeds,
the session never opens; ln003 the same, ln002 rejects the key) and the driver
died with it. 3516465 held its node until TIMEOUT (1:30) without a cell. Logins
were impossible from ~11:30 UTC on 2026-09-25 until 2026-09-26 12:40 UTC, when
ln001 answered (alias `clariden1`). The dead ln004 driver still holds the
default `flock`, so `eagle8b-measure.sh` now honours `STAGEA_LOCK`. The b6
baseline is re-run on 2026-09-26 (sml job 3526582), a day after the rest of b6.

Block b5 alone (same-block baseline, geometric mean over chat/code/summary):

| depth | all | C=1 | C=8 | worst | g |
| --- | --- | --- | --- | --- | --- |
| 2 | 1.383 | 1.419 | 1.347 | 1.153 | 1.72 |
| 3 | 1.369 | 1.392 | 1.346 | 1.088 | 1.84 |
| 5 | 1.274 | 1.302 | 1.248 | 0.954 | 1.96 |

Depth 2 by stratum (C=1 / C=8): code 1.90 / 1.82 (acceptance 61%, E2E p50
0.54 / 0.60x), summarization 1.26 / 1.16, chat 1.19 / 1.15. Depth 3: code 2.04 /
1.95 (50%), summarization 1.19 / 1.15, chat 1.11 / 1.09.

Full screen (b6 baseline re-run 2026-09-26 12:46–12:55, sml 3526582, ok):
`results/eagle/8b/screen-summary-mix10k.json`.

| depth | all (b5 / b6) | C=1 | C=8 | worst | g |
| --- | --- | --- | --- | --- | --- |
| 2 | 1.373 (1.383 / 1.363) | 1.410 | 1.336 | 1.140 | 1.72 |
| 3 | 1.366 (1.369 / 1.363) | 1.393 | 1.340 | 1.088 | 1.84 |
| 5 | 1.260 (1.274 / 1.246) | 1.289 | 1.233 | 0.951 | 1.96 |

Blocks agree within 0.03 per depth despite the day-later b6 baseline. Selected
for A6 (plan rule: best, plus any within 5%): depths 2 and 3. Confirmation plan
`configs/eagle/8b/a6-plan-confirm.txt` (3 blocks c1–c3, baseline + k2 + k3 each,
rotated order, untouched test strata, 128 requests per cell, C=1/8), campaign
started 2026-09-26 12:57 UTC from ln001 (`STAGEA_LOCK` override).

Colleague data: `/capstor/scratch/cscs/zyu` is `drwxr-x--- 30628:sm94`; reading
`spec/` from this account gives `Permission denied`.

### A5 inputs

Workloads (`training/apertus_eagle/build_workloads.py`, manifest
`results/eagle/8b/workloads/manifest.json`; JSONL stays on the cluster, it holds
source text): chat and code from SFT-mix rows hash-assigned to `heldout_test`
(excluding the 256 A4 rows and every training/validation row); summarization
from finepdfs-edu eng/deu/fra/ita with a same-language instruction. Exact Apertus
token counts with the served template.

| stratum | validation | test | input tokens (val / test) | output cap |
| --- | --- | --- | --- | --- |
| chat | 64 | 128 | 130–1657 / 128–1922 | 256 |
| code | 64 | 128 | 168–1001 / 221–1202 | 512 |
| summarization | 64 | 128 | 2058–3807 / 2048–4085 | 512 |

Profile set: 32 chat/code prompts with 512–1024 input tokens outside both splits.
Summarization languages EN/DE/FR/IT: validation 28/12/12/12, test 53/25/25/25
(floor quotas of 40/20/20/20). Limits: code prompts reach only ~1.2k tokens, so
the 4096 upper band is unused; chat language is not balanced (SFT-mix domain
only). Build fixes on the way: English finepdfs path (`eng_Latn/train`),
English token window (Apertus is more compact than the source count),
`splitlines()` breaking on U+2028 inside JSON strings.

Scheduling: the EAGLE and baseline smoke engines both log "Asynchronous
scheduling is enabled" and `max_num_batched_tokens=8192`, so speculation does
not change scheduler settings here; no matched baseline is needed. KV capacity
differs (466,144 tokens baseline vs 439,072 with the draft).

Compile cache: every A5/A6 deployment gets a fresh `VLLM_CACHE_ROOT` (injected
through `resolve-env.sh` `EXTRA_ENV`), so all arms and repeats start cold.

### vLLM overlay patch fix

`patches/vllm-apertus-eagle3-aux-layers.patch` was malformed ("malformed patch
at line 40": wrong hunk counts), so `launch/eagle.sh` could never have applied it.
Regenerated with `diff -u` against the image's `interfaces.py`, same change, and
dry-run against the image copies together with the image-token patch (fuzz 1).
Training jobs now mount both overlays from `~/.sml/vllm-patch-train`.

### Infrastructure incident, 2026-09-24

Clariden logins hung again from ~10:26 UTC (intermittent recoveries at 10:44 and
10:53). At the same time teacher extraction in job 3505554 fell from 2,495 to
377 tokens/s after the 100th validation sample, and `ls` on the iopsstor
feature directory hung for more than 7 minutes (10:57). This is shared
filesystem slowness, not a pipeline cost; A3 cost estimates use the first-100
figures (2.5k teacher tokens/s, 32.8 KB/token, 19–22 GB peak GPU).
`apertus_eagle.features` now checkpoints a partial manifest every 50 samples and
resumes matching samples after a time-out.

### A0

`python3 -m apertus_eagle.build_contract --stage 8b --hash-weights` on the login node
(57 s). Target `/capstor/.../swiss-ai/Apertus-v1.5-8B`, revision
`a411d838600baf0e3635a3daf66fb7c55fc97bb6`, weight manifest
`0a6c3ca1c1e8cf220c21971b3b96b0e8799827f7bb00ac056dfb3657a533b08a` (sha256 of every
safetensors file plus config), config `391963f8…`.

- text hidden 4096, 32 layers, 32/8 heads, head_dim 128, xIELU MLP 21504, qk_norm
- input vocab 266752, output vocab 131072; `lm_head.weight` [131072, 4096] BF16;
  `model.language_model.embed_tokens.weight` [266752, 4096]; final norm
  `model.language_model.norm.weight`
- llama3 RoPE theta 4e6, factor 32, original 8192; eps 1e-5; BF16 throughout
- tokenizer and chat template hashes are identical to the 70B record
  (`1f2f6198…`, `d23af285…`); BOS 1 owned by the template; EOS {2, 68, 72}
- `<|inner_prefix|>`/`<|inner_suffix|>` are ids 32/33. The 70B record's
  `think_start_id=69`/`think_end_id=70` is wrong: ids 69 and 70 are not added
  tokens. `renderer.THINK_*` constants are unused by the loss mask.
- aux layers: vLLM `(2, 16, 29)`, HF/TorchSpec decoder layers `[1, 15, 28]`.
  Verified in the image's vLLM source (unsquashed to
  `/iopsstor/scratch/cscs/faruk_zahiragic/apertus-eagle/vllm-src`):
  `interfaces.py` default `(2, L//2, L-3)`; `apertus.py` records index 0 before
  the loop and `idx + 1` after layer `idx`; `llama_eagle3.py` supports `fc_norm`
  and `norm_output` and maps `midlayer.` to `layers.0.`.
- internal inventory: no Apertus 1.5 8B head
  (`results/eagle/8b/preflight/internal-head-search.json`).
- draft template `configs/eagle/8b/draft-e31-config.json` (generated by
  `apertus_eagle.draft_config`): 1 layer, hidden 4096, SwiGLU 14336, vocab 131072.
  Parameters: 293,629,952 excluding embed/lm_head; 1,367,371,776 total.

GPU memory: 8B BF16 weights ~16.4 GB of 97.9 GB per GH200, so TP=1 serving and
sequential extract-then-train on one GPU are expected to fit; measured peaks
come from the job logs. Allocation: every job holds an exclusive 4-GPU node while
the pipeline uses GPU 0. Record GPU-hours as 4 x wall time.

### A1

- `training/apertus_eagle/contract.py`: no default contract (explicit path or
  `APERTUS_EAGLE_CONTRACT`); aux ids from the contract, cross-checked against
  depth; `target_identity()`. `target_adapter.py`, `parity.py`,
  `overfit_direct.py` take the contract; no fixed 80-layer assertion.
- `src/apertus_bench/eagle.py`: no default contract; head `apertus_target`
  provenance must match the contract's revision, config, tokenizer and template
  hashes; the 2509 head is refused by path or config text; config-only
  directories are never `trained_head`; `checkpoint_manifest_sha256` covers
  config plus weights; `--target-model` refuses a target that is not the
  contract checkpoint.
- `launch/stage-defaults.sh` (new), `launch/eagle.sh`, `launch/baseline.sh`:
  `STAGE=8b` gives target TP=1, draft TP=1, 32768 context, prefix caching off;
  `STAGE=70b` gives 4/4 and 131072. Draft TP must equal target TP. `baseline.sh`
  without `STAGE` keeps the historical 70B launch.
- `launch/submit-eagle-train.sh`: requires `EAGLE_TRAIN_CONFIG`, prints resolved
  settings, records allocated versus used GPUs. `launch/train-eagle-on-node.sh`
  runs `apertus_eagle.pipeline`. `launch/train-eagle.sh` (upstream Ray route) takes
  teacher/trainer GPU counts and aux ids from the contract.
  `launch/train-eagle-job.sh` is superseded and refuses to run.
- `src/apertus_bench/analysis.py`: baseline key now includes target model,
  revision, tokenizer, target TP and precision; draft TP must agree across a
  candidate's repeats. `cli.py`: EAGLE cells also need `target_model`,
  `target_revision`, `target_tensor_parallel_size`.
- Tests: `tests/test_eagle.py` (rewritten), `tests/test_launchers.py` (new), new
  cases in `tests/test_analysis.py`, `tests/test_apertus_eagle_training.py`,
  `tests/test_cli.py`. `pytest -q`: 84 passed.

### A2/A3 implementation

The serving image has no Ray or Mooncake, and TorchSpec's offline replay still
starts Ray actors and a Mooncake store. The pipeline therefore keeps TorchSpec's
training objective and replaces only its data plane:

- `apertus_eagle.generate_targets`: greedy, natural-EOS 8B responses via vLLM
  offline (TP=1), serving default `Deliberation: disabled`, developer spans
  dropped, max 2048 new tokens, sequences at most 4096. Training ids are the
  served stream `prompt_ids + generated_ids` (no detokenize/re-encode). First
  32 prompts (half multi-turn) must match `LLM.chat` prompt ids exactly.
- `apertus_eagle.features`: HF teacher on one GPU; decoder outputs at [1,15,28]
  plus post-norm final hidden, one safetensors per conversation; manifest names
  target identity, feature contract and corpus digest, and refuses reuse on any
  change. Bytes/token estimate after 100 samples; `--max-cache-gb` cap.
  Parity samples: reconstructed `lm_head(norm)` versus model logits, and a
  repeated forward as the noise control.
- `apertus_eagle.parity_vllm`: vLLM eager, hooks on every decoder layer; HF layer
  i compared with vLLM index i+1 and the neighbours i, i+2; argmax agreement via
  `prompt_logprobs`. The pipeline blocks training unless the offset is confirmed.
- `apertus_eagle.train_rollout`: `AutoEagle3DraftModel` + `Eagle3Model`
  (ttt_length 7, forward-KL, `compute_lazy_target_padded`, `0.8**i` weights) +
  `BF16Optimizer` (cosine, warmup 0.015, clip 0.5, wd 0), batches in
  `Eagle3Trainer._forward` layout. Accumulation 16 x micro-batch 1 x 1 rank =
  16 sequences per step. Embedding = first 131072 target rows, frozen; lm_head
  seeded from the target and frozen (TorchSpec `freeze_lm_head` path). Fails on
  non-finite loss, repeated non-finite grad norm or a no-op first update.
- `apertus_eagle.export_head`: TorchSpec `to_export_keys`, config with
  `apertus_target` and `apertus_training` provenance, then a fresh reload must
  reproduce rollout metrics and first-step logits exactly.
- Local CPU smoke on a synthetic tiny target (scratchpad only): 40 steps, loss
  12.2 → 1.9, held-out simulated acceptance length 0.01 → 0.58, export reload
  bit-exact. This checks mechanics only; it is not an Apertus result.

Updated 2026-09-23 11:57 UTC. This file records completed task IDs, commands,
artifacts, failures, and the next action. It does not claim GPU measurements
that were not taken. CSCS cert `/home/fzahiragic/.ssh/cscs-key-cert.pub` is
valid from 2026-09-23T09:48:55Z to **2026-09-24T09:49:55Z**. Use
`ssh -o AddressFamily=inet -o BatchMode=yes clariden` (`faruk_zahiragic` on
`clariden-ln003`). IPv6 to `ela.cscs.ch` can burn the connect timeout.
Tiny-overfit **3492049** is RUNNING on nid006905. No head files yet.

## Status

| Task | Status | Next action |
| --- | --- | --- |
| T0 pin and inspect | Contract recorded. GPU probe blocked on a dedicated tiny EAGLE launch (no head). | After T4 export, probe TP=4 feature shapes with the trained head. |
| T1 harness and launchers | Implemented; local tests cover baseline pairing and provenance. | Keep `--repeats 1` per independently launched deployment. |
| T2 round-cost diagnosis | All 24 screening cells + P2 summaries are on disk. Campaign driver ended 2026-09-19T22:10Z. P1 skipped by a harness bug (fixed in the campaign script). | P1 profile after the debug partition is free. Do not submit a second train job. |
| T3 data and parity | Splits done. Renderer/BOS probe **pass** on 32 overfit rows. TorchSpec `6c042a8` patched. | Numeric hidden-state parity still needs the T4 GPU job. |
| T4 train/export | No head yet. **3492049** RUNNING on nid006905 since 11:41 UTC. Draft import passed (`draft_import_ok`). 70B weights 924/1596 (~58%) at 12:00 UTC. | Leave 3492049 running. Do not submit another 70B train. Read `overfit-summary.json` when it ends. |
| T5–T8 | Not started. | After a reloadable 70B head. |

## T0

Recorded from the authorized Capstor checkpoint
`/capstor/store/cscs/swissai/infra01/hf_models/models/swiss-ai/Apertus-v1.5-70B`
and the pinned vLLM sources. Artifacts:

- `configs/eagle/lock.json`
- `results/eagle/preflight/compatibility.json`
- `results/eagle/preflight/target-config.json`

Exact values that matter for training:

- text hidden size 8192, 80 layers, 64/8 heads, head_dim 128 (derived)
- input vocab 266752, output vocab 131072, `lm_head.weight` shape `[131072, 8192]` BF16
- embeddings `model.language_model.embed_tokens.weight` `[266752, 8192]`
- final RMSNorm `model.language_model.norm.weight` `[8192]`
- tokenizer.json sha256 `1f2f6198ea5789e5a90ec7c5ec5cf0d5242cf5b8de007105537c91d777581582`
- chat template sha256 `d23af2850029e9df88bb017e06e0c14a0462604588f1dbe4f519cdfac3e260e0`
- vLLM default aux layers `(2, 40, 77)` with residual included, final norm not included
- TorchSpec `inference.aux_hidden_states_layers` must be `[1, 39, 76]`; the vLLM engine adds +1
- serving BOS is owned by the chat template (`add_special_tokens=False`)

Draft architecture template parameter count (not a trained checkpoint):

- E3.1 total 3,288,391,680 (~3.29B) including embed + lm_head at output vocab 131072
- unique draft compute excluding embed + lm_head: 1,140,908,032 (~1.14B)
- This is not a 200M head. Sharing embed/lm_head with the target still leaves ~1.14B unique parameters.

No Apertus 1.5 70B EAGLE head was found on Capstor. The public
`thomaskiefer/EAGLE3-Apertus-8B-Instruct-2509` checkpoint is refused by the
validator. `launch/eagle.sh` applies both source overlays.

TorchSpec `6c042a87140a84d13839e341ece2c5c3ada918bc` is cloned under gitignored
`scratch/TorchSpec`. Patches applied on Clariden:

- `patches/torchspec-apertus-target-lm-head.patch` (allocate `output_vocab_size`)
- `patches/torchspec-apertus-hf-target-layers.patch` (`language_model` walk)
- AutoModelForImageTextToText / AutoModel fallback (do not relabel as Llama)
- `torchspec.data.renderers.apertus` registered as `apertus`

Clariden `$HOME/model-launch` is `05ee56f`, not the experiment pin `909026a`.
Launchers must set `MODEL_LAUNCH_ROOT` to a 909026a checkout (`$HOME/model-launch-apertus` on cluster).

GPU probe of EAGLE feature shapes: blocked until a 70B head exists. Debug
replica **3491679** (D3, nid007320) was running at the start of this check
and was not scancelled. By 11:36 UTC it had left the queue: sacct shows
`CANCELLED by 1186` at 13:15:47 cluster time, and step `.1` is
`OUT_OF_MEMORY`. That is not an EAGLE probe.

## T1

Harness distinguishes engine `method=eagle3` from algorithm labels
`eagle3` / `eagle31` / `peagle`. `add_speedups()` refuses to pool operational
and matched baselines, different images, corpora, or `ignore_eos` settings.
EAGLE cells require `--metadata deployment_id=` and `checkpoint_sha256=`.
`VALIDATE_ONLY=1` on `launch/eagle.sh` allows config-only architecture templates.

Prometheus counter names remain those verified on the pinned image. Missing
counters stay `enabled=false`, not zero acceptance.

## T2

Config phase **completed** 2026-09-19T22:10:51Z. All 24 screening cells plus
P2 summaries are on disk. Those Slurm jobs are finished. As of 11:57 UTC
23 Sep, train **3492049** is RUNNING on nid006905 and no debug replica is
in the queue.

B0: 69.15 / 68.93 tok/s C=1 (jobs 3441989, 3446578), `t0` 14.19–14.22 ms,
KV ~513,700, async on.

B1: 63.73 / 63.74 tok/s C=1 (0.92× B0). Async-off costs ~8%.

B2: 63.83 / 63.71 tok/s C=1 (jobs 3446471, 3446749). **B2 ≈ B1**; the 7168
budget does not bind at this shape. Pooled B2 `t0 = 15.821 ms`.

D3: blk1 C=1 23.53 tok/s, `t_round` 103.5 ms, vs B2 **0.369×**; blk2 C=1
19.72 tok/s, `t_round` 126.5 ms, vs B2 **0.309×**. Acceptance 49–51%,
`g ≈ 2.47–2.53`. Break-even vs B2 is ~39–40 ms. Perfect-acceptance ceiling
0.46–0.62×. KV 251,488.

N3 / N3m: C=1 **1.09–1.13× B0** and **1.18–1.23× B2** at `t_round` 17–19 ms
despite ~21% acceptance. N3 C=8 is noisier (0.83–0.93× B0).

P2 job 3448470 summaries: C=1 **242.25 tok/s, TPOT p50 4.00 ms**, C=8
1723.07 tok/s / 4.27 ms, 32/32 success, 256 completion tokens. Slurm state
CANCELLED after 00:05:51. The first stored request `output` string was
empty, so this turn does not treat 4.00 ms as a confirmed 8B floor. P1
still has to profile the embedded draft path.

P1 was skipped because `run_config` treated existing screening summaries as
done. Fixed in `launch/run-diagnostics-campaign.sh` (profile skip looks for
`profile-c*/summary.json`). Next: `PHASES=profile DATE_TAG=20260919`.

Canonical artifacts:
`/users/faruk_zahiragic/apertus-bench-repo/results/diagnostics-20260919/`

Do not add an EAGLE arm until T4 exports a head. Do not start the 50k
alignment run until P1–P3 attribute the 103 ms round. Tiny-overfit
**3492049** is the one T4 retry already submitted; it is not that run.

## T3

Prompt source: `swiss-ai/Apertus-1.5-SFT-mix` revision
`d38f5b98b1da36405f6d55b939f096f154883103`, materialized at
`/capstor/store/cscs/swissai/infra01/datasets/Apertus-1.5-SFT-mix-pretrain-v1`
(3,647,256 accepted rows). Parquet columns include rendered `text`, `domain`,
and `dataset_source`. Use prompts only; regenerate assistant text with the
frozen 70B target.

```bash
bash launch/prepare-eagle-splits.sh
bash launch/probe-renderer.sh
```

Splits on Clariden (`results/eagle/data/`): overfit 32, validation 128, train
1000, heldout 256. `teacher_generation: not_run` (mix `original_assistant` is
teacher-forced until the frozen 70B regenerates). TorchSpec JSONL:
`results/eagle/data/torchspec/`.

Renderer/BOS probe **pass** (2026-09-19 20:10 UTC) on the 32 overfit rows.
Capstor tokenizer via `apertus-bench` conda (transformers 5.3 TokenizersBackend).
Artifact: `results/eagle/preflight/renderer-parity.json`. Mix sidecar:
`results/eagle/preflight/mix-text-by-id.json` (32/32 parquet ids).

- Serving renderer always starts with BOS id 1; no double BOS. `training_ok=true`.
- Mix parquet `text` never starts with `<s>` (`stored_always_bos=false`).
- Mix is **not** exactly `rendered[1:]`: the chat template injects the default
  system span (token 61 `<|system_start|>`) plus developer deliberation. First
  row rendered 2301 tokens vs mix 2255. Recorded; does not block training
  because the training renderer is the serving chat template
  (`tokenize=False` then `encode(..., add_special_tokens=False)`).
- Login python has pyarrow (not transformers). Probe looks up mix text with
  login python, then tokenizes with conda.

`apply_chat_template(tokenize=True)` on this TokenizersBackend returns a
BatchEncoding (`list(...)` is `['input_ids', 'attention_mask']`, which looked
like a 2-token render). That is why the renderer now encodes the template
string instead of trusting `tokenize=True`.

Known loader traps remain in `training/apertus_eagle/target_adapter.py`. Numeric
feature/logit parity still needs the allocated T4 GPU job.

## T4

No Apertus 1.5 70B EAGLE checkpoint exists yet. Do not use
`thomaskiefer/EAGLE3-Apertus-8B-Instruct-2509`.

Submitted a **separate** tiny-overfit E3.1 job on `partition=normal` so it
cannot steal debug-qos. Attempts:

| Job | Result |
| --- | --- |
| 3446482 | FAILED 14s: pyxis chdir `/users/...` missing (`/users` not mounted) |
| 3446503 | FAILED 14s: same pyxis chdir |
| 3446504 / 3446514 | FAILED: container venv blocks `pip --user`; `device_map=auto` needs accelerate |
| 3446526 | FAILED ~16 min on nid007418 after **70B weights loaded**. `LlamaForCausalLMEagle3` import pulled TorchSpec `wandb` → missing `platformdirs` |
| **3446638** | FAILED exit 1:0, 00:16:47, nid007611, `normal`, start 19 Sep 22:54, end 19 Sep 23:11 (cluster clock). Weights 1596/1596. Wandb stayed `0.0.0-stub`. `build_draft` then died: `LlamaForCausalLMEagle3` → Mooncake `utils.py` → `ModuleNotFoundError: No module named 'ray'`. Head dir empty. Exact sacct: `3446638\|FAILED\|1:0\|00:16:47\|nid007611\|19 Sep 22:54\|19 Sep 23:11` |
| **3491232** | FAILED exit 1:0, 00:00:53, nid006256, `normal`, 2026-09-23 10:38:46Z–10:39:24Z. Wandb file stub was on `PYTHONPATH` (`0.0.0-stub`). `overfit_direct` then replaced `sys.modules['wandb']` with a module whose `__spec__` is None. Accelerate `find_spec('wandb')` raised `ValueError: wandb.__spec__ is None` during the draft-class import, before the 70B load. Head dir still empty. |
| **3491998** | Submitted 13:29 cluster time, then scancelled while still PENDING. Pyxis `--environment` sbatch stored a script that failed `bash -n` (`:%M:%SZ)` appended after a strftime `%` in the body). Not a training result. |
| **3492049** | RUNNING on nid006905, `normal`, 6h, started 2026-09-23T11:41:32Z. `overfit-e31`, `EAGLE_FORCE_OVERFIT_DIRECT=1`, `--exclude=nid007129`. Stdout `results/eagle/logs/3492049.out` printed `{"preflight": "draft_import_ok", "cls": "LlamaForCausalLMEagle3"}` before the teacher load. Wandb stayed `0.0.0-stub`. Accelerate 1.15.0 is in `scratch/pydeps`. At 12:00 UTC weights were 924/1596 (~58%) and the head directory was still empty. Stored script ends with `date -u -Iseconds` (no broken `%M` fragment). Do not submit a second train. |

| Field | Value |
| --- | --- |
| Last submitted train job | `3492049` RUNNING nid006905 since 11:41 UTC |
| Last confirmed failure | `3491232` in 53s, `ValueError: wandb.__spec__ is None` (3446638 was the earlier missing-`ray` failure after the 70B load; 3446526 was wandb/platformdirs) |
| Ready launcher | `launch/submit-eagle-train.sh` (regenerates sbatch on the cluster) |
| Stale local sbatch | Workspace `results/eagle/logs/submit-overfit-e31.sbatch` still has `pip --user`, no `EAGLE_PYDEPS` / wandb stub / `EAGLE_FORCE_OVERFIT_DIRECT`. **Do not sbatch that file.** |
| Stage | `overfit-e31` (32 conversations; must actually overfit) |
| Config | `configs/eagle/train-e31-overfit.yaml` |
| Expected head | `/users/faruk_zahiragic/apertus-bench-repo/results/eagle/heads/e31-overfit/{config.json,model.safetensors,overfit-summary.json}` |

Do not move training to debug. At 11:57 UTC 23 Sep the only job in
`squeue -u faruk_zahiragic` was **3492049** RUNNING on nid006905.
Do not submit a second train job while 3492049 is running. Do not scancel it.

Cluster `results/eagle/heads/e31-overfit/` existed as a directory at that
snapshot with no `config.json` / `model.safetensors`. E3 control is a later
from-scratch job (`EAGLE_TRAIN_STAGE=overfit-e3`); do not flip `fc_norm` /
`norm_output`. Do not start 50k until 100 samples + 100 warm steps give cost.

Cluster access from this workspace is `ssh -o BatchMode=yes clariden`. The
cert in `/home/fzahiragic/.ssh/cscs-key-cert.pub` is valid until
2026-09-24T11:49:55 CEST. Next check:

```bash
ssh -o BatchMode=yes clariden \
  'squeue -u faruk_zahiragic; sacct -j 3492049 --format=JobID,State,ExitCode,Elapsed,NodeList -n; ls -la /users/faruk_zahiragic/apertus-bench-repo/results/eagle/heads/e31-overfit'
```

3492049 is that one resubmit. Do not submit another until it finishes and
the head is checked. `launch/train-eagle-on-node.sh` still copies
`wandb_offline_stub.py` onto `$EAGLE_PYDEPS/wandb/__init__.py` and prepends
`$EAGLE_PYDEPS` to `PYTHONPATH`. `import_stubs.ensure_import_stubs()` does
not replace that package when `__spec__` is already set, and synthetic
ray/datasets/pydantic/numba modules also get a `ModuleSpec`. Mooncake utils
stays lazy. The 3492049 script does not `pip install ray`.

## Resource note

70B teacher + four-GH200 node. At 11:57 UTC 23 Sep, **3492049** was
RUNNING on `normal` (nid006905, 6h) and had passed the draft-class import
that killed 3446638 and 3491232. P1 profiles
use debug, one replica at a time. T4 stays on `normal`. Do not put another
T4 on debug.

## Credentials still needed

- CSCS SSH cert is valid until **2026-09-24T11:49:55** CEST. Plain
  `ssh -o BatchMode=yes clariden` succeeded on 2026-09-23.
- Hugging Face token only if downloading gated copies; Capstor already has the
  target weights.
- No extra API token is required to *launch* via `sml` on Clariden. The
  on-cluster load generator talks to the replica IP as in the 13 September
  smoke run. `CSCS_SERVING_API` is only needed if the client is pointed at the
  gateway.
- Confirm the GPU-hour budget before T4 expansion. Tiny-overfit jobs on
  `normal` are the 70B head path; they must not use the debug partition.
