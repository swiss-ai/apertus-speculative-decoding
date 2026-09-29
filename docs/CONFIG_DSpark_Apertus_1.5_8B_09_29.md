# DSpark drafter for Apertus 1.5 8B, thinking off, Open-PerfectBlend: training configuration (run in progress, 2026-09-29)

Reference sheet for the full-corpus run started 2026-09-28 22:00 on Daint. The earlier 100k-sample Magpie run is documented in `CONFIG_DSpark_Apertus_1.5_8B.md`; the narrative and measurements are in the results log of `RUNBOOK_09_23_Train_Apertus_SD.md`. Scripts: `results/apertus15_8b_opb_2026-09-28/` in this repo, `/capstor/scratch/cscs/zyu/spec/jobs/` on Daint (`$SPEC` below).

## 1. Sequence lengths (three different numbers, do not mix them up)

| Length | Value | Where it applies |
|---|---|---|
| Answer cap at regeneration | 4,096 generated tokens per assistant turn | `regenerate-responses --max-tokens 4096`; 0.3 % of answers hit it |
| Maximum conversation length | **8,192 tokens** (prompt + all turns) | `prepare-data --seq-length 8192`; longer conversations are truncated. This is the number that corresponds to "sequence length" in the DSpark / DBLAST papers (DeepSpec configs: 4,096; DBLAST: 2,048) |
| Packed training sequence | 16,384 tokens per GPU per step | `--total-seq-len 16384`; several conversations packed into one sequence; 4 ranks → 65,536 tokens per optimizer step |

Supervision per step is set by `--max-anchors 1024`: 1,024 randomly sampled anchor positions per packed sequence, each predicting a block of 8 tokens, independent of the packed length.

## 2. Data

| Item | Value |
|---|---|
| Prompt source | `mlabonne/open-perfectblend`, Hugging Face revision `af60f3c18201652a83a93f46fcfee1b646ba3df7`, 1,420,909 rows; copy for the project at `/capstor/store/cscs/userlab/sm94/datasets/mlabonne__open-perfectblend/` |
| Filtering | 62,128 rows dropped: 62,124 single-message rows (all from the UltraFeedback slice, prompt without any turn structure) and 4 rows with an empty first message → 1,358,781 conversations kept; 15 % multi-turn |
| Order | one shuffle with seed 0 (`jobs/make_opb_shards.py`), four shards of ≈ 340k conversations; nested subsets (10 / 20 / 50 %) are prefixes of this order |
| Regeneration | Apertus-v1.5-8B answers every assistant turn given the preceding turns; thinking off; T 0.7 / top-p 0.8 / top-k 20; 4-way data-parallel server per shard; 1,809,684 generation rows, 933.7 M completion tokens, median 426 / mean 515 / p90 ≈ 980 tokens per answer; ≈ 0.05 % request errors |
| Rendered prefix | Apertus system prompt + developer block "Deliberation: disabled, Tool Capabilities: disabled" on every sample |
| Prepared dataset | `$SPEC/runs/apertus_opb/data/`, 58 arrow shards, 36.5 GB, packed at 8,192; `token_freq.pt` present but the full head is used |
| Split | trainer default `train_data_ratio 0.9`: ≈ 1.22 M conversations train, ≈ 136k held out for per-epoch validation and the confidence-head calibration. Decision 2026-09-29: keep the same 90/10 split (same seed) for the thinking-on run and the Eagle3 comparison so the held-out numbers are comparable |
| Epoch size | 24,940 optimizer steps × 65,536 tokens ≈ 1.63 B tokens (prompts and multi-turn prefixes included) |
| Shared copy of the corpus | `/capstor/store/cscs/userlab/sm94/datasets/apertus15_8b_regenerated/open-perfectblend_apertus15-8b_thinking-off_seed0.jsonl` (+ prompt manifest), for the Eagle3 comparison: same prompts, same Apertus answers, same split seed |

## 3. Drafter and trainer

Exact command (from the checkpoint directory's `train_command.txt`), one trainer rank per GPU, two ranks per node:

```
torchrun --nnodes 2 --nproc_per_node 2 --rdzv_backend c10d ... --no-python bash train_rank_wrapper.sh \
  --verifier-name-or-path <Apertus-v1.5-8B snapshot> --data-path $SPEC/runs/apertus_opb/data \
  --hidden-states-backend file --hidden-states-path /tmp/hidden_states --on-missing generate --on-generate delete \
  --target-layer-ids 1 8 15 22 29 --speculator-type dspark --num-layers 5 --block-size 8 \
  --markov-rank 256 --enable-confidence-head --confidence-head-with-markov --loss-fn '{"ce":0.1,"tv":0.9}' \
  --loss-implementation eager --optimizer muon --lr 3e-4 --epochs 10 --total-seq-len 16384 --max-anchors 1024 \
  --num-workers 4 --prefetch-factor 2 --checkpoint-freq 0.25 --save-path $SPEC/runs/apertus_opb/ckpt_dspark_nothink \
  --vllm-endpoint http://localhost:$((8001 + LOCAL_RANK))/v1        # added per rank by the wrapper
```

| Setting | Value | Note |
|---|---|---|
| Target | Apertus-v1.5-8B, frozen, bf16; the drafter reuses its embedding (266,752 rows) and its pruned output head (131,072 rows) frozen | speculators pruned-head patch required |
| Draft network | 5 Qwen3-style decoder layers (hidden 4096, 32 heads / 8 KV heads, MLP 21,504), block 8 → up to 7 draft tokens, `sample_from_anchor = true` | serving needs the fork's anchor-layout fix (`fork_patch.sh`) |
| Target features | layers 1/8/15/22/29 plus the last layer | five layers, as the fork's DSpark loader expects |
| Heads | Markov head rank 256; confidence head fed by backbone state + Markov embedding | DSpark defaults |
| Loss | CE 0.1 + total-variation 0.9 + confidence BCE, position weights exp(−(k−1)/γ) | DeepSpec's objective; TV = half the L1 |
| Optimizer | Muon on 2-D weights, AdamW on the rest, lr 3e-4, default cosine schedule | Muon ≈ 300 ms of the ≈ 650 ms step |
| Epochs | 10 | DeepSpec's count; validation curve decides early stop |
| Precision | bf16 hidden states and compute, fp32 master weights, dynamo on, eager loss | fused loss trips torch 2.11 dynamo |
| Checkpoints | every quarter epoch (`--checkpoint-freq 0.25`), model 3.4 GB + optimizer 3.6 GB, resume automatic from the save path | a 24 h job loses at most a quarter epoch |

## 4. Layout and job chain

| Job | Script | Layout | Status |
|---|---|---|---|
| regeneration 4883730 / 31 / 33 / 34 | `apertus_opb_regen.sh` | 1 node each, 4-way data-parallel server, concurrency 512 | done, 2 h 05 min each |
| prepare 4883735 | `apertus_opb_prepare.sh` | 1 node | done, 25 min |
| trial 4883736 | `apertus_opb_train.sh` TRIAL=1 | 2 nodes on the 5k pilot data | passed, 23.5k tok/s per rank |
| train A 4883806 | `apertus_opb_train.sh` (proxy layout) | 2 nodes: per node 2 extraction servers behind a TCP round-robin proxy + 2 trainer ranks | ran 00:45–05:39, steps 0–11.5k; died of a 10-min NCCL collective timeout after one rank stalled on the overloaded server |
| train B 4886635 | `apertus_opb_train_v2.sh` | 2 nodes: each trainer rank reads its own server (port 8001 + local rank), no proxy | running since 05:44, resumed from step 6,235; ≈ 40 steps/min |
| train C / D / E 4937512 → 13 → 14 | `apertus_opb_train_v3.sh` | as v2 plus `--num-workers 8 --prefetch-factor 4` and process-group timeout 60 min (`SPEC_PG_TIMEOUT_MIN`, patch in `train/distributed.py`) | queued, 24 h each, resume from the shared save path |
| verify 4937515 | `apertus_dspark_verify_param.sh` | 1 node | queued after E |

Node facts: trainer GPUs at 100 %, extraction GPUs at 0–9 % under the per-rank layout (extraction is no longer the limiter; the remaining gap to the 0.65 s profiled step is dataloader depth, addressed in v3); host RSS ≈ 130 GB for the trainers, ≈ 490 GB per node including the servers; node nid005669 excluded (faulty GPU 0).

Throughput and cost so far: regeneration ≈ 8.4 GPU-h per shard (33 GPU-h total, ≈ 31k tokens/s per node); training ≈ 40 steps/min on 8 GPUs → ≈ 10.4 h per epoch, ≈ 80 GPU-h per epoch; 10 epochs ≈ 800 GPU-h at this rate, less if v3 lifts the step rate.

## 5. Evaluation plan

| Layer | Prompts | Setting | When |
|---|---|---|---|
| Training-time validation | 10 % of the regenerated corpus (held out) | teacher-forced; expected accepted length, per-position accuracy, confidence error | every epoch, in the trainer |
| Verification probe | 32 math_reasoning + 32 HumanEval prompts from `RedHatAI/speculator_benchmarks` | greedy, both thinking modes, k = 3/5/7, 8 streams; single-stream latency on 8 math prompts; greedy equality vs a no-speculation server | job 4937515, after the chain; same probe as for the Magpie drafter (4.98 / 3.23 at k = 7) |
| DSpark paper protocol | nine benchmarks: GSM8K, MATH500, AIME25; MBPP, HumanEval, LiveCodeBench; MT-Bench, Alpaca, Arena-Hard | standard speculative sampling at T = 1.0, chain drafting, accepted length per round incl. bonus, non-thinking | to schedule on `checkpoint_best` (DeepSpec harness; `jobs/deepspec_eval_small.py` reproduced their Qwen3-8B table on Daint) |
| Workload coverage | the nine-subset Phase 1 harness (HumanEval, math, qa, question, rag, summarization, tool_call, translation, writing), both modes, sweeps | vLLM 0.29-style GuideLLM harness against the fork server | after the paper protocol; needed because Open-PerfectBlend is math/code-heavy |

## 6. What to tell a colleague training Eagle3 on the same data

- Use the regenerated corpus file on the project store, not a fresh regeneration: same prompts, same Apertus answers, same 90/10 split when the trainer seed is left at its default.
- Maximum conversation length 8,192 tokens, answers capped at 4,096 generated tokens, packed 16,384 tokens per GPU per step; Eagle3 reads target layers 2/16/29 (three) where DSpark reads five, everything else identical.
- Evaluate with the same three layers above, in this order: the 64-prompt probe first (10 minutes per checkpoint), the nine DeepSpec benchmarks at T = 1.0 for the paper comparison, the nine-subset harness for workload coverage.
