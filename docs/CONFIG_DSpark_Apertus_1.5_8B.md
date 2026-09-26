# Configuration: DSpark drafter for Apertus 1.5 8B (as run on Daint, 24–25 September 2026)

The configuration that produced the drafter measured at 4.98 accepted tokens per step (thinking off, k = 7, 3.5× single-stream). Scripts: `results/apertus15_8b_dspark_bench_2026-09-24/` in this repo, `/capstor/scratch/cscs/zyu/spec/jobs/` on Daint. The exact trainer invocation of every run is saved by the trainer itself in `<checkpoint dir>/train_command.txt` and `run.yaml`.

## 1. Environment

| Item | Value |
|---|---|
| Cluster | Daint (Alps), account `sm94`, partition `normal` (24 h limit), one node = 4 × GH200 96 GB, 870 GB host RAM |
| Container | `ghcr.io/swiss-ai/vllm_apertus_1.5_release:latest-arm64` via EDF `~/.edf/vllm-apertus15-ghcr.toml` (mounts `/capstor /iopsstor /users`, `HF_HOME=$SPEC/hf`, `HF_HUB_OFFLINE=1`) |
| Stack inside the image | Swiss AI vLLM fork 0.23.1rc1 (commit a601a9d9), torch 2.11.0+cu130, transformers 5.14.0.dev0 (`Apertus1p5ForConditionalGeneration`) |
| Trainer | RedHatAI `speculators` 0.9.0.dev36 (git 7046ac6) at `$SPEC/repos/speculators`, on `PYTHONPATH` (never pip-installed into the image) |
| Side dependencies | `$SPEC/venvs/apertus-extra-ghcr`, installed with `pip install --no-deps --target` by `jobs/side_deps.sh` |
| Target checkpoint | `$SPEC/hf/hub/models--swiss-ai--Apertus-v1.5-8B/snapshots/a411d838600baf0e3635a3daf66fb7c55fc97bb6` (gated repo; token at `~/.cache/huggingface/token`) |
| Work directory | `SPEC=/capstor/scratch/cscs/zyu/spec` (explicit path; `$SCRATCH` points elsewhere) |
| Slurm header | `-N 1 --ntasks=1 --ntasks-per-node=1 --gpus-per-node=4 --cpus-per-task=64`, `srun -n 1 --environment=vllm-apertus15-ghcr` |
| Compile cache | `TORCHINDUCTOR_CACHE_DIR=$SPEC/cache/inductor` (compile 165 s cold, ~80 s warm) |

## 2. Patches (all idempotent, applied by the job scripts)

| Patch | Where | Why |
|---|---|---|
| Pruned output head | `results/speculators_pruned_head.patch` on the speculators clone (3 files) | Apertus has `vocab_size` 266,752 but an `lm_head` of 131,072 rows (`output_vocab_size`); heads and t2d follow the output vocabulary, embeddings keep the input vocabulary; also `head_dim=resolved_head_dim` |
| Aux-hidden-state hooks | `jobs/fork_patch.sh` → scratch copy `$SPEC/vllm_fork_patched_ghcr`, shadowed via `PYTHONPATH` | the fork's `set_aux_hidden_state_layers` and the V2 speculator loaders assume an extra `.model` wrapper that Apertus's multimodal class does not have |
| DSpark anchor layout | same script, edits `transformers_utils/configs/speculators/algos.py` | the fork forces `dspark_bonus_anchor = True` (1+N block) for speculators checkpoints; speculators trains DSpark with `sample_from_anchor = True` (anchor-as-first). Patched to `dspark_bonus_anchor = not sample_from_anchor`. Without it the drafter serves at 1.96 instead of 4.98 |

## 3. Data

| Step | Setting |
|---|---|
| Prompts | `speculators regenerate-responses --dataset magpie --limit 100000` (Magpie-Llama-3.1-Pro-300K-Filtered, single turn) |
| Regeneration server | `vllm serve <target> --trust-remote-code --skip-mm-profiling --chat-template-content-format string --max-model-len 8192 --gpu-memory-utilization 0.90 --max-num-seqs 256 --data-parallel-size 4` on 4 GPUs (works in this fork with `--ntasks=1`) |
| Sampling | `{"temperature":0.7,"top_p":0.8,"top_k":20,"chat_template_kwargs":{"enable_thinking":false}}`, `--max-tokens 4096`, `--concurrency 512` |
| Throughput | 41.7k output tokens/s per node; 100k samples in 22 min; 0 errors, 0.5 % truncated; ≈ 550 generated tokens per sample |
| Prepare | `speculators prepare-data --model <target> --trust-remote-code --data <jsonl> --output <data> --token-freq-path <data>/token_freq.pt --no-skip-token-freq --seq-length 8192` (73 s for 100k; writes the 32k `d2t`/`t2d` maps, unused with the full head) |
| Result | ≈ 66 M tokens per epoch; 90 % train / 10 % validation (trainer default) |

Not yet covered: thinking-mode regeneration (needs the round-trip check of the inline thinking span through the chat template), tool-call traces, multilingual prompts.

## 4. Hidden-state extraction

```
python scripts/launch_vllm.py train <target> --target-layer-ids 1 8 15 22 29 --hidden-states-backend file --hidden-states-path <cache> \
  -- --trust-remote-code --skip-mm-profiling --renderer-num-workers 1 --mm-processor-cache-gb 0 --port 8000 --gpu-memory-utilization 0.85 --max-model-len 8192
```

- Served under the model path (no `--served-model-name`), one GPU (84 GB used, 55 GB KV cache).
- Five target layers plus the last layer are stored: 46 KB per token (bf16), one file per sample; 100k samples = 3.2 TB, 100k files (scratch quota 150 TB, 1 M files).
- Capacity ≈ 35k tokens/s per server including the capstor write; enough for two 16k trainers, not three.

## 5. Trainer (run 1 of the scale demo, `checkpoint_best` = epoch index 7)

```
CUDA_VISIBLE_DEVICES=1,2,3 torchrun --standalone --nproc_per_node 3 -m speculators.train \
  --verifier-name-or-path <target> --data-path <data> \
  --hidden-states-backend file --hidden-states-path <cache> --vllm-endpoint http://localhost:8000/v1 \
  --on-missing generate --on-generate cache --target-layer-ids 1 8 15 22 29 \
  --speculator-type dspark --num-layers 5 --block-size 8 --markov-rank 256 --enable-confidence-head --confidence-head-with-markov \
  --loss-fn '{"ce":0.1,"tv":0.9}' --loss-implementation eager \
  --optimizer muon --lr 3e-4 --epochs 10 --total-seq-len 16384 --max-anchors 1024 --checkpoint-freq 0.5 --save-path <ckpt>
```

| Setting | Value | Note |
|---|---|---|
| Draft architecture | 5 Qwen3-style decoder layers, hidden 4096, 32 heads / 8 KV heads, MLP 21,504; `sample_from_anchor = true`; block 8 → up to 7 draft tokens | derived from the Apertus config by the trainer |
| Vocabulary | full 131,072-row head (no `--draft-vocab-size`); embeddings 266,752 (frozen copy of the target) | 32k pruned head gives +9 % speed only |
| Target features | layers 1/8/15/22/29 (five, evenly spaced over 32); the fork's DSpark loader requires five | Eagle3 would use 2/16/29 |
| Loss | CE 0.1 + total variation 0.9, plus confidence-head loss; D-PACE position weights are the DSpark default | speculators' equivalent of DeepSpec's CE 0.1 + L1 0.9 |
| Optimizer | Muon (trainer default) for 2-D weights, AdamW for the rest; lr 3e-4; cosine schedule (default) | AdamW (`--optimizer adamw`) halves the step time but learned less per step at 5k scale |
| Packing / anchors | 16,384 tokens per packed batch per GPU, 1,024 anchors × 8 slots supervised per step | supervision per step is set by anchors, not tokens |
| Precision | bf16 hidden states and compute, fp32 master weights; dynamo on, eager loss (fused loss trips a torch-2.11 dynamo assertion) | |
| Dataloader | 12 workers × prefetch 4 per rank (defaults) | this is what exhausted host RAM at 16k packing; use `--num-workers 4 --prefetch-factor 2` |
| Checkpoints | every half epoch and at epoch end (model 3.4 GB + optimizer 3.6 GB), `checkpoint_best` symlink by validation loss; resume from numbered directories is automatic; the SIGINT `interrupted` directory carries no step position | |

Measured: 1,312 steps per epoch on 3 GPUs; 31 min for the online first epoch, 15–17 min per cached epoch; 26.5k tokens/s per GPU, step ≈ 610 ms (optimizer 306, backward 180, forward 110, fetch 14); 87–94 GB per trainer GPU; host RSS 821 GB peak. Validation expected accepted length by epoch: 3.01, 3.64, 3.89, 4.04, 4.16, 4.23, 4.28, 4.32, 4.36.

## 6. Serving and evaluation

```
vllm serve <target> --served-model-name apertus15-8b --trust-remote-code --skip-mm-profiling --max-model-len 8192 \
  --max-num-seqs 64 --max-num-batched-tokens 16384 --gpu-memory-utilization 0.85 \
  --speculative-config '{"method":"dspark","model":"<ckpt>/checkpoint_best","num_speculative_tokens":7}'
```

- Requires the anchor-layout patch (§2); `draft_sample_method` greedy (default). The drafter shares the GPU with the target.
- Probe: 64 prompts (32 math_reasoning + 32 HumanEval from `RedHatAI/speculator_benchmarks`), greedy, 8 concurrent, 384 tokens (thinking off) or 1,024 (thinking on); acceptance from the `vllm:spec_decode_*` counters; single-stream latency on 8 math prompts; greedy-equality against a no-speculation server (`jobs/apertus_dspark_verify.sh`, `probe.py`).

| k | accepted / step, thinking off | thinking on | single stream ms/token | 8 streams tok/s (off) |
|---|---|---|---|---|
| none | 1 | 1 | 5.79 | 1,242 |
| 3 | 3.41 | 2.66 | 2.32 | 3,089 |
| 5 | 4.41 | 3.06 | 1.84 | 3,580 |
| 7 | 4.98 | 3.23 | 1.65 | 4,152 |

## 7. Resources for this configuration

| Phase | Time | GPU-hours |
|---|---|---|
| Regeneration, 100k | 22 min on 4 GPUs | 1.5 |
| Extraction + 9 epochs of training | 3.4 h on 4 GPUs (1 + 3) | ≈ 14 |
| Serving probes | ≈ 10 min per server | < 1 |

Scaling to 300k non-thinking samples: ≈ 1 h regeneration, ≈ 10 h training at the same settings (≈ 40 GPU-h); with 2,048 anchors at 8k packing (same supervision density as the pilot recipe) roughly twice that.

## 8. Recommended changes for the next run

1. `--total-seq-len 8192 --max-anchors 2048` (or 16k with `--num-workers 4 --prefetch-factor 2`) to keep supervision density and stay under host RAM.
2. Two extraction servers, or cache first and train from the cache on all four GPUs (`--on-missing raise`).
3. `--checkpoint-freq 0.25` on long epochs.
4. Exclude node nid005669 (`--exclude`) until its GPU 0 is repaired.
5. Add thinking-mode regeneration once the chat-template round trip is verified, and a per-mode evaluation.
