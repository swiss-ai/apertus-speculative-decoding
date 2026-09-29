# DSpark drafter for Apertus 1.5 8B: training scripts

Job scripts, patches and run configurations used to train and serve speculative-decoding drafters for **Apertus-v1.5-8B** on the CSCS Alps/Daint GH200 partition with the [speculators](https://github.com/vllm-project/speculators) trainer and the Swiss AI vLLM fork.

The files are copied verbatim from the job directory on Daint (`$SPEC/jobs`, `SPEC=/capstor/scratch/cscs/zyu/spec`). Paths, the Slurm account (`sm94`) and the node layout are hard-coded; adapt them before reuse.

## Layout

| Path | Contents |
|---|---|
| `jobs/*.sbatch` | Slurm batch files; each one runs the shell script of the same name inside the container |
| `jobs/*.sh` | The scripts themselves (see the table below) |
| `jobs/*.py` | Helpers: prompt sharding, dataset statistics, acceptance probes, log summaries |
| `patches/` | Local diff on speculators commit `7046ac63fac4eeb218dc2e92a385c7d1ec8c4e00` and the commit record |
| `env/` | CSCS EDF container definitions |
| `configs/` | The trainer's own run records (`train_command.txt`, `run.yaml`, captured patch) for the two DSpark runs |

## Environment

- Container: `env/vllm-apertus15-ghcr.toml`, image `ghcr.io/swiss-ai/vllm_apertus_1.5_release:latest-arm64`. Copy to `~/.edf/` and use `srun --environment=vllm-apertus15-ghcr`.
- speculators: clone `vllm-project/speculators` at the commit above into `$SPEC/repos/speculators` and apply `patches/speculators_7046ac6_daint.patch`. The scripts add `src/` and `hs_connectors/src/` to `PYTHONPATH`; nothing is installed into the image.
- Extra Python packages missing from the image are installed without dependencies into `$SPEC/venvs/apertus-extra-ghcr` by `jobs/side_deps.sh`.
- `jobs/fork_patch.sh` copies the image's `vllm` package to `$SPEC/vllm_fork_patched_ghcr`, patches it (auxiliary hidden-state hooks, V2-runner speculator loaders, DSpark anchor layout following the checkpoint's `sample_from_anchor`) and prepends it to `PYTHONPATH`. `jobs/fork_patch_addendum.sh` holds the anchor-layout part on its own.
- Model: `jobs/download_apertus15_8b.sh` fetches the gated `swiss-ai/Apertus-v1.5-8B` into `$SPEC/hf` (requires a Hugging Face token with access; not included here).

## Scripts

| Script | Purpose | Slurm |
|---|---|---|
| `make_opb_shards.py` | Shuffle `mlabonne/open-perfectblend` once (seed 0), drop rows without a leading user turn, write N JSONL shards | login node or container |
| `apertus_opb_regen.sh` | Regenerate one shard with Apertus-v1.5-8B, thinking off (`SHARD`, `NSH` env vars); 4-way data-parallel server, concurrency 512, 4,096-token answer cap | 1 node, 4 h |
| `apertus_opb_regen_think.sh` | Same with thinking on, 8,192-token answer cap | 1 node |
| `apertus_opb_prepare.sh` / `_think.sh` | Merge the shards and run `speculators prepare-data --seq-length 8192` | 1 node, 2 h |
| `apertus_opb_train_v3.sh` | Two-node DSpark training: per node two extraction servers (GPUs 0, 1) and two trainer ranks (GPUs 2, 3), ranks joined by `torchrun` c10d rendezvous; each rank uses its own server through `train_rank_wrapper.sh`. Env knobs: `TRIAL`, `EPOCHS`, `SEQ`, `ANCHORS`, `LR`, `WORKERS`, `PREFETCH`, `SPEC_PG_TIMEOUT_MIN` | 2 nodes, 24 h, chain with `--dependency=afterany` |
| `apertus_opb_train.sh`, `apertus_opb_train_v2.sh`, `rr_proxy.py` | Earlier multi-node layouts (proxy-based, then per-rank servers without the deeper dataloader) | superseded by v3 |
| `apertus_opb_verify.sbatch` → `apertus_dspark_verify_param.sh` | Serve a checkpoint with `--speculative-config` and probe acceptance on 64 prompts, both thinking modes, k = 3/5/7, plus a no-speculation baseline | 1 node, 1 h |
| `apertus_dspark_verify.sh`, `_verify2.sh` | Earlier versions of the same probe | |
| `apertus_serve_probe.sh` | Serve one drafter checkpoint and probe acceptance (`NAME CKPT METHOD K`) | 1 node |
| `apertus_regen100k.sh`, `apertus_dspark_scale.sh` | The 100k-sample Magpie run: regeneration and one-node training | 1 node, 12 h |
| `apertus_dspark_bench.sh`, `apertus_dspark_diag.sh` | Trainer profiling and serving diagnostics | 1 node |
| `apertus_pipe5k.sh` | 5k-sample end-to-end check: regenerate, prepare, train Eagle3 / DFlash / DSpark, serve, probe | 1 node |
| `apertus_gate.sh`, `apertus_trainer_preflight.sh` | Serving and extraction check; trainer dry-run in a given image | 1 node |
| `bench_summary.py`, `spec_accept_apertus.py`, `deepspec_eval_small.py` | Parse training logs; measure accepted length against a served model; small DeepSpec-style evaluation | |
| `data_stats.py`, `magpie_stats.py`, `opb_stats.py`, `opb_dropped.py`, `regen_stats.py`, `peek_sft_mix.py` | Dataset statistics | |

## Submitting the main pipeline

```
for s in 0 1 2 3; do sbatch --export=ALL,SHARD=$s,NSH=4 jobs/apertus_opb_regen.sbatch; done
sbatch --dependency=afterok:<regen job ids> jobs/apertus_opb_prepare.sbatch
A=$(sbatch --parsable -N 2 -t 24:00:00 jobs/apertus_opb_train_v3.sbatch)
B=$(sbatch --parsable -N 2 -t 24:00:00 --dependency=afterany:$A jobs/apertus_opb_train_v3.sbatch)
sbatch --dependency=afterany:$B jobs/apertus_opb_verify.sbatch
```

Training resumes automatically from the numbered checkpoints in the save path, so chained jobs continue where the previous one stopped.

## Trainer settings (from `configs/opb_nothink/train_command.txt`)

```
--speculator-type dspark --num-layers 5 --block-size 8 --markov-rank 256 --enable-confidence-head --confidence-head-with-markov
--target-layer-ids 1 8 15 22 29 --loss-fn '{"ce":0.1,"tv":0.9}' --loss-implementation eager --optimizer muon --lr 3e-4
--epochs 10 --total-seq-len 16384 --max-anchors 1024 --num-workers 8 --prefetch-factor 4 --checkpoint-freq 0.125
--hidden-states-backend file --hidden-states-path /tmp/hidden_states --on-missing generate --on-generate delete
```

Data: Open-PerfectBlend answers regenerated by Apertus-v1.5-8B (temperature 0.7, top-p 0.8, top-k 20), conversations truncated at 8,192 tokens, trainer default 90/10 train/validation split.
