# Apertus speculative decoding — Agent Guide

## Separate branches for new work

- Start every new task or experiment on its own named Git branch before editing
  files, committing changes, or launching runs. This includes new training,
  evaluation, benchmark, and serving campaigns. Do not put new work on `main` or
  on an unrelated task or experiment branch.
- Check the current branch, working-tree changes, existing run artifacts, and
  submitted or running jobs before starting or resuming work. Preserve unrelated
  changes and reuse the existing task branch when continuing the same work.
  Check an experiment's status before restarting or submitting a replacement.
- Use an isolated worktree or checkout on the task branch when another task or
  running job uses the existing checkout. Keep the branch and commit associated
  with each run in its provenance; preserve the exact code used by existing runs.

## Slack action items

When the user refers to the "action items canvas", they mean [Spec Decoding Sync: Prioritized Actions and Owners](https://swissai-initiative.slack.com/docs/T0627JS49GT/F0C6431UBTP) (canvas ID: `F0C6431UBTP`) in Slack channel [#swiss-ai-spec-decode](https://swissai-initiative.slack.com/archives/C0C2QMYCQ5B) (channel ID: `C0C2QMYCQ5B`).

## Provenance

Training, evaluation, and serving runs must retain reproducibility artifacts so runs can be recreated later. Follow the same provenance policy as Speculators, using this repository's own artifact formats and launchers.

- **Training** (`methods/eagle/apertus_eagle/train_rollout.py`): preserve `provenance.json`, `train-summary.json`, `evals.json`, and `history.json` with the associated run. The pipeline copies these into its results tree. Keep the training configuration, target contract, and exported head's `config.json` and `export-report.json` with the run's reproducibility artifacts.
- **Benchmarks** (`src/apertus_bench/`): preserve each cell's `metadata.json`, `requests.jsonl`, `summary.json`, and raw metric snapshots when collected. Pass the relevant non-secret provenance with `--metadata KEY=VALUE`. EAGLE measurements require `deployment_id`, `checkpoint_sha256`, `target_model`, `target_revision`, and `target_tensor_parallel_size`; use the existing measurement launchers to populate these fields.
- **Serving** (`serving/` and `methods/*/launch/`): record Git revisions, the exact launch command, container path and digest, applied patches or source overlays, target and drafter identities and hashes, corpus digest, and Slurm job and node IDs alongside the associated results. Preserve launch logs, checkpoint validation reports, and resolved environment files produced by the launcher.

Use a new run directory for each independently launched deployment. Never overwrite completed raw result cells or earlier comparisons; follow `results/README.md` and `docs/protocol.md`.

When publishing a model or evaluation results (HuggingFace, GitHub comments, etc.), include the associated provenance, configurations, hashes, patches, and raw measurement artifacts so the run can be reproduced.

## When using Speculators

The following rules also apply when running training, evaluation, or serving through a Speculators checkout, such as `../speculators/` or a scratch checkout. Paths in this section are relative to that checkout.

- **Training** (`src/speculators/train/utils.py`): `save_train_command()` writes `train_command.txt` (timestamp, git SHA, world size, package versions, full argv) and `speculators.patch` (uncommitted diff) into the checkpoint directory.
- **Eval** (`scripts/evaluate/evaluate.py`): `save_eval_provenance()` writes `eval_command.txt` (timestamp, git SHA, package versions, full argv) into the output directory.
- **vLLM launch** (`scripts/launch_vllm.py`): `_save_vllm_provenance()` writes `vllm_command.txt`, `vllm.patch`, and `checkpoint_sha256.txt` (plus `drafter_checkpoint_sha256.txt` in eval mode) — but **only when `--provenance-dir` is passed**. Two subcommands:
  - `launch_vllm.py train MODEL` (default): hidden-states extraction for training data generation.
  - `launch_vllm.py eval MODEL --spec-model DRAFTER`: speculative decoding serving for evaluation.

**Always pass `--provenance-dir` when running Speculators' `scripts/launch_vllm.py`** — point it at the associated run's output/checkpoint directory (e.g. `--provenance-dir <output_dir>`) so the vLLM provenance is co-located with the training/eval it belongs to.

When publishing a model or eval results produced with Speculators, always include its provenance artifacts (`train_command.txt`, `eval_command.txt`, `vllm_command.txt`, `checkpoint_sha256.txt`, `drafter_checkpoint_sha256.txt`, patches) so the run can be reproduced.
