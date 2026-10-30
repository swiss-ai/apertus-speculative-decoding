#!/bin/bash
#SBATCH --job-name=bev-num-guard-preflight
#SBATCH --account=infra01
#SBATCH --partition=mi300
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --exclusive
#SBATCH --time=00:20:00
#SBATCH --chdir=/iopsstor/scratch/cscs/faruk_zahiragic/beverin-feasibility/repo-556a464d
#SBATCH --output=/iopsstor/scratch/cscs/faruk_zahiragic/beverin-feasibility/runs/guard-preflight-20261030-556a464d/slurm-%j.out
set -euo pipefail
export REPO_ROOT=/iopsstor/scratch/cscs/faruk_zahiragic/beverin-feasibility/repo-556a464d
export BEVERIN_PROBE_DIR=/iopsstor/scratch/cscs/faruk_zahiragic/beverin-feasibility/runs/guard-preflight-20261030-556a464d/probe
export BEVERIN_CAPTURE_BAD_BATCH=1
export BEVERIN_CHECK_ACTIVATIONS=1
export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH=/iopsstor/scratch/cscs/faruk_zahiragic/beverin-feasibility/repo-556a464d/methods/dspark/beverin
mkdir -p /iopsstor/scratch/cscs/faruk_zahiragic/beverin-feasibility/runs/guard-preflight-20261030-556a464d/deployments/${SLURM_JOB_ID}
DEPLOYMENT=/iopsstor/scratch/cscs/faruk_zahiragic/beverin-feasibility/runs/guard-preflight-20261030-556a464d/deployments/${SLURM_JOB_ID}
cp "$0" "$DEPLOYMENT/launch.sbatch"
cp /iopsstor/scratch/cscs/faruk_zahiragic/beverin-feasibility/runs/guard-preflight-20261030-556a464d/container-probe.sh "$DEPLOYMENT/container-probe.sh"
cp /iopsstor/scratch/cscs/faruk_zahiragic/beverin-feasibility/repo-556a464d/methods/dspark/beverin/env-vllm031.toml "$DEPLOYMENT/environment.toml"
cp /iopsstor/scratch/cscs/faruk_zahiragic/beverin-feasibility/repo-556a464d/REVISION "$DEPLOYMENT/repo-revision.txt"
cp /iopsstor/scratch/cscs/faruk_zahiragic/beverin-feasibility/repo-556a464d/SOURCE_SHA256SUMS "$DEPLOYMENT/source.sha256"
scontrol show job "$SLURM_JOB_ID" > "$DEPLOYMENT/slurm-job.txt"
scontrol show hostnames "$SLURM_JOB_NODELIST" > "$DEPLOYMENT/nodes.txt"
printf '%s\n' 'fix/beverin-nonfinite-diagnosis-20261030' > "$DEPLOYMENT/branch.txt"
printf '%s\n' 'torchrun --standalone --nproc-per-node=2 /iopsstor/scratch/cscs/faruk_zahiragic/beverin-feasibility/repo-556a464d/methods/dspark/beverin/check_numerical_guard.py' > "$DEPLOYMENT/command.txt"
sha256sum /capstor/scratch/cscs/faruk_zahiragic/container-images/vllm-openai-rocm-v0.31.0.sqsh > "$DEPLOYMENT/container.sha256" &
hash_pid=$!
printf 'START %s job=%s nodes=%s\n' "$(date -u -Iseconds)" "$SLURM_JOB_ID" "$SLURM_JOB_NODELIST"
rc=0
srun --input=none --kill-on-bad-exit=1 --environment=/iopsstor/scratch/cscs/faruk_zahiragic/beverin-feasibility/repo-556a464d/methods/dspark/beverin/env-vllm031.toml --export=ALL bash /iopsstor/scratch/cscs/faruk_zahiragic/beverin-feasibility/runs/guard-preflight-20261030-556a464d/container-probe.sh || rc=$?
wait "$hash_pid"
read actual_hash image_path < "$DEPLOYMENT/container.sha256"
[ "$actual_hash" = b58797bb6a430d7587f1c10ec24ac71f198bd8d590bde98dbbe68d7744a9816f ]
printf 'FINISH %s rc=%s\n' "$(date -u -Iseconds)" "$rc"
exit "$rc"
