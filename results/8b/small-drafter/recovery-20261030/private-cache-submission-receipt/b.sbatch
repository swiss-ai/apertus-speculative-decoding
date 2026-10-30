#!/usr/bin/env bash
#SBATCH --job-name=depth-recovery-b
#SBATCH --account=infra01
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --time=04:00:00
#SBATCH --chdir=/iopsstor/scratch/cscs/faruk_zahiragic/spec-exp-data/small-drafter/campaigns/depth-recovery-20261030-gpu-calibration-v2/source
#SBATCH --output=/iopsstor/scratch/cscs/faruk_zahiragic/spec-exp-data/small-drafter/campaigns/depth-recovery-20261030-gpu-calibration-v2/b-job.out
set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1
trap 'python3 "/iopsstor/scratch/cscs/faruk_zahiragic/spec-exp-data/small-drafter/campaigns/depth-recovery-20261030-gpu-calibration-v2/source/methods/dspark/train/depth_campaign.py" fail "/iopsstor/scratch/cscs/faruk_zahiragic/spec-exp-data/small-drafter/campaigns/depth-recovery-20261030-gpu-calibration-v2" --stage "b"' ERR
# Pyxis has rejected long overlay mount lists before. The Slurm job ID makes
# this short path private to this allocation; patch-vllm may safely replace it.
export VLLM_PATCH_DIR="/iopsstor/scratch/cscs/faruk_zahiragic/sd3-${SLURM_JOB_ID}"
export SML_ENV_DIR="/iopsstor/scratch/cscs/faruk_zahiragic/spec-exp-data/small-drafter/campaigns/depth-recovery-20261030-gpu-calibration-v2/environments"
export DEPTH_SOURCE_MANIFEST="/iopsstor/scratch/cscs/faruk_zahiragic/spec-exp-data/small-drafter/campaigns/depth-recovery-20261030-gpu-calibration-v2/source-manifest.json"
python3 "/iopsstor/scratch/cscs/faruk_zahiragic/spec-exp-data/small-drafter/campaigns/depth-recovery-20261030-gpu-calibration-v2/source/methods/dspark/train/depth_campaign.py" verify "/iopsstor/scratch/cscs/faruk_zahiragic/spec-exp-data/small-drafter/campaigns/depth-recovery-20261030-gpu-calibration-v2"
PATCHES=("/iopsstor/scratch/cscs/faruk_zahiragic/spec-exp-data/small-drafter/campaigns/depth-recovery-20261030-gpu-calibration-v2/source"/serving/patches/vllm-apertus-image-token.patch
  "/iopsstor/scratch/cscs/faruk_zahiragic/spec-exp-data/small-drafter/campaigns/depth-recovery-20261030-gpu-calibration-v2/source"/serving/patches/vllm-apertus-eagle3-aux-layers.patch
  "/iopsstor/scratch/cscs/faruk_zahiragic/spec-exp-data/small-drafter/campaigns/depth-recovery-20261030-gpu-calibration-v2/source"/serving/patches/vllm-apertus-eagle-mrv2-inner.patch
  "/iopsstor/scratch/cscs/faruk_zahiragic/spec-exp-data/small-drafter/campaigns/depth-recovery-20261030-gpu-calibration-v2/source"/serving/patches/vllm-apertus-dspark-mrv2-inner.patch
  "/iopsstor/scratch/cscs/faruk_zahiragic/spec-exp-data/small-drafter/campaigns/depth-recovery-20261030-gpu-calibration-v2/source"/serving/patches/vllm-apertus-dspark-anchor-layout.patch
  "/iopsstor/scratch/cscs/faruk_zahiragic/spec-exp-data/small-drafter/campaigns/depth-recovery-20261030-gpu-calibration-v2/source"/serving/patches/vllm-apertus-dspark-num-layers.patch)
EXTRA_MOUNTS="$("/iopsstor/scratch/cscs/faruk_zahiragic/spec-exp-data/small-drafter/campaigns/depth-recovery-20261030-gpu-calibration-v2/source/serving/patch-vllm.sh"   /capstor/store/cscs/swissai/infra01/container-images/ci/vllm_apertus_1.5_release-arm64.sqsh   "${PATCHES[@]}")"
export EXTRA_MOUNTS
DEPTH_RESOLVED_ENV="$("/iopsstor/scratch/cscs/faruk_zahiragic/spec-exp-data/small-drafter/campaigns/depth-recovery-20261030-gpu-calibration-v2/source/serving/resolve-env.sh"   "/iopsstor/scratch/cscs/faruk_zahiragic/spec-exp-data/small-drafter/campaigns/depth-recovery-20261030-gpu-calibration-v2/source/methods/eagle/configs/train-env.toml")"
export DEPTH_RESOLVED_ENV
cp "${DEPTH_RESOLVED_ENV}" "/iopsstor/scratch/cscs/faruk_zahiragic/spec-exp-data/small-drafter/campaigns/depth-recovery-20261030-gpu-calibration-v2/b-environment.toml"
printf '%s\n' "${PATCHES[@]}" > "/iopsstor/scratch/cscs/faruk_zahiragic/spec-exp-data/small-drafter/campaigns/depth-recovery-20261030-gpu-calibration-v2/b-patches.txt"
srun --overlap --nodes=1 --ntasks=1 nvidia-smi   --query-gpu=timestamp,index,memory.used,memory.total,utilization.gpu,power.draw   --format=csv,noheader,nounits -lms 1000 > "/iopsstor/scratch/cscs/faruk_zahiragic/spec-exp-data/small-drafter/campaigns/depth-recovery-20261030-gpu-calibration-v2/b-gpu-memory.csv" 2>/dev/null &
SAMPLER=$!
trap 'kill "${SAMPLER}" 2>/dev/null || true' EXIT
srun --overlap --nodes=1 --ntasks=1 --environment="${DEPTH_RESOLVED_ENV}"   env PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1   SWEEP_CACHE_ROOT="/iopsstor/scratch/cscs/faruk_zahiragic/spec-exp-data/small-drafter/campaigns/depth-recovery-20261030-gpu-calibration-v2/cache-b"   bash -c 'cd "/iopsstor/scratch/cscs/faruk_zahiragic/spec-exp-data/small-drafter/campaigns/depth-recovery-20261030-gpu-calibration-v2/source"; export PYTHONPATH="/iopsstor/scratch/cscs/faruk_zahiragic/spec-exp-data/small-drafter/campaigns/depth-recovery-20261030-gpu-calibration-v2/source/src:${PYTHONPATH:-}"; exec python3 "/iopsstor/scratch/cscs/faruk_zahiragic/spec-exp-data/small-drafter/campaigns/depth-recovery-20261030-gpu-calibration-v2/source/methods/dspark/train/depth_campaign.py" run "/iopsstor/scratch/cscs/faruk_zahiragic/spec-exp-data/small-drafter/campaigns/depth-recovery-20261030-gpu-calibration-v2" --stage "b"'
