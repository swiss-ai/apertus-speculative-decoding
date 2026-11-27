#!/bin/bash
#SBATCH --job-name=shape-qwen3-8b-download
#SBATCH --account=d_research_serving
#SBATCH --partition=normal
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --output=/iopsstor/scratch/cscs/faruk_zahiragic/workload-shape-qwen3-8b/models/download-%j.out
# Download the pinned Qwen3-8B target and the two official Red Hat drafters for the
# Qwen3-8B workload-shape study (branch experiment/workload-shape-qwen3-8b).
# The DFlash optimizer/scheduler states are training-resume files, not needed and skipped.
set -euo pipefail
M=/iopsstor/scratch/cscs/faruk_zahiragic/workload-shape-qwen3-8b/models
export HF_HOME=/tmp/hfhome-$SLURM_JOB_ID HF_HUB_CACHE=/tmp/hfhome-$SLURM_JOB_ID/hub HF_XET_CACHE=/tmp/hfhome-$SLURM_JOB_ID/xet HF_HUB_DISABLE_TELEMETRY=1
export HF_XET_HIGH_PERFORMANCE=1
echo "job=$SLURM_JOB_ID node=$SLURMD_NODENAME start=$(date -u +%FT%TZ)"
~/miniconda/bin/python - <<'PY'
from huggingface_hub import snapshot_download
M = "/iopsstor/scratch/cscs/faruk_zahiragic/workload-shape-qwen3-8b/models"
for repo, rev, ignore in [
    ("RedHatAI/Qwen3-8B-speculator.eagle3", "08610ffa01dd9f16731fe8f627b85905b6aa51c4", None),
    ("RedHatAI/Qwen3-8B-speculator.dflash", "1a11b170eb65c8a62c80ecd01dfe22a5907298e6", ["optimizer_state_dict.pt", "scheduler_state_dict.pt"]),
    ("Qwen/Qwen3-8B", "b968826d9c46dd6066d109eabc6255188de91218", None),
]:
    p = snapshot_download(repo, revision=rev, local_dir=f"{M}/{repo.replace('/', '--')}", max_workers=16, ignore_patterns=ignore)
    open(f"{p}/hf-revision.txt", "w").write(f"{repo} {rev}\n")
    print("done", repo, rev, p, flush=True)
PY
cd $M
for d in RedHatAI--Qwen3-8B-speculator.eagle3 RedHatAI--Qwen3-8B-speculator.dflash Qwen--Qwen3-8B; do
  (cd $d && find . -type f ! -path './.cache/*' ! -name SHA256SUMS -print0 | sort -z | xargs -0 -P 16 -n 4 sha256sum | sort -k2 > SHA256SUMS)
done
echo "finished=$(date -u +%FT%TZ)"
