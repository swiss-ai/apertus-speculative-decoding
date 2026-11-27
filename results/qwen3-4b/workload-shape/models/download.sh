#!/bin/bash
#SBATCH --job-name=wshape-q4b-download
#SBATCH --account=d_research_serving
#SBATCH --partition=normal
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --output=/iopsstor/scratch/cscs/faruk_zahiragic/workload-shape-qwen3-4b/models/download-%j.out
# Download the pinned Qwen3-4B target and the Red Hat DSpark speculator for the workload-shape study.
set -euo pipefail
M=/iopsstor/scratch/cscs/faruk_zahiragic/workload-shape-qwen3-4b/models
export HF_HOME=/iopsstor/scratch/cscs/faruk_zahiragic/workload-shape-qwen3-4b/hf_home
export HF_HUB_CACHE=$HF_HOME/hub HF_XET_CACHE=$HF_HOME/xet HF_HUB_DISABLE_TELEMETRY=1 HF_XET_HIGH_PERFORMANCE=1
echo "job=$SLURM_JOB_ID node=$SLURMD_NODENAME start=$(date -u +%FT%TZ)"
~/miniconda/bin/python - <<'PY'
from huggingface_hub import snapshot_download
M = "/iopsstor/scratch/cscs/faruk_zahiragic/workload-shape-qwen3-4b/models"
for repo, rev in [
    ("RedHatAI/Qwen3-4B-speculator.dspark", "bdffaa5f011da5a1a58ad841a1f6ced3e87f2f23"),
    ("Qwen/Qwen3-4B", "1cfa9a7208912126459214e8b04321603b3df60c"),
]:
    p = snapshot_download(repo, revision=rev, local_dir=f"{M}/{repo.replace('/', '--')}", max_workers=16)
    open(f"{p}/hf-revision.txt", "w").write(f"{repo} {rev}\n")
    print("done", repo, rev, p, flush=True)
PY
cd $M
for d in RedHatAI--Qwen3-4B-speculator.dspark Qwen--Qwen3-4B; do
  (cd $d && find . -type f ! -path './.cache/*' ! -name SHA256SUMS -print0 | sort -z | xargs -0 -n 4 sha256sum | sort -k2 > SHA256SUMS)
done
echo "finished=$(date -u +%FT%TZ)"
