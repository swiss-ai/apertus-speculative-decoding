#!/bin/bash
#SBATCH --job-name=k-headline-coverage
#SBATCH --account=d_research_serving
#SBATCH --partition=normal
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --output=/iopsstor/scratch/cscs/faruk_zahiragic/k-headline/coverage-build/job.out
set -euo pipefail
srun --ntasks=1 --cpus-per-task=16 --environment=/iopsstor/scratch/cscs/faruk_zahiragic/k-headline/pilot-dev-dspark-12h-20261029-a-restored/environment.toml   env PYTHONNOUSERSITE=1 HF_HUB_OFFLINE=1 bash -c "export PYTHONPATH=/iopsstor/scratch/cscs/faruk_zahiragic/k-headline/code/src:\${PYTHONPATH:-}; cd /iopsstor/scratch/cscs/faruk_zahiragic/k-headline; python3 /iopsstor/scratch/cscs/faruk_zahiragic/k-headline/coverage_workloads.py --raw /iopsstor/scratch/cscs/faruk_zahiragic/k-headline/coverage-raw --tokenizer /capstor/store/cscs/swissai/infra01/hf_models/models/swiss-ai/Apertus-v1.5-8B --out /iopsstor/scratch/cscs/faruk_zahiragic/k-headline/coverage-build/coverage.jsonl --manifest /iopsstor/scratch/cscs/faruk_zahiragic/k-headline/coverage-build/manifest.json > /iopsstor/scratch/cscs/faruk_zahiragic/k-headline/coverage-build/build.log 2>&1; sha256sum /iopsstor/scratch/cscs/faruk_zahiragic/k-headline/coverage-build/coverage.jsonl"
