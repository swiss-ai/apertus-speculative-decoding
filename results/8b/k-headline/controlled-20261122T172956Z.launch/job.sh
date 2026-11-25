#!/bin/bash
#SBATCH --job-name=task-speed
#SBATCH --account=d_research_serving
#SBATCH --partition=normal
#SBATCH --time=07:00:00
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --output=/iopsstor/scratch/cscs/faruk_zahiragic/k-headline/code/results/8b/k-headline/controlled-20261122T172956Z.launch/job.out
set -euo pipefail
srun --overlap --nodes=1 --ntasks=1 nvidia-smi   --query-gpu=timestamp,index,memory.used,utilization.gpu,power.draw,clocks.sm   --format=csv,noheader,nounits -lms 1000 > "/iopsstor/scratch/cscs/faruk_zahiragic/k-headline/code/results/8b/k-headline/controlled-20261122T172956Z.launch/gpu.csv" 2> /dev/null &
SAMPLER=$!
# Extend the image's PYTHONPATH (it carries /workspace/vllm), never replace it.
srun --overlap --nodes=1 --ntasks=1 --cpus-per-task=64 --environment="/iopsstor/scratch/cscs/faruk_zahiragic/k-headline/pilot-dev-dspark-12h-20261029-a-restored/environment.toml"   env PYTHONNOUSERSITE=1 TASK_SPEED_BRANCH="experiment/dspark-e10-k-headline" TASK_SPEED_REVISION="288126c40e9a505ebaaebca1331b9c7b37084ae8" bash -c   'export PYTHONPATH="/iopsstor/scratch/cscs/faruk_zahiragic/k-headline/code/src:${PYTHONPATH:-}"; exec python3 "/iopsstor/scratch/cscs/faruk_zahiragic/k-headline/code/experiments/task-speed/run-controlled.py" --suite "/iopsstor/scratch/cscs/faruk_zahiragic/k-headline/code/experiments/task-speed/suite.yaml" --pilot "/iopsstor/scratch/cscs/faruk_zahiragic/k-headline/pilot-dev-dspark-12h-20261029-a-restored" --inputs /iopsstor/scratch/cscs/faruk_zahiragic/k-headline/code/workloads/8b/task-speed-input  --ks 3,5,7 --gpus 0,1,2,3 --first-pair 1 --output "/iopsstor/scratch/cscs/faruk_zahiragic/k-headline/code/results/8b/k-headline/controlled-20261122T172956Z"'   || status=$?
kill ${SAMPLER} 2> /dev/null || true
exit ${status:-0}
