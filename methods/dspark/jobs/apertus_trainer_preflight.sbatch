#!/bin/bash -l
#SBATCH -A sm94 -p normal -N 1 --gpus-per-node=1 -t 00:40:00 -J apertus-trainer-preflight -o /capstor/scratch/cscs/zyu/spec/logs/%x-%j.out
srun --environment=${ENVNAME:-vllm29} bash /capstor/scratch/cscs/zyu/spec/jobs/apertus_trainer_preflight.sh
