#!/bin/bash -l
#SBATCH -A sm94 -p normal -N 1 --gpus-per-node=1 -t 01:00:00 -J apertus-gate -o /capstor/scratch/cscs/zyu/spec/logs/%x-%j.out
srun --environment=${ENVNAME:-vllm-apertus15} bash /capstor/scratch/cscs/zyu/spec/jobs/apertus_gate.sh
