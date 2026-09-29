#!/bin/bash -l
#SBATCH -A sm94 -p normal --ntasks-per-node=1 --gpus-per-node=4 --cpus-per-task=64 -J apertus-opb-train -o /capstor/scratch/cscs/zyu/spec/logs/%x-%j.out
srun --ntasks-per-node=1 --environment=${ENVNAME:-vllm-apertus15-ghcr} bash /capstor/scratch/cscs/zyu/spec/jobs/apertus_opb_train_v2.sh
