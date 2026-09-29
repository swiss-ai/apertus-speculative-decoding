#!/bin/bash -l
#SBATCH -A sm94 -p normal -N 1 --ntasks=1 --ntasks-per-node=1 --gpus-per-node=4 --cpus-per-task=64 -t 01:00:00 -J apertus-opb-verify -o /capstor/scratch/cscs/zyu/spec/logs/%x-%j.out
export OUT=/capstor/scratch/cscs/zyu/spec/runs/apertus_opb/verify CK=/capstor/scratch/cscs/zyu/spec/runs/apertus_opb/ckpt_dspark_nothink/checkpoint_best
srun -n 1 --environment=${ENVNAME:-vllm-apertus15-ghcr} bash /capstor/scratch/cscs/zyu/spec/jobs/apertus_dspark_verify_param.sh
