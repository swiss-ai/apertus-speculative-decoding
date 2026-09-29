#!/bin/bash -l
#SBATCH -A sm94 -p normal -N 1 --ntasks=1 --gpus-per-node=1 -t 00:30:00 -J apertus-serve -o /capstor/scratch/cscs/zyu/spec/logs/%x-%j.out
srun -n 1 --environment=vllm-apertus15-ghcr bash /capstor/scratch/cscs/zyu/spec/jobs/apertus_serve_probe.sh "$NAME" "$CKPT" "$METHOD" "$K" "$EXTRA"
