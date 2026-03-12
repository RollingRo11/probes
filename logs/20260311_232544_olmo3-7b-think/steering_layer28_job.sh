#!/bin/bash
#SBATCH --job-name=steer-L28
#SBATCH --partition=compute
#SBATCH --gres=gpu:1
#SBATCH --time=2:00:00
#SBATCH --output=logs/20260311_232544_olmo3-7b-think/steering_layer28_slurm.log
#SBATCH --error=logs/20260311_232544_olmo3-7b-think/steering_layer28_slurm.log

cd /home/rkathuria/probes
uv run python scripts/steer.py \
    --run-dir 'logs/20260311_232544_olmo3-7b-think' \
    --layer 28 \
    --probe-type linear \
    --batch \
    --alphas '-20,-10,-5,0,5,10,20' \
    --max-new-tokens 512 \
    --temperature 0.7
