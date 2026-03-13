#!/bin/bash
#SBATCH --job-name=steer-L24
#SBATCH --partition=compute
#SBATCH --gres=gpu:1
#SBATCH --time=2:00:00
#SBATCH --output=logs/20260312_215816_olmo3-7b-think/steering_layer24_slurm.log
#SBATCH --error=logs/20260312_215816_olmo3-7b-think/steering_layer24_slurm.log

cd /home/rkathuria/probes
uv run python scripts/steer.py \
    --run-dir 'logs/20260312_215816_olmo3-7b-think' \
    --layer 24 \
    --probe-type linear \
    --batch \
    --alphas='-20,-10,-5,0,5,10,20' \
    --max-new-tokens 512 \
    --temperature 0.7
