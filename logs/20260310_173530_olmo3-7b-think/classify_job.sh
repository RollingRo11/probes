#!/bin/bash
#SBATCH --job-name=probe-classify
#SBATCH --partition=compute
#SBATCH --gres=gpu:1
#SBATCH --time=6:00:00
#SBATCH --output=logs/20260310_173530_olmo3-7b-think/classify_slurm.log
#SBATCH --error=logs/20260310_173530_olmo3-7b-think/classify_slurm.log

cd /home/rkathuria/probes
uv run python scripts/classify_prompt.py \
    --prompt "How many r's are in the word strawberry" \
    --run-dir 'logs/20260310_173530_olmo3-7b-think' \
    --output 'logs/20260310_173530_olmo3-7b-think/classify_results.json'
