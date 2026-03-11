#!/bin/bash
#SBATCH --job-name=probe-eval
#SBATCH --partition=compute
#SBATCH --gres=gpu:1
#SBATCH --time=6:00:00
#SBATCH --output=logs/20260310_173530_olmo3-7b-think/eval_simple_contrastive_slurm.log
#SBATCH --error=logs/20260310_173530_olmo3-7b-think/eval_simple_contrastive_slurm.log

cd /home/rkathuria/probes
uv run python scripts/eval_probes.py \
    --dataset 'data/simple_contrastive.jsonl' \
    --run-dir 'logs/20260310_173530_olmo3-7b-think' \
    --output 'logs/20260310_173530_olmo3-7b-think/eval_simple_contrastive.json'
