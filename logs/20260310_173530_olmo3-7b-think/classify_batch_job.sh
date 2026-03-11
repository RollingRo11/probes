#!/bin/bash
#SBATCH --job-name=probe-classify-batch
#SBATCH --partition=compute
#SBATCH --gres=gpu:1
#SBATCH --time=6:00:00
#SBATCH --output=logs/20260310_173530_olmo3-7b-think/classify_batch_slurm.log
#SBATCH --error=logs/20260310_173530_olmo3-7b-think/classify_batch_slurm.log

cd /home/rkathuria/probes
uv run python scripts/classify_batch.py \
    --prompts-file 'data/eval_test_prompts.json' \
    --run-dir 'logs/20260310_173530_olmo3-7b-think' \
    --output 'logs/20260310_173530_olmo3-7b-think/classify_batch_results.json' \
    --no-cot
