#!/bin/bash
#SBATCH --job-name=batch-score
#SBATCH --partition=compute
#SBATCH --gres=gpu:1
#SBATCH --time=6:00:00
#SBATCH --output=logs/20260311_232544_olmo3-7b-think/batch_score_sad_slurm.log
#SBATCH --error=logs/20260311_232544_olmo3-7b-think/batch_score_sad_slurm.log

cd /home/rkathuria/probes
uv run python scripts/batch_score.py \
    --dataset 'data/sad.jsonl' \
    --run-dir 'logs/20260311_232544_olmo3-7b-think' \
    --output-dir 'logs/20260311_232544_olmo3-7b-think/viz_sad'
