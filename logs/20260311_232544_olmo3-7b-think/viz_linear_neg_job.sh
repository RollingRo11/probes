#!/bin/bash
#SBATCH --job-name=probe-viz-neg
#SBATCH --partition=compute
#SBATCH --gres=gpu:1
#SBATCH --time=4:00:00
#SBATCH --output=logs/20260311_232544_olmo3-7b-think/viz_linear_neg_slurm.log
#SBATCH --error=logs/20260311_232544_olmo3-7b-think/viz_linear_neg_slurm.log

cd /home/rkathuria/probes
uv run python scripts/visualize_tokens.py \
    --dataset data/test_awareness.jsonl --index 0 \
    --run-dir logs/20260311_232544_olmo3-7b-think \
    --probe-type linear \
    --output logs/20260311_232544_olmo3-7b-think/viz_linear_tokens_neg.html
