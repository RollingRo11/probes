#!/bin/bash
#SBATCH --job-name=probe-viz
#SBATCH --partition=compute
#SBATCH --gres=gpu:1
#SBATCH --time=4:00:00
#SBATCH --output=logs/20260311_232544_olmo3-7b-think/viz_linear_slurm.log
#SBATCH --error=logs/20260311_232544_olmo3-7b-think/viz_linear_slurm.log

cd /home/rkathuria/probes
uv run python scripts/visualize_tokens.py \
    --dataset data/test_awareness.jsonl --index 1 \
    --run-dir logs/20260311_232544_olmo3-7b-think \
    --probe-type linear
