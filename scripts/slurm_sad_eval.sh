#!/bin/bash
#SBATCH --job-name=sad_eval_all
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --partition=compute
#SBATCH --time=2:00:00
#SBATCH --output=logs/sad_eval_all_%j.out
#SBATCH --error=logs/sad_eval_all_%j.err

set -euo pipefail
cd /home/rkathuria/probes

# Evaluate all probe types on SAD:
#   - mean_diff probes from the eval_awareness run
#   - gradient probes (linear, ema, mlp, attention, multimax, max_rolling_mean) from the answer_token run
# Reuse already-extracted SAD activations.

uv run python scripts/eval_on_sad.py \
    --probes-dir \
        logs/20260310_051655_olmo3-7b-think/results \
        logs/20260310_173530_olmo3-7b-think/results \
    --config configs/all_probes.yaml \
    --output logs/sad_eval_all \
    --activations logs/20260310_051655_olmo3-7b-think/sad_eval/activations
