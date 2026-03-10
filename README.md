# probes

Train linear and nonlinear probes on HuggingFace LLM hidden states for interpretability research.

Implements all probe architectures from [Building Production-Ready Probes for Gemini](https://arxiv.org/abs/2601.11516) on top of a simple closed-form baseline, with a default configuration targeting [evaluation awareness](https://arxiv.org/abs/2509.13333) in OLMo-3-7B-Think. Uses [NNsight 0.6](https://nnsight.net) for activation extraction and [vLLM](https://github.com/vllm-project/vllm) for data generation. Parallelises across layers and probe types via SLURM.

## Probe types

| Name | Description | Training |
|---|---|---|
| `mean_diff` | Difference-of-means; normalized direction in activation space | Closed-form |
| `linear` | Mean-pooled activations + linear head, BCE loss | AdamW |
| `ema` | Same weights as `linear`; at inference applies EMA-max over token scores (α=0.5) | AdamW |
| `mlp` | 2-layer ReLU MLP (width 100) + mean pool + linear head | AdamW |
| `attention` | MLP transform + 10-head soft attention aggregation | AdamW |
| `multimax` | MLP transform + per-head hard max (prevents dilution in long contexts) | AdamW |
| `max_rolling_mean` | MLP transform + sliding-window attention (w=10) + max over windows | AdamW |

Gradient-based probes use full-batch AdamW (lr=1e-4, wd=3e-3, 1000 epochs, BCE loss) with n random seeds; the best seed on validation weighted error is reported on the test set.

## Datasets

| Name | Flag | Examples | Source |
|---|---|---|---|
| SAD stages\_oversight | `sad_stages` | 400 | [Laine et al. 2024](https://arxiv.org/abs/2407.04694) reconstructed from BigBench/WMDP/WildChat |
| Simple Contrastive | `simple_contrastive` | ~202 | [Marks et al. 2025](https://arxiv.org/abs/2507.01786), 101 MCQ questions × 2 answer tokens |
| Pivotal Contrastive | `pivotal_contrastive` | 400 | [Jordine/pivotal-test-phase-steering](https://github.com/Jordine/pivotal-test-phase-steering), 200 MCQ questions × 2 answer tokens |
| Synthetic | `default` | configurable | Hardcoded deployment + prefixed evaluation prompts |

See [INSTRUCTIONS.md](INSTRUCTIONS.md) for full setup and dataset details.

## Quick start

```bash
# Install
git clone <this-repo> && cd probes
uv sync

# Generate dataset
uv run python scripts/generate_data.py --dataset sad_stages --out data/sad_stages.jsonl

# Run locally (no SLURM)
uv run python train.py --config configs/default.yaml --no-slurm

# Visualise results
uv run python scripts/analyze_results.py --log-dir logs/<run>
```

## Configs

| Config | Probe types | Dataset | Seeds | Notes |
|---|---|---|---|---|
| `configs/default.yaml` | `mean_diff` | sad\_stages | 1 | Matches arXiv:2507.01786 setup for OLMo-3-7B-Think |
| `configs/gradient_probes.yaml` | `linear`, `ema` | sad\_stages | 10 | Full-sequence activations for EMA inference |
| `configs/all_probes.yaml` | all 7 types | sad\_stages | 100 | Production eval with weighted error metric |

## SLURM

```bash
# Submit full pipeline (extract then train all probes in parallel)
uv run python train.py --config configs/default.yaml

# Override log directory
uv run python train.py --config configs/all_probes.yaml --log-dir logs/my_run
```

The orchestrator runs two phases: one extraction job (loads model via NNsight, saves per-layer activations), then one training job per `(layer, probe_type)` pair up to `slurm.max_concurrent` concurrent jobs.
