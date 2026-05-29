"""Transfer evaluation: test if the eval-awareness probe generalizes
to held-out prompt families it was never trained on.

Runs prompts through the model (with chat template), extracts activations
at the specified layer, and scores them with the probe. Reports AUROC
on the new distribution.

This tests whether the probe learned a genuine eval-awareness feature
or just memorized artifacts of its training data.

Usage:
    uv run python scripts/probe_transfer_eval.py \
        --run-dir logs/20260312_215816_olmo3-7b-think \
        --dataset data/transfer_eval_deploy.jsonl \
        --score-layer 26 \
        --slurm
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def load_linear_probe(run_dir: Path, layer: int, hidden_size: int):
    from probes.architectures import build_probe
    ckpt_path = run_dir / "results" / f"layer_{layer}" / "linear" / "probe.pt"
    ckpt = torch.load(ckpt_path, weights_only=False, map_location="cpu")
    probe = build_probe("linear", hidden_size)
    probe.load_state_dict(ckpt["state_dict"])
    probe.eval()
    return probe


def extract_and_score(model, tokenizer, text, probe, score_layer, device):
    """Run text through model with chat template, extract activations, score."""
    obj = model.model.layers
    captured = {}

    def hook(_module, _input, output):
        if isinstance(output, tuple):
            hidden = output[0]
        else:
            hidden = output
        captured["act"] = hidden.detach()

    handle = obj[score_layer].register_forward_hook(hook)
    try:
        # Apply chat template
        messages = [{"role": "user", "content": text}]
        formatted = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )
        inputs = tokenizer(formatted, return_tensors="pt", truncation=True, max_length=2048).to(device)

        with torch.no_grad():
            model(**inputs)

        act = captured["act"].float().cpu()[0]  # (seq_len, hidden)
        with torch.no_grad():
            logits = probe.linear(act).squeeze(-1)
            scores = torch.sigmoid(logits).numpy()

        return float(np.mean(scores)), float(np.max(scores))
    finally:
        handle.remove()


def run(args):
    import yaml
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from sklearn.metrics import roc_auc_score, accuracy_score

    run_dir = Path(args.run_dir)
    cfg = yaml.safe_load((run_dir / "config.yaml").read_text())
    model_id = cfg["model"]["id"]
    hidden_size = cfg["model"]["hidden_size"]

    # Load dataset
    rows = []
    with open(args.dataset) as f:
        for line in f:
            rows.append(json.loads(line))

    print(f"Transfer dataset: {len(rows)} examples from {args.dataset}")

    # Load probe
    print(f"Loading probe (layer {args.score_layer})...")
    probe = load_linear_probe(run_dir, args.score_layer, hidden_size)

    # Optionally also evaluate multiple layers
    layers_to_eval = [args.score_layer]
    if args.all_layers:
        layers_to_eval = list(range(cfg["model"]["num_layers"]))

    probes = {}
    for layer in layers_to_eval:
        probes[layer] = load_linear_probe(run_dir, layer, hidden_size)

    # Load model
    print(f"Loading model {model_id}...")
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.bfloat16, device_map="auto",
    )
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    device = next(model.parameters()).device

    # Score all examples with each probe layer
    results_by_layer = {layer: [] for layer in layers_to_eval}
    labels = []

    for i, row in enumerate(rows):
        labels.append(row["label"])

        for layer in layers_to_eval:
            mean_score, max_score = extract_and_score(
                model, tokenizer, row["text"], probes[layer], layer, device,
            )
            results_by_layer[layer].append({
                "index": i,
                "source": row.get("source", "unknown"),
                "label": row["label"],
                "probe_mean": mean_score,
                "probe_max": max_score,
            })

        if (i + 1) % 20 == 0:
            print(f"  Scored {i+1}/{len(rows)}")

    labels = np.array(labels)

    # Report results
    print(f"\n{'='*60}")
    print("TRANSFER EVALUATION RESULTS")
    print(f"{'='*60}")

    for layer in layers_to_eval:
        scores_mean = np.array([r["probe_mean"] for r in results_by_layer[layer]])
        scores_max = np.array([r["probe_max"] for r in results_by_layer[layer]])

        auroc_mean = roc_auc_score(labels, scores_mean)
        auroc_max = roc_auc_score(labels, scores_max)

        # Per-source breakdown
        sources = set(r["source"] for r in results_by_layer[layer])
        print(f"\n  Layer {layer}:  AUROC (mean): {auroc_mean:.3f}  AUROC (max): {auroc_max:.3f}")

        for source in sorted(sources):
            mask = np.array([r["source"] == source for r in results_by_layer[layer]])
            avg_mean = np.mean(scores_mean[mask])
            avg_max = np.mean(scores_max[mask])
            lbl = labels[mask][0]
            print(f"    {source:20s} (label={lbl}): avg_mean={avg_mean:.3f}, avg_max={avg_max:.3f}")

    # Save results
    output_path = Path(args.output) if args.output else run_dir / "transfer_eval.json"
    output_data = {
        "dataset": args.dataset,
        "n_examples": len(rows),
        "layers_evaluated": layers_to_eval,
        "results_by_layer": {str(k): v for k, v in results_by_layer.items()},
    }
    with open(output_path, "w") as f:
        json.dump(output_data, f, indent=2)
    print(f"\nResults saved to {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", default="logs/20260312_215816_olmo3-7b-think")
    parser.add_argument("--dataset", default="data/transfer_eval_deploy.jsonl")
    parser.add_argument("--score-layer", type=int, default=26)
    parser.add_argument("--all-layers", action="store_true",
                        help="Evaluate all layers, not just score-layer")
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--slurm", action="store_true")
    args = parser.parse_args()

    if args.slurm:
        import subprocess, yaml
        run_dir = Path(args.run_dir)
        cfg = yaml.safe_load((run_dir / "config.yaml").read_text())
        partition = cfg.get("slurm", {}).get("partition", "compute")
        output_flag = f" \\\n    --output {repr(str(args.output))}" if args.output else ""
        all_layers_flag = " \\\n    --all-layers" if args.all_layers else ""
        log_path = run_dir / "transfer_eval_slurm.log"

        script = f"""#!/bin/bash
#SBATCH --job-name=transfer-eval
#SBATCH --partition={partition}
#SBATCH --gres=gpu:1
#SBATCH --time=2:00:00
#SBATCH --output={log_path}
#SBATCH --error={log_path}

cd {PROJECT_ROOT}
uv run python scripts/probe_transfer_eval.py \\
    --run-dir {repr(str(args.run_dir))} \\
    --dataset {repr(str(args.dataset))} \\
    --score-layer {args.score_layer}{all_layers_flag}{output_flag}
"""
        script_path = run_dir / "transfer_eval_job.sh"
        script_path.write_text(script)
        result = subprocess.run(["sbatch", str(script_path)], capture_output=True, text=True)
        print(result.stdout.strip())
        print(f"Log: {log_path}")
        return

    run(args)


if __name__ == "__main__":
    main()
