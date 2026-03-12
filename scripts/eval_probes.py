"""Evaluate trained probes on a new dataset.

Extracts activations from the dataset using the same token_strategy as
training, then loads all trained probes and computes metrics.

Usage (SLURM):
    uv run python scripts/eval_probes.py \
        --dataset data/simple_contrastive.jsonl \
        --run-dir logs/20260310_173530_olmo3-7b-think \
        --slurm
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from probes.architectures import EMAProbe, build_probe, SEQUENCE_PROBES
from probes.extraction import extract_activations, load_activations, load_metadata
from probes.metrics import auroc, evaluate_probe, youden_threshold


def eval_probes(dataset_path: str, run_dir: str, output_path: str | None = None) -> dict:
    run_dir = Path(run_dir)
    import yaml
    cfg = yaml.safe_load((run_dir / "config.yaml").read_text())

    model_id = cfg["model"]["id"]
    num_layers = cfg["model"]["num_layers"]
    hidden_size = cfg["model"]["hidden_size"]
    layer_path = cfg["model"].get("layer_path", "model.layers")
    layer_output_index = cfg["model"].get("layer_output_index")
    token_strategy = cfg["extraction"]["token_strategy"]
    batch_size = cfg["extraction"].get("batch_size", 4)
    max_length = cfg["extraction"].get("max_length", 128)
    results_dir = run_dir / "results"

    # Load dataset.
    rows = []
    with open(dataset_path) as f:
        for line in f:
            rows.append(json.loads(line))

    texts = [r["text"] for r in rows]
    labels = [r["label"] for r in rows]
    answer_token_chars = None
    evidence_char_spans = None

    # Determine eval extraction strategy: use the training token_strategy
    # if the eval dataset has the required fields, otherwise fall back to "last".
    eval_token_strategy = token_strategy
    if token_strategy == "answer_token":
        if all("answer_token_char" in r for r in rows):
            answer_token_chars = [r["answer_token_char"] for r in rows]
        else:
            print(f"[eval] Eval dataset lacks answer_token_char; falling back to 'last'.")
            eval_token_strategy = "last"
    elif token_strategy == "evidence_spans":
        if all("evidence_char_spans" in r for r in rows):
            evidence_char_spans = [r["evidence_char_spans"] for r in rows]
        else:
            print(f"[eval] Eval dataset lacks evidence_char_spans; falling back to 'last'.")
            eval_token_strategy = "last"

    # Extract activations.
    eval_act_dir = run_dir / "eval_activations" / Path(dataset_path).stem
    extract_activations(
        model_id=model_id,
        texts=texts,
        labels=labels,
        output_dir=eval_act_dir,
        num_layers=num_layers,
        layer_path=layer_path,
        layer_output_index=layer_output_index,
        token_strategy=eval_token_strategy,
        batch_size=batch_size,
        max_length=max_length,
        skip_if_exists=False,
        answer_token_chars=answer_token_chars,
        evidence_char_spans=evidence_char_spans,
    )

    # Discover all trained probes.
    probe_specs = []
    for layer_dir in sorted(results_dir.iterdir()):
        if not layer_dir.is_dir() or not layer_dir.name.startswith("layer_"):
            continue
        layer_idx = int(layer_dir.name.split("_")[1])
        for probe_dir in sorted(layer_dir.iterdir()):
            if not probe_dir.is_dir():
                continue
            ckpt = probe_dir / "probe.pt"
            met = probe_dir / "metrics.json"
            if ckpt.exists() and met.exists():
                probe_specs.append((layer_idx, probe_dir.name, ckpt, met))

    print(f"\n[eval] Evaluating {len(probe_specs)} probes on {len(rows)} examples from {dataset_path}\n")

    all_labels = torch.load(eval_act_dir / "labels.pt", weights_only=True).numpy()
    all_results = []

    for layer_idx, probe_type, ckpt_path, metrics_path in probe_specs:
        acts = torch.load(eval_act_dir / f"layer_{layer_idx:02d}.pt", weights_only=True)

        ckpt = torch.load(ckpt_path, weights_only=False, map_location="cpu")
        train_metrics = json.loads(metrics_path.read_text())

        if ckpt["type"] == "mean_diff":
            direction = ckpt["direction"]
            threshold = ckpt["threshold"]
            X = acts.numpy()
            if X.ndim == 3:
                X = X.mean(axis=1)
            scores = X @ direction
        else:
            probe = build_probe(probe_type, hidden_size)
            probe.load_state_dict(ckpt["state_dict"])
            probe.eval()

            x = acts
            if probe_type not in SEQUENCE_PROBES and x.dim() == 3:
                x = x.mean(dim=1)

            with torch.no_grad():
                if isinstance(probe, EMAProbe) and x.dim() == 3:
                    logits = probe.forward_ema(x)
                else:
                    logits = probe(x)
                scores = torch.sigmoid(logits).numpy()
            threshold = train_metrics["threshold"]

        auc = auroc(all_labels, scores)
        preds = (scores >= threshold).astype(int)
        acc = (preds == all_labels).mean()
        tp = ((preds == 1) & (all_labels == 1)).sum()
        fn = ((preds == 0) & (all_labels == 1)).sum()
        fp = ((preds == 1) & (all_labels == 0)).sum()
        tn = ((preds == 0) & (all_labels == 0)).sum()
        fnr = fn / (tp + fn) if (tp + fn) > 0 else 0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0

        result = {
            "layer": layer_idx,
            "probe_type": probe_type,
            "auroc": float(auc),
            "accuracy": float(acc),
            "threshold": float(threshold),
            "fnr": float(fnr),
            "fpr": float(fpr),
            "tp": int(tp), "fn": int(fn), "fp": int(fp), "tn": int(tn),
            "train_auroc": train_metrics.get("auroc"),
        }
        all_results.append(result)

    # Print summary by probe type.
    probe_types = sorted(set(r["probe_type"] for r in all_results))

    print("=" * 90)
    print(f"EVAL DATASET: {dataset_path} ({len(rows)} examples)")
    print(f"PROBES FROM:  {run_dir}")
    print("=" * 90)

    print(f"\n{'Probe Type':>18}  {'Avg AUROC':>10}  {'Avg Acc':>8}  {'Avg FNR':>8}  {'Avg FPR':>8}  {'Layers':>6}")
    print("-" * 70)
    for pt in probe_types:
        pt_results = [r for r in all_results if r["probe_type"] == pt]
        avg_auc = np.mean([r["auroc"] for r in pt_results])
        avg_acc = np.mean([r["accuracy"] for r in pt_results])
        avg_fnr = np.mean([r["fnr"] for r in pt_results])
        avg_fpr = np.mean([r["fpr"] for r in pt_results])
        print(f"{pt:>18}  {avg_auc:>10.4f}  {avg_acc:>7.1%}  {avg_fnr:>7.1%}  {avg_fpr:>7.1%}  {len(pt_results):>6}")
    print()

    # Per-layer breakdown.
    print(f"{'Layer':>6}", end="")
    for pt in probe_types:
        print(f"  {pt:>10}", end="")
    print("   (AUROC)")
    print("-" * (8 + 12 * len(probe_types)))

    layers = sorted(set(r["layer"] for r in all_results))
    for layer in layers:
        print(f"{layer:>6}", end="")
        for pt in probe_types:
            match = [r for r in all_results if r["layer"] == layer and r["probe_type"] == pt]
            if match:
                print(f"  {match[0]['auroc']:>10.4f}", end="")
            else:
                print(f"  {'---':>10}", end="")
        print()

    print("=" * 90)

    output = {
        "eval_dataset": dataset_path,
        "run_dir": str(run_dir),
        "n_examples": len(rows),
        "results": all_results,
    }

    out_path = Path(output_path) if output_path else run_dir / "eval_simple_contrastive.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nResults saved to: {out_path}")

    return output


def submit_slurm(args: argparse.Namespace) -> None:
    run_dir = Path(args.run_dir)
    import yaml
    cfg = yaml.safe_load((run_dir / "config.yaml").read_text())
    slurm_cfg = cfg.get("slurm", {})

    partition = slurm_cfg.get("partition", "compute")
    timeout = slurm_cfg.get("extract_timeout", "4:00:00")

    output_path = args.output or str(run_dir / f"eval_{Path(args.dataset).stem}.json")
    log_path = run_dir / f"eval_{Path(args.dataset).stem}_slurm.log"

    script = f"""#!/bin/bash
#SBATCH --job-name=probe-eval
#SBATCH --partition={partition}
#SBATCH --gres=gpu:1
#SBATCH --time={timeout}
#SBATCH --output={log_path}
#SBATCH --error={log_path}

cd {PROJECT_ROOT}
uv run python scripts/eval_probes.py \\
    --dataset {repr(str(args.dataset))} \\
    --run-dir {repr(str(args.run_dir))} \\
    --output {repr(output_path)}
"""

    script_path = run_dir / f"eval_{Path(args.dataset).stem}_job.sh"
    script_path.write_text(script)
    print(f"[eval] Submitting SLURM job ...")
    result = subprocess.run(["sbatch", str(script_path)], capture_output=True, text=True)
    print(result.stdout.strip())
    if result.returncode != 0:
        print(f"SLURM error: {result.stderr}", file=sys.stderr)
        sys.exit(1)
    print(f"Log: {log_path}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate trained probes on a new dataset")
    parser.add_argument("--dataset", required=True, help="Path to JSONL dataset")
    parser.add_argument("--run-dir", default="logs/20260310_173530_olmo3-7b-think")
    parser.add_argument("--output", help="Output JSON path")
    parser.add_argument("--slurm", action="store_true", help="Submit as SLURM job")
    args = parser.parse_args()

    if args.slurm:
        submit_slurm(args)
    else:
        eval_probes(args.dataset, args.run_dir, args.output)


if __name__ == "__main__":
    main()
