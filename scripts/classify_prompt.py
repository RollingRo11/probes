"""Classify a prompt for eval-awareness using all trained probes.

Loads the model, extracts activations for the given prompt at every layer,
then runs all trained probes (from a previous training run) and reports
per-layer, per-probe predictions.

Usage (local, requires GPU):
    uv run python scripts/classify_prompt.py \
        --prompt "How many r's are in the word strawberry" \
        --run-dir logs/20260310_173530_olmo3-7b-think

Usage (SLURM):
    uv run python scripts/classify_prompt.py \
        --prompt "How many r's are in the word strawberry" \
        --run-dir logs/20260310_173530_olmo3-7b-think \
        --slurm
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Add project root to path.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from probes.architectures import build_probe, SEQUENCE_PROBES


def extract_single_prompt(
    model_id: str,
    prompt: str,
    num_layers: int,
    layer_path: str = "model.layers",
    layer_output_index: int | None = None,
    max_length: int = 2048,
) -> dict[int, torch.Tensor]:
    """Extract last-token activations for a single prompt at every layer.

    Returns dict mapping layer_idx -> (1, hidden_size) tensor.
    """
    print(f"[classify] Loading model {model_id} ...")
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.bfloat16, device_map="auto",
    )
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Get layer list and register hooks.
    obj = model
    for attr in layer_path.split("."):
        obj = getattr(obj, attr)
    layers = obj

    captured: dict[int, torch.Tensor] = {}

    def _make_hook(idx: int):
        def hook(_module, _input, output):
            if layer_output_index is not None:
                hidden = output[layer_output_index]
            else:
                hidden = output if isinstance(output, torch.Tensor) else output[0]
            captured[idx] = hidden.detach()
        return hook

    hooks = []
    for i in range(num_layers):
        hooks.append(layers[i].register_forward_hook(_make_hook(i)))

    enc = tokenizer(
        [prompt], padding=True, truncation=True,
        max_length=max_length, return_tensors="pt",
    )
    seq_len = enc["attention_mask"].sum(dim=1).long().item()
    last_pos = seq_len - 1

    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in enc.items()}

    print(f"[classify] Running forward pass ({seq_len} tokens) ...")
    with torch.no_grad():
        model(**inputs)

    for h in hooks:
        h.remove()

    # Extract last-token activation per layer.
    activations = {}
    for i in range(num_layers):
        act = captured[i].float().cpu()  # (1, seq_len, hidden)
        last_act = act[:, last_pos, :]             # (1, hidden)
        activations[i] = {
            "last": last_act,
            "all": last_act.unsqueeze(1),          # (1, 1, hidden)
        }

    del model
    torch.cuda.empty_cache()
    return activations


def load_and_run_probe(
    probe_type: str,
    checkpoint_path: Path,
    metrics_path: Path,
    activation: dict[str, torch.Tensor],
    hidden_size: int,
) -> dict:
    """Load a trained probe and classify the activation."""
    ckpt = torch.load(checkpoint_path, weights_only=False, map_location="cpu")
    metrics = json.loads(metrics_path.read_text())
    threshold = metrics["threshold"]

    if ckpt["type"] == "mean_diff":
        direction = ckpt["direction"]
        thresh = ckpt["threshold"]
        act_np = activation["last"].numpy()  # (1, hidden)
        score = float(act_np @ direction)
        pred = int(score >= thresh)
        return {
            "score": score,
            "threshold": thresh,
            "prediction": pred,
            "label": "EVAL" if pred == 1 else "DEPLOY",
        }

    # Gradient-based probe: rebuild architecture and load weights.
    probe = build_probe(probe_type, hidden_size)
    probe.load_state_dict(ckpt["state_dict"])
    probe.eval()

    # Use sequence activations for probes that need them, else last-token.
    if probe_type in SEQUENCE_PROBES:
        x = activation["all"]
    else:
        x = activation["last"]

    with torch.no_grad():
        from probes.architectures import EMAProbe
        if isinstance(probe, EMAProbe) and x.dim() == 3:
            logit = probe.forward_ema(x)
        else:
            logit = probe(x)
        prob = torch.sigmoid(logit).item()

    pred = int(prob >= threshold)
    return {
        "score": prob,
        "threshold": threshold,
        "prediction": pred,
        "label": "EVAL" if pred == 1 else "DEPLOY",
    }


def classify_prompt(prompt: str, run_dir: str, output_path: str | None = None) -> dict:
    """Run all probes from a training run on a single prompt."""
    run_dir = Path(run_dir)
    config_path = run_dir / "config.yaml"

    import yaml
    cfg = yaml.safe_load(config_path.read_text())

    model_id = cfg["model"]["id"]
    num_layers = cfg["model"]["num_layers"]
    hidden_size = cfg["model"]["hidden_size"]
    layer_path = cfg["model"].get("layer_path", "model.layers")
    layer_output_index = cfg["model"].get("layer_output_index")

    results_dir = run_dir / "results"

    # Extract activations.
    activations = extract_single_prompt(
        model_id=model_id,
        prompt=prompt,
        num_layers=num_layers,
        layer_path=layer_path,
        layer_output_index=layer_output_index,
    )

    # Discover and run all trained probes.
    results = {"prompt": prompt, "model": model_id, "probes": []}
    vote_eval = 0
    vote_deploy = 0

    for layer_dir in sorted(results_dir.iterdir()):
        if not layer_dir.is_dir() or not layer_dir.name.startswith("layer_"):
            continue
        layer_idx = int(layer_dir.name.split("_")[1])

        for probe_dir in sorted(layer_dir.iterdir()):
            if not probe_dir.is_dir():
                continue
            probe_type = probe_dir.name
            ckpt_path = probe_dir / "probe.pt"
            metrics_path = probe_dir / "metrics.json"

            if not ckpt_path.exists() or not metrics_path.exists():
                continue

            try:
                result = load_and_run_probe(
                    probe_type=probe_type,
                    checkpoint_path=ckpt_path,
                    metrics_path=metrics_path,
                    activation=activations[layer_idx],
                    hidden_size=hidden_size,
                )
                result["layer"] = layer_idx
                result["probe_type"] = probe_type
                results["probes"].append(result)

                if result["prediction"] == 1:
                    vote_eval += 1
                else:
                    vote_deploy += 1
            except Exception as e:
                print(f"[classify] WARN: {probe_type} layer {layer_idx}: {e}")

    total = vote_eval + vote_deploy
    results["summary"] = {
        "vote_eval": vote_eval,
        "vote_deploy": vote_deploy,
        "total_probes": total,
        "eval_fraction": vote_eval / total if total > 0 else 0,
        "verdict": "EVAL-AWARE" if vote_eval > vote_deploy else "NOT EVAL-AWARE",
    }

    # Print summary.
    print("\n" + "=" * 70)
    print(f"PROMPT: {prompt}")
    print(f"MODEL:  {model_id}")
    print("=" * 70)

    # Per-layer summary.
    from collections import defaultdict
    layer_votes = defaultdict(lambda: {"eval": 0, "deploy": 0})
    for p in results["probes"]:
        if p["prediction"] == 1:
            layer_votes[p["layer"]]["eval"] += 1
        else:
            layer_votes[p["layer"]]["deploy"] += 1

    print(f"\n{'Layer':>6}  {'EVAL':>5}  {'DEPLOY':>7}  {'Verdict':>12}")
    print("-" * 40)
    for layer in sorted(layer_votes):
        v = layer_votes[layer]
        verdict = "EVAL" if v["eval"] > v["deploy"] else "DEPLOY"
        print(f"{layer:>6}  {v['eval']:>5}  {v['deploy']:>7}  {verdict:>12}")

    print("-" * 40)
    s = results["summary"]
    print(f"{'TOTAL':>6}  {s['vote_eval']:>5}  {s['vote_deploy']:>7}  {s['verdict']:>12}")
    print(f"\nEval fraction: {s['eval_fraction']:.2%}")
    print(f"VERDICT: {s['verdict']}")
    print("=" * 70)

    # Save results.
    if output_path:
        out = Path(output_path)
    else:
        out = run_dir / "classify_results.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nFull results saved to: {out}")

    return results


def submit_slurm(args: argparse.Namespace) -> None:
    """Submit the classification as a SLURM job."""
    run_dir = Path(args.run_dir)
    config_path = run_dir / "config.yaml"

    import yaml
    cfg = yaml.safe_load(config_path.read_text())
    slurm_cfg = cfg.get("slurm", {})

    partition = slurm_cfg.get("partition", "compute")
    timeout = slurm_cfg.get("extract_timeout", "4:00:00")
    job_name = "probe-classify"

    output_path = args.output or str(run_dir / "classify_results.json")
    log_path = run_dir / "classify_slurm.log"

    script = f"""#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --partition={partition}
#SBATCH --gres=gpu:1
#SBATCH --time={timeout}
#SBATCH --output={log_path}
#SBATCH --error={log_path}

cd {PROJECT_ROOT}
uv run python scripts/classify_prompt.py \\
    --prompt {repr(args.prompt)} \\
    --run-dir {repr(str(args.run_dir))} \\
    --output {repr(output_path)}
"""

    script_path = run_dir / "classify_job.sh"
    script_path.write_text(script)
    print(f"[classify] Submitting SLURM job ...")
    result = subprocess.run(
        ["sbatch", str(script_path)],
        capture_output=True, text=True,
    )
    print(result.stdout.strip())
    if result.returncode != 0:
        print(f"SLURM error: {result.stderr}", file=sys.stderr)
        sys.exit(1)
    print(f"Log: {log_path}")


def main():
    parser = argparse.ArgumentParser(description="Classify a prompt for eval-awareness")
    parser.add_argument("--prompt", required=True, help="The prompt to classify")
    parser.add_argument(
        "--run-dir",
        default="logs/20260310_173530_olmo3-7b-think",
        help="Path to training run directory with trained probes",
    )
    parser.add_argument("--output", help="Output JSON path (default: <run-dir>/classify_results.json)")
    parser.add_argument("--slurm", action="store_true", help="Submit as SLURM job instead of running locally")
    args = parser.parse_args()

    if args.slurm:
        submit_slurm(args)
    else:
        classify_prompt(args.prompt, args.run_dir, args.output)


if __name__ == "__main__":
    main()
