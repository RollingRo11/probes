"""Generate completions and probe hidden states at every token.

For each prompt, captures:
  1. Prompt-time scores: probe scores at the last token of the prompt
  2. Generation-time trajectory: probe scores at each generated token

This reveals whether the model's evaluation-awareness belief is static
(fully determined by the prompt) or dynamic (evolves during reasoning).

Usage:
    uv run python scripts/generate_and_probe.py \
        --run-dir logs/20260326_210903_olmo3-7b-think \
        --dataset data/sad_stages.jsonl \
        --layers 5,10,15,20,25 \
        --probe-type linear \
        --max-examples 20 \
        --max-new-tokens 128

    # SLURM:
    uv run python scripts/generate_and_probe.py \
        --run-dir logs/20260326_210903_olmo3-7b-think \
        --dataset data/sad_stages.jsonl \
        --slurm
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def load_probes(run_dir: Path, layers: list[int], probe_type: str, hidden_size: int):
    """Load trained probes for specified layers. Returns {layer: (probe, threshold)}."""
    from probes.architectures import MeanDiffProbe, build_probe

    probes = {}
    for layer in layers:
        ckpt_path = run_dir / "results" / f"layer_{layer:02d}" / probe_type / "probe.pt"
        metrics_path = run_dir / "results" / f"layer_{layer:02d}" / probe_type / "metrics.json"
        if not ckpt_path.exists():
            print(f"  Skipping layer {layer}: no checkpoint")
            continue

        ckpt = torch.load(ckpt_path, weights_only=False, map_location="cpu")
        metrics = json.loads(metrics_path.read_text()) if metrics_path.exists() else {}
        threshold = metrics.get("threshold", 0.5)

        if ckpt["type"] == "mean_diff":
            probe = MeanDiffProbe()
            probe.direction = ckpt["direction"]
            probe.threshold = ckpt["threshold"]
        else:
            probe = build_probe(probe_type, hidden_size)
            probe.load_state_dict(ckpt["state_dict"])
            probe.eval()

        probes[layer] = (probe, threshold)

    return probes


def score_activations(probe, probe_type: str, act: torch.Tensor) -> float:
    """Score a single (hidden_size,) activation vector with a probe."""
    from probes.architectures import MeanDiffProbe

    if isinstance(probe, MeanDiffProbe):
        score = float(act.numpy() @ probe.direction)
        return score
    else:
        with torch.no_grad():
            logit = probe(act.unsqueeze(0))
            return float(torch.sigmoid(logit).item())


def run(args):
    import yaml
    from transformers import AutoModelForCausalLM, AutoTokenizer

    run_dir = Path(args.run_dir)
    cfg = yaml.safe_load((run_dir / "config.yaml").read_text())
    model_id = cfg["model"]["id"]
    hidden_size = cfg["model"]["hidden_size"]
    num_layers = cfg["model"]["num_layers"]

    # Parse layers.
    if args.layers:
        layers = [int(x) for x in args.layers.split(",")]
    else:
        # Default: ~8 evenly spaced layers
        n = num_layers
        layers = [int(i * n / 8) for i in range(8)] + [n - 1]
        layers = sorted(set(layers))

    probe_type = args.probe_type

    # Load dataset.
    rows = []
    with open(args.dataset) as f:
        for line in f:
            rows.append(json.loads(line))
    if args.max_examples:
        rows = rows[:args.max_examples]

    print(f"Model: {model_id}")
    print(f"Layers: {layers}")
    print(f"Probe type: {probe_type}")
    print(f"Dataset: {len(rows)} examples from {args.dataset}")
    print(f"Max new tokens: {args.max_new_tokens}")

    # Load probes.
    print("\nLoading probes...")
    probes = load_probes(run_dir, layers, probe_type, hidden_size)
    active_layers = sorted(probes.keys())
    print(f"  Loaded probes for layers: {active_layers}")

    if not probes:
        print("ERROR: No probes loaded.")
        sys.exit(1)

    # Load model.
    print(f"\nLoading model {model_id}...")
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.bfloat16, device_map="auto",
    )
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Determine device.
    if hasattr(model, "hf_device_map"):
        first_device = min(
            set(model.hf_device_map.values()),
            key=lambda d: 0 if d == "cpu" else int(str(d).replace("cuda:", "")),
        )
        device = torch.device(first_device)
    else:
        device = next(model.parameters()).device

    # Get layer module list.
    obj = model
    for attr in cfg["model"].get("layer_path", "model.layers").split("."):
        obj = getattr(obj, attr)
    layer_modules = obj

    # Process each example.
    all_results = []

    for idx, row in enumerate(tqdm(rows, desc="examples")):
        text = row["text"]
        label = row.get("label")
        source = row.get("source", "")

        # Format with chat template.
        messages = [{"role": "user", "content": text}]
        formatted = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )

        input_ids = tokenizer(
            formatted, return_tensors="pt", truncation=True,
            max_length=args.max_length,
        ).input_ids.to(device)
        prompt_len = input_ids.shape[1]

        # --- Prompt-time scores ---
        captured = {}

        def make_hook(layer_idx):
            def hook(_module, _input, output):
                if isinstance(output, tuple):
                    hidden = output[0]
                else:
                    hidden = output
                captured[layer_idx] = hidden.detach()
            return hook

        handles = []
        for li in active_layers:
            handles.append(layer_modules[li].register_forward_hook(make_hook(li)))

        with torch.no_grad():
            model(input_ids)

        prompt_scores = {}
        for li in active_layers:
            act = captured[li].float().cpu()[0, -1, :]  # last token
            probe, threshold = probes[li]
            prompt_scores[li] = score_activations(probe, probe_type, act)

        for h in handles:
            h.remove()
        captured.clear()

        # --- Generation-time trajectory ---
        # Generate token by token, probing at each step.
        gen_ids = input_ids.clone()
        trajectories = {li: [] for li in active_layers}
        generated_tokens = []

        for step in range(args.max_new_tokens):
            handles = []
            for li in active_layers:
                handles.append(layer_modules[li].register_forward_hook(make_hook(li)))

            with torch.no_grad():
                outputs = model(gen_ids)

            # Score at the last (newly generated) position.
            for li in active_layers:
                act = captured[li].float().cpu()[0, -1, :]
                probe, threshold = probes[li]
                score = score_activations(probe, probe_type, act)
                trajectories[li].append(score)

            for h in handles:
                h.remove()
            captured.clear()

            # Sample next token (greedy for reproducibility).
            next_logits = outputs.logits[0, -1, :]
            if args.temperature == 0:
                next_token = next_logits.argmax(dim=-1, keepdim=True)
            else:
                probs = torch.softmax(next_logits / args.temperature, dim=-1)
                next_token = torch.multinomial(probs, 1)

            gen_ids = torch.cat([gen_ids, next_token.unsqueeze(0)], dim=1)
            token_str = tokenizer.decode(next_token[0], skip_special_tokens=False)
            generated_tokens.append(token_str)

            # Stop on EOS.
            if next_token.item() == tokenizer.eos_token_id:
                break

        completion = tokenizer.decode(
            gen_ids[0, prompt_len:], skip_special_tokens=True,
        )

        result = {
            "index": idx,
            "prompt": text,
            "label": label,
            "source": source,
            "completion": completion,
            "prompt_len_tokens": prompt_len,
            "completion_len_tokens": len(generated_tokens),
            "prompt_scores": {str(k): round(v, 6) for k, v in prompt_scores.items()},
            "trajectories": {
                str(k): [round(v, 6) for v in vals]
                for k, vals in trajectories.items()
            },
            "generated_tokens": generated_tokens,
        }
        all_results.append(result)

        if (idx + 1) % 5 == 0:
            # Periodic status.
            avg_prompt = np.mean([v for v in prompt_scores.values()])
            avg_final = np.mean([vals[-1] for vals in trajectories.values() if vals])
            print(
                f"  [{idx+1}/{len(rows)}] label={label} "
                f"prompt_score={avg_prompt:.3f} final_score={avg_final:.3f} "
                f"gen_len={len(generated_tokens)}"
            )

    # Save.
    output_dir = Path(args.output_dir) if args.output_dir else run_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"gen_probe_{Path(args.dataset).stem}.json"

    output_data = {
        "model": model_id,
        "run_dir": str(run_dir),
        "dataset": args.dataset,
        "probe_type": probe_type,
        "layers": active_layers,
        "n_examples": len(all_results),
        "max_new_tokens": args.max_new_tokens,
        "temperature": args.temperature,
        "results": all_results,
    }

    with open(output_path, "w") as f:
        json.dump(output_data, f, indent=2)
    print(f"\nResults saved to {output_path}")

    # Summary.
    print(f"\n{'Layer':>6} {'Avg Prompt':>12} {'Avg Final':>12} {'Avg Δ':>10}")
    print("-" * 45)
    for li in active_layers:
        ps = [r["prompt_scores"][str(li)] for r in all_results]
        fs = [r["trajectories"][str(li)][-1] for r in all_results if r["trajectories"][str(li)]]
        avg_p = np.mean(ps)
        avg_f = np.mean(fs) if fs else float("nan")
        delta = avg_f - avg_p
        print(f"{li:>6} {avg_p:>12.4f} {avg_f:>12.4f} {delta:>+10.4f}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate completions and probe hidden states at every token",
    )
    parser.add_argument("--run-dir", required=True, help="Probe run directory")
    parser.add_argument("--dataset", required=True, help="JSONL dataset")
    parser.add_argument("--layers", default=None,
                        help="Comma-separated layer indices (default: ~8 evenly spaced)")
    parser.add_argument("--probe-type", default="linear",
                        help="Probe type to use (default: linear)")
    parser.add_argument("--max-examples", type=int, default=20)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--temperature", type=float, default=0.0,
                        help="Generation temperature (0=greedy)")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--slurm", action="store_true")
    args = parser.parse_args()

    if args.slurm:
        import subprocess
        import yaml

        run_dir = Path(args.run_dir)
        cfg = yaml.safe_load((run_dir / "config.yaml").read_text())
        partition = cfg.get("slurm", {}).get("partition", "compute")

        output_dir = args.output_dir or str(run_dir)
        log_path = Path(output_dir) / f"gen_probe_{Path(args.dataset).stem}_slurm.log"
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        layers_flag = f" \\\n    --layers '{args.layers}'" if args.layers else ""
        output_flag = f" \\\n    --output-dir '{args.output_dir}'" if args.output_dir else ""

        script = f"""#!/bin/bash
#SBATCH --job-name=gen-probe
#SBATCH --partition={partition}
#SBATCH --gres=gpu:1
#SBATCH --time=6:00:00
#SBATCH --output={log_path}
#SBATCH --error={log_path}

cd {PROJECT_ROOT}
uv run python scripts/generate_and_probe.py \\
    --run-dir '{args.run_dir}' \\
    --dataset '{args.dataset}' \\
    --probe-type '{args.probe_type}' \\
    --max-examples {args.max_examples} \\
    --max-new-tokens {args.max_new_tokens} \\
    --temperature {args.temperature}{layers_flag}{output_flag}
"""
        script_path = Path(output_dir) / f"gen_probe_{Path(args.dataset).stem}_job.sh"
        script_path.write_text(script)
        result = subprocess.run(["sbatch", str(script_path)], capture_output=True, text=True)
        print(result.stdout.strip())
        print(f"Log: {log_path}")
        return

    run(args)


if __name__ == "__main__":
    main()
