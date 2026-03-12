"""Batch per-token probe scoring for a dataset.

Processes every example through the model once, scores with ALL trained
probe types at every token position, and saves one JSON per example to
an output directory.  Designed for SLURM GPU jobs.

Output structure:
    {output_dir}/
        manifest.json          – index of all examples
        example_0000.json      – per-token scores for example 0
        example_0001.json      – ...

Each example JSON:
    {
        "index": 0,
        "label": 1,
        "num_tokens": 512,
        "token_strs": ["Can", " you", ...],
        "probe_types": ["linear", "ema", "mlp", ...],
        "layers": [0, 1, 2, ...],
        "scores": {
            "linear":   {"0": [0.12, ...], "1": [0.34, ...], ...},
            "ema":      {"0": [0.11, ...], ...},
            ...
        }
    }

Usage:
    uv run python scripts/batch_score.py \
        --dataset data/sad_stages_full.jsonl \
        --run-dir logs/20260311_232544_olmo3-7b-think \
        --output-dir logs/20260311_232544_olmo3-7b-think/viz_sad \
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
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from probes.architectures import build_probe, EMAProbe, MeanDiffProbe, SEQUENCE_PROBES


def _load_probes(results_dir: Path, hidden_size: int) -> dict[str, dict[int, dict]]:
    """Load all trained probes.

    Returns {probe_type: {layer_idx: {"probe": probe_obj, "ckpt": ckpt_data, "threshold": float}}}
    """
    probes: dict[str, dict[int, dict]] = {}

    for layer_dir in sorted(results_dir.iterdir()):
        if not layer_dir.is_dir() or not layer_dir.name.startswith("layer_"):
            continue
        layer_idx = int(layer_dir.name.split("_")[1])

        for probe_dir in sorted(layer_dir.iterdir()):
            if not probe_dir.is_dir():
                continue
            probe_type = probe_dir.name
            ckpt_path = probe_dir / "probe.pt"
            met_path = probe_dir / "metrics.json"
            if not ckpt_path.exists() or not met_path.exists():
                continue

            ckpt = torch.load(ckpt_path, weights_only=False, map_location="cpu")
            metrics = json.loads(met_path.read_text())

            if ckpt["type"] == "mean_diff":
                probe_obj = None
                info = {"direction": ckpt["direction"], "threshold": ckpt["threshold"]}
            else:
                probe_obj = build_probe(probe_type, hidden_size)
                probe_obj.load_state_dict(ckpt["state_dict"])
                probe_obj.eval()
                info = {"probe": probe_obj, "threshold": metrics["threshold"]}

            info["type"] = ckpt["type"]
            probes.setdefault(probe_type, {})[layer_idx] = info

    return probes


def _score_example(
    activations: dict[int, torch.Tensor],
    probes: dict[str, dict[int, dict]],
) -> dict[str, dict[str, list[float]]]:
    """Score one example with all probes at all layers. Returns {probe_type: {layer: [scores]}}."""
    results: dict[str, dict[str, list[float]]] = {}

    for probe_type, layer_probes in probes.items():
        layer_scores: dict[str, list[float]] = {}
        for layer_idx, info in sorted(layer_probes.items()):
            act = activations[layer_idx]  # (1, seq_len, hidden)
            x = act[0]  # (seq_len, hidden)

            if info["type"] == "mean_diff":
                direction = info["direction"]
                raw = x.numpy() @ direction
                lo, hi = raw.min(), raw.max()
                if hi - lo > 1e-8:
                    scores = ((raw - lo) / (hi - lo)).tolist()
                else:
                    scores = [0.5] * len(raw)
            else:
                probe_obj = info["probe"]
                with torch.no_grad():
                    if hasattr(probe_obj, "transform") and hasattr(probe_obj, "head"):
                        y = probe_obj.transform(x)
                        logits = probe_obj.head(y).squeeze(-1)
                    elif hasattr(probe_obj, "linear"):
                        logits = probe_obj.linear(x).squeeze(-1)
                    else:
                        logits = probe_obj(x)
                    scores = torch.sigmoid(logits).numpy().tolist()

            # Round to 3 decimal places to save space.
            layer_scores[str(layer_idx)] = [round(s, 3) for s in scores]
        results[probe_type] = layer_scores

    return results


def batch_score(
    dataset_path: str,
    run_dir: str,
    output_dir: str,
    max_examples: int | None = None,
) -> None:
    """Process all examples, score with all probes, save per-example JSONs."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    run_dir = Path(run_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    import yaml
    cfg = yaml.safe_load((run_dir / "config.yaml").read_text())

    model_id = cfg["model"]["id"]
    num_layers = cfg["model"]["num_layers"]
    hidden_size = cfg["model"]["hidden_size"]
    layer_path = cfg["model"].get("layer_path", "model.layers")
    layer_output_index = cfg["model"].get("layer_output_index")
    max_length = cfg["extraction"].get("max_length", 2048)
    results_dir = run_dir / "results"

    # Load dataset.
    rows = []
    with open(dataset_path) as f:
        for line in f:
            rows.append(json.loads(line))
    if max_examples:
        rows = rows[:max_examples]

    print(f"[batch] {len(rows)} examples from {dataset_path}")

    # Load all probes once.
    print("[batch] Loading probes ...")
    probes = _load_probes(results_dir, hidden_size)
    probe_types = sorted(probes.keys())
    layers = sorted(next(iter(probes.values())).keys())
    print(f"[batch] {len(probe_types)} probe types, {len(layers)} layers")

    # Load model.
    print(f"[batch] Loading model {model_id} ...")
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.bfloat16, device_map="auto",
    )
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Get layer modules and register hooks.
    obj = model
    for attr in layer_path.split("."):
        obj = getattr(obj, attr)
    layers_module = obj

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
        hooks.append(layers_module[i].register_forward_hook(_make_hook(i)))

    if hasattr(model, "hf_device_map"):
        first_device = min(
            set(model.hf_device_map.values()),
            key=lambda d: 0 if d == "cpu" else int(str(d).replace("cuda:", "")),
        )
        device = torch.device(first_device)
    else:
        device = next(model.parameters()).device

    # Process examples one by one.
    manifest = []
    for idx, row in enumerate(tqdm(rows, desc="examples")):
        text = row["text"]
        label = row.get("label")

        enc = tokenizer(
            [text], padding=False, truncation=True,
            max_length=max_length, return_tensors="pt",
        )
        token_ids = enc["input_ids"][0].tolist()
        token_strs = [tokenizer.decode([tid]) for tid in token_ids]
        seq_len = len(token_strs)

        inputs = {k: v.to(device) for k, v in enc.items()}
        with torch.no_grad():
            model(**inputs)

        # Collect activations to CPU.
        activations = {}
        for i in layers:
            activations[i] = captured[i].float().cpu()
        captured.clear()

        # Score with all probes.
        all_scores = _score_example(activations, probes)
        del activations

        # Save per-example JSON.
        example_data = {
            "index": idx,
            "label": label,
            "num_tokens": seq_len,
            "token_strs": token_strs,
            "text_preview": text[:120],
            "probe_types": probe_types,
            "layers": layers,
            "scores": all_scores,
        }
        example_path = output_dir / f"example_{idx:04d}.json"
        with open(example_path, "w") as f:
            json.dump(example_data, f)

        manifest.append({
            "index": idx,
            "label": label,
            "num_tokens": seq_len,
            "text_preview": text[:120],
            "file": example_path.name,
        })

    for h in hooks:
        h.remove()
    del model
    torch.cuda.empty_cache()

    # Save manifest.
    manifest_data = {
        "dataset": dataset_path,
        "model_id": model_id,
        "run_dir": str(run_dir),
        "probe_types": probe_types,
        "layers": layers,
        "n_examples": len(rows),
        "examples": manifest,
    }
    with open(output_dir / "manifest.json", "w") as f:
        json.dump(manifest_data, f, indent=2)

    print(f"\n[batch] Done. {len(rows)} examples → {output_dir}")


def submit_slurm(args: argparse.Namespace) -> None:
    run_dir = Path(args.run_dir)
    import yaml
    cfg = yaml.safe_load((run_dir / "config.yaml").read_text())
    slurm_cfg = cfg.get("slurm", {})

    partition = slurm_cfg.get("partition", "compute")
    timeout = slurm_cfg.get("extract_timeout", "6:00:00")

    dataset_stem = Path(args.dataset).stem
    output_dir = args.output_dir or str(run_dir / f"viz_{dataset_stem}")
    log_path = run_dir / f"batch_score_{dataset_stem}_slurm.log"

    max_flag = f" --max-examples {args.max_examples}" if args.max_examples else ""

    script = f"""#!/bin/bash
#SBATCH --job-name=batch-score
#SBATCH --partition={partition}
#SBATCH --gres=gpu:1
#SBATCH --time={timeout}
#SBATCH --output={log_path}
#SBATCH --error={log_path}

cd {PROJECT_ROOT}
uv run python scripts/batch_score.py \\
    --dataset {repr(str(args.dataset))} \\
    --run-dir {repr(str(args.run_dir))} \\
    --output-dir {repr(output_dir)}{max_flag}
"""

    script_path = run_dir / f"batch_score_{dataset_stem}_job.sh"
    script_path.write_text(script)
    print(f"[batch] Submitting SLURM job ...")
    result = subprocess.run(["sbatch", str(script_path)], capture_output=True, text=True)
    print(result.stdout.strip())
    if result.returncode != 0:
        print(f"SLURM error: {result.stderr}", file=sys.stderr)
        sys.exit(1)
    print(f"Log: {log_path}")
    print(f"Output: {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Batch per-token probe scoring")
    parser.add_argument("--dataset", required=True, help="Path to JSONL dataset")
    parser.add_argument("--run-dir", default="logs/20260311_232544_olmo3-7b-think")
    parser.add_argument("--output-dir", help="Output directory for per-example JSONs")
    parser.add_argument("--max-examples", type=int, help="Limit number of examples")
    parser.add_argument("--slurm", action="store_true", help="Submit as SLURM job")
    args = parser.parse_args()

    if args.slurm:
        submit_slurm(args)
    else:
        batch_score(args.dataset, args.run_dir, args.output_dir, args.max_examples)


if __name__ == "__main__":
    main()
