"""Per-token probe heatmap visualization.

Runs a single text through the model, applies a trained probe at every
token position independently, and generates an interactive Plotly HTML
heatmap (x = token, y = layer, color = probe score).

Usage (SLURM):
    uv run python scripts/visualize_tokens.py \
        --text "The model said: I know this is an eval ..." \
        --run-dir logs/20260311_232544_olmo3-7b-think \
        --probe-type linear \
        --slurm

Usage (from JSONL, by index):
    uv run python scripts/visualize_tokens.py \
        --dataset data/test_awareness.jsonl --index 42 \
        --run-dir logs/20260311_232544_olmo3-7b-think \
        --probe-type linear
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

from probes.architectures import build_probe, EMAProbe, MeanDiffProbe, SEQUENCE_PROBES


def extract_all_positions(
    model_id: str,
    text: str,
    num_layers: int,
    layer_path: str = "model.layers",
    layer_output_index: int | None = None,
    max_length: int = 2048,
) -> tuple[dict[int, torch.Tensor], list[str]]:
    """Run forward pass, return per-layer (1, seq_len, hidden) activations and token strings."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"[viz] Loading model {model_id} ...")
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

    enc = tokenizer(
        [text], padding=False, truncation=True,
        max_length=max_length, return_tensors="pt",
    )
    seq_len = enc["input_ids"].shape[1]

    # Token strings for display.
    token_ids = enc["input_ids"][0].tolist()
    token_strs = [tokenizer.decode([tid]) for tid in token_ids]

    if hasattr(model, "hf_device_map"):
        first_device = min(
            set(model.hf_device_map.values()),
            key=lambda d: 0 if d == "cpu" else int(str(d).replace("cuda:", "")),
        )
        device = torch.device(first_device)
    else:
        device = next(model.parameters()).device

    inputs = {k: v.to(device) for k, v in enc.items()}

    print(f"[viz] Running forward pass ({seq_len} tokens) ...")
    with torch.no_grad():
        model(**inputs)

    for h in hooks:
        h.remove()

    # Move to CPU float32.
    activations = {}
    for i in range(num_layers):
        activations[i] = captured[i].float().cpu()  # (1, seq_len, hidden)

    del model
    torch.cuda.empty_cache()
    return activations, token_strs


def score_per_token(
    probe_type: str,
    ckpt_path: Path,
    metrics_path: Path,
    activations: torch.Tensor,
    hidden_size: int,
) -> np.ndarray:
    """Run probe on each token position independently. Returns (seq_len,) scores."""
    ckpt = torch.load(ckpt_path, weights_only=False, map_location="cpu")
    seq_len = activations.shape[1]

    if ckpt["type"] == "mean_diff":
        direction = ckpt["direction"]  # (hidden,)
        X = activations[0].numpy()  # (seq_len, hidden)
        scores = X @ direction  # (seq_len,)
        # Normalize to [0, 1] range for visualization.
        lo, hi = scores.min(), scores.max()
        if hi - lo > 1e-8:
            scores = (scores - lo) / (hi - lo)
        else:
            scores = np.full_like(scores, 0.5)
        return scores

    probe = build_probe(probe_type, hidden_size)
    probe.load_state_dict(ckpt["state_dict"])
    probe.eval()

    # For per-token scoring we apply the probe to each position as if it were
    # a single-token input: (seq_len, hidden) treated as a batch.
    x = activations[0]  # (seq_len, hidden)

    with torch.no_grad():
        # All probes accept (batch, hidden) → (batch,) logits
        # For probes with MLP transform + head, this gives per-position score.
        if hasattr(probe, "transform") and hasattr(probe, "head"):
            # MLP, Attention, MultiMax, MaxRollingMean — use transform + head directly
            y = probe.transform(x)  # (seq_len, mlp_hidden)
            logits = probe.head(y).squeeze(-1)  # (seq_len,)
        elif hasattr(probe, "linear"):
            # Linear / EMA — single linear layer
            logits = probe.linear(x).squeeze(-1)  # (seq_len,)
        else:
            logits = probe(x)

        scores = torch.sigmoid(logits).numpy()

    return scores


def visualize_tokens(
    text: str,
    run_dir: str,
    probe_type: str = "linear",
    output_path: str | None = None,
    max_length: int = 2048,
    label: int | None = None,
) -> str:
    """Generate per-token probe heatmap. Returns path to output HTML."""
    import plotly.graph_objects as go

    run_dir = Path(run_dir)
    import yaml
    cfg = yaml.safe_load((run_dir / "config.yaml").read_text())

    model_id = cfg["model"]["id"]
    num_layers = cfg["model"]["num_layers"]
    hidden_size = cfg["model"]["hidden_size"]
    layer_path = cfg["model"].get("layer_path", "model.layers")
    layer_output_index = cfg["model"].get("layer_output_index")
    results_dir = run_dir / "results"

    # Extract activations at all positions.
    activations, token_strs = extract_all_positions(
        model_id=model_id,
        text=text,
        num_layers=num_layers,
        layer_path=layer_path,
        layer_output_index=layer_output_index,
        max_length=max_length,
    )

    seq_len = len(token_strs)

    # Find all trained probes of the requested type.
    layer_scores = {}
    for layer_dir in sorted(results_dir.iterdir()):
        if not layer_dir.is_dir() or not layer_dir.name.startswith("layer_"):
            continue
        layer_idx = int(layer_dir.name.split("_")[1])
        probe_dir = layer_dir / probe_type
        ckpt = probe_dir / "probe.pt"
        met = probe_dir / "metrics.json"
        if not ckpt.exists() or not met.exists():
            continue

        scores = score_per_token(
            probe_type=probe_type,
            ckpt_path=ckpt,
            metrics_path=met,
            activations=activations[layer_idx],
            hidden_size=hidden_size,
        )
        layer_scores[layer_idx] = scores

    if not layer_scores:
        print(f"[viz] No trained {probe_type} probes found in {results_dir}")
        sys.exit(1)

    layers = sorted(layer_scores.keys())
    heatmap = np.stack([layer_scores[l] for l in layers], axis=0)  # (n_layers, seq_len)

    print(f"[viz] Heatmap shape: {heatmap.shape} ({len(layers)} layers × {seq_len} tokens)")

    # Truncate token strings for display (keep first 20 chars).
    display_tokens = [t.replace("\n", "\\n")[:20] for t in token_strs]

    # Build hover text with full token and score.
    hover_text = []
    for i, layer in enumerate(layers):
        row = []
        for j in range(seq_len):
            tok_repr = repr(token_strs[j])
            row.append(
                f"Layer {layer}<br>"
                f"Pos {j}: {tok_repr}<br>"
                f"Score: {heatmap[i, j]:.4f}"
            )
        hover_text.append(row)

    title = f"Per-token probe scores ({probe_type})"
    if label is not None:
        title += f" — label={label} ({'AWARE' if label == 1 else 'NOT AWARE'})"
    title += f"<br><sub>{model_id} | {seq_len} tokens</sub>"

    fig = go.Figure(data=go.Heatmap(
        z=heatmap,
        x=list(range(seq_len)),
        y=[f"L{l}" for l in layers],
        text=hover_text,
        hoverinfo="text",
        colorscale="RdYlGn",
        reversescale=True,  # Red = high score (aware), green = low
        colorbar=dict(title="Score"),
        zmin=0,
        zmax=1,
    ))

    fig.update_layout(
        title=title,
        xaxis_title="Token position",
        yaxis_title="Layer",
        height=max(400, 20 * len(layers) + 150),
        width=max(800, 4 * seq_len + 200),
        xaxis=dict(
            tickmode="array",
            tickvals=list(range(0, seq_len, max(1, seq_len // 50))),
            ticktext=[display_tokens[i] for i in range(0, seq_len, max(1, seq_len // 50))],
            tickangle=45,
        ),
        yaxis=dict(autorange="reversed"),
    )

    # Output path.
    if output_path:
        out = Path(output_path)
    else:
        out = run_dir / f"viz_{probe_type}_tokens.html"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(out))
    print(f"[viz] Saved heatmap → {out}")

    # Also save raw scores as JSON for further analysis.
    json_out = out.with_suffix(".json")
    with open(json_out, "w") as f:
        json.dump({
            "probe_type": probe_type,
            "model_id": model_id,
            "num_tokens": seq_len,
            "layers": layers,
            "token_strs": token_strs,
            "scores": {str(l): layer_scores[l].tolist() for l in layers},
            "label": label,
            "text_preview": text[:200],
        }, f, indent=2)
    print(f"[viz] Saved scores → {json_out}")

    return str(out)


def submit_slurm(args: argparse.Namespace) -> None:
    run_dir = Path(args.run_dir)
    import yaml
    cfg = yaml.safe_load((run_dir / "config.yaml").read_text())
    slurm_cfg = cfg.get("slurm", {})

    partition = slurm_cfg.get("partition", "compute")
    timeout = slurm_cfg.get("extract_timeout", "4:00:00")

    output_path = args.output or str(run_dir / f"viz_{args.probe_type}_tokens.html")
    log_path = run_dir / f"viz_{args.probe_type}_slurm.log"

    # Write text to a temp file so we don't have shell quoting issues.
    text_file = run_dir / "viz_input_text.txt"
    text_file.write_text(args.text)

    script = f"""#!/bin/bash
#SBATCH --job-name=probe-viz
#SBATCH --partition={partition}
#SBATCH --gres=gpu:1
#SBATCH --time={timeout}
#SBATCH --output={log_path}
#SBATCH --error={log_path}

cd {PROJECT_ROOT}
uv run python scripts/visualize_tokens.py \\
    --text-file {repr(str(text_file))} \\
    --run-dir {repr(str(args.run_dir))} \\
    --probe-type {args.probe_type} \\
    --output {repr(output_path)}
"""

    script_path = run_dir / f"viz_{args.probe_type}_job.sh"
    script_path.write_text(script)
    print(f"[viz] Submitting SLURM job ...")
    result = subprocess.run(["sbatch", str(script_path)], capture_output=True, text=True)
    print(result.stdout.strip())
    if result.returncode != 0:
        print(f"SLURM error: {result.stderr}", file=sys.stderr)
        sys.exit(1)
    print(f"Log: {log_path}")


def main():
    parser = argparse.ArgumentParser(description="Per-token probe heatmap visualization")
    text_group = parser.add_mutually_exclusive_group(required=True)
    text_group.add_argument("--text", help="Text to visualize")
    text_group.add_argument("--text-file", help="Read text from a file")
    text_group.add_argument("--dataset", help="JSONL dataset path (use with --index)")

    parser.add_argument("--index", type=int, default=0, help="Example index in dataset (default: 0)")
    parser.add_argument("--run-dir", default="logs/20260311_232544_olmo3-7b-think")
    parser.add_argument("--probe-type", default="linear", help="Probe type to visualize (default: linear)")
    parser.add_argument("--output", help="Output HTML path")
    parser.add_argument("--slurm", action="store_true", help="Submit as SLURM job")
    args = parser.parse_args()

    # Resolve text input.
    label = None
    if args.text:
        text = args.text
    elif args.text_file:
        text = Path(args.text_file).read_text()
    else:
        rows = []
        with open(args.dataset) as f:
            for line in f:
                rows.append(json.loads(line))
        if args.index >= len(rows):
            print(f"Index {args.index} out of range (dataset has {len(rows)} examples)")
            sys.exit(1)
        row = rows[args.index]
        text = row["text"]
        label = row.get("label")
        print(f"[viz] Using example {args.index}: label={label}, {len(text)} chars")

    args.text = text

    if args.slurm:
        submit_slurm(args)
    else:
        visualize_tokens(
            text=text,
            run_dir=args.run_dir,
            probe_type=args.probe_type,
            output_path=args.output,
            label=label,
        )


if __name__ == "__main__":
    main()
