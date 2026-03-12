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


def _build_html(
    token_strs: list[str],
    layers: list[int],
    scores: dict[int, list[float]],
    probe_type: str,
    model_id: str,
    label: int | None,
) -> str:
    """Build self-contained interactive HTML token viewer."""
    import html as html_mod

    # Prepare data for embedding.
    tokens_json = json.dumps([html_mod.escape(t) for t in token_strs])
    raw_tokens_json = json.dumps(token_strs)
    layers_json = json.dumps(layers)
    scores_json = json.dumps({str(l): scores[l] for l in layers})

    label_str = ""
    if label is not None:
        label_str = f' &mdash; label={label} ({"AWARE" if label == 1 else "NOT AWARE"})'

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Probe Token Viewer</title>
<style>
* {{ margin: 0; padding: 0; box-sizing: border-box; }}
body {{ font-family: 'SF Mono', 'Fira Code', 'Consolas', monospace; background: #1a1a2e; color: #e0e0e0; padding: 20px; }}
h1 {{ font-size: 16px; margin-bottom: 4px; color: #fff; }}
.subtitle {{ font-size: 12px; color: #888; margin-bottom: 16px; }}
.controls {{ display: flex; align-items: center; gap: 16px; margin-bottom: 16px; background: #16213e; padding: 12px 16px; border-radius: 8px; position: sticky; top: 0; z-index: 10; }}
.controls label {{ font-size: 13px; white-space: nowrap; }}
.controls input[type=range] {{ flex: 1; max-width: 400px; accent-color: #e94560; }}
.layer-display {{ font-size: 14px; font-weight: bold; color: #e94560; min-width: 70px; }}
.legend {{ display: flex; align-items: center; gap: 4px; font-size: 11px; color: #888; }}
.legend-bar {{ width: 120px; height: 12px; border-radius: 3px; background: linear-gradient(to right, #1a5276, #2ecc71, #f39c12, #e74c3c); }}
#token-container {{ line-height: 2.2; padding: 16px; background: #0f3460; border-radius: 8px; margin-bottom: 16px; }}
.tok {{ display: inline; padding: 2px 1px; border-radius: 3px; cursor: pointer; transition: outline 0.1s; font-size: 13px; white-space: pre-wrap; }}
.tok:hover {{ outline: 2px solid #fff; outline-offset: 1px; }}
.tok.selected {{ outline: 2px solid #e94560; outline-offset: 1px; }}
#detail-panel {{ background: #16213e; border-radius: 8px; padding: 16px; min-height: 200px; }}
#detail-panel h2 {{ font-size: 14px; margin-bottom: 12px; color: #e94560; }}
.detail-token {{ font-size: 16px; background: #0f3460; padding: 4px 8px; border-radius: 4px; margin-bottom: 12px; display: inline-block; }}
#bar-chart {{ display: flex; align-items: flex-end; gap: 2px; height: 150px; }}
.bar-group {{ display: flex; flex-direction: column; align-items: center; flex: 1; min-width: 0; }}
.bar {{ width: 100%; border-radius: 2px 2px 0 0; transition: height 0.15s; cursor: pointer; min-height: 1px; }}
.bar-label {{ font-size: 9px; color: #666; margin-top: 2px; writing-mode: vertical-lr; text-orientation: mixed; }}
.bar:hover {{ outline: 1px solid #fff; }}
#detail-info {{ font-size: 12px; color: #aaa; margin-top: 8px; }}
.empty-state {{ color: #555; font-size: 13px; padding: 40px; text-align: center; }}
</style>
</head>
<body>

<h1>Per-token Probe Scores ({html_mod.escape(probe_type)}){label_str}</h1>
<div class="subtitle">{html_mod.escape(model_id)} &middot; <span id="n-tokens"></span> tokens &middot; <span id="n-layers"></span> layers</div>

<div class="controls">
    <label for="layer-slider">Layer:</label>
    <input type="range" id="layer-slider" min="0" max="0" value="0">
    <span class="layer-display" id="layer-label">L0</span>
    <div class="legend">
        <span>0</span>
        <div class="legend-bar"></div>
        <span>1</span>
    </div>
</div>

<div id="token-container"></div>

<div id="detail-panel">
    <div class="empty-state">Click any token to see its score across all layers</div>
</div>

<script>
const tokens = {tokens_json};
const rawTokens = {raw_tokens_json};
const layers = {layers_json};
const scores = {scores_json};
const nTokens = tokens.length;
const nLayers = layers.length;

document.getElementById('n-tokens').textContent = nTokens;
document.getElementById('n-layers').textContent = nLayers;

// Color interpolation: blue(0) -> green(0.35) -> yellow(0.6) -> red(1)
function scoreColor(s) {{
    const stops = [
        [0.0,  26,  82, 118],
        [0.35, 46, 204, 113],
        [0.6, 243, 156,  18],
        [1.0, 231,  76,  60],
    ];
    let i = 0;
    while (i < stops.length - 2 && s > stops[i+1][0]) i++;
    const [t0, r0, g0, b0] = stops[i];
    const [t1, r1, g1, b1] = stops[i+1];
    const f = (s - t0) / (t1 - t0);
    const r = Math.round(r0 + f * (r1 - r0));
    const g = Math.round(g0 + f * (g1 - g0));
    const b = Math.round(b0 + f * (b1 - b0));
    // Darker background, use alpha for readability
    return `rgba(${{r}},${{g}},${{b}},0.7)`;
}}

// Build token spans.
const container = document.getElementById('token-container');
const tokenEls = [];
for (let j = 0; j < nTokens; j++) {{
    const span = document.createElement('span');
    span.className = 'tok';
    span.dataset.idx = j;
    span.innerHTML = tokens[j];
    span.addEventListener('click', () => selectToken(j));
    container.appendChild(span);
    tokenEls.push(span);
}}

// Layer slider.
const slider = document.getElementById('layer-slider');
slider.max = nLayers - 1;
// Default to best layer (layer 20 = index for layer 20).
const defaultLayerIdx = layers.indexOf(20);
slider.value = defaultLayerIdx >= 0 ? defaultLayerIdx : Math.floor(nLayers / 2);
const layerLabel = document.getElementById('layer-label');

function updateColors() {{
    const li = parseInt(slider.value);
    const layer = layers[li];
    layerLabel.textContent = 'L' + layer;
    const layerScores = scores[String(layer)];
    for (let j = 0; j < nTokens; j++) {{
        tokenEls[j].style.backgroundColor = scoreColor(layerScores[j]);
    }}
}}

slider.addEventListener('input', updateColors);
updateColors();

// Token detail panel.
let selectedIdx = null;

function selectToken(j) {{
    if (selectedIdx !== null) tokenEls[selectedIdx].classList.remove('selected');
    selectedIdx = j;
    tokenEls[j].classList.add('selected');

    const panel = document.getElementById('detail-panel');
    const tokenScores = layers.map(l => scores[String(l)][j]);
    const maxScore = Math.max(...tokenScores);
    const maxLayer = layers[tokenScores.indexOf(maxScore)];

    let html = '<h2>Token ' + j + '</h2>';
    html += '<div class="detail-token">' + tokens[j] + '</div>';
    html += '<div id="bar-chart">';
    for (let i = 0; i < nLayers; i++) {{
        const s = tokenScores[i];
        const h = Math.max(1, s * 140);
        const color = scoreColor(s);
        html += '<div class="bar-group">';
        html += '<div class="bar" style="height:' + h + 'px;background:' + color + '" ';
        html += 'title="Layer ' + layers[i] + ': ' + s.toFixed(4) + '">';
        html += '</div>';
        if (i % 4 === 0) html += '<div class="bar-label">L' + layers[i] + '</div>';
        html += '</div>';
    }}
    html += '</div>';
    html += '<div id="detail-info">Peak: <b>L' + maxLayer + '</b> (score ' + maxScore.toFixed(4) + ') &middot; Raw token: <code>' + JSON.stringify(rawTokens[j]) + '</code></div>';
    panel.innerHTML = html;
}}
</script>
</body>
</html>"""


def visualize_tokens(
    text: str,
    run_dir: str,
    probe_type: str = "linear",
    output_path: str | None = None,
    max_length: int = 2048,
    label: int | None = None,
) -> str:
    """Generate interactive per-token probe viewer. Returns path to output HTML."""
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
        layer_scores[layer_idx] = scores.tolist()

    if not layer_scores:
        print(f"[viz] No trained {probe_type} probes found in {results_dir}")
        sys.exit(1)

    layers = sorted(layer_scores.keys())
    print(f"[viz] {len(layers)} layers × {seq_len} tokens")

    # Build HTML.
    html = _build_html(
        token_strs=token_strs,
        layers=layers,
        scores=layer_scores,
        probe_type=probe_type,
        model_id=model_id,
        label=label,
    )

    # Output path.
    if output_path:
        out = Path(output_path)
    else:
        out = run_dir / f"viz_{probe_type}_tokens.html"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html)
    print(f"[viz] Saved → {out}")

    # Also save raw scores as JSON.
    json_out = out.with_suffix(".json")
    with open(json_out, "w") as f:
        json.dump({
            "probe_type": probe_type,
            "model_id": model_id,
            "num_tokens": seq_len,
            "layers": layers,
            "token_strs": token_strs,
            "scores": {str(l): layer_scores[l] for l in layers},
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


def rebuild_from_json(json_path: str, output_path: str | None = None) -> str:
    """Rebuild interactive HTML from a previously saved JSON scores file."""
    data = json.loads(Path(json_path).read_text())
    html = _build_html(
        token_strs=data["token_strs"],
        layers=data["layers"],
        scores={int(k): v for k, v in data["scores"].items()},
        probe_type=data["probe_type"],
        model_id=data["model_id"],
        label=data.get("label"),
    )
    out = Path(output_path) if output_path else Path(json_path).with_suffix(".html")
    out.write_text(html)
    print(f"[viz] Rebuilt → {out}")
    return str(out)


def main():
    parser = argparse.ArgumentParser(description="Per-token probe heatmap visualization")
    text_group = parser.add_mutually_exclusive_group(required=True)
    text_group.add_argument("--text", help="Text to visualize")
    text_group.add_argument("--text-file", help="Read text from a file")
    text_group.add_argument("--dataset", help="JSONL dataset path (use with --index)")
    text_group.add_argument("--from-json", help="Rebuild HTML from existing JSON scores file (no GPU needed)")

    parser.add_argument("--index", type=int, default=0, help="Example index in dataset (default: 0)")
    parser.add_argument("--run-dir", default="logs/20260311_232544_olmo3-7b-think")
    parser.add_argument("--probe-type", default="linear", help="Probe type to visualize (default: linear)")
    parser.add_argument("--output", help="Output HTML path")
    parser.add_argument("--slurm", action="store_true", help="Submit as SLURM job")
    args = parser.parse_args()

    if args.from_json:
        rebuild_from_json(args.from_json, args.output)
        return

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
