"""Streamlit probe token viewer.

Loads per-example JSON files produced by batch_score.py and renders an
interactive token-level probe visualization.

Usage:
    uv run streamlit run scripts/viewer.py -- \
        --data-dir logs/20260311_232544_olmo3-7b-think/viz_sad
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import streamlit as st
import streamlit.components.v1 as components

# ---------------------------------------------------------------------------
# Args (parsed from sys.argv after Streamlit's "--" separator)
# ---------------------------------------------------------------------------

def get_run_dir() -> Path:
    """Resolve training run directory."""
    import os
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--run-dir", default=None)
    args, _ = parser.parse_known_args()
    if args.run_dir:
        return Path(args.run_dir)
    if args.data_dir:
        # Legacy: --data-dir points to a viz_* subdir; derive run dir from parent.
        return Path(args.data_dir).parent
    if os.environ.get("PROBE_DATA_DIR"):
        return Path(os.environ["PROBE_DATA_DIR"]).parent
    repo_root = Path(__file__).resolve().parent.parent
    default = repo_root / "logs" / "20260311_232544_olmo3-7b-think"
    if default.exists():
        return default
    st.error("No run directory found. Set --run-dir or PROBE_DATA_DIR env var.")
    st.stop()

RUN_DIR = get_run_dir()


def discover_datasets() -> dict[str, Path]:
    """Find all viz_* subdirectories in the run dir."""
    datasets = {}
    for d in sorted(RUN_DIR.iterdir()):
        if d.is_dir() and d.name.startswith("viz_"):
            manifest = d / "manifest.json.gz"
            if not manifest.exists():
                manifest = d / "manifest.json"
            if manifest.exists():
                label = d.name.removeprefix("viz_")
                datasets[label] = d
    return datasets

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------

@st.cache_data
def load_manifest(data_dir: str):
    import gzip
    data_dir = Path(data_dir)
    gz_path = data_dir / "manifest.json.gz"
    raw_path = data_dir / "manifest.json"
    if gz_path.exists():
        with gzip.open(gz_path, "rt") as f:
            return json.load(f)
    with open(raw_path) as f:
        return json.load(f)

@st.cache_data
def load_example(data_dir: str, filename: str):
    import gzip
    data_dir = Path(data_dir)
    gz_path = data_dir / (filename + ".gz")
    raw_path = data_dir / filename
    if gz_path.exists():
        with gzip.open(gz_path, "rt") as f:
            return json.load(f)
    with open(raw_path) as f:
        return json.load(f)

# ---------------------------------------------------------------------------
# Analysis helpers
# ---------------------------------------------------------------------------

@st.cache_data
def load_eval_results(run_dir: str) -> dict[str, list[dict]]:
    """Load all eval_*.json files from the run directory."""
    import glob
    results = {}
    for path in sorted(glob.glob(f"{run_dir}/eval_*.json")):
        name = Path(path).stem.removeprefix("eval_")
        with open(path) as f:
            data = json.load(f)
        results[name] = data.get("results", [])
    return results


@st.cache_data
def load_training_metrics(run_dir: str) -> list[dict]:
    """Load per-layer per-probe training metrics."""
    import glob
    metrics = []
    results_dir = Path(run_dir) / "results"
    if not results_dir.exists():
        return metrics
    for metrics_path in sorted(results_dir.glob("layer_*/*/metrics.json")):
        parts = metrics_path.parts
        layer = int(parts[-3].split("_")[1])
        probe_type = parts[-2]
        with open(metrics_path) as f:
            m = json.load(f)
        m["layer"] = layer
        m["probe_type"] = probe_type
        metrics.append(m)
    return metrics


@st.cache_data
def compute_dataset_stats(
    data_dir: str, manifest_data: dict, max_examples: int = 200,
) -> dict:
    """Compute aggregate per-token score statistics across examples.

    Samples up to max_examples for speed. Returns stats per probe per layer.
    """
    import gzip
    import numpy as np
    data_dir = Path(data_dir)
    examples = manifest_data["examples"][:max_examples]
    probe_types = manifest_data["probe_types"]
    layers = manifest_data["layers"]

    # Accumulate: for each (probe, layer), collect mean-of-token-scores per example.
    stats: dict[str, dict[str, list[float]]] = {
        pt: {str(l): [] for l in layers} for pt in probe_types
    }

    for ex in examples:
        gz_path = data_dir / (ex["file"] + ".gz")
        raw_path = data_dir / ex["file"]
        if gz_path.exists():
            with gzip.open(gz_path, "rt") as f:
                data = json.load(f)
        elif raw_path.exists():
            with open(raw_path) as f:
                data = json.load(f)
        else:
            continue

        for pt in probe_types:
            pt_scores = data["scores"].get(pt, {})
            for l in layers:
                arr = pt_scores.get(str(l), [])
                if arr:
                    stats[pt][str(l)].append(float(np.mean(arr)))

    # Summarise.
    summary = {}
    for pt in probe_types:
        pt_summary = {}
        for l in layers:
            vals = stats[pt][str(l)]
            if vals:
                arr = np.array(vals)
                pt_summary[str(l)] = {
                    "mean": float(np.mean(arr)),
                    "std": float(np.std(arr)),
                    "median": float(np.median(arr)),
                    "p90": float(np.percentile(arr, 90)),
                    "above_05": float(np.mean(arr > 0.5)),
                }
        summary[pt] = pt_summary

    return {"summary": summary, "n_sampled": len(examples)}


def _render_analysis(data_dir: str, run_dir: str, manifest: dict, dataset_name: str):
    """Render the analysis pane."""
    import numpy as np

    st.markdown(f"## Analysis: **{dataset_name}**")
    st.caption(
        f"{manifest['model_id']} · {manifest['n_examples']} examples · "
        f"{len(manifest['probe_types'])} probes · {len(manifest['layers'])} layers"
    )

    # --- Training metrics ---
    train_metrics = load_training_metrics(run_dir)
    if train_metrics:
        st.markdown("### Training Metrics (held-out test set)")
        # Build a table: rows = layers, columns = probes, cells = AUROC
        probe_types = manifest["probe_types"]
        layers = manifest["layers"]
        header = "| Layer | " + " | ".join(probe_types) + " |"
        sep = "|---:|" + "|".join(["---:"] * len(probe_types)) + "|"
        rows = []
        # Index training metrics by (layer, probe_type)
        tm_idx = {(m["layer"], m["probe_type"]): m for m in train_metrics}
        best_auroc = 0
        best_cell = ""
        for l in layers:
            cells = []
            for pt in probe_types:
                m = tm_idx.get((l, pt))
                if m:
                    auroc = m.get("auroc", 0)
                    if auroc > best_auroc:
                        best_auroc = auroc
                        best_cell = f"{pt} L{l}"
                    cells.append(f"{auroc:.3f}")
                else:
                    cells.append("—")
            rows.append(f"| {l} | " + " | ".join(cells) + " |")
        st.markdown(f"Best: **{best_cell}** ({best_auroc:.4f} AUROC)")
        # Show only a few key layers to avoid huge table
        key_layers = [0, 4, 8, 12, 16, 20, 24, 28, 31]
        key_rows = [r for l, r in zip(layers, rows) if l in key_layers]
        st.markdown(header + "\n" + sep + "\n" + "\n".join(key_rows))

    # --- Cross-dataset eval results ---
    eval_results = load_eval_results(run_dir)
    if eval_results:
        st.markdown("### Cross-Dataset Eval (AUROC)")
        for eval_name, results in eval_results.items():
            if not results:
                continue
            st.markdown(f"**{eval_name}** ({len(results)} entries)")
            # Find best AUROC per probe type
            probe_best: dict[str, tuple[float, int]] = {}
            for r in results:
                pt = r["probe_type"]
                auroc = r.get("auroc", 0)
                if pt not in probe_best or auroc > probe_best[pt][0]:
                    probe_best[pt] = (auroc, r["layer"])
            cols = st.columns(len(probe_best))
            for col, (pt, (auroc, layer)) in zip(cols, sorted(probe_best.items())):
                col.metric(pt, f"{auroc:.3f}", f"Layer {layer}")

    # --- Per-token score stats from viz data ---
    st.markdown("### Per-Token Score Distribution")
    st.caption("Mean of per-token scores across examples (sampled up to 200)")
    ds_stats = compute_dataset_stats(data_dir, manifest)
    summary = ds_stats["summary"]
    st.caption(f"Sampled {ds_stats['n_sampled']} / {manifest['n_examples']} examples")

    probe_types = manifest["probe_types"]
    layers = manifest["layers"]

    # Heatmap: mean score per (probe, layer) using an HTML table
    selected_stat = st.selectbox(
        "Statistic", ["mean", "median", "p90", "above_05"], index=0,
        format_func=lambda x: {
            "mean": "Mean score",
            "median": "Median score",
            "p90": "90th percentile",
            "above_05": "% examples > 0.5",
        }.get(x, x),
    )

    # Build HTML heatmap table
    def val_color(v: float) -> str:
        # Blue to red scale
        r = int(min(255, v * 2 * 255))
        b = int(min(255, (1 - v) * 2 * 255))
        g = int(min(255, (1 - abs(v - 0.5) * 2) * 200))
        return f"rgb({r},{g},{b})"

    html_parts = ['<table style="border-collapse:collapse;font-size:12px;font-family:monospace;width:100%;">']
    html_parts.append("<tr><th style='padding:4px 8px;color:#aaa;'>Layer</th>")
    for pt in probe_types:
        html_parts.append(f"<th style='padding:4px 8px;color:#ccc;'>{pt}</th>")
    html_parts.append("</tr>")

    key_layers = [l for l in layers if l % 2 == 0 or l == layers[-1]]
    for l in key_layers:
        html_parts.append(f"<tr><td style='padding:3px 8px;color:#aaa;text-align:right;'>{l}</td>")
        for pt in probe_types:
            s = summary.get(pt, {}).get(str(l), {})
            v = s.get(selected_stat, 0)
            bg = val_color(v)
            display = f"{v:.3f}" if selected_stat != "above_05" else f"{v:.0%}"
            html_parts.append(
                f"<td style='padding:3px 8px;text-align:center;background:{bg};color:#fff;'>{display}</td>"
            )
        html_parts.append("</tr>")
    html_parts.append("</table>")
    st.markdown("".join(html_parts), unsafe_allow_html=True)

    # --- Score distribution chart per probe at best layer ---
    st.markdown("### Score by Layer (best probe overview)")
    # For each probe, plot mean score across layers as a line
    chart_html = _build_analysis_chart(summary, probe_types, layers, selected_stat)
    components.html(chart_html, height=350, scrolling=False)


def _build_analysis_chart(
    summary: dict, probe_types: list[str], layers: list[int], stat: str,
) -> str:
    """Build an HTML canvas chart showing stat across layers for each probe."""
    probe_colors = {
        'linear': '#e94560',
        'ema': '#f39c12',
        'mlp': '#2ecc71',
        'attention': '#3498db',
        'multimax': '#9b59b6',
        'max_rolling_mean': '#e67e22',
        'mean_diff': '#95a5a6',
    }
    # Prepare series data
    series = {}
    for pt in probe_types:
        vals = []
        for l in layers:
            s = summary.get(pt, {}).get(str(l), {})
            vals.append(s.get(stat, 0))
        series[pt] = vals

    series_json = json.dumps(series)
    layers_json = json.dumps(layers)
    colors_json = json.dumps(probe_colors)

    return f"""
<html><body style="margin:0;background:#1e1e1e;">
<canvas id="c" style="width:100%;height:320px;"></canvas>
<script>
const series = {series_json};
const layers = {layers_json};
const colors = {colors_json};
const canvas = document.getElementById('c');
const ctx = canvas.getContext('2d');
const dpr = window.devicePixelRatio || 1;

function draw() {{
    const rect = canvas.getBoundingClientRect();
    canvas.width = rect.width * dpr;
    canvas.height = rect.height * dpr;
    ctx.scale(dpr, dpr);
    const W = rect.width, H = rect.height;
    const pad = {{ left: 50, right: 20, top: 20, bottom: 30 }};
    const pW = W - pad.left - pad.right;
    const pH = H - pad.top - pad.bottom;
    const xScale = i => pad.left + (i / (layers.length - 1)) * pW;
    const yScale = v => pad.top + (1 - v) * pH;

    ctx.clearRect(0, 0, W, H);

    // Grid
    ctx.strokeStyle = '#333'; ctx.lineWidth = 0.5;
    for (let v = 0; v <= 1; v += 0.25) {{
        ctx.beginPath(); ctx.moveTo(pad.left, yScale(v));
        ctx.lineTo(W - pad.right, yScale(v)); ctx.stroke();
    }}
    ctx.fillStyle = '#666'; ctx.font = '10px monospace'; ctx.textAlign = 'right';
    for (let v = 0; v <= 1; v += 0.25) ctx.fillText(v.toFixed(2), pad.left - 4, yScale(v) + 3);
    ctx.textAlign = 'center';
    const step = Math.max(1, Math.floor(layers.length / 8));
    for (let i = 0; i < layers.length; i += step)
        ctx.fillText('L' + layers[i], xScale(i), H - 6);

    // Lines
    Object.entries(series).forEach(([pt, vals]) => {{
        ctx.strokeStyle = colors[pt] || '#888';
        ctx.lineWidth = 2;
        ctx.beginPath();
        vals.forEach((v, i) => {{
            const x = xScale(i), y = yScale(Math.max(0, Math.min(1, v)));
            i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
        }});
        ctx.stroke();
    }});

    // Legend
    let lx = pad.left;
    ctx.font = '11px monospace';
    Object.entries(series).forEach(([pt]) => {{
        ctx.fillStyle = colors[pt] || '#888';
        ctx.fillRect(lx, 4, 14, 3);
        ctx.fillStyle = '#aaa';
        ctx.textAlign = 'left';
        ctx.fillText(pt, lx + 18, 10);
        lx += ctx.measureText(pt).width + 34;
    }});
}}
draw();
window.addEventListener('resize', draw);
</script></body></html>
"""


# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Probe Token Viewer",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Full-width layout with dark gray background.
st.markdown("""
<style>
    .stApp { background-color: #1e1e1e; }
    /* Hide Streamlit header/toolbar/footer */
    header[data-testid="stHeader"],
    .stDeployButton,
    #MainMenu,
    footer,
    .stDecoration {
        display: none !important;
        height: 0 !important;
    }
    .block-container,
    .stMainBlockContainer,
    [data-testid="stAppViewBlockContainer"] {
        max-width: 100% !important;
        width: 100% !important;
        padding-left: 1rem !important;
        padding-right: 1rem !important;
        padding-top: 0 !important;
    }
    /* Kill top margin/padding on the main view */
    [data-testid="stAppViewContainer"],
    [data-testid="stAppViewContainer"] > section {
        padding-top: 0 !important;
        margin-top: 0 !important;
    }
    /* Prevent ALL Streamlit wrappers from constraining size or clipping */
    .stMainBlockContainer,
    .stVerticalBlock,
    .stVerticalBlockBorderWrapper,
    [data-testid="stVerticalBlock"],
    [data-testid="stVerticalBlockBorderWrapper"],
    .stHorizontalBlock,
    .element-container,
    .stElementContainer,
    [data-testid="element-container"],
    [data-testid="stElementContainer"] {
        max-width: 100% !important;
        width: 100% !important;
        overflow: visible !important;
        max-height: none !important;
        height: auto !important;
    }
    /* The main scrollable area */
    .main .block-container,
    [data-testid="stAppViewContainer"] > section > div {
        max-width: 100% !important;
        overflow: visible !important;
    }
    iframe {
        width: 100% !important;
        max-width: 100% !important;
    }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Sidebar controls
# ---------------------------------------------------------------------------

st.sidebar.title("Probe Token Viewer")

# Dataset selector.
available_datasets = discover_datasets()
if not available_datasets:
    st.error("No viz_* datasets found in run directory.")
    st.stop()
dataset_names = list(available_datasets.keys())
selected_dataset = st.sidebar.selectbox("Dataset", dataset_names, index=0)
DATA_DIR = available_datasets[selected_dataset]

manifest = load_manifest(str(DATA_DIR))
st.sidebar.caption(f"{manifest['model_id']}")
st.sidebar.caption(f"{manifest['n_examples']} examples from {Path(manifest['dataset']).name}")

# View mode.
view_mode = st.sidebar.radio("View", ["Token Viewer", "Analysis"], index=0)

# Filter by label.
label_filter = st.sidebar.radio("Filter by label", ["All", "Positive (1)", "Negative (0)"], index=0)
filtered = manifest["examples"]
if label_filter == "Positive (1)":
    filtered = [e for e in filtered if e["label"] == 1]
elif label_filter == "Negative (0)":
    filtered = [e for e in filtered if e["label"] == 0]

# Example selector.
example_options = [
    f"{'[+]' if e['label'] == 1 else '[-]'} #{e['index']} ({e['num_tokens']} tok) {e['text_preview'][:60]}..."
    for e in filtered
]
selected_idx = st.sidebar.selectbox(
    f"Example ({len(filtered)} shown)",
    range(len(filtered)),
    format_func=lambda i: example_options[i],
)

# Probe type.
probe_type = st.sidebar.selectbox("Probe type", manifest["probe_types"], index=0)

# Layer slider.
layers = manifest["layers"]
default_layer = 20 if 20 in layers else layers[len(layers) // 2]
layer = st.sidebar.select_slider("Layer", options=layers, value=default_layer)

st.sidebar.markdown("---")
st.sidebar.markdown(
    """
    **Color scale**
    - <span style="color:rgba(26,82,118,1);">&#9632;</span> 0.0 (not aware)
    - <span style="color:rgba(46,204,113,1);">&#9632;</span> 0.35
    - <span style="color:rgba(243,156,18,1);">&#9632;</span> 0.60
    - <span style="color:rgba(231,76,60,1);">&#9632;</span> 1.0 (aware)
    """,
    unsafe_allow_html=True,
)

# ===========================================================================
# ANALYSIS VIEW
# ===========================================================================

if view_mode == "Analysis":
    _render_analysis(str(DATA_DIR), str(RUN_DIR), manifest, selected_dataset)
    st.stop()

# ===========================================================================
# TOKEN VIEWER (below here)
# ===========================================================================

example = load_example(str(DATA_DIR), filtered[selected_idx]["file"])
token_strs = example["token_strs"]
num_tokens = example["num_tokens"]
label = example["label"]
scores_for_probe = example["scores"].get(probe_type, {})
layer_scores = scores_for_probe.get(str(layer), [0.0] * num_tokens)

# For sequence probes, use cumulative scores for token highlighting if available.
has_cumulative = "cumulative" in example and probe_type in example.get("cumulative", {})
has_contributions = "contributions" in example and probe_type in example.get("contributions", {})
if has_cumulative:
    cumulative_for_probe = example["cumulative"][probe_type]
    cumulative_scores = cumulative_for_probe.get(str(layer), [0.0] * num_tokens)
else:
    cumulative_scores = None
# Use contributions for highlighting (cumulative saturates to 1.0 too quickly).
if has_contributions:
    contribution_for_probe_hl = example["contributions"][probe_type]
    highlight_scores = contribution_for_probe_hl.get(str(layer), [0.0] * num_tokens)
else:
    highlight_scores = layer_scores
if has_contributions:
    contribution_for_probe = example["contributions"][probe_type]
    contribution_scores = contribution_for_probe.get(str(layer), [0.0] * num_tokens)
else:
    contribution_scores = None

# ---------------------------------------------------------------------------
# Color function
# ---------------------------------------------------------------------------

def score_color(s: float) -> str:
    stops = [
        (0.0,  26,  82, 118),
        (0.35, 46, 204, 113),
        (0.6, 243, 156,  18),
        (1.0, 231,  76,  60),
    ]
    i = 0
    while i < len(stops) - 2 and s > stops[i + 1][0]:
        i += 1
    t0, r0, g0, b0 = stops[i]
    t1, r1, g1, b1 = stops[i + 1]
    f = (s - t0) / (t1 - t0) if t1 != t0 else 0
    r = int(r0 + f * (r1 - r0))
    g = int(g0 + f * (g1 - g0))
    b = int(b0 + f * (b1 - b0))
    return f"rgba({r},{g},{b},0.75)"

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------

label_text = "AWARE" if label == 1 else "NOT AWARE"
label_color = "#e74c3c" if label == 1 else "#2ecc71"
highlight_mode = " · contribution highlight" if has_contributions else ""
st.markdown(
    f"### Example #{example['index']} &nbsp; "
    f"<span style='color:{label_color};font-size:14px;'>[{label_text}]</span> &nbsp; "
    f"<span style='color:#888;font-size:13px;'>{num_tokens} tokens · Layer {layer} · {probe_type}{highlight_mode}</span>",
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Session state for selected token
# ---------------------------------------------------------------------------

if "selected_token" not in st.session_state:
    st.session_state.selected_token = 0

# ---------------------------------------------------------------------------
# Render tokens as clickable HTML component
# ---------------------------------------------------------------------------

import html as html_mod

# Build token data for the JS component.
token_data = []
for j, (tok, h_score, raw_score) in enumerate(
    zip(token_strs, highlight_scores, layer_scores)
):
    contrib = contribution_scores[j] if contribution_scores else None
    td = {
        "idx": j,
        "text": html_mod.escape(tok).replace("\n", "<br>"),
        "score": round(raw_score, 3),
        "highlight": round(h_score, 3),
        "color": score_color(h_score),
    }
    if contrib is not None:
        td["contribution"] = round(contrib, 3)
    token_data.append(td)

token_data_json = json.dumps(token_data)
selected = st.session_state.selected_token

# Pass full scores to JS so it can recompute chart for any clicked token.
all_layers = example["layers"]
full_scores_json = json.dumps(example["scores"])
full_cumulative_json = json.dumps(example.get("cumulative", {}))
full_contributions_json = json.dumps(example.get("contributions", {}))
layers_json = json.dumps(all_layers)
probe_types_json = json.dumps(example["probe_types"])

# Single HTML component with tokens + embedded line chart.
component_html = f"""
<html>
<head>
<style>
* {{ margin: 0; padding: 0; box-sizing: border-box; }}
body {{ background: #1e1e1e; color: #e0e0e0; font-family: 'SF Mono', 'Fira Code', 'Consolas', monospace; padding: 0; }}

.token-container {{
    line-height: 2.4;
    padding: 16px;
    background: #2a2a2a;
    border-radius: 8px;
    font-size: 13px;
    margin-bottom: 16px;
}}

.tok {{
    display: inline;
    padding: 2px 1px;
    border-radius: 3px;
    cursor: pointer;
    white-space: pre-wrap;
    transition: outline 0.1s;
}}
.tok:hover {{ outline: 2px solid #fff; outline-offset: 1px; }}
.tok.selected {{ outline: 2px solid #e94560; outline-offset: 1px; }}

.detail-panel {{
    background: #2a2a2a;
    border-radius: 8px;
    padding: 16px;
    display: flex;
    gap: 24px;
    align-items: flex-start;
}}

.detail-left {{
    min-width: 160px;
}}
.detail-left h3 {{
    font-size: 14px;
    color: #e94560;
    margin-bottom: 8px;
}}
.detail-token {{
    font-size: 16px;
    background: #363636;
    padding: 4px 8px;
    border-radius: 4px;
    margin-bottom: 8px;
    display: inline-block;
}}
.detail-score {{
    font-size: 12px;
    color: #aaa;
}}

.chart-container {{
    flex: 1;
    min-width: 0;
}}
.chart-header {{
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 8px;
}}
.chart-header h3 {{
    font-size: 14px;
    color: #e94560;
}}
.chart-header label {{
    font-size: 11px;
    color: #888;
    cursor: pointer;
}}
.chart-header input {{
    margin-right: 4px;
    accent-color: #e94560;
}}
canvas {{
    width: 100%;
    border-radius: 4px;
    background: #222;
}}
.peak-info {{
    font-size: 11px;
    color: #888;
    margin-top: 4px;
}}
.legend {{
    display: flex;
    gap: 12px;
    flex-wrap: wrap;
    margin-top: 6px;
    font-size: 11px;
}}
.legend-item {{
    display: flex;
    align-items: center;
    gap: 4px;
    color: #aaa;
}}
.legend-swatch {{
    width: 12px;
    height: 3px;
    border-radius: 1px;
}}
</style>
</head>
<body>

<div class="token-container" id="token-container"></div>

<div class="detail-panel">
    <div class="detail-left">
        <h3 id="detail-title">Token 0</h3>
        <div class="detail-token" id="detail-token"></div>
        <div class="detail-score" id="detail-score"></div>
        <div class="detail-score" id="detail-contribution" style="margin-top:4px;"></div>
        <div class="detail-score" id="detail-cumulative" style="margin-top:4px;"></div>
    </div>
    <div class="chart-container">
        <div class="chart-header">
            <h3>Score across layers</h3>
            <div>
                <label><input type="checkbox" id="show-cumulative" {'checked' if has_cumulative else ''} onchange="drawChart()"> Cumulative</label>
                <label style="margin-left:12px;"><input type="checkbox" id="show-all" onchange="drawChart()"> All probes</label>
            </div>
        </div>
        <canvas id="chart" height="180"></canvas>
        <div class="peak-info" id="peak-info"></div>
        <div class="legend" id="legend"></div>
    </div>
</div>

<script>
const tokenData = {token_data_json};
const fullScores = {full_scores_json};
const fullCumulative = {full_cumulative_json};
const fullContributions = {full_contributions_json};
const layers = {layers_json};
const probeTypes = {probe_types_json};
const currentProbe = {json.dumps(probe_type)};
const hasCumulative = {json.dumps(has_cumulative)};
const nTokens = tokenData.length;
let selectedIdx = {selected};

// Extract chart series for a given token index (per-token independent scores).
function getChartSeries(tokIdx, useCumulative) {{
    const series = {{}};
    const source = useCumulative ? fullCumulative : fullScores;
    probeTypes.forEach(pt => {{
        // Fall back to fullScores if cumulative not available for this probe.
        const ptData = (useCumulative && source[pt]) ? source[pt] : (fullScores[pt] || {{}});
        series[pt] = layers.map(l => {{
            const layerArr = ptData[String(l)] || [];
            return layerArr[tokIdx] || 0;
        }});
    }});
    return series;
}}

const probeColors = {{
    'linear': '#e94560',
    'ema': '#f39c12',
    'mlp': '#2ecc71',
    'attention': '#3498db',
    'multimax': '#9b59b6',
    'max_rolling_mean': '#e67e22',
}};

// Build tokens.
const container = document.getElementById('token-container');
const tokenEls = [];
tokenData.forEach((t, j) => {{
    const span = document.createElement('span');
    span.className = 'tok' + (j === selectedIdx ? ' selected' : '');
    span.innerHTML = t.text;
    span.style.background = t.color;
    const tipParts = [`pos ${{j}} | score ${{t.score.toFixed(3)}}`];
    if (t.highlight !== undefined && t.highlight !== t.score) tipParts.push(`cumulative ${{t.highlight.toFixed(3)}}`);
    if (t.contribution !== undefined) tipParts.push(`contribution ${{t.contribution.toFixed(3)}}`);
    span.title = tipParts.join(' | ');
    span.addEventListener('click', () => selectToken(j));
    container.appendChild(span);
    tokenEls.push(span);
}});

function selectToken(j) {{
    if (selectedIdx >= 0 && selectedIdx < tokenEls.length) {{
        tokenEls[selectedIdx].classList.remove('selected');
    }}
    selectedIdx = j;
    tokenEls[j].classList.add('selected');
    tokenEls[j].scrollIntoView({{ behavior: 'smooth', block: 'nearest' }});
    updateDetail();
    drawChart();

    // Communicate back to Streamlit.
    window.parent.postMessage({{
        type: 'streamlit:setComponentValue',
        value: j,
    }}, '*');
}}

function updateDetail() {{
    const t = tokenData[selectedIdx];
    document.getElementById('detail-title').textContent = 'Token ' + selectedIdx;
    document.getElementById('detail-token').innerHTML = t.text;
    document.getElementById('detail-score').textContent = 'Per-token score: ' + t.score.toFixed(4);
    const cumEl = document.getElementById('detail-cumulative');
    const contribEl = document.getElementById('detail-contribution');
    if (t.highlight !== undefined && t.highlight !== t.score) {{
        cumEl.textContent = 'Cumulative score: ' + t.highlight.toFixed(4);
        cumEl.style.display = '';
    }} else {{
        cumEl.style.display = 'none';
    }}
    if (t.contribution !== undefined) {{
        contribEl.textContent = 'Contribution: ' + t.contribution.toFixed(4);
        contribEl.style.display = '';
    }} else {{
        contribEl.style.display = 'none';
    }}
}}

function drawChart() {{
    const canvas = document.getElementById('chart');
    const ctx = canvas.getContext('2d');
    const dpr = window.devicePixelRatio || 1;
    const rect = canvas.getBoundingClientRect();
    canvas.width = rect.width * dpr;
    canvas.height = rect.height * dpr;
    ctx.scale(dpr, dpr);
    const W = rect.width;
    const H = rect.height;

    ctx.clearRect(0, 0, W, H);

    const showAll = document.getElementById('show-all').checked;
    const showCumulative = document.getElementById('show-cumulative').checked;
    const chartSeries = getChartSeries(selectedIdx, showCumulative);
    const probes = showAll ? probeTypes : [currentProbe];

    const pad = {{ left: 40, right: 16, top: 12, bottom: 24 }};
    const plotW = W - pad.left - pad.right;
    const plotH = H - pad.top - pad.bottom;

    // Y axis: 0 to 1.
    // X axis: layers.
    const xScale = (i) => pad.left + (i / (layers.length - 1)) * plotW;
    const yScale = (v) => pad.top + (1 - v) * plotH;

    // Grid lines.
    ctx.strokeStyle = '#333';
    ctx.lineWidth = 0.5;
    for (let v = 0; v <= 1; v += 0.25) {{
        ctx.beginPath();
        ctx.moveTo(pad.left, yScale(v));
        ctx.lineTo(W - pad.right, yScale(v));
        ctx.stroke();
    }}

    // Y labels.
    ctx.fillStyle = '#666';
    ctx.font = '10px monospace';
    ctx.textAlign = 'right';
    for (let v = 0; v <= 1; v += 0.25) {{
        ctx.fillText(v.toFixed(2), pad.left - 4, yScale(v) + 3);
    }}

    // X labels.
    ctx.textAlign = 'center';
    const xStep = Math.max(1, Math.floor(layers.length / 8));
    for (let i = 0; i < layers.length; i += xStep) {{
        ctx.fillText('L' + layers[i], xScale(i), H - 4);
    }}

    // Draw lines.
    let peakVal = -1, peakLayer = 0;
    probes.forEach(pt => {{
        const series = chartSeries[pt];
        if (!series) return;
        const vals = series;  // already per-token scores across layers

        ctx.strokeStyle = probeColors[pt] || '#888';
        ctx.lineWidth = pt === currentProbe ? 2.5 : 1.2;
        ctx.globalAlpha = pt === currentProbe ? 1.0 : 0.5;

        ctx.beginPath();
        vals.forEach((v, i) => {{
            const x = xScale(i);
            const y = yScale(Math.max(0, Math.min(1, v)));
            if (i === 0) ctx.moveTo(x, y);
            else ctx.lineTo(x, y);

            if (pt === currentProbe && v > peakVal) {{
                peakVal = v;
                peakLayer = layers[i];
            }}
        }});
        ctx.stroke();
        ctx.globalAlpha = 1.0;
    }});

    // Peak dot.
    if (peakVal >= 0) {{
        const peakIdx = layers.indexOf(peakLayer);
        ctx.fillStyle = probeColors[currentProbe] || '#e94560';
        ctx.beginPath();
        ctx.arc(xScale(peakIdx), yScale(peakVal), 4, 0, Math.PI * 2);
        ctx.fill();
    }}

    document.getElementById('peak-info').textContent =
        'Peak for ' + currentProbe + ': Layer ' + peakLayer + ' (score ' + peakVal.toFixed(4) + ')';

    // Legend.
    const legendEl = document.getElementById('legend');
    legendEl.innerHTML = '';
    probes.forEach(pt => {{
        const color = probeColors[pt] || '#888';
        legendEl.innerHTML += `<div class="legend-item"><div class="legend-swatch" style="background:${{color}}"></div>${{pt}}</div>`;
    }});
}}

updateDetail();
// Draw chart after a frame so canvas has correct dimensions.
requestAnimationFrame(() => drawChart());
window.addEventListener('resize', drawChart);
</script>
</body>
</html>
"""

clicked = components.html(component_html, height=4000, scrolling=False)

# Update session state if a token was clicked.
if clicked is not None and isinstance(clicked, (int, float)):
    st.session_state.selected_token = int(clicked)
    st.rerun()
