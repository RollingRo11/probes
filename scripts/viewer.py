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

def get_data_dir() -> Path:
    """Resolve data directory from CLI args, env var, or default."""
    import os
    # 1. CLI arg (for local use: streamlit run viewer.py -- --data-dir ...)
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default=None)
    args, _ = parser.parse_known_args()
    if args.data_dir:
        return Path(args.data_dir)
    # 2. Environment variable (for Streamlit Cloud)
    if os.environ.get("PROBE_DATA_DIR"):
        return Path(os.environ["PROBE_DATA_DIR"])
    # 3. Default: look relative to repo root
    repo_root = Path(__file__).resolve().parent.parent
    default = repo_root / "logs" / "20260311_232544_olmo3-7b-think" / "viz_sad"
    if default.exists():
        return default
    st.error("No data directory found. Set --data-dir or PROBE_DATA_DIR env var.")
    st.stop()

DATA_DIR = get_data_dir()

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------

@st.cache_data
def load_manifest():
    import gzip
    gz_path = DATA_DIR / "manifest.json.gz"
    raw_path = DATA_DIR / "manifest.json"
    if gz_path.exists():
        with gzip.open(gz_path, "rt") as f:
            return json.load(f)
    with open(raw_path) as f:
        return json.load(f)

@st.cache_data
def load_example(filename: str):
    import gzip
    gz_path = DATA_DIR / (filename + ".gz")
    raw_path = DATA_DIR / filename
    if gz_path.exists():
        with gzip.open(gz_path, "rt") as f:
            return json.load(f)
    with open(raw_path) as f:
        return json.load(f)

manifest = load_manifest()

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Probe Token Viewer",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Sidebar controls
# ---------------------------------------------------------------------------

st.sidebar.title("Probe Token Viewer")
st.sidebar.caption(f"{manifest['model_id']}")
st.sidebar.caption(f"{manifest['n_examples']} examples from {Path(manifest['dataset']).name}")

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

# ---------------------------------------------------------------------------
# Load selected example
# ---------------------------------------------------------------------------

example = load_example(filtered[selected_idx]["file"])
token_strs = example["token_strs"]
num_tokens = example["num_tokens"]
label = example["label"]
scores_for_probe = example["scores"].get(probe_type, {})
layer_scores = scores_for_probe.get(str(layer), [0.0] * num_tokens)

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
st.markdown(
    f"### Example #{example['index']} &nbsp; "
    f"<span style='color:{label_color};font-size:14px;'>[{label_text}]</span> &nbsp; "
    f"<span style='color:#888;font-size:13px;'>{num_tokens} tokens · Layer {layer} · {probe_type}</span>",
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
for j, (tok, score) in enumerate(zip(token_strs, layer_scores)):
    token_data.append({
        "idx": j,
        "text": html_mod.escape(tok).replace("\n", "<br>"),
        "score": round(score, 3),
        "color": score_color(score),
    })

token_data_json = json.dumps(token_data)
selected = st.session_state.selected_token

# All scores for the selected token across layers and probe types (for the chart).
all_layers = example["layers"]
chart_series = {}
for pt in example["probe_types"]:
    pt_scores = example["scores"].get(pt, {})
    chart_series[pt] = [
        pt_scores.get(str(l), [0.0] * num_tokens)[selected]
        for l in all_layers
    ]
chart_series_json = json.dumps(chart_series)
layers_json = json.dumps(all_layers)

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
    </div>
    <div class="chart-container">
        <div class="chart-header">
            <h3>Score across layers</h3>
            <label><input type="checkbox" id="show-all" onchange="drawChart()"> Show all probes</label>
        </div>
        <canvas id="chart" height="180"></canvas>
        <div class="peak-info" id="peak-info"></div>
        <div class="legend" id="legend"></div>
    </div>
</div>

<script>
const tokenData = {token_data_json};
const chartSeries = {chart_series_json};
const layers = {layers_json};
const currentProbe = {json.dumps(probe_type)};
const nTokens = tokenData.length;
let selectedIdx = {selected};

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
    span.title = `pos ${{j}} | score ${{t.score.toFixed(3)}}`;
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
    document.getElementById('detail-score').textContent = 'Score at current layer: ' + t.score.toFixed(4);
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
    const probes = showAll ? Object.keys(chartSeries) : [currentProbe];

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

# Calculate height based on content.
# Rough estimate: token container ~300px + detail panel ~280px + padding.
estimated_height = 600
clicked = components.html(component_html, height=estimated_height, scrolling=True)

# Update session state if a token was clicked.
if clicked is not None and isinstance(clicked, (int, float)):
    st.session_state.selected_token = int(clicked)
    st.rerun()
