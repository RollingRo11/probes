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

# ---------------------------------------------------------------------------
# Args (parsed from sys.argv after Streamlit's "--" separator)
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True, help="Directory with manifest.json + example_*.json")
    # Streamlit passes its own args before "--", so filter.
    args, _ = parser.parse_known_args()
    return args

args = parse_args()
DATA_DIR = Path(args.data_dir)

# ---------------------------------------------------------------------------
# Load manifest
# ---------------------------------------------------------------------------

@st.cache_data
def load_manifest():
    with open(DATA_DIR / "manifest.json") as f:
        return json.load(f)

@st.cache_data
def load_example(filename: str):
    with open(DATA_DIR / filename) as f:
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

# Custom CSS for dark gray theme and token styling.
st.markdown("""
<style>
    .stApp { background-color: #1e1e1e; }
    .token-container {
        line-height: 2.4;
        padding: 16px;
        background: #2a2a2a;
        border-radius: 8px;
        font-family: 'SF Mono', 'Fira Code', 'Consolas', monospace;
        font-size: 13px;
    }
    .tok {
        display: inline;
        padding: 2px 1px;
        border-radius: 3px;
        cursor: pointer;
        white-space: pre-wrap;
    }
    .tok:hover { outline: 2px solid #fff; outline-offset: 1px; }
    .tok.selected { outline: 2px solid #e94560; outline-offset: 1px; }
</style>
""", unsafe_allow_html=True)

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
layer = st.sidebar.select_slider(
    "Layer",
    options=layers,
    value=default_layer,
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
    """Map score [0,1] to rgba color. Blue → green → yellow → red."""
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
# Render tokens
# ---------------------------------------------------------------------------

import html as html_mod

token_html_parts = []
for j, (tok, score) in enumerate(zip(token_strs, layer_scores)):
    bg = score_color(score)
    escaped = html_mod.escape(tok)
    # Use non-breaking markers for newlines so they render.
    escaped = escaped.replace("\n", "<br>")
    token_html_parts.append(
        f'<span class="tok" data-idx="{j}" title="pos {j} | score {score:.3f}" '
        f'style="background:{bg};">{escaped}</span>'
    )

st.markdown(
    f'<div class="token-container">{"".join(token_html_parts)}</div>',
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Token detail: pick a token position to inspect
# ---------------------------------------------------------------------------

st.markdown("---")

col1, col2 = st.columns([1, 3])

with col1:
    selected_token = st.number_input(
        "Token position",
        min_value=0,
        max_value=num_tokens - 1,
        value=0,
        help="Enter a token position or click a score in the chart",
    )
    tok_display = repr(token_strs[selected_token])
    st.code(tok_display, language=None)
    st.caption(f"Score at L{layer}: **{layer_scores[selected_token]:.4f}**")

with col2:
    # Line chart: score across layers for the selected token.
    import pandas as pd

    all_layers = example["layers"]
    chart_data = {}

    # Show selected probe + optionally all probes.
    show_all = st.checkbox("Show all probe types", value=False)

    if show_all:
        for pt in example["probe_types"]:
            pt_scores = example["scores"].get(pt, {})
            chart_data[pt] = [
                pt_scores.get(str(l), [0.0] * num_tokens)[selected_token]
                for l in all_layers
            ]
    else:
        chart_data[probe_type] = [
            scores_for_probe.get(str(l), [0.0] * num_tokens)[selected_token]
            for l in all_layers
        ]

    df = pd.DataFrame(chart_data, index=[f"L{l}" for l in all_layers])
    st.line_chart(df, height=250, use_container_width=True)

    # Find peak.
    peak_scores = [
        scores_for_probe.get(str(l), [0.0] * num_tokens)[selected_token]
        for l in all_layers
    ]
    peak_layer = all_layers[peak_scores.index(max(peak_scores))]
    st.caption(f"Peak for **{probe_type}**: Layer {peak_layer} (score {max(peak_scores):.4f})")

# ---------------------------------------------------------------------------
# Legend
# ---------------------------------------------------------------------------

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
