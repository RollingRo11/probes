"""Streamlit dashboard for probe score visualization.

Shows prompt-time scores and generation-time score trajectories
from generate_and_probe.py results.

Usage:
    uv run streamlit run dashboard.py
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

st.set_page_config(page_title="Probe Dashboard", layout="wide")
st.title("Evaluation Awareness Probe Dashboard")

# ---------------------------------------------------------------------------
# Sidebar: file selection
# ---------------------------------------------------------------------------

st.sidebar.header("Data")

# Find all gen_probe_*.json files.
results_files = sorted(Path(".").rglob("gen_probe_*.json"))
if not results_files:
    st.error("No `gen_probe_*.json` files found. Run `scripts/generate_and_probe.py` first.")
    st.stop()

file_options = {str(f): f for f in results_files}
selected_file = st.sidebar.selectbox(
    "Results file",
    options=list(file_options.keys()),
    format_func=lambda x: x.replace("logs/", "").replace("/gen_probe_", " → "),
)

data = json.loads(file_options[selected_file].read_text())
results = data["results"]
layers = [str(l) for l in data["layers"]]
probe_type = data.get("probe_type", "linear")
model_name = data.get("model", "unknown")
dataset_name = Path(data.get("dataset", "")).stem

st.sidebar.markdown(f"""
**Model:** `{model_name}`
**Probe:** `{probe_type}`
**Dataset:** `{dataset_name}`
**Examples:** {len(results)}
**Layers:** {', '.join(layers)}
""")

# Filter by label.
all_labels = sorted(set(r["label"] for r in results if r["label"] is not None))
if all_labels:
    label_filter = st.sidebar.multiselect(
        "Filter by label", all_labels, default=all_labels,
    )
    results = [r for r in results if r["label"] in label_filter]

if not results:
    st.warning("No examples match the current filter.")
    st.stop()

# Layer selection for detailed views.
selected_layer = st.sidebar.selectbox("Primary layer", layers, index=len(layers) // 2)

# ---------------------------------------------------------------------------
# Tab layout
# ---------------------------------------------------------------------------

tab_overview, tab_trajectories, tab_examples, tab_compare = st.tabs([
    "Overview", "Trajectories", "Per-Example", "Prompt vs. Final",
])

# ---------------------------------------------------------------------------
# Tab 1: Overview
# ---------------------------------------------------------------------------

with tab_overview:
    st.header("Prompt-Time Scores by Layer")

    # Build a DataFrame of prompt scores.
    prompt_rows = []
    for r in results:
        for l in layers:
            prompt_rows.append({
                "index": r["index"],
                "label": r["label"],
                "source": r.get("source", ""),
                "layer": int(l),
                "prompt_score": r["prompt_scores"].get(l, np.nan),
            })
    df_prompt = pd.DataFrame(prompt_rows)

    # Box plot of prompt scores by layer, colored by label.
    fig_box = px.box(
        df_prompt, x="layer", y="prompt_score", color="label",
        title="Prompt-Time Probe Score Distribution by Layer",
        labels={"prompt_score": "Probe Score", "layer": "Layer", "label": "Label"},
        color_discrete_map={0: "#636EFA", 1: "#EF553B"},
    )
    fig_box.update_layout(height=450)
    st.plotly_chart(fig_box, use_container_width=True)

    # AUROC-like separation per layer.
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Mean Prompt Score by Layer & Label")
        pivot = df_prompt.groupby(["layer", "label"])["prompt_score"].mean().unstack()
        fig_mean = go.Figure()
        for lab in sorted(pivot.columns):
            name = "Deploy (0)" if lab == 0 else "Eval (1)"
            color = "#636EFA" if lab == 0 else "#EF553B"
            fig_mean.add_trace(go.Scatter(
                x=pivot.index, y=pivot[lab], mode="lines+markers",
                name=name, line=dict(color=color),
            ))
        fig_mean.update_layout(
            xaxis_title="Layer", yaxis_title="Mean Probe Score", height=350,
        )
        st.plotly_chart(fig_mean, use_container_width=True)

    with col2:
        st.subheader("Score Separation (Eval - Deploy)")
        if 0 in pivot.columns and 1 in pivot.columns:
            separation = pivot[1] - pivot[0]
            fig_sep = px.bar(
                x=separation.index, y=separation.values,
                labels={"x": "Layer", "y": "Score Separation"},
                color=separation.values,
                color_continuous_scale="RdBu",
                color_continuous_midpoint=0,
            )
            fig_sep.update_layout(height=350)
            st.plotly_chart(fig_sep, use_container_width=True)
        else:
            st.info("Need both labels (0 and 1) to show separation.")

# ---------------------------------------------------------------------------
# Tab 2: Trajectories
# ---------------------------------------------------------------------------

with tab_trajectories:
    st.header(f"Generation-Time Score Trajectories — Layer {selected_layer}")

    # Aggregate trajectories.
    max_len = max(
        len(r["trajectories"].get(selected_layer, []))
        for r in results
    )

    if max_len == 0:
        st.warning(f"No trajectory data for layer {selected_layer}.")
    else:
        # Individual traces.
        fig_traj = go.Figure()

        # Group by label.
        for label_val, color, name in [(0, "#636EFA", "Deploy"), (1, "#EF553B", "Eval")]:
            label_results = [r for r in results if r["label"] == label_val]
            for r in label_results:
                traj = r["trajectories"].get(selected_layer, [])
                if not traj:
                    continue
                fig_traj.add_trace(go.Scatter(
                    x=list(range(len(traj))),
                    y=traj,
                    mode="lines",
                    line=dict(color=color, width=0.7),
                    opacity=0.3,
                    name=name,
                    legendgroup=name,
                    showlegend=False,
                    hovertemplate=(
                        f"Example {r['index']}<br>"
                        f"Token: %{{x}}<br>Score: %{{y:.4f}}<br>"
                        f"Label: {label_val}"
                    ),
                ))

        # Mean trajectories.
        for label_val, color, name in [(0, "#636EFA", "Deploy (mean)"), (1, "#EF553B", "Eval (mean)")]:
            label_results = [r for r in results if r["label"] == label_val]
            trajs = [r["trajectories"].get(selected_layer, []) for r in label_results]
            if not trajs:
                continue
            # Pad to max length.
            padded = np.full((len(trajs), max_len), np.nan)
            for i, t in enumerate(trajs):
                padded[i, :len(t)] = t
            mean_traj = np.nanmean(padded, axis=0)
            fig_traj.add_trace(go.Scatter(
                x=list(range(len(mean_traj))),
                y=mean_traj,
                mode="lines",
                line=dict(color=color, width=3),
                name=name,
            ))

        fig_traj.update_layout(
            title=f"Probe Score During Generation — Layer {selected_layer}",
            xaxis_title="Generated Token Index",
            yaxis_title="Probe Score",
            height=500,
            legend=dict(yanchor="top", y=0.99, xanchor="right", x=0.99),
        )
        st.plotly_chart(fig_traj, use_container_width=True)

        # Multi-layer trajectory heatmap for mean across examples.
        st.subheader("Mean Trajectory Across All Layers")

        heatmap_data = []
        for l in layers:
            trajs = [r["trajectories"].get(l, []) for r in results]
            if not trajs or all(len(t) == 0 for t in trajs):
                heatmap_data.append([np.nan] * max_len)
                continue
            padded = np.full((len(trajs), max_len), np.nan)
            for i, t in enumerate(trajs):
                padded[i, :len(t)] = t
            heatmap_data.append(np.nanmean(padded, axis=0).tolist())

        fig_heat = px.imshow(
            heatmap_data,
            x=list(range(max_len)),
            y=[f"L{l}" for l in layers],
            aspect="auto",
            color_continuous_scale="RdBu",
            color_continuous_midpoint=np.nanmedian(heatmap_data),
            labels={"x": "Token Index", "y": "Layer", "color": "Score"},
            title="Mean Probe Score Heatmap (all layers × generation steps)",
        )
        fig_heat.update_layout(height=350)
        st.plotly_chart(fig_heat, use_container_width=True)

        # Prompt vs generation score shift.
        st.subheader("Score Shift: Prompt → Generation")
        shift_rows = []
        for r in results:
            for l in layers:
                traj = r["trajectories"].get(l, [])
                ps = r["prompt_scores"].get(l, np.nan)
                if traj:
                    shift_rows.append({
                        "index": r["index"],
                        "label": r["label"],
                        "layer": int(l),
                        "prompt_score": ps,
                        "final_score": traj[-1],
                        "delta": traj[-1] - ps,
                        "max_score": max(traj),
                        "min_score": min(traj),
                    })

        df_shift = pd.DataFrame(shift_rows)
        if not df_shift.empty:
            fig_delta = px.box(
                df_shift, x="layer", y="delta", color="label",
                title="Score Change (Final Gen Token - Prompt) by Layer",
                labels={"delta": "Δ Score", "layer": "Layer"},
                color_discrete_map={0: "#636EFA", 1: "#EF553B"},
            )
            fig_delta.update_layout(height=400)
            st.plotly_chart(fig_delta, use_container_width=True)

# ---------------------------------------------------------------------------
# Tab 3: Per-Example Browser
# ---------------------------------------------------------------------------

with tab_examples:
    st.header("Per-Example Browser")

    example_idx = st.selectbox(
        "Select example",
        range(len(results)),
        format_func=lambda i: (
            f"#{results[i]['index']} — "
            f"label={results[i]['label']} — "
            f"{results[i]['prompt'][:80]}..."
        ),
    )

    r = results[example_idx]

    col_info, col_scores = st.columns([1, 1])

    with col_info:
        st.subheader("Prompt")
        st.text_area("", r["prompt"], height=150, disabled=True, key="prompt_text")

        st.subheader("Completion")
        st.text_area("", r["completion"], height=200, disabled=True, key="completion_text")

        st.caption(
            f"Label: **{r['label']}** | Source: {r.get('source', 'N/A')} | "
            f"Prompt tokens: {r['prompt_len_tokens']} | "
            f"Gen tokens: {r['completion_len_tokens']}"
        )

    with col_scores:
        st.subheader("Prompt-Time Scores")

        ps_data = {f"L{k}": v for k, v in r["prompt_scores"].items()}
        fig_ps = px.bar(
            x=list(ps_data.keys()), y=list(ps_data.values()),
            labels={"x": "Layer", "y": "Score"},
            title="Prompt Score by Layer",
        )
        fig_ps.update_layout(height=250)
        st.plotly_chart(fig_ps, use_container_width=True)

        st.subheader("Generation Trajectory")
        fig_ex_traj = go.Figure()
        for l in layers:
            traj = r["trajectories"].get(l, [])
            if traj:
                fig_ex_traj.add_trace(go.Scatter(
                    x=list(range(len(traj))),
                    y=traj,
                    mode="lines",
                    name=f"L{l}",
                ))
        fig_ex_traj.update_layout(
            title="Score Trajectory During Generation",
            xaxis_title="Token Index",
            yaxis_title="Score",
            height=300,
        )
        st.plotly_chart(fig_ex_traj, use_container_width=True)

    # Token-level detail.
    if st.checkbox("Show token-level scores", key="token_detail"):
        tokens = r.get("generated_tokens", [])
        traj = r["trajectories"].get(selected_layer, [])
        if tokens and traj:
            token_df = pd.DataFrame({
                "token_idx": range(len(tokens)),
                "token": [t.replace("\n", "\\n") for t in tokens[:len(traj)]],
                "score": traj[:len(tokens)],
            })
            st.dataframe(token_df, use_container_width=True, height=300)

# ---------------------------------------------------------------------------
# Tab 4: Prompt vs Final Scatter
# ---------------------------------------------------------------------------

with tab_compare:
    st.header(f"Prompt Score vs. Final Generation Score — Layer {selected_layer}")

    scatter_rows = []
    for r in results:
        traj = r["trajectories"].get(selected_layer, [])
        ps = r["prompt_scores"].get(selected_layer, np.nan)
        if traj:
            scatter_rows.append({
                "index": r["index"],
                "label": r["label"],
                "source": r.get("source", ""),
                "prompt_score": ps,
                "final_score": traj[-1],
                "mean_gen_score": np.mean(traj),
                "max_gen_score": max(traj),
                "prompt": r["prompt"][:100],
                "gen_tokens": r["completion_len_tokens"],
            })

    df_scatter = pd.DataFrame(scatter_rows)

    if df_scatter.empty:
        st.warning("No data to plot.")
    else:
        fig_scatter = px.scatter(
            df_scatter,
            x="prompt_score",
            y="final_score",
            color="label",
            hover_data=["prompt", "source", "gen_tokens"],
            title=f"Prompt vs. Final Generation Score (Layer {selected_layer})",
            labels={
                "prompt_score": "Prompt-Time Score",
                "final_score": "Final Gen Token Score",
            },
            color_discrete_map={0: "#636EFA", 1: "#EF553B"},
        )
        # Add diagonal line.
        score_range = [
            min(df_scatter["prompt_score"].min(), df_scatter["final_score"].min()),
            max(df_scatter["prompt_score"].max(), df_scatter["final_score"].max()),
        ]
        fig_scatter.add_trace(go.Scatter(
            x=score_range, y=score_range,
            mode="lines", line=dict(dash="dash", color="gray"),
            name="y=x (no change)", showlegend=True,
        ))
        fig_scatter.update_layout(height=500)
        st.plotly_chart(fig_scatter, use_container_width=True)

        st.caption(
            "Points above the diagonal: score increased during generation. "
            "Points below: score decreased."
        )

        # Stats table.
        st.subheader("Summary Statistics")
        summary = df_scatter.groupby("label").agg({
            "prompt_score": ["mean", "std"],
            "final_score": ["mean", "std"],
        }).round(4)
        summary.columns = [
            "Prompt Mean", "Prompt Std", "Final Mean", "Final Std",
        ]
        summary["Δ Mean"] = (summary["Final Mean"] - summary["Prompt Mean"]).round(4)
        st.dataframe(summary)
