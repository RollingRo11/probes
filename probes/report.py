"""Generate an interactive HTML report from probe training results."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


# Consistent color palette for probe types.
PROBE_COLORS = {
    "mean_diff": "#636EFA",
    "linear": "#EF553B",
    "ema": "#00CC96",
    "mlp": "#AB63FA",
    "attention": "#FFA15A",
    "multimax": "#19D3F3",
    "max_rolling_mean": "#FF6692",
}


def _color(probe_type: str) -> str:
    return PROBE_COLORS.get(probe_type, "#B6B6B6")


def generate_report(log_dir: Path, config: dict | None = None) -> Path:
    """Read summary.csv and write an interactive HTML report.

    Returns the path to the generated HTML file.
    """
    csv_path = log_dir / "summary.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"No summary.csv in {log_dir}")

    df = pd.read_csv(csv_path)

    # Ensure numeric columns.
    for col in ["layer", "auroc", "weighted_error", "val_auroc", "val_weighted_error",
                "accuracy", "fnr", "fpr", "elapsed_s", "n_seeds"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    probe_types = sorted(df["probe_type"].unique(), key=lambda p: list(PROBE_COLORS).index(p) if p in PROBE_COLORS else 99)

    figs = []

    # --- Figure 1: AUROC by layer ---
    fig_auroc = go.Figure()
    for pt in probe_types:
        sub = df[df["probe_type"] == pt].sort_values("layer")
        fig_auroc.add_trace(go.Scatter(
            x=sub["layer"], y=sub["auroc"],
            mode="lines+markers", name=pt,
            line=dict(color=_color(pt), width=2),
            marker=dict(size=5),
            hovertemplate="Layer %{x}<br>AUROC: %{y:.4f}<extra>" + pt + "</extra>",
        ))
    fig_auroc.add_hline(y=0.5, line_dash="dash", line_color="gray",
                        annotation_text="chance", annotation_position="bottom left")
    fig_auroc.update_layout(
        title="Test AUROC by Layer",
        xaxis_title="Layer", yaxis_title="AUROC",
        yaxis_range=[0, 1.05],
        legend_title="Probe Type",
        template="plotly_white",
        height=500,
    )
    figs.append(("auroc_by_layer", fig_auroc))

    # --- Figure 2: Weighted Error by layer ---
    if "weighted_error" in df.columns and df["weighted_error"].notna().any():
        fig_we = go.Figure()
        for pt in probe_types:
            sub = df[df["probe_type"] == pt].sort_values("layer")
            fig_we.add_trace(go.Scatter(
                x=sub["layer"], y=sub["weighted_error"],
                mode="lines+markers", name=pt,
                line=dict(color=_color(pt), width=2),
                marker=dict(size=5),
                hovertemplate="Layer %{x}<br>WE: %{y:.2f}<extra>" + pt + "</extra>",
            ))
        fig_we.update_layout(
            title="Weighted Error by Layer (lower is better)",
            xaxis_title="Layer", yaxis_title="Weighted Error",
            legend_title="Probe Type",
            template="plotly_white",
            height=500,
        )
        figs.append(("weighted_error_by_layer", fig_we))

    # --- Figure 3: Heatmap of AUROC (probe_type x layer) ---
    pivot = df.pivot_table(index="probe_type", columns="layer", values="auroc")
    pivot = pivot.reindex(probe_types)
    fig_heat = go.Figure(data=go.Heatmap(
        z=pivot.values,
        x=[str(c) for c in pivot.columns],
        y=list(pivot.index),
        colorscale="RdYlGn",
        zmin=0, zmax=1,
        colorbar_title="AUROC",
        hovertemplate="Probe: %{y}<br>Layer: %{x}<br>AUROC: %{z:.4f}<extra></extra>",
    ))
    fig_heat.update_layout(
        title="AUROC Heatmap (Probe Type x Layer)",
        xaxis_title="Layer", yaxis_title="Probe Type",
        template="plotly_white",
        height=max(300, 60 * len(probe_types) + 100),
    )
    figs.append(("auroc_heatmap", fig_heat))

    # --- Figure 4: Best probe per layer bar chart ---
    best_idx = df.groupby("layer")["auroc"].idxmax()
    best = df.loc[best_idx].sort_values("layer")
    fig_best = go.Figure()
    fig_best.add_trace(go.Bar(
        x=best["layer"], y=best["auroc"],
        marker_color=[_color(pt) for pt in best["probe_type"]],
        text=best["probe_type"],
        textposition="outside",
        hovertemplate="Layer %{x}<br>AUROC: %{y:.4f}<br>Probe: %{text}<extra></extra>",
    ))
    fig_best.add_hline(y=0.5, line_dash="dash", line_color="gray")
    fig_best.update_layout(
        title="Best Probe per Layer",
        xaxis_title="Layer", yaxis_title="AUROC",
        yaxis_range=[0, 1.05],
        template="plotly_white",
        height=400,
    )
    figs.append(("best_per_layer", fig_best))

    # --- Summary statistics table ---
    summary_rows = []
    for pt in probe_types:
        sub = df[df["probe_type"] == pt]
        best_row = sub.loc[sub["auroc"].idxmax()]
        summary_rows.append({
            "Probe": pt,
            "Best Layer": int(best_row["layer"]),
            "Best AUROC": f"{best_row['auroc']:.4f}",
            "Mean AUROC": f"{sub['auroc'].mean():.4f}",
            "Best WE": f"{best_row.get('weighted_error', float('nan')):.2f}",
            "Accuracy": f"{best_row.get('accuracy', float('nan')):.2%}" if pd.notna(best_row.get("accuracy")) else "—",
        })

    # --- Build HTML ---
    html_parts = [
        "<!DOCTYPE html>",
        "<html><head>",
        '<meta charset="utf-8">',
        "<title>Probe Training Report</title>",
        '<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>',
        "<style>",
        "  body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;",
        "         max-width: 1200px; margin: 0 auto; padding: 20px; background: #fafafa; }",
        "  h1 { color: #333; border-bottom: 2px solid #ddd; padding-bottom: 10px; }",
        "  h2 { color: #555; margin-top: 40px; }",
        "  .chart { background: white; border-radius: 8px; box-shadow: 0 1px 3px rgba(0,0,0,0.1);",
        "           padding: 10px; margin: 20px 0; }",
        "  table { border-collapse: collapse; width: 100%; background: white;",
        "          border-radius: 8px; overflow: hidden; box-shadow: 0 1px 3px rgba(0,0,0,0.1); }",
        "  th { background: #4a5568; color: white; padding: 12px 16px; text-align: left; }",
        "  td { padding: 10px 16px; border-bottom: 1px solid #e2e8f0; }",
        "  tr:hover td { background: #f7fafc; }",
        "  .meta { color: #888; font-size: 0.9em; margin-bottom: 30px; }",
        "</style>",
        "</head><body>",
        "<h1>Probe Training Report</h1>",
        f'<p class="meta">Log directory: <code>{log_dir}</code><br>',
        f"Probe types: {', '.join(probe_types)}<br>",
        f"Layers: {int(df['layer'].min())}–{int(df['layer'].max())} ({df['layer'].nunique()} layers)<br>",
        f"Total runs: {len(df)}</p>",
    ]

    # Summary table
    html_parts.append("<h2>Summary</h2>")
    html_parts.append("<table><thead><tr>")
    for col in summary_rows[0]:
        html_parts.append(f"<th>{col}</th>")
    html_parts.append("</tr></thead><tbody>")
    for row in summary_rows:
        html_parts.append("<tr>")
        for v in row.values():
            html_parts.append(f"<td>{v}</td>")
        html_parts.append("</tr>")
    html_parts.append("</tbody></table>")

    # Charts
    for div_id, fig in figs:
        fig_json = fig.to_json()
        html_parts.append(f'<div class="chart" id="{div_id}"></div>')
        html_parts.append("<script>")
        html_parts.append(f"(function() {{ var fig = {fig_json}; Plotly.newPlot('{div_id}', fig.data, fig.layout, {{responsive: true}}); }})();")
        html_parts.append("</script>")

    html_parts.append("</body></html>")

    out_path = log_dir / "report.html"
    out_path.write_text("\n".join(html_parts))
    return out_path
