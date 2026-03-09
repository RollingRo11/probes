#!/usr/bin/env python
"""Aggregate probe results and generate visualisations.

Reads all metrics.json files from a run's results/ directory and produces:
  - summary.csv      : one row per (layer, probe_type)
  - auroc_heatmap.html : Plotly heatmap of AUROC across layers × probe types
  - layer_curves.html  : AUROC as a function of layer depth per probe type

Usage
-----
  uv run python scripts/analyze_results.py --log-dir logs/<run>
  uv run python scripts/analyze_results.py --log-dir logs/<run> --metric auroc
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def load_results(log_dir: Path) -> list[dict]:
    rows = []
    for path in sorted((log_dir / "results").glob("layer_*/**/metrics.json")):
        with open(path) as f:
            rows.append(json.load(f))
    if not rows:
        raise FileNotFoundError(f"No metrics.json found under {log_dir}/results/")
    return rows


def write_csv(rows: list[dict], path: Path) -> None:
    scalar_rows = []
    for r in rows:
        scalar_rows.append({k: v for k, v in r.items() if not isinstance(v, (list, dict))})
    fieldnames = sorted({k for r in scalar_rows for k in r})
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(scalar_rows)
    print(f"CSV → {path}")


def plot_heatmap(rows: list[dict], metric: str, out_path: Path) -> None:
    import plotly.graph_objects as go

    probe_types = sorted({r["probe_type"] for r in rows})
    layers = sorted({r["layer"] for r in rows})

    z = []
    for pt in probe_types:
        row_vals = []
        by_layer = {r["layer"]: r.get(metric, None) for r in rows if r["probe_type"] == pt}
        for l in layers:
            v = by_layer.get(l, None)
            row_vals.append(float(v) if v is not None else None)
        z.append(row_vals)

    fig = go.Figure(
        data=go.Heatmap(
            z=z,
            x=[str(l) for l in layers],
            y=probe_types,
            colorscale="Viridis",
            text=[[f"{v:.3f}" if v is not None else "" for v in row] for row in z],
            texttemplate="%{text}",
            colorbar=dict(title=metric.upper()),
        )
    )
    fig.update_layout(
        title=f"{metric.upper()} by layer and probe type",
        xaxis_title="Layer",
        yaxis_title="Probe type",
        height=max(300, 80 * len(probe_types)),
    )
    fig.write_html(str(out_path))
    print(f"Heatmap → {out_path}")


def plot_layer_curves(rows: list[dict], metric: str, out_path: Path) -> None:
    import plotly.graph_objects as go

    probe_types = sorted({r["probe_type"] for r in rows})
    fig = go.Figure()

    for pt in probe_types:
        pt_rows = sorted([r for r in rows if r["probe_type"] == pt], key=lambda r: r["layer"])
        xs = [r["layer"] for r in pt_rows]
        ys = [r.get(metric, None) for r in pt_rows]
        ys = [float(v) if v is not None else None for v in ys]
        fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines+markers", name=pt))

    fig.update_layout(
        title=f"{metric.upper()} vs layer depth",
        xaxis_title="Layer",
        yaxis_title=metric.upper(),
        legend_title="Probe type",
    )
    fig.write_html(str(out_path))
    print(f"Layer curves → {out_path}")


def print_summary(rows: list[dict], metric: str) -> None:
    probe_types = sorted({r["probe_type"] for r in rows})
    print(f"\n{'='*60}")
    print(f"Best {metric.upper()} per probe type")
    print("=" * 60)
    for pt in probe_types:
        vals = [r.get(metric) for r in rows if r["probe_type"] == pt and r.get(metric) is not None]
        if not vals:
            continue
        best = max(vals)
        best_layer = next(
            r["layer"] for r in rows
            if r["probe_type"] == pt and r.get(metric) == best
        )
        print(f"  {pt:>20}  best={best:.4f}  at layer {best_layer}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Analyze probe training results")
    ap.add_argument("--log-dir", required=True)
    ap.add_argument(
        "--metric",
        default="auroc",
        choices=["auroc", "weighted_error", "accuracy", "fnr", "fpr"],
    )
    args = ap.parse_args()

    log_dir = Path(args.log_dir)
    rows = load_results(log_dir)

    plots_dir = log_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    write_csv(rows, log_dir / "summary.csv")
    plot_heatmap(rows, args.metric, plots_dir / f"{args.metric}_heatmap.html")
    plot_layer_curves(rows, args.metric, plots_dir / f"{args.metric}_by_layer.html")
    print_summary(rows, args.metric)


if __name__ == "__main__":
    main()
