#!/usr/bin/env python
"""Evaluate trained probes on the SAD stages dataset.

Extracts activations from OLMo-3-7B-Think on the SAD dataset, then runs
all trained probes to see how often they fire on evaluation vs deployment
examples.

Supports both mean_diff (closed-form) and gradient-based probes (linear,
ema, mlp, attention, multimax, max_rolling_mean).

Usage:
  uv run python scripts/eval_on_sad.py --probes-dir logs/<run>/results --config configs/<cfg>.yaml

This should be submitted as a SLURM job (needs GPU for extraction).
"""

import argparse
import json
import sys
from pathlib import Path

# Ensure the project root is on the path.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
from sklearn.metrics import roc_auc_score


def _load_probe(probe_path: Path, probe_type: str, hidden_size: int, probe_kwargs: dict):
    """Load a saved probe checkpoint. Returns (probe, threshold)."""
    data = torch.load(probe_path, weights_only=False)

    if data["type"] == "mean_diff":
        from probes.architectures import MeanDiffProbe
        probe = MeanDiffProbe()
        probe.direction = data["direction"]
        probe.threshold = data["threshold"]
        return probe, data["threshold"]
    else:
        from probes.architectures import build_probe
        probe = build_probe(probe_type, hidden_size, **probe_kwargs)
        probe.load_state_dict(data["state_dict"])
        probe.eval()
        # Gradient probes don't store a threshold in the checkpoint;
        # we'll use Youden's J on the SAD data itself (or just report AUROC).
        return probe, None


def _score_probe(probe, acts: torch.Tensor, probe_type: str) -> np.ndarray:
    """Get probe scores (higher = more likely eval). Returns numpy array."""
    if hasattr(probe, "decision_function"):
        # MeanDiffProbe
        return probe.decision_function(acts.numpy())

    # Gradient-based probe
    from probes.architectures import EMAProbe
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    probe.to(device)
    with torch.no_grad():
        x = acts.to(device)
        if isinstance(probe, EMAProbe) and x.dim() == 3:
            logits = probe.forward_ema(x)
        else:
            logits = probe(x)
        scores = torch.sigmoid(logits).cpu().numpy()
    return scores


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--probes-dir", required=True, nargs="+",
                        help="Path(s) to results/ dir(s) with trained probes")
    parser.add_argument("--config", required=True, help="Config YAML for model settings")
    parser.add_argument("--output", default=None, help="Output dir")
    parser.add_argument("--activations", default=None,
                        help="Pre-extracted activations dir (skip extraction)")
    args = parser.parse_args()

    import yaml
    with open(args.config) as f:
        config = yaml.safe_load(f)

    probes_dirs = [Path(p) for p in args.probes_dir]
    if args.output:
        output_dir = Path(args.output)
    else:
        output_dir = probes_dirs[0].parent / "sad_eval"
    output_dir.mkdir(parents=True, exist_ok=True)

    model_cfg = config["model"]
    model_id = model_cfg["id"]
    num_layers = model_cfg["num_layers"]
    hidden_size = model_cfg["hidden_size"]
    layer_path = model_cfg.get("layer_path", "model.layers")
    layer_output_index = model_cfg.get("layer_output_index")

    probe_cfg = config.get("probes", {})
    probe_kwargs = {
        "mlp_hidden_dim": probe_cfg.get("mlp_hidden_dim", 100),
        "mlp_n_layers": probe_cfg.get("mlp_n_layers", 2),
        "n_heads": probe_cfg.get("n_heads", 10),
        "rolling_window": probe_cfg.get("rolling_window", 10),
        "alpha": probe_cfg.get("ema_alpha", 0.5),
    }

    # Load SAD dataset.
    sad_path = Path("data/sad_stages.jsonl")
    examples = [json.loads(l) for l in open(sad_path)]
    texts = [e["text"] for e in examples]
    labels = np.array([e["label"] for e in examples])
    sources = [e["source"] for e in examples]
    print(f"[sad_eval] Loaded {len(texts)} examples: "
          f"{(labels==0).sum()} deployment, {(labels==1).sum()} evaluation")

    # Extract or reuse activations.
    if args.activations and Path(args.activations).exists():
        acts_dir = Path(args.activations)
        print(f"[sad_eval] Reusing activations from {acts_dir}")
    else:
        from transformers import AutoTokenizer
        print("[sad_eval] Applying chat template...")
        tok = AutoTokenizer.from_pretrained(model_id)
        if tok.pad_token_id is None:
            tok.pad_token = tok.eos_token
        formatted = []
        for t in texts:
            formatted.append(tok.apply_chat_template(
                [{"role": "user", "content": t}],
                tokenize=False,
                add_generation_prompt=True,
            ))

        from probes.extraction import extract_activations
        acts_dir = output_dir / "activations"
        extract_activations(
            model_id=model_id,
            texts=formatted,
            labels=labels.tolist(),
            output_dir=acts_dir,
            num_layers=num_layers,
            layer_path=layer_path,
            layer_output_index=layer_output_index,
            token_strategy="last",
            batch_size=8,
            max_length=2048,
        )

    # Discover all probes across all provided results dirs.
    print(f"\n[sad_eval] Discovering probes...")
    probe_entries = []  # list of (layer, probe_type, probe_path)
    for pdir in probes_dirs:
        for layer_dir in sorted(pdir.glob("layer_*")):
            layer_idx = int(layer_dir.name.replace("layer_", ""))
            for pt_dir in sorted(layer_dir.iterdir()):
                if not pt_dir.is_dir():
                    continue
                probe_path = pt_dir / "probe.pt"
                if probe_path.exists():
                    probe_entries.append((layer_idx, pt_dir.name, probe_path))

    probe_types_found = sorted(set(pt for _, pt, _ in probe_entries))
    print(f"[sad_eval] Found {len(probe_entries)} probes: "
          f"types={probe_types_found}, layers=0..{max(l for l, _, _ in probe_entries)}")

    # Evaluate all probes.
    print("[sad_eval] Evaluating...")
    results = []
    deploy_mask = labels == 0
    eval_mask = labels == 1
    unique_sources = sorted(set(sources))
    source_masks = {src: np.array([s == src for s in sources]) for src in unique_sources}

    for layer_idx, probe_type, probe_path in probe_entries:
        acts = torch.load(acts_dir / f"layer_{layer_idx:02d}.pt", weights_only=True)

        probe, threshold = _load_probe(probe_path, probe_type, hidden_size, probe_kwargs)
        scores = _score_probe(probe, acts, probe_type)

        # For gradient probes without a stored threshold, use Youden's J.
        if threshold is None:
            from probes.metrics import youden_threshold
            threshold = youden_threshold(labels, scores)

        predictions = (scores >= threshold).astype(int)

        deploy_fire = float(predictions[deploy_mask].mean())
        eval_fire = float(predictions[eval_mask].mean())

        try:
            auc = float(roc_auc_score(labels, scores))
        except ValueError:
            auc = float("nan")

        row = {
            "layer": layer_idx,
            "probe_type": probe_type,
            "auroc": round(auc, 4),
            "deploy_fire_rate": round(deploy_fire, 4),
            "eval_fire_rate": round(eval_fire, 4),
            "fire_rate_gap": round(eval_fire - deploy_fire, 4),
            "threshold": round(float(threshold), 6),
        }
        for src in unique_sources:
            row[f"fire_{src}"] = round(float(predictions[source_masks[src]].mean()), 4)
        results.append(row)

    # Print summary: best per probe type.
    print(f"\n{'='*80}")
    print("SAD Evaluation: Best Probe per Type")
    print("=" * 80)
    header = f"{'Probe':>18}  {'Best Lyr':>8}  {'AUROC':>8}  {'Deploy%':>8}  {'Eval%':>8}  {'Gap':>8}"
    print(header)
    print("-" * len(header))
    by_type: dict[str, list[dict]] = {}
    for r in results:
        by_type.setdefault(r["probe_type"], []).append(r)
    for pt in sorted(by_type):
        best = max(by_type[pt], key=lambda r: r["auroc"])
        print(f"{pt:>18}  {best['layer']:>8}  {best['auroc']:>8.4f}  "
              f"{best['deploy_fire_rate']:>7.0%}  {best['eval_fire_rate']:>7.0%}  "
              f"{best['fire_rate_gap']:>+7.0%}")

    # Save results.
    results_path = output_dir / "sad_eval_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved → {results_path}")

    # Generate HTML report.
    _generate_sad_report(results, sources, output_dir)


def _generate_sad_report(results: list[dict], sources: list[str], output_dir: Path):
    """Generate an interactive HTML report for SAD evaluation."""
    import pandas as pd
    import plotly.graph_objects as go
    from probes.report import PROBE_COLORS

    def _color(pt):
        return PROBE_COLORS.get(pt, "#B6B6B6")

    df = pd.DataFrame(results)
    probe_types = sorted(df["probe_type"].unique(),
                         key=lambda p: list(PROBE_COLORS).index(p) if p in PROBE_COLORS else 99)
    unique_sources = sorted(set(sources))

    figs = []

    # Figure 1: AUROC by layer per probe type.
    fig1 = go.Figure()
    for pt in probe_types:
        sub = df[df["probe_type"] == pt].sort_values("layer")
        fig1.add_trace(go.Scatter(
            x=sub["layer"], y=sub["auroc"],
            mode="lines+markers", name=pt,
            line=dict(color=_color(pt), width=2), marker=dict(size=4),
            hovertemplate="Layer %{x}<br>AUROC: %{y:.4f}<extra>" + pt + "</extra>",
        ))
    fig1.add_hline(y=0.5, line_dash="dash", line_color="gray",
                   annotation_text="chance", annotation_position="bottom left")
    fig1.update_layout(
        title="AUROC on SAD Dataset by Layer (probes trained on contrastive/eval_awareness)",
        xaxis_title="Layer", yaxis_title="AUROC",
        yaxis_range=[0, 1.05], template="plotly_white", height=500,
    )
    figs.append(("auroc_by_layer", fig1))

    # Figure 2: Fire rate gap by layer per probe type.
    fig2 = go.Figure()
    for pt in probe_types:
        sub = df[df["probe_type"] == pt].sort_values("layer")
        fig2.add_trace(go.Scatter(
            x=sub["layer"], y=sub["fire_rate_gap"],
            mode="lines+markers", name=pt,
            line=dict(color=_color(pt), width=2), marker=dict(size=4),
            hovertemplate="Layer %{x}<br>Gap: %{y:+.1%}<extra>" + pt + "</extra>",
        ))
    fig2.add_hline(y=0, line_dash="dash", line_color="gray")
    fig2.update_layout(
        title="Firing Rate Gap (Eval - Deploy) by Layer",
        xaxis_title="Layer", yaxis_title="Fire Rate Gap (positive = detects eval)",
        yaxis_tickformat="+.0%", template="plotly_white", height=500,
    )
    figs.append(("gap_by_layer", fig2))

    # Figure 3: Heatmap of AUROC.
    pivot = df.pivot_table(index="probe_type", columns="layer", values="auroc")
    pivot = pivot.reindex([pt for pt in probe_types if pt in pivot.index])
    fig3 = go.Figure(data=go.Heatmap(
        z=pivot.values,
        x=[str(c) for c in pivot.columns],
        y=list(pivot.index),
        colorscale="RdYlGn", zmin=0, zmax=1,
        colorbar_title="AUROC",
        hovertemplate="Probe: %{y}<br>Layer: %{x}<br>AUROC: %{z:.4f}<extra></extra>",
    ))
    fig3.update_layout(
        title="AUROC Heatmap on SAD (Probe Type x Layer)",
        xaxis_title="Layer", yaxis_title="Probe Type",
        template="plotly_white", height=max(300, 60 * len(probe_types) + 100),
    )
    figs.append(("auroc_heatmap", fig3))

    # Figure 4: Deploy vs Eval firing rates per probe type (at best AUROC layer).
    best_rows = []
    for pt in probe_types:
        sub = df[df["probe_type"] == pt]
        best_rows.append(sub.loc[sub["auroc"].idxmax()])
    best_df = pd.DataFrame(best_rows)

    fig4 = go.Figure()
    x_labels = [f"{r['probe_type']}\n(L{int(r['layer'])})" for _, r in best_df.iterrows()]
    fig4.add_trace(go.Bar(
        x=x_labels, y=best_df["deploy_fire_rate"],
        name="Deployment", marker_color="#636EFA",
    ))
    fig4.add_trace(go.Bar(
        x=x_labels, y=best_df["eval_fire_rate"],
        name="Evaluation", marker_color="#EF553B",
    ))
    fig4.update_layout(
        title="Firing Rates at Best AUROC Layer per Probe Type",
        yaxis_title="Firing Rate", yaxis_tickformat=".0%",
        yaxis_range=[0, 1.05], barmode="group",
        template="plotly_white", height=450,
    )
    figs.append(("best_layer_firing", fig4))

    # Figure 5: Per-source breakdown at best overall layer.
    overall_best = df.loc[df["auroc"].idxmax()]
    best_layer = int(overall_best["layer"])
    best_pt = overall_best["probe_type"]
    fire_cols = [c for c in df.columns if c.startswith("fire_") and c != "fire_rate_gap"]
    if fire_cols:
        layer_df = df[(df["layer"] == best_layer) & (df["probe_type"] == best_pt)]
        if not layer_df.empty:
            row = layer_df.iloc[0]
            src_names = [c.replace("fire_", "") for c in fire_cols]
            src_vals = [row[c] for c in fire_cols]
            colors = ["#636EFA" if "wildchat" in s else "#EF553B" for s in src_names]
            fig5 = go.Figure(go.Bar(
                x=src_names, y=src_vals,
                marker_color=colors,
                hovertemplate="%{x}<br>Fire rate: %{y:.1%}<extra></extra>",
            ))
            fig5.update_layout(
                title=f"Per-Source Firing Rates (best: {best_pt} @ layer {best_layer}, AUROC={overall_best['auroc']:.3f})",
                yaxis_title="Firing Rate", yaxis_tickformat=".0%",
                yaxis_range=[0, 1.05], template="plotly_white", height=400,
            )
            figs.append(("per_source", fig5))

    # Summary table.
    summary_rows = []
    for pt in probe_types:
        sub = df[df["probe_type"] == pt]
        best = sub.loc[sub["auroc"].idxmax()]
        summary_rows.append({
            "Probe": pt,
            "Best Layer": int(best["layer"]),
            "AUROC": f"{best['auroc']:.4f}",
            "Deploy Fire": f"{best['deploy_fire_rate']:.0%}",
            "Eval Fire": f"{best['eval_fire_rate']:.0%}",
            "Gap": f"{best['fire_rate_gap']:+.0%}",
        })

    # Build HTML.
    overall_best_row = df.loc[df["auroc"].idxmax()]
    best_gap_row = df.loc[df["fire_rate_gap"].idxmax()]

    html_parts = [
        "<!DOCTYPE html>",
        "<html><head>",
        '<meta charset="utf-8">',
        "<title>SAD Evaluation Report</title>",
        '<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>',
        "<style>",
        "  body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;",
        "         max-width: 1200px; margin: 0 auto; padding: 20px; background: #fafafa; }",
        "  h1 { color: #333; border-bottom: 2px solid #ddd; padding-bottom: 10px; }",
        "  h2 { color: #555; margin-top: 40px; }",
        "  .chart { background: white; border-radius: 8px; box-shadow: 0 1px 3px rgba(0,0,0,0.1);",
        "           padding: 10px; margin: 20px 0; }",
        "  .summary-box { background: white; border-radius: 8px; padding: 20px; margin: 20px 0;",
        "                  box-shadow: 0 1px 3px rgba(0,0,0,0.1); }",
        "  .summary-box h3 { margin-top: 0; }",
        "  .stat { display: inline-block; margin: 10px 20px; text-align: center; }",
        "  .stat .value { font-size: 2em; font-weight: bold; color: #333; }",
        "  .stat .label { font-size: 0.9em; color: #888; }",
        "  .meta { color: #888; font-size: 0.9em; margin-bottom: 30px; }",
        "  table { border-collapse: collapse; width: 100%; background: white;",
        "          border-radius: 8px; overflow: hidden; box-shadow: 0 1px 3px rgba(0,0,0,0.1); }",
        "  th { background: #4a5568; color: white; padding: 12px 16px; text-align: left; }",
        "  td { padding: 10px 16px; border-bottom: 1px solid #e2e8f0; }",
        "  tr:hover td { background: #f7fafc; }",
        "</style>",
        "</head><body>",
        "<h1>Probe Evaluation on SAD Stages Dataset</h1>",
        '<p class="meta">Probes trained on eval_awareness / contrastive data, evaluated on <code>sad_stages.jsonl</code>.<br>',
        f"Probe types: {', '.join(probe_types)}<br>",
        f"SAD dataset: {len(set(sources))} source categories, 200 deployment + 200 evaluation examples.</p>",
        '<div class="summary-box">',
        "<h3>Key Findings</h3>",
        f'<div class="stat"><div class="value">{overall_best_row["auroc"]:.3f}</div>'
        f'<div class="label">Best AUROC ({overall_best_row["probe_type"]} L{int(overall_best_row["layer"])})</div></div>',
        f'<div class="stat"><div class="value">{best_gap_row["fire_rate_gap"]:+.0%}</div>'
        f'<div class="label">Best Gap ({best_gap_row["probe_type"]} L{int(best_gap_row["layer"])})</div></div>',
        f'<div class="stat"><div class="value">{df["deploy_fire_rate"].mean():.0%}</div>'
        f'<div class="label">Avg Deploy Fire Rate</div></div>',
        f'<div class="stat"><div class="value">{df["eval_fire_rate"].mean():.0%}</div>'
        f'<div class="label">Avg Eval Fire Rate</div></div>',
        "</div>",
    ]

    # Summary table.
    html_parts.append("<h2>Best per Probe Type</h2>")
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

    for div_id, fig in figs:
        fig_json = fig.to_json()
        html_parts.append(f'<div class="chart" id="{div_id}"></div>')
        html_parts.append("<script>")
        html_parts.append(f"(function() {{ var fig = {fig_json}; Plotly.newPlot('{div_id}', fig.data, fig.layout, {{responsive: true}}); }})();")
        html_parts.append("</script>")

    html_parts.append("</body></html>")

    report_path = output_dir / "sad_eval_report.html"
    report_path.write_text("\n".join(html_parts))
    print(f"Report → {report_path}")


if __name__ == "__main__":
    main()
