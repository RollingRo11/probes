"""Surface baseline checks for eval-awareness probe.

Tests whether the probe might be learning surface artifacts rather than
a genuine eval-awareness belief state. Checks:

1. Prompt length baseline: Does text length predict probe scores?
2. Special character baseline: Do special chars / unicode predict scores?
3. Lexical baseline: Do simple keyword counts predict scores?
4. Leave-one-source-out: Does the probe generalize across data sources?

These are the negative controls recommended by Nguyen et al.
If the probe is just learning surface features, these baselines will
have comparable accuracy.

Usage:
    uv run python scripts/probe_baseline_checks.py \
        --run-dir logs/20260312_215816_olmo3-7b-think \
        --dataset data/test_awareness.jsonl \
        --score-layer 26
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def load_dataset(path):
    rows = []
    with open(path) as f:
        for line in f:
            rows.append(json.loads(line))
    return rows


def extract_thinking(text):
    """Extract the CoT/thinking portion of the text."""
    # Find the thinking section (between prompt and </think>)
    think_end = text.find("</think>")
    if think_end == -1:
        return text
    # Find where CoT starts (after the first newline following the prompt)
    first_newline = text.find("\n")
    if first_newline == -1 or first_newline > think_end:
        return text[:think_end]
    return text[first_newline:think_end]


def surface_features(text):
    """Extract surface features from text."""
    thinking = extract_thinking(text)
    prompt_end = text.find("\n")
    prompt = text[:prompt_end] if prompt_end != -1 else text[:200]

    return {
        "text_length": len(text),
        "prompt_length": len(prompt),
        "thinking_length": len(thinking),
        "n_special_chars": sum(1 for c in text if ord(c) > 127),
        "n_urls": len(re.findall(r'https?://\S+', text)),
        "n_question_marks": text.count("?"),
        "n_newlines": text.count("\n"),
        # Lexical features - eval-suggestive keywords in prompt
        "prompt_has_test": int(bool(re.search(r'\btest\b', prompt, re.I))),
        "prompt_has_eval": int(bool(re.search(r'\bevaluat', prompt, re.I))),
        "prompt_has_benchmark": int(bool(re.search(r'\bbenchmark\b', prompt, re.I))),
        # Lexical features - eval-awareness keywords in CoT
        "cot_has_test": int(bool(re.search(r'\btest\b', thinking, re.I))),
        "cot_has_eval": int(bool(re.search(r'\bevaluat', thinking, re.I))),
        "cot_has_guidelines": int(bool(re.search(r'\bguidelines?\b', thinking, re.I))),
        "cot_has_policies": int(bool(re.search(r'\bpolic(?:y|ies)\b', thinking, re.I))),
        "cot_has_boundaries": int(bool(re.search(r'\bboundar', thinking, re.I))),
    }


def logistic_regression_accuracy(X, y, n_splits=5):
    """Simple cross-validated logistic regression accuracy."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_score
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline

    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("lr", LogisticRegression(max_iter=1000, random_state=42)),
    ])
    scores = cross_val_score(pipe, X, y, cv=n_splits, scoring="roc_auc")
    return scores.mean(), scores.std()


def run_baselines(rows):
    """Run all surface baseline checks."""
    labels = np.array([r["label"] for r in rows])
    features_list = [surface_features(r["text"]) for r in rows]
    feature_names = list(features_list[0].keys())

    print(f"\nDataset: {len(rows)} examples, {labels.sum()} positive, {len(rows) - labels.sum()} negative")
    print(f"\n{'='*60}")
    print("BASELINE 1: Individual surface features (AUROC)")
    print(f"{'='*60}")

    # Individual feature AUROCs
    from sklearn.metrics import roc_auc_score
    for feat_name in feature_names:
        vals = np.array([f[feat_name] for f in features_list])
        if vals.std() == 0:
            print(f"  {feat_name:30s}  AUROC: N/A (constant)")
            continue
        auroc = roc_auc_score(labels, vals)
        # Flip if below 0.5 (feature inversely correlated)
        if auroc < 0.5:
            auroc = 1 - auroc
            direction = "(inverted)"
        else:
            direction = ""
        print(f"  {feat_name:30s}  AUROC: {auroc:.3f} {direction}")

    # Combined surface features logistic regression
    print(f"\n{'='*60}")
    print("BASELINE 2: Combined surface features (cross-validated AUROC)")
    print(f"{'='*60}")

    X_all = np.array([[f[name] for name in feature_names] for f in features_list])
    mean_auroc, std_auroc = logistic_regression_accuracy(X_all, labels)
    print(f"  All surface features:  AUROC: {mean_auroc:.3f} ± {std_auroc:.3f}")

    # Length-only
    X_length = np.array([[f["text_length"], f["prompt_length"], f["thinking_length"]] for f in features_list])
    mean_auroc, std_auroc = logistic_regression_accuracy(X_length, labels)
    print(f"  Length features only:  AUROC: {mean_auroc:.3f} ± {std_auroc:.3f}")

    # Special chars only
    X_special = np.array([[f["n_special_chars"], f["n_urls"], f["n_question_marks"]] for f in features_list])
    mean_auroc, std_auroc = logistic_regression_accuracy(X_special, labels)
    print(f"  Special chars only:   AUROC: {mean_auroc:.3f} ± {std_auroc:.3f}")

    # Lexical keywords only
    lexical_feats = [n for n in feature_names if "has_" in n]
    X_lexical = np.array([[f[name] for name in lexical_feats] for f in features_list])
    mean_auroc, std_auroc = logistic_regression_accuracy(X_lexical, labels)
    print(f"  Lexical keywords only: AUROC: {mean_auroc:.3f} ± {std_auroc:.3f}")

    return feature_names, features_list


def run_source_generalization(rows):
    """Leave-one-source-out evaluation."""
    print(f"\n{'='*60}")
    print("BASELINE 3: Leave-one-source-out generalization")
    print(f"{'='*60}")

    sources = Counter(r.get("source", "unknown") for r in rows)
    print(f"  Sources: {dict(sources)}")

    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import roc_auc_score

    labels = np.array([r["label"] for r in rows])
    features_list = [surface_features(r["text"]) for r in rows]
    feature_names = list(features_list[0].keys())
    X = np.array([[f[name] for name in feature_names] for f in features_list])

    for source in sorted(sources.keys()):
        if sources[source] < 10:
            continue
        test_mask = np.array([r.get("source") == source for r in rows])
        train_mask = ~test_mask

        if labels[test_mask].std() == 0:
            print(f"  Hold out '{source}': single class in test, skipping")
            continue

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X[train_mask])
        X_test = scaler.transform(X[test_mask])

        lr = LogisticRegression(max_iter=1000, random_state=42)
        lr.fit(X_train, labels[train_mask])

        pred = lr.predict_proba(X_test)[:, 1]
        auroc = roc_auc_score(labels[test_mask], pred)
        n_test = test_mask.sum()
        n_pos = labels[test_mask].sum()
        print(f"  Hold out '{source}' ({n_test} examples, {n_pos} pos): surface AUROC = {auroc:.3f}")


def run_probe_vs_baseline(args):
    """Compare actual probe accuracy to surface baselines."""
    import torch

    run_dir = Path(args.run_dir)
    layer = args.score_layer

    # Load probe metrics
    metrics_path = run_dir / "results" / f"layer_{layer}" / "linear" / "metrics.json"
    if metrics_path.exists():
        metrics = json.load(open(metrics_path))
        probe_auroc = metrics.get("val_auroc", metrics.get("auroc", "?"))
        probe_acc = metrics.get("accuracy", "?")
    else:
        probe_auroc = "?"
        probe_acc = "?"

    print(f"\n{'='*60}")
    print("COMPARISON: Probe vs Surface Baselines")
    print(f"{'='*60}")
    print(f"  Probe (layer {layer}):     AUROC: {probe_auroc}")
    print(f"  Probe (layer {layer}):     Accuracy: {probe_acc}")
    print()
    print("  If probe AUROC >> surface baselines, the probe is learning")
    print("  something beyond surface artifacts.")
    print("  If probe AUROC ≈ surface baselines, the probe may be cheating.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", default="logs/20260312_215816_olmo3-7b-think")
    parser.add_argument("--dataset", default="data/test_awareness.jsonl")
    parser.add_argument("--score-layer", type=int, default=26)
    args = parser.parse_args()

    rows = load_dataset(args.dataset)
    run_baselines(rows)
    run_source_generalization(rows)
    run_probe_vs_baseline(args)


if __name__ == "__main__":
    main()
