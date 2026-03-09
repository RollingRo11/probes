"""Evaluation metrics for probe classifiers.

Standard metrics:
  auroc           - Area under ROC curve.
  accuracy        - Accuracy at chosen threshold.
  fnr             - False Negative Rate at threshold.
  fpr             - False Positive Rate at threshold.

Production-probe weighted error (arXiv:2601.11516 §3.2):
  E_w = w_fnr · FNR + w_hard · hard_neg_FPR + w_over · overtrigger_FPR

  where the three FPR components correspond to different "negative" subsets:
    - hard negatives: examples that look similar to positives
    - overtriggering negatives: benign examples that must not be flagged

  For datasets without explicit subsets, we fall back to a single overall FPR.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve


def auroc(y_true: np.ndarray, scores: np.ndarray) -> float:
    """AUROC. Handles single-class edge case."""
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(roc_auc_score(y_true, scores))


def youden_threshold(y_true: np.ndarray, scores: np.ndarray) -> float:
    """Threshold maximising Youden's J = TPR - FPR."""
    fpr, tpr, thresholds = roc_curve(y_true, scores)
    j = tpr - fpr
    return float(thresholds[int(np.argmax(j))])


def threshold_metrics(
    y_true: np.ndarray,
    scores: np.ndarray,
    threshold: float,
    hard_neg_mask: Optional[np.ndarray] = None,
    overtrigger_mask: Optional[np.ndarray] = None,
) -> dict[str, float]:
    """Compute FPR, FNR, accuracy at a fixed threshold.

    Parameters
    ----------
    y_true          : binary labels (0/1).
    scores          : continuous probe scores.
    threshold       : decision boundary.
    hard_neg_mask   : boolean mask for hard-negative examples (y_true==0 subset).
    overtrigger_mask: boolean mask for overtriggering examples (y_true==0 subset).
    """
    preds = (scores >= threshold).astype(int)

    pos = y_true == 1
    neg = y_true == 0

    tp = int(((preds == 1) & pos).sum())
    fn = int(((preds == 0) & pos).sum())
    fp = int(((preds == 1) & neg).sum())
    tn = int(((preds == 0) & neg).sum())

    fnr = fn / (tp + fn) if (tp + fn) > 0 else float("nan")
    fpr = fp / (fp + tn) if (fp + tn) > 0 else float("nan")
    acc = (tp + tn) / len(y_true)

    out: dict[str, float] = {
        "accuracy": acc,
        "fnr": fnr,
        "fpr": fpr,
        "tp": float(tp),
        "fn": float(fn),
        "fp": float(fp),
        "tn": float(tn),
    }

    # Subset FPRs
    if hard_neg_mask is not None:
        h_neg = neg & hard_neg_mask
        h_fp = int(((preds == 1) & h_neg).sum())
        h_tn = int(((preds == 0) & h_neg).sum())
        out["hard_neg_fpr"] = h_fp / (h_fp + h_tn) if (h_fp + h_tn) > 0 else float("nan")
    else:
        out["hard_neg_fpr"] = fpr

    if overtrigger_mask is not None:
        o_neg = neg & overtrigger_mask
        o_fp = int(((preds == 1) & o_neg).sum())
        o_tn = int(((preds == 0) & o_neg).sum())
        out["overtrigger_fpr"] = o_fp / (o_fp + o_tn) if (o_fp + o_tn) > 0 else float("nan")
    else:
        out["overtrigger_fpr"] = fpr

    return out


def weighted_error(
    metrics: dict[str, float],
    fnr_weight: float = 5.0,
    hard_neg_fpr_weight: float = 2.0,
    overtrigger_fpr_weight: float = 50.0,
) -> float:
    """Production-probe weighted error from arXiv:2601.11516 §3.2.

    E_w = w_fnr · FNR + w_hard · hard_neg_FPR + w_over · overtrigger_FPR
    """
    fnr = metrics.get("fnr", float("nan"))
    hard_neg_fpr = metrics.get("hard_neg_fpr", float("nan"))
    overtrigger_fpr = metrics.get("overtrigger_fpr", float("nan"))

    components = [
        (fnr_weight, fnr),
        (hard_neg_fpr_weight, hard_neg_fpr),
        (overtrigger_fpr_weight, overtrigger_fpr),
    ]
    if any(np.isnan(v) for _, v in components):
        return float("nan")
    return sum(w * v for w, v in components)


def evaluate_probe(
    y_true: np.ndarray,
    scores: np.ndarray,
    threshold: Optional[float] = None,
    y_val: Optional[np.ndarray] = None,
    val_scores: Optional[np.ndarray] = None,
    hard_neg_mask: Optional[np.ndarray] = None,
    overtrigger_mask: Optional[np.ndarray] = None,
    fnr_weight: float = 5.0,
    hard_neg_fpr_weight: float = 2.0,
    overtrigger_fpr_weight: float = 50.0,
) -> dict[str, float]:
    """Full evaluation for a single probe on a test set.

    If threshold is None, selects via Youden's J on val_scores/y_val.
    """
    if threshold is None:
        if y_val is None or val_scores is None:
            raise ValueError("Provide val_scores/y_val or a fixed threshold.")
        threshold = youden_threshold(y_val, val_scores)

    auc = auroc(y_true, scores)
    tm = threshold_metrics(
        y_true, scores, threshold,
        hard_neg_mask=hard_neg_mask,
        overtrigger_mask=overtrigger_mask,
    )
    we = weighted_error(tm, fnr_weight, hard_neg_fpr_weight, overtrigger_fpr_weight)

    return {
        "auroc": auc,
        "threshold": threshold,
        "weighted_error": we,
        **tm,
    }
