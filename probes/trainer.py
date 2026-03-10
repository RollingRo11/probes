"""Probe training and evaluation.

Handles both closed-form probes (MeanDiffProbe) and gradient-based probes
(LinearProbe, EMAProbe, MLPProbe, AttentionProbe, MultiMaxProbe, MaxRollingMeanProbe).

Training protocol (gradient-based, arXiv:2601.11516):
  - AdamW, lr=1e-4, weight_decay=3e-3, β=(0.9, 0.999)
  - Full-batch gradient descent for 1000 epochs
  - BCE loss: L = BCE(σ(f(X) + b), y)
  - n_seeds random initialisations; best selected on val weighted error

EMA probe note:
  - Trains with mean-pooled forward pass (same as LinearProbe)
  - Evaluates using forward_ema() (EMA-max aggregation over sequence)
  - Requires "all" token_strategy activations for proper EMA evaluation
"""

from __future__ import annotations

import json
import random
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .architectures import (
    EMAProbe,
    MeanDiffProbe,
    build_probe,
)
from .metrics import auroc, evaluate_probe, weighted_error, youden_threshold


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _to_numpy(t: torch.Tensor) -> np.ndarray:
    return t.detach().cpu().float().numpy()


def _split_data(
    X: torch.Tensor,
    y: torch.Tensor,
    train_frac: float,
    val_frac: float,
    seed: int,
) -> tuple[torch.Tensor, ...]:
    """Returns X_train, X_val, X_test, y_train, y_val, y_test."""
    n = len(y)
    idx = torch.randperm(n, generator=torch.Generator().manual_seed(seed))

    n_train = int(n * train_frac)
    n_val = int(n * val_frac)

    i_train = idx[:n_train]
    i_val = idx[n_train : n_train + n_val]
    i_test = idx[n_train + n_val :]

    return (
        X[i_train], X[i_val], X[i_test],
        y[i_train], y[i_val], y[i_test],
    )


def _aggregate_for_inference(
    probe: nn.Module, X: torch.Tensor, device: torch.device
) -> np.ndarray:
    """Run probe forward, using EMA-max aggregation for EMAProbe."""
    probe.eval()
    X = X.to(device)
    with torch.no_grad():
        if isinstance(probe, EMAProbe) and X.dim() == 3:
            logits = probe.forward_ema(X)
        else:
            logits = probe(X)
    return torch.sigmoid(logits).cpu().numpy()


# ---------------------------------------------------------------------------
# Closed-form training
# ---------------------------------------------------------------------------


def train_mean_diff(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    eval_cfg: dict,
) -> dict:
    """Train and evaluate a MeanDiffProbe."""
    probe = MeanDiffProbe()
    probe.fit(X_train, y_train)
    probe.fit_threshold(X_val, y_val)

    val_scores = probe.decision_function(X_val)
    test_scores = probe.decision_function(X_test)

    results = evaluate_probe(
        y_test,
        test_scores,
        threshold=probe.threshold,
        fnr_weight=eval_cfg.get("fnr_weight", 5.0),
        hard_neg_fpr_weight=eval_cfg.get("hard_neg_fpr_weight", 2.0),
        overtrigger_fpr_weight=eval_cfg.get("overtrigger_fpr_weight", 50.0),
    )
    results["val_auroc"] = auroc(y_val, val_scores)
    return results, probe


# ---------------------------------------------------------------------------
# Gradient-based training
# ---------------------------------------------------------------------------


def _train_one_seed(
    probe: nn.Module,
    X_train: torch.Tensor,
    y_train: torch.Tensor,
    X_val: torch.Tensor,
    y_val: torch.Tensor,
    cfg: dict,
    device: torch.device,
) -> tuple[float, float, list[float]]:
    """Train probe for one seed. Returns (val_auroc, val_weighted_error, epoch_losses)."""
    probe.to(device)

    lr = float(cfg.get("lr", 1e-4))
    wd = float(cfg.get("weight_decay", 3e-3))
    betas = tuple(cfg.get("betas", [0.9, 0.999]))
    epochs = int(cfg.get("epochs", 1000))

    opt = torch.optim.AdamW(probe.parameters(), lr=lr, weight_decay=wd, betas=betas)

    X_tr = X_train.to(device)
    y_tr = y_train.float().to(device)

    # Full-batch gradient descent
    epoch_losses: list[float] = []
    probe.train()
    for _ in range(epochs):
        opt.zero_grad()

        if isinstance(probe, EMAProbe):
            # Train with mean-pooled activations for EMA probe.
            x_input = X_tr.mean(dim=1) if X_tr.dim() == 3 else X_tr
            logits = probe.linear(x_input).squeeze(-1)
        else:
            logits = probe(X_tr)

        loss = F.binary_cross_entropy_with_logits(logits, y_tr)
        epoch_losses.append(loss.item())
        loss.backward()
        opt.step()

    val_scores = _aggregate_for_inference(probe, X_val, device)
    y_val_np = _to_numpy(y_val)

    val_auc = auroc(y_val_np, val_scores)
    thresh = youden_threshold(y_val_np, val_scores)
    val_results = evaluate_probe(
        y_val_np,
        val_scores,
        threshold=thresh,
        fnr_weight=cfg.get("fnr_weight", 5.0),
        hard_neg_fpr_weight=cfg.get("hard_neg_fpr_weight", 2.0),
        overtrigger_fpr_weight=cfg.get("overtrigger_fpr_weight", 50.0),
    )
    val_we = val_results["weighted_error"]
    return val_auc, val_we, epoch_losses


def train_gradient_probe(
    probe_type: str,
    hidden_size: int,
    X_train: torch.Tensor,
    y_train: torch.Tensor,
    X_val: torch.Tensor,
    y_val: torch.Tensor,
    X_test: torch.Tensor,
    y_test: torch.Tensor,
    training_cfg: dict,
    eval_cfg: dict,
    probe_kwargs: dict | None = None,
    n_seeds: int = 1,
    base_seed: int = 42,
    wandb_run=None,
) -> tuple[dict, nn.Module]:
    """Train a gradient-based probe with n_seeds initialisations.

    Selects the best probe on validation weighted error, then evaluates on test.
    If wandb_run is provided, logs per-seed validation metrics and the best
    seed's per-epoch training loss curve.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    probe_kwargs = probe_kwargs or {}

    best_probe = None
    best_val_we = float("inf")
    best_val_auc = 0.0
    best_epoch_losses: list[float] = []
    seed_logs = []

    merged_cfg = {**training_cfg, **eval_cfg}

    for s in range(n_seeds):
        seed = base_seed + s
        _set_seed(seed)
        probe = build_probe(probe_type, hidden_size, **probe_kwargs)

        val_auc, val_we, epoch_losses = _train_one_seed(
            probe, X_train, y_train, X_val, y_val, merged_cfg, device
        )
        seed_logs.append({"seed": seed, "val_auroc": val_auc, "val_weighted_error": val_we})

        if wandb_run is not None:
            wandb_run.log({
                "seed": seed,
                "seed/val_auroc": val_auc,
                "seed/val_weighted_error": val_we,
            })

        if val_we < best_val_we or best_probe is None:
            best_val_we = val_we
            best_val_auc = val_auc
            best_probe = probe
            best_epoch_losses = epoch_losses

    # Log per-epoch training loss curve for the best seed.
    if wandb_run is not None and best_epoch_losses:
        for epoch, loss_val in enumerate(best_epoch_losses):
            wandb_run.log({"epoch": epoch, "train/bce_loss": loss_val})

    # Evaluate best probe on test set.
    test_scores = _aggregate_for_inference(best_probe, X_test.to(device), device)
    y_val_np = _to_numpy(y_val)
    val_scores_best = _aggregate_for_inference(best_probe, X_val.to(device), device)

    results = evaluate_probe(
        _to_numpy(y_test),
        test_scores,
        y_val=y_val_np,
        val_scores=val_scores_best,
        fnr_weight=eval_cfg.get("fnr_weight", 5.0),
        hard_neg_fpr_weight=eval_cfg.get("hard_neg_fpr_weight", 2.0),
        overtrigger_fpr_weight=eval_cfg.get("overtrigger_fpr_weight", 50.0),
    )
    results["val_auroc"] = best_val_auc
    results["val_weighted_error"] = best_val_we
    results["n_seeds"] = n_seeds
    results["seed_logs"] = seed_logs

    return results, best_probe


# ---------------------------------------------------------------------------
# Unified entry point
# ---------------------------------------------------------------------------


def train_and_evaluate(
    probe_type: str,
    activations: torch.Tensor,
    labels: torch.Tensor,
    training_cfg: dict,
    eval_cfg: dict,
    dataset_cfg: dict,
    probe_kwargs: dict | None = None,
    wandb_run=None,
) -> tuple[dict, object]:
    """Train and evaluate one (probe_type, layer) combination.

    Parameters
    ----------
    activations : (n, hidden) or (n, seq_len, hidden) tensor.
    labels      : (n,) int64 tensor.
    training_cfg: fields: lr, weight_decay, betas, epochs, n_seeds, full_batch, …
    eval_cfg    : fields: fnr_weight, hard_neg_fpr_weight, overtrigger_fpr_weight.
    dataset_cfg : fields: train_frac, val_frac, test_frac, seed.
    wandb_run   : optional wandb run object for logging.

    Returns (metrics_dict, trained_probe).
    """
    train_frac = float(dataset_cfg.get("train_frac", 0.8))
    val_frac = float(dataset_cfg.get("val_frac", 0.1))
    seed = int(dataset_cfg.get("seed", 42))
    n_seeds = int(training_cfg.get("n_seeds", 1))

    X_tr, X_val, X_te, y_tr, y_val, y_te = _split_data(
        activations, labels, train_frac, val_frac, seed
    )

    if probe_type == "mean_diff":
        # Mean-pool if full-sequence activations provided.
        X_tr_np = _to_numpy(X_tr.mean(1) if X_tr.dim() == 3 else X_tr)
        X_val_np = _to_numpy(X_val.mean(1) if X_val.dim() == 3 else X_val)
        X_te_np = _to_numpy(X_te.mean(1) if X_te.dim() == 3 else X_te)
        y_tr_np = _to_numpy(y_tr)
        y_val_np = _to_numpy(y_val)
        y_te_np = _to_numpy(y_te)
        return train_mean_diff(X_tr_np, y_tr_np, X_val_np, y_val_np, X_te_np, y_te_np, eval_cfg)

    hidden_size = activations.shape[-1]
    return train_gradient_probe(
        probe_type=probe_type,
        hidden_size=hidden_size,
        X_train=X_tr,
        y_train=y_tr,
        X_val=X_val,
        y_val=y_val,
        X_test=X_te,
        y_test=y_te,
        training_cfg=training_cfg,
        eval_cfg=eval_cfg,
        probe_kwargs=probe_kwargs,
        n_seeds=n_seeds,
        base_seed=seed,
        wandb_run=wandb_run,
    )


# ---------------------------------------------------------------------------
# Checkpoint I/O
# ---------------------------------------------------------------------------


def save_probe(probe: object, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(probe, MeanDiffProbe):
        torch.save({"type": "mean_diff", "direction": probe.direction, "threshold": probe.threshold}, path)
    else:
        torch.save({"type": "torch", "state_dict": probe.state_dict()}, path)


def save_results(results: dict, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    # Convert non-serialisable types.
    def _convert(obj):
        if isinstance(obj, (np.floating, np.integer)):
            return obj.item()
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    clean = {k: _convert(v) for k, v in results.items()}
    with open(path, "w") as f:
        json.dump(clean, f, indent=2)
