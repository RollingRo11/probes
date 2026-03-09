"""Probe architectures for LLM interpretability.

Implements the 6-state composition framework from arXiv:2601.11516
"Building Production-Ready Probes for Gemini":

  All probing classifiers follow:
    (1) Residual stream activations S_i = {x_{i,j}} for j token positions
    (2) Per-position transformation T(x_{i,j}) → y_{i,j}
    (3) Aggregation {y_{i,j}} → scalar score

Probe types
-----------
MeanDiffProbe   : Closed-form difference-of-means (arXiv:2507.01786).
                  T=identity, aggregation=mean; direction = normalized(μ₁ − μ₀).
LinearProbe     : Gradient-based BCE. T=identity, aggregation=mean pool + linear.
                  f(S_i) = (1/n_i) Σ_j w·x_{i,j} + b
EMAProbe        : Trained as LinearProbe; at inference uses EMA-max aggregation.
                  ema_j = α·score_j + (1−α)·ema_{j−1},  f = max_j ema_j,  α=0.5
MLPProbe        : T=2-layer ReLU MLP (width 100), aggregation=mean pool + linear head.
AttentionProbe  : T=MLP, aggregation=multi-head soft attention.
                  f(S_i) = Σ_h Σ_j softmax(q_h·y_j)_j · (v_h·y_j)
MultiMaxProbe   : T=MLP, aggregation=per-head hard max (prevents dilution in long ctx).
                  f(S_i) = Σ_h max_j (v_h·y_j)
MaxRollingMeanProbe: T=MLP, aggregation=sliding-window attention-weighted mean + max.
                  Window width w=10, 10 heads.
"""

from __future__ import annotations

import inspect
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_curve

# Probe types that require per-position (full-sequence) activations.
SEQUENCE_PROBES: frozenset[str] = frozenset(
    {"ema", "attention", "multimax", "max_rolling_mean"}
)


def needs_sequence_activations(probe_types: list[str]) -> bool:
    return any(p in SEQUENCE_PROBES for p in probe_types)


# ---------------------------------------------------------------------------
# Closed-form probe
# ---------------------------------------------------------------------------


class MeanDiffProbe:
    """Difference-of-means probe (arXiv:2507.01786).

    Closed-form: direction = normalize(mean(X[y==1]) - mean(X[y==0])).
    Threshold chosen to maximise Youden's J = TPR - FPR on the validation set.
    """

    direction: Optional[np.ndarray]
    threshold: float

    def __init__(self) -> None:
        self.direction = None
        self.threshold = 0.0

    def fit(self, X: np.ndarray, y: np.ndarray) -> "MeanDiffProbe":
        """X: (n, d)  y: (n,) with values in {0, 1}."""
        X0, X1 = X[y == 0], X[y == 1]
        if len(X0) == 0 or len(X1) == 0:
            raise ValueError("Both classes must have at least one example.")
        diff = X1.mean(axis=0) - X0.mean(axis=0)
        norm = np.linalg.norm(diff)
        if norm < 1e-8:
            raise ValueError("Classes have identical mean activations.")
        self.direction = diff / norm
        return self

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        """Returns probe scores (higher = more likely class 1)."""
        if self.direction is None:
            raise RuntimeError("Call fit() before predict().")
        return X @ self.direction

    def fit_threshold(self, X_val: np.ndarray, y_val: np.ndarray) -> float:
        """Set threshold maximising Youden's J on validation set."""
        scores = self.decision_function(X_val)
        fpr, tpr, thresholds = roc_curve(y_val, scores)
        j = tpr - fpr
        best_idx = int(np.argmax(j))
        self.threshold = float(thresholds[best_idx])
        return self.threshold

    def predict(self, X: np.ndarray) -> np.ndarray:
        return (self.decision_function(X) >= self.threshold).astype(int)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Returns pseudo-probabilities via min-max normalisation of scores."""
        scores = self.decision_function(X)
        lo, hi = scores.min(), scores.max()
        if hi - lo < 1e-8:
            return np.full_like(scores, 0.5)
        return (scores - lo) / (hi - lo)


# ---------------------------------------------------------------------------
# Gradient-based probes (all nn.Module subclasses)
# ---------------------------------------------------------------------------


class LinearProbe(nn.Module):
    """Gradient-based linear probe (arXiv:2601.11516).

    T = identity, aggregation = mean-pool over sequence + linear.
    f(S_i) = (1/n_i) Σ_j  w·x_{i,j} + b
    """

    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.linear = nn.Linear(hidden_size, 1)
        nn.init.normal_(self.linear.weight, std=0.01)
        nn.init.zeros_(self.linear.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (batch, seq, hidden) or (batch, hidden) → (batch,) logits."""
        if x.dim() == 3:
            x = x.mean(dim=1)
        return self.linear(x).squeeze(-1)


class EMAProbe(nn.Module):
    """EMA probe (arXiv:2601.11516 §2.2).

    Trains identically to LinearProbe (mean-pool forward used during training).
    At inference on full-sequence inputs, applies EMA over per-token scores
    and returns the maximum.

        score_j = w·x_j + b
        ema_j   = α · score_j + (1−α) · ema_{j−1}
        f(S_i)  = max_j ema_j          (α = 0.5 by default)
    """

    def __init__(self, hidden_size: int, alpha: float = 0.5) -> None:
        super().__init__()
        self.linear = nn.Linear(hidden_size, 1)
        self.alpha = alpha
        nn.init.normal_(self.linear.weight, std=0.01)
        nn.init.zeros_(self.linear.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Training forward: mean-pool then linear (identical to LinearProbe)."""
        if x.dim() == 3:
            x = x.mean(dim=1)
        return self.linear(x).squeeze(-1)

    def forward_ema(self, x: torch.Tensor) -> torch.Tensor:
        """Inference forward: EMA-max over per-token scores.

        x: (batch, seq, hidden) — requires full sequence.
        """
        if x.dim() == 2:
            # Fallback if sequence dim unavailable.
            return self.linear(x).squeeze(-1)

        scores = self.linear(x).squeeze(-1)  # (batch, seq)
        batch, seq_len = scores.shape

        ema = torch.zeros_like(scores)
        ema[:, 0] = scores[:, 0]
        for j in range(1, seq_len):
            ema[:, j] = self.alpha * scores[:, j] + (1 - self.alpha) * ema[:, j - 1]

        return ema.max(dim=1).values


# ---------------------------------------------------------------------------
# Shared MLP transform used by MLPProbe, AttentionProbe, MultiMaxProbe, etc.
# ---------------------------------------------------------------------------


class _MLPTransform(nn.Module):
    """M-layer ReLU MLP. Input→(hidden)^{M-1}→hidden (no final nonlinearity)."""

    def __init__(
        self, input_dim: int, hidden_dim: int = 100, n_layers: int = 2
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        in_d = input_dim
        for _ in range(n_layers - 1):
            layers += [nn.Linear(in_d, hidden_dim), nn.ReLU()]
            in_d = hidden_dim
        layers.append(nn.Linear(in_d, hidden_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class MLPProbe(nn.Module):
    """MLP probe (arXiv:2601.11516 §2.3).

    T = 2-layer ReLU MLP (width 100), aggregation = mean-pool + linear head.
    f(S_i) = (1/n_i) Σ_j  head(MLP(x_{i,j}))
    """

    def __init__(
        self,
        hidden_size: int,
        mlp_hidden_dim: int = 100,
        mlp_n_layers: int = 2,
    ) -> None:
        super().__init__()
        self.transform = _MLPTransform(hidden_size, mlp_hidden_dim, mlp_n_layers)
        self.head = nn.Linear(mlp_hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (batch, seq, hidden) or (batch, hidden) → (batch,) logits."""
        if x.dim() == 2:
            x = x.unsqueeze(1)
        y = self.transform(x)       # (batch, seq, mlp_hidden)
        y = y.mean(dim=1)           # (batch, mlp_hidden)
        return self.head(y).squeeze(-1)


class AttentionProbe(nn.Module):
    """Multi-head soft-attention probe (arXiv:2601.11516 §2.4).

    T = MLP(x_{i,j}) → y_{i,j}
    f(S_i) = Σ_h  Σ_j softmax_j(q_h·y_j) · (v_h·y_j)

    q_h, v_h ∈ ℝ^{mlp_hidden} are learned per-head query/value vectors.
    Default: 10 heads.
    """

    def __init__(
        self,
        hidden_size: int,
        n_heads: int = 10,
        mlp_hidden_dim: int = 100,
        mlp_n_layers: int = 2,
    ) -> None:
        super().__init__()
        self.n_heads = n_heads
        self.transform = _MLPTransform(hidden_size, mlp_hidden_dim, mlp_n_layers)
        self.queries = nn.Parameter(torch.randn(n_heads, mlp_hidden_dim) * 0.01)
        self.values = nn.Parameter(torch.randn(n_heads, mlp_hidden_dim) * 0.01)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (batch, seq, hidden) or (batch, hidden) → (batch,) logits."""
        if x.dim() == 2:
            x = x.unsqueeze(1)

        y = self.transform(x)  # (batch, seq, mlp_hidden)

        # q_scores[b, h, j] = q_h · y_{b,j}
        q_scores = torch.einsum("bsh,nh->bns", y, self.queries)   # (batch, heads, seq)
        attn = torch.softmax(q_scores, dim=-1)                     # (batch, heads, seq)

        # v_scores[b, h, j] = v_h · y_{b,j}
        v_scores = torch.einsum("bsh,nh->bns", y, self.values)     # (batch, heads, seq)

        # Σ_j attn_{h,j} · v_score_{h,j}, then Σ_h
        return (attn * v_scores).sum(dim=-1).sum(dim=-1)            # (batch,)


class MultiMaxProbe(nn.Module):
    """MultiMax probe (arXiv:2601.11516 §3.1) — main contribution of the paper.

    Replaces softmax with hard-max to prevent signal dilution in long contexts.

    T = MLP(x_{i,j}) → y_{i,j}
    f(S_i) = Σ_h  max_j (v_h · y_{i,j})

    Default: 10 heads.
    """

    def __init__(
        self,
        hidden_size: int,
        n_heads: int = 10,
        mlp_hidden_dim: int = 100,
        mlp_n_layers: int = 2,
    ) -> None:
        super().__init__()
        self.n_heads = n_heads
        self.transform = _MLPTransform(hidden_size, mlp_hidden_dim, mlp_n_layers)
        self.values = nn.Parameter(torch.randn(n_heads, mlp_hidden_dim) * 0.01)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (batch, seq, hidden) or (batch, hidden) → (batch,) logits."""
        if x.dim() == 2:
            x = x.unsqueeze(1)

        y = self.transform(x)                                       # (batch, seq, mlp_hidden)
        v_scores = torch.einsum("bsh,nh->bns", y, self.values)     # (batch, heads, seq)

        # Hard max over positions, then sum over heads.
        return v_scores.max(dim=-1).values.sum(dim=-1)              # (batch,)


class MaxRollingMeanProbe(nn.Module):
    """Max of Rolling Means Attention Probe (arXiv:2601.11516 §3.2).

    Computes attention-weighted averages within sliding windows of width w,
    then takes the max over window endpoints.

    For each window ending at position t (window = [t, t+w)):
      α_j = exp(q_h · y_j)  (unnormalised)
      v̄_t^h = Σ_{j in window} α_j · (v_h · y_j) / Σ_{j in window} α_j

    f(S_i) = Σ_h  max_t  v̄_t^h

    Default: 10 heads, window width 10.
    """

    def __init__(
        self,
        hidden_size: int,
        n_heads: int = 10,
        rolling_window: int = 10,
        mlp_hidden_dim: int = 100,
        mlp_n_layers: int = 2,
    ) -> None:
        super().__init__()
        self.n_heads = n_heads
        self.window = rolling_window
        self.transform = _MLPTransform(hidden_size, mlp_hidden_dim, mlp_n_layers)
        self.queries = nn.Parameter(torch.randn(n_heads, mlp_hidden_dim) * 0.01)
        self.values = nn.Parameter(torch.randn(n_heads, mlp_hidden_dim) * 0.01)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (batch, seq, hidden) or (batch, hidden) → (batch,) logits."""
        if x.dim() == 2:
            x = x.unsqueeze(1)

        batch, seq_len, _ = x.shape
        y = self.transform(x)                                       # (batch, seq, mlp_hidden)

        q_scores = torch.einsum("bsh,nh->bns", y, self.queries)    # (batch, heads, seq)
        v_scores = torch.einsum("bsh,nh->bns", y, self.values)     # (batch, heads, seq)

        w = min(self.window, seq_len)
        n_windows = seq_len - w + 1

        # (batch, heads, n_windows)
        window_vals = torch.zeros(
            batch, self.n_heads, n_windows, device=x.device, dtype=x.dtype
        )
        for t in range(n_windows):
            q_win = q_scores[:, :, t : t + w]   # (batch, heads, w)
            v_win = v_scores[:, :, t : t + w]   # (batch, heads, w)
            attn_win = torch.softmax(q_win, dim=-1)
            window_vals[:, :, t] = (attn_win * v_win).sum(dim=-1)

        # Max over windows, sum over heads → (batch,)
        return window_vals.max(dim=-1).values.sum(dim=-1)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

_PROBE_REGISTRY: dict[str, type] = {
    "linear": LinearProbe,
    "ema": EMAProbe,
    "mlp": MLPProbe,
    "attention": AttentionProbe,
    "multimax": MultiMaxProbe,
    "max_rolling_mean": MaxRollingMeanProbe,
}


def build_probe(probe_type: str, hidden_size: int, **kwargs) -> nn.Module | MeanDiffProbe:
    """Instantiate a probe by name. Passes only recognised kwargs to the constructor."""
    if probe_type == "mean_diff":
        return MeanDiffProbe()
    if probe_type not in _PROBE_REGISTRY:
        raise ValueError(
            f"Unknown probe type '{probe_type}'. "
            f"Available: {['mean_diff'] + list(_PROBE_REGISTRY)}"
        )
    cls = _PROBE_REGISTRY[probe_type]
    sig = inspect.signature(cls.__init__)
    filtered = {k: v for k, v in kwargs.items() if k in sig.parameters}
    return cls(hidden_size=hidden_size, **filtered)
