"""LoopClosureHead — calibrated same-place probability over embedding pairs.

Stage 3a of the topological-nav build (``docs/v3_topological_nav_plan.md`` §5.3;
decision registered in ``docs/lewm_topological_nav_stage2_belief_encoder_2026-06-09.md``
"v6 + decision"). The topological memory consumes loop-closure decisions at very
high precision (a false closure corrupts the graph; a miss merely duplicates a
node), so the spec gates on **precision >= 99% at the deployment threshold** and
**ECE <= 5% after calibration** — not on retrieval recall. This module is also
the *consumer-side* arbiter of the Stage 2 BeliefEncoder: the encoder is adopted
only if it lifts loop-closure recall at the 99%-precision operating point over
the frozen single-frame baseline.

Head: symmetric pair features ``[a*b, |a-b|]`` over two L2-normalized place
embeddings -> small MLP -> logit. Symmetry is by construction (both features are
order-invariant), matching the undirected same-place relation. Calibration is
post-hoc Platt scaling (§5.3 names isotonic/Platt; Platt is two parameters and
cannot overfit the calibration split). Labels follow the repo's three-bucket
scheme: same-cell positives, BFS-distance >= 2 negatives, adjacent cells masked.
"""
from __future__ import annotations

import torch
import torch.nn as nn


def pair_features(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Symmetric pair features for two (N, D) embedding batches -> (N, 2D)."""
    if a.shape != b.shape or a.dim() != 2:
        raise ValueError(f"expected matching (N, D) tensors, got {tuple(a.shape)} vs {tuple(b.shape)}")
    return torch.cat([a * b, (a - b).abs()], dim=-1)


class LoopClosureHead(nn.Module):
    def __init__(self, embedding_dim: int, hidden: int = 128, dropout: float = 0.1) -> None:
        super().__init__()
        self.embedding_dim = int(embedding_dim)
        self.net = nn.Sequential(
            nn.Linear(2 * embedding_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

    def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """Returns (N,) logits for P(same place)."""
        return self.net(pair_features(a, b)).squeeze(-1)


@torch.no_grad()
def expected_calibration_error(probs: torch.Tensor, labels: torch.Tensor, n_bins: int = 15) -> float:
    """Standard equal-width-bin ECE over P(same place)."""
    probs = probs.float().clamp(0.0, 1.0)
    labels = labels.float()
    edges = torch.linspace(0.0, 1.0, n_bins + 1, device=probs.device)
    ece = torch.zeros((), device=probs.device)
    for i in range(n_bins):
        in_bin = (probs > edges[i]) & (probs <= edges[i + 1]) if i > 0 else (probs >= edges[i]) & (probs <= edges[i + 1])
        if in_bin.any():
            ece = ece + in_bin.float().mean() * (probs[in_bin].mean() - labels[in_bin].mean()).abs()
    return float(ece)


def fit_platt(scores: torch.Tensor, labels: torch.Tensor, *, steps: int = 500, lr: float = 0.05) -> tuple[float, float]:
    """Fit Platt scaling p = sigmoid(a*score + b) by logistic regression.

    Returns (a, b). Deterministic (full-batch gradient descent from a=1, b=0).
    """
    scores = scores.detach().float()
    labels = labels.detach().float()
    a = torch.ones((), requires_grad=True)
    b = torch.zeros((), requires_grad=True)
    optimizer = torch.optim.Adam([a, b], lr=lr)
    loss_fn = nn.BCEWithLogitsLoss()
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        loss = loss_fn(a * scores + b, labels)
        loss.backward()
        optimizer.step()
    return float(a.detach()), float(b.detach())


@torch.no_grad()
def threshold_at_precision(probs: torch.Tensor, labels: torch.Tensor, precision: float) -> float | None:
    """Lowest threshold whose ``p >= t`` decision rule reaches the target precision.

    Scans the precision/recall curve from the most-confident pair down and keeps
    the largest prefix whose precision stays >= target (maximizing recall at that
    precision). Returns None if no threshold achieves it.
    """
    order = torch.argsort(probs, descending=True)
    sorted_labels = labels.float()[order]
    cumulative_tp = sorted_labels.cumsum(0)
    counts = torch.arange(1, len(sorted_labels) + 1, dtype=torch.float64, device=probs.device)
    precisions = cumulative_tp.double() / counts
    qualifying = torch.nonzero(precisions >= precision).flatten()
    if len(qualifying) == 0:
        return None
    return float(probs[order[int(qualifying[-1])]])


@torch.no_grad()
def precision_recall_at(probs: torch.Tensor, labels: torch.Tensor, threshold: float) -> tuple[float, float]:
    predicted = probs >= threshold
    labels = labels.bool()
    tp = (predicted & labels).sum().float()
    predicted_n = predicted.sum().float().clamp(min=1.0)
    positive_n = labels.sum().float().clamp(min=1.0)
    return float(tp / predicted_n), float(tp / positive_n)
