"""Target-grounded masked pairwise margin over counterfactual branches.

Replaces the nine-way soft-label matching cross-entropy.  The metric-alignment
audit found that the ranking component of the old objective was strongly aligned
with the thesis endpoint (row Spearman against frozen Q correlated +0.95 with
correct-minus-shuffled) while the probability component was not, and that ~53%
of the nine-way comparisons are near-degenerate, so cross-entropy over all cells
spent most of its mass on distinctions the successor targets do not contain.

For prediction ``p_i`` and frozen-reference successors ``t_bar_i``, ``t_bar_j``::

    s_ii = cos(p_i, t_bar_i)
    s_ij = cos(p_i, t_bar_j)

An ordered pair ``(i, j)`` is included only when both frozen separation criteria
hold::

    endpoint_displacement_ij > 0.05 m   AND   cos(t_bar_i, t_bar_j) < 0.90

The required margin is the target gap itself, so the objective can never demand
more separation than the successors actually exhibit::

    m_ij = 1 - cos(t_bar_i, t_bar_j)
    L    = mean_g mean_i mean_{j : M_ij = 1} max(0, m_ij - (s_ii - s_ij))

Valid negatives are averaged within each anchor and anchors within each group,
so a group with more separated pairs does not receive greater implicit weight.
"""

from __future__ import annotations

from typing import NamedTuple

import torch
import torch.nn.functional as F

DISPLACEMENT_THRESHOLD_M = 0.05
FROZEN_COSINE_THRESHOLD = 0.90


class MaskedRankingStatsV1(NamedTuple):
    """Ranking diagnostics over the masked pairs only."""

    loss: torch.Tensor
    pairwise_accuracy: float
    masked_mrr: float
    mean_achieved_margin: float
    valid_pairs: int
    valid_anchors: int


def separation_mask_v1(
    frozen_cosine: torch.Tensor, endpoints: torch.Tensor
) -> torch.Tensor:
    """Ordered-pair mask ``M`` from the two frozen criteria.

    ``frozen_cosine`` is ``(G, 9, 9)`` cosine between frozen-reference
    successors; ``endpoints`` is ``(G, 9, 2)`` world-frame branch endpoints.
    """
    if frozen_cosine.ndim != 3 or frozen_cosine.shape[-1] != frozen_cosine.shape[-2]:
        raise ValueError("frozen_cosine must be (G, n, n)")
    n = frozen_cosine.shape[-1]
    displacement = (endpoints[:, :, None, :] - endpoints[:, None, :, :]).norm(dim=-1)
    off_diagonal = ~torch.eye(n, dtype=torch.bool, device=frozen_cosine.device)
    return (
        (displacement > DISPLACEMENT_THRESHOLD_M)
        & (frozen_cosine < FROZEN_COSINE_THRESHOLD)
        & off_diagonal[None]
    )


def masked_pairwise_margin_v1(
    predicted_flat: torch.Tensor,
    frozen_flat: torch.Tensor,
    mask: torch.Tensor,
    frozen_cosine: torch.Tensor,
) -> MaskedRankingStatsV1:
    """Hinge on the target gap, averaged per anchor then per group.

    ``predicted_flat`` and ``frozen_flat`` are ``(G, 9, D)`` group-level token
    vectors; they are L2-normalised here so the scores are cosines.
    """
    if predicted_flat.shape != frozen_flat.shape:
        raise ValueError("predicted and frozen successor shapes differ")
    predicted = F.normalize(predicted_flat, dim=-1)
    frozen = F.normalize(frozen_flat, dim=-1)

    scores = torch.einsum("gid,gjd->gij", predicted, frozen)      # s_ij
    own = torch.diagonal(scores, dim1=1, dim2=2)                  # s_ii
    margin_required = 1.0 - frozen_cosine                         # m_ij
    difference = own[:, :, None] - scores                         # s_ii - s_ij
    hinge = (margin_required - difference).clamp_min(0.0)

    weights = mask.to(hinge.dtype)
    per_anchor_count = weights.sum(-1)                            # (G, 9)
    anchor_valid = per_anchor_count > 0
    per_anchor = (hinge * weights).sum(-1) / per_anchor_count.clamp_min(1.0)

    group_count = anchor_valid.sum(-1)
    group_valid = group_count > 0
    per_group = (per_anchor * anchor_valid).sum(-1) / group_count.clamp_min(1).to(
        per_anchor.dtype
    )
    loss = per_group[group_valid].mean() if bool(group_valid.any()) else scores.sum() * 0.0

    with torch.no_grad():
        correct = ((difference > 0) & mask).sum()
        total = mask.sum()
        accuracy = float(correct) / float(total) if int(total) else float("nan")
        # Masked MRR: rank of the own successor among itself plus its valid negatives.
        neg_better = ((scores >= own[:, :, None]) & mask).sum(-1)
        rank = (neg_better + 1).to(torch.float64)
        mrr = float((1.0 / rank)[anchor_valid].mean()) if bool(anchor_valid.any()) else float("nan")
        achieved = (difference * weights).sum() / weights.sum().clamp_min(1.0)

    return MaskedRankingStatsV1(
        loss=loss,
        pairwise_accuracy=accuracy,
        masked_mrr=mrr,
        mean_achieved_margin=float(achieved),
        valid_pairs=int(mask.sum()),
        valid_anchors=int(anchor_valid.sum()),
    )


__all__ = [
    "DISPLACEMENT_THRESHOLD_M",
    "FROZEN_COSINE_THRESHOLD",
    "MaskedRankingStatsV1",
    "masked_pairwise_margin_v1",
    "separation_mask_v1",
]
