"""GoalAdapter — map a single goal-frame latent into BeliefEncoder space (§5.2).

A goal image is one frame; the memory's keys are history-conditioned belief
embeddings. Passing a goal image through the BeliefEncoder (padded or repeated)
is a silent train/deploy distribution mismatch (§5.2) — so a small adapter maps
the frozen single-frame LeWM latent directly into the belief retrieval space,
trained cross-modally: adapter(goal-frame latent) should land next to the
belief embeddings of windows that *end at the same (cell, yaw_bin)*.

Yaw scope follows the Stage 3a node design: nodes are view keyframes, so the
goal must match the node *as seen from the goal's heading* — positives are
same-(cell, yaw_bin) pairs, same-cell-different-yaw is masked out (the §5.1
λ_yaw_weak→0 registered choice), negatives are BFS≥2 (any yaw). The self-pair
(a window matched to its own terminal frame) is excluded as trivial.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class GoalAdapter(nn.Module):
    def __init__(self, latent_dim: int, embedding_dim: int = 64, hidden: int = 128, dropout: float = 0.1) -> None:
        super().__init__()
        self.latent_dim = int(latent_dim)
        self.embedding_dim = int(embedding_dim)
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, embedding_dim),
        )

    def forward(self, z_goal: torch.Tensor) -> torch.Tensor:
        """(B, latent_dim) frozen single-frame latent -> (B, embedding_dim), L2-normalized."""
        return F.normalize(self.net(z_goal), dim=-1)


def masked_cross_modal_supcon(
    goal_embeddings: torch.Tensor,
    belief_embeddings: torch.Tensor,
    positive_mask: torch.Tensor,
    valid_pair_mask: torch.Tensor,
    *,
    temperature: float = 0.1,
) -> torch.Tensor:
    """Supervised contrastive over the goal->belief cross-similarity matrix.

    ``positive_mask[i, j]``: goal frame i and window j share (cell, yaw_bin);
    ``valid_pair_mask``: positives plus BFS>=2 negatives. The diagonal (a
    window's own terminal frame) is excluded — it is trivially solvable and
    never occurs at deployment.
    """
    if goal_embeddings.shape != belief_embeddings.shape:
        raise ValueError("goal and belief embedding batches must align")
    n = goal_embeddings.shape[0]
    diagonal = torch.eye(n, dtype=torch.bool, device=goal_embeddings.device)
    positive = positive_mask.bool() & ~diagonal
    valid = (valid_pair_mask.bool() | positive) & ~diagonal
    negative = valid & ~positive
    eligible = positive.any(dim=1) & negative.any(dim=1)
    if not bool(eligible.any()):
        raise ValueError("no goal anchors with both a positive and a valid negative")
    similarity = goal_embeddings @ belief_embeddings.T / temperature
    denominator = torch.logsumexp(similarity.masked_fill(~valid, -torch.inf), dim=1)
    numerator = torch.logsumexp(similarity.masked_fill(~positive, -torch.inf), dim=1)
    return -(numerator[eligible] - denominator[eligible]).mean()
