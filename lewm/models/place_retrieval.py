"""Metric-learning head for place retrieval from frozen LeWM features."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class PlaceRetrievalHead(nn.Module):
    """Map a pooled frozen feature to an L2-normalized retrieval embedding."""

    def __init__(
        self,
        latent_dim: int = 192,
        hidden: int = 256,
        embedding_dim: int = 128,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.embedding_dim = embedding_dim
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, embedding_dim),
        )

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.net(latent), dim=-1)


def masked_supervised_contrastive_loss(
    embedding: torch.Tensor,
    positive_mask: torch.Tensor,
    valid_pair_mask: torch.Tensor,
    *,
    temperature: float = 0.1,
) -> torch.Tensor:
    """Pull positives together while ignoring invalid or ambiguous pairs."""
    if embedding.ndim != 2:
        raise ValueError("embedding must have shape [observations, dimensions]")
    pair_shape = (embedding.shape[0], embedding.shape[0])
    if positive_mask.shape != pair_shape or valid_pair_mask.shape != pair_shape:
        raise ValueError("pair masks must have shape [observations, observations]")

    positive_mask = positive_mask.bool()
    valid_pair_mask = valid_pair_mask.bool() | positive_mask
    diagonal = torch.eye(embedding.shape[0], dtype=torch.bool, device=embedding.device)
    positive_mask = positive_mask & ~diagonal
    valid_pair_mask = valid_pair_mask & ~diagonal
    negative_mask = valid_pair_mask & ~positive_mask
    eligible = positive_mask.any(dim=1) & negative_mask.any(dim=1)
    if not bool(eligible.any()):
        raise ValueError("no anchors have both a positive and a valid negative")

    similarity = embedding @ embedding.T / temperature
    denominator = torch.logsumexp(similarity.masked_fill(~valid_pair_mask, -torch.inf), dim=1)
    numerator = torch.logsumexp(similarity.masked_fill(~positive_mask, -torch.inf), dim=1)
    return -(numerator[eligible] - denominator[eligible]).mean()

