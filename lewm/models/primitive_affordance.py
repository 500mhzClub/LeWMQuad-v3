"""Source-image first-primitive affordance head for Phase 2M diagnostics."""
from __future__ import annotations

from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F

from .encoders import VisionEncoder

PrimitiveAffordanceRankingLoss = Literal["hard_ce", "soft_ce"]


class PrimitiveAffordanceModel(nn.Module):
    """Predict one source-local utility score for each candidate first primitive."""

    def __init__(
        self,
        *,
        primitive_count: int,
        latent_dim: int = 48,
        hidden_dim: int = 96,
        image_size: int = 224,
        patch_size: int = 14,
        encoder_depth: int = 2,
        encoder_heads: int = 3,
        encoder_mlp_ratio: int = 2,
        encoder_dropout: float = 0.0,
    ):
        super().__init__()
        if primitive_count < 2:
            raise ValueError("primitive_count must be at least 2")
        self.primitive_count = int(primitive_count)
        self.latent_dim = int(latent_dim)
        self.encoder = VisionEncoder(
            image_size=image_size,
            patch_size=patch_size,
            hidden_dim=latent_dim,
            depth=encoder_depth,
            n_heads=encoder_heads,
            mlp_ratio=encoder_mlp_ratio,
            dropout=encoder_dropout,
        )
        self.head = nn.Sequential(
            nn.LayerNorm(latent_dim),
            nn.Linear(latent_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, primitive_count),
        )

    def forward(self, start_vision: torch.Tensor) -> torch.Tensor:
        """Return primitive utility scores with shape ``(B, primitive_count)``."""

        if start_vision.ndim != 4:
            raise ValueError("start_vision must have shape (B, C, H, W)")
        return self.head(self.encoder(start_vision))


def primitive_affordance_losses(
    *,
    primitive_scores: torch.Tensor,
    primitive_utility_targets: torch.Tensor,
    primitive_utility_mask: torch.Tensor,
    primitive_class_weights: torch.Tensor | None = None,
    regression_weight: float = 1.0,
    ranking_loss: PrimitiveAffordanceRankingLoss = "soft_ce",
    softmax_temperature: float = 0.25,
) -> dict[str, torch.Tensor]:
    """Return masked source-local primitive affordance losses.

    The ranking term is computed independently per source state over valid
    primitive labels. Regression uses centered scores and centered targets so
    the loss emphasizes source-local ordering rather than global scene
    difficulty.
    """

    if primitive_scores.shape != primitive_utility_targets.shape:
        raise ValueError("primitive_scores and primitive_utility_targets must align")
    if primitive_utility_mask.shape != primitive_scores.shape:
        raise ValueError("primitive_utility_mask must align with primitive_scores")
    if primitive_class_weights is not None and primitive_class_weights.shape != (
        primitive_scores.shape[1],
    ):
        raise ValueError("primitive_class_weights must have shape (primitive_count,)")
    if regression_weight < 0.0:
        raise ValueError("regression_weight must be non-negative")
    if ranking_loss not in ("hard_ce", "soft_ce"):
        raise ValueError(f"unsupported primitive ranking loss: {ranking_loss}")
    if softmax_temperature <= 0.0:
        raise ValueError("softmax_temperature must be positive")

    zero = primitive_scores.sum() * 0.0
    ce_terms = []
    regression_terms = []
    valid_count = primitive_scores.new_zeros(())
    source_count = primitive_scores.new_zeros(())
    for index in range(primitive_scores.shape[0]):
        valid = primitive_utility_mask[index].bool()
        if int(valid.sum().detach().cpu()) < 2:
            continue
        scores = primitive_scores[index][valid]
        targets = primitive_utility_targets[index][valid]
        target_centered = targets - targets.mean()
        score_centered = scores - scores.mean()
        if ranking_loss == "hard_ce":
            label = torch.argmax(targets)
            ce = F.cross_entropy(scores.reshape(1, -1), label.reshape(1))
            if primitive_class_weights is not None:
                global_label = torch.nonzero(valid, as_tuple=False).flatten()[label]
                ce = ce * primitive_class_weights.to(scores.device)[global_label]
            ce_terms.append(ce)
        else:
            target_distribution = torch.softmax(
                target_centered / softmax_temperature,
                dim=0,
            )
            ce_terms.append(
                -(target_distribution * F.log_softmax(scores, dim=0)).sum()
            )
        regression_terms.append(F.mse_loss(score_centered, target_centered))
        valid_count = valid_count + valid.sum()
        source_count = source_count + 1.0

    if ce_terms:
        ce_loss = torch.stack(ce_terms).mean()
        regression_loss = torch.stack(regression_terms).mean()
    else:
        ce_loss = zero
        regression_loss = zero
    total = ce_loss + float(regression_weight) * regression_loss
    return {
        "primitive_affordance_loss": total,
        "primitive_affordance_ce_loss": ce_loss,
        "primitive_affordance_regression_loss": regression_loss,
        "primitive_affordance_valid_count": valid_count,
        "primitive_affordance_source_count": source_count,
    }
