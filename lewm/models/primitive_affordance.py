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


class FactorizedPrimitiveAffordanceModel(nn.Module):
    """Predict factorized primitive affordance labels from one source image."""

    def __init__(
        self,
        *,
        primitive_count: int,
        factor_count: int,
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
        if factor_count < 2:
            raise ValueError("factor_count must be at least 2")
        self.primitive_count = int(primitive_count)
        self.factor_count = int(factor_count)
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
            nn.Linear(hidden_dim, primitive_count * factor_count),
        )

    def forward(self, start_vision: torch.Tensor) -> torch.Tensor:
        """Return raw factor logits with shape ``(B, primitive_count, factor_count)``."""

        if start_vision.ndim != 4:
            raise ValueError("start_vision must have shape (B, C, H, W)")
        logits = self.head(self.encoder(start_vision))
        return logits.reshape(
            start_vision.shape[0],
            self.primitive_count,
            self.factor_count,
        )


class GeometryPrimitiveAffordanceModel(nn.Module):
    """Predict factorized primitive affordance labels from geometry features."""

    def __init__(
        self,
        *,
        feature_dim: int,
        primitive_count: int,
        factor_count: int,
        hidden_dim: int = 128,
        depth: int = 3,
        dropout: float = 0.0,
    ):
        super().__init__()
        if feature_dim < 1:
            raise ValueError("feature_dim must be positive")
        if primitive_count < 2:
            raise ValueError("primitive_count must be at least 2")
        if factor_count < 2:
            raise ValueError("factor_count must be at least 2")
        if hidden_dim < 1:
            raise ValueError("hidden_dim must be positive")
        if depth < 1:
            raise ValueError("depth must be positive")
        self.feature_dim = int(feature_dim)
        self.primitive_count = int(primitive_count)
        self.factor_count = int(factor_count)
        layers: list[nn.Module] = [nn.LayerNorm(feature_dim)]
        in_dim = feature_dim
        for _index in range(int(depth)):
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.GELU())
            if dropout > 0.0:
                layers.append(nn.Dropout(dropout))
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, primitive_count * factor_count))
        self.net = nn.Sequential(*layers)

    def forward(self, geometry_features: torch.Tensor) -> torch.Tensor:
        """Return raw factor logits with shape ``(B, primitive_count, factor_count)``."""

        if geometry_features.ndim != 2:
            raise ValueError("geometry_features must have shape (B, feature_dim)")
        if geometry_features.shape[1] != self.feature_dim:
            raise ValueError("geometry_features feature dimension mismatch")
        logits = self.net(geometry_features)
        return logits.reshape(
            geometry_features.shape[0],
            self.primitive_count,
            self.factor_count,
        )


def factorized_affordance_values(factor_logits: torch.Tensor) -> torch.Tensor:
    """Map raw factor logits to the registered target value domains."""

    if factor_logits.ndim != 3:
        raise ValueError("factor_logits must have shape (B, primitive_count, factors)")
    if factor_logits.shape[-1] < 6:
        raise ValueError("factor_logits must contain the six Phase 2O factors")
    return torch.stack(
        [
            torch.sigmoid(factor_logits[..., 0]),
            torch.tanh(factor_logits[..., 1]),
            torch.sigmoid(factor_logits[..., 2]),
            torch.sigmoid(factor_logits[..., 3]),
            torch.sigmoid(factor_logits[..., 4]),
            torch.sigmoid(factor_logits[..., 5]),
        ],
        dim=-1,
    )


def factorized_affordance_losses(
    *,
    factor_logits: torch.Tensor,
    factor_targets: torch.Tensor,
    factor_mask: torch.Tensor,
    safety_weight: float = 1.0,
    value_weight: float = 1.0,
) -> dict[str, torch.Tensor]:
    """Return masked losses for factorized primitive affordance prediction."""

    if factor_logits.shape != factor_targets.shape:
        raise ValueError("factor_logits and factor_targets must align")
    if factor_mask.shape != factor_logits.shape:
        raise ValueError("factor_mask must align with factor_logits")
    if factor_logits.shape[-1] < 6:
        raise ValueError("factorized affordance loss requires six factors")
    if safety_weight < 0.0:
        raise ValueError("safety_weight must be non-negative")
    if value_weight < 0.0:
        raise ValueError("value_weight must be non-negative")

    zero = factor_logits.sum() * 0.0
    safety_mask = factor_mask[..., 0].bool()
    if bool(safety_mask.any()):
        safety_loss = F.binary_cross_entropy_with_logits(
            factor_logits[..., 0][safety_mask],
            factor_targets[..., 0][safety_mask],
        )
    else:
        safety_loss = zero

    values = factorized_affordance_values(factor_logits)
    value_mask = factor_mask[..., 1:].bool()
    if bool(value_mask.any()):
        value_loss = F.mse_loss(
            values[..., 1:][value_mask],
            factor_targets[..., 1:][value_mask],
        )
    else:
        value_loss = zero
    total = float(safety_weight) * safety_loss + float(value_weight) * value_loss
    return {
        "factorized_affordance_loss": total,
        "factorized_safety_bce_loss": safety_loss,
        "factorized_value_mse_loss": value_loss,
        "factorized_safety_valid_count": safety_mask.sum(),
        "factorized_value_valid_count": value_mask.sum(),
    }


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
