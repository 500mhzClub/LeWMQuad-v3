"""RGB full-whitened predictive-state H4 JEPA V1.

This is the single structural follow-up to WDPS-D8.  It keeps the same fixed
N320 future-delta target, learned zero-preserving D8 compressor, dense
three-frame/action predictor, and joint backward.  The failed marginal
variance plus weak raw-covariance objective is replaced by a per-horizon full
within-branch covariance-to-identity constraints plus a cross-covariance-to-
identity alignment.  Copying one scalar into all eight state coordinates or
shrinking the learned target toward the zero-initialized predictor therefore
cannot satisfy the objective.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Mapping

import torch
import torch.nn.functional as F

from .go2_rgb_fixed_teacher_dense_spatiotemporal_cross_attention_h4_jepa_v1 import (
    JointRecurrentH4JEPAConfig as _DenseConfig,
)
from .go2_rgb_whitened_delta_predictive_state_h4_jepa_v1 import (
    WhitenedDeltaPredictiveStateConfig,
    WhitenedDeltaPredictiveStateH4JEPA,
    WhitenedDeltaPredictiveStateOutput,
    _mean_loss,
)


@dataclass(frozen=True)
class FullWhitenedPredictiveStateConfig(WhitenedDeltaPredictiveStateConfig):
    """Exact full-covariance objective contract."""

    variance_regularization_weight: float = 0.0
    covariance_regularization_weight: float = 25.0

    def __post_init__(self) -> None:
        # Deliberately bypass the predecessor's exact old loss-weight check
        # while retaining the inherited dense architecture validation.
        _DenseConfig.__post_init__(self)
        if self.state_dim != 8:
            raise ValueError("state_dim must remain exactly eight")
        expected = {
            "similarity_weight": 25.0,
            "variance_regularization_weight": 0.0,
            "mean_regularization_weight": 25.0,
            "covariance_regularization_weight": 25.0,
            "history_teacher_alignment_weight": 1.0,
            "variance_target_std": 1.0,
            "variance_epsilon": 1e-4,
        }
        for name, required in expected.items():
            value = getattr(self, name)
            if not math.isfinite(value) or value != required:
                raise ValueError(f"{name} must remain exactly {required}")


def _covariance_identity_loss(state: torch.Tensor) -> torch.Tensor:
    """Mean per-horizon squared Frobenius distance from covariance identity."""

    if state.ndim != 3 or state.shape[0] < 2:
        raise ValueError("state must have shape (B,H,D) with B at least two")
    batch, _horizons, dim = state.shape
    centered = state - state.mean(dim=0, keepdim=True)
    covariance = torch.einsum("bhd,bhe->hde", centered, centered) / float(batch - 1)
    identity = torch.eye(dim, dtype=state.dtype, device=state.device)[None]
    return (covariance - identity).square().sum(dim=(-2, -1)).div(float(dim)).mean()


def _cross_covariance_identity_loss(
    predicted: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    """Per-horizon CCA alignment without inverse covariance operations."""

    if predicted.shape != target.shape or predicted.ndim != 3:
        raise ValueError("predicted and target must share shape (B,H,D)")
    if predicted.shape[0] < 2:
        raise ValueError("cross covariance requires at least two batch rows")
    batch, _horizons, dim = predicted.shape
    predicted_centered = predicted - predicted.mean(dim=0, keepdim=True)
    target_centered = target - target.mean(dim=0, keepdim=True)
    cross_covariance = torch.einsum(
        "bhd,bhe->hde", predicted_centered, target_centered
    ) / float(batch - 1)
    identity = torch.eye(
        dim, dtype=predicted.dtype, device=predicted.device
    )[None]
    return (
        (cross_covariance - identity)
        .square()
        .sum(dim=(-2, -1))
        .div(float(dim))
        .mean()
    )


class FullWhitenedPredictiveStateH4JEPA(WhitenedDeltaPredictiveStateH4JEPA):
    """Joint predictor/target JEPA with a full covariance identity constraint."""

    def __init__(
        self,
        n320_encoder_state_dict: Mapping[str, torch.Tensor],
        *,
        config: FullWhitenedPredictiveStateConfig | None = None,
    ) -> None:
        exact_config = config or FullWhitenedPredictiveStateConfig()
        if not isinstance(exact_config, FullWhitenedPredictiveStateConfig):
            raise TypeError("config must be FullWhitenedPredictiveStateConfig")
        super().__init__(n320_encoder_state_dict, config=exact_config)

    def forward(
        self,
        history_rgb: torch.Tensor,
        past_actions: torch.Tensor,
        future_actions: torch.Tensor,
        future_rgb: torch.Tensor | None = None,
    ) -> WhitenedDeltaPredictiveStateOutput:
        history, belief = self.encode_history(history_rgb, past_actions)
        predicted = self.predict_from_belief(belief, future_actions)
        target: torch.Tensor | None = None
        similarity: torch.Tensor | None = None
        predicted_whitening: torch.Tensor | None = None
        target_whitening: torch.Tensor | None = None
        predicted_mean: torch.Tensor | None = None
        target_mean: torch.Tensor | None = None
        alignment: torch.Tensor | None = None
        total: torch.Tensor | None = None
        fixed_history: torch.Tensor | None = None
        zero: torch.Tensor | None = None
        if future_rgb is not None:
            fixed_history = self._encode_fixed_history(history_rgb)
            target = self.encode_target_state(fixed_history[:, 2], future_rgb)
            similarity = _cross_covariance_identity_loss(predicted, target)
            predicted_whitening = _covariance_identity_loss(predicted)
            target_whitening = _covariance_identity_loss(target)
            predicted_mean = _mean_loss(predicted)
            target_mean = _mean_loss(target)
            online_history = F.normalize(
                history,
                dim=-1,
                eps=self.config.normalization_epsilon,
            )
            alignment = (
                (online_history - fixed_history).square().sum(dim=-1).mean()
            )
            total = (
                self.config.similarity_weight * similarity
                + self.config.covariance_regularization_weight
                * 0.5
                * (predicted_whitening + target_whitening)
                + self.config.mean_regularization_weight
                * 0.5
                * (predicted_mean + target_mean)
                + self.config.history_teacher_alignment_weight * alignment
            )
            # Preserve the reviewed runner output schema while explicitly
            # recording that the predecessor's hinge-variance term is absent.
            zero = similarity.new_zeros(())
        return WhitenedDeltaPredictiveStateOutput(
            predicted_state=predicted,
            target_state=target,
            history_latents=history,
            fixed_history_latents=fixed_history,
            belief_latents=belief,
            state_prediction_loss=similarity,
            predicted_variance_loss=zero,
            target_variance_loss=zero,
            predicted_mean_loss=predicted_mean,
            target_mean_loss=target_mean,
            predicted_covariance_loss=predicted_whitening,
            target_covariance_loss=target_whitening,
            history_teacher_alignment_loss=alignment,
            total_loss=total,
        )


__all__ = [
    "FullWhitenedPredictiveStateConfig",
    "FullWhitenedPredictiveStateH4JEPA",
    "_covariance_identity_loss",
    "_cross_covariance_identity_loss",
]
