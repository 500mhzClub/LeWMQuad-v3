"""RGB whitened-delta predictive-state H4 JEPA V1.

The model learns an eight-dimensional state for each future horizon from the
intersection of (a) what a dense RGB/action history predictor can infer and
(b) what changes in a fixed N320 future representation.  A shared,
zero-preserving attention compressor maps fixed-teacher future-minus-e2 patch
deltas to learned target states.  A dense causal history/action predictor
emits the corresponding predicted states.  The compressor, online encoder,
and predictor train jointly with VICReg-style invariance, variance, zero-mean,
and covariance terms; the teacher is permanently fixed.

The online encoder is initialized from the accepted N320 encoder and aligned
to the fixed copy on the three observed frames.  The compact target pool is
learned, but its RGB inputs cannot co-adapt with it.  The final state head is
exactly zero initialized, making zero change the exact update-0 persistence
prediction.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass
import math
from typing import Mapping

import torch
import torch.nn as nn
import torch.nn.functional as F

from .go2_recurrent_h4_joint_jepa import (
    GO2_H4_PRIMITIVE_VOCABULARY,
    _validate_encoder_state,
)
from .go2_rgb_fixed_teacher_dense_spatiotemporal_cross_attention_h4_jepa_v1 import (
    JointRecurrentH4JEPAConfig as _DenseConfig,
    _DenseHistoricalContext,
    _DenseHorizonCrossAttention,
)
from .encoders import VisionEncoder


@dataclass(frozen=True)
class WhitenedDeltaPredictiveStateConfig(_DenseConfig):
    """Exact compact-state and whitening objective contract."""

    state_dim: int = 8
    similarity_weight: float = 25.0
    variance_regularization_weight: float = 25.0
    mean_regularization_weight: float = 25.0
    covariance_regularization_weight: float = 1.0
    history_teacher_alignment_weight: float = 1.0
    variance_target_std: float = 1.0
    variance_epsilon: float = 1e-4

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.state_dim != 8:
            raise ValueError("state_dim must remain exactly eight")
        expected = {
            "similarity_weight": 25.0,
            "variance_regularization_weight": 25.0,
            "mean_regularization_weight": 25.0,
            "covariance_regularization_weight": 1.0,
            "history_teacher_alignment_weight": 1.0,
            "variance_target_std": 1.0,
            "variance_epsilon": 1e-4,
        }
        for name, required in expected.items():
            value = getattr(self, name)
            if not math.isfinite(value) or value != required:
                raise ValueError(f"{name} must remain exactly {required}")


@dataclass(frozen=True)
class WhitenedDeltaPredictiveStateOutput:
    predicted_state: torch.Tensor
    target_state: torch.Tensor | None
    history_latents: torch.Tensor
    fixed_history_latents: torch.Tensor | None
    belief_latents: torch.Tensor
    state_prediction_loss: torch.Tensor | None
    predicted_variance_loss: torch.Tensor | None
    target_variance_loss: torch.Tensor | None
    predicted_mean_loss: torch.Tensor | None
    target_mean_loss: torch.Tensor | None
    predicted_covariance_loss: torch.Tensor | None
    target_covariance_loss: torch.Tensor | None
    history_teacher_alignment_loss: torch.Tensor | None
    total_loss: torch.Tensor | None


class _BiasFreeDeltaAttentionPool(nn.Module):
    """Eight spatial attention tests whose values are only teacher deltas."""

    def __init__(self, feature_dim: int, state_dim: int, spatial_tokens: int) -> None:
        super().__init__()
        self.query = nn.Parameter(torch.empty(state_dim, feature_dim))
        self.position_logits = nn.Parameter(torch.empty(state_dim, spatial_tokens))
        self.value_weight = nn.Parameter(torch.empty(state_dim, feature_dim))
        for parameter in (self.query, self.position_logits):
            nn.init.normal_(parameter, mean=0.0, std=0.02)
        nn.init.orthogonal_(self.value_weight)
        self.output_scale = math.sqrt(float(spatial_tokens))

    def forward(self, delta_tokens: torch.Tensor) -> torch.Tensor:
        if delta_tokens.ndim != 4:
            raise ValueError("delta_tokens must have shape (B,H,P,D)")
        feature_dim = int(delta_tokens.shape[-1])
        if feature_dim != self.query.shape[1]:
            raise ValueError("delta token feature width changed")
        if delta_tokens.shape[-2] != self.position_logits.shape[1]:
            raise ValueError("delta token spatial count changed")
        scores = (
            torch.einsum("bhpd,sd->bhsp", delta_tokens, self.query)
            / math.sqrt(float(feature_dim))
            + self.position_logits[None, None]
        )
        weights = torch.softmax(scores, dim=-1)
        pooled = torch.einsum("bhsp,bhpd->bhsd", weights, delta_tokens)
        return self.output_scale * torch.einsum(
            "bhsd,sd->bhs", pooled, self.value_weight
        )


def _variance_loss(
    state: torch.Tensor,
    *,
    target_std: float,
    epsilon: float,
) -> torch.Tensor:
    if state.ndim != 3 or state.shape[0] < 2:
        raise ValueError("state must have shape (B,H,D) with B at least two")
    std = state.var(dim=0, unbiased=True).add(epsilon).sqrt()
    return F.relu(target_std - std).square().mean()


def _covariance_loss(state: torch.Tensor) -> torch.Tensor:
    if state.ndim != 3 or state.shape[0] < 2:
        raise ValueError("state must have shape (B,H,D) with B at least two")
    batch, horizons, dim = state.shape
    centered = state - state.mean(dim=0, keepdim=True)
    covariance = torch.einsum("bhd,bhe->hde", centered, centered) / float(batch - 1)
    diagonal = torch.diagonal(covariance, dim1=-2, dim2=-1)
    off_diagonal_square_sum = covariance.square().sum(dim=(-2, -1)) - diagonal.square().sum(
        dim=-1
    )
    return (off_diagonal_square_sum / float(dim)).mean()


def _mean_loss(state: torch.Tensor) -> torch.Tensor:
    if state.ndim != 3 or state.shape[0] < 2:
        raise ValueError("state must have shape (B,H,D) with B at least two")
    return state.mean(dim=0).square().mean()


class WhitenedDeltaPredictiveStateH4JEPA(nn.Module):
    """Joint RGB/action predictor and learned compact future-change target."""

    history_steps = 3
    past_action_steps = 2
    future_steps = 4

    def __init__(
        self,
        n320_encoder_state_dict: Mapping[str, torch.Tensor],
        *,
        config: WhitenedDeltaPredictiveStateConfig | None = None,
    ) -> None:
        super().__init__()
        self.config = config or WhitenedDeltaPredictiveStateConfig()
        if not isinstance(self.config, WhitenedDeltaPredictiveStateConfig):
            raise TypeError("config must be WhitenedDeltaPredictiveStateConfig")
        dim = self.config.feature_dim
        self.encoder = VisionEncoder(
            image_size=self.config.image_size,
            patch_size=self.config.patch_size,
            hidden_dim=dim,
            depth=self.config.encoder_depth,
            n_heads=self.config.encoder_heads,
            mlp_ratio=self.config.encoder_mlp_ratio,
            dropout=self.config.dropout,
        )
        _validate_encoder_state(self.encoder, n320_encoder_state_dict)
        self.encoder.load_state_dict(n320_encoder_state_dict, strict=True)
        self.target_encoder = copy.deepcopy(self.encoder)

        self.action_embedding = nn.Embedding(self.config.action_count, dim)
        nn.init.normal_(self.action_embedding.weight, mean=0.0, std=0.02)
        self.initial_belief = _DenseHistoricalContext(
            spatial_tokens=self.spatial_token_count,
            feature_dim=dim,
            history_steps=self.history_steps,
            past_action_steps=self.past_action_steps,
            heads=self.config.cross_attention_heads,
            mlp_ratio=self.config.cross_attention_mlp_ratio,
            dropout=self.config.dropout,
        )
        self.future_cell = _DenseHorizonCrossAttention(
            spatial_tokens=self.spatial_token_count,
            feature_dim=dim,
            future_steps=self.future_steps,
            heads=self.config.cross_attention_heads,
            mlp_ratio=self.config.cross_attention_mlp_ratio,
            dropout=self.config.dropout,
        )
        self.state_projector = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, self.config.state_dim),
        )
        nn.init.zeros_(self.state_projector[-1].weight)
        nn.init.zeros_(self.state_projector[-1].bias)
        self.target_state_compressor = _BiasFreeDeltaAttentionPool(
            dim,
            self.config.state_dim,
            self.spatial_token_count,
        )
        self.register_buffer(
            "ema_update_count",
            torch.zeros((), dtype=torch.long),
            persistent=True,
        )
        self._freeze_target()

    @property
    def action_vocabulary(self) -> tuple[str, ...]:
        return self.config.action_vocabulary

    @property
    def spatial_token_count(self) -> int:
        return self.config.spatial_token_count

    def _freeze_target(self) -> None:
        self.target_encoder.requires_grad_(False)
        self.target_encoder.eval()

    def train(self, mode: bool = True) -> "WhitenedDeltaPredictiveStateH4JEPA":
        super().train(mode)
        self._freeze_target()
        return self

    def hard_sync_target(self) -> None:
        self.ema_update_count.zero_()
        self._freeze_target()

    def update_target(self, momentum: float | None = None) -> None:
        del momentum
        self._freeze_target()

    def _validate_rgb(self, rgb: torch.Tensor, *, steps: int, name: str) -> int:
        expected = (steps, 3, self.config.image_size, self.config.image_size)
        if rgb.ndim != 5 or tuple(rgb.shape[1:]) != expected:
            raise ValueError(f"{name} must have shape (B,{','.join(map(str, expected))})")
        if rgb.shape[0] < 1 or rgb.dtype != torch.float32:
            raise TypeError(f"{name} must contain float32 rows")
        if rgb.device != self.action_embedding.weight.device:
            raise TypeError(f"{name} and model must share a device")
        if not bool(torch.isfinite(rgb).all()):
            raise FloatingPointError(f"{name} contains a nonfinite value")
        return int(rgb.shape[0])

    def _validate_actions(
        self,
        actions: torch.Tensor,
        *,
        batch: int,
        steps: int,
        name: str,
    ) -> None:
        if actions.shape != (batch, steps) or actions.dtype != torch.long:
            raise TypeError(f"{name} must be long with shape ({batch},{steps})")
        if actions.device != self.action_embedding.weight.device:
            raise TypeError(f"{name} and model must share a device")
        if bool((actions < 0).any()) or bool(
            (actions >= self.config.action_count).any()
        ):
            raise ValueError(f"{name} entries are outside the action vocabulary")

    def _encode_online(self, rgb: torch.Tensor) -> torch.Tensor:
        batch, steps = rgb.shape[:2]
        tokens = self.encoder.forward_tokens(
            rgb.reshape(batch * steps, *rgb.shape[2:])
        )[:, 1:]
        return tokens.reshape(
            batch,
            steps,
            self.spatial_token_count,
            self.config.feature_dim,
        )

    @torch.no_grad()
    def _encode_fixed_history(self, history_rgb: torch.Tensor) -> torch.Tensor:
        batch = int(history_rgb.shape[0])
        tokens = self.target_encoder.forward_tokens(
            history_rgb.reshape(batch * self.history_steps, *history_rgb.shape[2:])
        )[:, 1:]
        return F.normalize(
            tokens.reshape(
                batch,
                self.history_steps,
                self.spatial_token_count,
                self.config.feature_dim,
            ),
            dim=-1,
            eps=self.config.normalization_epsilon,
        ).detach()

    @torch.no_grad()
    def _encode_fixed_future(self, future_rgb: torch.Tensor) -> torch.Tensor:
        batch = int(future_rgb.shape[0])
        tokens = self.target_encoder.forward_tokens(
            future_rgb.reshape(batch * self.future_steps, *future_rgb.shape[2:])
        )[:, 1:]
        return F.normalize(
            tokens.reshape(
                batch,
                self.future_steps,
                self.spatial_token_count,
                self.config.feature_dim,
            ),
            dim=-1,
            eps=self.config.normalization_epsilon,
        ).detach()

    def encode_history(
        self,
        history_rgb: torch.Tensor,
        past_actions: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch = self._validate_rgb(
            history_rgb, steps=self.history_steps, name="history_rgb"
        )
        self._validate_actions(
            past_actions,
            batch=batch,
            steps=self.past_action_steps,
            name="past_actions",
        )
        history = self._encode_online(history_rgb)
        normalized = F.normalize(
            history,
            dim=-1,
            eps=self.config.normalization_epsilon,
        )
        memory = self.initial_belief(normalized, self.action_embedding(past_actions))
        belief = torch.cat((history[:, 2], memory), dim=1)
        return history, belief

    def predict_from_belief(
        self,
        belief_latents: torch.Tensor,
        future_actions: torch.Tensor,
    ) -> torch.Tensor:
        tokens = self.spatial_token_count
        dim = self.config.feature_dim
        expected_tokens = tokens + self.history_steps * tokens + self.past_action_steps
        if belief_latents.shape[1:] != (expected_tokens, dim):
            raise ValueError("belief latent shape changed")
        batch = int(belief_latents.shape[0])
        self._validate_actions(
            future_actions,
            batch=batch,
            steps=self.future_steps,
            name="future_actions",
        )
        anchor = belief_latents[:, :tokens]
        memory = belief_latents[:, tokens:]
        hidden = self.future_cell(
            F.normalize(anchor, dim=-1, eps=self.config.normalization_epsilon),
            memory,
            self.action_embedding(future_actions),
            self.initial_belief.spatial_embedding.weight,
        )
        return self.state_projector(hidden.mean(dim=2))

    def encode_target_state(
        self,
        current_teacher_tokens: torch.Tensor,
        future_rgb: torch.Tensor,
    ) -> torch.Tensor:
        batch = self._validate_rgb(
            future_rgb, steps=self.future_steps, name="future_rgb"
        )
        if current_teacher_tokens.shape != (
            batch,
            self.spatial_token_count,
            self.config.feature_dim,
        ):
            raise ValueError("current fixed-teacher token shape changed")
        future = self._encode_fixed_future(future_rgb)
        delta = future - current_teacher_tokens.detach()[:, None]
        return self.target_state_compressor(delta)

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
        predicted_variance: torch.Tensor | None = None
        target_variance: torch.Tensor | None = None
        predicted_mean: torch.Tensor | None = None
        target_mean: torch.Tensor | None = None
        predicted_covariance: torch.Tensor | None = None
        target_covariance: torch.Tensor | None = None
        alignment: torch.Tensor | None = None
        total: torch.Tensor | None = None
        fixed_history: torch.Tensor | None = None
        if future_rgb is not None:
            fixed_history = self._encode_fixed_history(history_rgb)
            target = self.encode_target_state(fixed_history[:, 2], future_rgb)
            similarity = (predicted - target).square().mean()
            predicted_variance = _variance_loss(
                predicted,
                target_std=self.config.variance_target_std,
                epsilon=self.config.variance_epsilon,
            )
            target_variance = _variance_loss(
                target,
                target_std=self.config.variance_target_std,
                epsilon=self.config.variance_epsilon,
            )
            predicted_mean = _mean_loss(predicted)
            target_mean = _mean_loss(target)
            predicted_covariance = _covariance_loss(predicted)
            target_covariance = _covariance_loss(target)
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
                + self.config.variance_regularization_weight
                * 0.5
                * (predicted_variance + target_variance)
                + self.config.mean_regularization_weight
                * 0.5
                * (predicted_mean + target_mean)
                + self.config.covariance_regularization_weight
                * 0.5
                * (predicted_covariance + target_covariance)
                + self.config.history_teacher_alignment_weight * alignment
            )
        return WhitenedDeltaPredictiveStateOutput(
            predicted_state=predicted,
            target_state=target,
            history_latents=history,
            fixed_history_latents=fixed_history,
            belief_latents=belief,
            state_prediction_loss=similarity,
            predicted_variance_loss=predicted_variance,
            target_variance_loss=target_variance,
            predicted_mean_loss=predicted_mean,
            target_mean_loss=target_mean,
            predicted_covariance_loss=predicted_covariance,
            target_covariance_loss=target_covariance,
            history_teacher_alignment_loss=alignment,
            total_loss=total,
        )


__all__ = [
    "GO2_H4_PRIMITIVE_VOCABULARY",
    "WhitenedDeltaPredictiveStateConfig",
    "WhitenedDeltaPredictiveStateH4JEPA",
    "WhitenedDeltaPredictiveStateOutput",
]
