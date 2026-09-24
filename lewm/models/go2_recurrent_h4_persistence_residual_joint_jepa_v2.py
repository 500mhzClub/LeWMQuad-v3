"""Persistence-anchored recurrent H4 joint JEPA V2.

V1 learned strong action identity but discarded the useful current visual state:
its absolute prediction was worse than persistence and ordered history was
harmful.  V2 changes only those failed mechanisms.  The online e2 spatial
tokens are an identity anchor, ordered history is a zero-gated correction, and
each future action contributes a zero-gated residual update.  With both gates
at initialization, every horizon is exactly the current online latent.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Mapping

import torch
import torch.nn as nn
import torch.nn.functional as F

from .go2_recurrent_h4_joint_jepa import (
    GO2_H4_PRIMITIVE_VOCABULARY,
    JointRecurrentH4JEPA as _V1JointRecurrentH4JEPA,
    JointRecurrentH4JEPAConfig as _V1Config,
    JointRecurrentH4JEPAOutput,
    OnlineH4Context,
)


@dataclass(frozen=True)
class JointRecurrentH4JEPAConfig(_V1Config):
    """V2 configuration; inherited defaults keep V1 science identical."""

    persistence_target_ratio: float = 0.90
    persistence_ranking_weight: float = 1.0
    history_margin_fraction: float = 0.03
    history_ranking_weight: float = 1.0

    def __post_init__(self) -> None:
        super().__post_init__()
        values = (
            self.persistence_target_ratio,
            self.persistence_ranking_weight,
            self.history_margin_fraction,
            self.history_ranking_weight,
        )
        if any(not math.isfinite(value) for value in values):
            raise ValueError("V2 ranking constants must be finite")
        if not 0.0 < self.persistence_target_ratio < 1.0:
            raise ValueError("persistence_target_ratio must lie in (0,1)")
        if self.persistence_ranking_weight < 0.0:
            raise ValueError("persistence_ranking_weight must be non-negative")
        if self.history_margin_fraction < 0.0:
            raise ValueError("history_margin_fraction must be non-negative")
        if self.history_ranking_weight < 0.0:
            raise ValueError("history_ranking_weight must be non-negative")


class _ZeroGatedResidual(nn.Module):
    """A learnable residual whose initial output is exactly zero."""

    def __init__(self, feature_dim: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(feature_dim)
        self.linear = nn.Linear(feature_dim, feature_dim)
        self.gate = nn.Parameter(torch.zeros(()))

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.gate) * self.linear(self.norm(value))


class JointRecurrentH4JEPA(_V1JointRecurrentH4JEPA):
    """V1-compatible API with persistence-anchored residual dynamics."""

    def __init__(
        self,
        n320_encoder_state_dict: Mapping[str, torch.Tensor] | None = None,
        *,
        config: JointRecurrentH4JEPAConfig | None = None,
    ) -> None:
        resolved = config or JointRecurrentH4JEPAConfig()
        if not isinstance(resolved, JointRecurrentH4JEPAConfig):
            raise TypeError("config must be the V2 JointRecurrentH4JEPAConfig")
        super().__init__(
            n320_encoder_state_dict=n320_encoder_state_dict,
            config=resolved,
        )
        dim = self.config.feature_dim
        # Keep the V1 module names so the reviewed optimizer inventory remains
        # complete while changing their semantics from absolute to residual.
        self.initial_belief = _ZeroGatedResidual(dim)
        self.prediction_projector = _ZeroGatedResidual(dim)

    def encode_history(
        self,
        history_rgb: torch.Tensor,
        past_actions: torch.Tensor,
    ) -> OnlineH4Context:
        """Return e2 plus a zero-gated correction from e0,p0,e1,p1,e2."""

        batch = self._validate_rgb_sequence(
            history_rgb,
            steps=self.history_steps,
            name="history_rgb",
        )
        self._validate_actions(
            past_actions,
            batch=batch,
            steps=self.past_action_steps,
            name="past_actions",
        )
        history = self._encode_online_spatial(history_rgb)
        hidden = history[:, 0]
        tokens = self.spatial_token_count
        dim = self.config.feature_dim
        for step in range(self.past_action_steps):
            observation = self.history_observation_norm(history[:, step + 1])
            action = self.action_embedding(past_actions[:, step])[:, None].expand(
                -1, tokens, -1
            )
            recurrent_input = torch.cat((observation, action), dim=-1)
            hidden = self.history_cell(
                recurrent_input.reshape(batch * tokens, 2 * dim),
                hidden.reshape(batch * tokens, dim),
            ).reshape(batch, tokens, dim)
            hidden = self.history_spatial_refiner(hidden)
        current = history[:, 2]
        belief = current + self.initial_belief(hidden - current)
        return OnlineH4Context(
            history_latents=history,
            belief_latents=belief,
        )

    def predict_from_belief(
        self,
        belief_latents: torch.Tensor,
        future_actions: torch.Tensor,
    ) -> torch.Tensor:
        """Accumulate shared action-conditioned residuals from the e2 anchor."""

        expected = (self.spatial_token_count, self.config.feature_dim)
        if belief_latents.ndim != 3 or tuple(belief_latents.shape[1:]) != expected:
            raise ValueError(
                f"belief_latents must have shape (B,{expected[0]},{expected[1]})"
            )
        batch = int(belief_latents.shape[0])
        if not belief_latents.is_floating_point():
            raise TypeError("belief_latents must be floating point")
        if belief_latents.device != self.action_embedding.weight.device:
            raise TypeError("belief_latents and model must share a device")
        if not bool(torch.isfinite(belief_latents).all()):
            raise FloatingPointError("belief_latents contains a nonfinite value")
        self._validate_actions(
            future_actions,
            batch=batch,
            steps=self.future_steps,
            name="future_actions",
        )

        tokens = self.spatial_token_count
        dim = self.config.feature_dim
        state = belief_latents
        predictions: list[torch.Tensor] = []
        for step in range(self.future_steps):
            action = self.action_embedding(future_actions[:, step])[:, None].expand(
                -1, tokens, -1
            )
            candidate = self.future_cell(
                action.reshape(batch * tokens, dim),
                state.reshape(batch * tokens, dim),
            ).reshape(batch, tokens, dim)
            candidate = self.future_spatial_refiner(candidate)
            state = state + self.prediction_projector(candidate)
            predictions.append(
                F.normalize(
                    state,
                    p=2.0,
                    dim=-1,
                    eps=self.config.normalization_epsilon,
                )
            )
        return torch.stack(predictions, dim=1)

    @staticmethod
    def _distance(predicted: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if predicted.shape != target.shape or predicted.ndim != 4:
            raise ValueError("V2 ranking latents must share shape (B,4,N,D)")
        return (predicted - target).square().sum(dim=-1).mean(dim=-1)

    def training_auxiliary_losses(
        self,
        *,
        history_rgb: torch.Tensor,
        past_actions: torch.Tensor,
        future_actions: torch.Tensor,
        target_latents: torch.Tensor,
        output: JointRecurrentH4JEPAOutput,
    ) -> dict[str, torch.Tensor]:
        """Require real residuals to beat persistence and corrupted history."""

        if not isinstance(output, JointRecurrentH4JEPAOutput):
            raise TypeError("output must be JointRecurrentH4JEPAOutput")
        predicted = output.predicted_latents
        correct_distance = self._distance(predicted, target_latents)

        repeated_current = history_rgb[:, 2:3].expand(-1, 4, -1, -1, -1).contiguous()
        persistence = self.encode_target(repeated_current)
        persistence_distance = self._distance(
            persistence,
            target_latents,
        ).detach()
        active = persistence_distance >= 1e-4

        def active_mean(value: torch.Tensor) -> torch.Tensor:
            weights = active.to(value.dtype)
            return (value * weights).sum() / weights.sum().clamp_min(1.0)

        persistence_ranking = active_mean(F.relu(
            correct_distance
            - self.config.persistence_target_ratio * persistence_distance
        ))

        # The causal history ablation keeps e2 and the entire future/action path
        # identical and disables only the learned history correction.  Detach
        # it so the margin cannot be won by making the control worse.
        with torch.no_grad():
            no_history_prediction = self.predict_from_belief(
                output.history_latents[:, 2].detach(),
                future_actions,
            )
            no_history_distance = self._distance(
                no_history_prediction,
                target_latents,
            )
        history_ranking = active_mean(F.relu(
            correct_distance
            + self.config.history_margin_fraction * persistence_distance
            - no_history_distance.detach()
        ))
        return {
            "persistence_ranking": (
                self.config.persistence_ranking_weight * persistence_ranking
            ),
            "history_ranking": self.config.history_ranking_weight * history_ranking,
        }


__all__ = [
    "GO2_H4_PRIMITIVE_VOCABULARY",
    "JointRecurrentH4JEPA",
    "JointRecurrentH4JEPAConfig",
]
