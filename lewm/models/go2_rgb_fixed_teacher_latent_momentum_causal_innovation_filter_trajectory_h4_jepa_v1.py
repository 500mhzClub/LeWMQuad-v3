"""Fixed-teacher latent-momentum causal innovation-filter H4 JEPA V1.

The model keeps four causal state atoms ``(q, v)``.  One shared prior uses a
mean-centered categorical action correction to update latent momentum and then
integrates that momentum into latent content.  On the two observed edges the
prior is emitted before one shared observer assimilates the newly available
online innovation.  The four future edges reuse the same prior recursively and
receive no RGB latent, explicit frame difference, anchor, or horizon query.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import torch
import torch.nn as nn
import torch.nn.functional as F

from .encoders import ViTBlock
from .go2_rgb_fixed_teacher_factorized_conditional_increment_trajectory_h4_jepa_v1 import (
    FactorizedConditionalIncrementTrajectoryH4JEPA,
    FactorizedConditionalIncrementTrajectoryH4JEPAConfig,
    FactualSharedTransitionTrajectoryH4JEPAOutput,
    GO2_H4_PRIMITIVE_VOCABULARY,
    fixed_teacher_local_innovations,
    realized_trajectory_innovations,
    trajectory_energy_score,
)
from .go2_rgb_fixed_teacher_local_innovation_trajectory_h4_jepa_v1 import (
    _renormalized_local_step,
)


@dataclass(frozen=True)
class LatentMomentumCausalInnovationFilterTrajectoryH4JEPAConfig(
    FactorizedConditionalIncrementTrajectoryH4JEPAConfig
):
    """The inherited fixed-teacher, K4, and proper-score contract."""


def _tangent_projection(
    content: torch.Tensor,
    momentum: torch.Tensor,
    *,
    epsilon: float,
) -> torch.Tensor:
    """Remove each feature token's radial momentum component."""

    if content.shape != momentum.shape:
        raise ValueError("content and momentum shapes differ")
    radius_squared = content.square().sum(dim=-1, keepdim=True).clamp_min(
        epsilon
    )
    radial_scale = (content * momentum).sum(
        dim=-1,
        keepdim=True,
    ) / radius_squared
    return momentum - radial_scale * content


class _CenteredStateModes(nn.Module):
    """Create the fixed initial state and own centered K/spatial identities."""

    def __init__(
        self,
        *,
        spatial_tokens: int,
        feature_dim: int,
        atom_count: int,
    ) -> None:
        super().__init__()
        self.spatial_tokens = spatial_tokens
        self.atom_count = atom_count
        self.mode_embedding = nn.Embedding(atom_count, feature_dim)
        self.spatial_embedding = nn.Embedding(spatial_tokens, feature_dim)
        nn.init.normal_(self.mode_embedding.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.spatial_embedding.weight, mean=0.0, std=0.02)

    def centered_context(
        self,
        *,
        batch: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        modes = self.mode_embedding.weight
        modes = modes - modes.mean(dim=0, keepdim=True)
        context = modes[:, None] + self.spatial_embedding.weight[None]
        return context.to(device=device, dtype=dtype)[None].expand(
            batch,
            -1,
            -1,
            -1,
        )

    def forward(
        self,
        initial_content: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if initial_content.ndim != 3:
            raise ValueError("initial content must have shape B,P,D")
        batch, tokens, _dim = initial_content.shape
        if tokens != self.spatial_tokens:
            raise ValueError("initial content spatial token count changed")
        content = initial_content[:, None].expand(
            batch,
            self.atom_count,
            tokens,
            initial_content.shape[-1],
        )
        return content, torch.zeros_like(content)


class _InnovationObserver(nn.Module):
    """Assimilate one post-prior online observation innovation into (q, v)."""

    layer_count = 1

    def __init__(
        self,
        *,
        spatial_tokens: int,
        feature_dim: int,
        heads: int,
        mlp_ratio: int,
        dropout: float,
        epsilon: float,
    ) -> None:
        super().__init__()
        self.spatial_tokens = spatial_tokens
        self.epsilon = epsilon
        self.content_path = nn.Sequential(
            nn.LayerNorm(feature_dim, elementwise_affine=False),
            nn.Linear(feature_dim, feature_dim, bias=False),
        )
        self.momentum_path = nn.Sequential(
            nn.LayerNorm(feature_dim, elementwise_affine=False),
            nn.Linear(feature_dim, feature_dim, bias=False),
        )
        self.innovation_path = nn.Sequential(
            nn.LayerNorm(feature_dim, elementwise_affine=False),
            nn.Linear(feature_dim, feature_dim, bias=False),
        )
        self.spatial_block = ViTBlock(
            hidden_dim=feature_dim,
            n_heads=heads,
            mlp_ratio=mlp_ratio,
            dropout=dropout,
        )
        self.content_gain_head = nn.Linear(
            feature_dim,
            feature_dim,
            bias=False,
        )
        self.momentum_correction_head = nn.Linear(
            feature_dim,
            feature_dim,
            bias=False,
        )
        nn.init.zeros_(self.content_gain_head.weight)
        nn.init.zeros_(self.momentum_correction_head.weight)

    def forward(
        self,
        prior_content: torch.Tensor,
        prior_momentum: torch.Tensor,
        innovation: torch.Tensor,
        mode_context: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if (
            prior_content.ndim != 4
            or prior_momentum.shape != prior_content.shape
            or innovation.shape != prior_content.shape
            or mode_context.shape != prior_content.shape
        ):
            raise ValueError("observer inputs must share shape B,K,P,D")
        batch, atoms, tokens, dim = prior_content.shape
        if tokens != self.spatial_tokens:
            raise ValueError("observer spatial token count changed")
        observer_input = (
            self.content_path(prior_content)
            + self.momentum_path(prior_momentum)
            + self.innovation_path(innovation)
            + mode_context
        )
        context = self.spatial_block(
            observer_input.reshape(batch * atoms, tokens, dim)
        ).reshape(batch, atoms, tokens, dim)
        gain = 1.0 + torch.tanh(self.content_gain_head(context))
        posterior_content = _renormalized_local_step(
            prior_content,
            gain * innovation,
            epsilon=self.epsilon,
        )
        corrected_momentum = (
            prior_momentum + self.momentum_correction_head(context)
        )
        posterior_momentum = _tangent_projection(
            posterior_content,
            corrected_momentum,
            epsilon=self.epsilon,
        )
        return posterior_content, posterior_momentum, context


class LatentMomentumCausalInnovationFilterTrajectoryH4JEPA(
    FactorizedConditionalIncrementTrajectoryH4JEPA
):
    """K4 predict-before-observe filter with state-only future rollout."""

    def __init__(
        self,
        n320_encoder_state_dict: Mapping[str, torch.Tensor] | None = None,
        *,
        config: (
            LatentMomentumCausalInnovationFilterTrajectoryH4JEPAConfig | None
        ) = None,
    ) -> None:
        selected = (
            config
            or LatentMomentumCausalInnovationFilterTrajectoryH4JEPAConfig()
        )
        if not isinstance(
            selected,
            LatentMomentumCausalInnovationFilterTrajectoryH4JEPAConfig,
        ):
            raise TypeError(
                "config must be "
                "LatentMomentumCausalInnovationFilterTrajectoryH4JEPAConfig"
            )
        super().__init__(
            n320_encoder_state_dict=n320_encoder_state_dict,
            config=selected,
        )
        dim = self.config.feature_dim
        self.initial_belief = _CenteredStateModes(
            spatial_tokens=self.spatial_token_count,
            feature_dim=dim,
            atom_count=self.config.trajectory_atom_count,
        )
        self.history_cell = _InnovationObserver(
            spatial_tokens=self.spatial_token_count,
            feature_dim=dim,
            heads=self.config.cross_attention_heads,
            mlp_ratio=self.config.cross_attention_mlp_ratio,
            dropout=self.config.dropout,
            epsilon=self.config.normalization_epsilon,
        )

    def _mode_context(self, reference: torch.Tensor) -> torch.Tensor:
        return self.initial_belief.centered_context(
            batch=int(reference.shape[0]),
            device=reference.device,
            dtype=reference.dtype,
        )

    def _transition_step(
        self,
        content: torch.Tensor,
        momentum: torch.Tensor,
        action_indices: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Apply the sole state-only prior transition and identity readout."""

        if content.ndim != 4 or momentum.shape != content.shape:
            raise ValueError("transition state must have shape B,K,P,D")
        if action_indices.ndim != 1 or action_indices.dtype != torch.long:
            raise TypeError("one-step actions must be long with shape (B,)")
        batch, _atoms, _tokens, dim = content.shape
        if action_indices.shape != (batch,):
            raise ValueError("one-step action batch size changed")
        if action_indices.device != content.device:
            raise TypeError("state and action must share a device")
        state_context = self.future_cell(
            content,
            momentum,
            self._mode_context(content),
        )
        action_codes = self._centered_action_codes()
        selected_codes = action_codes.index_select(0, action_indices)
        if selected_codes.shape != (batch, dim):
            raise ValueError("selected action-code shape changed")
        acceleration = self.prediction_projector(
            state_context * selected_codes[:, None, None]
        )
        candidate_momentum = momentum + acceleration
        next_content = _renormalized_local_step(
            content,
            candidate_momentum,
            epsilon=self.config.normalization_epsilon,
        )
        next_momentum = _tangent_projection(
            next_content,
            candidate_momentum,
            epsilon=self.config.normalization_epsilon,
        )
        realized_increment = next_content - content
        return next_content, next_momentum, acceleration, realized_increment

    def _observe(
        self,
        prior_content: torch.Tensor,
        prior_momentum: torch.Tensor,
        observation: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if observation.ndim != 3:
            raise ValueError("observation must have shape B,P,D")
        expanded = observation[:, None].expand_as(prior_content)
        innovation = expanded - prior_content
        posterior_content, posterior_momentum, _context = self.history_cell(
            prior_content,
            prior_momentum,
            innovation,
            self._mode_context(prior_content),
        )
        return posterior_content, posterior_momentum, innovation

    def _pack_belief(
        self,
        content: torch.Tensor,
        momentum: torch.Tensor,
    ) -> torch.Tensor:
        if content.ndim != 4 or momentum.shape != content.shape:
            raise ValueError("belief state must have matching q/v lattices")
        return torch.cat((content, momentum), dim=1)

    def _unpack_belief(
        self,
        belief_latents: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        atom_count = self.config.trajectory_atom_count
        expected = (
            2 * atom_count,
            self.spatial_token_count,
            self.config.feature_dim,
        )
        if belief_latents.ndim != 4 or tuple(belief_latents.shape[1:]) != expected:
            raise ValueError(
                "belief_latents must contain only packed q/v with shape "
                f"(B,{expected[0]},{expected[1]},{expected[2]})"
            )
        if not torch.is_floating_point(belief_latents):
            raise TypeError("belief_latents must have a floating dtype")
        if belief_latents.device != self.action_embedding.weight.device:
            raise TypeError("belief_latents and model must share a device")
        if not bool(torch.isfinite(belief_latents).all()):
            raise FloatingPointError("belief_latents contains a nonfinite value")
        return (
            belief_latents[:, :atom_count],
            belief_latents[:, atom_count:],
        )

    def _encode_factual_history(
        self,
        history_rgb: torch.Tensor,
        past_actions: torch.Tensor,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
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
        normalized = F.normalize(
            history,
            p=2.0,
            dim=-1,
            eps=self.config.normalization_epsilon,
        )
        content, momentum = self.initial_belief(normalized[:, 0])
        priors: list[torch.Tensor] = []
        realized: list[torch.Tensor] = []
        scored_innovations: list[torch.Tensor] = []
        for step in range(self.past_action_steps):
            prior_content, prior_momentum, _acceleration, realized_increment = (
                self._transition_step(
                    content,
                    momentum,
                    past_actions[:, step],
                )
            )
            priors.append(prior_content)
            realized.append(realized_increment)
            factual_source = normalized[:, step, None].expand_as(prior_content)
            scored_innovations.append(prior_content - factual_source)
            content, momentum, _innovation = self._observe(
                prior_content,
                prior_momentum,
                normalized[:, step + 1],
            )

        belief = self._pack_belief(content, momentum)
        return (
            history,
            belief,
            torch.stack(priors, dim=2),
            torch.stack(realized, dim=2),
            torch.stack(scored_innovations, dim=2),
        )

    def _rollout_future(
        self,
        belief_latents: torch.Tensor,
        future_actions: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        content, momentum = self._unpack_belief(belief_latents)
        batch = int(belief_latents.shape[0])
        self._validate_actions(
            future_actions,
            batch=batch,
            steps=self.future_steps,
            name="future_actions",
        )
        states: list[torch.Tensor] = []
        realized: list[torch.Tensor] = []
        for step in range(self.future_steps):
            content, momentum, _acceleration, realized_increment = (
                self._transition_step(
                    content,
                    momentum,
                    future_actions[:, step],
                )
            )
            states.append(content)
            realized.append(realized_increment)
        stacked_realized = torch.stack(realized, dim=2)
        return (
            torch.stack(states, dim=2),
            stacked_realized,
            stacked_realized,
            momentum,
        )


# Preserve the reviewed shared-runner constructor API.
JointRecurrentH4JEPAConfig = (
    LatentMomentumCausalInnovationFilterTrajectoryH4JEPAConfig
)
JointRecurrentH4JEPA = LatentMomentumCausalInnovationFilterTrajectoryH4JEPA


__all__ = [
    "FactualSharedTransitionTrajectoryH4JEPAOutput",
    "GO2_H4_PRIMITIVE_VOCABULARY",
    "JointRecurrentH4JEPA",
    "JointRecurrentH4JEPAConfig",
    "LatentMomentumCausalInnovationFilterTrajectoryH4JEPA",
    "LatentMomentumCausalInnovationFilterTrajectoryH4JEPAConfig",
    "fixed_teacher_local_innovations",
    "realized_trajectory_innovations",
    "trajectory_energy_score",
]
