"""Fixed-teacher factorized conditional-increment trajectory-H4 JEPA V1.

The learned transition is factorized into an action-free belief context ``B``,
an incoming-increment modulation ``D``, and a uniformly centered categorical
action code ``c_a``.  There is deliberately no learned current-state-only
successor path::

    B = S(z, h, D(d))
    c_a = A(E[a]) - mean_j A(E[j])
    raw = d + B * (1 + tanh(D(d))) * c_a
    increment = W0(raw)
    z_next = normalize(z + increment)

``W0`` is one shared, bias-free, zero-initialized linear map.  Thus update zero
is exact persistence.  If the action tower collapses, the state-conditioned
term is identically zero; the only remaining signal is the explicit incoming
realized increment ``d``.  The action-free belief context can never directly
emit a generic successor.

The two observed edges retain the factual-carrier contract.  ``p0`` receives
``d=0``; ``p1`` receives the factual online increment ``e1-e0``.  The packed
future belief carries ``e2``, factual ``e2-e1``, and four causal hidden
particles.  Future increments are then carried recursively.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import torch
import torch.nn as nn
import torch.nn.functional as F

from .encoders import ViTBlock
from .go2_recurrent_h4_joint_jepa import GO2_H4_PRIMITIVE_VOCABULARY
from .go2_rgb_fixed_teacher_factual_shared_transition_trajectory_h4_jepa_v1 import (
    FactualSharedTransitionTrajectoryH4JEPA,
    FactualSharedTransitionTrajectoryH4JEPAConfig,
    FactualSharedTransitionTrajectoryH4JEPAOutput,
    fixed_teacher_local_innovations,
    realized_trajectory_innovations,
    trajectory_energy_score,
)
from .go2_rgb_fixed_teacher_local_innovation_trajectory_h4_jepa_v1 import (
    _renormalized_local_step,
)


@dataclass(frozen=True)
class FactorizedConditionalIncrementTrajectoryH4JEPAConfig(
    FactualSharedTransitionTrajectoryH4JEPAConfig
):
    """The inherited K4, dimensions, and proper-score contract are unchanged."""


class _IncomingIncrementModulator(nn.Module):
    """Compute D(d), preserving the exact zero-input fixed point."""

    def __init__(self, feature_dim: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(feature_dim, elementwise_affine=False)
        self.projection = nn.Linear(feature_dim, feature_dim, bias=False)

    def forward(self, incoming_increment: torch.Tensor) -> torch.Tensor:
        if incoming_increment.ndim != 4:
            raise ValueError("incoming increment must have shape B,K,P,D")
        return self.projection(self.norm(incoming_increment))


class _ActionFreeBeliefContext(nn.Module):
    """Update the causal particle context without seeing the current action."""

    layer_count = 1

    def __init__(
        self,
        *,
        spatial_tokens: int,
        feature_dim: int,
        heads: int,
        mlp_ratio: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.spatial_tokens = spatial_tokens
        self.visual_path = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Linear(feature_dim, feature_dim),
        )
        self.spatial_block = ViTBlock(
            hidden_dim=feature_dim,
            n_heads=heads,
            mlp_ratio=mlp_ratio,
            dropout=dropout,
        )

    def forward(
        self,
        visual: torch.Tensor,
        hidden: torch.Tensor,
        increment_modulation: torch.Tensor,
    ) -> torch.Tensor:
        if (
            visual.ndim != 4
            or hidden.shape != visual.shape
            or increment_modulation.shape != visual.shape
        ):
            raise ValueError("B inputs must share shape B,K,P,D")
        batch, atom_count, tokens, dim = visual.shape
        if tokens != self.spatial_tokens:
            raise ValueError("B spatial token count changed")
        transition_input = (
            hidden
            + self.visual_path(visual)
            + increment_modulation
        )
        transitioned = self.spatial_block(
            transition_input.reshape(batch * atom_count, tokens, dim)
        )
        return transitioned.reshape(batch, atom_count, tokens, dim)


class _CenteredCategoricalActionTower(nn.Module):
    """Map all action embeddings, then remove their complete-tower mean."""

    def __init__(self, feature_dim: int) -> None:
        super().__init__()
        self.tower = nn.Sequential(
            nn.LayerNorm(feature_dim, elementwise_affine=False),
            nn.Linear(feature_dim, feature_dim, bias=False),
            nn.Tanh(),
            nn.Linear(feature_dim, feature_dim, bias=False),
        )

    def forward(self, action_embeddings: torch.Tensor) -> torch.Tensor:
        if action_embeddings.ndim != 2:
            raise ValueError("action embeddings must have shape A,D")
        transformed = self.tower(action_embeddings)
        return transformed - transformed.mean(dim=0, keepdim=True)


class FactorizedConditionalIncrementTrajectoryH4JEPA(
    FactualSharedTransitionTrajectoryH4JEPA
):
    """K4 shared transition with explicit B/D/centered-action factorization."""

    def __init__(
        self,
        n320_encoder_state_dict: Mapping[str, torch.Tensor] | None = None,
        *,
        config: FactorizedConditionalIncrementTrajectoryH4JEPAConfig | None = None,
    ) -> None:
        selected = config or FactorizedConditionalIncrementTrajectoryH4JEPAConfig()
        if not isinstance(
            selected,
            FactorizedConditionalIncrementTrajectoryH4JEPAConfig,
        ):
            raise TypeError(
                "config must be FactorizedConditionalIncrementTrajectoryH4JEPAConfig"
            )
        super().__init__(
            n320_encoder_state_dict=n320_encoder_state_dict,
            config=selected,
        )

        dim = self.config.feature_dim
        # These exact attribute names preserve the shared runner's reviewed
        # encoder/history/predictor parameter grouping.
        self.history_cell = _IncomingIncrementModulator(dim)
        self.future_cell = _ActionFreeBeliefContext(
            spatial_tokens=self.spatial_token_count,
            feature_dim=dim,
            heads=self.config.cross_attention_heads,
            mlp_ratio=self.config.cross_attention_mlp_ratio,
            dropout=self.config.dropout,
        )
        self.future_spatial_refiner = _CenteredCategoricalActionTower(dim)
        self.prediction_projector = nn.Sequential(
            nn.Linear(dim, dim, bias=False),
        )
        nn.init.zeros_(self.prediction_projector[-1].weight)

    def _centered_action_codes(self) -> torch.Tensor:
        codes = self.future_spatial_refiner(self.action_embedding.weight)
        expected = (len(GO2_H4_PRIMITIVE_VOCABULARY), self.config.feature_dim)
        if tuple(codes.shape) != expected:
            raise ValueError(f"centered action codes must have shape {expected}")
        return codes

    def _transition_step(
        self,
        visual: torch.Tensor,
        hidden: torch.Tensor,
        incoming_increment: torch.Tensor,
        action_indices: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Apply one shared factorized transition without a generic B skip."""

        if visual.ndim != 4 or hidden.shape != visual.shape:
            raise ValueError("transition visual and hidden shapes must be B,K,P,D")
        if incoming_increment.shape != visual.shape:
            raise ValueError("incoming increment must match the visual lattice")
        if action_indices.ndim != 1 or action_indices.dtype != torch.long:
            raise TypeError("one-step actions must be long with shape (B,)")
        batch, _atom_count, _tokens, dim = visual.shape
        if action_indices.shape[0] != batch:
            raise ValueError("one-step action batch size changed")
        if action_indices.device != visual.device:
            raise TypeError("one-step actions and visual carrier must share a device")
        modulation = self.history_cell(incoming_increment)
        belief_context = self.future_cell(visual, hidden, modulation)
        action_codes = self._centered_action_codes()
        selected_codes = action_codes.index_select(0, action_indices)
        if selected_codes.shape != (batch, dim):
            raise ValueError("selected action-code shape changed")
        interaction = (
            belief_context
            * (1.0 + torch.tanh(modulation))
            * selected_codes[:, None, None]
        )
        raw = incoming_increment + interaction
        projected_increment = self.prediction_projector(raw)
        next_visual = _renormalized_local_step(
            visual,
            projected_increment,
            epsilon=self.config.normalization_epsilon,
        )
        return next_visual, belief_context, projected_increment

    def _pack_belief(
        self,
        anchor: torch.Tensor,
        incoming_increment: torch.Tensor,
        hidden: torch.Tensor,
    ) -> torch.Tensor:
        if anchor.ndim != 3 or incoming_increment.shape != anchor.shape:
            raise ValueError("belief anchor and incoming increment must be B,P,D")
        if hidden.ndim != 4:
            raise ValueError("belief hidden state must be B,K,P,D")
        if (
            anchor.shape[0] != hidden.shape[0]
            or anchor.shape[1:] != hidden.shape[2:]
        ):
            raise ValueError("belief anchor and hidden lattices differ")
        return torch.cat(
            (anchor[:, None], incoming_increment[:, None], hidden),
            dim=1,
        )

    def _unpack_belief(
        self,
        belief_latents: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        expected = (
            self.config.trajectory_atom_count + 2,
            self.spatial_token_count,
            self.config.feature_dim,
        )
        if belief_latents.ndim != 4 or tuple(belief_latents.shape[1:]) != expected:
            raise ValueError(
                "belief_latents must have shape "
                f"(B,{expected[0]},{expected[1]},{expected[2]})"
            )
        if not torch.is_floating_point(belief_latents):
            raise TypeError("belief_latents must have a floating dtype")
        if belief_latents.device != self.action_embedding.weight.device:
            raise TypeError("belief_latents and model must share a device")
        if not bool(torch.isfinite(belief_latents).all()):
            raise FloatingPointError("belief_latents contains a nonfinite value")
        anchor = belief_latents[:, 0]
        incoming = belief_latents[:, 1]
        hidden = belief_latents[:, 2:]
        visual = anchor[:, None].expand(
            -1,
            self.config.trajectory_atom_count,
            -1,
            -1,
        )
        incoming_particles = incoming[:, None].expand_as(visual)
        return visual, incoming_particles, hidden

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
        hidden = self.initial_belief(normalized[:, 0])
        visual = normalized[:, 0, None].expand(
            -1,
            self.config.trajectory_atom_count,
            -1,
            -1,
        )
        incoming = torch.zeros_like(visual)
        priors: list[torch.Tensor] = []
        projected: list[torch.Tensor] = []
        innovations: list[torch.Tensor] = []
        for step in range(self.past_action_steps):
            prior, hidden, projected_increment = self._transition_step(
                visual,
                hidden,
                incoming,
                past_actions[:, step],
            )
            priors.append(prior)
            projected.append(projected_increment)
            innovations.append(prior - visual)
            # Insert the factual carrier only after its causal prior exists,
            # then derive the next edge's incoming increment from real RGB.
            factual = normalized[:, step + 1, None].expand_as(prior)
            incoming = factual - visual
            visual = factual

        final_incoming = normalized[:, 2] - normalized[:, 1]
        belief = self._pack_belief(normalized[:, 2], final_incoming, hidden)
        return (
            history,
            belief,
            torch.stack(priors, dim=2),
            torch.stack(projected, dim=2),
            torch.stack(innovations, dim=2),
        )

    def _rollout_future(
        self,
        belief_latents: torch.Tensor,
        future_actions: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        visual, incoming, hidden = self._unpack_belief(belief_latents)
        batch = int(belief_latents.shape[0])
        self._validate_actions(
            future_actions,
            batch=batch,
            steps=self.future_steps,
            name="future_actions",
        )
        states: list[torch.Tensor] = []
        projected: list[torch.Tensor] = []
        innovations: list[torch.Tensor] = []
        for step in range(self.future_steps):
            next_visual, hidden, projected_increment = self._transition_step(
                visual,
                hidden,
                incoming,
                future_actions[:, step],
            )
            realized_increment = next_visual - visual
            states.append(next_visual)
            projected.append(projected_increment)
            innovations.append(realized_increment)
            incoming = realized_increment
            visual = next_visual
        return (
            torch.stack(states, dim=2),
            torch.stack(projected, dim=2),
            torch.stack(innovations, dim=2),
            hidden,
        )


# Preserve the reviewed shared-runner constructor API.
JointRecurrentH4JEPAConfig = FactorizedConditionalIncrementTrajectoryH4JEPAConfig
JointRecurrentH4JEPA = FactorizedConditionalIncrementTrajectoryH4JEPA


__all__ = [
    "FactorizedConditionalIncrementTrajectoryH4JEPA",
    "FactorizedConditionalIncrementTrajectoryH4JEPAConfig",
    "FactualSharedTransitionTrajectoryH4JEPAOutput",
    "GO2_H4_PRIMITIVE_VOCABULARY",
    "JointRecurrentH4JEPA",
    "JointRecurrentH4JEPAConfig",
    "fixed_teacher_local_innovations",
    "realized_trajectory_innovations",
    "trajectory_energy_score",
]
