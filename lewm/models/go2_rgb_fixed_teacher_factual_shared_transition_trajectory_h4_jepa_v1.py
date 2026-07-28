"""Fixed-teacher factual shared-transition trajectory-H4 JEPA V1.

Exactly one spatial Transformer transition is reused on the two observed
history edges and the four open-loop future edges.  The observed-edge priors
are formed before their destination RGB is inserted; afterwards the factual
online visual carrier replaces the prior while the causal particle hidden
state is retained.  Four equal-mass particles then roll forward coherently.

The shared residual head is zero initialized, so all observed priors and all
future atoms begin at exact visual persistence.  Training uses only the
preregistered half factual-local, half cumulative-future proper score and
three-frame online-to-fixed-teacher alignment.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Mapping

import torch
import torch.nn as nn
import torch.nn.functional as F

from .encoders import ViTBlock
from .go2_recurrent_h4_joint_jepa import (
    GO2_H4_PRIMITIVE_VOCABULARY,
    JointRecurrentH4JEPAOutput,
)
from .go2_rgb_fixed_teacher_dense_spatiotemporal_cross_attention_h4_jepa_v1 import (
    JointRecurrentH4JEPA as _FixedTeacherH4JEPA,
    JointRecurrentH4JEPAConfig as _FixedTeacherH4JEPAConfig,
)
from .go2_rgb_fixed_teacher_local_innovation_trajectory_h4_jepa_v1 import (
    _renormalized_local_step,
    fixed_teacher_local_innovations,
    realized_trajectory_innovations,
)
from .go2_rgb_fixed_teacher_trajectory_distribution_h4_jepa_v1 import (
    TrajectoryDistributionH4JEPAOutput,
    trajectory_energy_score,
)


@dataclass(frozen=True)
class FactualSharedTransitionTrajectoryH4JEPAConfig(_FixedTeacherH4JEPAConfig):
    """Frozen K4 and equal-domain-weight scientific contract."""

    trajectory_atom_count: int = 4
    local_innovation_score_weight: float = 0.5
    cumulative_trajectory_score_weight: float = 0.5

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.trajectory_atom_count != 4:
            raise ValueError("trajectory_atom_count must remain exactly four")
        for name in (
            "local_innovation_score_weight",
            "cumulative_trajectory_score_weight",
        ):
            value = getattr(self, name)
            if not math.isfinite(value) or value != 0.5:
                raise ValueError(f"{name} must remain exactly 0.5")
        for name in ("teacher_alignment_weight", "teacher_delta_weight"):
            value = getattr(self, name)
            if not math.isfinite(value) or value != 1.0:
                raise ValueError(f"{name} must remain exactly 1.0")
        if self.action_vocabulary != GO2_H4_PRIMITIVE_VOCABULARY:
            raise ValueError("factual shared-transition action vocabulary changed")


@dataclass(frozen=True)
class FactualSharedTransitionTrajectoryH4JEPAOutput(
    TrajectoryDistributionH4JEPAOutput
):
    """Future K4 atoms plus the two scored pre-observation priors."""

    trajectory_innovations: torch.Tensor
    observed_prior_latents: torch.Tensor
    observed_prior_deltas: torch.Tensor
    all_six_trajectory_innovations: torch.Tensor
    final_hidden_particles: torch.Tensor


class _FactualParticleInitializer(nn.Module):
    """Construct K spatial hidden states from x0, mode, and patch identity."""

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
        self.visual_path = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Linear(feature_dim, feature_dim),
            nn.Tanh(),
        )
        self.mode_embedding = nn.Embedding(atom_count, feature_dim)
        self.spatial_embedding = nn.Embedding(spatial_tokens, feature_dim)
        nn.init.normal_(self.mode_embedding.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.spatial_embedding.weight, mean=0.0, std=0.02)

    def forward(self, initial_visual: torch.Tensor) -> torch.Tensor:
        if initial_visual.ndim != 3:
            raise ValueError("initial visual carrier must have shape B,P,D")
        batch, tokens, _dim = initial_visual.shape
        if tokens != self.spatial_tokens:
            raise ValueError("initial visual spatial token count changed")
        modes = self.mode_embedding.weight
        spatial = self.spatial_embedding.weight
        return (
            self.visual_path(initial_visual)[:, None]
            + modes[None, :, None]
            + spatial[None, None]
        ).reshape(batch, self.atom_count, tokens, -1)


class _SharedSpatialTransition(nn.Module):
    """The sole weight-shared action-conditioned spatial transition core."""

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
        self.action_path = nn.Linear(feature_dim, feature_dim)
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
        action_embedding: torch.Tensor,
    ) -> torch.Tensor:
        if visual.ndim != 4 or hidden.shape != visual.shape:
            raise ValueError("transition visual and hidden shapes must be B,K,P,D")
        batch, atom_count, tokens, dim = visual.shape
        if tokens != self.spatial_tokens:
            raise ValueError("transition spatial token count changed")
        if action_embedding.shape != (batch, dim):
            raise ValueError("transition action embedding shape changed")
        transition_input = (
            hidden
            + self.visual_path(visual)
            + self.action_path(action_embedding)[:, None, None]
        )
        transitioned = self.spatial_block(
            transition_input.reshape(batch * atom_count, tokens, dim)
        )
        return transitioned.reshape(batch, atom_count, tokens, dim)


class FactualSharedTransitionTrajectoryH4JEPA(_FixedTeacherH4JEPA):
    """One shared factual-and-future spatial transition with K4 rollouts."""

    def __init__(
        self,
        n320_encoder_state_dict: Mapping[str, torch.Tensor] | None = None,
        *,
        config: FactualSharedTransitionTrajectoryH4JEPAConfig | None = None,
    ) -> None:
        selected = config or FactualSharedTransitionTrajectoryH4JEPAConfig()
        if not isinstance(
            selected,
            FactualSharedTransitionTrajectoryH4JEPAConfig,
        ):
            raise TypeError(
                "config must be FactualSharedTransitionTrajectoryH4JEPAConfig"
            )
        super().__init__(
            n320_encoder_state_dict=n320_encoder_state_dict,
            config=selected,
        )

        dim = self.config.feature_dim
        self.initial_belief = _FactualParticleInitializer(
            spatial_tokens=self.spatial_token_count,
            feature_dim=dim,
            atom_count=self.config.trajectory_atom_count,
        )
        # Compatibility names preserve the reviewed optimizer inventory.  The
        # only learned history mechanism is the same future_cell used below.
        self.history_observation_norm = nn.Identity()
        self.history_cell = nn.Identity()
        self.history_spatial_refiner = nn.Identity()
        self.future_cell = _SharedSpatialTransition(
            spatial_tokens=self.spatial_token_count,
            feature_dim=dim,
            heads=self.config.cross_attention_heads,
            mlp_ratio=self.config.cross_attention_mlp_ratio,
            dropout=self.config.dropout,
        )
        self.future_spatial_refiner = nn.Identity()
        self.prediction_projector = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim),
        )
        final = self.prediction_projector[-1]
        nn.init.zeros_(final.weight)
        nn.init.zeros_(final.bias)

    def _transition_step(
        self,
        visual: torch.Tensor,
        hidden: torch.Tensor,
        action_indices: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Apply the sole transition and shared residual head once."""

        if action_indices.ndim != 1 or action_indices.dtype != torch.long:
            raise TypeError("one-step actions must be long with shape (B,)")
        if action_indices.shape[0] != visual.shape[0]:
            raise ValueError("one-step action batch size changed")
        next_hidden = self.future_cell(
            visual,
            hidden,
            self.action_embedding(action_indices),
        )
        raw_delta = self.prediction_projector(next_hidden)
        next_visual = _renormalized_local_step(
            visual,
            raw_delta,
            epsilon=self.config.normalization_epsilon,
        )
        return next_visual, next_hidden, raw_delta

    def _pack_belief(
        self,
        anchor: torch.Tensor,
        hidden: torch.Tensor,
    ) -> torch.Tensor:
        if anchor.ndim != 3 or hidden.ndim != 4:
            raise ValueError("belief anchor and hidden ranks changed")
        if anchor.shape[0] != hidden.shape[0] or anchor.shape[1:] != hidden.shape[2:]:
            raise ValueError("belief anchor and hidden lattices differ")
        return torch.cat((anchor[:, None], hidden), dim=1)

    def _unpack_belief(
        self,
        belief_latents: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        expected = (
            self.config.trajectory_atom_count + 1,
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
        hidden = belief_latents[:, 1:]
        visual = anchor[:, None].expand(-1, self.config.trajectory_atom_count, -1, -1)
        return visual, hidden

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
        priors: list[torch.Tensor] = []
        raw_deltas: list[torch.Tensor] = []
        innovations: list[torch.Tensor] = []
        for step in range(self.past_action_steps):
            prior, hidden, raw_delta = self._transition_step(
                visual,
                hidden,
                past_actions[:, step],
            )
            priors.append(prior)
            raw_deltas.append(raw_delta)
            innovations.append(prior - visual)
            # Insert the factual online carrier only after its prior is made.
            visual = normalized[:, step + 1, None].expand_as(prior)

        belief = self._pack_belief(normalized[:, 2], hidden)
        return (
            history,
            belief,
            torch.stack(priors, dim=2),
            torch.stack(raw_deltas, dim=2),
            torch.stack(innovations, dim=2),
        )

    def encode_history(
        self,
        history_rgb: torch.Tensor,
        past_actions: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        history, belief, _priors, _deltas, _innovations = (
            self._encode_factual_history(history_rgb, past_actions)
        )
        return history, belief

    def _rollout_future(
        self,
        belief_latents: torch.Tensor,
        future_actions: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        visual, hidden = self._unpack_belief(belief_latents)
        batch = int(belief_latents.shape[0])
        self._validate_actions(
            future_actions,
            batch=batch,
            steps=self.future_steps,
            name="future_actions",
        )
        states: list[torch.Tensor] = []
        raw_deltas: list[torch.Tensor] = []
        innovations: list[torch.Tensor] = []
        for step in range(self.future_steps):
            next_visual, hidden, raw_delta = self._transition_step(
                visual,
                hidden,
                future_actions[:, step],
            )
            states.append(next_visual)
            raw_deltas.append(raw_delta)
            innovations.append(next_visual - visual)
            visual = next_visual
        return (
            torch.stack(states, dim=2),
            torch.stack(raw_deltas, dim=2),
            torch.stack(innovations, dim=2),
            hidden,
        )

    def _predict_trajectory_with_deltas(
        self,
        belief_latents: torch.Tensor,
        future_actions: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        atoms, raw_deltas, _innovations, _hidden = self._rollout_future(
            belief_latents,
            future_actions,
        )
        return atoms, raw_deltas

    def predict_trajectory_atoms_from_belief(
        self,
        belief_latents: torch.Tensor,
        future_actions: torch.Tensor,
    ) -> torch.Tensor:
        atoms, _raw_deltas = self._predict_trajectory_with_deltas(
            belief_latents,
            future_actions,
        )
        return atoms

    def predict_from_belief(
        self,
        belief_latents: torch.Tensor,
        future_actions: torch.Tensor,
    ) -> torch.Tensor:
        atoms = self.predict_trajectory_atoms_from_belief(
            belief_latents,
            future_actions,
        )
        return F.normalize(
            atoms.mean(dim=1),
            p=2.0,
            dim=-1,
            eps=self.config.normalization_epsilon,
        )

    def forward(
        self,
        history_rgb: torch.Tensor,
        past_actions: torch.Tensor,
        future_actions: torch.Tensor,
        future_rgb: torch.Tensor | None = None,
    ) -> FactualSharedTransitionTrajectoryH4JEPAOutput:
        (
            history,
            belief,
            observed_priors,
            observed_deltas,
            observed_innovations,
        ) = self._encode_factual_history(history_rgb, past_actions)
        atoms, deltas, future_innovations, final_hidden = self._rollout_future(
            belief,
            future_actions,
        )
        all_six_innovations = torch.cat(
            (observed_innovations, future_innovations),
            dim=2,
        )
        centroid = F.normalize(
            atoms.mean(dim=1),
            p=2.0,
            dim=-1,
            eps=self.config.normalization_epsilon,
        )
        variance_loss = history.sum() * 0.0

        targets: torch.Tensor | None = None
        per_sample: torch.Tensor | None = None
        per_horizon: torch.Tensor | None = None
        prediction_loss: torch.Tensor | None = None
        joint_score: torch.Tensor | None = None
        if future_rgb is not None:
            if future_rgb.shape[0] != history_rgb.shape[0]:
                raise ValueError("history_rgb and future_rgb batch sizes differ")
            targets = self.encode_target(future_rgb)
            per_sample, joint_score, combined = trajectory_energy_score(
                atoms,
                targets,
            )
            per_horizon = per_sample.mean(dim=0)
            prediction_loss = combined.mean()

        return FactualSharedTransitionTrajectoryH4JEPAOutput(
            predicted_latents=centroid,
            target_latents=targets,
            history_latents=history,
            belief_latents=belief,
            per_sample_horizon_loss=per_sample,
            per_horizon_loss=per_horizon,
            prediction_loss=prediction_loss,
            variance_loss=variance_loss,
            total_loss=None,
            predicted_deltas=deltas.mean(dim=1),
            trajectory_latents=atoms,
            trajectory_deltas=deltas,
            joint_energy_score=joint_score,
            trajectory_innovations=future_innovations,
            observed_prior_latents=observed_priors,
            observed_prior_deltas=observed_deltas,
            all_six_trajectory_innovations=all_six_innovations,
            final_hidden_particles=final_hidden,
        )

    def training_auxiliary_losses(
        self,
        *,
        history_rgb: torch.Tensor,
        past_actions: torch.Tensor,
        future_actions: torch.Tensor,
        target_latents: torch.Tensor,
        output: JointRecurrentH4JEPAOutput,
    ) -> dict[str, torch.Tensor]:
        """Return exactly alignment and the two half-weight proper scores."""

        if not isinstance(
            output,
            FactualSharedTransitionTrajectoryH4JEPAOutput,
        ):
            raise TypeError(
                "output must be FactualSharedTransitionTrajectoryH4JEPAOutput"
            )
        batch = int(target_latents.shape[0])
        self._validate_actions(
            past_actions,
            batch=batch,
            steps=self.past_action_steps,
            name="past_actions",
        )
        self._validate_actions(
            future_actions,
            batch=batch,
            steps=self.future_steps,
            name="future_actions",
        )
        expected_future = (
            batch,
            self.config.trajectory_atom_count,
            self.future_steps,
            self.spatial_token_count,
            self.config.feature_dim,
        )
        expected_all = (
            batch,
            self.config.trajectory_atom_count,
            self.past_action_steps + self.future_steps,
            self.spatial_token_count,
            self.config.feature_dim,
        )
        if output.trajectory_latents.shape != expected_future:
            raise ValueError("target and cumulative trajectory shapes differ")
        if output.all_six_trajectory_innovations.shape != expected_all:
            raise ValueError("all-six factual innovation shape changed")

        online_history = F.normalize(
            output.history_latents,
            p=2.0,
            dim=-1,
            eps=self.config.normalization_epsilon,
        )
        teacher_history = self._encode_fixed_teacher_history(history_rgb)
        if online_history.shape != teacher_history.shape:
            raise ValueError("online and fixed-teacher history shapes differ")
        alignment = (
            (online_history - teacher_history).square().sum(dim=-1).mean()
        )

        target = target_latents.detach()
        observed_target_innovations = teacher_history[:, 1:] - teacher_history[:, :-1]
        future_target_innovations = fixed_teacher_local_innovations(
            teacher_history[:, 2],
            target,
        )
        all_target_innovations = torch.cat(
            (observed_target_innovations, future_target_innovations),
            dim=1,
        ).detach()
        _local_horizon, _local_joint, local = trajectory_energy_score(
            output.all_six_trajectory_innovations,
            all_target_innovations,
        )
        _future_horizon, _future_joint, cumulative = trajectory_energy_score(
            output.trajectory_latents,
            target,
        )
        return {
            "history_teacher_alignment": (
                self.config.teacher_alignment_weight * alignment
            ),
            "half_all_six_factual_local_innovation_energy_score": (
                self.config.local_innovation_score_weight * local.mean()
            ),
            "half_open_loop_future_cumulative_trajectory_energy_score": (
                self.config.cumulative_trajectory_score_weight
                * cumulative.mean()
            ),
        }


# Preserve the reviewed shared-runner constructor API.
JointRecurrentH4JEPAConfig = FactualSharedTransitionTrajectoryH4JEPAConfig
JointRecurrentH4JEPA = FactualSharedTransitionTrajectoryH4JEPA


__all__ = [
    "FactualSharedTransitionTrajectoryH4JEPA",
    "FactualSharedTransitionTrajectoryH4JEPAConfig",
    "FactualSharedTransitionTrajectoryH4JEPAOutput",
    "GO2_H4_PRIMITIVE_VOCABULARY",
    "JointRecurrentH4JEPA",
    "JointRecurrentH4JEPAConfig",
    "fixed_teacher_local_innovations",
    "realized_trajectory_innovations",
    "trajectory_energy_score",
]
