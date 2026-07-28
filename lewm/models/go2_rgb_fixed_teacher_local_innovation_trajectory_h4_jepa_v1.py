"""Fixed-teacher local-innovation trajectory-distribution H4 JEPA V1.

This successor keeps the reviewed four-atom dense trajectory backbone and its
permanently fixed N320 target.  The shared zero-initialized head now emits one
local transition increment per atom and horizon.  Each atom is integrated
recursively from normalized online ``e2`` and renormalized after every step.

Training scores the realized local transition innovations against adjacent
fixed-teacher transitions with the same proper 50/50 joint-and-marginal energy
score used by the parent trajectory model.  Same-RGB teacher alignment and
normalized action/history ranking terms keep the online encoder, dense history,
and action-conditioned predictor in one joint JEPA backward pass.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Mapping

import torch
import torch.nn.functional as F

from .go2_recurrent_h4_joint_jepa import (
    GO2_H4_PRIMITIVE_VOCABULARY,
    JointRecurrentH4JEPAOutput,
)
from .go2_rgb_fixed_teacher_trajectory_distribution_h4_jepa_v1 import (
    JointRecurrentH4JEPA as _TrajectoryH4JEPA,
    JointRecurrentH4JEPAConfig as _TrajectoryH4JEPAConfig,
    TrajectoryDistributionH4JEPAOutput,
    trajectory_energy_score,
)


@dataclass(frozen=True)
class LocalInnovationTrajectoryH4JEPAConfig(_TrajectoryH4JEPAConfig):
    """Exact weights and margins for the local-innovation falsification."""

    cyclic_wrong_action_ranking_weight: float = 1.0
    history_ranking_weight: float = 1.0
    cyclic_wrong_action_margin: float = 0.05
    history_margin: float = 0.03

    def __post_init__(self) -> None:
        super().__post_init__()
        exact_weights = {
            "teacher_alignment_weight": self.teacher_alignment_weight,
            "teacher_delta_weight": self.teacher_delta_weight,
            "cyclic_wrong_action_ranking_weight": (
                self.cyclic_wrong_action_ranking_weight
            ),
            "history_ranking_weight": self.history_ranking_weight,
        }
        for name, value in exact_weights.items():
            if not math.isfinite(value) or value != 1.0:
                raise ValueError(f"{name} must remain exactly 1.0")
        exact_margins = {
            "cyclic_wrong_action_margin": (
                self.cyclic_wrong_action_margin,
                0.05,
            ),
            "history_margin": (self.history_margin, 0.03),
        }
        for name, (value, expected) in exact_margins.items():
            if not math.isfinite(value) or value != expected:
                raise ValueError(f"{name} must remain exactly {expected}")
        if self.action_vocabulary != GO2_H4_PRIMITIVE_VOCABULARY:
            raise ValueError("local-innovation action vocabulary changed")


@dataclass(frozen=True)
class LocalInnovationTrajectoryH4JEPAOutput(TrajectoryDistributionH4JEPAOutput):
    """Trajectory output plus realized adjacent normalized-state changes."""

    trajectory_innovations: torch.Tensor


def fixed_teacher_local_innovations(
    current: torch.Tensor,
    future: torch.Tensor,
) -> torch.Tensor:
    """Return exact ``e3-e2, e4-e3, e5-e4, e6-e5`` target changes."""

    if current.ndim != 3 or future.ndim != 4:
        raise ValueError("local innovation target expects B,P,D and B,H,P,D")
    if current.shape[0] != future.shape[0] or current.shape[1:] != future.shape[2:]:
        raise ValueError("current and future target lattices differ")
    path = torch.cat((current[:, None], future), dim=1)
    return path[:, 1:] - path[:, :-1]


def realized_trajectory_innovations(
    anchor: torch.Tensor,
    atoms: torch.Tensor,
) -> torch.Tensor:
    """Return successive changes along each realized trajectory atom."""

    if anchor.ndim != 3 or atoms.ndim != 5:
        raise ValueError("trajectory innovations expect B,P,D and B,K,H,P,D")
    if anchor.shape[0] != atoms.shape[0] or anchor.shape[1:] != atoms.shape[3:]:
        raise ValueError("trajectory anchor and atom lattices differ")
    repeated_anchor = anchor[:, None, None].expand(
        anchor.shape[0], atoms.shape[1], 1, anchor.shape[1], anchor.shape[2]
    )
    path = torch.cat((repeated_anchor, atoms), dim=2)
    return path[:, :, 1:] - path[:, :, :-1]


def _renormalized_local_step(
    previous: torch.Tensor,
    increment: torch.Tensor,
    *,
    epsilon: float,
) -> torch.Tensor:
    """Apply one increment while preserving the normalized anchor radius.

    Dividing the candidate norm by the previous norm makes the zero-increment
    path multiply by exact one.  Update zero therefore remains bitwise
    persistence while gradients through a zero-initialized head stay open.
    """

    if previous.shape != increment.shape:
        raise ValueError("local trajectory step shapes differ")
    candidate = previous + increment
    previous_norm = previous.norm(p=2.0, dim=-1, keepdim=True).clamp_min(epsilon)
    candidate_norm = candidate.norm(p=2.0, dim=-1, keepdim=True).clamp_min(epsilon)
    return candidate * (previous_norm / candidate_norm)


class LocalInnovationTrajectoryH4JEPA(_TrajectoryH4JEPA):
    """Four-atom fixed-teacher JEPA with recursive local innovations."""

    def __init__(
        self,
        n320_encoder_state_dict: Mapping[str, torch.Tensor] | None = None,
        *,
        config: LocalInnovationTrajectoryH4JEPAConfig | None = None,
    ) -> None:
        selected = config or LocalInnovationTrajectoryH4JEPAConfig()
        if not isinstance(selected, LocalInnovationTrajectoryH4JEPAConfig):
            raise TypeError("config must be LocalInnovationTrajectoryH4JEPAConfig")
        super().__init__(
            n320_encoder_state_dict=n320_encoder_state_dict,
            config=selected,
        )

    def _predict_trajectory_with_deltas(
        self,
        belief_latents: torch.Tensor,
        future_actions: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Decode local increments and recursively integrate four horizons."""

        tokens = self.spatial_token_count
        dim = self.config.feature_dim
        memory_tokens = self.history_steps * tokens + self.past_action_steps
        expected = (tokens + memory_tokens, dim)
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

        anchor = belief_latents[:, :tokens]
        memory = belief_latents[:, tokens:]
        normalized_anchor = F.normalize(
            anchor,
            p=2.0,
            dim=-1,
            eps=self.config.normalization_epsilon,
        )
        hidden = self.future_cell(
            normalized_anchor,
            memory,
            self.action_embedding(future_actions),
            self.initial_belief.spatial_embedding.weight,
        )
        raw_increments = self.prediction_projector(hidden)
        state = normalized_anchor[:, None].expand(
            batch,
            self.config.trajectory_atom_count,
            tokens,
            dim,
        )
        states: list[torch.Tensor] = []
        for horizon in range(self.future_steps):
            state = _renormalized_local_step(
                state,
                raw_increments[:, :, horizon],
                epsilon=self.config.normalization_epsilon,
            )
            states.append(state)
        return torch.stack(states, dim=2), raw_increments

    def _belief_from_encoded_history(
        self,
        history_latents: torch.Tensor,
        past_actions: torch.Tensor,
    ) -> torch.Tensor:
        """Pack dense history controls without another online RGB encoding."""

        expected = (
            self.history_steps,
            self.spatial_token_count,
            self.config.feature_dim,
        )
        if history_latents.ndim != 4 or tuple(history_latents.shape[1:]) != expected:
            raise ValueError(
                "history_latents must have shape "
                f"(B,{expected[0]},{expected[1]},{expected[2]})"
            )
        batch = int(history_latents.shape[0])
        if not history_latents.is_floating_point():
            raise TypeError("history_latents must be floating point")
        if history_latents.device != self.action_embedding.weight.device:
            raise TypeError("history_latents and model must share a device")
        if not bool(torch.isfinite(history_latents).all()):
            raise FloatingPointError("history_latents contains a nonfinite value")
        self._validate_actions(
            past_actions,
            batch=batch,
            steps=self.past_action_steps,
            name="past_actions",
        )
        normalized = F.normalize(
            history_latents,
            p=2.0,
            dim=-1,
            eps=self.config.normalization_epsilon,
        )
        memory = self.initial_belief(
            normalized,
            self.action_embedding(past_actions),
        )
        return torch.cat((history_latents[:, 2], memory), dim=1)

    def forward(
        self,
        history_rgb: torch.Tensor,
        past_actions: torch.Tensor,
        future_actions: torch.Tensor,
        future_rgb: torch.Tensor | None = None,
    ) -> LocalInnovationTrajectoryH4JEPAOutput:
        parent = super().forward(
            history_rgb,
            past_actions,
            future_actions,
            future_rgb,
        )
        normalized_e2 = F.normalize(
            parent.history_latents[:, 2],
            p=2.0,
            dim=-1,
            eps=self.config.normalization_epsilon,
        )
        innovations = realized_trajectory_innovations(
            normalized_e2,
            parent.trajectory_latents,
        )
        return LocalInnovationTrajectoryH4JEPAOutput(
            predicted_latents=parent.predicted_latents,
            target_latents=parent.target_latents,
            history_latents=parent.history_latents,
            belief_latents=parent.belief_latents,
            per_sample_horizon_loss=parent.per_sample_horizon_loss,
            per_horizon_loss=parent.per_horizon_loss,
            prediction_loss=parent.prediction_loss,
            variance_loss=parent.variance_loss,
            total_loss=parent.total_loss,
            predicted_deltas=parent.predicted_deltas,
            trajectory_latents=parent.trajectory_latents,
            trajectory_deltas=parent.trajectory_deltas,
            joint_energy_score=parent.joint_energy_score,
            trajectory_innovations=innovations,
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
        """Score local innovations and require action/history necessity."""

        if not isinstance(output, LocalInnovationTrajectoryH4JEPAOutput):
            raise TypeError("output must be LocalInnovationTrajectoryH4JEPAOutput")
        expected_atoms = (
            target_latents.shape[0],
            self.config.trajectory_atom_count,
            *target_latents.shape[1:],
        )
        if output.trajectory_innovations.shape != expected_atoms:
            raise ValueError("target and local-innovation trajectory shapes differ")
        self._validate_actions(
            future_actions,
            batch=int(target_latents.shape[0]),
            steps=self.future_steps,
            name="future_actions",
        )

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
        target_innovations = fixed_teacher_local_innovations(
            teacher_history[:, 2],
            target_latents.detach(),
        )
        _real_horizon, _real_joint, real_combined = trajectory_energy_score(
            output.trajectory_innovations,
            target_innovations,
        )

        zero_innovations = torch.zeros_like(output.trajectory_innovations)
        _zero_horizon, _zero_joint, persistence_scale = trajectory_energy_score(
            zero_innovations,
            target_innovations,
        )
        persistence_scale = persistence_scale.detach()
        denominator = persistence_scale.clamp_min(self.config.normalization_epsilon)
        real_normalized = real_combined / denominator

        wrong_actions = (future_actions + 1) % self.config.action_count
        wrong_atoms, _wrong_raw = self._predict_trajectory_with_deltas(
            output.belief_latents,
            wrong_actions,
        )
        normalized_e2 = F.normalize(
            output.history_latents[:, 2],
            p=2.0,
            dim=-1,
            eps=self.config.normalization_epsilon,
        )
        wrong_innovations = realized_trajectory_innovations(
            normalized_e2,
            wrong_atoms,
        )
        _wrong_horizon, _wrong_joint, wrong_combined = trajectory_energy_score(
            wrong_innovations,
            target_innovations,
        )
        wrong_normalized = wrong_combined / denominator
        action_ranking = F.relu(
            self.config.cyclic_wrong_action_margin
            + real_normalized
            - wrong_normalized
        ).mean()

        reversed_history = output.history_latents[:, [1, 0, 2]]
        reversed_past = past_actions.flip(dims=(1,))
        reset_history = output.history_latents[:, 2:3].expand(
            -1,
            self.history_steps,
            -1,
            -1,
        )
        hold_index = self.action_vocabulary.index("hold")
        reset_past = torch.full_like(past_actions, hold_index)
        reversed_belief = self._belief_from_encoded_history(
            reversed_history,
            reversed_past,
        )
        reset_belief = self._belief_from_encoded_history(
            reset_history,
            reset_past,
        )
        reversed_atoms, _ = self._predict_trajectory_with_deltas(
            reversed_belief,
            future_actions,
        )
        reset_atoms, _ = self._predict_trajectory_with_deltas(
            reset_belief,
            future_actions,
        )
        reversed_innovations = realized_trajectory_innovations(
            normalized_e2,
            reversed_atoms,
        )
        reset_innovations = realized_trajectory_innovations(
            normalized_e2,
            reset_atoms,
        )
        _rh, _rj, reversed_combined = trajectory_energy_score(
            reversed_innovations,
            target_innovations,
        )
        _sh, _sj, reset_combined = trajectory_energy_score(
            reset_innovations,
            target_innovations,
        )
        history_control_normalized = torch.minimum(
            reversed_combined,
            reset_combined,
        ) / denominator
        history_ranking = F.relu(
            self.config.history_margin
            + real_normalized
            - history_control_normalized
        ).mean()

        return {
            "history_teacher_alignment": (
                self.config.teacher_alignment_weight * alignment
            ),
            "future_teacher_local_innovation_energy_score": (
                self.config.teacher_delta_weight * real_combined.mean()
            ),
            "cyclic_wrong_action_score_ranking": (
                self.config.cyclic_wrong_action_ranking_weight * action_ranking
            ),
            "history_counterfactual_score_ranking": (
                self.config.history_ranking_weight * history_ranking
            ),
        }


# Preserve the shared runner's conventional constructor names while also
# exposing mechanism-specific names for focused reviews and tests.
JointRecurrentH4JEPAConfig = LocalInnovationTrajectoryH4JEPAConfig
JointRecurrentH4JEPA = LocalInnovationTrajectoryH4JEPA


__all__ = [
    "GO2_H4_PRIMITIVE_VOCABULARY",
    "JointRecurrentH4JEPA",
    "JointRecurrentH4JEPAConfig",
    "LocalInnovationTrajectoryH4JEPA",
    "LocalInnovationTrajectoryH4JEPAConfig",
    "LocalInnovationTrajectoryH4JEPAOutput",
    "fixed_teacher_local_innovations",
    "realized_trajectory_innovations",
    "trajectory_energy_score",
]
