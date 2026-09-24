"""Fixed-teacher dual-domain trajectory-distribution H4 JEPA V1.

The architecture is exactly the local-innovation predecessor.  Its four
recursive trajectory atoms are scored in two linked domains: realized adjacent
innovations and integrated cumulative future states.  The same 50/50 mixed
proper score governs the prediction, cyclic-action, and ordered-history terms.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Mapping

import torch
import torch.nn.functional as F

from .go2_rgb_fixed_teacher_local_innovation_trajectory_h4_jepa_v1 import (
    GO2_H4_PRIMITIVE_VOCABULARY,
    LocalInnovationTrajectoryH4JEPA as _LocalInnovationTrajectoryH4JEPA,
    LocalInnovationTrajectoryH4JEPAConfig as _LocalInnovationTrajectoryH4JEPAConfig,
    LocalInnovationTrajectoryH4JEPAOutput,
    fixed_teacher_local_innovations,
    realized_trajectory_innovations,
    trajectory_energy_score,
)


@dataclass(frozen=True)
class DualDomainTrajectoryH4JEPAConfig(_LocalInnovationTrajectoryH4JEPAConfig):
    """Exact equal domain weights for the dual-domain falsification."""

    local_innovation_score_weight: float = 0.5
    cumulative_trajectory_score_weight: float = 0.5

    def __post_init__(self) -> None:
        super().__post_init__()
        for name in (
            "local_innovation_score_weight",
            "cumulative_trajectory_score_weight",
        ):
            value = getattr(self, name)
            if not math.isfinite(value) or value != 0.5:
                raise ValueError(f"{name} must remain exactly 0.5")


class DualDomainTrajectoryH4JEPA(_LocalInnovationTrajectoryH4JEPA):
    """One joint JEPA using local and integrated trajectory proper scores."""

    def __init__(
        self,
        n320_encoder_state_dict: Mapping[str, torch.Tensor] | None = None,
        *,
        config: DualDomainTrajectoryH4JEPAConfig | None = None,
    ) -> None:
        selected = config or DualDomainTrajectoryH4JEPAConfig()
        if not isinstance(selected, DualDomainTrajectoryH4JEPAConfig):
            raise TypeError("config must be DualDomainTrajectoryH4JEPAConfig")
        super().__init__(
            n320_encoder_state_dict=n320_encoder_state_dict,
            config=selected,
        )

    def _domain_scores(
        self,
        atoms: torch.Tensor,
        innovations: torch.Tensor,
        target: torch.Tensor,
        target_innovations: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return local, cumulative, and exact 50/50 per-sample scores."""

        _local_horizon, _local_joint, local = trajectory_energy_score(
            innovations,
            target_innovations,
        )
        _cumulative_horizon, _cumulative_joint, cumulative = (
            trajectory_energy_score(atoms, target)
        )
        mixed = (
            self.config.local_innovation_score_weight * local
            + self.config.cumulative_trajectory_score_weight * cumulative
        )
        return local, cumulative, mixed

    def training_auxiliary_losses(
        self,
        *,
        history_rgb: torch.Tensor,
        past_actions: torch.Tensor,
        future_actions: torch.Tensor,
        target_latents: torch.Tensor,
        output: LocalInnovationTrajectoryH4JEPAOutput,
    ) -> dict[str, torch.Tensor]:
        """Apply one coherent dual-domain score to fit and controls."""

        if not isinstance(output, LocalInnovationTrajectoryH4JEPAOutput):
            raise TypeError("output must be LocalInnovationTrajectoryH4JEPAOutput")
        expected_atoms = (
            target_latents.shape[0],
            self.config.trajectory_atom_count,
            *target_latents.shape[1:],
        )
        if output.trajectory_latents.shape != expected_atoms:
            raise ValueError("target and cumulative trajectory shapes differ")
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

        target = target_latents.detach()
        target_innovations = fixed_teacher_local_innovations(
            teacher_history[:, 2],
            target,
        )
        real_local, real_cumulative, real_mixed = self._domain_scores(
            output.trajectory_latents,
            output.trajectory_innovations,
            target,
            target_innovations,
        )

        zero_innovations = torch.zeros_like(output.trajectory_innovations)
        teacher_anchor = teacher_history[:, 2]
        persistence_atoms = teacher_anchor[:, None, None].expand_as(
            output.trajectory_latents
        )
        _zero_local, _zero_cumulative, persistence_mixed = self._domain_scores(
            persistence_atoms,
            zero_innovations,
            target,
            target_innovations,
        )
        denominator = persistence_mixed.detach().clamp_min(
            self.config.normalization_epsilon
        )
        real_normalized = real_mixed / denominator

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
        _wrong_local, _wrong_cumulative, wrong_mixed = self._domain_scores(
            wrong_atoms,
            wrong_innovations,
            target,
            target_innovations,
        )
        action_ranking = F.relu(
            self.config.cyclic_wrong_action_margin
            + real_normalized
            - wrong_mixed / denominator
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
        reversed_atoms, _reversed_raw = self._predict_trajectory_with_deltas(
            reversed_belief,
            future_actions,
        )
        reset_atoms, _reset_raw = self._predict_trajectory_with_deltas(
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
        _reverse_local, _reverse_cumulative, reversed_mixed = self._domain_scores(
            reversed_atoms,
            reversed_innovations,
            target,
            target_innovations,
        )
        _reset_local, _reset_cumulative, reset_mixed = self._domain_scores(
            reset_atoms,
            reset_innovations,
            target,
            target_innovations,
        )
        history_control_normalized = torch.minimum(
            reversed_mixed,
            reset_mixed,
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
            "half_future_teacher_local_innovation_energy_score": (
                self.config.local_innovation_score_weight * real_local.mean()
            ),
            "half_future_teacher_cumulative_trajectory_energy_score": (
                self.config.cumulative_trajectory_score_weight
                * real_cumulative.mean()
            ),
            "dual_domain_cyclic_wrong_action_score_ranking": (
                self.config.cyclic_wrong_action_ranking_weight * action_ranking
            ),
            "dual_domain_history_counterfactual_score_ranking": (
                self.config.history_ranking_weight * history_ranking
            ),
        }


JointRecurrentH4JEPAConfig = DualDomainTrajectoryH4JEPAConfig
JointRecurrentH4JEPA = DualDomainTrajectoryH4JEPA


__all__ = [
    "DualDomainTrajectoryH4JEPA",
    "DualDomainTrajectoryH4JEPAConfig",
    "GO2_H4_PRIMITIVE_VOCABULARY",
    "JointRecurrentH4JEPA",
    "JointRecurrentH4JEPAConfig",
    "LocalInnovationTrajectoryH4JEPAOutput",
]
