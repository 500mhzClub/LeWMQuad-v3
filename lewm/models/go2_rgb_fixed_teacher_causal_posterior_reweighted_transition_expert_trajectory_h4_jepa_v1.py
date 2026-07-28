"""Fixed-teacher causal posterior-reweighted transition-expert H4 JEPA V1.

Four mode-conditioned transition experts share the inherited action-free
context, centered categorical action tower, and zero-initialized output head.
Observed priors update only four expert probabilities; factual content is then
assimilated.  The probabilities are frozen during the four-step future and
affect only distribution mass, never expert content or increments.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import torch
import torch.nn as nn
import torch.nn.functional as F

from .go2_recurrent_h4_joint_jepa import (
    GO2_H4_PRIMITIVE_VOCABULARY,
    JointRecurrentH4JEPAOutput,
)
from .go2_rgb_fixed_teacher_factorized_conditional_increment_trajectory_h4_jepa_v1 import (
    FactorizedConditionalIncrementTrajectoryH4JEPA,
    FactorizedConditionalIncrementTrajectoryH4JEPAConfig,
)
from .go2_rgb_fixed_teacher_factual_shared_transition_trajectory_h4_jepa_v1 import (
    FactualSharedTransitionTrajectoryH4JEPAOutput,
    fixed_teacher_local_innovations,
    realized_trajectory_innovations,
    trajectory_energy_score,
)
from .go2_rgb_fixed_teacher_local_innovation_trajectory_h4_jepa_v1 import (
    _renormalized_local_step,
)
from .go2_rgb_fixed_teacher_trajectory_distribution_h4_jepa_v1 import (
    _lattice_distance,
)


def _validate_probabilities(
    probabilities: torch.Tensor,
    *,
    batch: int,
    atom_count: int,
) -> None:
    if probabilities.ndim != 2 or tuple(probabilities.shape) != (
        batch,
        atom_count,
    ):
        raise ValueError("expert probabilities must have shape B,K")
    if not torch.is_floating_point(probabilities):
        raise TypeError("expert probabilities must have a floating dtype")
    if not bool(torch.isfinite(probabilities).all()):
        raise FloatingPointError("expert probabilities contain a nonfinite value")
    if not bool((probabilities > 0.0).all()):
        raise ValueError("expert probabilities must be strictly positive")
    totals = probabilities.sum(dim=1)
    if not torch.allclose(
        totals,
        torch.ones_like(totals),
        rtol=0.0,
        atol=1e-6,
    ):
        raise ValueError("expert probabilities must sum to one")


def _validate_distribution_weights(
    weights: torch.Tensor,
    *,
    batch: int,
    atom_count: int,
) -> None:
    """Validate readout weights, permitting analytic zero-mass controls."""

    if weights.ndim != 2 or tuple(weights.shape) != (batch, atom_count):
        raise ValueError("expert weights must have shape B,K")
    if not torch.is_floating_point(weights):
        raise TypeError("expert weights must have a floating dtype")
    if not bool(torch.isfinite(weights).all()):
        raise FloatingPointError("expert weights contain a nonfinite value")
    if not bool((weights >= 0.0).all()):
        raise ValueError("expert weights must be nonnegative")
    totals = weights.sum(dim=1)
    if not torch.allclose(
        totals,
        torch.ones_like(totals),
        rtol=0.0,
        atol=1e-6,
    ):
        raise ValueError("expert weights must sum to one")


def weighted_trajectory_energy_score(
    atoms: torch.Tensor,
    target: torch.Tensor,
    weights: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return posterior-weighted marginal, joint, and combined energy scores."""

    if atoms.ndim != 5 or target.ndim != 4:
        raise ValueError("weighted energy expects B,K,H,P,D and B,H,P,D")
    if atoms.shape[0] != target.shape[0] or atoms.shape[2:] != target.shape[1:]:
        raise ValueError("trajectory atoms and target shapes differ")
    batch, atom_count, horizons, tokens, dim = atoms.shape
    if atom_count != 4:
        raise ValueError("weighted energy score requires exactly four experts")
    _validate_distribution_weights(weights, batch=batch, atom_count=atom_count)
    if weights.device != atoms.device or weights.dtype != atoms.dtype:
        raise TypeError("atoms and expert probabilities must share device and dtype")
    if target.device != atoms.device or target.dtype != atoms.dtype:
        raise TypeError("atoms and target must share device and dtype")

    fit_distance = _lattice_distance(atoms, target[:, None])
    pair_distance = _lattice_distance(
        atoms[:, :, None],
        atoms[:, None, :],
    )
    fit_horizon = (fit_distance * weights[:, :, None]).sum(dim=1)
    pair_weights = weights[:, :, None] * weights[:, None, :]
    pair_horizon = (
        pair_distance * pair_weights[:, :, :, None]
    ).sum(dim=(1, 2))
    horizon_score = fit_horizon - 0.5 * pair_horizon

    flat_atoms = atoms.reshape(batch, atom_count, horizons * tokens, dim)
    flat_target = target.reshape(batch, horizons * tokens, dim)
    joint_fit_distance = _lattice_distance(flat_atoms, flat_target[:, None])
    joint_pair_distance = _lattice_distance(
        flat_atoms[:, :, None],
        flat_atoms[:, None, :],
    )
    joint_fit = (joint_fit_distance * weights).sum(dim=1)
    joint_pair = (joint_pair_distance * pair_weights).sum(dim=(1, 2))
    joint_score = joint_fit - 0.5 * joint_pair
    combined = 0.5 * joint_score + 0.5 * horizon_score.mean(dim=1)
    return horizon_score, joint_score, combined


def weighted_spherical_centroid(
    atoms: torch.Tensor,
    weights: torch.Tensor,
    *,
    epsilon: float = 1e-6,
) -> torch.Tensor:
    """Return the normalized posterior mean of a B,K,H,P,D distribution."""

    if atoms.ndim != 5:
        raise ValueError("weighted centroid expects B,K,H,P,D atoms")
    _validate_distribution_weights(
        weights,
        batch=int(atoms.shape[0]),
        atom_count=int(atoms.shape[1]),
    )
    if weights.device != atoms.device or weights.dtype != atoms.dtype:
        raise TypeError("atoms and expert probabilities must share device and dtype")
    mean = (atoms * weights[:, :, None, None, None]).sum(dim=1)
    return F.normalize(mean, p=2.0, dim=-1, eps=epsilon)


def weighted_pairwise_spread(
    atoms: torch.Tensor,
    weights: torch.Tensor,
) -> torch.Tensor:
    """Return posterior-weighted ordered-pair spread for every horizon."""

    if atoms.ndim != 5:
        raise ValueError("weighted spread expects B,K,H,P,D atoms")
    _validate_distribution_weights(
        weights,
        batch=int(atoms.shape[0]),
        atom_count=int(atoms.shape[1]),
    )
    if weights.device != atoms.device or weights.dtype != atoms.dtype:
        raise TypeError("atoms and expert probabilities must share device and dtype")
    pair_distance = _lattice_distance(
        atoms[:, :, None],
        atoms[:, None, :],
    )
    pair_weights = weights[:, :, None] * weights[:, None, :]
    return (pair_distance * pair_weights[:, :, :, None]).sum(dim=(1, 2))


@dataclass(frozen=True)
class CausalPosteriorReweightedTransitionExpertTrajectoryH4JEPAConfig(
    FactorizedConditionalIncrementTrajectoryH4JEPAConfig
):
    """Frozen K4 posterior-expert contract with exact evidence epsilon."""

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.normalization_epsilon != 1e-6:
            raise ValueError("normalization_epsilon must remain exactly 1e-6")


@dataclass(frozen=True)
class CausalPosteriorReweightedTransitionExpertTrajectoryH4JEPAOutput(
    FactualSharedTransitionTrajectoryH4JEPAOutput
):
    """K4 future distribution and its two-update causal posterior evidence."""

    posterior_probabilities: torch.Tensor
    posterior_history: torch.Tensor
    evidence_squared_errors: torch.Tensor
    evidence_likelihoods: torch.Tensor


class _PosteriorExpertInitializer(nn.Module):
    """Own centered expert/spatial context and initialize q with uniform mass."""

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
        probabilities = initial_content.new_full(
            (batch, self.atom_count),
            1.0 / float(self.atom_count),
        )
        return content, probabilities


class CausalPosteriorReweightedTransitionExpertTrajectoryH4JEPA(
    FactorizedConditionalIncrementTrajectoryH4JEPA
):
    """K4 predict-update-assimilate experts with probability-only history."""

    def __init__(
        self,
        n320_encoder_state_dict: Mapping[str, torch.Tensor] | None = None,
        *,
        config: (
            CausalPosteriorReweightedTransitionExpertTrajectoryH4JEPAConfig
            | None
        ) = None,
    ) -> None:
        selected = (
            config
            or CausalPosteriorReweightedTransitionExpertTrajectoryH4JEPAConfig()
        )
        if not isinstance(
            selected,
            CausalPosteriorReweightedTransitionExpertTrajectoryH4JEPAConfig,
        ):
            raise TypeError(
                "config must be CausalPosteriorReweightedTransitionExpert"
                "TrajectoryH4JEPAConfig"
            )
        super().__init__(
            n320_encoder_state_dict=n320_encoder_state_dict,
            config=selected,
        )
        self.initial_belief = _PosteriorExpertInitializer(
            spatial_tokens=self.spatial_token_count,
            feature_dim=self.config.feature_dim,
            atom_count=self.config.trajectory_atom_count,
        )
        # Compatibility slots preserve the reviewed disjoint optimizer groups;
        # probability updating itself has no learned module or parameter.
        self.history_observation_norm = nn.Identity()
        self.history_cell = nn.Identity()
        self.history_spatial_refiner = nn.Identity()

    def _mode_context(self, reference: torch.Tensor) -> torch.Tensor:
        return self.initial_belief.centered_context(
            batch=int(reference.shape[0]),
            device=reference.device,
            dtype=reference.dtype,
        )

    def _transition_step(
        self,
        content: torch.Tensor,
        action_indices: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        expected = (
            self.config.trajectory_atom_count,
            self.spatial_token_count,
            self.config.feature_dim,
        )
        if content.ndim != 4 or tuple(content.shape[1:]) != expected:
            raise ValueError("expert content must have shape B,K,P,D")
        batch = int(content.shape[0])
        if action_indices.ndim != 1 or action_indices.dtype != torch.long:
            raise TypeError("one-step actions must be long with shape (B,)")
        if tuple(action_indices.shape) != (batch,):
            raise ValueError("one-step action batch size changed")
        if action_indices.device != content.device:
            raise TypeError("expert content and action must share a device")

        context = self.future_cell(
            content,
            self._mode_context(content),
            torch.zeros_like(content),
        )
        action_codes = self._centered_action_codes()
        selected_codes = action_codes.index_select(0, action_indices)
        interaction = context * selected_codes[:, None, None, :]
        projected_increment = self.prediction_projector(interaction)
        next_content = _renormalized_local_step(
            content,
            projected_increment,
            epsilon=self.config.normalization_epsilon,
        )
        return next_content, projected_increment, next_content - content

    def _evidence_update(
        self,
        probabilities: torch.Tensor,
        prior_content: torch.Tensor,
        observation: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        expected_tail = (
            self.config.trajectory_atom_count,
            self.spatial_token_count,
            self.config.feature_dim,
        )
        if prior_content.ndim != 4 or tuple(prior_content.shape[1:]) != expected_tail:
            raise ValueError("evidence prior must have shape B,K,P,D")
        batch = int(prior_content.shape[0])
        _validate_probabilities(
            probabilities,
            batch=batch,
            atom_count=self.config.trajectory_atom_count,
        )
        if observation.ndim != 3 or tuple(observation.shape) != (
            batch,
            self.spatial_token_count,
            self.config.feature_dim,
        ):
            raise ValueError("evidence observation must have shape B,P,D")
        if probabilities.device != prior_content.device:
            raise TypeError("prior and probabilities must share a device")
        if observation.device != prior_content.device:
            raise TypeError("prior and evidence observation must share a device")
        if probabilities.dtype != prior_content.dtype or observation.dtype != prior_content.dtype:
            raise TypeError("prior, probabilities, and observation must share dtype")

        destination = observation[:, None].expand_as(prior_content)
        squared_errors = (
            (prior_content - destination).square().sum(dim=-1).mean(dim=-1)
        )
        mean_error = squared_errors.mean(dim=1, keepdim=True)
        likelihoods = torch.exp(
            -squared_errors
            / (mean_error + self.config.normalization_epsilon)
        )
        unnormalized = probabilities * likelihoods
        updated = unnormalized / unnormalized.sum(dim=1, keepdim=True)
        _validate_probabilities(
            updated,
            batch=batch,
            atom_count=self.config.trajectory_atom_count,
        )
        return updated, squared_errors, likelihoods

    def _assimilate_observation(
        self,
        prior_content: torch.Tensor,
        observation: torch.Tensor,
    ) -> torch.Tensor:
        """Replace every expert content lattice with the factual online q."""

        if observation.ndim != 3:
            raise ValueError("assimilated observation must have shape B,P,D")
        if (
            prior_content.ndim != 4
            or prior_content.shape[0] != observation.shape[0]
            or tuple(prior_content.shape[2:]) != tuple(observation.shape[1:])
        ):
            raise ValueError("prior and assimilated observation lattices differ")
        if observation.device != prior_content.device:
            raise TypeError("prior and assimilated observation must share device")
        if observation.dtype != prior_content.dtype:
            raise TypeError("prior and assimilated observation must share dtype")
        return observation[:, None].expand_as(prior_content)

    def _pack_belief(
        self,
        content: torch.Tensor,
        probabilities: torch.Tensor,
    ) -> torch.Tensor:
        expected = (
            self.config.trajectory_atom_count,
            self.spatial_token_count,
            self.config.feature_dim,
        )
        if content.ndim != 4 or tuple(content.shape[1:]) != expected:
            raise ValueError("belief content must have shape B,K,P,D")
        batch = int(content.shape[0])
        _validate_probabilities(
            probabilities,
            batch=batch,
            atom_count=self.config.trajectory_atom_count,
        )
        if probabilities.device != content.device or probabilities.dtype != content.dtype:
            raise TypeError("belief content and probabilities must share device/dtype")
        carrier_size = self.spatial_token_count * self.config.feature_dim
        padded = F.pad(
            probabilities,
            (0, carrier_size - self.config.trajectory_atom_count),
        ).reshape(batch, 1, self.spatial_token_count, self.config.feature_dim)
        return torch.cat((content, padded), dim=1)

    def _unpack_belief(
        self,
        belief_latents: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        atom_count = self.config.trajectory_atom_count
        expected = (
            atom_count + 1,
            self.spatial_token_count,
            self.config.feature_dim,
        )
        if belief_latents.ndim != 4 or tuple(belief_latents.shape[1:]) != expected:
            raise ValueError(
                "belief_latents must contain four q lattices and one probability carrier"
            )
        if not torch.is_floating_point(belief_latents):
            raise TypeError("belief_latents must have a floating dtype")
        if belief_latents.device != self.action_embedding.weight.device:
            raise TypeError("belief_latents and model must share a device")
        if not bool(torch.isfinite(belief_latents).all()):
            raise FloatingPointError("belief_latents contains a nonfinite value")
        content = belief_latents[:, :atom_count]
        flattened = belief_latents[:, atom_count].reshape(
            belief_latents.shape[0],
            -1,
        )
        probabilities = flattened[:, :atom_count]
        padding = flattened[:, atom_count:]
        if padding.numel() and int(torch.count_nonzero(padding).item()) != 0:
            raise ValueError("serialized probability padding is nonzero")
        _validate_probabilities(
            probabilities,
            batch=int(belief_latents.shape[0]),
            atom_count=atom_count,
        )
        return content, probabilities

    def posterior_probabilities_from_belief(
        self,
        belief_latents: torch.Tensor,
    ) -> torch.Tensor:
        """Return the four serialized causal probabilities."""

        _content, probabilities = self._unpack_belief(belief_latents)
        return probabilities

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
        content, probabilities = self.initial_belief(normalized[:, 0])
        posterior_history = [probabilities]
        priors: list[torch.Tensor] = []
        projected: list[torch.Tensor] = []
        innovations: list[torch.Tensor] = []
        errors: list[torch.Tensor] = []
        likelihoods: list[torch.Tensor] = []
        for step in range(self.past_action_steps):
            prior, projected_increment, _realized = self._transition_step(
                content,
                past_actions[:, step],
            )
            priors.append(prior)
            projected.append(projected_increment)
            innovations.append(prior - content)
            probabilities, squared_error, likelihood = self._evidence_update(
                probabilities,
                prior,
                normalized[:, step + 1],
            )
            errors.append(squared_error)
            likelihoods.append(likelihood)
            posterior_history.append(probabilities)
            content = self._assimilate_observation(
                prior,
                normalized[:, step + 1],
            )

        belief = self._pack_belief(content, probabilities)
        return (
            history,
            belief,
            torch.stack(priors, dim=2),
            torch.stack(projected, dim=2),
            torch.stack(innovations, dim=2),
            torch.stack(posterior_history, dim=1),
            torch.stack(errors, dim=1),
            torch.stack(likelihoods, dim=1),
        )

    def encode_history(
        self,
        history_rgb: torch.Tensor,
        past_actions: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        history, belief, *_rest = self._encode_factual_history(
            history_rgb,
            past_actions,
        )
        return history, belief

    def _rollout_future(
        self,
        belief_latents: torch.Tensor,
        future_actions: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        content, probabilities = self._unpack_belief(belief_latents)
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
            next_content, projected_increment, realized_increment = (
                self._transition_step(content, future_actions[:, step])
            )
            states.append(next_content)
            projected.append(projected_increment)
            innovations.append(realized_increment)
            content = next_content
        return (
            torch.stack(states, dim=2),
            torch.stack(projected, dim=2),
            torch.stack(innovations, dim=2),
            probabilities,
        )

    def predict_trajectory_atoms_and_probabilities_from_belief(
        self,
        belief_latents: torch.Tensor,
        future_actions: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        atoms, _deltas, _innovations, probabilities = self._rollout_future(
            belief_latents,
            future_actions,
        )
        return atoms, probabilities

    def predict_trajectory_atoms_from_belief(
        self,
        belief_latents: torch.Tensor,
        future_actions: torch.Tensor,
    ) -> torch.Tensor:
        atoms, _probabilities = (
            self.predict_trajectory_atoms_and_probabilities_from_belief(
                belief_latents,
                future_actions,
            )
        )
        return atoms

    def predict_from_belief(
        self,
        belief_latents: torch.Tensor,
        future_actions: torch.Tensor,
    ) -> torch.Tensor:
        atoms, probabilities = (
            self.predict_trajectory_atoms_and_probabilities_from_belief(
                belief_latents,
                future_actions,
            )
        )
        return weighted_spherical_centroid(
            atoms,
            probabilities,
            epsilon=self.config.normalization_epsilon,
        )

    def forward(
        self,
        history_rgb: torch.Tensor,
        past_actions: torch.Tensor,
        future_actions: torch.Tensor,
        future_rgb: torch.Tensor | None = None,
    ) -> CausalPosteriorReweightedTransitionExpertTrajectoryH4JEPAOutput:
        (
            history,
            belief,
            observed_priors,
            observed_deltas,
            observed_innovations,
            posterior_history,
            evidence_squared_errors,
            evidence_likelihoods,
        ) = self._encode_factual_history(history_rgb, past_actions)
        atoms, deltas, future_innovations, probabilities = self._rollout_future(
            belief,
            future_actions,
        )
        all_six_innovations = torch.cat(
            (observed_innovations, future_innovations),
            dim=2,
        )
        centroid = weighted_spherical_centroid(
            atoms,
            probabilities,
            epsilon=self.config.normalization_epsilon,
        )
        weighted_deltas = (
            deltas * probabilities[:, :, None, None, None]
        ).sum(dim=1)
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
            per_sample, joint_score, combined = weighted_trajectory_energy_score(
                atoms,
                targets,
                probabilities,
            )
            per_horizon = per_sample.mean(dim=0)
            prediction_loss = combined.mean()

        return CausalPosteriorReweightedTransitionExpertTrajectoryH4JEPAOutput(
            predicted_latents=centroid,
            target_latents=targets,
            history_latents=history,
            belief_latents=belief,
            per_sample_horizon_loss=per_sample,
            per_horizon_loss=per_horizon,
            prediction_loss=prediction_loss,
            variance_loss=variance_loss,
            total_loss=None,
            predicted_deltas=weighted_deltas,
            trajectory_latents=atoms,
            trajectory_deltas=deltas,
            joint_energy_score=joint_score,
            trajectory_innovations=future_innovations,
            observed_prior_latents=observed_priors,
            observed_prior_deltas=observed_deltas,
            all_six_trajectory_innovations=all_six_innovations,
            final_hidden_particles=probabilities,
            posterior_probabilities=probabilities,
            posterior_history=posterior_history,
            evidence_squared_errors=evidence_squared_errors,
            evidence_likelihoods=evidence_likelihoods,
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
        if not isinstance(
            output,
            CausalPosteriorReweightedTransitionExpertTrajectoryH4JEPAOutput,
        ):
            raise TypeError(
                "output must be the causal posterior-reweighted expert output"
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
        _validate_probabilities(
            output.posterior_probabilities,
            batch=batch,
            atom_count=self.config.trajectory_atom_count,
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
        _future_horizon, _future_joint, cumulative = (
            weighted_trajectory_energy_score(
                output.trajectory_latents,
                target,
                output.posterior_probabilities,
            )
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


JointRecurrentH4JEPAConfig = (
    CausalPosteriorReweightedTransitionExpertTrajectoryH4JEPAConfig
)
JointRecurrentH4JEPA = CausalPosteriorReweightedTransitionExpertTrajectoryH4JEPA


__all__ = [
    "CausalPosteriorReweightedTransitionExpertTrajectoryH4JEPA",
    "CausalPosteriorReweightedTransitionExpertTrajectoryH4JEPAConfig",
    "CausalPosteriorReweightedTransitionExpertTrajectoryH4JEPAOutput",
    "FactualSharedTransitionTrajectoryH4JEPAOutput",
    "GO2_H4_PRIMITIVE_VOCABULARY",
    "JointRecurrentH4JEPA",
    "JointRecurrentH4JEPAConfig",
    "fixed_teacher_local_innovations",
    "realized_trajectory_innovations",
    "trajectory_energy_score",
    "weighted_pairwise_spread",
    "weighted_spherical_centroid",
    "weighted_trajectory_energy_score",
]
