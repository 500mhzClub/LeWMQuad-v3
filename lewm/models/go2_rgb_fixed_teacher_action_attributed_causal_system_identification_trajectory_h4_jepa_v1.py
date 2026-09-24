"""Fixed-teacher action-attributed causal system-ID H4 JEPA V1.

The model keeps four causal state atoms ``(q, M)``.  ``q`` is the current
normalized feature lattice.  ``M`` is a compact nonspatial response matrix
written exactly twice from observed prior errors and the centered key of the
requested action that caused each error.  During open-loop prediction ``M``
is fixed and may affect a successor only as a bounded multiplier inside the
mean-centered categorical action interaction::

    b = B(q, centered_mode_and_spatial_context, 0)
    mu = 1 + tanh(P_M(vec(M)))
    delta = W0(b * mu * c_action)
    q_next = renormalize(q + delta)

There is no momentum, incoming increment, generic state-only successor,
future observation, or dense-history carrier.  ``W0`` is shared, bias-free,
and zero initialized, so all paths begin at exact persistence.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Mapping

import torch
import torch.nn as nn
import torch.nn.functional as F

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
class ActionAttributedCausalSystemIdentificationTrajectoryH4JEPAConfig(
    FactorizedConditionalIncrementTrajectoryH4JEPAConfig
):
    """Inherited K4 contract plus the compact system-ID matrix width."""

    system_identification_dim: int = 16

    def __post_init__(self) -> None:
        super().__post_init__()
        value = self.system_identification_dim
        if isinstance(value, bool) or not isinstance(value, int) or value != 16:
            raise ValueError("system_identification_dim must be exactly 16")


class _CenteredSystemIdentificationState(nn.Module):
    """Initialize q/M and own the inherited centered mode/spatial context."""

    def __init__(
        self,
        *,
        spatial_tokens: int,
        feature_dim: int,
        atom_count: int,
        system_identification_dim: int,
    ) -> None:
        super().__init__()
        self.spatial_tokens = spatial_tokens
        self.atom_count = atom_count
        self.system_identification_dim = system_identification_dim
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
        memory = initial_content.new_zeros(
            batch,
            self.atom_count,
            self.system_identification_dim,
            self.system_identification_dim,
        )
        return content, memory


class _SystemIdentificationMemoryProjections(nn.Module):
    """Own the exact bias-free P_c and P_M projections."""

    def __init__(
        self,
        *,
        feature_dim: int,
        system_identification_dim: int,
    ) -> None:
        super().__init__()
        self.key_projection = nn.Linear(
            feature_dim,
            system_identification_dim,
            bias=False,
        )
        self.memory_projection = nn.Linear(
            system_identification_dim * system_identification_dim,
            feature_dim,
            bias=False,
        )


class ActionAttributedCausalSystemIdentificationTrajectoryH4JEPA(
    FactorizedConditionalIncrementTrajectoryH4JEPA
):
    """K4 predict-before-observe model with fixed bilinear system-ID writes."""

    def __init__(
        self,
        n320_encoder_state_dict: Mapping[str, torch.Tensor] | None = None,
        *,
        config: (
            ActionAttributedCausalSystemIdentificationTrajectoryH4JEPAConfig
            | None
        ) = None,
    ) -> None:
        selected = (
            config
            or ActionAttributedCausalSystemIdentificationTrajectoryH4JEPAConfig()
        )
        if not isinstance(
            selected,
            ActionAttributedCausalSystemIdentificationTrajectoryH4JEPAConfig,
        ):
            raise TypeError(
                "config must be "
                "ActionAttributedCausalSystemIdentificationTrajectoryH4JEPAConfig"
            )
        super().__init__(
            n320_encoder_state_dict=n320_encoder_state_dict,
            config=selected,
        )

        dim = self.config.feature_dim
        memory_dim = self.config.system_identification_dim
        carrier_capacity = self.spatial_token_count * dim
        if memory_dim * memory_dim > carrier_capacity:
            raise ValueError(
                "system-identification matrix exceeds one belief carrier"
            )

        self.initial_belief = _CenteredSystemIdentificationState(
            spatial_tokens=self.spatial_token_count,
            feature_dim=dim,
            atom_count=self.config.trajectory_atom_count,
            system_identification_dim=memory_dim,
        )
        # These inherited inventory names keep every fresh parameter in the
        # reviewed history/predictor optimizer groups.  They implement exactly
        # LN, P_r, and the paired P_c/P_M projections, respectively.
        self.history_observation_norm = nn.LayerNorm(
            dim,
            elementwise_affine=False,
            eps=1e-5,
        )
        self.history_cell = nn.Linear(dim, memory_dim, bias=False)
        self.history_spatial_refiner = _SystemIdentificationMemoryProjections(
            feature_dim=dim,
            system_identification_dim=memory_dim,
        )
        # ``future_cell``, ``future_spatial_refiner``, and
        # ``prediction_projector`` remain the exact inherited B, centered A,
        # and bias-free zero-initialized W0 constructions.

    @property
    def response_projection(self) -> nn.Linear:
        """Return P_r without registering a duplicate module alias."""

        return self.history_cell

    @property
    def action_key_projection(self) -> nn.Linear:
        """Return P_c without registering a duplicate module alias."""

        return self.history_spatial_refiner.key_projection

    @property
    def memory_projection(self) -> nn.Linear:
        """Return P_M without registering a duplicate module alias."""

        return self.history_spatial_refiner.memory_projection

    def _mode_context(self, reference: torch.Tensor) -> torch.Tensor:
        return self.initial_belief.centered_context(
            batch=int(reference.shape[0]),
            device=reference.device,
            dtype=reference.dtype,
        )

    def _centered_action_keys(self) -> torch.Tensor:
        """Project the complete centered action table, bound, then recenter."""

        action_codes = self._centered_action_codes()
        raw_keys = torch.tanh(self.action_key_projection(action_codes))
        keys = raw_keys - raw_keys.mean(dim=0, keepdim=True)
        expected = (
            len(GO2_H4_PRIMITIVE_VOCABULARY),
            self.config.system_identification_dim,
        )
        if tuple(keys.shape) != expected:
            raise ValueError(f"centered action keys must have shape {expected}")
        return keys

    def _memory_response(self, innovation: torch.Tensor) -> torch.Tensor:
        """Pool a spatial prior error into the bounded compact rho vector."""

        if innovation.ndim != 4:
            raise ValueError("innovation must have shape B,K,P,D")
        expected_tail = (
            self.config.trajectory_atom_count,
            self.spatial_token_count,
            self.config.feature_dim,
        )
        if tuple(innovation.shape[1:]) != expected_tail:
            raise ValueError("innovation state shape changed")
        normalized = self.history_observation_norm(innovation)
        pooled = normalized.mean(dim=2)
        response = torch.tanh(self.response_projection(pooled))
        expected = (
            int(innovation.shape[0]),
            self.config.trajectory_atom_count,
            self.config.system_identification_dim,
        )
        if tuple(response.shape) != expected:
            raise ValueError(f"memory response must have shape {expected}")
        return response

    def _memory_modulation(self, memory: torch.Tensor) -> torch.Tensor:
        """Read nonspatial M only as the bounded feature multiplier mu."""

        memory_dim = self.config.system_identification_dim
        if memory.ndim != 4 or tuple(memory.shape[1:]) != (
            self.config.trajectory_atom_count,
            memory_dim,
            memory_dim,
        ):
            raise ValueError("memory must have shape B,K,I,I")
        flattened = memory.reshape(
            memory.shape[0],
            memory.shape[1],
            memory_dim * memory_dim,
        )
        modulation = 1.0 + torch.tanh(self.memory_projection(flattened))
        expected = (
            int(memory.shape[0]),
            self.config.trajectory_atom_count,
            self.config.feature_dim,
        )
        if tuple(modulation.shape) != expected:
            raise ValueError(f"memory modulation must have shape {expected}")
        return modulation

    def _write_memory(
        self,
        memory: torch.Tensor,
        innovation: torch.Tensor,
        action_indices: torch.Tensor,
    ) -> torch.Tensor:
        """Apply the sole additive rho outer centered-action-key write."""

        batch = int(innovation.shape[0]) if innovation.ndim == 4 else -1
        memory_dim = self.config.system_identification_dim
        if memory.ndim != 4 or tuple(memory.shape) != (
            batch,
            self.config.trajectory_atom_count,
            memory_dim,
            memory_dim,
        ):
            raise ValueError("memory and innovation batch/state shapes differ")
        if action_indices.ndim != 1 or action_indices.dtype != torch.long:
            raise TypeError("memory-write actions must be long with shape (B,)")
        if tuple(action_indices.shape) != (batch,):
            raise ValueError("memory-write action batch size changed")
        if action_indices.device != memory.device or innovation.device != memory.device:
            raise TypeError("memory, innovation, and action must share a device")
        response = self._memory_response(innovation)
        keys = self._centered_action_keys()
        selected_keys = keys.index_select(0, action_indices)
        outer = response.unsqueeze(-1) * selected_keys[:, None, None, :]
        return memory + outer * (1.0 / math.sqrt(float(memory_dim)))

    def _transition_step(
        self,
        content: torch.Tensor,
        memory: torch.Tensor,
        action_indices: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Apply one shared modulation-only prior and preserve M bitwise."""

        expected_content = (
            self.config.trajectory_atom_count,
            self.spatial_token_count,
            self.config.feature_dim,
        )
        if content.ndim != 4 or tuple(content.shape[1:]) != expected_content:
            raise ValueError("content must have shape B,K,P,D")
        batch = int(content.shape[0])
        memory_dim = self.config.system_identification_dim
        if tuple(memory.shape) != (
            batch,
            self.config.trajectory_atom_count,
            memory_dim,
            memory_dim,
        ):
            raise ValueError("transition memory must have shape B,K,I,I")
        if action_indices.ndim != 1 or action_indices.dtype != torch.long:
            raise TypeError("one-step actions must be long with shape (B,)")
        if tuple(action_indices.shape) != (batch,):
            raise ValueError("one-step action batch size changed")
        if action_indices.device != content.device or memory.device != content.device:
            raise TypeError("state and action must share a device")

        context = self.future_cell(
            content,
            self._mode_context(content),
            torch.zeros_like(content),
        )
        modulation = self._memory_modulation(memory)
        action_codes = self._centered_action_codes()
        selected_codes = action_codes.index_select(0, action_indices)
        interaction = (
            context
            * modulation[:, :, None, :]
            * selected_codes[:, None, None, :]
        )
        projected_increment = self.prediction_projector(interaction)
        next_content = _renormalized_local_step(
            content,
            projected_increment,
            epsilon=self.config.normalization_epsilon,
        )
        realized_increment = next_content - content
        return next_content, memory, projected_increment, realized_increment

    def _observe(
        self,
        prior_content: torch.Tensor,
        memory: torch.Tensor,
        observation: torch.Tensor,
        past_action_indices: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Write the scored prior error, then insert factual online q."""

        if observation.ndim != 3:
            raise ValueError("observation must have shape B,P,D")
        expanded = observation[:, None].expand_as(prior_content)
        innovation = expanded - prior_content
        updated_memory = self._write_memory(
            memory,
            innovation,
            past_action_indices,
        )
        return expanded, updated_memory, innovation

    def _pack_belief(
        self,
        content: torch.Tensor,
        memory: torch.Tensor,
    ) -> torch.Tensor:
        """Pack q and row-major M with exact trailing-zero carrier padding."""

        expected_content = (
            self.config.trajectory_atom_count,
            self.spatial_token_count,
            self.config.feature_dim,
        )
        if content.ndim != 4 or tuple(content.shape[1:]) != expected_content:
            raise ValueError("belief content must have shape B,K,P,D")
        batch = int(content.shape[0])
        memory_dim = self.config.system_identification_dim
        if tuple(memory.shape) != (
            batch,
            self.config.trajectory_atom_count,
            memory_dim,
            memory_dim,
        ):
            raise ValueError("belief memory must have shape B,K,I,I")
        if memory.device != content.device or memory.dtype != content.dtype:
            raise TypeError("belief q and M must share device and dtype")
        carrier_size = self.spatial_token_count * self.config.feature_dim
        memory_size = memory_dim * memory_dim
        flattened = memory.reshape(batch, self.config.trajectory_atom_count, -1)
        padded = F.pad(flattened, (0, carrier_size - memory_size))
        carriers = padded.reshape(
            batch,
            self.config.trajectory_atom_count,
            self.spatial_token_count,
            self.config.feature_dim,
        )
        return torch.cat((content, carriers), dim=1)

    def _unpack_belief(
        self,
        belief_latents: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Unpack q/M and reject any nonzero serialized padding."""

        atom_count = self.config.trajectory_atom_count
        expected = (
            2 * atom_count,
            self.spatial_token_count,
            self.config.feature_dim,
        )
        if belief_latents.ndim != 4 or tuple(belief_latents.shape[1:]) != expected:
            raise ValueError(
                "belief_latents must contain packed q/M with shape "
                f"(B,{expected[0]},{expected[1]},{expected[2]})"
            )
        if not torch.is_floating_point(belief_latents):
            raise TypeError("belief_latents must have a floating dtype")
        if belief_latents.device != self.action_embedding.weight.device:
            raise TypeError("belief_latents and model must share a device")
        if not bool(torch.isfinite(belief_latents).all()):
            raise FloatingPointError("belief_latents contains a nonfinite value")

        content = belief_latents[:, :atom_count]
        carrier = belief_latents[:, atom_count:]
        flattened = carrier.reshape(carrier.shape[0], atom_count, -1)
        memory_dim = self.config.system_identification_dim
        memory_size = memory_dim * memory_dim
        padding = flattened[..., memory_size:]
        if padding.numel() and int(torch.count_nonzero(padding).item()) != 0:
            raise ValueError("serialized system-identification padding is nonzero")
        memory = flattened[..., :memory_size].reshape(
            carrier.shape[0],
            atom_count,
            memory_dim,
            memory_dim,
        )
        return content, memory

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
        content, memory = self.initial_belief(normalized[:, 0])
        priors: list[torch.Tensor] = []
        projected_deltas: list[torch.Tensor] = []
        scored_innovations: list[torch.Tensor] = []
        for step in range(self.past_action_steps):
            prior_content, memory, projected_increment, _realized_increment = (
                self._transition_step(
                    content,
                    memory,
                    past_actions[:, step],
                )
            )
            priors.append(prior_content)
            projected_deltas.append(projected_increment)
            factual_source = normalized[:, step, None].expand_as(prior_content)
            scored_innovations.append(prior_content - factual_source)
            content, memory, _innovation = self._observe(
                prior_content,
                memory,
                normalized[:, step + 1],
                past_actions[:, step],
            )

        belief = self._pack_belief(content, memory)
        return (
            history,
            belief,
            torch.stack(priors, dim=2),
            torch.stack(projected_deltas, dim=2),
            torch.stack(scored_innovations, dim=2),
        )

    def _rollout_future(
        self,
        belief_latents: torch.Tensor,
        future_actions: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        content, memory = self._unpack_belief(belief_latents)
        batch = int(belief_latents.shape[0])
        self._validate_actions(
            future_actions,
            batch=batch,
            steps=self.future_steps,
            name="future_actions",
        )
        states: list[torch.Tensor] = []
        projected_deltas: list[torch.Tensor] = []
        realized: list[torch.Tensor] = []
        for step in range(self.future_steps):
            content, memory, projected_increment, realized_increment = (
                self._transition_step(
                    content,
                    memory,
                    future_actions[:, step],
                )
            )
            states.append(content)
            projected_deltas.append(projected_increment)
            realized.append(realized_increment)
        return (
            torch.stack(states, dim=2),
            torch.stack(projected_deltas, dim=2),
            torch.stack(realized, dim=2),
            memory,
        )


# Preserve the reviewed shared-runner constructor API.
JointRecurrentH4JEPAConfig = (
    ActionAttributedCausalSystemIdentificationTrajectoryH4JEPAConfig
)
JointRecurrentH4JEPA = (
    ActionAttributedCausalSystemIdentificationTrajectoryH4JEPA
)


__all__ = [
    "ActionAttributedCausalSystemIdentificationTrajectoryH4JEPA",
    "ActionAttributedCausalSystemIdentificationTrajectoryH4JEPAConfig",
    "FactualSharedTransitionTrajectoryH4JEPAOutput",
    "GO2_H4_PRIMITIVE_VOCABULARY",
    "JointRecurrentH4JEPA",
    "JointRecurrentH4JEPAConfig",
    "fixed_teacher_local_innovations",
    "realized_trajectory_innovations",
    "trajectory_energy_score",
]
