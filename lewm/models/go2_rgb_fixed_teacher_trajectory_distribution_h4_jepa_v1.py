"""RGB fixed-teacher finite-support trajectory-distribution H4 JEPA V1.

This model keeps the reviewed dense three-frame history substrate, but replaces
the single deterministic future with four equal-mass, coherent H1--H4
trajectory atoms.  One learned mode embedding is shared across all horizons of
an atom.  The online encoder and trajectory predictor train jointly with a
proper energy score against fixed-teacher future latents plus the inherited
three-frame online-to-teacher alignment.

The final shared delta projection is exactly zero initialized.  Consequently
all four atoms are exact e2 persistence before the first optimizer update.
There is no learned variance, mixture weight, best-of-K loss, control ranking,
navigation label, or target-encoder update.
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
    JointRecurrentH4JEPAOutput,
)
from .go2_rgb_fixed_teacher_dense_spatiotemporal_cross_attention_h4_jepa_v1 import (
    DenseCrossAttentionH4JEPAOutput,
    JointRecurrentH4JEPA as _DenseH4JEPA,
    JointRecurrentH4JEPAConfig as _DenseH4JEPAConfig,
)


@dataclass(frozen=True)
class JointRecurrentH4JEPAConfig(_DenseH4JEPAConfig):
    """Exact finite-support trajectory-distribution contract."""

    trajectory_atom_count: int = 4

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.trajectory_atom_count != 4:
            raise ValueError("trajectory_atom_count must remain exactly four")


@dataclass(frozen=True)
class TrajectoryDistributionH4JEPAOutput(DenseCrossAttentionH4JEPAOutput):
    """Shared-runner fields plus all equal-mass trajectory atoms."""

    trajectory_latents: torch.Tensor
    trajectory_deltas: torch.Tensor
    joint_energy_score: torch.Tensor | None


class _TrajectoryDistributionCrossAttention(nn.Module):
    """Decode K coherent trajectory atoms from one dense causal context."""

    layer_count = 2

    def __init__(
        self,
        *,
        spatial_tokens: int,
        feature_dim: int,
        future_steps: int,
        atom_count: int,
        heads: int,
        mlp_ratio: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.spatial_tokens = spatial_tokens
        self.future_steps = future_steps
        self.atom_count = atom_count
        self.horizon_embedding = nn.Embedding(future_steps, feature_dim)
        self.mode_embedding = nn.Embedding(atom_count, feature_dim)
        self.future_action_path = nn.Sequential(
            nn.Linear(future_steps * feature_dim, feature_dim),
            nn.GELU(),
            nn.Linear(feature_dim, feature_dim),
        )
        self.decoder = nn.ModuleList(
            [
                nn.TransformerDecoderLayer(
                    d_model=feature_dim,
                    nhead=heads,
                    dim_feedforward=feature_dim * mlp_ratio,
                    dropout=dropout,
                    activation="gelu",
                    batch_first=True,
                    norm_first=True,
                )
                for _ in range(self.layer_count)
            ]
        )
        nn.init.normal_(self.horizon_embedding.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.mode_embedding.weight, mean=0.0, std=0.02)
        self.register_buffer(
            "prefix_mask",
            torch.tril(torch.ones(future_steps, future_steps, dtype=torch.bool)),
            persistent=True,
        )

    def forward(
        self,
        anchor: torch.Tensor,
        memory: torch.Tensor,
        future_action_embeddings: torch.Tensor,
        spatial_embeddings: torch.Tensor,
    ) -> torch.Tensor:
        batch, tokens, dim = anchor.shape
        if tokens != self.spatial_tokens:
            raise ValueError("future spatial token count changed")
        if future_action_embeddings.shape != (batch, self.future_steps, dim):
            raise ValueError("future action embedding shape changed")
        if memory.ndim != 3 or memory.shape[0] != batch or memory.shape[2] != dim:
            raise ValueError("dense history memory shape changed")
        if spatial_embeddings.shape != (tokens, dim):
            raise ValueError("shared spatial embedding shape changed")

        device = anchor.device
        horizons = self.horizon_embedding(torch.arange(self.future_steps, device=device))
        modes = self.mode_embedding(torch.arange(self.atom_count, device=device))
        fixed_slots = future_action_embeddings[:, None].expand(
            batch,
            self.future_steps,
            self.future_steps,
            dim,
        )
        fixed_slots = fixed_slots * self.prefix_mask[None, :, :, None]
        action_prefix = self.future_action_path(
            fixed_slots.reshape(batch, self.future_steps, self.future_steps * dim)
        )
        queries = (
            anchor[:, None, None]
            + spatial_embeddings[None, None, None]
            + horizons[None, None, :, None]
            + modes[None, :, None, None]
            + action_prefix[:, None, :, None]
        )
        flat_queries = queries.reshape(
            batch * self.atom_count * self.future_steps,
            tokens,
            dim,
        )
        repeated_memory = memory[:, None, None].expand(
            batch,
            self.atom_count,
            self.future_steps,
            memory.shape[1],
            dim,
        ).reshape(
            batch * self.atom_count * self.future_steps,
            memory.shape[1],
            dim,
        )
        decoded = flat_queries
        for layer in self.decoder:
            decoded = layer(decoded, repeated_memory)
        return decoded.reshape(
            batch,
            self.atom_count,
            self.future_steps,
            tokens,
            dim,
        )


def _lattice_distance(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    """Euclidean token-lattice distance divided by sqrt(token count)."""

    if left.ndim < 2 or right.ndim < 2 or left.shape[-2:] != right.shape[-2:]:
        raise ValueError("energy-score lattice shapes are not broadcastable")
    try:
        torch.broadcast_shapes(left.shape, right.shape)
    except RuntimeError as error:
        raise ValueError(
            "energy-score lattice shapes are not broadcastable"
        ) from error
    token_count = int(left.shape[-2])
    return torch.linalg.vector_norm(left - right, dim=(-2, -1)) / math.sqrt(
        float(token_count)
    )


def trajectory_energy_score(
    atoms: torch.Tensor,
    target: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return per-horizon, joint, and equally weighted combined energy scores.

    ``atoms`` is ``(B,K,H,P,D)`` and ``target`` is ``(B,H,P,D)``.  The
    empirical distribution has fixed equal mass.  Distances are ordinary
    Euclidean distances on tokenwise unit-normalized latent lattices.
    """

    if atoms.ndim != 5 or target.ndim != 4:
        raise ValueError("trajectory energy score expects B,K,H,P,D and B,H,P,D")
    if atoms.shape[0] != target.shape[0] or atoms.shape[2:] != target.shape[1:]:
        raise ValueError("trajectory atoms and target shapes differ")
    if atoms.shape[1] < 2:
        raise ValueError("trajectory energy score requires at least two atoms")
    fit_horizon = _lattice_distance(atoms, target[:, None]).mean(dim=1)
    pair_horizon = _lattice_distance(atoms[:, :, None], atoms[:, None, :]).mean(
        dim=(1, 2)
    )
    horizon_score = fit_horizon - 0.5 * pair_horizon

    batch, atom_count, horizons, tokens, dim = atoms.shape
    flat_atoms = atoms.reshape(batch, atom_count, horizons * tokens, dim)
    flat_target = target.reshape(batch, horizons * tokens, dim)
    joint_fit = _lattice_distance(flat_atoms, flat_target[:, None]).mean(dim=1)
    joint_pair = _lattice_distance(
        flat_atoms[:, :, None],
        flat_atoms[:, None, :],
    ).mean(dim=(1, 2))
    joint_score = joint_fit - 0.5 * joint_pair
    combined = 0.5 * joint_score + 0.5 * horizon_score.mean(dim=1)
    return horizon_score, joint_score, combined


class JointRecurrentH4JEPA(_DenseH4JEPA):
    """Shared-runner-compatible equal-mass trajectory-distribution JEPA."""

    def __init__(
        self,
        n320_encoder_state_dict: Mapping[str, torch.Tensor] | None = None,
        *,
        config: JointRecurrentH4JEPAConfig | None = None,
    ) -> None:
        selected = config or JointRecurrentH4JEPAConfig()
        if not isinstance(selected, JointRecurrentH4JEPAConfig):
            raise TypeError("config must be the trajectory V1 config")
        super().__init__(
            n320_encoder_state_dict=n320_encoder_state_dict,
            config=selected,
        )
        dim = self.config.feature_dim
        self.future_cell = _TrajectoryDistributionCrossAttention(
            spatial_tokens=self.spatial_token_count,
            feature_dim=dim,
            future_steps=self.future_steps,
            atom_count=self.config.trajectory_atom_count,
            heads=self.config.cross_attention_heads,
            mlp_ratio=self.config.cross_attention_mlp_ratio,
            dropout=self.config.dropout,
        )

    def _predict_trajectory_with_deltas(
        self,
        belief_latents: torch.Tensor,
        future_actions: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
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
        future_embeddings = self.action_embedding(future_actions)
        normalized_anchor = F.normalize(
            anchor,
            p=2.0,
            dim=-1,
            eps=self.config.normalization_epsilon,
        )
        hidden = self.future_cell(
            normalized_anchor,
            memory,
            future_embeddings,
            self.initial_belief.spatial_embedding.weight,
        )
        deltas = self.prediction_projector(hidden)
        anchor_norm = anchor.norm(p=2.0, dim=-1, keepdim=True).clamp_min(
            self.config.normalization_epsilon
        )
        atoms = F.normalize(
            anchor[:, None, None] + anchor_norm[:, None, None] * deltas,
            p=2.0,
            dim=-1,
            eps=self.config.normalization_epsilon,
        )
        return atoms, deltas

    def predict_trajectory_atoms_from_belief(
        self,
        belief_latents: torch.Tensor,
        future_actions: torch.Tensor,
    ) -> torch.Tensor:
        atoms, _deltas = self._predict_trajectory_with_deltas(
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
    ) -> TrajectoryDistributionH4JEPAOutput:
        history, belief = self.encode_history(history_rgb, past_actions)
        atoms, deltas = self._predict_trajectory_with_deltas(belief, future_actions)
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

        return TrajectoryDistributionH4JEPAOutput(
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
        del past_actions, future_actions
        if not isinstance(output, TrajectoryDistributionH4JEPAOutput):
            raise TypeError("output must be TrajectoryDistributionH4JEPAOutput")
        expected = (
            target_latents.shape[0],
            self.config.trajectory_atom_count,
            *target_latents.shape[1:],
        )
        if output.trajectory_latents.shape != expected:
            raise ValueError("target and trajectory-atom shapes differ")

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
        _per_horizon, _joint, combined = trajectory_energy_score(
            output.trajectory_latents,
            target_latents.detach(),
        )
        return {
            "history_teacher_alignment": (
                self.config.teacher_alignment_weight * alignment
            ),
            "future_teacher_trajectory_energy_score": (
                self.config.teacher_delta_weight * combined.mean()
            ),
        }


__all__ = [
    "GO2_H4_PRIMITIVE_VOCABULARY",
    "JointRecurrentH4JEPA",
    "JointRecurrentH4JEPAConfig",
    "TrajectoryDistributionH4JEPAOutput",
    "trajectory_energy_score",
]
