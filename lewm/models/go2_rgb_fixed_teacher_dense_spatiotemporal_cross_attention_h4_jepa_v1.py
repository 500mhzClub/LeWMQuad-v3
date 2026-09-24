"""RGB fixed-teacher dense spatiotemporal cross-attention H4 JEPA V1.

The online path keeps every normalized spatial token from e0, e1, and e2.
Learned spatial and time embeddings identify frame tokens, while p0 and p1
are explicit interleaved transition tokens.  Exactly two pre-norm transformer
encoder blocks create the 770-token historical context.  Four independent
future query grids use ordered, fixed-slot action-prefix encodings and exactly
two shared pre-norm transformer decoder blocks.

The accepted N320 encoder copy is a permanently fixed teacher.  The online
encoder and dense attention predictor train jointly through raw fixed-teacher
future-minus-e2 delta regression and all-three-frame same-RGB alignment.  One
final delta Linear is zero-initialized, so update zero is exact persistence.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass
import math
from typing import Mapping

import torch
import torch.nn as nn
import torch.nn.functional as F

from .encoders import VisionEncoder
from .go2_recurrent_h4_joint_jepa import (
    GO2_H4_PRIMITIVE_VOCABULARY,
    JointRecurrentH4JEPAConfig as _V1Config,
    JointRecurrentH4JEPAOutput,
    _validate_encoder_state,
)


@dataclass(frozen=True)
class JointRecurrentH4JEPAConfig(_V1Config):
    """Dense-attention geometry and fixed-teacher objective weights."""

    cross_attention_heads: int = 6
    cross_attention_mlp_ratio: int = 4
    teacher_alignment_weight: float = 1.0
    teacher_delta_weight: float = 1.0

    def __post_init__(self) -> None:
        super().__post_init__()
        for name in ("cross_attention_heads", "cross_attention_mlp_ratio"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if self.feature_dim % self.cross_attention_heads != 0:
            raise ValueError("feature_dim must be divisible by cross_attention_heads")
        if self.cross_attention_mlp_ratio != 4:
            raise ValueError("cross_attention_mlp_ratio must remain exactly 4")
        if self.dropout != 0.0:
            raise ValueError("dense H4 V1 dropout must remain exactly zero")
        for name in ("teacher_alignment_weight", "teacher_delta_weight"):
            value = getattr(self, name)
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and non-negative")


@dataclass(frozen=True)
class DenseCrossAttentionH4JEPAOutput(JointRecurrentH4JEPAOutput):
    """Shared runner output contract plus raw direct future deltas."""

    predicted_deltas: torch.Tensor


class _DenseHistoricalContext(nn.Module):
    """Build and encode frame/action/frame/action/frame context tokens."""

    layer_count = 2

    def __init__(
        self,
        *,
        spatial_tokens: int,
        feature_dim: int,
        history_steps: int,
        past_action_steps: int,
        heads: int,
        mlp_ratio: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.history_steps = history_steps
        self.past_action_steps = past_action_steps
        self.spatial_embedding = nn.Embedding(spatial_tokens, feature_dim)
        self.time_embedding = nn.Embedding(history_steps, feature_dim)
        self.transition_step_embedding = nn.Embedding(
            past_action_steps,
            feature_dim,
        )
        self.encoder = nn.ModuleList(
            [
                nn.TransformerEncoderLayer(
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
        for embedding in (
            self.spatial_embedding,
            self.time_embedding,
            self.transition_step_embedding,
        ):
            nn.init.normal_(embedding.weight, mean=0.0, std=0.02)

    def forward(
        self,
        history_tokens: torch.Tensor,
        past_action_embeddings: torch.Tensor,
    ) -> torch.Tensor:
        batch, steps, tokens, dim = history_tokens.shape
        if steps != self.history_steps:
            raise ValueError("history token step count changed")
        if past_action_embeddings.shape != (
            batch,
            self.past_action_steps,
            dim,
        ):
            raise ValueError("past action embedding shape changed")
        if tokens != self.spatial_embedding.num_embeddings:
            raise ValueError("history spatial token count changed")
        spatial = self.spatial_embedding.weight
        time = self.time_embedding(
            torch.arange(steps, device=history_tokens.device)
        )
        frames = (
            history_tokens
            + spatial[None, None]
            + time[None, :, None]
        )
        transition = self.transition_step_embedding(
            torch.arange(self.past_action_steps, device=history_tokens.device)
        )
        action_tokens = past_action_embeddings + transition[None]
        interleaved = torch.cat(
            (
                frames[:, 0],
                action_tokens[:, 0:1],
                frames[:, 1],
                action_tokens[:, 1:2],
                frames[:, 2],
            ),
            dim=1,
        )
        expected_tokens = self.history_steps * tokens + self.past_action_steps
        if interleaved.shape != (batch, expected_tokens, dim):
            raise RuntimeError("interleaved dense history shape changed")
        encoded = interleaved
        for layer in self.encoder:
            encoded = layer(encoded)
        return encoded


class _DenseHorizonCrossAttention(nn.Module):
    """Decode four independent dense queries from one shared context."""

    layer_count = 2

    def __init__(
        self,
        *,
        spatial_tokens: int,
        feature_dim: int,
        future_steps: int,
        heads: int,
        mlp_ratio: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.future_steps = future_steps
        self.horizon_embedding = nn.Embedding(future_steps, feature_dim)
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
        self.spatial_tokens = spatial_tokens
        nn.init.normal_(self.horizon_embedding.weight, mean=0.0, std=0.02)

        prefix_mask = torch.tril(
            torch.ones(future_steps, future_steps, dtype=torch.bool)
        )
        self.register_buffer("prefix_mask", prefix_mask, persistent=True)

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

        horizons = self.horizon_embedding(
            torch.arange(self.future_steps, device=anchor.device)
        )
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
            anchor[:, None]
            + spatial_embeddings[None, None]
            + horizons[None, :, None]
            + action_prefix[:, :, None]
        )
        independent_queries = queries.reshape(
            batch * self.future_steps,
            tokens,
            dim,
        )
        repeated_memory = memory[:, None].expand(
            batch,
            self.future_steps,
            memory.shape[1],
            dim,
        ).reshape(batch * self.future_steps, memory.shape[1], dim)
        decoded = independent_queries
        for layer in self.decoder:
            decoded = layer(decoded, repeated_memory)
        return decoded.reshape(batch, self.future_steps, tokens, dim)


class JointRecurrentH4JEPA(nn.Module):
    """Shared-runner-compatible dense cross-attention fixed-teacher JEPA."""

    history_steps = 3
    past_action_steps = 2
    future_steps = 4

    def __init__(
        self,
        n320_encoder_state_dict: Mapping[str, torch.Tensor] | None = None,
        *,
        config: JointRecurrentH4JEPAConfig | None = None,
    ) -> None:
        super().__init__()
        self.config = config or JointRecurrentH4JEPAConfig()
        if not isinstance(self.config, JointRecurrentH4JEPAConfig):
            raise TypeError("config must be the dense V1 JointRecurrentH4JEPAConfig")

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
        if n320_encoder_state_dict is not None:
            _validate_encoder_state(self.encoder, n320_encoder_state_dict)
            self.encoder.load_state_dict(n320_encoder_state_dict, strict=True)

        self.action_embedding = nn.Embedding(self.config.action_count, dim)
        nn.init.normal_(self.action_embedding.weight, mean=0.0, std=0.02)

        # These compatibility names preserve the shared runner's reviewed
        # optimizer inventory.  None is a recurrent module.
        self.initial_belief = _DenseHistoricalContext(
            spatial_tokens=self.spatial_token_count,
            feature_dim=dim,
            history_steps=self.history_steps,
            past_action_steps=self.past_action_steps,
            heads=self.config.cross_attention_heads,
            mlp_ratio=self.config.cross_attention_mlp_ratio,
            dropout=self.config.dropout,
        )
        self.history_observation_norm = nn.Identity()
        self.history_cell = nn.Identity()
        self.history_spatial_refiner = nn.Identity()

        self.future_cell = _DenseHorizonCrossAttention(
            spatial_tokens=self.spatial_token_count,
            feature_dim=dim,
            future_steps=self.future_steps,
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

        self.target_encoder = copy.deepcopy(self.encoder)
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

    def train(self, mode: bool = True) -> "JointRecurrentH4JEPA":
        super().train(mode)
        self._freeze_target()
        return self

    def _validate_rgb_sequence(
        self,
        rgb: torch.Tensor,
        *,
        steps: int,
        name: str,
    ) -> int:
        expected = (steps, 3, self.config.image_size, self.config.image_size)
        if rgb.ndim != 5 or tuple(rgb.shape[1:]) != expected:
            raise ValueError(
                f"{name} must have shape "
                f"(B,{steps},3,{self.config.image_size},{self.config.image_size})"
            )
        if rgb.shape[0] < 1:
            raise ValueError(f"{name} must contain at least one row")
        if rgb.dtype != torch.float32:
            raise TypeError(f"{name} must be exact float32")
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
            raise ValueError(
                f"{name} entries must lie in [0,{self.config.action_count - 1}]"
            )

    def _encode_online_spatial(self, rgb: torch.Tensor) -> torch.Tensor:
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

    def encode_history(
        self,
        history_rgb: torch.Tensor,
        past_actions: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return online history and packed e2-anchor plus dense memory."""

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
        normalized_history = F.normalize(
            history,
            p=2.0,
            dim=-1,
            eps=self.config.normalization_epsilon,
        )
        past_embeddings = self.action_embedding(past_actions)
        memory = self.initial_belief(normalized_history, past_embeddings)
        anchor = history[:, 2]
        belief = torch.cat((anchor, memory), dim=1)
        return history, belief

    def _predict_with_deltas(
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
        predictions = F.normalize(
            anchor[:, None] + anchor_norm[:, None] * deltas,
            p=2.0,
            dim=-1,
            eps=self.config.normalization_epsilon,
        )
        return predictions, deltas

    def predict_from_belief(
        self,
        belief_latents: torch.Tensor,
        future_actions: torch.Tensor,
    ) -> torch.Tensor:
        predictions, _deltas = self._predict_with_deltas(
            belief_latents,
            future_actions,
        )
        return predictions

    @torch.no_grad()
    def encode_target(self, future_rgb: torch.Tensor) -> torch.Tensor:
        batch = self._validate_rgb_sequence(
            future_rgb,
            steps=self.future_steps,
            name="future_rgb",
        )
        tokens = self.target_encoder.forward_tokens(
            future_rgb.reshape(batch * self.future_steps, *future_rgb.shape[2:])
        )[:, 1:]
        tokens = tokens.reshape(
            batch,
            self.future_steps,
            self.spatial_token_count,
            self.config.feature_dim,
        )
        return F.normalize(
            tokens,
            p=2.0,
            dim=-1,
            eps=self.config.normalization_epsilon,
        ).detach()

    @torch.no_grad()
    def _encode_fixed_teacher_history(
        self,
        history_rgb: torch.Tensor,
    ) -> torch.Tensor:
        batch = self._validate_rgb_sequence(
            history_rgb,
            steps=self.history_steps,
            name="history_rgb",
        )
        tokens = self.target_encoder.forward_tokens(
            history_rgb.reshape(batch * self.history_steps, *history_rgb.shape[2:])
        )[:, 1:]
        tokens = tokens.reshape(
            batch,
            self.history_steps,
            self.spatial_token_count,
            self.config.feature_dim,
        )
        return F.normalize(
            tokens,
            p=2.0,
            dim=-1,
            eps=self.config.normalization_epsilon,
        ).detach()

    def forward(
        self,
        history_rgb: torch.Tensor,
        past_actions: torch.Tensor,
        future_actions: torch.Tensor,
        future_rgb: torch.Tensor | None = None,
    ) -> DenseCrossAttentionH4JEPAOutput:
        history, belief = self.encode_history(history_rgb, past_actions)
        predictions, predicted_deltas = self._predict_with_deltas(
            belief,
            future_actions,
        )
        variance_loss = history.sum() * 0.0

        targets: torch.Tensor | None = None
        per_sample: torch.Tensor | None = None
        per_horizon: torch.Tensor | None = None
        prediction_loss: torch.Tensor | None = None
        if future_rgb is not None:
            if future_rgb.shape[0] != history_rgb.shape[0]:
                raise ValueError("history_rgb and future_rgb batch sizes differ")
            targets = self.encode_target(future_rgb)
            per_sample = (predictions - targets).square().sum(dim=-1).mean(dim=-1)
            per_horizon = per_sample.mean(dim=0)
            prediction_loss = per_horizon.mean()

        return DenseCrossAttentionH4JEPAOutput(
            predicted_latents=predictions,
            target_latents=targets,
            history_latents=history,
            belief_latents=belief,
            per_sample_horizon_loss=per_sample,
            per_horizon_loss=per_horizon,
            prediction_loss=prediction_loss,
            variance_loss=variance_loss,
            total_loss=None,
            predicted_deltas=predicted_deltas,
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
        if not isinstance(output, DenseCrossAttentionH4JEPAOutput):
            raise TypeError("output must be DenseCrossAttentionH4JEPAOutput")
        if target_latents.shape != output.predicted_latents.shape:
            raise ValueError("target_latents shape differs from dense predictions")

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

        teacher_delta = target_latents.detach() - teacher_history[:, 2:3]
        if output.predicted_deltas.shape != teacher_delta.shape:
            raise ValueError("predicted and fixed-teacher delta shapes differ")
        delta_loss = (
            (output.predicted_deltas - teacher_delta)
            .square()
            .sum(dim=-1)
            .mean()
        )
        return {
            "history_teacher_alignment": (
                self.config.teacher_alignment_weight * alignment
            ),
            "future_teacher_delta": self.config.teacher_delta_weight * delta_loss,
        }

    @torch.no_grad()
    def hard_sync_target(self) -> None:
        self.ema_update_count.zero_()
        self._freeze_target()

    @torch.no_grad()
    def update_target(self, momentum: float | None = None) -> None:
        value = self.config.target_ema_momentum if momentum is None else momentum
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            raise TypeError("momentum must be a real number")
        value = float(value)
        if not math.isfinite(value) or not 0.0 <= value < 1.0:
            raise ValueError("momentum must lie in [0,1)")
        self._freeze_target()


__all__ = [
    "DenseCrossAttentionH4JEPAOutput",
    "GO2_H4_PRIMITIVE_VOCABULARY",
    "JointRecurrentH4JEPA",
    "JointRecurrentH4JEPAConfig",
]
