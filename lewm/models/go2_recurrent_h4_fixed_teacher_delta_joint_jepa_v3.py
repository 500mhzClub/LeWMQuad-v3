"""Fixed-teacher recurrent H4 joint JEPA V3.

The accepted N320 representation is retained as a permanently fixed target
encoder.  The online encoder, ordered-history recurrence, and sequential
action predictor remain jointly trainable.  A belief tensor carries a hard
online-e2 anchor alongside an ordinary recurrent history context.
The shared future recurrence emits one direct cumulative e2-to-horizon delta
at each step; predicted deltas are never recursively accumulated.  Exactly
one final delta Linear is zero-initialized, so every horizon begins at exact
persistence without zero-gating any upstream trainable path.

The training objectives align all three online history frames with their
fixed-teacher counterparts and regress raw predicted future deltas onto
fixed-teacher future-minus-e2 deltas.  Absolute predictions are evaluation
outputs only; they do not own a V3 training-loss term.
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
    """V3 fixed-teacher alignment and delta-regression weights."""

    teacher_alignment_weight: float = 1.0
    teacher_delta_weight: float = 1.0

    def __post_init__(self) -> None:
        super().__post_init__()
        for name in ("teacher_alignment_weight", "teacher_delta_weight"):
            value = getattr(self, name)
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and non-negative")


@dataclass(frozen=True)
class FixedTeacherDeltaH4JEPAOutput(JointRecurrentH4JEPAOutput):
    """V1 output contract plus the raw direct future deltas."""

    predicted_deltas: torch.Tensor


class JointRecurrentH4JEPA(_V1JointRecurrentH4JEPA):
    """V1-compatible recurrent JEPA with a permanently fixed teacher."""

    def __init__(
        self,
        n320_encoder_state_dict: Mapping[str, torch.Tensor] | None = None,
        *,
        config: JointRecurrentH4JEPAConfig | None = None,
    ) -> None:
        resolved = config or JointRecurrentH4JEPAConfig()
        if not isinstance(resolved, JointRecurrentH4JEPAConfig):
            raise TypeError("config must be the V3 JointRecurrentH4JEPAConfig")
        super().__init__(
            n320_encoder_state_dict=n320_encoder_state_dict,
            config=resolved,
        )
        dim = self.config.feature_dim
        # Preserve the reviewed optimizer module name.  Only this final Linear
        # is zero-initialized; the future recurrence and all history modules
        # retain their ordinary initialization and receive gradients through it.
        self.prediction_projector = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim),
        )
        final = self.prediction_projector[-1]
        nn.init.zeros_(final.weight)
        nn.init.zeros_(final.bias)
        self._freeze_target()

    def encode_history(
        self,
        history_rgb: torch.Tensor,
        past_actions: torch.Tensor,
    ) -> OnlineH4Context:
        """Return concat(hard online-e2 anchor, ordered-history context)."""

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
        context = self.initial_belief(normalized_history[:, 0])
        tokens = self.spatial_token_count
        dim = self.config.feature_dim
        for step in range(self.past_action_steps):
            observation = self.history_observation_norm(
                normalized_history[:, step + 1]
            )
            action = self.action_embedding(past_actions[:, step])[:, None].expand(
                -1, tokens, -1
            )
            recurrent_input = torch.cat((observation, action), dim=-1)
            context = self.history_cell(
                recurrent_input.reshape(batch * tokens, 2 * dim),
                context.reshape(batch * tokens, dim),
            ).reshape(batch, tokens, dim)
            context = self.history_spatial_refiner(context)

        # Retain the raw online token as the hard anchor so a zero delta takes
        # exactly the same single normalization path as persistence.  Deltas
        # are expressed in unit-latent coordinates and rescaled by the anchor
        # norm only for the absolute output reconstruction below.
        anchor = history[:, 2]
        belief = torch.cat((anchor, context), dim=-1)
        return OnlineH4Context(
            history_latents=history,
            belief_latents=belief,
        )

    def _predict_with_deltas(
        self,
        belief_latents: torch.Tensor,
        future_actions: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        expected = (self.spatial_token_count, 2 * self.config.feature_dim)
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

        dim = self.config.feature_dim
        tokens = self.spatial_token_count
        anchor, action_state = belief_latents.split(dim, dim=-1)
        predictions: list[torch.Tensor] = []
        deltas: list[torch.Tensor] = []
        for step in range(self.future_steps):
            action = self.action_embedding(future_actions[:, step])[:, None].expand(
                -1, tokens, -1
            )
            action_state = self.future_cell(
                action.reshape(batch * tokens, dim),
                action_state.reshape(batch * tokens, dim),
            ).reshape(batch, tokens, dim)
            action_state = self.future_spatial_refiner(action_state)
            delta = self.prediction_projector(action_state)
            deltas.append(delta)
            anchor_norm = anchor.norm(p=2.0, dim=-1, keepdim=True).clamp_min(
                self.config.normalization_epsilon
            )
            predictions.append(
                F.normalize(
                    anchor + anchor_norm * delta,
                    p=2.0,
                    dim=-1,
                    eps=self.config.normalization_epsilon,
                )
            )
        return torch.stack(predictions, dim=1), torch.stack(deltas, dim=1)

    def predict_from_belief(
        self,
        belief_latents: torch.Tensor,
        future_actions: torch.Tensor,
    ) -> torch.Tensor:
        """Return direct non-cumulative e2-to-horizon predictions."""

        predictions, _deltas = self._predict_with_deltas(
            belief_latents,
            future_actions,
        )
        return predictions

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

    def _variance_floor(self, history_latents: torch.Tensor) -> torch.Tensor:
        """V3 uses the fixed teacher rather than a variance auxiliary."""

        return history_latents.sum() * 0.0

    def forward(
        self,
        history_rgb: torch.Tensor,
        past_actions: torch.Tensor,
        future_actions: torch.Tensor,
        future_rgb: torch.Tensor | None = None,
    ) -> FixedTeacherDeltaH4JEPAOutput:
        """Run the online path and optionally score fixed-teacher futures."""

        context = self.encode_history(history_rgb, past_actions)
        predictions, predicted_deltas = self._predict_with_deltas(
            context.belief_latents,
            future_actions,
        )
        variance_loss = self._variance_floor(context.history_latents)

        targets: torch.Tensor | None = None
        per_sample: torch.Tensor | None = None
        per_horizon: torch.Tensor | None = None
        prediction_loss: torch.Tensor | None = None
        total_loss: torch.Tensor | None = None
        if future_rgb is not None:
            if future_rgb.shape[0] != history_rgb.shape[0]:
                raise ValueError("history_rgb and future_rgb batch sizes differ")
            targets = self.encode_target(future_rgb)
            per_sample = (predictions - targets).square().sum(dim=-1).mean(dim=-1)
            per_horizon = per_sample.mean(dim=0)
            prediction_loss = per_horizon.mean()
            # Retained as an API-compatible evaluation diagnostic only.  V3
            # training is supplied by the fixed-teacher auxiliary hook.
            total_loss = None

        return FixedTeacherDeltaH4JEPAOutput(
            predicted_latents=predictions,
            target_latents=targets,
            history_latents=context.history_latents,
            belief_latents=context.belief_latents,
            per_sample_horizon_loss=per_sample,
            per_horizon_loss=per_horizon,
            prediction_loss=prediction_loss,
            variance_loss=variance_loss,
            total_loss=total_loss,
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
        """Return only fixed-teacher history alignment and future-delta loss."""

        del past_actions, future_actions
        if not isinstance(output, FixedTeacherDeltaH4JEPAOutput):
            raise TypeError("output must be FixedTeacherDeltaH4JEPAOutput")
        if target_latents.shape != output.predicted_latents.shape:
            raise ValueError("target_latents shape differs from V3 predictions")

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
        """Reset the audit counter without ever copying online weights."""

        self.ema_update_count.zero_()
        self._freeze_target()

    @torch.no_grad()
    def update_target(self, momentum: float | None = None) -> None:
        """Validate the legacy call while leaving teacher and count untouched."""

        value = self.config.target_ema_momentum if momentum is None else momentum
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            raise TypeError("momentum must be a real number")
        value = float(value)
        if not math.isfinite(value) or not 0.0 <= value < 1.0:
            raise ValueError("momentum must lie in [0,1)")
        self._freeze_target()


__all__ = [
    "FixedTeacherDeltaH4JEPAOutput",
    "GO2_H4_PRIMITIVE_VOCABULARY",
    "JointRecurrentH4JEPA",
    "JointRecurrentH4JEPAConfig",
]
