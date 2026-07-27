"""Joint recurrent H4 JEPA over reset-safe RGB/action sequences.

The online path receives exactly three ordered RGB observations and the two
primitive actions between them.  A token-wise recurrent belief preserves the
encoder's spatial token lattice.  One shared action-conditioned predictor is
then unrolled over four future primitive actions.  Future RGB is optional and
is visible only to a frozen EMA target encoder under ``torch.no_grad``.

The default visual encoder geometry matches the qualified N320 encoder:
112x112 RGB, 7x7 patches, 192 features, depth 6, and 6 attention heads.  This
module never opens a checkpoint; callers may pass an already validated N320
encoder state mapping to the constructor.
"""
from __future__ import annotations

import copy
import math
from dataclasses import dataclass
from typing import Mapping

import torch
import torch.nn as nn
import torch.nn.functional as F

from .encoders import ViTBlock, VisionEncoder


GO2_H4_PRIMITIVE_VOCABULARY = (
    "arc_left",
    "arc_right",
    "backward",
    "forward_fast",
    "forward_medium",
    "forward_slow",
    "hold",
    "yaw_left",
    "yaw_right",
)


@dataclass(frozen=True)
class JointRecurrentH4JEPAConfig:
    """Constructor contract for :class:`JointRecurrentH4JEPA`.

    Non-default visual sizes are useful for source-only synthetic tests.  An
    actual N320 state is compatible with the default visual fields only.
    """

    image_size: int = 112
    patch_size: int = 7
    feature_dim: int = 192
    encoder_depth: int = 6
    encoder_heads: int = 6
    encoder_mlp_ratio: int = 4
    recurrent_spatial_heads: int = 6
    recurrent_spatial_mlp_ratio: int = 2
    dropout: float = 0.0
    target_ema_momentum: float = 0.996
    normalization_epsilon: float = 1e-6
    variance_weight: float = 0.1
    variance_target_std: float | None = None
    action_vocabulary: tuple[str, ...] = GO2_H4_PRIMITIVE_VOCABULARY

    def __post_init__(self) -> None:
        integer_fields = (
            "image_size",
            "patch_size",
            "feature_dim",
            "encoder_depth",
            "encoder_heads",
            "encoder_mlp_ratio",
            "recurrent_spatial_heads",
            "recurrent_spatial_mlp_ratio",
        )
        for name in integer_fields:
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if self.image_size % self.patch_size != 0:
            raise ValueError("image_size must be divisible by patch_size")
        if self.feature_dim % self.encoder_heads != 0:
            raise ValueError("feature_dim must be divisible by encoder_heads")
        if self.feature_dim % self.recurrent_spatial_heads != 0:
            raise ValueError(
                "feature_dim must be divisible by recurrent_spatial_heads"
            )
        if not math.isfinite(self.dropout) or not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must lie in [0,1)")
        if (
            not math.isfinite(self.target_ema_momentum)
            or not 0.0 <= self.target_ema_momentum < 1.0
        ):
            raise ValueError("target_ema_momentum must lie in [0,1)")
        if (
            not math.isfinite(self.normalization_epsilon)
            or self.normalization_epsilon <= 0.0
        ):
            raise ValueError("normalization_epsilon must be positive")
        if not math.isfinite(self.variance_weight) or self.variance_weight < 0.0:
            raise ValueError("variance_weight must be non-negative")
        if self.variance_target_std is not None and (
            not math.isfinite(self.variance_target_std)
            or self.variance_target_std <= 0.0
        ):
            raise ValueError("variance_target_std must be positive or None")
        vocabulary = tuple(self.action_vocabulary)
        if not vocabulary or len(set(vocabulary)) != len(vocabulary):
            raise ValueError("action_vocabulary must be nonempty and unique")
        if any(not isinstance(value, str) or not value for value in vocabulary):
            raise ValueError("action_vocabulary entries must be nonempty strings")
        object.__setattr__(self, "action_vocabulary", vocabulary)

    @property
    def action_count(self) -> int:
        return len(self.action_vocabulary)

    @property
    def spatial_token_count(self) -> int:
        return (self.image_size // self.patch_size) ** 2

    @property
    def effective_variance_target_std(self) -> float:
        if self.variance_target_std is not None:
            return float(self.variance_target_std)
        return 1.0 / math.sqrt(float(self.feature_dim))


@dataclass(frozen=True)
class OnlineH4Context:
    """Online spatial history and its final causal belief state."""

    history_latents: torch.Tensor
    belief_latents: torch.Tensor


@dataclass(frozen=True)
class JointRecurrentH4JEPAOutput:
    """Model outputs and optional loss terms for one H4 batch."""

    predicted_latents: torch.Tensor
    target_latents: torch.Tensor | None
    history_latents: torch.Tensor
    belief_latents: torch.Tensor
    per_sample_horizon_loss: torch.Tensor | None
    per_horizon_loss: torch.Tensor | None
    prediction_loss: torch.Tensor | None
    variance_loss: torch.Tensor
    total_loss: torch.Tensor | None


def _validate_encoder_state(
    encoder: VisionEncoder,
    state: Mapping[str, torch.Tensor],
) -> None:
    expected = encoder.state_dict()
    if set(state) != set(expected):
        missing = sorted(set(expected) - set(state))
        extra = sorted(set(state) - set(expected))
        raise ValueError(
            f"N320 encoder state keys changed; missing={missing}, extra={extra}"
        )
    for name, expected_tensor in expected.items():
        value = state[name]
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"N320 encoder state {name!r} is not a tensor")
        if value.shape != expected_tensor.shape:
            raise ValueError(
                f"N320 encoder state {name!r} has shape {tuple(value.shape)}, "
                f"expected {tuple(expected_tensor.shape)}"
            )
        if value.dtype != torch.float32:
            raise TypeError(f"N320 encoder state {name!r} must be float32")
        if not bool(torch.isfinite(value).all()):
            raise FloatingPointError(f"N320 encoder state {name!r} is nonfinite")


class JointRecurrentH4JEPA(nn.Module):
    """RGB-only joint recurrent JEPA for three-history/four-future H4.

    Action tensors contain indices into ``config.action_vocabulary``.  The
    online encoder, history recurrence, future recurrence, refiners, action
    embedding, and prediction projector are jointly trainable.  The target
    encoder is never trainable and is updated only through :meth:`update_target`.
    """

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
            raise TypeError("config must be JointRecurrentH4JEPAConfig")

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
        nn.init.normal_(self.action_embedding.weight, std=0.02)

        self.initial_belief = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim),
            nn.Tanh(),
        )
        self.history_observation_norm = nn.LayerNorm(dim)
        self.history_cell = nn.GRUCell(input_size=2 * dim, hidden_size=dim)
        self.history_spatial_refiner = ViTBlock(
            hidden_dim=dim,
            n_heads=self.config.recurrent_spatial_heads,
            mlp_ratio=self.config.recurrent_spatial_mlp_ratio,
            dropout=self.config.dropout,
        )

        # These modules are shared across all four future steps.
        self.future_cell = nn.GRUCell(input_size=dim, hidden_size=dim)
        self.future_spatial_refiner = ViTBlock(
            hidden_dim=dim,
            n_heads=self.config.recurrent_spatial_heads,
            mlp_ratio=self.config.recurrent_spatial_mlp_ratio,
            dropout=self.config.dropout,
        )
        self.prediction_projector = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim),
        )

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
        expected = (
            steps,
            3,
            self.config.image_size,
            self.config.image_size,
        )
        if rgb.ndim != 5 or tuple(rgb.shape[1:]) != expected:
            raise ValueError(
                f"{name} must have shape (B,{steps},3,"
                f"{self.config.image_size},{self.config.image_size})"
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
    ) -> OnlineH4Context:
        """Encode ``e0,p0,e1,p1,e2`` into one spatial belief state."""

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
        hidden = self.initial_belief(history[:, 0])
        tokens = self.spatial_token_count
        dim = self.config.feature_dim
        for step in range(self.past_action_steps):
            observation = self.history_observation_norm(history[:, step + 1])
            action = self.action_embedding(past_actions[:, step])[:, None].expand(
                -1, tokens, -1
            )
            recurrent_input = torch.cat((observation, action), dim=-1)
            hidden = self.history_cell(
                recurrent_input.reshape(batch * tokens, 2 * dim),
                hidden.reshape(batch * tokens, dim),
            ).reshape(batch, tokens, dim)
            hidden = self.history_spatial_refiner(hidden)
        return OnlineH4Context(
            history_latents=history,
            belief_latents=hidden,
        )

    def predict_from_belief(
        self,
        belief_latents: torch.Tensor,
        future_actions: torch.Tensor,
    ) -> torch.Tensor:
        """Unroll the shared predictor over ``p2,p3,p4,p5``.

        This method is also the counterfactual-action API: callers can reuse a
        fixed belief with alternative future action sequences without
        re-encoding RGB.
        """

        expected_belief = (
            self.spatial_token_count,
            self.config.feature_dim,
        )
        if belief_latents.ndim != 3 or tuple(belief_latents.shape[1:]) != expected_belief:
            raise ValueError(
                "belief_latents must have shape "
                f"(B,{expected_belief[0]},{expected_belief[1]})"
            )
        batch = int(belief_latents.shape[0])
        # An outer autocast context may make this online intermediate bf16/fp16
        # even though runner-owned RGB inputs remain exact float32.
        if not torch.is_floating_point(belief_latents):
            raise TypeError("belief_latents must have a floating dtype")
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

        tokens = self.spatial_token_count
        dim = self.config.feature_dim
        hidden = belief_latents
        predictions: list[torch.Tensor] = []
        for step in range(self.future_steps):
            action = self.action_embedding(future_actions[:, step])[:, None].expand(
                -1, tokens, -1
            )
            hidden = self.future_cell(
                action.reshape(batch * tokens, dim),
                hidden.reshape(batch * tokens, dim),
            ).reshape(batch, tokens, dim)
            hidden = self.future_spatial_refiner(hidden)
            projected = self.prediction_projector(hidden)
            predictions.append(
                F.normalize(
                    projected,
                    p=2.0,
                    dim=-1,
                    eps=self.config.normalization_epsilon,
                )
            )
        return torch.stack(predictions, dim=1)

    @torch.no_grad()
    def encode_target(self, future_rgb: torch.Tensor) -> torch.Tensor:
        """Encode ``e3,e4,e5,e6`` with the detached EMA target encoder."""

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

    def _variance_floor(self, history_latents: torch.Tensor) -> torch.Tensor:
        normalized = F.normalize(
            history_latents,
            p=2.0,
            dim=-1,
            eps=self.config.normalization_epsilon,
        )
        # Preserve spatial identity: variance is measured across batch and time
        # independently at each patch position and feature dimension.
        std = torch.sqrt(
            normalized.float().var(dim=(0, 1), unbiased=False) + 1e-4
        )
        return F.relu(self.config.effective_variance_target_std - std).mean().to(
            history_latents.dtype
        )

    def forward(
        self,
        history_rgb: torch.Tensor,
        past_actions: torch.Tensor,
        future_actions: torch.Tensor,
        future_rgb: torch.Tensor | None = None,
    ) -> JointRecurrentH4JEPAOutput:
        """Run the online H4 rollout and optionally construct JEPA targets.

        ``future_rgb`` never enters the online call graph.  When it is absent,
        the returned target and prediction-loss fields are ``None`` and the
        result is suitable for inference or counterfactual scoring.
        """

        context = self.encode_history(history_rgb, past_actions)
        predictions = self.predict_from_belief(
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
            # Unit-normalized token energy is bounded by [0,4].
            per_sample = (predictions - targets).square().sum(dim=-1).mean(dim=-1)
            per_horizon = per_sample.mean(dim=0)
            prediction_loss = per_horizon.mean()
            total_loss = (
                prediction_loss + self.config.variance_weight * variance_loss
            )

        return JointRecurrentH4JEPAOutput(
            predicted_latents=predictions,
            target_latents=targets,
            history_latents=context.history_latents,
            belief_latents=context.belief_latents,
            per_sample_horizon_loss=per_sample,
            per_horizon_loss=per_horizon,
            prediction_loss=prediction_loss,
            variance_loss=variance_loss,
            total_loss=total_loss,
        )

    @torch.no_grad()
    def hard_sync_target(self) -> None:
        """Copy the online encoder exactly into the target encoder."""

        self.target_encoder.load_state_dict(self.encoder.state_dict(), strict=True)
        self.ema_update_count.zero_()
        self._freeze_target()

    @torch.no_grad()
    def update_target(self, momentum: float | None = None) -> None:
        """EMA-update the target after one successful optimizer step."""

        value = self.config.target_ema_momentum if momentum is None else momentum
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            raise TypeError("momentum must be a real number")
        value = float(value)
        if not math.isfinite(value) or not 0.0 <= value < 1.0:
            raise ValueError("momentum must lie in [0,1)")
        online_parameters = dict(self.encoder.named_parameters())
        target_parameters = dict(self.target_encoder.named_parameters())
        if online_parameters.keys() != target_parameters.keys():
            raise RuntimeError("online and target encoder parameter inventories differ")
        for name, online in online_parameters.items():
            target_parameters[name].mul_(value).add_(online, alpha=1.0 - value)

        online_buffers = dict(self.encoder.named_buffers())
        target_buffers = dict(self.target_encoder.named_buffers())
        if online_buffers.keys() != target_buffers.keys():
            raise RuntimeError("online and target encoder buffer inventories differ")
        for name, online in online_buffers.items():
            target_buffers[name].copy_(online)
        self.ema_update_count.add_(1)
        self._freeze_target()


__all__ = [
    "GO2_H4_PRIMITIVE_VOCABULARY",
    "JointRecurrentH4JEPA",
    "JointRecurrentH4JEPAConfig",
    "JointRecurrentH4JEPAOutput",
    "OnlineH4Context",
]
