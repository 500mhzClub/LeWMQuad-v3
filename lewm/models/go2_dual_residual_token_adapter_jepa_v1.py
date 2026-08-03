"""Residual token-adapter JEPA for the matched-branch capacity screen.

The pretrained image encoders are not part of this module.  It operates only
on their bound, frozen 16 by 16 token caches.  The action-conditioned
predictor is constructed before either adapter so that its seeded
initialization remains identical to the predecessor screen.
"""
from __future__ import annotations

import copy
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from lewm.models.go2_matched_branch_successor_screen_v1 import (
    DenseActionConditionedPredictorV1,
)


__all__ = (
    "JointResidualTokenAdapterJEPAV1",
    "ResidualSpatialTokenAdapterBlockV1",
    "ResidualSpatialTokenAdapterV1",
)


_ACTION_COUNT = 9
_BOTTLENECK_DIM = 64
_CONTEXT_STEPS = 3
_EMA_MOMENTUM = 0.996
_GRID_SIZE = 16
_HIDDEN_DIM = 128
_LAYER_NORM_EPSILON = 1.0e-5
_NORMALIZATION_EPSILON = 1.0e-12
_RESIDUAL_SCALE = 0.125
_TOKEN_COUNT = _GRID_SIZE * _GRID_SIZE


def _require_feature_dim(feature_dim: int) -> int:
    if type(feature_dim) is not int or feature_dim < 1:
        raise ValueError("feature_dim must be a positive integer")
    return feature_dim


def _require_token_tensor(
    tokens: torch.Tensor,
    *,
    feature_dim: int,
    device: torch.device,
    label: str,
    rank: int | None = None,
) -> None:
    if not isinstance(tokens, torch.Tensor):
        raise TypeError(f"{label} must be a tensor")
    if rank is not None and tokens.ndim != rank:
        raise ValueError(f"{label} must have rank {rank}")
    if tokens.ndim < 3 or tuple(tokens.shape[-2:]) != (
        _TOKEN_COUNT,
        feature_dim,
    ):
        raise ValueError(
            f"{label} must end with shape ({_TOKEN_COUNT},{feature_dim})"
        )
    if any(size < 1 for size in tokens.shape[:-2]):
        raise ValueError(f"{label} must have nonempty leading dimensions")
    if tokens.dtype != torch.float32:
        raise TypeError(f"{label} must use exact float32")
    if tokens.device != device:
        raise TypeError(f"{label} and adapter parameters must share one device")
    if not bool(torch.isfinite(tokens).all()):
        raise FloatingPointError(f"{label} contains a nonfinite value")
    norms = torch.linalg.vector_norm(tokens, ord=2, dim=-1)
    if bool((norms <= 0.0).any()) or not bool(torch.isfinite(norms).all()):
        raise FloatingPointError(f"{label} contains a zero or nonfinite token")


class ResidualSpatialTokenAdapterBlockV1(nn.Module):
    """One exact bounded residual spatial-adapter block."""

    def __init__(self, *, feature_dim: int) -> None:
        super().__init__()
        self.feature_dim = _require_feature_dim(feature_dim)
        self.bottleneck_dim = _BOTTLENECK_DIM
        self.norm = nn.LayerNorm(
            self.feature_dim,
            eps=_LAYER_NORM_EPSILON,
            elementwise_affine=True,
            bias=True,
        )
        self.input_projection = nn.Linear(
            self.feature_dim,
            self.bottleneck_dim,
            bias=True,
        )
        self.activation = nn.GELU(approximate="none")
        self.depthwise = nn.Conv2d(
            self.bottleneck_dim,
            self.bottleneck_dim,
            kernel_size=3,
            stride=1,
            padding=1,
            dilation=1,
            groups=self.bottleneck_dim,
            bias=True,
        )
        self.channel_mixing = nn.Conv2d(
            self.bottleneck_dim,
            self.bottleneck_dim,
            kernel_size=1,
            stride=1,
            padding=0,
            dilation=1,
            groups=1,
            bias=True,
        )
        self.output_projection = nn.Linear(
            self.bottleneck_dim,
            self.feature_dim,
            bias=True,
        )
        nn.init.zeros_(self.output_projection.weight)
        nn.init.zeros_(self.output_projection.bias)

    def bounded_update(self, tokens: torch.Tensor) -> torch.Tensor:
        """Return the preregistered per-token update with norm below 0.125."""

        reference = self.norm.weight
        if reference.dtype != torch.float32:
            raise TypeError("adapter parameters must remain float32")
        _require_token_tensor(
            tokens,
            feature_dim=self.feature_dim,
            device=reference.device,
            label="block tokens",
            rank=3,
        )
        batch = tokens.shape[0]
        hidden = self.activation(self.input_projection(self.norm(tokens)))
        grid = hidden.transpose(1, 2).reshape(
            batch,
            self.bottleneck_dim,
            _GRID_SIZE,
            _GRID_SIZE,
        )
        grid = self.channel_mixing(self.depthwise(grid))
        hidden = grid.flatten(2).transpose(1, 2)
        raw = self.output_projection(hidden)
        raw_norm = torch.linalg.vector_norm(
            raw,
            ord=2,
            dim=-1,
            keepdim=True,
        )
        update = _RESIDUAL_SCALE * raw / (1.0 + raw_norm)
        if not bool(torch.isfinite(update).all()):
            raise FloatingPointError("bounded adapter update became nonfinite")
        return update

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        update = self.bounded_update(tokens)
        result = F.normalize(
            tokens + update,
            p=2.0,
            dim=-1,
            eps=_NORMALIZATION_EPSILON,
        )
        if not bool(torch.isfinite(result).all()):
            raise FloatingPointError("adapter block output became nonfinite")
        return result


class ResidualSpatialTokenAdapterV1(nn.Module):
    """Two exact residual spatial blocks over native-width cached tokens."""

    def __init__(self, *, feature_dim: int) -> None:
        super().__init__()
        self.feature_dim = _require_feature_dim(feature_dim)
        self.blocks = nn.ModuleList(
            (
                ResidualSpatialTokenAdapterBlockV1(feature_dim=self.feature_dim),
                ResidualSpatialTokenAdapterBlockV1(feature_dim=self.feature_dim),
            )
        )

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        reference = self.blocks[0].norm.weight
        if reference.dtype != torch.float32:
            raise TypeError("adapter parameters must remain float32")
        _require_token_tensor(
            tokens,
            feature_dim=self.feature_dim,
            device=reference.device,
            label="adapter tokens",
        )
        leading_shape = tokens.shape[:-2]
        adapted = tokens.reshape(-1, _TOKEN_COUNT, self.feature_dim)
        for block in self.blocks:
            adapted = block(adapted)
        result = adapted.reshape(*leading_shape, _TOKEN_COUNT, self.feature_dim)
        if not bool(torch.isfinite(result).all()):
            raise FloatingPointError("adapter output became nonfinite")
        return result


class JointResidualTokenAdapterJEPAV1(nn.Module):
    """Action-conditioned predecessor predictor with online and EMA adapters."""

    def __init__(self, *, feature_dim: int) -> None:
        super().__init__()
        self.feature_dim = _require_feature_dim(feature_dim)

        # This must remain the first RNG-consuming construction in the model.
        self.predictor = DenseActionConditionedPredictorV1(
            feature_dim=self.feature_dim,
            hidden_dim=_HIDDEN_DIM,
            action_count=_ACTION_COUNT,
        )
        self.online_adapter = ResidualSpatialTokenAdapterV1(
            feature_dim=self.feature_dim
        )
        self.target_adapter = copy.deepcopy(self.online_adapter)
        self.register_buffer(
            "ema_update_count",
            torch.zeros((), dtype=torch.long),
            persistent=True,
        )
        self._freeze_target_adapter()

    def _freeze_target_adapter(self) -> None:
        self.target_adapter.requires_grad_(False)
        self.target_adapter.eval()

    def train(self, mode: bool = True) -> JointResidualTokenAdapterJEPAV1:
        if not isinstance(mode, bool):
            raise TypeError("mode must be a bool")
        super().train(mode)
        self._freeze_target_adapter()
        return self

    def _ema_parameter_pairs(
        self,
    ) -> tuple[tuple[torch.Tensor, torch.Tensor], ...]:
        online = dict(self.online_adapter.named_parameters())
        target = dict(self.target_adapter.named_parameters())
        if online.keys() != target.keys():
            raise RuntimeError("online and target adapter inventories differ")
        return tuple((online[name], target[name]) for name in online)

    def adapt_online(self, tokens: torch.Tensor) -> torch.Tensor:
        """Adapt cached observations through the trainable online route."""

        return self.online_adapter(tokens)

    @torch.no_grad()
    def adapt_target(self, tokens: torch.Tensor) -> torch.Tensor:
        """Adapt successor observations through the detached EMA route."""

        self._freeze_target_adapter()
        return self.target_adapter(tokens).detach()

    def predict_from_adapted_context(
        self,
        adapted_context: torch.Tensor,
        history_actions: torch.Tensor,
        candidate_action: torch.Tensor,
    ) -> torch.Tensor:
        """Predict native-width successor tokens from already-adapted context."""

        return self.predictor(
            adapted_context,
            history_actions,
            candidate_action,
        )

    @torch.no_grad()
    def update_target_ema_(self, momentum: float) -> None:
        """Apply the sole allowed target update, in place, after optimization."""

        if type(momentum) is not float or not math.isfinite(momentum):
            raise TypeError("EMA momentum must be one finite float")
        if momentum != _EMA_MOMENTUM:
            raise ValueError("EMA momentum must remain exactly 0.996")
        for online, target in self._ema_parameter_pairs():
            target.mul_(momentum).add_(online, alpha=1.0 - momentum)
        self.ema_update_count.add_(1)
        self._freeze_target_adapter()

    @torch.no_grad()
    def ema_update_(self, momentum: float) -> None:
        """Alias for callers using the shorter in-place EMA name."""

        self.update_target_ema_(momentum)

    def forward(
        self,
        context: torch.Tensor,
        history_actions: torch.Tensor,
        candidate_action: torch.Tensor,
    ) -> torch.Tensor:
        if context.ndim != 4 or tuple(context.shape[1:3]) != (
            _CONTEXT_STEPS,
            _TOKEN_COUNT,
        ):
            raise ValueError("context must have shape (B,3,256,D)")
        adapted = self.adapt_online(context)
        prediction = self.predict_from_adapted_context(
            adapted,
            history_actions,
            candidate_action,
        )
        if not bool(torch.isfinite(prediction).all()):
            raise FloatingPointError("joint adapter prediction became nonfinite")
        return prediction
