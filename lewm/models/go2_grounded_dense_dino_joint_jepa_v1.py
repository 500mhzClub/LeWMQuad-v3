"""Production model for the grounded dense-DINO joint-JEPA V1 experiment.

The filesystem-facing runner owns the frozen DINO trunk.  This module begins
at the exact trunk boundary: full class-plus-patch token sequences produced by
DINOv2 block 9.  It owns the trainable online blocks 10--11 and final norm, a
detached EMA target copy, the existing action-conditioned dense predictor, and
the embodiment-grounded physical outcome head.

True-successor trunk tokens are accepted only by :meth:`encode_target`.  The
inference ``forward`` signature has no future observation or target input.
"""
from __future__ import annotations

from collections.abc import Sequence
from copy import deepcopy
from numbers import Real
from typing import NamedTuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from lewm.models.dense_dinov2_temporal_predictor import (
    DenseDINOv2TemporalPredictorV1,
)


FEATURE_DIM = 384
FULL_TOKEN_COUNT = 257
PATCH_TOKEN_COUNT = 256
CONTEXT_STEPS = 3
HISTORY_STEPS = 2
ACTION_DIM = 15
PHYSICAL_INPUT_DIM = 12
PHYSICAL_OUTPUT_DIM = 4
PHYSICAL_HEAD_WIDTH = 128
ONLINE_TAIL_BLOCK_COUNT = 2
DEFAULT_INITIALIZATION_SEED = 2_026_080_405
DEFAULT_EMA_MOMENTUM = 0.996


__all__ = (
    "DEFAULT_EMA_MOMENTUM",
    "DEFAULT_INITIALIZATION_SEED",
    "DINOv2TrainableTailV1",
    "DenseRelationalPhysicalHeadOutputV1",
    "DenseRelationalPhysicalHeadV1",
    "GroundedBatchPredictionV1",
    "GroundedDenseDINOJointJEPAV1",
)


def _validate_initialization_seed(value: int) -> int:
    if type(value) is not int or not 0 <= value < 2**63:
        raise ValueError("initialization_seed must be an integer in [0, 2**63)")
    return value


def _validate_float_tensor(
    value: object,
    *,
    shape: tuple[int, ...],
    device: torch.device,
    name: str,
) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if tuple(value.shape) != shape:
        raise ValueError(f"{name} must have shape {shape}")
    if value.dtype != torch.float32:
        raise TypeError(f"{name} must use exact torch.float32")
    if value.device != device:
        raise TypeError(f"{name} and model state must share one device")
    if not bool(torch.isfinite(value).all()):
        raise FloatingPointError(f"{name} contains a nonfinite value")
    return value


def _module_device(module: nn.Module) -> torch.device:
    devices = {
        tensor.device
        for tensor in (*tuple(module.parameters()), *tuple(module.buffers()))
    }
    if len(devices) > 1:
        raise RuntimeError("module state spans multiple devices")
    return next(iter(devices), torch.device("cpu"))


class DINOv2TrainableTailV1(nn.Module):
    """Exactly two DINOv2 blocks followed by the DINO final norm.

    The wrapper consumes the full class-plus-patch sequence produced by the
    frozen block-0--9 trunk.  ``forward`` retains the class token, while
    :meth:`patch_tokens` returns unit-normalized patch tokens for prediction.
    """

    def __init__(self, blocks: Sequence[nn.Module], norm: nn.Module) -> None:
        super().__init__()
        if isinstance(blocks, (str, bytes)) or len(blocks) != ONLINE_TAIL_BLOCK_COUNT:
            raise ValueError("online DINO tail requires exactly two blocks")
        if any(not isinstance(block, nn.Module) for block in blocks):
            raise TypeError("each online DINO tail block must be an nn.Module")
        if not isinstance(norm, nn.Module):
            raise TypeError("online DINO final norm must be an nn.Module")
        self.blocks = nn.ModuleList(tuple(blocks))
        self.norm = norm
        self.requires_grad_(True)

    def forward(self, trunk_tokens: torch.Tensor) -> torch.Tensor:
        if not isinstance(trunk_tokens, torch.Tensor):
            raise TypeError("trunk_tokens must be a torch.Tensor")
        if (
            trunk_tokens.ndim != 3
            or trunk_tokens.shape[0] < 1
            or tuple(trunk_tokens.shape[1:]) != (FULL_TOKEN_COUNT, FEATURE_DIM)
        ):
            raise ValueError("trunk_tokens must have shape [F,257,384] with F >= 1")
        if trunk_tokens.dtype != torch.float32:
            raise TypeError("trunk_tokens must use exact torch.float32")
        if trunk_tokens.device != _module_device(self):
            raise TypeError("trunk_tokens and DINO tail must share one device")
        if not bool(torch.isfinite(trunk_tokens).all()):
            raise FloatingPointError("trunk_tokens contain a nonfinite value")

        hidden = trunk_tokens
        for block in self.blocks:
            hidden = block(hidden)
            if not isinstance(hidden, torch.Tensor) or tuple(hidden.shape) != tuple(
                trunk_tokens.shape
            ):
                raise RuntimeError("DINO tail block output contract changed")
        hidden = self.norm(hidden)
        if (
            not isinstance(hidden, torch.Tensor)
            or tuple(hidden.shape) != tuple(trunk_tokens.shape)
            or hidden.dtype != torch.float32
            or hidden.device != trunk_tokens.device
        ):
            raise RuntimeError("DINO tail norm output contract changed")
        if not bool(torch.isfinite(hidden).all()):
            raise FloatingPointError("DINO tail produced a nonfinite value")
        return hidden

    def patch_tokens(self, trunk_tokens: torch.Tensor) -> torch.Tensor:
        patches = self(trunk_tokens)[:, 1:]
        result = F.normalize(patches, p=2.0, dim=-1, eps=1.0e-8)
        if tuple(result.shape[1:]) != (PATCH_TOKEN_COUNT, FEATURE_DIM):
            raise RuntimeError("DINO patch-token contract changed")
        return result


class DenseRelationalPhysicalHeadOutputV1(NamedTuple):
    standardized_residuals: torch.Tensor
    attention: torch.Tensor


class DenseRelationalPhysicalHeadV1(nn.Module):
    """Conditioned nonlinear readout over dense successor relations."""

    def __init__(self, width: int = PHYSICAL_HEAD_WIDTH) -> None:
        super().__init__()
        if type(width) is not int or width < 1:
            raise ValueError("physical head width must be a positive integer")
        self.width = width
        relational_width = 3 * FEATURE_DIM
        self.relational_norm = nn.LayerNorm(relational_width)
        self.relational_projection = nn.Linear(relational_width, width)
        self.value_projection = nn.Linear(relational_width, width)
        self.condition_projection = nn.Sequential(
            nn.LayerNorm(PHYSICAL_INPUT_DIM),
            nn.Linear(PHYSICAL_INPUT_DIM, width),
            nn.SiLU(),
            nn.Linear(width, width),
        )
        self.position_embedding = nn.Parameter(
            torch.empty(1, PATCH_TOKEN_COUNT, width)
        )
        self.attention_projection = nn.Linear(width, 1, bias=False)
        self.output_hidden = nn.Sequential(
            nn.LayerNorm(width),
            nn.Linear(width, width),
            nn.SiLU(),
        )
        self.output_projection = nn.Linear(width, PHYSICAL_OUTPUT_DIM)
        nn.init.trunc_normal_(self.position_embedding, std=0.02)
        # Zero standardized residual is decoded by the runner as the exact
        # train-only action mean at update zero.
        nn.init.zeros_(self.output_projection.weight)
        nn.init.zeros_(self.output_projection.bias)

    def forward(
        self,
        current_tokens: torch.Tensor,
        predicted_tokens: torch.Tensor,
        physical_inputs: torch.Tensor,
    ) -> DenseRelationalPhysicalHeadOutputV1:
        if not isinstance(current_tokens, torch.Tensor):
            raise TypeError("current_tokens must be a torch.Tensor")
        if current_tokens.ndim != 3 or current_tokens.shape[0] < 1:
            raise ValueError("current_tokens must have shape [N,256,384] with N >= 1")
        batch = int(current_tokens.shape[0])
        device = self.position_embedding.device
        current = _validate_float_tensor(
            current_tokens,
            shape=(batch, PATCH_TOKEN_COUNT, FEATURE_DIM),
            device=device,
            name="current_tokens",
        )
        predicted = _validate_float_tensor(
            predicted_tokens,
            shape=(batch, PATCH_TOKEN_COUNT, FEATURE_DIM),
            device=device,
            name="predicted_tokens",
        )
        condition_input = _validate_float_tensor(
            physical_inputs,
            shape=(batch, PHYSICAL_INPUT_DIM),
            device=device,
            name="physical_inputs",
        )

        relational = torch.cat((current, predicted, predicted - current), dim=-1)
        normalized = self.relational_norm(relational)
        condition = self.condition_projection(condition_input)
        attention_hidden = torch.tanh(
            self.relational_projection(normalized)
            + self.position_embedding
            + condition.unsqueeze(1)
        )
        attention = torch.softmax(
            self.attention_projection(attention_hidden).squeeze(-1), dim=-1
        )
        values = self.value_projection(normalized)
        pooled = torch.sum(attention.unsqueeze(-1) * values, dim=1)
        residuals = self.output_projection(self.output_hidden(pooled + condition))
        if (
            tuple(residuals.shape) != (batch, PHYSICAL_OUTPUT_DIM)
            or tuple(attention.shape) != (batch, PATCH_TOKEN_COUNT)
            or not bool(torch.isfinite(residuals).all())
            or not bool(torch.isfinite(attention).all())
        ):
            raise FloatingPointError("dense physical head output is invalid")
        return DenseRelationalPhysicalHeadOutputV1(residuals, attention)


class GroundedBatchPredictionV1(NamedTuple):
    successor_tokens: torch.Tensor
    standardized_physical_residuals: torch.Tensor
    physical_attention: torch.Tensor


class GroundedDenseDINOJointJEPAV1(nn.Module):
    """Identical production model instantiated for both learned arms."""

    def __init__(
        self,
        online_tail_blocks: Sequence[nn.Module],
        online_tail_norm: nn.Module,
        *,
        initialization_seed: int = DEFAULT_INITIALIZATION_SEED,
        ema_momentum: float = DEFAULT_EMA_MOMENTUM,
    ) -> None:
        super().__init__()
        seed = _validate_initialization_seed(initialization_seed)
        if (
            isinstance(ema_momentum, bool)
            or not isinstance(ema_momentum, Real)
            or not 0.0 <= float(ema_momentum) < 1.0
        ):
            raise ValueError("ema_momentum must be a real number in [0,1)")
        self.initialization_seed = seed
        self.ema_momentum = float(ema_momentum)
        self.online_tail = DINOv2TrainableTailV1(
            online_tail_blocks, online_tail_norm
        )
        self.target_tail = deepcopy(self.online_tail)
        self.target_tail.requires_grad_(False)
        self.target_tail.eval()

        # Arm construction does not consume or depend on ambient CPU RNG.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            self.predictor = DenseDINOv2TemporalPredictorV1()
            self.physical_head = DenseRelationalPhysicalHeadV1()

    def train(self, mode: bool = True) -> GroundedDenseDINOJointJEPAV1:
        if not isinstance(mode, bool):
            raise TypeError("training mode must be a bool")
        super().train(mode)
        # ``Module.train`` recurses into every child; the EMA target must never
        # acquire train-mode stochastic behavior.
        self.target_tail.eval()
        return self

    def _validate_inference_inputs(
        self,
        context_trunk_tokens: torch.Tensor,
        history_commands: torch.Tensor,
        candidate_commands: torch.Tensor,
        physical_inputs: torch.Tensor,
    ) -> tuple[int, int]:
        if not isinstance(context_trunk_tokens, torch.Tensor):
            raise TypeError("context_trunk_tokens must be a torch.Tensor")
        if context_trunk_tokens.ndim != 4 or context_trunk_tokens.shape[0] < 1:
            raise ValueError(
                "context_trunk_tokens must have shape [B,3,257,384] with B >= 1"
            )
        batch = int(context_trunk_tokens.shape[0])
        if not isinstance(candidate_commands, torch.Tensor) or candidate_commands.ndim != 3:
            raise ValueError("candidate_commands must have shape [B,A,15]")
        action_count = int(candidate_commands.shape[1])
        if action_count < 1:
            raise ValueError("candidate_commands requires at least one action")
        device = self.physical_head.position_embedding.device
        _validate_float_tensor(
            context_trunk_tokens,
            shape=(batch, CONTEXT_STEPS, FULL_TOKEN_COUNT, FEATURE_DIM),
            device=device,
            name="context_trunk_tokens",
        )
        _validate_float_tensor(
            history_commands,
            shape=(batch, HISTORY_STEPS, ACTION_DIM),
            device=device,
            name="history_commands",
        )
        _validate_float_tensor(
            candidate_commands,
            shape=(batch, action_count, ACTION_DIM),
            device=device,
            name="candidate_commands",
        )
        _validate_float_tensor(
            physical_inputs,
            shape=(batch, action_count, PHYSICAL_INPUT_DIM),
            device=device,
            name="physical_inputs",
        )
        if _module_device(self.online_tail) != device:
            raise RuntimeError("online DINO tail and grounded model state differ in device")
        return batch, action_count

    def forward(
        self,
        context_trunk_tokens: torch.Tensor,
        history_commands: torch.Tensor,
        candidate_commands: torch.Tensor,
        physical_inputs: torch.Tensor,
    ) -> GroundedBatchPredictionV1:
        """Predict all requested branches without accepting a future input."""

        batch, action_count = self._validate_inference_inputs(
            context_trunk_tokens,
            history_commands,
            candidate_commands,
            physical_inputs,
        )
        context = self.online_tail.patch_tokens(
            context_trunk_tokens.reshape(
                batch * CONTEXT_STEPS, FULL_TOKEN_COUNT, FEATURE_DIM
            )
        ).reshape(batch, CONTEXT_STEPS, PATCH_TOKEN_COUNT, FEATURE_DIM)
        branch_context = (
            context[:, None]
            .expand(-1, action_count, -1, -1, -1)
            .reshape(
                batch * action_count,
                CONTEXT_STEPS,
                PATCH_TOKEN_COUNT,
                FEATURE_DIM,
            )
            .contiguous()
        )
        branch_history = (
            history_commands[:, None]
            .expand(-1, action_count, -1, -1)
            .reshape(batch * action_count, HISTORY_STEPS, ACTION_DIM)
            .contiguous()
        )
        flat_candidates = candidate_commands.reshape(
            batch * action_count, ACTION_DIM
        ).contiguous()
        successor = self.predictor(
            branch_context, branch_history, flat_candidates
        )
        current = (
            context[:, None, -1]
            .expand(-1, action_count, -1, -1)
            .reshape(batch * action_count, PATCH_TOKEN_COUNT, FEATURE_DIM)
            .contiguous()
        )
        physical = self.physical_head(
            current,
            successor,
            physical_inputs.reshape(
                batch * action_count, PHYSICAL_INPUT_DIM
            ).contiguous(),
        )
        result = GroundedBatchPredictionV1(
            successor_tokens=successor.reshape(
                batch, action_count, PATCH_TOKEN_COUNT, FEATURE_DIM
            ),
            standardized_physical_residuals=physical.standardized_residuals.reshape(
                batch, action_count, PHYSICAL_OUTPUT_DIM
            ),
            physical_attention=physical.attention.reshape(
                batch, action_count, PATCH_TOKEN_COUNT
            ),
        )
        if not all(bool(torch.isfinite(value).all()) for value in result):
            raise FloatingPointError("grounded batch prediction became nonfinite")
        return result

    @torch.no_grad()
    def encode_target(self, target_trunk_tokens: torch.Tensor) -> torch.Tensor:
        """Encode detached EMA successor targets for joint-JEPA training only."""

        if not isinstance(target_trunk_tokens, torch.Tensor):
            raise TypeError("target_trunk_tokens must be a torch.Tensor")
        if target_trunk_tokens.ndim != 4 or target_trunk_tokens.shape[0] < 1:
            raise ValueError(
                "target_trunk_tokens must have shape [B,A,257,384] with B >= 1"
            )
        batch, action_count = (
            int(target_trunk_tokens.shape[0]),
            int(target_trunk_tokens.shape[1]),
        )
        if action_count < 1:
            raise ValueError("target_trunk_tokens requires at least one action")
        device = _module_device(self.target_tail)
        target = _validate_float_tensor(
            target_trunk_tokens,
            shape=(batch, action_count, FULL_TOKEN_COUNT, FEATURE_DIM),
            device=device,
            name="target_trunk_tokens",
        )
        self.target_tail.eval()
        encoded = self.target_tail.patch_tokens(
            target.reshape(
                batch * action_count, FULL_TOKEN_COUNT, FEATURE_DIM
            )
        ).reshape(batch, action_count, PATCH_TOKEN_COUNT, FEATURE_DIM)
        return encoded.detach()

    @torch.no_grad()
    def update_target_ema(self, momentum: float | None = None) -> None:
        """Apply the exact parameter EMA and synchronize nonparameter buffers."""

        selected = self.ema_momentum if momentum is None else momentum
        if (
            isinstance(selected, bool)
            or not isinstance(selected, Real)
            or not 0.0 <= float(selected) < 1.0
        ):
            raise ValueError("EMA momentum must be a real number in [0,1)")
        coefficient = float(selected)
        online_parameters = dict(self.online_tail.named_parameters())
        target_parameters = dict(self.target_tail.named_parameters())
        if online_parameters.keys() != target_parameters.keys():
            raise RuntimeError("online and target DINO parameter inventories differ")
        for name in online_parameters:
            online = online_parameters[name]
            target = target_parameters[name]
            if (
                target.shape != online.shape
                or target.dtype != online.dtype
                or target.device != online.device
            ):
                raise RuntimeError("online and target DINO parameters are incompatible")
            target.mul_(coefficient).add_(online.detach(), alpha=1.0 - coefficient)
            if not bool(torch.isfinite(target).all()):
                raise FloatingPointError("EMA target parameter became nonfinite")

        online_buffers = dict(self.online_tail.named_buffers())
        target_buffers = dict(self.target_tail.named_buffers())
        if online_buffers.keys() != target_buffers.keys():
            raise RuntimeError("online and target DINO buffer inventories differ")
        for name in online_buffers:
            online = online_buffers[name]
            target = target_buffers[name]
            if target.shape != online.shape or target.dtype != online.dtype:
                raise RuntimeError("online and target DINO buffers are incompatible")
            target.copy_(online)
        self.target_tail.requires_grad_(False)
        self.target_tail.eval()
