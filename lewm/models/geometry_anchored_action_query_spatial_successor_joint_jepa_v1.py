"""Geometry-anchored Action-Query Spatial Successor joint JEPA V1.

The representation is constructed afresh from the frozen deformable-BEV
machinery.  The predictor sees only the normalized online current latent and
asks nine learned action queries for continuous successor residuals.  Future
and deranged latents are detached targets used only by the objective helpers.
"""
from __future__ import annotations

import copy
import math
from typing import Any, Mapping, NamedTuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from lewm.models.geometry_anchored_deformable_bev_lift_joint_jepa_v1 import (
    ACTION_VOCABULARY_V1,
    FREE_CLASS_V1,
    OCCUPIED_CLASS_V1,
    UNKNOWN_CLASS_V1,
    GeometryAnchoredBevSamplingV1,
    GeometryAnchoredDeformableBevLiftJointJepaV1 as _FrozenRepresentationJointJepaV1,
    GeometryAnchoredDeformableBevLiftJointJepaV1Config,
    GeometryAnchoredDeformableBevLiftV1,
    _construct_n320_encoder_without_rng_draw,
    _validate_n320_encoder_state,
    final_class_macro_nll_per_row,
    latent_energy_per_row,
)


ACTION_QUERY_WIDTH_V1 = 128
ACTION_QUERY_HEADS_V1 = 4
ACTION_QUERY_BLOCKS_V1 = 2
ACTION_QUERY_TOKEN_SIDE_V1 = 16
ACTION_QUERY_TOKEN_COUNT_V1 = 256
LOCAL_POOL_SIZE_V1 = 4
SSM_TEMPERATURE_V1 = 0.25
LOCAL_SCALE_FLOOR_V1 = 1e-3
LATENT_LAYER_NORM_EPSILON_V1 = 1e-5
PREDICTOR_PARAMETER_COUNT_V1 = 504_384
PREDICTOR_PARAMETER_TENSOR_COUNT_V1 = 34

PREDICTOR_ORDERED_PARAMETER_NAMES_V1 = (
    "future_queries",
    "current_downsampler.weight",
    "current_downsampler.bias",
    "action_embedding.weight",
    "blocks.0.query_norm.weight",
    "blocks.0.query_norm.bias",
    "blocks.0.memory_norm.weight",
    "blocks.0.memory_norm.bias",
    "blocks.0.attention.in_proj_weight",
    "blocks.0.attention.in_proj_bias",
    "blocks.0.attention.out_proj.weight",
    "blocks.0.attention.out_proj.bias",
    "blocks.0.ffn_norm.weight",
    "blocks.0.ffn_norm.bias",
    "blocks.0.linear1.weight",
    "blocks.0.linear1.bias",
    "blocks.0.linear2.weight",
    "blocks.0.linear2.bias",
    "blocks.1.query_norm.weight",
    "blocks.1.query_norm.bias",
    "blocks.1.memory_norm.weight",
    "blocks.1.memory_norm.bias",
    "blocks.1.attention.in_proj_weight",
    "blocks.1.attention.in_proj_bias",
    "blocks.1.attention.out_proj.weight",
    "blocks.1.attention.out_proj.bias",
    "blocks.1.ffn_norm.weight",
    "blocks.1.ffn_norm.bias",
    "blocks.1.linear1.weight",
    "blocks.1.linear1.bias",
    "blocks.1.linear2.weight",
    "blocks.1.linear2.bias",
    "output_head.weight",
    "output_head.bias",
)


class ActionQueryLocalEnergies(NamedTuple):
    """Token-local successor, deranged, and persistence energies."""

    positive: torch.Tensor
    negative: torch.Tensor
    persistence: torch.Tensor

    @property
    def e_pos(self) -> torch.Tensor:
        return self.positive

    @property
    def e_neg(self) -> torch.Tensor:
        return self.negative

    @property
    def e_per(self) -> torch.Tensor:
        return self.persistence


class ActionQueryReportingEnergies(NamedTuple):
    """Smooth-soft-min scores used only for reporting."""

    action: torch.Tensor
    correct: torch.Tensor
    deranged: torch.Tensor


class ActionQueryJointObjective(NamedTuple):
    """Exact dynamics terms and all auditable local intermediates."""

    positive: torch.Tensor
    negative: torch.Tensor
    persistence: torch.Tensor
    action_scale: torch.Tensor
    target_scale: torch.Tensor
    local_action_ce: torch.Tensor
    local_target_ce: torch.Tensor
    executed_positive: torch.Tensor
    executed_negative: torch.Tensor
    P_successor: torch.Tensor
    R_local_action: torch.Tensor
    C_deranged: torch.Tensor
    dynamics: torch.Tensor

    @property
    def e_pos(self) -> torch.Tensor:
        return self.positive

    @property
    def e_neg(self) -> torch.Tensor:
        return self.negative

    @property
    def e_per(self) -> torch.Tensor:
        return self.persistence

    @property
    def e_exec(self) -> torch.Tensor:
        return self.executed_positive

    @property
    def e_exec_neg(self) -> torch.Tensor:
        return self.executed_negative


def _validate_latent_nchw(value: torch.Tensor, *, name: str) -> None:
    if value.ndim != 4 or tuple(value.shape[1:]) != (64, 64, 64):
        raise ValueError(f"{name} must have shape (B,64,64,64)")
    if value.shape[0] < 1:
        raise ValueError(f"{name} must contain at least one row")
    if value.dtype != torch.float32:
        raise TypeError(f"{name} must use exact float32")
    if not bool(torch.isfinite(value).all()):
        raise FloatingPointError(f"{name} is nonfinite")


def _validate_action_indices(
    action_indices: torch.Tensor,
    *,
    batch: int,
    device: torch.device,
) -> torch.Tensor:
    if action_indices.shape != (batch,):
        raise ValueError("executed action indices must have shape (B,)")
    if action_indices.device != device:
        raise TypeError("action indices and latent must share a device")
    if action_indices.dtype == torch.bool or action_indices.is_floating_point():
        raise TypeError("action indices must use an integer dtype")
    indices = action_indices.to(dtype=torch.long)
    if bool(((indices < 0) | (indices >= 9)).any()):
        raise ValueError("action indices must lie in [0,8]")
    return indices


def normalize_latent_per_cell_v1(latent: torch.Tensor) -> torch.Tensor:
    """Apply the exact non-affine channel LayerNorm at each BEV cell."""

    _validate_latent_nchw(latent, name="latent")
    normalized = F.layer_norm(
        latent.movedim(1, -1),
        (64,),
        weight=None,
        bias=None,
        eps=LATENT_LAYER_NORM_EPSILON_V1,
    ).movedim(-1, 1)
    if normalized.dtype != torch.float32 or not bool(torch.isfinite(normalized).all()):
        raise FloatingPointError("normalized latent is nonfinite or not float32")
    return normalized


def fixed_spatial_position_table_v1() -> torch.Tensor:
    """Return the exact row-major 16-by-16 float32 sine/cosine table."""

    coordinates = torch.arange(ACTION_QUERY_TOKEN_SIDE_V1, dtype=torch.float32)
    frequency_index = torch.arange(32, dtype=torch.float32)
    frequencies = torch.pow(
        torch.tensor(10000.0, dtype=torch.float32), -frequency_index / 32.0
    )
    angles = coordinates[:, None] * frequencies[None, :]
    components = torch.stack((torch.sin(angles), torch.cos(angles)), dim=-1)
    components = components.flatten(1)
    rows = components[:, None, :].expand(16, 16, 64)
    columns = components[None, :, :].expand(16, 16, 64)
    return torch.cat((rows, columns), dim=-1).reshape(256, 128).contiguous()


class ActionQueryFutureBlockV1(nn.Module):
    """One separately parameterized pre-norm future-query cross-attention block."""

    def __init__(self) -> None:
        super().__init__()
        self.query_norm = nn.LayerNorm(128, eps=1e-5, elementwise_affine=True)
        self.memory_norm = nn.LayerNorm(128, eps=1e-5, elementwise_affine=True)
        self.attention = nn.MultiheadAttention(
            128,
            4,
            dropout=0.0,
            bias=True,
            batch_first=True,
        )
        self.ffn_norm = nn.LayerNorm(128, eps=1e-5, elementwise_affine=True)
        self.linear1 = nn.Linear(128, 256, bias=True)
        self.activation = nn.GELU()
        self.linear2 = nn.Linear(256, 128, bias=True)

    def forward(self, query: torch.Tensor, memory: torch.Tensor) -> torch.Tensor:
        normalized_memory = self.memory_norm(memory)
        attention_output = self.attention(
            self.query_norm(query),
            normalized_memory,
            normalized_memory,
            need_weights=False,
        )[0]
        attended = query + attention_output
        return attended + self.linear2(
            self.activation(self.linear1(self.ffn_norm(attended)))
        )


class ActionQuerySpatialSuccessorPredictorV1(nn.Module):
    """Vectorized nine-action spatial successor predictor."""

    def __init__(
        self, config: GeometryAnchoredDeformableBevLiftJointJepaV1Config
    ) -> None:
        super().__init__()
        self.config = config
        self.current_downsampler = nn.Conv2d(
            64, 128, kernel_size=4, stride=4, padding=0, bias=True
        )
        self.register_buffer(
            "position_encoding", fixed_spatial_position_table_v1(), persistent=True
        )
        self.action_embedding = nn.Embedding(9, 128)
        nn.init.xavier_uniform_(self.action_embedding.weight)
        self.future_queries = nn.Parameter(torch.empty(256, 128))
        nn.init.xavier_uniform_(self.future_queries)
        self.blocks = nn.ModuleList(
            [ActionQueryFutureBlockV1() for _ in range(ACTION_QUERY_BLOCKS_V1)]
        )
        self.output_head = nn.Conv2d(
            128, 64, kernel_size=3, stride=1, padding=1, bias=True
        )
        nn.init.xavier_uniform_(self.output_head.weight)
        nn.init.zeros_(self.output_head.bias)
        # The three Xavier draws and output-bias zero above are the only
        # explicit overrides, in module order, on the caller-owned continuing
        # 20260712 CPU generator stream.
        self._validate_initialization_and_inventory()

    def _validate_initialization_and_inventory(self) -> None:
        names = tuple(name for name, _ in self.named_parameters())
        if names != PREDICTOR_ORDERED_PARAMETER_NAMES_V1:
            raise RuntimeError("action-query predictor parameter order changed")
        parameters = tuple(self.parameters())
        if len(parameters) != PREDICTOR_PARAMETER_TENSOR_COUNT_V1 or sum(
            value.numel() for value in parameters
        ) != PREDICTOR_PARAMETER_COUNT_V1:
            raise RuntimeError("action-query predictor inventory changed")
        initialized = (
            self.action_embedding.weight,
            self.future_queries,
            self.output_head.weight,
        )
        if any(
            not bool(torch.isfinite(value).all()) or float(value.detach().norm()) <= 0.0
            for value in initialized
        ):
            raise RuntimeError("explicit action-query initialization is invalid")
        rows = self.action_embedding.weight.detach()
        if any(
            torch.equal(rows[first], rows[second])
            for first in range(9)
            for second in range(first + 1, 9)
        ):
            raise RuntimeError("action-token rows are not pairwise distinct")

    def ordered_parameter_receipt(self) -> tuple[dict[str, Any], ...]:
        """Return the predictor-local ordered tensor inventory."""

        return tuple(
            {
                "name": name,
                "shape": list(parameter.shape),
                "parameter_count": parameter.numel(),
            }
            for name, parameter in self.named_parameters()
        )

    def ordered_module_receipt(self) -> tuple[dict[str, str], ...]:
        """Return the predictor-local module traversal inventory."""

        return tuple(
            {"name": name, "type": type(module).__name__}
            for name, module in self.named_modules()
        )

    def _residuals_and_head_inputs(
        self, normalized_current_latent: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        _validate_latent_nchw(
            normalized_current_latent, name="normalized_current_latent"
        )
        if normalized_current_latent.device != self.future_queries.device:
            raise TypeError("normalized_current_latent and predictor must share a device")
        batch = normalized_current_latent.shape[0]
        current_tokens = self.current_downsampler(normalized_current_latent)
        if tuple(current_tokens.shape) != (batch, 128, 16, 16):
            raise RuntimeError("current token downsampling shape changed")
        current_tokens = current_tokens.flatten(2).transpose(1, 2)
        position = self.position_encoding.to(
            device=current_tokens.device, dtype=current_tokens.dtype
        )
        actions = self.action_embedding.weight[None].expand(batch, -1, -1)
        query = (
            self.future_queries[None, None]
            + position[None, None]
            + actions[:, :, None, :]
        )
        current_memory = (current_tokens + position[None]).unsqueeze(1).expand(
            -1, 9, -1, -1
        )
        memory = torch.cat((actions[:, :, None, :], current_memory), dim=2)
        query = query.reshape(batch * 9, 256, 128)
        memory = memory.reshape(batch * 9, 257, 128)
        for block in self.blocks:
            query = block(query, memory)
        head_inputs = query.transpose(1, 2).reshape(batch * 9, 128, 16, 16)
        upsampled = F.interpolate(
            head_inputs,
            scale_factor=4.0,
            mode="bilinear",
            align_corners=False,
        )
        residuals = self.output_head(upsampled).reshape(batch, 9, 64, 64, 64)
        head_inputs = head_inputs.reshape(batch, 9, 128, 16, 16)
        if not bool(torch.isfinite(residuals).all()) or not bool(
            torch.isfinite(head_inputs).all()
        ):
            raise FloatingPointError("action-query predictor output is nonfinite")
        return residuals, head_inputs

    def predict_residuals_all_actions(
        self, normalized_current_latent: torch.Tensor
    ) -> torch.Tensor:
        return self._residuals_and_head_inputs(normalized_current_latent)[0]

    def predict_residuals_selected_actions(
        self,
        normalized_current_latent: torch.Tensor,
        action_indices: torch.Tensor,
    ) -> torch.Tensor:
        """Evaluate only each row's fixed action for observation rollouts."""

        _validate_latent_nchw(
            normalized_current_latent, name="normalized_current_latent"
        )
        if normalized_current_latent.device != self.future_queries.device:
            raise TypeError("normalized_current_latent and predictor must share a device")
        batch = normalized_current_latent.shape[0]
        indices = _validate_action_indices(
            action_indices,
            batch=batch,
            device=normalized_current_latent.device,
        )
        current_tokens = self.current_downsampler(normalized_current_latent)
        if tuple(current_tokens.shape) != (batch, 128, 16, 16):
            raise RuntimeError("selected current token downsampling shape changed")
        current_tokens = current_tokens.flatten(2).transpose(1, 2)
        position = self.position_encoding.to(
            device=current_tokens.device, dtype=current_tokens.dtype
        )
        actions = self.action_embedding(indices)
        query = self.future_queries[None] + position[None] + actions[:, None, :]
        memory = torch.cat((
            actions[:, None, :],
            current_tokens + position[None],
        ), dim=1)
        for block in self.blocks:
            query = block(query, memory)
        head_inputs = query.transpose(1, 2).reshape(batch, 128, 16, 16)
        upsampled = F.interpolate(
            head_inputs,
            scale_factor=4.0,
            mode="bilinear",
            align_corners=False,
        )
        residuals = self.output_head(upsampled)
        if tuple(residuals.shape) != (batch, 64, 64, 64) or not bool(
            torch.isfinite(residuals).all()
        ):
            raise FloatingPointError("selected action-query output is invalid")
        return residuals

    def head_inputs_all_actions(
        self, normalized_current_latent: torch.Tensor
    ) -> torch.Tensor:
        """Expose the shared-head input solely for source-bound U0 proof."""

        return self._residuals_and_head_inputs(normalized_current_latent)[1]

    def forward(self, normalized_current_latent: torch.Tensor) -> torch.Tensor:
        residuals = self.predict_residuals_all_actions(normalized_current_latent)
        return normalized_current_latent[:, None] + residuals


def select_action_successor_v1(
    all_successors: torch.Tensor, action_indices: torch.Tensor
) -> torch.Tensor:
    """Take one executed-action slice from a vectorized all-action call."""

    if all_successors.ndim != 5 or tuple(all_successors.shape[1:]) != (
        9,
        64,
        64,
        64,
    ):
        raise ValueError("all_successors must have shape (B,9,64,64,64)")
    indices = _validate_action_indices(
        action_indices,
        batch=all_successors.shape[0],
        device=all_successors.device,
    )
    rows = torch.arange(all_successors.shape[0], device=all_successors.device)
    return all_successors[rows, indices]


def action_query_local_energies_v1(
    predictions: torch.Tensor,
    target_next: torch.Tensor,
    target_negative: torch.Tensor,
    current: torch.Tensor,
) -> ActionQueryLocalEnergies:
    """Compute exact beta-one Smooth-L1 energies on the 16-by-16 grid."""

    if predictions.ndim != 5 or tuple(predictions.shape[1:]) != (9, 64, 64, 64):
        raise ValueError("predictions must have shape (B,9,64,64,64)")
    for name, value in (
        ("target_next", target_next),
        ("target_negative", target_negative),
        ("current", current),
    ):
        _validate_latent_nchw(value, name=name)
        if value.shape[0] != predictions.shape[0]:
            raise ValueError(f"{name} batch differs from predictions")
        if value.device != predictions.device or value.dtype != predictions.dtype:
            raise TypeError(f"{name} and predictions must share dtype and device")
    if predictions.dtype != torch.float32 or not bool(torch.isfinite(predictions).all()):
        raise FloatingPointError("predictions are nonfinite or not float32")
    next_detached = target_next.detach()
    negative_detached = target_negative.detach()
    positive_cells = F.smooth_l1_loss(
        predictions,
        next_detached[:, None].expand_as(predictions),
        beta=1.0,
        reduction="none",
    ).mean(dim=2)
    negative_cells = F.smooth_l1_loss(
        predictions,
        negative_detached[:, None].expand_as(predictions),
        beta=1.0,
        reduction="none",
    ).mean(dim=2)
    persistence_cells = F.smooth_l1_loss(
        current,
        next_detached,
        beta=1.0,
        reduction="none",
    ).mean(dim=1, keepdim=True)
    positive = F.avg_pool2d(positive_cells, 4, stride=4).flatten(2)
    negative = F.avg_pool2d(negative_cells, 4, stride=4).flatten(2)
    persistence = F.avg_pool2d(persistence_cells, 4, stride=4).flatten(1)
    if (
        tuple(positive.shape) != (predictions.shape[0], 9, 256)
        or negative.shape != positive.shape
        or tuple(persistence.shape) != (predictions.shape[0], 256)
        or not all(
            bool(torch.isfinite(value).all())
            for value in (positive, negative, persistence)
        )
    ):
        raise FloatingPointError("local successor energies are invalid")
    return ActionQueryLocalEnergies(positive, negative, persistence)


def detached_local_action_scale_v1(positive: torch.Tensor) -> torch.Tensor:
    if positive.ndim != 3 or tuple(positive.shape[1:]) != (9, 256):
        raise ValueError("positive energies must have shape (B,9,256)")
    if not bool(torch.isfinite(positive).all()) or bool((positive < 0.0).any()):
        raise FloatingPointError("positive energies are invalid")
    return positive.mean(dim=1).detach().clamp_min(LOCAL_SCALE_FLOOR_V1)


def gather_executed_action_v1(
    action_values: torch.Tensor, executed_actions: torch.Tensor
) -> torch.Tensor:
    if action_values.ndim < 2 or action_values.shape[1] != 9:
        raise ValueError("action_values must have action axis one of length nine")
    indices = _validate_action_indices(
        executed_actions,
        batch=action_values.shape[0],
        device=action_values.device,
    )
    rows = torch.arange(action_values.shape[0], device=action_values.device)
    return action_values[rows, indices]


def detached_local_target_scale_v1(
    executed_positive: torch.Tensor, executed_negative: torch.Tensor
) -> torch.Tensor:
    if (
        executed_positive.ndim != 2
        or tuple(executed_positive.shape[1:]) != (256,)
        or executed_negative.shape != executed_positive.shape
    ):
        raise ValueError("executed energies must both have shape (B,256)")
    if not bool(torch.isfinite(executed_positive).all()) or not bool(
        torch.isfinite(executed_negative).all()
    ):
        raise FloatingPointError("executed energies are nonfinite")
    return (0.5 * (executed_positive + executed_negative)).detach().clamp_min(
        LOCAL_SCALE_FLOOR_V1
    )


def local_action_cross_entropy_v1(
    positive: torch.Tensor,
    executed_actions: torch.Tensor,
    action_scale: torch.Tensor | None = None,
) -> torch.Tensor:
    scale = (
        detached_local_action_scale_v1(positive)
        if action_scale is None
        else action_scale
    )
    if scale.shape != positive.shape[:1] + positive.shape[2:]:
        raise ValueError("action_scale must have shape (B,256)")
    if scale.requires_grad:
        raise ValueError("action_scale must be detached")
    indices = _validate_action_indices(
        executed_actions, batch=positive.shape[0], device=positive.device
    )
    labels = indices[:, None].expand(-1, 256)
    return F.cross_entropy(-positive / scale[:, None, :], labels, reduction="none")


def local_target_cross_entropy_v1(
    executed_positive: torch.Tensor,
    executed_negative: torch.Tensor,
    target_scale: torch.Tensor | None = None,
) -> torch.Tensor:
    scale = (
        detached_local_target_scale_v1(executed_positive, executed_negative)
        if target_scale is None
        else target_scale
    )
    if scale.shape != executed_positive.shape:
        raise ValueError("target_scale must have shape (B,256)")
    if scale.requires_grad:
        raise ValueError("target_scale must be detached")
    logits = torch.stack(
        (-executed_positive / scale, -executed_negative / scale), dim=1
    )
    labels = torch.zeros_like(executed_positive, dtype=torch.long)
    return F.cross_entropy(logits, labels, reduction="none")


def smooth_spatial_soft_min_v1(values: torch.Tensor) -> torch.Tensor:
    """Exact corrected SSM over the final 256-token axis."""

    if values.ndim < 1 or values.shape[-1] != ACTION_QUERY_TOKEN_COUNT_V1:
        raise ValueError("SSM values must have a final axis of length 256")
    if not bool(torch.isfinite(values).all()):
        raise FloatingPointError("SSM values are nonfinite")
    result = -SSM_TEMPERATURE_V1 * (
        torch.logsumexp(-values / SSM_TEMPERATURE_V1, dim=-1)
        - math.log(ACTION_QUERY_TOKEN_COUNT_V1)
    )
    if not bool(torch.isfinite(result).all()):
        raise FloatingPointError("SSM result is nonfinite")
    return result


def action_reporting_energies_v1(
    positive: torch.Tensor, action_scale: torch.Tensor | None = None
) -> torch.Tensor:
    scale = (
        detached_local_action_scale_v1(positive)
        if action_scale is None
        else action_scale
    )
    if scale.shape != positive.shape[:1] + positive.shape[2:]:
        raise ValueError("action_scale must have shape (B,256)")
    return smooth_spatial_soft_min_v1(positive / scale[:, None, :])


def target_reporting_energies_v1(
    executed_positive: torch.Tensor,
    executed_negative: torch.Tensor,
    target_scale: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    scale = (
        detached_local_target_scale_v1(executed_positive, executed_negative)
        if target_scale is None
        else target_scale
    )
    if scale.shape != executed_positive.shape:
        raise ValueError("target_scale must have shape (B,256)")
    return (
        smooth_spatial_soft_min_v1(executed_positive / scale),
        smooth_spatial_soft_min_v1(executed_negative / scale),
    )


def reporting_energy_helpers_v1(
    positive: torch.Tensor,
    negative: torch.Tensor,
    executed_actions: torch.Tensor,
) -> ActionQueryReportingEnergies:
    """Return all reporting scores from energies, never from future predictor input."""

    action_scale = detached_local_action_scale_v1(positive)
    executed_positive = gather_executed_action_v1(positive, executed_actions)
    executed_negative = gather_executed_action_v1(negative, executed_actions)
    target_scale = detached_local_target_scale_v1(
        executed_positive, executed_negative
    )
    correct, deranged = target_reporting_energies_v1(
        executed_positive, executed_negative, target_scale
    )
    return ActionQueryReportingEnergies(
        action_reporting_energies_v1(positive, action_scale), correct, deranged
    )


def reporting_action_cross_entropy_v1(
    action_energies: torch.Tensor, executed_actions: torch.Tensor
) -> torch.Tensor:
    if action_energies.ndim != 2 or action_energies.shape[1] != 9:
        raise ValueError("action_energies must have shape (B,9)")
    indices = _validate_action_indices(
        executed_actions,
        batch=action_energies.shape[0],
        device=action_energies.device,
    )
    return F.cross_entropy(-action_energies, indices, reduction="none")


def reporting_target_cross_entropy_v1(
    correct_energy: torch.Tensor, deranged_energy: torch.Tensor
) -> torch.Tensor:
    if correct_energy.ndim != 1 or deranged_energy.shape != correct_energy.shape:
        raise ValueError("reporting target energies must both have shape (B,)")
    logits = torch.stack((-correct_energy, -deranged_energy), dim=1)
    return F.cross_entropy(
        logits, torch.zeros_like(correct_energy, dtype=torch.long), reduction="none"
    )


def action_query_joint_objective_v1(
    predictions: torch.Tensor,
    target_next: torch.Tensor,
    target_negative: torch.Tensor,
    current: torch.Tensor,
    executed_actions: torch.Tensor,
) -> ActionQueryJointObjective:
    """Return the three exact dynamics losses and their unweighted sum."""

    energies = action_query_local_energies_v1(
        predictions, target_next, target_negative, current
    )
    action_scale = detached_local_action_scale_v1(energies.positive)
    executed_positive = gather_executed_action_v1(
        energies.positive, executed_actions
    )
    executed_negative = gather_executed_action_v1(
        energies.negative, executed_actions
    )
    target_scale = detached_local_target_scale_v1(
        executed_positive, executed_negative
    )
    local_action_ce = local_action_cross_entropy_v1(
        energies.positive, executed_actions, action_scale
    )
    local_target_ce = local_target_cross_entropy_v1(
        executed_positive, executed_negative, target_scale
    )
    p_successor = executed_positive.mean()
    r_local_action = smooth_spatial_soft_min_v1(
        local_action_ce / math.log(9.0)
    ).mean()
    c_deranged = smooth_spatial_soft_min_v1(
        local_target_ce / math.log(2.0)
    ).mean()
    dynamics = p_successor + r_local_action + c_deranged
    if not all(
        bool(torch.isfinite(value))
        for value in (p_successor, r_local_action, c_deranged, dynamics)
    ):
        raise FloatingPointError("action-query objective is nonfinite")
    return ActionQueryJointObjective(
        energies.positive,
        energies.negative,
        energies.persistence,
        action_scale,
        target_scale,
        local_action_ce,
        local_target_ce,
        executed_positive,
        executed_negative,
        p_successor,
        r_local_action,
        c_deranged,
        dynamics,
    )


class GeometryAnchoredActionQuerySpatialSuccessorJointJepaV1(
    _FrozenRepresentationJointJepaV1
):
    """Fresh frozen-geometry representation with the action-query predictor."""

    def __init__(
        self,
        n320_encoder_state_dict: Mapping[str, torch.Tensor],
        config: GeometryAnchoredDeformableBevLiftJointJepaV1Config | None = None,
    ) -> None:
        # Preserve the inherited construction order without constructing its
        # closed predecessor predictor.
        nn.Module.__init__(self)
        self.config = config or GeometryAnchoredDeformableBevLiftJointJepaV1Config()
        self.encoder = _construct_n320_encoder_without_rng_draw(self.config)
        _validate_n320_encoder_state(self.encoder, n320_encoder_state_dict)
        self.encoder.load_state_dict(n320_encoder_state_dict, strict=True)

        caller_rng = torch.random.get_rng_state().clone()
        try:
            torch.random.default_generator.manual_seed(self.config.initialization_seed)
            self.bev_lift = GeometryAnchoredDeformableBevLiftV1(self.config)
            self.semantic_head = nn.Conv2d(
                self.config.bev_dim,
                self.config.state_classes,
                kernel_size=1,
                bias=True,
            )
            self.predictor = ActionQuerySpatialSuccessorPredictorV1(self.config)
        finally:
            torch.random.set_rng_state(caller_rng)

        self.target_encoder = copy.deepcopy(self.encoder)
        self.target_bev_lift = copy.deepcopy(self.bev_lift)
        self.register_buffer(
            "target_hard_sync_count", torch.zeros((), dtype=torch.long), persistent=True
        )
        self.register_buffer(
            "ema_update_count", torch.zeros((), dtype=torch.long), persistent=True
        )
        self.hard_sync_target_from_online()

    def predict_all_actions(
        self, normalized_current_latent: torch.Tensor
    ) -> torch.Tensor:
        return self.predictor(normalized_current_latent)

    def predict_residuals_all_actions(
        self, normalized_current_latent: torch.Tensor
    ) -> torch.Tensor:
        return self.predictor.predict_residuals_all_actions(
            normalized_current_latent
        )

    def predict(
        self,
        normalized_current_latent: torch.Tensor,
        action_indices: torch.Tensor,
    ) -> torch.Tensor:
        return normalized_current_latent + self.predictor.predict_residuals_selected_actions(
            normalized_current_latent, action_indices
        )

    def predictor_component_parameter_receipt(self) -> dict[str, dict[str, Any]]:
        """Return the eight source-bound component inventories with full names."""

        relative_groups = {
            "downsampler": (
                "current_downsampler.weight",
                "current_downsampler.bias",
            ),
            "action_embedding": ("action_embedding.weight",),
            "future_queries": ("future_queries",),
            "block_0_attention": tuple(
                name
                for name in PREDICTOR_ORDERED_PARAMETER_NAMES_V1
                if name.startswith(
                    (
                        "blocks.0.query_norm.",
                        "blocks.0.memory_norm.",
                        "blocks.0.attention.",
                    )
                )
            ),
            "block_0_mlp": tuple(
                name
                for name in PREDICTOR_ORDERED_PARAMETER_NAMES_V1
                if name.startswith(
                    (
                        "blocks.0.ffn_norm.",
                        "blocks.0.linear1.",
                        "blocks.0.linear2.",
                    )
                )
            ),
            "block_1_attention": tuple(
                name
                for name in PREDICTOR_ORDERED_PARAMETER_NAMES_V1
                if name.startswith(
                    (
                        "blocks.1.query_norm.",
                        "blocks.1.memory_norm.",
                        "blocks.1.attention.",
                    )
                )
            ),
            "block_1_mlp": tuple(
                name
                for name in PREDICTOR_ORDERED_PARAMETER_NAMES_V1
                if name.startswith(
                    (
                        "blocks.1.ffn_norm.",
                        "blocks.1.linear1.",
                        "blocks.1.linear2.",
                    )
                )
            ),
            "output_head": ("output_head.weight", "output_head.bias"),
        }
        parameters = dict(self.predictor.named_parameters())
        receipt: dict[str, dict[str, Any]] = {}
        flattened: list[str] = []
        for component, relative_names in relative_groups.items():
            flattened.extend(relative_names)
            receipt[component] = {
                "ordered_parameter_names": [
                    f"predictor.{name}" for name in relative_names
                ],
                "tensor_count": len(relative_names),
                "parameter_count": sum(parameters[name].numel() for name in relative_names),
            }
        if set(flattened) != set(PREDICTOR_ORDERED_PARAMETER_NAMES_V1) or len(
            flattened
        ) != len(PREDICTOR_ORDERED_PARAMETER_NAMES_V1):
            raise RuntimeError("predictor component receipt is incomplete")
        return receipt

    def predictor_ordered_parameter_receipt(self) -> tuple[dict[str, Any], ...]:
        return tuple(
            {
                **row,
                "name": f"predictor.{row['name']}",
            }
            for row in self.predictor.ordered_parameter_receipt()
        )

    def predictor_ordered_module_receipt(self) -> tuple[dict[str, str], ...]:
        return tuple(
            {
                **row,
                "name": "predictor" + (f".{row['name']}" if row["name"] else ""),
            }
            for row in self.predictor.ordered_module_receipt()
        )


GeometryAnchoredActionQuerySpatialSuccessorJointJepaV1Config = (
    GeometryAnchoredDeformableBevLiftJointJepaV1Config
)
# The inherited runner resolves this historical name from the selected module.
GeometryAnchoredDeformableBevLiftJointJepaV1 = (
    GeometryAnchoredActionQuerySpatialSuccessorJointJepaV1
)

# Concise source-bound runner aliases.
normalize_latent_per_cell = normalize_latent_per_cell_v1
action_query_local_energies = action_query_local_energies_v1
detached_local_action_scale = detached_local_action_scale_v1
detached_local_target_scale = detached_local_target_scale_v1
local_action_cross_entropy = local_action_cross_entropy_v1
local_target_cross_entropy = local_target_cross_entropy_v1
smooth_spatial_soft_min = smooth_spatial_soft_min_v1
action_energy_scores_v1 = action_reporting_energies_v1
target_energy_scores_v1 = target_reporting_energies_v1
action_energy_scores = action_reporting_energies_v1
target_energy_scores = target_reporting_energies_v1
action_query_joint_objective = action_query_joint_objective_v1
final_class_macro_nll_per_row_v1 = final_class_macro_nll_per_row
latent_energy_per_row_v1 = latent_energy_per_row


__all__ = [
    "ACTION_QUERY_BLOCKS_V1",
    "ACTION_QUERY_HEADS_V1",
    "ACTION_QUERY_TOKEN_COUNT_V1",
    "ACTION_QUERY_TOKEN_SIDE_V1",
    "ACTION_QUERY_WIDTH_V1",
    "ACTION_VOCABULARY_V1",
    "ActionQueryFutureBlockV1",
    "ActionQueryJointObjective",
    "ActionQueryLocalEnergies",
    "ActionQueryReportingEnergies",
    "ActionQuerySpatialSuccessorPredictorV1",
    "FREE_CLASS_V1",
    "GeometryAnchoredActionQuerySpatialSuccessorJointJepaV1",
    "GeometryAnchoredActionQuerySpatialSuccessorJointJepaV1Config",
    "GeometryAnchoredBevSamplingV1",
    "GeometryAnchoredDeformableBevLiftJointJepaV1",
    "GeometryAnchoredDeformableBevLiftJointJepaV1Config",
    "GeometryAnchoredDeformableBevLiftV1",
    "LATENT_LAYER_NORM_EPSILON_V1",
    "LOCAL_POOL_SIZE_V1",
    "LOCAL_SCALE_FLOOR_V1",
    "OCCUPIED_CLASS_V1",
    "PREDICTOR_ORDERED_PARAMETER_NAMES_V1",
    "PREDICTOR_PARAMETER_COUNT_V1",
    "PREDICTOR_PARAMETER_TENSOR_COUNT_V1",
    "SSM_TEMPERATURE_V1",
    "UNKNOWN_CLASS_V1",
    "action_energy_scores",
    "action_energy_scores_v1",
    "action_query_joint_objective",
    "action_query_joint_objective_v1",
    "action_query_local_energies",
    "action_query_local_energies_v1",
    "action_reporting_energies_v1",
    "detached_local_action_scale",
    "detached_local_action_scale_v1",
    "detached_local_target_scale",
    "detached_local_target_scale_v1",
    "final_class_macro_nll_per_row",
    "final_class_macro_nll_per_row_v1",
    "fixed_spatial_position_table_v1",
    "gather_executed_action_v1",
    "latent_energy_per_row",
    "latent_energy_per_row_v1",
    "local_action_cross_entropy",
    "local_action_cross_entropy_v1",
    "local_target_cross_entropy",
    "local_target_cross_entropy_v1",
    "normalize_latent_per_cell",
    "normalize_latent_per_cell_v1",
    "reporting_action_cross_entropy_v1",
    "reporting_energy_helpers_v1",
    "reporting_target_cross_entropy_v1",
    "select_action_successor_v1",
    "smooth_spatial_soft_min",
    "smooth_spatial_soft_min_v1",
    "target_energy_scores",
    "target_energy_scores_v1",
    "target_reporting_energies_v1",
]
