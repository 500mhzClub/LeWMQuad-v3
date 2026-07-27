"""Geometry-anchored local deformable RGB-to-BEV joint JEPA V1.

This module implements the narrow model mechanism frozen in the 2026-07-27
preregistration.  A fixed level-camera projection gives every BEV cell a
projective image anchor.  The learned lift may take four bounded, local
samples around that anchor, but it has no global-attention, pooling, pose, or
coordinate bypass.  The target branch contains only the EMA encoder and lift.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass
import math
from typing import Iterator, Mapping

import torch
import torch.nn as nn
import torch.nn.functional as F

from lewm.models.encoders import VisionEncoder


UNKNOWN_CLASS_V1 = 0
FREE_CLASS_V1 = 1
OCCUPIED_CLASS_V1 = 2
ACTION_VOCABULARY_V1 = (
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
class GeometryAnchoredDeformableBevLiftJointJepaV1Config:
    """Immutable architecture constants from the frozen preregistration."""

    image_size: int = 112
    patch_size: int = 7
    encoder_dim: int = 192
    encoder_depth: int = 6
    encoder_heads: int = 6
    encoder_mlp_ratio: int = 4
    encoder_dropout: float = 0.0
    token_side: int = 16
    bev_dim: int = 64
    bev_size: tuple[int, int] = (64, 64)
    forward_range_m: tuple[float, float] = (-0.95, 5.35)
    left_range_m: tuple[float, float] = (-3.15, 3.15)
    samples_per_cell: int = 4
    offset_radius_token_cells: float = 2.0
    state_classes: int = 3
    action_dim: int = 9
    target_ema_momentum: float = 0.996
    initialization_seed: int = 20260712
    camera_origin_xyz_m: tuple[float, float, float] = (0.326, 0.0, 0.043)
    camera_forward_xyz: tuple[float, float, float] = (1.0, 0.0, 0.0)
    camera_right_xyz: tuple[float, float, float] = (0.0, -1.0, 0.0)
    camera_up_xyz: tuple[float, float, float] = (0.0, 0.0, 1.0)
    ground_z_m: float = -0.333
    horizontal_fov_degrees: float = 78.323
    vertical_fov_degrees: float = 62.8370386364
    camera_near_m: float = 0.05

    def __post_init__(self) -> None:
        expected = {
            "image_size": 112,
            "patch_size": 7,
            "encoder_dim": 192,
            "encoder_depth": 6,
            "encoder_heads": 6,
            "encoder_mlp_ratio": 4,
            "encoder_dropout": 0.0,
            "token_side": 16,
            "bev_dim": 64,
            "bev_size": (64, 64),
            "forward_range_m": (-0.95, 5.35),
            "left_range_m": (-3.15, 3.15),
            "samples_per_cell": 4,
            "offset_radius_token_cells": 2.0,
            "state_classes": 3,
            "action_dim": 9,
            "target_ema_momentum": 0.996,
            "initialization_seed": 20260712,
            "camera_origin_xyz_m": (0.326, 0.0, 0.043),
            "camera_forward_xyz": (1.0, 0.0, 0.0),
            "camera_right_xyz": (0.0, -1.0, 0.0),
            "camera_up_xyz": (0.0, 0.0, 1.0),
            "ground_z_m": -0.333,
            "horizontal_fov_degrees": 78.323,
            "vertical_fov_degrees": 62.8370386364,
            "camera_near_m": 0.05,
        }
        changed = [
            name for name, value in expected.items() if getattr(self, name) != value
        ]
        if changed:
            raise ValueError(
                "Geometry-anchored joint-JEPA V1 constants cannot change: "
                + ", ".join(changed)
            )


@dataclass(frozen=True)
class GeometryAnchoredBevSamplingV1:
    """Auditable intermediate values from one local deformable lift call."""

    latent: torch.Tensor
    anchor_in_frustum: torch.Tensor
    sample_valid_mask: torch.Tensor
    cell_valid_mask: torch.Tensor
    sample_grid_xy: torch.Tensor
    offsets_token_cells: torch.Tensor
    sample_weights: torch.Tensor


def _construct_n320_encoder_without_rng_draw(
    config: GeometryAnchoredDeformableBevLiftJointJepaV1Config,
) -> VisionEncoder:
    caller_rng = torch.random.get_rng_state().clone()
    try:
        return VisionEncoder(
            image_size=config.image_size,
            patch_size=config.patch_size,
            hidden_dim=config.encoder_dim,
            depth=config.encoder_depth,
            n_heads=config.encoder_heads,
            mlp_ratio=config.encoder_mlp_ratio,
            dropout=config.encoder_dropout,
        )
    finally:
        torch.random.set_rng_state(caller_rng)


def _validate_n320_encoder_state(
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
            raise TypeError(f"N320 state {name!r} is not a tensor")
        if value.shape != expected_tensor.shape:
            raise ValueError(
                f"N320 state {name!r} has shape {tuple(value.shape)}, "
                f"expected {tuple(expected_tensor.shape)}"
            )
        if value.dtype != torch.float32:
            raise TypeError(f"N320 state {name!r} must be exact float32")
        if not bool(torch.isfinite(value).all()):
            raise FloatingPointError(f"N320 state {name!r} is nonfinite")


class _LocalResidualBlockV1(nn.Module):
    """A spatially local residual block with no statistic-sharing layer."""

    def __init__(self, width: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(width, width, kernel_size=3, stride=1, padding=1)
        self.activation = nn.GELU()
        self.conv2 = nn.Conv2d(width, width, kernel_size=3, stride=1, padding=1)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return value + self.conv2(self.activation(self.conv1(value)))


def _fixed_projective_anchor(
    config: GeometryAnchoredDeformableBevLiftJointJepaV1Config,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return frozen projective buffers, calculated in float64 then stored f32."""

    dtype = torch.float64
    # The registered metric ranges name the first and last cell centres.
    forward_centers = torch.linspace(
        config.forward_range_m[0],
        config.forward_range_m[1],
        config.bev_size[0],
        dtype=dtype,
    )
    left_centers = torch.linspace(
        config.left_range_m[0],
        config.left_range_m[1],
        config.bev_size[1],
        dtype=dtype,
    )
    forward_grid, left_grid = torch.meshgrid(
        forward_centers, left_centers, indexing="ij"
    )
    ground = torch.stack(
        (
            forward_grid,
            left_grid,
            torch.full_like(forward_grid, config.ground_z_m),
        ),
        dim=-1,
    )
    origin = torch.tensor(config.camera_origin_xyz_m, dtype=dtype)
    basis = torch.tensor(
        (
            config.camera_forward_xyz,
            config.camera_right_xyz,
            config.camera_up_xyz,
        ),
        dtype=dtype,
    )
    relative = ground - origin
    camera_forward = torch.einsum("hwc,c->hw", relative, basis[0])
    camera_right = torch.einsum("hwc,c->hw", relative, basis[1])
    camera_up = torch.einsum("hwc,c->hw", relative, basis[2])
    tan_half_h = math.tan(math.radians(config.horizontal_fov_degrees) / 2.0)
    tan_half_v = math.tan(math.radians(config.vertical_fov_degrees) / 2.0)
    safe_forward = camera_forward.clamp_min(torch.finfo(dtype).eps)
    grid_x = camera_right / (safe_forward * tan_half_h)
    grid_y = -camera_up / (safe_forward * tan_half_v)
    raw_grid = torch.stack((grid_x, grid_y), dim=-1)
    visible = (
        (camera_forward > config.camera_near_m)
        & (grid_x >= -1.0)
        & (grid_x <= 1.0)
        & (grid_y >= -1.0)
        & (grid_y <= 1.0)
    )
    safe_grid = torch.where(
        visible[..., None], raw_grid, torch.full_like(raw_grid, 2.0)
    )
    return (
        safe_grid.to(torch.float32),
        visible,
        origin.to(torch.float32),
        basis.to(torch.float32),
        ground.to(torch.float32),
    )


class GeometryAnchoredDeformableBevLiftV1(nn.Module):
    """Fixed-anchor, four-sample, bounded local RGB-token to BEV lift."""

    def __init__(
        self,
        config: GeometryAnchoredDeformableBevLiftJointJepaV1Config,
    ) -> None:
        super().__init__()
        self.config = config
        anchor_grid, visible, origin, basis, ground = _fixed_projective_anchor(config)
        self.register_buffer("anchor_grid_xy", anchor_grid, persistent=True)
        self.register_buffer("anchor_in_frustum", visible, persistent=True)
        self.register_buffer("camera_origin_xyz_m", origin, persistent=True)
        self.register_buffer("camera_basis_forward_right_up", basis, persistent=True)
        self.register_buffer("bev_ground_xyz_m", ground, persistent=True)
        self.register_buffer(
            "ground_z_m",
            torch.tensor(config.ground_z_m, dtype=torch.float32),
            persistent=True,
        )
        self.register_buffer(
            "horizontal_fov_degrees",
            torch.tensor(config.horizontal_fov_degrees, dtype=torch.float32),
            persistent=True,
        )
        self.register_buffer(
            "vertical_fov_degrees",
            torch.tensor(config.vertical_fov_degrees, dtype=torch.float32),
            persistent=True,
        )
        self.register_buffer(
            "camera_near_m",
            torch.tensor(config.camera_near_m, dtype=torch.float32),
            persistent=True,
        )

        self.token_projection = nn.Conv2d(
            config.encoder_dim, config.bev_dim, kernel_size=1, bias=True
        )
        initial_raw_offset = math.atanh(0.25)
        symmetric_offsets = torch.tensor(
            (
                (-initial_raw_offset, -initial_raw_offset),
                (-initial_raw_offset, initial_raw_offset),
                (initial_raw_offset, -initial_raw_offset),
                (initial_raw_offset, initial_raw_offset),
            ),
            dtype=torch.float32,
        )
        self.raw_offsets = nn.Parameter(
            symmetric_offsets[None, None].expand(
                *config.bev_size, config.samples_per_cell, 2
            ).clone()
        )
        self.weight_logits = nn.Parameter(
            torch.zeros(
                *config.bev_size,
                config.samples_per_cell,
                dtype=torch.float32,
            )
        )
        self.null_evidence = nn.Parameter(torch.zeros(config.bev_dim))
        self.refinement_blocks = nn.ModuleList(
            [_LocalResidualBlockV1(config.bev_dim) for _ in range(2)]
        )

    def _validate_tokens(self, patch_tokens: torch.Tensor) -> None:
        expected_tokens = self.config.token_side * self.config.token_side
        if patch_tokens.ndim != 3 or tuple(patch_tokens.shape[1:]) != (
            expected_tokens,
            self.config.encoder_dim,
        ):
            raise ValueError(
                "patch_tokens must have shape "
                f"(B,{expected_tokens},{self.config.encoder_dim})"
            )
        if patch_tokens.shape[0] < 1:
            raise ValueError("patch_tokens must contain at least one row")
        if patch_tokens.dtype != torch.float32:
            raise TypeError("patch_tokens must use exact float32")
        if not bool(torch.isfinite(patch_tokens).all()):
            raise FloatingPointError("patch_tokens are nonfinite")

    def forward_with_sampling(
        self, patch_tokens: torch.Tensor
    ) -> GeometryAnchoredBevSamplingV1:
        self._validate_tokens(patch_tokens)
        batch = patch_tokens.shape[0]
        height, width = self.config.bev_size
        samples = self.config.samples_per_cell
        token_map = patch_tokens.transpose(1, 2).reshape(
            batch,
            self.config.encoder_dim,
            self.config.token_side,
            self.config.token_side,
        )
        projected = self.token_projection(token_map)
        anchor_grid = self.anchor_grid_xy.to(dtype=projected.dtype)[None].expand(
            batch, -1, -1, -1
        )
        raw_offsets = self.raw_offsets[None].expand(batch, -1, -1, -1, -1)
        raw_weight_logits = self.weight_logits[None].expand(batch, -1, -1, -1)
        offsets = self.config.offset_radius_token_cells * torch.tanh(raw_offsets)
        normalized_offsets = offsets * (2.0 / self.config.token_side)
        proposed_grid = anchor_grid[..., None, :] + normalized_offsets
        anchor_visible = self.anchor_in_frustum[None].expand(batch, -1, -1)
        within_grid = (
            (proposed_grid[..., 0] >= -1.0)
            & (proposed_grid[..., 0] <= 1.0)
            & (proposed_grid[..., 1] >= -1.0)
            & (proposed_grid[..., 1] <= 1.0)
        )
        sample_valid = anchor_visible[..., None] & within_grid
        safe_sample_grid = torch.where(
            sample_valid[..., None],
            proposed_grid,
            torch.full_like(proposed_grid, 2.0),
        )
        packed_grid = safe_sample_grid.reshape(batch, height, width * samples, 2)
        sampled = F.grid_sample(
            projected,
            packed_grid,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=False,
        )
        sampled = sampled.reshape(
            batch, self.config.bev_dim, height, width, samples
        )
        unmasked_weights = torch.softmax(raw_weight_logits, dim=-1)
        weights = unmasked_weights * sample_valid.to(unmasked_weights.dtype)
        weight_sum = weights.sum(dim=-1, keepdim=True)
        weights = torch.where(
            weight_sum > 0.0,
            weights / weight_sum.clamp_min(torch.finfo(weights.dtype).tiny),
            torch.zeros_like(weights),
        )
        lifted = (sampled * weights.unsqueeze(1)).sum(dim=-1)
        cell_valid = sample_valid.any(dim=-1)
        null = self.null_evidence[None, :, None, None].to(dtype=lifted.dtype)
        lifted = torch.where(cell_valid[:, None], lifted, null)
        for block in self.refinement_blocks:
            lifted = block(lifted)
            lifted = torch.where(cell_valid[:, None], lifted, null)
        return GeometryAnchoredBevSamplingV1(
            latent=lifted,
            anchor_in_frustum=anchor_visible,
            sample_valid_mask=sample_valid,
            cell_valid_mask=cell_valid,
            sample_grid_xy=safe_sample_grid,
            offsets_token_cells=offsets,
            sample_weights=weights,
        )

    def forward(self, patch_tokens: torch.Tensor) -> torch.Tensor:
        return self.forward_with_sampling(patch_tokens).latent


def _validate_action_one_hot(
    action_one_hot: torch.Tensor,
    *,
    batch: int,
    reference: torch.Tensor,
) -> torch.Tensor:
    if action_one_hot.shape != (batch, 9):
        raise ValueError("action_one_hot must have shape (B,9)")
    if action_one_hot.dtype != reference.dtype or not action_one_hot.is_floating_point():
        raise TypeError("action_one_hot and latent must share a floating dtype")
    if action_one_hot.device != reference.device:
        raise TypeError("action_one_hot and latent must share a device")
    if not bool(torch.isfinite(action_one_hot).all()):
        raise FloatingPointError("action_one_hot is nonfinite")
    if not bool(((action_one_hot == 0.0) | (action_one_hot == 1.0)).all()):
        raise ValueError("action_one_hot must contain exact zeros and ones")
    if not torch.equal(
        action_one_hot.sum(dim=1),
        torch.ones(batch, dtype=reference.dtype, device=reference.device),
    ):
        raise ValueError("each action row must contain exactly one active action")
    return action_one_hot.argmax(dim=1)


class _LocalActionConditionedPredictorV1(nn.Module):
    """Two-block local action-conditioned residual predictor."""

    def __init__(
        self, config: GeometryAnchoredDeformableBevLiftJointJepaV1Config
    ) -> None:
        super().__init__()
        self.config = config
        self.action_embedding = nn.Embedding(config.action_dim, config.bev_dim)
        self.input_projection = nn.Conv2d(
            config.bev_dim * 2,
            config.bev_dim,
            kernel_size=3,
            stride=1,
            padding=1,
        )
        self.residual_blocks = nn.ModuleList(
            [_LocalResidualBlockV1(config.bev_dim) for _ in range(2)]
        )
        self.residual_head = nn.Conv2d(
            config.bev_dim,
            config.bev_dim,
            kernel_size=3,
            stride=1,
            padding=1,
        )
        nn.init.zeros_(self.residual_head.weight)
        nn.init.zeros_(self.residual_head.bias)

    def forward(
        self, current_latent: torch.Tensor, action_one_hot: torch.Tensor
    ) -> torch.Tensor:
        expected = (self.config.bev_dim, *self.config.bev_size)
        if current_latent.ndim != 4 or tuple(current_latent.shape[1:]) != expected:
            raise ValueError(f"current_latent must have shape (B,{expected})")
        action_indices = _validate_action_one_hot(
            action_one_hot,
            batch=current_latent.shape[0],
            reference=current_latent,
        )
        action = self.action_embedding(action_indices)
        action = action[:, :, None, None].expand_as(current_latent)
        value = self.input_projection(torch.cat((current_latent, action), dim=1))
        for block in self.residual_blocks:
            value = block(value)
        return current_latent + self.residual_head(value)


def final_class_macro_nll_per_row(
    semantic_logits: torch.Tensor,
    target_labels: torch.Tensor,
) -> torch.Tensor:
    """Equal-present-class semantic NLL, independently for each batch row."""

    if semantic_logits.ndim != 4 or semantic_logits.shape[1] != 3:
        raise ValueError("semantic_logits must have shape (B,3,H,W)")
    if target_labels.shape != semantic_logits.shape[:1] + semantic_logits.shape[2:]:
        raise ValueError("target_labels must have shape (B,H,W)")
    if target_labels.device != semantic_logits.device:
        raise TypeError("target_labels and semantic_logits must share a device")
    if target_labels.is_floating_point() or target_labels.dtype == torch.bool:
        raise TypeError("target_labels must use an integer dtype")
    valid = (
        (target_labels == UNKNOWN_CLASS_V1)
        | (target_labels == FREE_CLASS_V1)
        | (target_labels == OCCUPIED_CLASS_V1)
    )
    if not bool(valid.all()):
        raise ValueError("target_labels contain an unsupported final class")
    per_cell = F.cross_entropy(semantic_logits, target_labels.long(), reduction="none")
    rows: list[torch.Tensor] = []
    for row_index in range(semantic_logits.shape[0]):
        class_means = [
            per_cell[row_index][target_labels[row_index] == class_index].mean()
            for class_index in range(3)
            if bool((target_labels[row_index] == class_index).any())
        ]
        rows.append(torch.stack(class_means).mean())
    return torch.stack(rows)


def latent_energy_per_row(
    predicted_latent: torch.Tensor,
    target_latent: torch.Tensor,
) -> torch.Tensor:
    """Per-row LayerNorm Smooth-L1 energy for B or B-by-action latents."""

    if predicted_latent.shape != target_latent.shape:
        raise ValueError("predicted_latent and target_latent shapes differ")
    if predicted_latent.ndim not in (4, 5) or predicted_latent.shape[-3] != 64:
        raise ValueError("latents must have shape (B,64,H,W) or (B,A,64,H,W)")
    if predicted_latent.dtype != target_latent.dtype:
        raise TypeError("predicted_latent and target_latent dtypes differ")
    predicted = predicted_latent.movedim(-3, -1)
    target = target_latent.detach().movedim(-3, -1)
    predicted = F.layer_norm(predicted, (predicted.shape[-1],))
    target = F.layer_norm(target, (target.shape[-1],))
    loss = F.smooth_l1_loss(predicted, target, beta=1.0, reduction="none")
    return loss.mean(dim=(-3, -2, -1))


class GeometryAnchoredDeformableBevLiftJointJepaV1(nn.Module):
    """RGB-only local BEV representation with an EMA joint-JEPA target."""

    def __init__(
        self,
        n320_encoder_state_dict: Mapping[str, torch.Tensor],
        config: GeometryAnchoredDeformableBevLiftJointJepaV1Config | None = None,
    ) -> None:
        super().__init__()
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
            self.predictor = _LocalActionConditionedPredictorV1(self.config)
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

    @property
    def action_vocabulary(self) -> tuple[str, ...]:
        return ACTION_VOCABULARY_V1

    def online_target_modules(self) -> tuple[nn.Module, nn.Module]:
        """Online modules that have EMA counterparts, in binding order."""

        return self.encoder, self.bev_lift

    def target_modules(self) -> tuple[nn.Module, nn.Module]:
        """EMA target modules, in the same binding order as online modules."""

        return self.target_encoder, self.target_bev_lift

    def iter_online_target_modules(self) -> Iterator[nn.Module]:
        return iter(self.online_target_modules())

    def iter_target_modules(self) -> Iterator[nn.Module]:
        return iter(self.target_modules())

    def _freeze_target(self) -> None:
        for module in self.target_modules():
            module.requires_grad_(False)
            module.eval()

    def train(
        self, mode: bool = True
    ) -> GeometryAnchoredDeformableBevLiftJointJepaV1:
        super().train(mode)
        self._freeze_target()
        return self

    @torch.no_grad()
    def hard_sync_target_from_online(self) -> None:
        for target, online in zip(
            self.target_modules(), self.online_target_modules(), strict=True
        ):
            target.load_state_dict(online.state_dict(), strict=True)
        self.target_hard_sync_count.add_(1)
        self.ema_update_count.zero_()
        self._freeze_target()

    @torch.no_grad()
    def update_target_ema_after_optimizer_step(self) -> None:
        momentum = self.config.target_ema_momentum
        for target_module, online_module in zip(
            self.target_modules(), self.online_target_modules(), strict=True
        ):
            target_parameters = dict(target_module.named_parameters())
            online_parameters = dict(online_module.named_parameters())
            if target_parameters.keys() != online_parameters.keys():
                raise RuntimeError("online and target parameter inventories differ")
            for name, target in target_parameters.items():
                target.mul_(momentum).add_(
                    online_parameters[name], alpha=1.0 - momentum
                )
            target_buffers = dict(target_module.named_buffers())
            online_buffers = dict(online_module.named_buffers())
            if target_buffers.keys() != online_buffers.keys():
                raise RuntimeError("online and target buffer inventories differ")
            for name, target in target_buffers.items():
                target.copy_(online_buffers[name])
        self.ema_update_count.add_(1)
        self._freeze_target()

    def _validate_rgb(self, rgb: torch.Tensor, *, name: str) -> None:
        expected = (3, self.config.image_size, self.config.image_size)
        if rgb.ndim != 4 or tuple(rgb.shape[1:]) != expected:
            raise ValueError(f"{name} must have shape (B,3,112,112)")
        if rgb.shape[0] < 1:
            raise ValueError(f"{name} must contain at least one row")
        if rgb.dtype != torch.float32:
            raise TypeError(f"{name} must use exact float32")
        if not bool(torch.isfinite(rgb).all()):
            raise FloatingPointError(f"{name} is nonfinite")
        if rgb.device != next(self.parameters()).device:
            raise TypeError(f"{name} and model must share a device")

    @staticmethod
    def _encode(
        rgb: torch.Tensor,
        encoder: VisionEncoder,
        lift: GeometryAnchoredDeformableBevLiftV1,
    ) -> torch.Tensor:
        patch_tokens = encoder.forward_tokens(rgb)[:, 1:]
        return lift(patch_tokens)

    def encode_online(self, rgb: torch.Tensor) -> torch.Tensor:
        self._validate_rgb(rgb, name="online_rgb")
        return self._encode(rgb, self.encoder, self.bev_lift)

    def encode_online_with_sampling(
        self, rgb: torch.Tensor
    ) -> GeometryAnchoredBevSamplingV1:
        self._validate_rgb(rgb, name="online_rgb")
        patch_tokens = self.encoder.forward_tokens(rgb)[:, 1:]
        return self.bev_lift.forward_with_sampling(patch_tokens)

    @torch.no_grad()
    def encode_target(self, rgb: torch.Tensor) -> torch.Tensor:
        self._validate_rgb(rgb, name="target_rgb")
        return self._encode(rgb, self.target_encoder, self.target_bev_lift).detach()

    @torch.no_grad()
    def encode_target_with_sampling(
        self, rgb: torch.Tensor
    ) -> GeometryAnchoredBevSamplingV1:
        self._validate_rgb(rgb, name="target_rgb")
        patch_tokens = self.target_encoder.forward_tokens(rgb)[:, 1:]
        state = self.target_bev_lift.forward_with_sampling(patch_tokens)
        return GeometryAnchoredBevSamplingV1(
            latent=state.latent.detach(),
            anchor_in_frustum=state.anchor_in_frustum,
            sample_valid_mask=state.sample_valid_mask,
            cell_valid_mask=state.cell_valid_mask,
            sample_grid_xy=state.sample_grid_xy,
            offsets_token_cells=state.offsets_token_cells,
            sample_weights=state.sample_weights,
        )

    def semantic_logits_from_latent(self, latent: torch.Tensor) -> torch.Tensor:
        expected = (self.config.bev_dim, *self.config.bev_size)
        if latent.ndim != 4 or tuple(latent.shape[1:]) != expected:
            raise ValueError(f"latent must have shape (B,{expected})")
        logits = self.semantic_head(latent)
        visible = self.bev_lift.anchor_in_frustum[None, None].expand(
            latent.shape[0], 1, -1, -1
        )
        invalid_logits = logits.new_tensor((0.0, -20.0, -20.0))[
            None, :, None, None
        ]
        return torch.where(visible, logits, invalid_logits)

    def online_state(self, rgb: torch.Tensor) -> torch.Tensor:
        return self.semantic_logits_from_latent(self.encode_online(rgb))

    def predict(
        self, current_latent: torch.Tensor, action_one_hot: torch.Tensor
    ) -> torch.Tensor:
        return self.predictor(current_latent, action_one_hot)

    def predict_all_actions(self, current_latent: torch.Tensor) -> torch.Tensor:
        expected = (self.config.bev_dim, *self.config.bev_size)
        if current_latent.ndim != 4 or tuple(current_latent.shape[1:]) != expected:
            raise ValueError(f"current_latent must have shape (B,{expected})")
        batch = current_latent.shape[0]
        repeated = current_latent[:, None].expand(
            -1, self.config.action_dim, -1, -1, -1
        ).reshape(batch * self.config.action_dim, *current_latent.shape[1:])
        actions = torch.eye(
            self.config.action_dim,
            dtype=current_latent.dtype,
            device=current_latent.device,
        )[None].expand(batch, -1, -1).reshape(
            batch * self.config.action_dim, self.config.action_dim
        )
        return self.predict(repeated, actions).reshape(
            batch,
            self.config.action_dim,
            self.config.bev_dim,
            *self.config.bev_size,
        )


final_class_macro_nll_per_row_v1 = final_class_macro_nll_per_row
latent_energy_per_row_v1 = latent_energy_per_row


__all__ = [
    "ACTION_VOCABULARY_V1",
    "FREE_CLASS_V1",
    "GeometryAnchoredBevSamplingV1",
    "GeometryAnchoredDeformableBevLiftJointJepaV1",
    "GeometryAnchoredDeformableBevLiftJointJepaV1Config",
    "GeometryAnchoredDeformableBevLiftV1",
    "OCCUPIED_CLASS_V1",
    "UNKNOWN_CLASS_V1",
    "final_class_macro_nll_per_row",
    "final_class_macro_nll_per_row_v1",
    "latent_energy_per_row",
    "latent_energy_per_row_v1",
]
