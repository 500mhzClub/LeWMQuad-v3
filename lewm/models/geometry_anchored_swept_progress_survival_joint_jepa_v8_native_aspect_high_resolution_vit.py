"""V4 swept-progress joint JEPA with native-aspect high-resolution ViT tokens.

V8 preserves the accepted V4 transformer and every non-positional encoder
tensor.  It decodes the camera at its native 168-by-224 aspect, forms a
24-by-32 patch-7 lattice, and adapts only the inherited BEV lift's token-map
shape.  The proposed sampling grid deliberately retains V4's exact legacy
offset arithmetic.
"""
from __future__ import annotations

import copy
from typing import Mapping

import torch
import torch.nn as nn
import torch.nn.functional as F

from lewm.models.encoders import VisionEncoder
from lewm.models.geometry_anchored_deformable_bev_lift_joint_jepa_v1 import (
    GeometryAnchoredDeformableBevLiftV1,
)
from lewm.models.geometry_anchored_swept_progress_survival_joint_jepa_v4_residual_local_semantic_decoder import (
    ACTION_VOCABULARY_V1,
    FREE_CLASS_V1,
    OCCUPIED_CLASS_V1,
    SWEEP_PROGRESS_BIN_COUNT_V1,
    UNKNOWN_CLASS_V1,
    GeometryAnchoredBevSamplingV1,
    GeometryAnchoredDeformableBevLiftJointJepaV1Config,
    GeometryAnchoredSweptProgressSurvivalJointJepaV4,
    SweptProgressSurvivalHeadV1,
    SweptProgressSurvivalPredictionV1,
    final_class_macro_nll_per_row,
    latent_energy_per_row,
)


NATIVE_IMAGE_HEIGHT_V8 = 168
NATIVE_IMAGE_WIDTH_V8 = 224
PATCH_SIZE_V8 = 7
TOKEN_HEIGHT_V8 = 24
TOKEN_WIDTH_V8 = 32
SPATIAL_TOKEN_COUNT_V8 = 768
TOKEN_COUNT_WITH_CLS_V8 = 769
ENCODER_DIM_V8 = 192
NATIVE_ASPECT_HIGH_RESOLUTION_ENCODER_TRAINABLE_PARAMETER_COUNT_V8 = 2_845_824
NATIVE_TOKEN_CELL_RADII_XY_V8 = (4.0, 3.0)
# Frozen public integration names used by the V8 executor.
NATIVE_TOKEN_HEIGHT_V8 = TOKEN_HEIGHT_V8
NATIVE_TOKEN_WIDTH_V8 = TOKEN_WIDTH_V8
NATIVE_SPATIAL_TOKEN_COUNT_V8 = SPATIAL_TOKEN_COUNT_V8
NATIVE_ENCODER_TRAINABLE_PARAMETER_COUNT_V8 = (
    NATIVE_ASPECT_HIGH_RESOLUTION_ENCODER_TRAINABLE_PARAMETER_COUNT_V8
)


def resize_v4_positional_embedding_v8(position: torch.Tensor) -> torch.Tensor:
    """Resize V4's spatial positions once on CPU float32, retaining CLS."""

    if not isinstance(position, torch.Tensor):
        raise TypeError("position must be a tensor")
    if tuple(position.shape) != (1, 257, ENCODER_DIM_V8):
        raise ValueError("V4 position must have shape (1,257,192)")
    if position.device.type != "cpu" or position.dtype != torch.float32:
        raise TypeError("V4 position must use CPU float32")
    if not bool(torch.isfinite(position).all()):
        raise FloatingPointError("V4 position is nonfinite")

    cls_position = position[:, :1].detach().clone()
    spatial = (
        position[:, 1:]
        .detach()
        .reshape(1, 16, 16, ENCODER_DIM_V8)
        .permute(0, 3, 1, 2)
        .contiguous()
    )
    resized = F.interpolate(
        spatial,
        size=(TOKEN_HEIGHT_V8, TOKEN_WIDTH_V8),
        mode="bicubic",
        align_corners=False,
        antialias=False,
    )
    spatial_position = resized.flatten(start_dim=2).transpose(1, 2).contiguous()
    result = torch.cat((cls_position, spatial_position), dim=1)
    if tuple(result.shape) != (1, TOKEN_COUNT_WITH_CLS_V8, ENCODER_DIM_V8):
        raise RuntimeError("resized V8 positional shape changed")
    if result.device.type != "cpu" or result.dtype != torch.float32:
        raise TypeError("resized V8 position must use CPU float32")
    if not bool(torch.isfinite(result).all()):
        raise FloatingPointError("resized V8 position is nonfinite")
    return result


def _validate_v4_encoder_v8(encoder: VisionEncoder) -> None:
    if not isinstance(encoder, VisionEncoder):
        raise TypeError("V8 requires the clean V4 VisionEncoder")
    if (
        encoder.image_size,
        encoder.patch_size,
        encoder.hidden_dim,
        encoder.num_patches,
        len(encoder.blocks),
    ) != (112, 7, 192, 256, 6):
        raise ValueError("clean V4 encoder architecture changed")
    if encoder.pos_drop.p != 0.0:
        raise ValueError("clean V4 encoder dropout changed")
    if encoder.pos_embed.device.type != "cpu":
        raise TypeError("clean V4 encoder must be constructed on CPU")
    if any(
        block.attn.embed_dim != 192 or block.attn.num_heads != 6
        for block in encoder.blocks
    ):
        raise ValueError("clean V4 global-attention blocks changed")


class NativeAspectHighResolutionVisionEncoderV8(nn.Module):
    """Exact V4 ViT weights on a native 24-by-32 patch lattice."""

    image_height = NATIVE_IMAGE_HEIGHT_V8
    image_width = NATIVE_IMAGE_WIDTH_V8
    image_size = (NATIVE_IMAGE_HEIGHT_V8, NATIVE_IMAGE_WIDTH_V8)
    patch_size = PATCH_SIZE_V8
    hidden_dim = ENCODER_DIM_V8
    token_height = TOKEN_HEIGHT_V8
    token_width = TOKEN_WIDTH_V8
    num_patches = SPATIAL_TOKEN_COUNT_V8

    def __init__(self, v4_encoder: VisionEncoder) -> None:
        super().__init__()
        _validate_v4_encoder_v8(v4_encoder)

        # Deep copies preserve the accepted state without any initialization or
        # RNG draw.  Only the spatial positional tensor is deterministically
        # resized below.
        self.patch_embed = copy.deepcopy(v4_encoder.patch_embed)
        self.cls_token = nn.Parameter(v4_encoder.cls_token.detach().clone())
        self.pos_embed = nn.Parameter(
            resize_v4_positional_embedding_v8(v4_encoder.pos_embed)
        )
        self.pos_drop = copy.deepcopy(v4_encoder.pos_drop)
        self.blocks = copy.deepcopy(v4_encoder.blocks)
        self.norm = copy.deepcopy(v4_encoder.norm)

        parameter_count = sum(parameter.numel() for parameter in self.parameters())
        if parameter_count != (
            NATIVE_ASPECT_HIGH_RESOLUTION_ENCODER_TRAINABLE_PARAMETER_COUNT_V8
        ):
            raise RuntimeError("native-aspect V8 encoder parameter count changed")

        source_state = v4_encoder.state_dict()
        migrated_state = self.state_dict()
        if source_state.keys() != migrated_state.keys():
            raise RuntimeError("V8 encoder state inventory changed")
        for name, source in source_state.items():
            if name == "pos_embed":
                continue
            if not torch.equal(source, migrated_state[name]):
                raise RuntimeError(f"V8 encoder changed inherited tensor {name!r}")

    def _validate_rgb(self, rgb: torch.Tensor) -> None:
        if not isinstance(rgb, torch.Tensor):
            raise TypeError("rgb must be a tensor")
        if rgb.ndim != 4 or tuple(rgb.shape[1:]) != (
            3,
            NATIVE_IMAGE_HEIGHT_V8,
            NATIVE_IMAGE_WIDTH_V8,
        ):
            raise ValueError("rgb must have shape (B,3,168,224)")
        if rgb.shape[0] < 1:
            raise ValueError("rgb must contain at least one row")
        if rgb.dtype != torch.float32:
            raise TypeError("rgb must use exact float32")
        if rgb.device != next(self.parameters()).device:
            raise TypeError("rgb and encoder must share a device")
        if not bool(torch.isfinite(rgb).all()):
            raise FloatingPointError("rgb is nonfinite")

    def forward_tokens(self, rgb: torch.Tensor) -> torch.Tensor:
        """Return normalized CLS and 24-by-32 row-major spatial tokens."""

        self._validate_rgb(rgb)
        patch_map = self.patch_embed(rgb)
        if tuple(patch_map.shape[1:]) != (
            ENCODER_DIM_V8,
            TOKEN_HEIGHT_V8,
            TOKEN_WIDTH_V8,
        ):
            raise RuntimeError("native V8 patch-map shape changed")
        spatial = patch_map.flatten(start_dim=2).transpose(1, 2)
        cls = self.cls_token.expand(spatial.shape[0], -1, -1)
        value = torch.cat((cls, spatial), dim=1)
        value = self.pos_drop(value + self.pos_embed)
        for block in self.blocks:
            value = block(value)
        result = self.norm(value)
        if tuple(result.shape[1:]) != (
            TOKEN_COUNT_WITH_CLS_V8,
            ENCODER_DIM_V8,
        ):
            raise RuntimeError("native V8 token shape changed")
        if result.dtype != torch.float32 or not bool(torch.isfinite(result).all()):
            raise FloatingPointError("native V8 tokens are nonfinite")
        return result

    def forward(self, rgb: torch.Tensor) -> torch.Tensor:
        return self.forward_tokens(rgb)[:, 0]


def _validate_v4_lift_v8(lift: GeometryAnchoredDeformableBevLiftV1) -> None:
    if not isinstance(lift, GeometryAnchoredDeformableBevLiftV1):
        raise TypeError("V8 requires the clean V4 deformable BEV lift")
    config = lift.config
    if (
        config.encoder_dim,
        config.token_side,
        config.bev_dim,
        config.bev_size,
        config.samples_per_cell,
        config.offset_radius_token_cells,
    ) != (192, 16, 64, (64, 64), 4, 2.0):
        raise ValueError("clean V4 BEV lift architecture changed")
    if set(dict(lift.named_parameters(recurse=False))) != {
        "raw_offsets",
        "weight_logits",
        "null_evidence",
    }:
        raise RuntimeError("clean V4 direct lift-parameter inventory changed")
    if set(dict(lift.named_children())) != {
        "token_projection",
        "refinement_blocks",
    }:
        raise RuntimeError("clean V4 lift-module inventory changed")


class NativeAspectHighResolutionBevLiftV8(nn.Module):
    """Exact V4 lift state consuming a rectangular 24-by-32 token map."""

    token_height = TOKEN_HEIGHT_V8
    token_width = TOKEN_WIDTH_V8
    native_token_cell_radii_xy = NATIVE_TOKEN_CELL_RADII_XY_V8

    def __init__(self, v4_lift: GeometryAnchoredDeformableBevLiftV1) -> None:
        super().__init__()
        _validate_v4_lift_v8(v4_lift)
        self.config = v4_lift.config

        for name, parameter in v4_lift.named_parameters(recurse=False):
            self.register_parameter(
                name,
                nn.Parameter(
                    parameter.detach().clone(),
                    requires_grad=parameter.requires_grad,
                ),
            )
        for name, buffer in v4_lift.named_buffers(recurse=False):
            self.register_buffer(
                name,
                buffer.detach().clone(),
                persistent=name not in v4_lift._non_persistent_buffers_set,
            )
        for name, module in v4_lift.named_children():
            self.add_module(name, copy.deepcopy(module))

        source_state = v4_lift.state_dict()
        migrated_state = self.state_dict()
        if source_state.keys() != migrated_state.keys():
            raise RuntimeError("V8 lift state inventory changed")
        if any(
            not torch.equal(value, migrated_state[name])
            for name, value in source_state.items()
        ):
            raise RuntimeError("V8 lift state differs from clean V4")

    def _validate_tokens(self, patch_tokens: torch.Tensor) -> None:
        if not isinstance(patch_tokens, torch.Tensor):
            raise TypeError("patch_tokens must be a tensor")
        if patch_tokens.ndim != 3 or tuple(patch_tokens.shape[1:]) != (
            SPATIAL_TOKEN_COUNT_V8,
            ENCODER_DIM_V8,
        ):
            raise ValueError("patch_tokens must have shape (B,768,192)")
        if patch_tokens.shape[0] < 1:
            raise ValueError("patch_tokens must contain at least one row")
        if patch_tokens.dtype != torch.float32:
            raise TypeError("patch_tokens must use exact float32")
        if patch_tokens.device != self.raw_offsets.device:
            raise TypeError("patch_tokens and lift must share a device")
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
            TOKEN_HEIGHT_V8,
            TOKEN_WIDTH_V8,
        )
        projected = self.token_projection(token_map)
        anchor_grid = self.anchor_grid_xy.to(dtype=projected.dtype)[None].expand(
            batch, -1, -1, -1
        )
        raw_offsets = self.raw_offsets[None].expand(batch, -1, -1, -1, -1)
        raw_weight_logits = self.weight_logits[None].expand(batch, -1, -1, -1)

        # These two operations intentionally reproduce V4's arithmetic and
        # operation order exactly.  The proposed normalized grid therefore does
        # not shrink when the token lattice becomes denser and rectangular.
        legacy_offsets = self.config.offset_radius_token_cells * torch.tanh(
            raw_offsets
        )
        normalized_offsets = legacy_offsets * (2.0 / self.config.token_side)
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

        native_radii = raw_offsets.new_tensor(NATIVE_TOKEN_CELL_RADII_XY_V8)
        native_token_cell_offsets = torch.tanh(raw_offsets) * native_radii
        return GeometryAnchoredBevSamplingV1(
            latent=lifted,
            anchor_in_frustum=anchor_visible,
            sample_valid_mask=sample_valid,
            cell_valid_mask=cell_valid,
            sample_grid_xy=safe_sample_grid,
            offsets_token_cells=native_token_cell_offsets,
            sample_weights=weights,
        )

    def forward(self, patch_tokens: torch.Tensor) -> torch.Tensor:
        return self.forward_with_sampling(patch_tokens).latent


# Frozen public integration names; the descriptive names remain compatibility
# aliases for the focused model source and tests.
NativeAspectVisionEncoderV8 = NativeAspectHighResolutionVisionEncoderV8
NativeAspectGeometryAnchoredDeformableBevLiftV8 = (
    NativeAspectHighResolutionBevLiftV8
)


class GeometryAnchoredSweptProgressSurvivalJointJepaV8(
    GeometryAnchoredSweptProgressSurvivalJointJepaV4
):
    """Clean V4 with native-aspect tokens and an exact-state rectangular lift."""

    def __init__(
        self,
        n320_encoder_state_dict: Mapping[str, torch.Tensor],
        sweep_masks: torch.Tensor,
        config: GeometryAnchoredDeformableBevLiftJointJepaV1Config | None = None,
    ) -> None:
        super().__init__(n320_encoder_state_dict, sweep_masks, config)
        if (
            self.config.image_size,
            self.config.patch_size,
            self.config.encoder_dim,
            self.config.encoder_depth,
            self.config.encoder_heads,
            self.config.token_side,
            self.config.offset_radius_token_cells,
        ) != (112, 7, 192, 6, 6, 16, 2.0):
            raise ValueError("V8 requires the exact clean V4 encoder/lift contract")
        if int(self.target_hard_sync_count.item()) != 1:
            raise RuntimeError("clean V4 initial hard-sync count changed")
        if int(self.ema_update_count.item()) != 0:
            raise RuntimeError("clean V4 initial EMA count changed")

        self.encoder = NativeAspectHighResolutionVisionEncoderV8(self.encoder)
        self.bev_lift = NativeAspectHighResolutionBevLiftV8(self.bev_lift)
        self.target_encoder = copy.deepcopy(self.encoder)
        self.target_bev_lift = copy.deepcopy(self.bev_lift)
        self._freeze_target()

    def _validate_rgb(self, rgb: torch.Tensor, *, name: str) -> None:
        if not isinstance(rgb, torch.Tensor):
            raise TypeError(f"{name} must be a tensor")
        if rgb.ndim != 4 or tuple(rgb.shape[1:]) != (
            3,
            NATIVE_IMAGE_HEIGHT_V8,
            NATIVE_IMAGE_WIDTH_V8,
        ):
            raise ValueError(f"{name} must have shape (B,3,168,224)")
        if rgb.shape[0] < 1:
            raise ValueError(f"{name} must contain at least one row")
        if rgb.dtype != torch.float32:
            raise TypeError(f"{name} must use exact float32")
        if not bool(torch.isfinite(rgb).all()):
            raise FloatingPointError(f"{name} is nonfinite")
        if rgb.device != next(self.parameters()).device:
            raise TypeError(f"{name} and model must share a device")


GeometryAnchoredSweptProgressSurvivalJointJepaV8Config = (
    GeometryAnchoredDeformableBevLiftJointJepaV1Config
)
# The frozen runner resolves this historical name from its selected model module.
GeometryAnchoredDeformableBevLiftJointJepaV1 = (
    GeometryAnchoredSweptProgressSurvivalJointJepaV8
)
final_class_macro_nll_per_row_v1 = final_class_macro_nll_per_row
latent_energy_per_row_v1 = latent_energy_per_row


__all__ = [
    "ACTION_VOCABULARY_V1",
    "ENCODER_DIM_V8",
    "FREE_CLASS_V1",
    "GeometryAnchoredBevSamplingV1",
    "GeometryAnchoredDeformableBevLiftJointJepaV1",
    "GeometryAnchoredDeformableBevLiftJointJepaV1Config",
    "GeometryAnchoredSweptProgressSurvivalJointJepaV8",
    "GeometryAnchoredSweptProgressSurvivalJointJepaV8Config",
    "NATIVE_ASPECT_HIGH_RESOLUTION_ENCODER_TRAINABLE_PARAMETER_COUNT_V8",
    "NATIVE_ENCODER_TRAINABLE_PARAMETER_COUNT_V8",
    "NATIVE_IMAGE_HEIGHT_V8",
    "NATIVE_IMAGE_WIDTH_V8",
    "NATIVE_SPATIAL_TOKEN_COUNT_V8",
    "NATIVE_TOKEN_CELL_RADII_XY_V8",
    "NATIVE_TOKEN_HEIGHT_V8",
    "NATIVE_TOKEN_WIDTH_V8",
    "NativeAspectGeometryAnchoredDeformableBevLiftV8",
    "NativeAspectHighResolutionBevLiftV8",
    "NativeAspectHighResolutionVisionEncoderV8",
    "NativeAspectVisionEncoderV8",
    "OCCUPIED_CLASS_V1",
    "PATCH_SIZE_V8",
    "SPATIAL_TOKEN_COUNT_V8",
    "SWEEP_PROGRESS_BIN_COUNT_V1",
    "SweptProgressSurvivalHeadV1",
    "SweptProgressSurvivalPredictionV1",
    "TOKEN_COUNT_WITH_CLS_V8",
    "TOKEN_HEIGHT_V8",
    "TOKEN_WIDTH_V8",
    "UNKNOWN_CLASS_V1",
    "final_class_macro_nll_per_row",
    "final_class_macro_nll_per_row_v1",
    "latent_energy_per_row",
    "latent_energy_per_row_v1",
    "resize_v4_positional_embedding_v8",
]
