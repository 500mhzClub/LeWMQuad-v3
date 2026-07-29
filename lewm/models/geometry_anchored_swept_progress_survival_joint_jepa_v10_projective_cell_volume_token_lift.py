"""V4 joint-JEPA with a projective cell-volume token lift.

V10 preserves the clean V4 encoder, residual-local semantic decoder,
predictor, survival head, and BEV refinement.  Its sole scientific
intervention replaces V9's ground-centre-gated token neighbourhood with 25
independently projected cell-volume supports while retaining V9's attention
projections and attention calculation.
"""
from __future__ import annotations

import copy
import hashlib
import math
from typing import Mapping, NamedTuple

import torch
import torch.nn as nn
import torch.nn.functional as F

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
from lewm.models.geometry_anchored_swept_progress_survival_joint_jepa_v9_content_adaptive_dense_local_token_lift import (
    DENSE_LOCAL_ATTENTION_ADDED_PARAMETER_COUNT_V9,
    DENSE_LOCAL_ATTENTION_HEADS_V9,
    DENSE_LOCAL_ATTENTION_HEAD_WIDTH_V9,
    DENSE_LOCAL_ATTENTION_INITIALIZATION_SEED_V9,
    DENSE_LOCAL_ATTENTION_PARAMETER_TENSOR_COUNT_V9,
    ContentAdaptiveDenseLocalTokenLiftV9,
)


CELL_VOLUME_HORIZONTAL_OFFSETS_M_V10 = (
    (0.0, 0.0),
    (-0.05, -0.05),
    (-0.05, 0.05),
    (0.05, -0.05),
    (0.05, 0.05),
)
CELL_VOLUME_HEIGHTS_M_V10 = (-0.333, -0.133, 0.067, 0.267, 0.467)
CELL_VOLUME_HORIZONTAL_SUPPORT_COUNT_V10 = 5
CELL_VOLUME_HEIGHT_COUNT_V10 = 5
CELL_VOLUME_SUPPORT_COUNT_V10 = 25

# These aliases make the inherited V9 attention identity explicit and
# fail closed below if that frozen implementation changes.
CELL_VOLUME_ATTENTION_INITIALIZATION_SEED_V10 = 20260729
CELL_VOLUME_ATTENTION_HEADS_V10 = 4
CELL_VOLUME_ATTENTION_HEAD_WIDTH_V10 = 16
CELL_VOLUME_ATTENTION_ADDED_PARAMETER_COUNT_V10 = 16_576
CELL_VOLUME_ATTENTION_PARAMETER_TENSOR_COUNT_V10 = 7

CELL_VOLUME_VALID_CELL_COUNT_V10 = 2_062
CELL_VOLUME_VALID_MASK_SHA256_V10 = (
    "4ebbafb6d4dd5fb13b96df978abfa7b81bc2f879b2ba6dec2fcda38dec54e60b"
)
CELL_VOLUME_WITHIN_TWO_METERS_CELL_COUNT_V10 = 1_016
CELL_VOLUME_WITHIN_TWO_METERS_VALID_CELL_COUNT_V10 = 222


class ProjectiveCellVolumeTokenLiftSamplingV10(NamedTuple):
    """Auditable V10 geometry, masked mean, and attention values."""

    latent: torch.Tensor
    anchor_in_frustum: torch.Tensor
    support_valid_mask: torch.Tensor
    cell_valid_mask: torch.Tensor
    support_grid_xy: torch.Tensor
    support_xyz_m: torch.Tensor
    support_offsets_xy_m: torch.Tensor
    support_heights_m: torch.Tensor
    masked_mean: torch.Tensor
    attention_weights: torch.Tensor


def _construct_fixed_cell_volume_geometry_v10(
    config: GeometryAnchoredDeformableBevLiftJointJepaV1Config,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    """Project all 25 registered supports independently in float64."""

    dtype = torch.float64
    forward_centres = torch.linspace(
        config.forward_range_m[0],
        config.forward_range_m[1],
        config.bev_size[0],
        dtype=dtype,
    )
    left_centres = torch.linspace(
        config.left_range_m[0],
        config.left_range_m[1],
        config.bev_size[1],
        dtype=dtype,
    )
    forward_grid, left_grid = torch.meshgrid(
        forward_centres, left_centres, indexing="ij"
    )
    centres_xy = torch.stack((forward_grid, left_grid), dim=-1)
    offsets_xy = torch.tensor(CELL_VOLUME_HORIZONTAL_OFFSETS_M_V10, dtype=dtype)
    heights = torch.tensor(CELL_VOLUME_HEIGHTS_M_V10, dtype=dtype)
    horizontal = centres_xy[..., None, :] + offsets_xy
    support_xyz = torch.empty(
        *config.bev_size,
        CELL_VOLUME_HORIZONTAL_SUPPORT_COUNT_V10,
        CELL_VOLUME_HEIGHT_COUNT_V10,
        3,
        dtype=dtype,
    )
    support_xyz[..., :2] = horizontal[..., :, None, :]
    support_xyz[..., 2] = heights
    support_xyz = support_xyz.reshape(
        *config.bev_size, CELL_VOLUME_SUPPORT_COUNT_V10, 3
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
    relative = support_xyz - origin
    camera_forward = torch.einsum("hwsc,c->hws", relative, basis[0])
    camera_right = torch.einsum("hwsc,c->hws", relative, basis[1])
    camera_up = torch.einsum("hwsc,c->hws", relative, basis[2])
    safe_forward = torch.where(
        camera_forward.abs() > torch.finfo(dtype).eps,
        camera_forward,
        torch.ones_like(camera_forward),
    )
    tan_half_horizontal = math.tan(
        math.radians(config.horizontal_fov_degrees) / 2.0
    )
    tan_half_vertical = math.tan(
        math.radians(config.vertical_fov_degrees) / 2.0
    )
    grid_x = camera_right / (safe_forward * tan_half_horizontal)
    grid_y = -camera_up / (safe_forward * tan_half_vertical)
    raw_grid = torch.stack((grid_x, grid_y), dim=-1)
    support_valid = (
        (camera_forward >= config.camera_near_m)
        & (grid_x >= -1.0)
        & (grid_x <= 1.0)
        & (grid_y >= -1.0)
        & (grid_y <= 1.0)
    )
    safe_grid = torch.where(
        support_valid[..., None], raw_grid, torch.full_like(raw_grid, 2.0)
    )
    cell_valid = support_valid.any(dim=-1)

    if int(cell_valid.sum()) != CELL_VOLUME_VALID_CELL_COUNT_V10:
        raise RuntimeError("V10 cell-volume valid-cell count changed")
    mask_payload = bytes(cell_valid.to(torch.uint8).reshape(-1).tolist())
    if hashlib.sha256(mask_payload).hexdigest() != (
        CELL_VOLUME_VALID_MASK_SHA256_V10
    ):
        raise RuntimeError("V10 cell-volume validity hash changed")
    within_two_metres = torch.linalg.vector_norm(centres_xy, dim=-1) <= 2.0
    if int(within_two_metres.sum()) != (
        CELL_VOLUME_WITHIN_TWO_METERS_CELL_COUNT_V10
    ):
        raise RuntimeError("V10 <=2 m cell count changed")
    if int((within_two_metres & cell_valid).sum()) != (
        CELL_VOLUME_WITHIN_TWO_METERS_VALID_CELL_COUNT_V10
    ):
        raise RuntimeError("V10 <=2 m valid-cell count changed")

    return (
        safe_grid.to(torch.float32),
        support_valid,
        cell_valid,
        support_xyz.to(torch.float32),
        offsets_xy.to(torch.float32),
        heights.to(torch.float32),
    )


class ProjectiveCellVolumeTokenLiftV10(ContentAdaptiveDenseLocalTokenLiftV9):
    """V9 attention over independently projected cell-volume supports."""

    def __init__(self, v4_lift: nn.Module) -> None:
        super().__init__(v4_lift)
        inherited_attention = (
            DENSE_LOCAL_ATTENTION_INITIALIZATION_SEED_V9,
            DENSE_LOCAL_ATTENTION_HEADS_V9,
            DENSE_LOCAL_ATTENTION_HEAD_WIDTH_V9,
            DENSE_LOCAL_ATTENTION_ADDED_PARAMETER_COUNT_V9,
            DENSE_LOCAL_ATTENTION_PARAMETER_TENSOR_COUNT_V9,
        )
        frozen_attention = (
            CELL_VOLUME_ATTENTION_INITIALIZATION_SEED_V10,
            CELL_VOLUME_ATTENTION_HEADS_V10,
            CELL_VOLUME_ATTENTION_HEAD_WIDTH_V10,
            CELL_VOLUME_ATTENTION_ADDED_PARAMETER_COUNT_V10,
            CELL_VOLUME_ATTENTION_PARAMETER_TENSOR_COUNT_V10,
        )
        if inherited_attention != frozen_attention:
            raise RuntimeError("V9 attention constants changed")

        del self.support_offsets_token_cells
        (
            support_grid,
            support_valid,
            cell_valid,
            support_xyz,
            offsets_xy,
            heights,
        ) = _construct_fixed_cell_volume_geometry_v10(self.config)
        self.register_buffer("support_grid_xy", support_grid, persistent=True)
        self.register_buffer(
            "support_valid_mask", support_valid, persistent=True
        )
        self.register_buffer("cell_valid_mask", cell_valid, persistent=True)
        self.register_buffer("support_xyz_m", support_xyz, persistent=True)
        self.register_buffer(
            "support_offsets_xy_m", offsets_xy, persistent=True
        )
        self.register_buffer("support_heights_m", heights, persistent=True)

    def _attend_valid_cells(
        self,
        masked_mean: torch.Tensor,
        support_samples: torch.Tensor,
        support_valid: torch.Tensor,
        cell_valid: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply V9's unchanged Q/K/V/O attention to valid V10 cells."""

        flat_mean = masked_mean.reshape(-1, self.config.bev_dim)
        flat_supports = support_samples.reshape(
            -1, CELL_VOLUME_SUPPORT_COUNT_V10, self.config.bev_dim
        )
        flat_support_valid = support_valid.reshape(
            -1, CELL_VOLUME_SUPPORT_COUNT_V10
        )
        flat_cell_valid = cell_valid.reshape(-1)
        valid_indices = torch.nonzero(flat_cell_valid, as_tuple=False).flatten()

        outputs = flat_mean.new_zeros(flat_mean.shape)
        report_weights = flat_mean.new_zeros(
            (
                flat_mean.shape[0],
                CELL_VOLUME_ATTENTION_HEADS_V10,
                CELL_VOLUME_SUPPORT_COUNT_V10,
            )
        )
        if valid_indices.numel() == 0:
            return outputs, report_weights

        means = flat_mean.index_select(0, valid_indices)
        supports = flat_supports.index_select(0, valid_indices)
        valid = flat_support_valid.index_select(0, valid_indices)
        if not bool(valid.any(dim=-1).all()):
            raise RuntimeError("valid V10 cell has no valid support")

        query = self.query_projection(means).reshape(
            -1,
            CELL_VOLUME_ATTENTION_HEADS_V10,
            CELL_VOLUME_ATTENTION_HEAD_WIDTH_V10,
        )
        key = self.key_projection(supports).reshape(
            -1,
            CELL_VOLUME_SUPPORT_COUNT_V10,
            CELL_VOLUME_ATTENTION_HEADS_V10,
            CELL_VOLUME_ATTENTION_HEAD_WIDTH_V10,
        )
        value = self.value_projection(supports).reshape_as(key)
        logits = torch.einsum("nhd,nshd->nhs", query, key)
        logits = logits * (
            1.0 / math.sqrt(CELL_VOLUME_ATTENTION_HEAD_WIDTH_V10)
        )
        logits = logits.masked_fill(~valid[:, None, :], float("-inf"))
        weights = torch.softmax(logits, dim=-1)
        attended = torch.einsum("nhs,nshd->nhd", weights, value).reshape(
            -1, self.config.bev_dim
        )
        attended = self.output_projection(attended)
        outputs = outputs.index_copy(0, valid_indices, attended)
        report_weights = report_weights.index_copy(0, valid_indices, weights)
        return outputs, report_weights

    def forward_with_sampling(
        self, patch_tokens: torch.Tensor
    ) -> ProjectiveCellVolumeTokenLiftSamplingV10:
        self._validate_tokens(patch_tokens)
        batch = patch_tokens.shape[0]
        height, width = self.config.bev_size
        token_map = patch_tokens.transpose(1, 2).reshape(
            batch,
            self.config.encoder_dim,
            self.config.token_side,
            self.config.token_side,
        )
        projected = self.token_projection(token_map)

        support_grid = self.support_grid_xy.to(dtype=projected.dtype)[None].expand(
            batch, -1, -1, -1, -1
        )
        support_valid = self.support_valid_mask[None].expand(
            batch, -1, -1, -1
        )
        cell_valid = self.cell_valid_mask[None].expand(batch, -1, -1)
        packed_grid = support_grid.reshape(
            batch, height, width * CELL_VOLUME_SUPPORT_COUNT_V10, 2
        )
        sampled = F.grid_sample(
            projected,
            packed_grid,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=False,
        ).reshape(
            batch,
            self.config.bev_dim,
            height,
            width,
            CELL_VOLUME_SUPPORT_COUNT_V10,
        )
        support_samples = sampled.permute(0, 2, 3, 4, 1).contiguous()
        support_samples = torch.where(
            support_valid[..., None],
            support_samples,
            torch.zeros_like(support_samples),
        )
        valid_count = support_valid.sum(dim=-1, keepdim=True).to(
            dtype=support_samples.dtype
        )
        masked_mean = support_samples.sum(dim=-2) / valid_count.clamp_min(1.0)
        attention_output, attention_weights = self._attend_valid_cells(
            masked_mean,
            support_samples,
            support_valid,
            cell_valid,
        )
        attention_output = attention_output.reshape(
            batch, height, width, self.config.bev_dim
        )
        attention_weights = attention_weights.reshape(
            batch,
            height,
            width,
            CELL_VOLUME_ATTENTION_HEADS_V10,
            CELL_VOLUME_SUPPORT_COUNT_V10,
        )
        lifted = (masked_mean + attention_output).permute(0, 3, 1, 2)
        null = self.null_evidence[None, :, None, None].to(dtype=lifted.dtype)
        lifted = torch.where(cell_valid[:, None], lifted, null)
        for block in self.refinement_blocks:
            lifted = block(lifted)
            lifted = torch.where(cell_valid[:, None], lifted, null)

        if not bool(torch.isfinite(masked_mean).all()):
            raise FloatingPointError("V10 masked mean is nonfinite")
        if not bool(torch.isfinite(lifted).all()):
            raise FloatingPointError("V10 lift latent is nonfinite")
        if not bool(torch.isfinite(attention_weights).all()):
            raise FloatingPointError("V10 attention weights are nonfinite")
        anchor_visible = self.anchor_in_frustum[None].expand(batch, -1, -1)
        return ProjectiveCellVolumeTokenLiftSamplingV10(
            latent=lifted,
            anchor_in_frustum=anchor_visible,
            support_valid_mask=support_valid,
            cell_valid_mask=cell_valid,
            support_grid_xy=support_grid,
            support_xyz_m=self.support_xyz_m,
            support_offsets_xy_m=self.support_offsets_xy_m,
            support_heights_m=self.support_heights_m,
            masked_mean=masked_mean,
            attention_weights=attention_weights,
        )


# Descriptive compatibility alias retained for callers that spell out BEV.
ProjectiveCellVolumeTokenBevLiftV10 = ProjectiveCellVolumeTokenLiftV10


class GeometryAnchoredSweptProgressSurvivalJointJepaV10(
    GeometryAnchoredSweptProgressSurvivalJointJepaV4
):
    """Clean V4 with only V9's support routing replaced by V10 geometry."""

    def __init__(
        self,
        n320_encoder_state_dict: Mapping[str, torch.Tensor],
        sweep_masks: torch.Tensor,
        config: GeometryAnchoredDeformableBevLiftJointJepaV1Config | None = None,
    ) -> None:
        super().__init__(n320_encoder_state_dict, sweep_masks, config)
        if int(self.target_hard_sync_count.item()) != 1:
            raise RuntimeError("clean V4 initial hard-sync count changed")
        if int(self.ema_update_count.item()) != 0:
            raise RuntimeError("clean V4 initial EMA count changed")
        self.bev_lift = ProjectiveCellVolumeTokenLiftV10(self.bev_lift)
        self.target_bev_lift = copy.deepcopy(self.bev_lift)
        self._freeze_target()

    def encode_online_with_sampling(
        self, rgb: torch.Tensor
    ) -> ProjectiveCellVolumeTokenLiftSamplingV10:
        self._validate_rgb(rgb, name="online_rgb")
        patch_tokens = self.encoder.forward_tokens(rgb)[:, 1:]
        return self.bev_lift.forward_with_sampling(patch_tokens)

    @torch.no_grad()
    def encode_target_with_sampling(
        self, rgb: torch.Tensor
    ) -> ProjectiveCellVolumeTokenLiftSamplingV10:
        self._validate_rgb(rgb, name="target_rgb")
        patch_tokens = self.target_encoder.forward_tokens(rgb)[:, 1:]
        state = self.target_bev_lift.forward_with_sampling(patch_tokens)
        return ProjectiveCellVolumeTokenLiftSamplingV10(
            latent=state.latent.detach(),
            anchor_in_frustum=state.anchor_in_frustum,
            support_valid_mask=state.support_valid_mask,
            cell_valid_mask=state.cell_valid_mask,
            support_grid_xy=state.support_grid_xy,
            support_xyz_m=state.support_xyz_m,
            support_offsets_xy_m=state.support_offsets_xy_m,
            support_heights_m=state.support_heights_m,
            masked_mean=state.masked_mean.detach(),
            attention_weights=state.attention_weights.detach(),
        )

    def semantic_logits_from_latent(self, latent: torch.Tensor) -> torch.Tensor:
        expected = (self.config.bev_dim, *self.config.bev_size)
        if latent.ndim != 4 or tuple(latent.shape[1:]) != expected:
            raise ValueError(f"latent must have shape (B,{expected})")
        logits = self.semantic_head(latent)
        valid = self.bev_lift.cell_valid_mask[None, None].expand(
            latent.shape[0], 1, -1, -1
        )
        invalid_logits = logits.new_tensor((0.0, -20.0, -20.0))[
            None, :, None, None
        ]
        return torch.where(valid, logits, invalid_logits)


GeometryAnchoredSweptProgressSurvivalJointJepaV10Config = (
    GeometryAnchoredDeformableBevLiftJointJepaV1Config
)
# The frozen runner resolves this historical name from its selected model module.
GeometryAnchoredDeformableBevLiftJointJepaV1 = (
    GeometryAnchoredSweptProgressSurvivalJointJepaV10
)
final_class_macro_nll_per_row_v1 = final_class_macro_nll_per_row
latent_energy_per_row_v1 = latent_energy_per_row


__all__ = [
    "ACTION_VOCABULARY_V1",
    "CELL_VOLUME_ATTENTION_ADDED_PARAMETER_COUNT_V10",
    "CELL_VOLUME_ATTENTION_HEADS_V10",
    "CELL_VOLUME_ATTENTION_HEAD_WIDTH_V10",
    "CELL_VOLUME_ATTENTION_INITIALIZATION_SEED_V10",
    "CELL_VOLUME_ATTENTION_PARAMETER_TENSOR_COUNT_V10",
    "CELL_VOLUME_HEIGHT_COUNT_V10",
    "CELL_VOLUME_HEIGHTS_M_V10",
    "CELL_VOLUME_HORIZONTAL_OFFSETS_M_V10",
    "CELL_VOLUME_HORIZONTAL_SUPPORT_COUNT_V10",
    "CELL_VOLUME_SUPPORT_COUNT_V10",
    "CELL_VOLUME_VALID_CELL_COUNT_V10",
    "CELL_VOLUME_VALID_MASK_SHA256_V10",
    "CELL_VOLUME_WITHIN_TWO_METERS_CELL_COUNT_V10",
    "CELL_VOLUME_WITHIN_TWO_METERS_VALID_CELL_COUNT_V10",
    "FREE_CLASS_V1",
    "GeometryAnchoredBevSamplingV1",
    "GeometryAnchoredDeformableBevLiftJointJepaV1",
    "GeometryAnchoredDeformableBevLiftJointJepaV1Config",
    "GeometryAnchoredSweptProgressSurvivalJointJepaV10",
    "GeometryAnchoredSweptProgressSurvivalJointJepaV10Config",
    "OCCUPIED_CLASS_V1",
    "ProjectiveCellVolumeTokenBevLiftV10",
    "ProjectiveCellVolumeTokenLiftSamplingV10",
    "ProjectiveCellVolumeTokenLiftV10",
    "SWEEP_PROGRESS_BIN_COUNT_V1",
    "SweptProgressSurvivalHeadV1",
    "SweptProgressSurvivalPredictionV1",
    "UNKNOWN_CLASS_V1",
    "final_class_macro_nll_per_row",
    "final_class_macro_nll_per_row_v1",
    "latent_energy_per_row",
    "latent_energy_per_row_v1",
]
