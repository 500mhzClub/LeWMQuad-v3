"""V4 joint-JEPA with a content-adaptive dense local token lift.

V9 preserves the clean V4 encoder, semantic decoder, predictor, survival
head, projective ground anchors, and local BEV refinement.  Its sole
scientific intervention replaces the four learned, content-independent
samples with four-head attention over a fixed 5-by-5 token neighbourhood.
"""
from __future__ import annotations

import copy
import math
from typing import Mapping, NamedTuple

import torch
import torch.nn as nn
import torch.nn.functional as F

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


DENSE_LOCAL_ATTENTION_INITIALIZATION_SEED_V9 = 20260729
DENSE_LOCAL_SUPPORT_SIDE_V9 = 5
DENSE_LOCAL_SUPPORT_COUNT_V9 = 25
DENSE_LOCAL_SUPPORT_CENTER_INDEX_V9 = 12
DENSE_LOCAL_ATTENTION_HEADS_V9 = 4
DENSE_LOCAL_ATTENTION_HEAD_WIDTH_V9 = 16
DENSE_LOCAL_ATTENTION_ADDED_PARAMETER_COUNT_V9 = 16_576
DENSE_LOCAL_ATTENTION_PARAMETER_TENSOR_COUNT_V9 = 7


class ContentAdaptiveDenseLocalTokenLiftSamplingV9(NamedTuple):
    """Auditable V9 sampling and per-head local-attention values."""

    latent: torch.Tensor
    anchor_in_frustum: torch.Tensor
    support_valid_mask: torch.Tensor
    cell_valid_mask: torch.Tensor
    support_grid_xy: torch.Tensor
    support_offsets_token_cells: torch.Tensor
    attention_weights: torch.Tensor


def _validate_clean_v4_lift_v9(
    lift: GeometryAnchoredDeformableBevLiftV1,
) -> None:
    if not isinstance(lift, GeometryAnchoredDeformableBevLiftV1):
        raise TypeError("V9 requires the clean V4 deformable BEV lift")
    config = lift.config
    if (
        config.image_size,
        config.patch_size,
        config.encoder_dim,
        config.token_side,
        config.bev_dim,
        config.bev_size,
        config.samples_per_cell,
        config.offset_radius_token_cells,
    ) != (112, 7, 192, 16, 64, (64, 64), 4, 2.0):
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
    if set(dict(lift.named_buffers(recurse=False))) != {
        "anchor_grid_xy",
        "anchor_in_frustum",
        "camera_origin_xyz_m",
        "camera_basis_forward_right_up",
        "bev_ground_xyz_m",
        "ground_z_m",
        "horizontal_fov_degrees",
        "vertical_fov_degrees",
        "camera_near_m",
    }:
        raise RuntimeError("clean V4 lift-buffer inventory changed")
    projection = lift.token_projection
    if not isinstance(projection, nn.Conv2d) or (
        projection.in_channels,
        projection.out_channels,
        projection.kernel_size,
        projection.bias is not None,
    ) != (192, 64, (1, 1), True):
        raise RuntimeError("clean V4 token projection changed")
    if not isinstance(lift.refinement_blocks, nn.ModuleList) or len(
        lift.refinement_blocks
    ) != 2:
        raise RuntimeError("clean V4 refinement blocks changed")


def _fixed_support_offsets_v9() -> torch.Tensor:
    axis = torch.arange(-2, 3, dtype=torch.float32)
    offset_y, offset_x = torch.meshgrid(axis, axis, indexing="ij")
    result = torch.stack((offset_x, offset_y), dim=-1).reshape(
        DENSE_LOCAL_SUPPORT_COUNT_V9, 2
    )
    if not torch.equal(
        result[DENSE_LOCAL_SUPPORT_CENTER_INDEX_V9], torch.zeros(2)
    ):
        raise RuntimeError("V9 support centre/order changed")
    return result


class ContentAdaptiveDenseLocalTokenLiftV9(nn.Module):
    """Exact V4 geometry/state with one fixed-window local attention lift."""

    def __init__(self, v4_lift: GeometryAnchoredDeformableBevLiftV1) -> None:
        super().__init__()
        _validate_clean_v4_lift_v9(v4_lift)
        self.config = v4_lift.config

        # Retain every V4 tensor except the two removed four-sample parameters.
        null_evidence = v4_lift.null_evidence
        self.register_parameter(
            "null_evidence",
            nn.Parameter(
                null_evidence.detach().clone(),
                requires_grad=null_evidence.requires_grad,
            ),
        )
        for name, buffer in v4_lift.named_buffers(recurse=False):
            self.register_buffer(
                name,
                buffer.detach().clone(),
                persistent=name not in v4_lift._non_persistent_buffers_set,
            )
        self.token_projection = copy.deepcopy(v4_lift.token_projection)
        self.refinement_blocks = copy.deepcopy(v4_lift.refinement_blocks)
        self.register_buffer(
            "support_offsets_token_cells",
            _fixed_support_offsets_v9(),
            persistent=True,
        )

        # The constructors draw from the global CPU generator, so the complete
        # block is isolated and the caller's state is restored byte-for-byte.
        caller_rng = torch.random.get_rng_state().clone()
        try:
            self.query_projection = nn.Linear(64, 64, bias=True)
            self.key_projection = nn.Linear(64, 64, bias=False)
            self.value_projection = nn.Linear(64, 64, bias=True)
            self.output_projection = nn.Linear(64, 64, bias=True)
            generator = torch.Generator(device="cpu")
            generator.manual_seed(DENSE_LOCAL_ATTENTION_INITIALIZATION_SEED_V9)
            for projection in (
                self.query_projection,
                self.key_projection,
                self.value_projection,
                self.output_projection,
            ):
                nn.init.xavier_uniform_(
                    projection.weight, gain=1.0, generator=generator
                )
                if projection.bias is not None:
                    nn.init.zeros_(projection.bias)
        finally:
            torch.random.set_rng_state(caller_rng)

        attention_parameters = tuple(
            parameter
            for projection in (
                self.query_projection,
                self.key_projection,
                self.value_projection,
                self.output_projection,
            )
            for parameter in projection.parameters()
        )
        if len(attention_parameters) != (
            DENSE_LOCAL_ATTENTION_PARAMETER_TENSOR_COUNT_V9
        ):
            raise RuntimeError("V9 attention parameter inventory changed")
        if sum(parameter.numel() for parameter in attention_parameters) != (
            DENSE_LOCAL_ATTENTION_ADDED_PARAMETER_COUNT_V9
        ):
            raise RuntimeError("V9 attention parameter count changed")

    def _validate_tokens(self, patch_tokens: torch.Tensor) -> None:
        if not isinstance(patch_tokens, torch.Tensor):
            raise TypeError("patch_tokens must be a tensor")
        if patch_tokens.ndim != 3 or tuple(patch_tokens.shape[1:]) != (256, 192):
            raise ValueError("patch_tokens must have shape (B,256,192)")
        if patch_tokens.shape[0] < 1:
            raise ValueError("patch_tokens must contain at least one row")
        if patch_tokens.dtype != torch.float32:
            raise TypeError("patch_tokens must use exact float32")
        if patch_tokens.device != self.null_evidence.device:
            raise TypeError("patch_tokens and lift must share a device")
        if not bool(torch.isfinite(patch_tokens).all()):
            raise FloatingPointError("patch_tokens are nonfinite")

    def _attend_valid_cells(
        self,
        centre_samples: torch.Tensor,
        support_samples: torch.Tensor,
        support_valid: torch.Tensor,
        cell_valid: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return flat attention outputs and reportable per-head weights."""

        flat_centres = centre_samples.reshape(-1, self.config.bev_dim)
        flat_supports = support_samples.reshape(
            -1, DENSE_LOCAL_SUPPORT_COUNT_V9, self.config.bev_dim
        )
        flat_support_valid = support_valid.reshape(
            -1, DENSE_LOCAL_SUPPORT_COUNT_V9
        )
        flat_cell_valid = cell_valid.reshape(-1)
        valid_indices = torch.nonzero(flat_cell_valid, as_tuple=False).flatten()

        outputs = flat_centres.new_zeros(flat_centres.shape)
        report_weights = flat_centres.new_zeros(
            (
                flat_centres.shape[0],
                DENSE_LOCAL_ATTENTION_HEADS_V9,
                DENSE_LOCAL_SUPPORT_COUNT_V9,
            )
        )
        if valid_indices.numel() == 0:
            return outputs, report_weights

        centres = flat_centres.index_select(0, valid_indices)
        supports = flat_supports.index_select(0, valid_indices)
        valid = flat_support_valid.index_select(0, valid_indices)
        if not bool(valid[:, DENSE_LOCAL_SUPPORT_CENTER_INDEX_V9].all()):
            raise RuntimeError("visible V9 cell has an invalid centre support")

        query = self.query_projection(centres).reshape(
            -1,
            DENSE_LOCAL_ATTENTION_HEADS_V9,
            DENSE_LOCAL_ATTENTION_HEAD_WIDTH_V9,
        )
        key = self.key_projection(supports).reshape(
            -1,
            DENSE_LOCAL_SUPPORT_COUNT_V9,
            DENSE_LOCAL_ATTENTION_HEADS_V9,
            DENSE_LOCAL_ATTENTION_HEAD_WIDTH_V9,
        )
        value = self.value_projection(supports).reshape_as(key)
        logits = torch.einsum("nhd,nshd->nhs", query, key)
        logits = logits * (1.0 / math.sqrt(DENSE_LOCAL_ATTENTION_HEAD_WIDTH_V9))
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
    ) -> ContentAdaptiveDenseLocalTokenLiftSamplingV9:
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

        anchor_grid = self.anchor_grid_xy.to(dtype=projected.dtype)[None].expand(
            batch, -1, -1, -1
        )
        offsets = self.support_offsets_token_cells.to(dtype=projected.dtype)
        normalized_offsets = offsets * (2.0 / self.config.token_side)
        proposed_grid = anchor_grid[..., None, :] + normalized_offsets
        anchor_visible = self.anchor_in_frustum[None].expand(batch, -1, -1)
        within_grid = (
            (proposed_grid[..., 0] >= -1.0)
            & (proposed_grid[..., 0] <= 1.0)
            & (proposed_grid[..., 1] >= -1.0)
            & (proposed_grid[..., 1] <= 1.0)
        )
        support_valid = anchor_visible[..., None] & within_grid
        safe_support_grid = torch.where(
            support_valid[..., None],
            proposed_grid,
            torch.full_like(proposed_grid, 2.0),
        )
        packed_grid = safe_support_grid.reshape(
            batch, height, width * DENSE_LOCAL_SUPPORT_COUNT_V9, 2
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
            DENSE_LOCAL_SUPPORT_COUNT_V9,
        )
        support_samples = sampled.permute(0, 2, 3, 4, 1).contiguous()
        centre_samples = support_samples[..., DENSE_LOCAL_SUPPORT_CENTER_INDEX_V9, :]
        cell_valid = support_valid.any(dim=-1)
        attention_output, attention_weights = self._attend_valid_cells(
            centre_samples,
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
            DENSE_LOCAL_ATTENTION_HEADS_V9,
            DENSE_LOCAL_SUPPORT_COUNT_V9,
        )
        lifted = (centre_samples + attention_output).permute(0, 3, 1, 2)
        null = self.null_evidence[None, :, None, None].to(dtype=lifted.dtype)
        lifted = torch.where(cell_valid[:, None], lifted, null)
        for block in self.refinement_blocks:
            lifted = block(lifted)
            lifted = torch.where(cell_valid[:, None], lifted, null)

        if not bool(torch.isfinite(lifted).all()):
            raise FloatingPointError("V9 lift latent is nonfinite")
        if not bool(torch.isfinite(attention_weights).all()):
            raise FloatingPointError("V9 attention weights are nonfinite")
        return ContentAdaptiveDenseLocalTokenLiftSamplingV9(
            latent=lifted,
            anchor_in_frustum=anchor_visible,
            support_valid_mask=support_valid,
            cell_valid_mask=cell_valid,
            support_grid_xy=safe_support_grid,
            support_offsets_token_cells=self.support_offsets_token_cells,
            attention_weights=attention_weights,
        )

    def forward(self, patch_tokens: torch.Tensor) -> torch.Tensor:
        return self.forward_with_sampling(patch_tokens).latent


# Descriptive compatibility alias retained for callers that spell out BEV.
ContentAdaptiveDenseLocalTokenBevLiftV9 = ContentAdaptiveDenseLocalTokenLiftV9


class GeometryAnchoredSweptProgressSurvivalJointJepaV9(
    GeometryAnchoredSweptProgressSurvivalJointJepaV4
):
    """Clean V4 with only its sparse lift aggregation replaced."""

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
        self.bev_lift = ContentAdaptiveDenseLocalTokenLiftV9(self.bev_lift)
        self.target_bev_lift = copy.deepcopy(self.bev_lift)
        self._freeze_target()

    def encode_online_with_sampling(
        self, rgb: torch.Tensor
    ) -> ContentAdaptiveDenseLocalTokenLiftSamplingV9:
        self._validate_rgb(rgb, name="online_rgb")
        patch_tokens = self.encoder.forward_tokens(rgb)[:, 1:]
        return self.bev_lift.forward_with_sampling(patch_tokens)

    @torch.no_grad()
    def encode_target_with_sampling(
        self, rgb: torch.Tensor
    ) -> ContentAdaptiveDenseLocalTokenLiftSamplingV9:
        self._validate_rgb(rgb, name="target_rgb")
        patch_tokens = self.target_encoder.forward_tokens(rgb)[:, 1:]
        state = self.target_bev_lift.forward_with_sampling(patch_tokens)
        return ContentAdaptiveDenseLocalTokenLiftSamplingV9(
            latent=state.latent.detach(),
            anchor_in_frustum=state.anchor_in_frustum,
            support_valid_mask=state.support_valid_mask,
            cell_valid_mask=state.cell_valid_mask,
            support_grid_xy=state.support_grid_xy,
            support_offsets_token_cells=state.support_offsets_token_cells,
            attention_weights=state.attention_weights.detach(),
        )


GeometryAnchoredSweptProgressSurvivalJointJepaV9Config = (
    GeometryAnchoredDeformableBevLiftJointJepaV1Config
)
# The frozen runner resolves this historical name from its selected model module.
GeometryAnchoredDeformableBevLiftJointJepaV1 = (
    GeometryAnchoredSweptProgressSurvivalJointJepaV9
)
final_class_macro_nll_per_row_v1 = final_class_macro_nll_per_row
latent_energy_per_row_v1 = latent_energy_per_row


__all__ = [
    "ACTION_VOCABULARY_V1",
    "ContentAdaptiveDenseLocalTokenBevLiftV9",
    "ContentAdaptiveDenseLocalTokenLiftV9",
    "ContentAdaptiveDenseLocalTokenLiftSamplingV9",
    "DENSE_LOCAL_ATTENTION_ADDED_PARAMETER_COUNT_V9",
    "DENSE_LOCAL_ATTENTION_HEADS_V9",
    "DENSE_LOCAL_ATTENTION_HEAD_WIDTH_V9",
    "DENSE_LOCAL_ATTENTION_INITIALIZATION_SEED_V9",
    "DENSE_LOCAL_ATTENTION_PARAMETER_TENSOR_COUNT_V9",
    "DENSE_LOCAL_SUPPORT_CENTER_INDEX_V9",
    "DENSE_LOCAL_SUPPORT_COUNT_V9",
    "DENSE_LOCAL_SUPPORT_SIDE_V9",
    "FREE_CLASS_V1",
    "GeometryAnchoredBevSamplingV1",
    "GeometryAnchoredDeformableBevLiftJointJepaV1",
    "GeometryAnchoredDeformableBevLiftJointJepaV1Config",
    "GeometryAnchoredSweptProgressSurvivalJointJepaV9",
    "GeometryAnchoredSweptProgressSurvivalJointJepaV9Config",
    "OCCUPIED_CLASS_V1",
    "SWEEP_PROGRESS_BIN_COUNT_V1",
    "SweptProgressSurvivalHeadV1",
    "SweptProgressSurvivalPredictionV1",
    "UNKNOWN_CLASS_V1",
    "final_class_macro_nll_per_row",
    "final_class_macro_nll_per_row_v1",
    "latent_energy_per_row",
    "latent_energy_per_row_v1",
]
