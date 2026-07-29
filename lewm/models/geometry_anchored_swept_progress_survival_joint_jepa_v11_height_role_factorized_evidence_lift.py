"""V10 geometry with height-routed FREE and OCCUPIED evidence branches.

V11 preserves the clean N320-seeded encoder, V10 projective support geometry,
shared 64-channel JEPA state, action predictor, survival head, and EMA update.
The 25 sampled supports are split before aggregation into five floor supports
and twenty elevated supports.  Their learned 32-channel branches remain
separate halves of the state consumed by the predictor and semantic decoder.
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
from lewm.models.geometry_anchored_swept_progress_survival_joint_jepa_v10_projective_cell_volume_token_lift import (
    CELL_VOLUME_HEIGHTS_M_V10,
    CELL_VOLUME_HORIZONTAL_OFFSETS_M_V10,
    CELL_VOLUME_SUPPORT_COUNT_V10,
    CELL_VOLUME_VALID_CELL_COUNT_V10,
    CELL_VOLUME_VALID_MASK_SHA256_V10,
    CELL_VOLUME_WITHIN_TWO_METERS_CELL_COUNT_V10,
    CELL_VOLUME_WITHIN_TWO_METERS_VALID_CELL_COUNT_V10,
    ProjectiveCellVolumeTokenLiftV10,
)


HEIGHT_ROLE_INITIALIZATION_SEED_V11 = 20_260_730
HEIGHT_ROLE_BRANCH_WIDTH_V11 = 32
HEIGHT_ROLE_ATTENTION_HEADS_V11 = 2
HEIGHT_ROLE_ATTENTION_HEAD_WIDTH_V11 = 16
HEIGHT_ROLE_ATTENTION_PARAMETER_TENSOR_COUNT_V11 = 14
HEIGHT_ROLE_ATTENTION_PARAMETER_COUNT_V11 = 14_528
HEIGHT_ROLE_SEMANTIC_PARAMETER_TENSOR_COUNT_V11 = 12
HEIGHT_ROLE_SEMANTIC_PARAMETER_COUNT_V11 = 18_628

FLOOR_SUPPORT_INDICES_V11 = (0, 5, 10, 15, 20)
ELEVATED_SUPPORT_INDICES_V11 = tuple(
    index
    for index in range(CELL_VOLUME_SUPPORT_COUNT_V10)
    if index not in FLOOR_SUPPORT_INDICES_V11
)
FLOOR_SUPPORT_COUNT_V11 = 5
ELEVATED_SUPPORT_COUNT_V11 = 20
FLOOR_VALID_CELL_COUNT_V11 = 2_024
ELEVATED_VALID_CELL_COUNT_V11 = 2_062
FLOOR_VALID_MASK_SHA256_V11 = (
    "8b6b4202d04cf08de9813a4fc12deff9ea35de8d8c7adc8eb40a117593694bbc"
)
ELEVATED_VALID_MASK_SHA256_V11 = CELL_VOLUME_VALID_MASK_SHA256_V10
FLOOR_WITHIN_TWO_METERS_VALID_CELL_COUNT_V11 = 184
ELEVATED_WITHIN_TWO_METERS_VALID_CELL_COUNT_V11 = 222
ELEVATED_ONLY_VALID_CELL_COUNT_V11 = 38

HEIGHT_ROLE_ATTENTION_PARAMETER_SUFFIXES_V11 = (
    "floor_query_projection.weight",
    "floor_query_projection.bias",
    "floor_key_projection.weight",
    "floor_value_projection.weight",
    "floor_value_projection.bias",
    "floor_output_projection.weight",
    "floor_output_projection.bias",
    "elevated_query_projection.weight",
    "elevated_query_projection.bias",
    "elevated_key_projection.weight",
    "elevated_value_projection.weight",
    "elevated_value_projection.bias",
    "elevated_output_projection.weight",
    "elevated_output_projection.bias",
)
HEIGHT_ROLE_SEMANTIC_PARAMETER_SUFFIXES_V11 = (
    "free_axis.base.weight",
    "free_axis.base.bias",
    "free_axis.local.weight",
    "free_axis.local.bias",
    "free_axis.residual_output.weight",
    "free_axis.residual_output.bias",
    "occupied_axis.base.weight",
    "occupied_axis.base.bias",
    "occupied_axis.local.weight",
    "occupied_axis.local.bias",
    "occupied_axis.residual_output.weight",
    "occupied_axis.residual_output.bias",
)


class HeightRoleFactorizedEvidenceLiftSamplingV11(NamedTuple):
    """Auditable V11 support roles, branch values, and shared latent."""

    latent: torch.Tensor
    anchor_in_frustum: torch.Tensor
    support_valid_mask: torch.Tensor
    cell_valid_mask: torch.Tensor
    support_grid_xy: torch.Tensor
    support_xyz_m: torch.Tensor
    support_offsets_xy_m: torch.Tensor
    support_heights_m: torch.Tensor
    floor_support_role_mask: torch.Tensor
    elevated_support_role_mask: torch.Tensor
    floor_support_valid_mask: torch.Tensor
    elevated_support_valid_mask: torch.Tensor
    floor_cell_valid_mask: torch.Tensor
    elevated_cell_valid_mask: torch.Tensor
    floor_masked_mean: torch.Tensor
    elevated_masked_mean: torch.Tensor
    floor_attention_weights: torch.Tensor
    elevated_attention_weights: torch.Tensor


def _mask_sha256_v11(mask: torch.Tensor) -> str:
    payload = bytes(mask.to(torch.uint8).reshape(-1).tolist())
    return hashlib.sha256(payload).hexdigest()


class HeightRoleFactorizedEvidenceLiftV11(ProjectiveCellVolumeTokenLiftV10):
    """Two fixed-role attentions over the unchanged V10 sampled supports."""

    def __init__(self, v4_lift: nn.Module) -> None:
        super().__init__(v4_lift)
        del self.query_projection
        del self.key_projection
        del self.value_projection
        del self.output_projection

        floor_role = torch.zeros(CELL_VOLUME_SUPPORT_COUNT_V10, dtype=torch.bool)
        floor_role[list(FLOOR_SUPPORT_INDICES_V11)] = True
        elevated_role = ~floor_role
        if (
            int(floor_role.sum()) != FLOOR_SUPPORT_COUNT_V11
            or int(elevated_role.sum()) != ELEVATED_SUPPORT_COUNT_V11
            or bool((floor_role & elevated_role).any())
            or not bool((floor_role | elevated_role).all())
        ):
            raise RuntimeError("V11 support-role partition changed")
        self.register_buffer("floor_support_role_mask", floor_role, persistent=True)
        self.register_buffer(
            "elevated_support_role_mask", elevated_role, persistent=True
        )

        floor_valid = (self.support_valid_mask & floor_role).any(dim=-1)
        elevated_valid = (self.support_valid_mask & elevated_role).any(dim=-1)
        if (
            int(floor_valid.sum()) != FLOOR_VALID_CELL_COUNT_V11
            or _mask_sha256_v11(floor_valid) != FLOOR_VALID_MASK_SHA256_V11
            or int(elevated_valid.sum()) != ELEVATED_VALID_CELL_COUNT_V11
            or _mask_sha256_v11(elevated_valid)
            != ELEVATED_VALID_MASK_SHA256_V11
            or not torch.equal(elevated_valid, self.cell_valid_mask)
            or int((elevated_valid & ~floor_valid).sum())
            != ELEVATED_ONLY_VALID_CELL_COUNT_V11
        ):
            raise RuntimeError("V11 frozen role-valid masks changed")
        self.register_buffer("floor_cell_valid_mask", floor_valid, persistent=True)
        self.register_buffer(
            "elevated_cell_valid_mask", elevated_valid, persistent=True
        )

        caller_rng = torch.random.get_rng_state().clone()
        try:
            self.floor_query_projection = nn.Linear(64, 32, bias=True)
            self.floor_key_projection = nn.Linear(64, 32, bias=False)
            self.floor_value_projection = nn.Linear(64, 32, bias=True)
            self.floor_output_projection = nn.Linear(32, 32, bias=True)
            self.elevated_query_projection = nn.Linear(64, 32, bias=True)
            self.elevated_key_projection = nn.Linear(64, 32, bias=False)
            self.elevated_value_projection = nn.Linear(64, 32, bias=True)
            self.elevated_output_projection = nn.Linear(32, 32, bias=True)
            generator = torch.Generator(device="cpu")
            generator.manual_seed(HEIGHT_ROLE_INITIALIZATION_SEED_V11)
            for projection in (
                self.floor_query_projection,
                self.floor_key_projection,
                self.floor_value_projection,
                self.floor_output_projection,
                self.elevated_query_projection,
                self.elevated_key_projection,
                self.elevated_value_projection,
                self.elevated_output_projection,
            ):
                nn.init.xavier_uniform_(
                    projection.weight, gain=1.0, generator=generator
                )
                if projection.bias is not None:
                    nn.init.zeros_(projection.bias)
        finally:
            torch.random.set_rng_state(caller_rng)

        attention = tuple(
            parameter
            for suffix, parameter in self.named_parameters()
            if suffix in HEIGHT_ROLE_ATTENTION_PARAMETER_SUFFIXES_V11
        )
        if len(attention) != HEIGHT_ROLE_ATTENTION_PARAMETER_TENSOR_COUNT_V11:
            raise RuntimeError("V11 attention tensor count changed")
        if sum(value.numel() for value in attention) != (
            HEIGHT_ROLE_ATTENTION_PARAMETER_COUNT_V11
        ):
            raise RuntimeError("V11 attention parameter count changed")

    def _attend_role(
        self,
        masked_mean: torch.Tensor,
        support_samples: torch.Tensor,
        role_support_valid: torch.Tensor,
        role_cell_valid: torch.Tensor,
        *,
        query_projection: nn.Linear,
        key_projection: nn.Linear,
        value_projection: nn.Linear,
        output_projection: nn.Linear,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        flat_mean = masked_mean.reshape(-1, self.config.bev_dim)
        flat_supports = support_samples.reshape(
            -1, CELL_VOLUME_SUPPORT_COUNT_V10, self.config.bev_dim
        )
        flat_valid = role_support_valid.reshape(
            -1, CELL_VOLUME_SUPPORT_COUNT_V10
        )
        flat_cell_valid = role_cell_valid.reshape(-1)
        valid_indices = torch.nonzero(flat_cell_valid, as_tuple=False).flatten()
        outputs = flat_mean.new_zeros(
            (flat_mean.shape[0], HEIGHT_ROLE_BRANCH_WIDTH_V11)
        )
        report_weights = flat_mean.new_zeros(
            (
                flat_mean.shape[0],
                HEIGHT_ROLE_ATTENTION_HEADS_V11,
                CELL_VOLUME_SUPPORT_COUNT_V10,
            )
        )
        if valid_indices.numel() == 0:
            return outputs, report_weights

        means = flat_mean.index_select(0, valid_indices)
        supports = flat_supports.index_select(0, valid_indices)
        valid = flat_valid.index_select(0, valid_indices)
        if not bool(valid.any(dim=-1).all()):
            raise RuntimeError("valid V11 role cell has no valid role support")
        query_flat = query_projection(means)
        query = query_flat.reshape(
            -1,
            HEIGHT_ROLE_ATTENTION_HEADS_V11,
            HEIGHT_ROLE_ATTENTION_HEAD_WIDTH_V11,
        )
        key = key_projection(supports).reshape(
            -1,
            CELL_VOLUME_SUPPORT_COUNT_V10,
            HEIGHT_ROLE_ATTENTION_HEADS_V11,
            HEIGHT_ROLE_ATTENTION_HEAD_WIDTH_V11,
        )
        value = value_projection(supports).reshape_as(key)
        logits = torch.einsum("nhd,nshd->nhs", query, key)
        logits = logits * (1.0 / math.sqrt(HEIGHT_ROLE_ATTENTION_HEAD_WIDTH_V11))
        logits = logits.masked_fill(~valid[:, None, :], float("-inf"))
        weights = torch.softmax(logits, dim=-1)
        attended = torch.einsum("nhs,nshd->nhd", weights, value).reshape(
            -1, HEIGHT_ROLE_BRANCH_WIDTH_V11
        )
        branch = query_flat + output_projection(attended)
        outputs = outputs.index_copy(0, valid_indices, branch)
        report_weights = report_weights.index_copy(0, valid_indices, weights)
        return outputs, report_weights

    def _refine_role_half(
        self,
        branch: torch.Tensor,
        role_valid: torch.Tensor,
        *,
        retain_first_half: bool,
    ) -> torch.Tensor:
        zeros = torch.zeros_like(branch)
        channel_last = (
            torch.cat((branch, zeros), dim=-1)
            if retain_first_half
            else torch.cat((zeros, branch), dim=-1)
        )
        lifted = channel_last.permute(0, 3, 1, 2).contiguous()
        null = self.null_evidence[None, :, None, None].to(dtype=lifted.dtype)
        lifted = torch.where(role_valid[:, None], lifted, null)
        for block in self.refinement_blocks:
            lifted = block(lifted)
            lifted = torch.where(role_valid[:, None], lifted, null)
        return lifted[:, :32] if retain_first_half else lifted[:, 32:]

    def forward_with_sampling(
        self, patch_tokens: torch.Tensor
    ) -> HeightRoleFactorizedEvidenceLiftSamplingV11:
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

        floor_support_valid = support_valid & self.floor_support_role_mask
        elevated_support_valid = support_valid & self.elevated_support_role_mask
        floor_cell_valid = floor_support_valid.any(dim=-1)
        elevated_cell_valid = elevated_support_valid.any(dim=-1)
        cell_valid = floor_cell_valid | elevated_cell_valid
        if not torch.equal(cell_valid, support_valid.any(dim=-1)):
            raise RuntimeError("V11 role masks no longer exhaust V10 supports")

        def masked_mean(role_valid: torch.Tensor) -> torch.Tensor:
            count = role_valid.sum(dim=-1, keepdim=True).to(
                dtype=support_samples.dtype
            )
            values = torch.where(
                role_valid[..., None],
                support_samples,
                torch.zeros_like(support_samples),
            )
            return values.sum(dim=-2) / count.clamp_min(1.0)

        floor_mean = masked_mean(floor_support_valid)
        elevated_mean = masked_mean(elevated_support_valid)
        floor_branch, floor_weights = self._attend_role(
            floor_mean,
            support_samples,
            floor_support_valid,
            floor_cell_valid,
            query_projection=self.floor_query_projection,
            key_projection=self.floor_key_projection,
            value_projection=self.floor_value_projection,
            output_projection=self.floor_output_projection,
        )
        elevated_branch, elevated_weights = self._attend_role(
            elevated_mean,
            support_samples,
            elevated_support_valid,
            elevated_cell_valid,
            query_projection=self.elevated_query_projection,
            key_projection=self.elevated_key_projection,
            value_projection=self.elevated_value_projection,
            output_projection=self.elevated_output_projection,
        )
        floor_branch = floor_branch.reshape(batch, height, width, 32)
        elevated_branch = elevated_branch.reshape(batch, height, width, 32)
        floor_weights = floor_weights.reshape(batch, height, width, 2, 25)
        elevated_weights = elevated_weights.reshape(batch, height, width, 2, 25)
        floor_latent = self._refine_role_half(
            floor_branch, floor_cell_valid, retain_first_half=True
        )
        elevated_latent = self._refine_role_half(
            elevated_branch, elevated_cell_valid, retain_first_half=False
        )
        latent = torch.cat((floor_latent, elevated_latent), dim=1)

        for name, value in (
            ("floor masked mean", floor_mean),
            ("elevated masked mean", elevated_mean),
            ("floor attention", floor_weights),
            ("elevated attention", elevated_weights),
            ("shared latent", latent),
        ):
            if not bool(torch.isfinite(value).all()):
                raise FloatingPointError(f"V11 {name} is nonfinite")
        anchor_visible = self.anchor_in_frustum[None].expand(batch, -1, -1)
        return HeightRoleFactorizedEvidenceLiftSamplingV11(
            latent=latent,
            anchor_in_frustum=anchor_visible,
            support_valid_mask=support_valid,
            cell_valid_mask=cell_valid,
            support_grid_xy=support_grid,
            support_xyz_m=self.support_xyz_m,
            support_offsets_xy_m=self.support_offsets_xy_m,
            support_heights_m=self.support_heights_m,
            floor_support_role_mask=self.floor_support_role_mask,
            elevated_support_role_mask=self.elevated_support_role_mask,
            floor_support_valid_mask=floor_support_valid,
            elevated_support_valid_mask=elevated_support_valid,
            floor_cell_valid_mask=floor_cell_valid,
            elevated_cell_valid_mask=elevated_cell_valid,
            floor_masked_mean=floor_mean,
            elevated_masked_mean=elevated_mean,
            floor_attention_weights=floor_weights,
            elevated_attention_weights=elevated_weights,
        )


class ResidualLocalEvidenceAxisV11(nn.Module):
    """Half-width V4-style local decoder producing one evidence logit."""

    def __init__(self, *, generator: torch.Generator) -> None:
        super().__init__()
        self.base = nn.Conv2d(32, 1, kernel_size=1, bias=True)
        self.local = nn.Conv2d(32, 32, kernel_size=3, padding=1, bias=True)
        self.activation = nn.GELU(approximate="none")
        self.residual_output = nn.Conv2d(32, 1, kernel_size=1, bias=True)
        nn.init.xavier_uniform_(self.base.weight, gain=1.0, generator=generator)
        nn.init.zeros_(self.base.bias)
        nn.init.xavier_uniform_(self.local.weight, gain=1.0, generator=generator)
        nn.init.zeros_(self.local.bias)
        nn.init.zeros_(self.residual_output.weight)
        nn.init.zeros_(self.residual_output.bias)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        if value.ndim != 4 or value.shape[1] != 32:
            raise ValueError("V11 evidence input must have shape (B,32,H,W)")
        residual = self.residual_output(self.activation(self.local(value)))
        return (self.base(value) + residual).squeeze(1)


def occupied_priority_log_probabilities_v11(
    free_evidence: torch.Tensor, occupied_evidence: torch.Tensor
) -> torch.Tensor:
    """Map two evidence axes to normalized UNKNOWN/FREE/OCCUPIED log-probs."""

    if free_evidence.shape != occupied_evidence.shape or free_evidence.ndim != 3:
        raise ValueError("V11 evidence axes must have matching shape (B,H,W)")
    if not bool(torch.isfinite(free_evidence).all()) or not bool(
        torch.isfinite(occupied_evidence).all()
    ):
        raise FloatingPointError("V11 evidence axes are nonfinite")
    log_not_occupied = F.logsigmoid(-occupied_evidence)
    log_unknown = log_not_occupied + F.logsigmoid(-free_evidence)
    log_free = log_not_occupied + F.logsigmoid(free_evidence)
    log_occupied = F.logsigmoid(occupied_evidence)
    result = torch.stack((log_unknown, log_free, log_occupied), dim=1)
    if not bool(torch.isfinite(result).all()):
        raise FloatingPointError("V11 semantic log probabilities are nonfinite")
    return result


class HeightRoleOccupiedPrioritySemanticDecoderV11(nn.Module):
    """Disjoint floor/FREE and elevated/OCCUPIED local evidence axes."""

    def __init__(self) -> None:
        super().__init__()
        caller_rng = torch.random.get_rng_state().clone()
        try:
            generator = torch.Generator(device="cpu")
            generator.manual_seed(HEIGHT_ROLE_INITIALIZATION_SEED_V11)
            self.free_axis = ResidualLocalEvidenceAxisV11(generator=generator)
            self.occupied_axis = ResidualLocalEvidenceAxisV11(generator=generator)
        finally:
            torch.random.set_rng_state(caller_rng)
        parameters = tuple(self.parameters())
        if len(parameters) != HEIGHT_ROLE_SEMANTIC_PARAMETER_TENSOR_COUNT_V11:
            raise RuntimeError("V11 semantic tensor count changed")
        if sum(value.numel() for value in parameters) != (
            HEIGHT_ROLE_SEMANTIC_PARAMETER_COUNT_V11
        ):
            raise RuntimeError("V11 semantic parameter count changed")

    def evidence_logits(self, latent: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if latent.ndim != 4 or tuple(latent.shape[1:]) != (64, 64, 64):
            raise ValueError("V11 latent must have shape (B,64,64,64)")
        return self.free_axis(latent[:, :32]), self.occupied_axis(latent[:, 32:])

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        free, occupied = self.evidence_logits(latent)
        return occupied_priority_log_probabilities_v11(free, occupied)


class GeometryAnchoredSweptProgressSurvivalJointJepaV11(
    GeometryAnchoredSweptProgressSurvivalJointJepaV4
):
    """Joint JEPA whose shared latent preserves floor/elevated evidence roles."""

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
        self.bev_lift = HeightRoleFactorizedEvidenceLiftV11(self.bev_lift)
        self.target_bev_lift = copy.deepcopy(self.bev_lift)
        self.semantic_head = HeightRoleOccupiedPrioritySemanticDecoderV11()
        self._freeze_target()

    def encode_online_with_sampling(
        self, rgb: torch.Tensor
    ) -> HeightRoleFactorizedEvidenceLiftSamplingV11:
        self._validate_rgb(rgb, name="online_rgb")
        patch_tokens = self.encoder.forward_tokens(rgb)[:, 1:]
        return self.bev_lift.forward_with_sampling(patch_tokens)

    @torch.no_grad()
    def encode_target_with_sampling(
        self, rgb: torch.Tensor
    ) -> HeightRoleFactorizedEvidenceLiftSamplingV11:
        self._validate_rgb(rgb, name="target_rgb")
        patch_tokens = self.target_encoder.forward_tokens(rgb)[:, 1:]
        state = self.target_bev_lift.forward_with_sampling(patch_tokens)
        values = []
        for field, value in zip(state._fields, state, strict=True):
            if field in (
                "latent",
                "floor_masked_mean",
                "elevated_masked_mean",
                "floor_attention_weights",
                "elevated_attention_weights",
            ):
                value = value.detach()
            values.append(value)
        return HeightRoleFactorizedEvidenceLiftSamplingV11(*values)

    def semantic_logits_from_latent(self, latent: torch.Tensor) -> torch.Tensor:
        expected = (self.config.bev_dim, *self.config.bev_size)
        if latent.ndim != 4 or tuple(latent.shape[1:]) != expected:
            raise ValueError(f"latent must have shape (B,{expected})")
        free, occupied = self.semantic_head.evidence_logits(latent)
        floor_valid = self.bev_lift.floor_cell_valid_mask[None].expand(
            latent.shape[0], -1, -1
        )
        elevated_valid = self.bev_lift.elevated_cell_valid_mask[None].expand(
            latent.shape[0], -1, -1
        )
        free = torch.where(floor_valid, free, torch.full_like(free, -20.0))
        occupied = torch.where(
            elevated_valid, occupied, torch.full_like(occupied, -20.0)
        )
        logits = occupied_priority_log_probabilities_v11(free, occupied)
        valid = self.bev_lift.cell_valid_mask[None, None].expand(
            latent.shape[0], 1, -1, -1
        )
        invalid_logits = logits.new_tensor((0.0, -20.0, -20.0))[
            None, :, None, None
        ]
        return torch.where(valid, logits, invalid_logits)


HeightRoleFactorizedEvidenceTokenBevLiftV11 = HeightRoleFactorizedEvidenceLiftV11
GeometryAnchoredSweptProgressSurvivalJointJepaV11Config = (
    GeometryAnchoredDeformableBevLiftJointJepaV1Config
)
GeometryAnchoredDeformableBevLiftJointJepaV1 = (
    GeometryAnchoredSweptProgressSurvivalJointJepaV11
)
final_class_macro_nll_per_row_v1 = final_class_macro_nll_per_row
latent_energy_per_row_v1 = latent_energy_per_row


__all__ = [
    "ACTION_VOCABULARY_V1",
    "CELL_VOLUME_HEIGHTS_M_V10",
    "CELL_VOLUME_HORIZONTAL_OFFSETS_M_V10",
    "CELL_VOLUME_SUPPORT_COUNT_V10",
    "CELL_VOLUME_VALID_CELL_COUNT_V10",
    "CELL_VOLUME_VALID_MASK_SHA256_V10",
    "CELL_VOLUME_WITHIN_TWO_METERS_CELL_COUNT_V10",
    "CELL_VOLUME_WITHIN_TWO_METERS_VALID_CELL_COUNT_V10",
    "ELEVATED_ONLY_VALID_CELL_COUNT_V11",
    "ELEVATED_SUPPORT_COUNT_V11",
    "ELEVATED_SUPPORT_INDICES_V11",
    "ELEVATED_VALID_CELL_COUNT_V11",
    "ELEVATED_VALID_MASK_SHA256_V11",
    "ELEVATED_WITHIN_TWO_METERS_VALID_CELL_COUNT_V11",
    "FLOOR_SUPPORT_COUNT_V11",
    "FLOOR_SUPPORT_INDICES_V11",
    "FLOOR_VALID_CELL_COUNT_V11",
    "FLOOR_VALID_MASK_SHA256_V11",
    "FLOOR_WITHIN_TWO_METERS_VALID_CELL_COUNT_V11",
    "FREE_CLASS_V1",
    "GeometryAnchoredBevSamplingV1",
    "GeometryAnchoredDeformableBevLiftJointJepaV1",
    "GeometryAnchoredDeformableBevLiftJointJepaV1Config",
    "GeometryAnchoredSweptProgressSurvivalJointJepaV11",
    "GeometryAnchoredSweptProgressSurvivalJointJepaV11Config",
    "HEIGHT_ROLE_ATTENTION_HEADS_V11",
    "HEIGHT_ROLE_ATTENTION_HEAD_WIDTH_V11",
    "HEIGHT_ROLE_ATTENTION_PARAMETER_COUNT_V11",
    "HEIGHT_ROLE_ATTENTION_PARAMETER_SUFFIXES_V11",
    "HEIGHT_ROLE_ATTENTION_PARAMETER_TENSOR_COUNT_V11",
    "HEIGHT_ROLE_BRANCH_WIDTH_V11",
    "HEIGHT_ROLE_INITIALIZATION_SEED_V11",
    "HEIGHT_ROLE_SEMANTIC_PARAMETER_COUNT_V11",
    "HEIGHT_ROLE_SEMANTIC_PARAMETER_SUFFIXES_V11",
    "HEIGHT_ROLE_SEMANTIC_PARAMETER_TENSOR_COUNT_V11",
    "HeightRoleFactorizedEvidenceLiftSamplingV11",
    "HeightRoleFactorizedEvidenceLiftV11",
    "HeightRoleFactorizedEvidenceTokenBevLiftV11",
    "HeightRoleOccupiedPrioritySemanticDecoderV11",
    "OCCUPIED_CLASS_V1",
    "ResidualLocalEvidenceAxisV11",
    "SWEEP_PROGRESS_BIN_COUNT_V1",
    "SweptProgressSurvivalHeadV1",
    "SweptProgressSurvivalPredictionV1",
    "UNKNOWN_CLASS_V1",
    "final_class_macro_nll_per_row",
    "final_class_macro_nll_per_row_v1",
    "latent_energy_per_row",
    "latent_energy_per_row_v1",
    "occupied_priority_log_probabilities_v11",
]
