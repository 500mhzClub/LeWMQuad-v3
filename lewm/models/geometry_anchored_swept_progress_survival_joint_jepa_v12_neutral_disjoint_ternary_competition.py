"""V11 height-role evidence with neutral disjoint ternary competition.

V12 preserves the complete jointly trained V11 representation and changes
only the final semantic algebra.  UNKNOWN is a fixed zero-evidence reference;
the inherited floor and elevated axes compete directly as FREE and OCCUPIED.
"""
from __future__ import annotations

from typing import Mapping

import torch
import torch.nn as nn
import torch.nn.functional as F

from lewm.models.geometry_anchored_swept_progress_survival_joint_jepa_v11_height_role_factorized_evidence_lift import (
    ACTION_VOCABULARY_V1,
    CELL_VOLUME_HEIGHTS_M_V10,
    CELL_VOLUME_HORIZONTAL_OFFSETS_M_V10,
    CELL_VOLUME_SUPPORT_COUNT_V10,
    CELL_VOLUME_VALID_CELL_COUNT_V10,
    CELL_VOLUME_VALID_MASK_SHA256_V10,
    CELL_VOLUME_WITHIN_TWO_METERS_CELL_COUNT_V10,
    CELL_VOLUME_WITHIN_TWO_METERS_VALID_CELL_COUNT_V10,
    ELEVATED_ONLY_VALID_CELL_COUNT_V11,
    ELEVATED_SUPPORT_COUNT_V11,
    ELEVATED_SUPPORT_INDICES_V11,
    ELEVATED_VALID_CELL_COUNT_V11,
    ELEVATED_VALID_MASK_SHA256_V11,
    ELEVATED_WITHIN_TWO_METERS_VALID_CELL_COUNT_V11,
    FLOOR_SUPPORT_COUNT_V11,
    FLOOR_SUPPORT_INDICES_V11,
    FLOOR_VALID_CELL_COUNT_V11,
    FLOOR_VALID_MASK_SHA256_V11,
    FLOOR_WITHIN_TWO_METERS_VALID_CELL_COUNT_V11,
    FREE_CLASS_V1,
    HEIGHT_ROLE_ATTENTION_HEADS_V11,
    HEIGHT_ROLE_ATTENTION_HEAD_WIDTH_V11,
    HEIGHT_ROLE_ATTENTION_PARAMETER_COUNT_V11,
    HEIGHT_ROLE_ATTENTION_PARAMETER_SUFFIXES_V11,
    HEIGHT_ROLE_ATTENTION_PARAMETER_TENSOR_COUNT_V11,
    HEIGHT_ROLE_BRANCH_WIDTH_V11,
    HEIGHT_ROLE_INITIALIZATION_SEED_V11,
    HEIGHT_ROLE_SEMANTIC_PARAMETER_COUNT_V11,
    HEIGHT_ROLE_SEMANTIC_PARAMETER_SUFFIXES_V11,
    HEIGHT_ROLE_SEMANTIC_PARAMETER_TENSOR_COUNT_V11,
    OCCUPIED_CLASS_V1,
    SWEEP_PROGRESS_BIN_COUNT_V1,
    UNKNOWN_CLASS_V1,
    GeometryAnchoredBevSamplingV1,
    GeometryAnchoredDeformableBevLiftJointJepaV1Config,
    GeometryAnchoredSweptProgressSurvivalJointJepaV11,
    HeightRoleFactorizedEvidenceLiftSamplingV11,
    HeightRoleFactorizedEvidenceLiftV11,
    HeightRoleFactorizedEvidenceTokenBevLiftV11,
    HeightRoleOccupiedPrioritySemanticDecoderV11,
    ResidualLocalEvidenceAxisV11,
    SweptProgressSurvivalHeadV1,
    SweptProgressSurvivalPredictionV1,
    final_class_macro_nll_per_row,
    latent_energy_per_row,
)


def neutral_disjoint_ternary_log_probabilities_v12(
    free_evidence: torch.Tensor, occupied_evidence: torch.Tensor
) -> torch.Tensor:
    """Return normalized UNKNOWN/FREE/OCCUPIED log probabilities."""

    if free_evidence.shape != occupied_evidence.shape or free_evidence.ndim != 3:
        raise ValueError("V12 evidence axes must have matching shape (B,H,W)")
    if not bool(torch.isfinite(free_evidence).all()) or not bool(
        torch.isfinite(occupied_evidence).all()
    ):
        raise FloatingPointError("V12 evidence axes are nonfinite")
    neutral = torch.zeros_like(free_evidence)
    result = F.log_softmax(
        torch.stack((neutral, free_evidence, occupied_evidence), dim=1),
        dim=1,
    )
    if not bool(torch.isfinite(result).all()):
        raise FloatingPointError("V12 semantic log probabilities are nonfinite")
    return result


class HeightRoleNeutralDisjointTernarySemanticDecoderV12(
    HeightRoleOccupiedPrioritySemanticDecoderV11
):
    """Zero-parameter wrapper reusing the exact two V11 evidence axes."""

    def __init__(
        self, v11_decoder: HeightRoleOccupiedPrioritySemanticDecoderV11
    ) -> None:
        if not isinstance(
            v11_decoder, HeightRoleOccupiedPrioritySemanticDecoderV11
        ):
            raise TypeError("V12 semantic wrapper requires the V11 decoder")
        caller_rng = torch.random.get_rng_state().clone()
        before = tuple(v11_decoder.named_parameters())
        try:
            nn.Module.__init__(self)
            self.free_axis = v11_decoder.free_axis
            self.occupied_axis = v11_decoder.occupied_axis
        finally:
            torch.random.set_rng_state(caller_rng)
        after = tuple(self.named_parameters())
        if tuple(name for name, _ in after) != tuple(name for name, _ in before):
            raise RuntimeError("V12 semantic parameter names changed")
        if any(new is not old for (_, new), (_, old) in zip(after, before, strict=True)):
            raise RuntimeError("V12 semantic wrapper replaced a V11 axis tensor")

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        free, occupied = self.evidence_logits(latent)
        return neutral_disjoint_ternary_log_probabilities_v12(free, occupied)


class GeometryAnchoredSweptProgressSurvivalJointJepaV12(
    GeometryAnchoredSweptProgressSurvivalJointJepaV11
):
    """V11 joint JEPA with only neutral semantic competition substituted."""

    def __init__(
        self,
        n320_encoder_state_dict: Mapping[str, torch.Tensor],
        sweep_masks: torch.Tensor,
        config: GeometryAnchoredDeformableBevLiftJointJepaV1Config | None = None,
    ) -> None:
        super().__init__(n320_encoder_state_dict, sweep_masks, config)
        self.semantic_head = HeightRoleNeutralDisjointTernarySemanticDecoderV12(
            self.semantic_head
        )

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
        logits = neutral_disjoint_ternary_log_probabilities_v12(free, occupied)
        valid = self.bev_lift.cell_valid_mask[None, None].expand(
            latent.shape[0], 1, -1, -1
        )
        invalid_logits = logits.new_tensor((0.0, -20.0, -20.0))[
            None, :, None, None
        ]
        return torch.where(valid, logits, invalid_logits)


HeightRoleFactorizedEvidenceTokenBevLiftV12 = HeightRoleFactorizedEvidenceLiftV11
GeometryAnchoredSweptProgressSurvivalJointJepaV12Config = (
    GeometryAnchoredDeformableBevLiftJointJepaV1Config
)
GeometryAnchoredDeformableBevLiftJointJepaV1 = (
    GeometryAnchoredSweptProgressSurvivalJointJepaV12
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
    "GeometryAnchoredSweptProgressSurvivalJointJepaV12",
    "GeometryAnchoredSweptProgressSurvivalJointJepaV12Config",
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
    "HeightRoleFactorizedEvidenceTokenBevLiftV12",
    "HeightRoleNeutralDisjointTernarySemanticDecoderV12",
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
    "neutral_disjoint_ternary_log_probabilities_v12",
]
