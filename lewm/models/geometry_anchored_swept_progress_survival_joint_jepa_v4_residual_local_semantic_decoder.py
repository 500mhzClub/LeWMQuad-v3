"""Swept-progress joint JEPA with a residual local semantic decoder.

V4 preserves the complete V1 swept-progress model and wraps its existing
semantic projection with one zero-gated nonlinear local residual branch.  The
inherited visibility masking remains implemented by the parent model.
"""
from __future__ import annotations

from typing import Mapping

import torch
import torch.nn as nn

from lewm.models.geometry_anchored_swept_progress_survival_joint_jepa_v1 import (
    ACTION_VOCABULARY_V1,
    FREE_CLASS_V1,
    OCCUPIED_CLASS_V1,
    SWEEP_PROGRESS_BIN_COUNT_V1,
    UNKNOWN_CLASS_V1,
    GeometryAnchoredBevSamplingV1,
    GeometryAnchoredDeformableBevLiftJointJepaV1Config,
    GeometryAnchoredSweptProgressSurvivalJointJepaV1,
    SweptProgressSurvivalHeadV1,
    SweptProgressSurvivalPredictionV1,
    final_class_macro_nll_per_row,
    latent_energy_per_row,
)


RESIDUAL_LOCAL_SEMANTIC_DECODER_ADDED_PARAMETER_COUNT_V4 = 37_123
RESIDUAL_BRANCH_INITIALIZATION_SEED_OFFSET_V4 = 1


class ResidualLocalSemanticDecoderV4(nn.Module):
    """The inherited linear decoder plus one zero-gated local residual."""

    def __init__(self, base: nn.Conv2d, *, initialization_seed: int) -> None:
        super().__init__()
        if not isinstance(base, nn.Conv2d) or (
            base.in_channels,
            base.out_channels,
            base.kernel_size,
            base.bias is not None,
        ) != (64, 3, (1, 1), True):
            raise TypeError("base must be the inherited biased 64-to-3 1x1 Conv2d")
        self.base = base

        caller_rng = torch.random.get_rng_state().clone()
        try:
            torch.random.default_generator.manual_seed(initialization_seed)
            self.local = nn.Conv2d(
                64, 64, kernel_size=3, padding=1, bias=True
            )
            self.activation = nn.GELU(approximate="none")
            self.residual_output = nn.Conv2d(
                64, 3, kernel_size=1, bias=True
            )
            nn.init.zeros_(self.residual_output.weight)
            nn.init.zeros_(self.residual_output.bias)
        finally:
            torch.random.set_rng_state(caller_rng)

        added = sum(
            parameter.numel()
            for module in (self.local, self.residual_output)
            for parameter in module.parameters()
        )
        if added != RESIDUAL_LOCAL_SEMANTIC_DECODER_ADDED_PARAMETER_COUNT_V4:
            raise RuntimeError("residual semantic-decoder parameter count changed")

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        residual = self.residual_output(self.activation(self.local(latent)))
        return self.base(latent) + residual


class GeometryAnchoredSweptProgressSurvivalJointJepaV4(
    GeometryAnchoredSweptProgressSurvivalJointJepaV1
):
    """V1 swept-progress joint JEPA with the preregistered V4 decoder."""

    def __init__(
        self,
        n320_encoder_state_dict: Mapping[str, torch.Tensor],
        sweep_masks: torch.Tensor,
        config: GeometryAnchoredDeformableBevLiftJointJepaV1Config | None = None,
    ) -> None:
        super().__init__(n320_encoder_state_dict, sweep_masks, config)
        inherited_base = self.semantic_head
        self.semantic_head = ResidualLocalSemanticDecoderV4(
            inherited_base,
            initialization_seed=(
                self.config.initialization_seed
                + RESIDUAL_BRANCH_INITIALIZATION_SEED_OFFSET_V4
            ),
        )


GeometryAnchoredSweptProgressSurvivalJointJepaV4Config = (
    GeometryAnchoredDeformableBevLiftJointJepaV1Config
)
# The frozen runner resolves this historical name from its selected model module.
GeometryAnchoredDeformableBevLiftJointJepaV1 = (
    GeometryAnchoredSweptProgressSurvivalJointJepaV4
)
final_class_macro_nll_per_row_v1 = final_class_macro_nll_per_row
latent_energy_per_row_v1 = latent_energy_per_row


__all__ = [
    "ACTION_VOCABULARY_V1",
    "FREE_CLASS_V1",
    "GeometryAnchoredBevSamplingV1",
    "GeometryAnchoredDeformableBevLiftJointJepaV1",
    "GeometryAnchoredDeformableBevLiftJointJepaV1Config",
    "GeometryAnchoredSweptProgressSurvivalJointJepaV4",
    "GeometryAnchoredSweptProgressSurvivalJointJepaV4Config",
    "OCCUPIED_CLASS_V1",
    "RESIDUAL_BRANCH_INITIALIZATION_SEED_OFFSET_V4",
    "RESIDUAL_LOCAL_SEMANTIC_DECODER_ADDED_PARAMETER_COUNT_V4",
    "ResidualLocalSemanticDecoderV4",
    "SWEEP_PROGRESS_BIN_COUNT_V1",
    "SweptProgressSurvivalHeadV1",
    "SweptProgressSurvivalPredictionV1",
    "UNKNOWN_CLASS_V1",
    "final_class_macro_nll_per_row",
    "final_class_macro_nll_per_row_v1",
    "latent_energy_per_row",
    "latent_energy_per_row_v1",
]
