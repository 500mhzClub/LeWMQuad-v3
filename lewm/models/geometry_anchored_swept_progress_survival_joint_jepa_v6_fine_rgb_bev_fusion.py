"""V4 swept-progress joint JEPA with fine-RGB BEV residual fusion."""
from __future__ import annotations

import copy
from typing import Mapping

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


FINE_RGB_BRANCH_INITIALIZATION_SEED_V6 = 20260714
FINE_RGB_BRANCH_ADDED_TRAINABLE_PARAMETER_COUNT_V6 = 12_256


class FineRgbBevResidualV6(nn.Module):
    """Pixel-resolution RGB features ending in an exact-zero projection."""

    def __init__(self) -> None:
        super().__init__()
        caller_rng = torch.random.get_rng_state().clone()
        try:
            torch.random.default_generator.manual_seed(
                FINE_RGB_BRANCH_INITIALIZATION_SEED_V6
            )
            self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
            self.activation1 = nn.GELU(approximate="none")
            self.conv2 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)
            self.activation2 = nn.GELU(approximate="none")
            self.output = nn.Conv2d(32, 64, kernel_size=1, stride=1, padding=0)
            nn.init.zeros_(self.output.weight)
            nn.init.zeros_(self.output.bias)
        finally:
            torch.random.set_rng_state(caller_rng)

        if sum(parameter.numel() for parameter in self.parameters()) != (
            FINE_RGB_BRANCH_ADDED_TRAINABLE_PARAMETER_COUNT_V6
        ):
            raise RuntimeError("fine-RGB branch parameter count changed")

    def forward(self, rgb: torch.Tensor) -> torch.Tensor:
        return self.output(
            self.activation2(self.conv2(self.activation1(self.conv1(rgb))))
        )


def _fuse_fine_rgb_v6(
    rgb: torch.Tensor,
    inherited: GeometryAnchoredBevSamplingV1,
    branch: FineRgbBevResidualV6,
) -> GeometryAnchoredBevSamplingV1:
    """Sample fine RGB at the inherited locations and add its weighted residual."""

    fine = branch(rgb)
    batch, channels = fine.shape[:2]
    height, width = inherited.cell_valid_mask.shape[-2:]
    samples = inherited.sample_grid_xy.shape[-2]
    packed_grid = inherited.sample_grid_xy.reshape(
        batch, height, width * samples, 2
    )
    sampled = F.grid_sample(
        fine,
        packed_grid,
        mode="bilinear",
        padding_mode="zeros",
        align_corners=False,
    ).reshape(batch, channels, height, width, samples)
    residual = (sampled * inherited.sample_weights.unsqueeze(1)).sum(dim=-1)
    residual = torch.where(
        inherited.cell_valid_mask[:, None], residual, torch.zeros_like(residual)
    )
    return GeometryAnchoredBevSamplingV1(
        latent=inherited.latent + residual,
        anchor_in_frustum=inherited.anchor_in_frustum,
        sample_valid_mask=inherited.sample_valid_mask,
        cell_valid_mask=inherited.cell_valid_mask,
        sample_grid_xy=inherited.sample_grid_xy,
        offsets_token_cells=inherited.offsets_token_cells,
        sample_weights=inherited.sample_weights,
    )


class GeometryAnchoredSweptProgressSurvivalJointJepaV6(
    GeometryAnchoredSweptProgressSurvivalJointJepaV4
):
    """V4 with one zero-gated pixel-resolution RGB path into the BEV latent."""

    def __init__(
        self,
        n320_encoder_state_dict: Mapping[str, torch.Tensor],
        sweep_masks: torch.Tensor,
        config: GeometryAnchoredDeformableBevLiftJointJepaV1Config | None = None,
    ) -> None:
        super().__init__(n320_encoder_state_dict, sweep_masks, config)
        self.bev_lift.fine_rgb_branch = FineRgbBevResidualV6()
        self.target_bev_lift.fine_rgb_branch = copy.deepcopy(
            self.bev_lift.fine_rgb_branch
        )
        self._freeze_target()

    @staticmethod
    def _encode_with_sampling_v6(
        rgb: torch.Tensor, encoder: nn.Module, lift: nn.Module
    ) -> GeometryAnchoredBevSamplingV1:
        patch_tokens = encoder.forward_tokens(rgb)[:, 1:]
        inherited = lift.forward_with_sampling(patch_tokens)
        return _fuse_fine_rgb_v6(rgb, inherited, lift.fine_rgb_branch)

    @staticmethod
    def _encode(rgb: torch.Tensor, encoder: nn.Module, lift: nn.Module) -> torch.Tensor:
        return GeometryAnchoredSweptProgressSurvivalJointJepaV6._encode_with_sampling_v6(
            rgb, encoder, lift
        ).latent

    def encode_online_with_sampling(
        self, rgb: torch.Tensor
    ) -> GeometryAnchoredBevSamplingV1:
        self._validate_rgb(rgb, name="online_rgb")
        return self._encode_with_sampling_v6(rgb, self.encoder, self.bev_lift)

    @torch.no_grad()
    def encode_target_with_sampling(
        self, rgb: torch.Tensor
    ) -> GeometryAnchoredBevSamplingV1:
        self._validate_rgb(rgb, name="target_rgb")
        state = self._encode_with_sampling_v6(
            rgb, self.target_encoder, self.target_bev_lift
        )
        return GeometryAnchoredBevSamplingV1(
            latent=state.latent.detach(),
            anchor_in_frustum=state.anchor_in_frustum,
            sample_valid_mask=state.sample_valid_mask,
            cell_valid_mask=state.cell_valid_mask,
            sample_grid_xy=state.sample_grid_xy,
            offsets_token_cells=state.offsets_token_cells,
            sample_weights=state.sample_weights,
        )


GeometryAnchoredSweptProgressSurvivalJointJepaV6Config = (
    GeometryAnchoredDeformableBevLiftJointJepaV1Config
)
# The frozen runner resolves this historical name from its selected model module.
GeometryAnchoredDeformableBevLiftJointJepaV1 = (
    GeometryAnchoredSweptProgressSurvivalJointJepaV6
)
final_class_macro_nll_per_row_v1 = final_class_macro_nll_per_row
latent_energy_per_row_v1 = latent_energy_per_row


__all__ = [
    "ACTION_VOCABULARY_V1",
    "FINE_RGB_BRANCH_ADDED_TRAINABLE_PARAMETER_COUNT_V6",
    "FINE_RGB_BRANCH_INITIALIZATION_SEED_V6",
    "FREE_CLASS_V1",
    "FineRgbBevResidualV6",
    "GeometryAnchoredBevSamplingV1",
    "GeometryAnchoredDeformableBevLiftJointJepaV1",
    "GeometryAnchoredDeformableBevLiftJointJepaV1Config",
    "GeometryAnchoredSweptProgressSurvivalJointJepaV6",
    "GeometryAnchoredSweptProgressSurvivalJointJepaV6Config",
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
