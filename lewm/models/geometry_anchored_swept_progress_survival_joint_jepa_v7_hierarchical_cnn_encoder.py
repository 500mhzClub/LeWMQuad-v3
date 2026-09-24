"""V4 swept-progress joint JEPA with a hierarchical convolutional encoder."""
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


HIERARCHICAL_CNN_ENCODER_INITIALIZATION_SEED_V7 = 20260715
HIERARCHICAL_CNN_ENCODER_TRAINABLE_PARAMETER_COUNT_V7 = 1_994_880


class HierarchicalCnnResidualBlockV7(nn.Module):
    """Two normalized convolutions followed by a residual GELU."""

    def __init__(self, width: int, groups: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(
            width, width, kernel_size=3, stride=1, padding=1, bias=True
        )
        self.norm1 = nn.GroupNorm(groups, width, affine=True)
        self.activation1 = nn.GELU(approximate="none")
        self.conv2 = nn.Conv2d(
            width, width, kernel_size=3, stride=1, padding=1, bias=True
        )
        self.norm2 = nn.GroupNorm(groups, width, affine=True)
        self.activation2 = nn.GELU(approximate="none")

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        residual = value
        value = self.activation1(self.norm1(self.conv1(value)))
        value = self.norm2(self.conv2(value))
        return self.activation2(residual + value)


class HierarchicalCnnEncoderV7(nn.Module):
    """Fresh RGB CNN exposing the existing 257-by-192 token interface."""

    image_size = 112
    hidden_dim = 192
    token_side = 16
    num_patches = 256

    def __init__(self) -> None:
        super().__init__()
        caller_rng = torch.random.get_rng_state().clone()
        try:
            torch.random.default_generator.manual_seed(
                HIERARCHICAL_CNN_ENCODER_INITIALIZATION_SEED_V7
            )
            self.stem_conv = nn.Conv2d(
                3, 48, kernel_size=5, stride=2, padding=2, bias=True
            )
            self.stem_norm = nn.GroupNorm(6, 48, affine=True)
            self.stem_activation = nn.GELU(approximate="none")
            self.stage48 = nn.ModuleList(
                [HierarchicalCnnResidualBlockV7(48, 6) for _ in range(2)]
            )

            self.down96_conv = nn.Conv2d(
                48, 96, kernel_size=3, stride=2, padding=1, bias=True
            )
            self.down96_norm = nn.GroupNorm(8, 96, affine=True)
            self.down96_activation = nn.GELU(approximate="none")
            self.stage96 = nn.ModuleList(
                [HierarchicalCnnResidualBlockV7(96, 8) for _ in range(2)]
            )

            self.down192_conv = nn.Conv2d(
                96, 192, kernel_size=3, stride=2, padding=1, bias=True
            )
            self.down192_norm = nn.GroupNorm(12, 192, affine=True)
            self.down192_activation = nn.GELU(approximate="none")
            self.stage192 = nn.ModuleList(
                [HierarchicalCnnResidualBlockV7(192, 12) for _ in range(2)]
            )

            self.output_projection = nn.Conv2d(
                192, 192, kernel_size=1, stride=1, padding=0, bias=True
            )
        finally:
            torch.random.set_rng_state(caller_rng)

        parameter_count = sum(
            parameter.numel() for parameter in self.parameters()
        )
        if parameter_count != HIERARCHICAL_CNN_ENCODER_TRAINABLE_PARAMETER_COUNT_V7:
            raise RuntimeError("hierarchical CNN encoder parameter count changed")

    def _validate_rgb(self, rgb: torch.Tensor) -> None:
        if not isinstance(rgb, torch.Tensor):
            raise TypeError("rgb must be a tensor")
        if rgb.ndim != 4 or tuple(rgb.shape[1:]) != (3, 112, 112):
            raise ValueError("rgb must have shape (B,3,112,112)")
        if rgb.shape[0] < 1:
            raise ValueError("rgb must contain at least one row")
        if rgb.dtype != torch.float32:
            raise TypeError("rgb must use exact float32")
        if rgb.device != next(self.parameters()).device:
            raise TypeError("rgb and encoder must share a device")
        if not bool(torch.isfinite(rgb).all()):
            raise FloatingPointError("rgb is nonfinite")

    def _spatial_features(self, rgb: torch.Tensor) -> torch.Tensor:
        value = self.stem_activation(self.stem_norm(self.stem_conv(rgb)))
        for block in self.stage48:
            value = block(value)
        value = self.down96_activation(self.down96_norm(self.down96_conv(value)))
        for block in self.stage96:
            value = block(value)
        value = self.down192_activation(
            self.down192_norm(self.down192_conv(value))
        )
        for block in self.stage192:
            value = block(value)
        value = F.interpolate(
            value,
            size=(self.token_side, self.token_side),
            mode="bilinear",
            align_corners=False,
        )
        return self.output_projection(value)

    def forward_tokens(self, rgb: torch.Tensor) -> torch.Tensor:
        """Return mean CLS followed by row-major 16x16 spatial tokens."""

        self._validate_rgb(rgb)
        spatial = self._spatial_features(rgb)
        if tuple(spatial.shape[1:]) != (self.hidden_dim, 16, 16):
            raise RuntimeError("hierarchical CNN spatial shape changed")
        tokens = spatial.flatten(start_dim=2).transpose(1, 2).contiguous()
        cls = tokens.mean(dim=1, keepdim=True)
        output = torch.cat((cls, tokens), dim=1)
        if tuple(output.shape[1:]) != (257, self.hidden_dim):
            raise RuntimeError("hierarchical CNN token shape changed")
        if output.dtype != torch.float32 or not bool(torch.isfinite(output).all()):
            raise FloatingPointError("hierarchical CNN tokens are nonfinite")
        return output

    def forward(self, rgb: torch.Tensor) -> torch.Tensor:
        return self.forward_tokens(rgb)[:, 0]


class GeometryAnchoredSweptProgressSurvivalJointJepaV7(
    GeometryAnchoredSweptProgressSurvivalJointJepaV4
):
    """Clean V4 model with its inherited ViT replaced by one fresh CNN."""

    def __init__(
        self,
        n320_encoder_state_dict: Mapping[str, torch.Tensor],
        sweep_masks: torch.Tensor,
        config: GeometryAnchoredDeformableBevLiftJointJepaV1Config | None = None,
    ) -> None:
        super().__init__(n320_encoder_state_dict, sweep_masks, config)
        if (
            self.config.image_size,
            self.config.encoder_dim,
            self.config.token_side,
        ) != (112, 192, 16):
            raise ValueError(
                "V7 hierarchical CNN requires image_size=112, "
                "encoder_dim=192, and token_side=16"
            )
        self.encoder = HierarchicalCnnEncoderV7()
        self.target_encoder = copy.deepcopy(self.encoder)
        self._freeze_target()


GeometryAnchoredSweptProgressSurvivalJointJepaV7Config = (
    GeometryAnchoredDeformableBevLiftJointJepaV1Config
)
# The frozen runner resolves this historical name from its selected model module.
GeometryAnchoredDeformableBevLiftJointJepaV1 = (
    GeometryAnchoredSweptProgressSurvivalJointJepaV7
)
final_class_macro_nll_per_row_v1 = final_class_macro_nll_per_row
latent_energy_per_row_v1 = latent_energy_per_row


__all__ = [
    "ACTION_VOCABULARY_V1",
    "FREE_CLASS_V1",
    "GeometryAnchoredBevSamplingV1",
    "GeometryAnchoredDeformableBevLiftJointJepaV1",
    "GeometryAnchoredDeformableBevLiftJointJepaV1Config",
    "GeometryAnchoredSweptProgressSurvivalJointJepaV7",
    "GeometryAnchoredSweptProgressSurvivalJointJepaV7Config",
    "HIERARCHICAL_CNN_ENCODER_INITIALIZATION_SEED_V7",
    "HIERARCHICAL_CNN_ENCODER_TRAINABLE_PARAMETER_COUNT_V7",
    "HierarchicalCnnEncoderV7",
    "HierarchicalCnnResidualBlockV7",
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
