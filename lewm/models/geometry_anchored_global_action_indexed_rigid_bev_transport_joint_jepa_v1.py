"""Geometry-anchored global action-indexed rigid BEV transport joint JEPA V1.

The RGB encoder, geometry-anchored lift, semantic head, and EMA target are the
frozen V1 representation.  The sole scientific change is the predictor: each
action selects one learned, bounded SE(2)-style transform of the complete BEV,
then all actions share the same local residual corrector.
"""
from __future__ import annotations

import copy
import math
from typing import Mapping

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
    _LocalResidualBlockV1,
    _validate_action_one_hot,
    _validate_n320_encoder_state,
    final_class_macro_nll_per_row,
    latent_energy_per_row,
)


class GlobalActionIndexedRigidBevTransportPredictorV1(nn.Module):
    """One bounded global rigid warp per action plus one shared corrector."""

    maximum_translation_cells: float = 8.0
    maximum_yaw_radians: float = math.pi / 4.0

    def __init__(
        self, config: GeometryAnchoredDeformableBevLiftJointJepaV1Config
    ) -> None:
        super().__init__()
        self.config = config
        self.raw_twist = nn.Parameter(
            torch.zeros((config.action_dim, 3), dtype=torch.float32)
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
            bias=True,
        )
        nn.init.zeros_(self.residual_head.weight)
        nn.init.zeros_(self.residual_head.bias)

    def _validate_inputs(
        self, current_latent: torch.Tensor, action_one_hot: torch.Tensor
    ) -> torch.Tensor:
        expected = (self.config.bev_dim, *self.config.bev_size)
        if current_latent.ndim != 4 or tuple(current_latent.shape[1:]) != expected:
            raise ValueError(f"current_latent must have shape (B,{expected})")
        if current_latent.shape[0] < 1:
            raise ValueError("current_latent must contain at least one row")
        if current_latent.dtype != torch.float32:
            raise TypeError("current_latent must use exact float32")
        if current_latent.device != self.raw_twist.device:
            raise TypeError("current_latent and predictor must share a device")
        if not bool(torch.isfinite(current_latent).all()):
            raise FloatingPointError("current_latent is nonfinite")
        if not bool(torch.isfinite(self.raw_twist).all()):
            raise FloatingPointError("raw_twist is nonfinite")
        return _validate_action_one_hot(
            action_one_hot,
            batch=current_latent.shape[0],
            reference=current_latent,
        )

    def _affine_from_indices(self, action_indices: torch.Tensor) -> torch.Tensor:
        selected = self.raw_twist[action_indices]
        forward_cells = self.maximum_translation_cells * torch.tanh(selected[:, 0])
        left_cells = self.maximum_translation_cells * torch.tanh(selected[:, 1])
        theta = self.maximum_yaw_radians * torch.tanh(selected[:, 2])
        height, width = self.config.bev_size
        tx = 2.0 * left_cells / float(width)
        ty = 2.0 * forward_cells / float(height)
        cosine = torch.cos(theta)
        sine = torch.sin(theta)
        return torch.stack(
            (
                torch.stack((cosine, -sine, tx), dim=1),
                torch.stack((sine, cosine, ty), dim=1),
            ),
            dim=1,
        )

    def selected_affine(
        self, current_latent: torch.Tensor, action_one_hot: torch.Tensor
    ) -> torch.Tensor:
        """Return the validated output-to-input affine for each batch row."""

        action_indices = self._validate_inputs(current_latent, action_one_hot)
        return self._affine_from_indices(action_indices)

    def transport(
        self, current_latent: torch.Tensor, action_one_hot: torch.Tensor
    ) -> torch.Tensor:
        """Apply the selected global rigid transport to the complete BEV."""

        affine = self.selected_affine(current_latent, action_one_hot)
        grid = F.affine_grid(affine, current_latent.shape, align_corners=False)
        return F.grid_sample(
            current_latent,
            grid,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=False,
        )

    def forward(
        self, current_latent: torch.Tensor, action_one_hot: torch.Tensor
    ) -> torch.Tensor:
        transported = self.transport(current_latent, action_one_hot)
        corrected = transported
        for block in self.residual_blocks:
            corrected = block(corrected)
        return transported + self.residual_head(corrected)


class GeometryAnchoredGlobalActionIndexedRigidBevTransportJointJepaV1(
    _FrozenRepresentationJointJepaV1
):
    """Frozen geometry-anchored representation with the rigid predictor."""

    def __init__(
        self,
        n320_encoder_state_dict: Mapping[str, torch.Tensor],
        config: GeometryAnchoredDeformableBevLiftJointJepaV1Config | None = None,
    ) -> None:
        # Do not call the predecessor constructor: it would construct and then
        # discard the closed broadcast-action predictor.  This is the same
        # construction sequence with the replacement installed at that point.
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
            self.predictor = GlobalActionIndexedRigidBevTransportPredictorV1(
                self.config
            )
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


GeometryAnchoredGlobalActionIndexedRigidBevTransportJointJepaV1Config = (
    GeometryAnchoredDeformableBevLiftJointJepaV1Config
)
# The frozen runner resolves this historical API name from its selected model
# module.  In this module it intentionally denotes the registered replacement.
GeometryAnchoredDeformableBevLiftJointJepaV1 = (
    GeometryAnchoredGlobalActionIndexedRigidBevTransportJointJepaV1
)
final_class_macro_nll_per_row_v1 = final_class_macro_nll_per_row
latent_energy_per_row_v1 = latent_energy_per_row


__all__ = [
    "ACTION_VOCABULARY_V1",
    "FREE_CLASS_V1",
    "GeometryAnchoredBevSamplingV1",
    "GeometryAnchoredDeformableBevLiftJointJepaV1",
    "GeometryAnchoredGlobalActionIndexedRigidBevTransportJointJepaV1",
    "GeometryAnchoredGlobalActionIndexedRigidBevTransportJointJepaV1Config",
    "GlobalActionIndexedRigidBevTransportPredictorV1",
    "OCCUPIED_CLASS_V1",
    "UNKNOWN_CLASS_V1",
    "final_class_macro_nll_per_row",
    "final_class_macro_nll_per_row_v1",
    "latent_energy_per_row",
    "latent_energy_per_row_v1",
]
