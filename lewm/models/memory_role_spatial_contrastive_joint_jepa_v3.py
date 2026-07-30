"""Spatial, anti-collapse memory-role interface for the V18 RGB joint-JEPA.

V3 keeps the complete V1 shared, physical, and local-control routes.  Only the
place factorizer and its predictor change: the key retains a small spatial
grid, and the learned predictor starts as an exact identity residual.
"""
from __future__ import annotations

import copy

import torch
import torch.nn as nn
import torch.nn.functional as F

from lewm.models.geometry_anchored_swept_progress_survival_joint_jepa_v18_object_space_height_volume import (
    GeometryAnchoredDeformableBevLiftJointJepaV1Config,
    PROJECTION_INITIALIZATION_SEED_V13,
)
from lewm.models.memory_role_factorized_joint_jepa_v1 import (
    LOCAL_CONTROL_SHAPE_MEMORY_ROLE_FACTORIZED_V1,
    PLACE_KEY_WIDTH_MEMORY_ROLE_FACTORIZED_V1,
    SOURCE_LATENT_SHAPE_MEMORY_ROLE_FACTORIZED_V1,
    MemoryRoleEncodingV1,
    MemoryRoleFactorizedJointJepaV1,
    _validate_float_tensor,
)
from lewm.models.observable_camera_ray_evidence_v4 import (
    ObservableCameraRayEvidenceV4Model,
)


PLACE_SPATIAL_CHANNELS_MEMORY_ROLE_V3 = 16
PLACE_SPATIAL_GRID_MEMORY_ROLE_V3 = (4, 4)
PLACE_FLAT_WIDTH_MEMORY_ROLE_V3 = 256
PLACE_PREDICTOR_HIDDEN_WIDTH_MEMORY_ROLE_V3 = 128
INITIALIZATION_SEED_MEMORY_ROLE_SPATIAL_CONTRASTIVE_V3 = 20_260_732


class MemoryRoleSpatialFactorizerV3(nn.Module):
    """Retain a 4x4 spatial summary while preserving V1's local projection."""

    def __init__(self, *, local_projection: nn.Module) -> None:
        super().__init__()
        if not isinstance(local_projection, nn.Module):
            raise TypeError("local_projection must be an nn.Module")
        self.place_projection = nn.Conv2d(
            SOURCE_LATENT_SHAPE_MEMORY_ROLE_FACTORIZED_V1[0],
            PLACE_SPATIAL_CHANNELS_MEMORY_ROLE_V3,
            kernel_size=1,
        )
        self.activation = nn.GELU(approximate="none")
        self.place_pool = nn.AdaptiveAvgPool2d(PLACE_SPATIAL_GRID_MEMORY_ROLE_V3)
        self.place_output = nn.Linear(
            PLACE_FLAT_WIDTH_MEMORY_ROLE_V3,
            PLACE_KEY_WIDTH_MEMORY_ROLE_FACTORIZED_V1,
        )
        self.local_projection = local_projection

    def forward(self, latent: torch.Tensor) -> MemoryRoleEncodingV1:
        _validate_float_tensor(
            latent,
            shape_after_batch=SOURCE_LATENT_SHAPE_MEMORY_ROLE_FACTORIZED_V1,
            name="latent",
            reference=self.place_projection.weight,
        )
        place_map = self.activation(self.place_projection(latent))
        place_grid = self.place_pool(place_map)
        if tuple(place_grid.shape[1:]) != (
            PLACE_SPATIAL_CHANNELS_MEMORY_ROLE_V3,
            *PLACE_SPATIAL_GRID_MEMORY_ROLE_V3,
        ):
            raise RuntimeError("place spatial-grid output shape changed")
        place_key = F.normalize(
            self.place_output(place_grid.flatten(start_dim=1)),
            dim=1,
            eps=1.0e-6,
        )
        local_control = self.local_projection(latent)
        if tuple(local_control.shape[1:]) != LOCAL_CONTROL_SHAPE_MEMORY_ROLE_FACTORIZED_V1:
            raise RuntimeError("local-control output shape changed")
        return MemoryRoleEncodingV1(place_key, local_control)


class PlaceKeyIdentityResidualPredictorV3(nn.Module):
    """A learned place predictor initialized to the normalized identity map."""

    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(
                PLACE_KEY_WIDTH_MEMORY_ROLE_FACTORIZED_V1,
                PLACE_PREDICTOR_HIDDEN_WIDTH_MEMORY_ROLE_V3,
            ),
            nn.GELU(approximate="none"),
            nn.Linear(
                PLACE_PREDICTOR_HIDDEN_WIDTH_MEMORY_ROLE_V3,
                PLACE_KEY_WIDTH_MEMORY_ROLE_FACTORIZED_V1,
            ),
        )
        nn.init.zeros_(self.net[2].weight)
        nn.init.zeros_(self.net[2].bias)

    def forward(self, place_key: torch.Tensor) -> torch.Tensor:
        _validate_float_tensor(
            place_key,
            shape_after_batch=(PLACE_KEY_WIDTH_MEMORY_ROLE_FACTORIZED_V1,),
            name="place_key",
            reference=self.net[0].weight,
        )
        return F.normalize(place_key + self.net(place_key), dim=1, eps=1.0e-6)


class MemoryRoleSpatialContrastiveJointJepaV3(MemoryRoleFactorizedJointJepaV1):
    """V1 joint-JEPA controls with the preregistered V3 place mechanism."""

    def __init__(
        self,
        n320_fit_model: ObservableCameraRayEvidenceV4Model,
        sweep_masks: torch.Tensor,
        config: GeometryAnchoredDeformableBevLiftJointJepaV1Config | None = None,
    ) -> None:
        caller_rng = torch.random.get_rng_state().clone()
        try:
            super().__init__(n320_fit_model, sweep_masks, config)
            inherited_local_projection = self.role_factorizer.local_projection
            torch.random.default_generator.manual_seed(
                INITIALIZATION_SEED_MEMORY_ROLE_SPATIAL_CONTRASTIVE_V3
            )
            self.role_factorizer = MemoryRoleSpatialFactorizerV3(
                local_projection=inherited_local_projection
            )
            self.place_predictor = PlaceKeyIdentityResidualPredictorV3()
            self.target_role_factorizer = copy.deepcopy(self.role_factorizer)
        finally:
            torch.random.set_rng_state(caller_rng)

        self._freeze_target()
        self._assert_role_target_binding()
        self.trainable_parameter_groups_memory_role_factorized_v1()


__all__ = [
    "INITIALIZATION_SEED_MEMORY_ROLE_SPATIAL_CONTRASTIVE_V3",
    "MemoryRoleSpatialContrastiveJointJepaV3",
    "MemoryRoleSpatialFactorizerV3",
    "PLACE_FLAT_WIDTH_MEMORY_ROLE_V3",
    "PLACE_PREDICTOR_HIDDEN_WIDTH_MEMORY_ROLE_V3",
    "PLACE_SPATIAL_CHANNELS_MEMORY_ROLE_V3",
    "PLACE_SPATIAL_GRID_MEMORY_ROLE_V3",
    "PROJECTION_INITIALIZATION_SEED_V13",
    "PlaceKeyIdentityResidualPredictorV3",
]
