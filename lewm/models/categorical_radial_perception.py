"""Geometry-structured categorical radial perception from ego RGB.

The model keeps every registered camera-height anchor separate while lifting
ordered ViT patch tokens into a polar occupancy lattice.  A fixed, audited
factorization then gathers the polar UNKNOWN/FREE/OCCUPIED logits into the
registered 64 x 64 Cartesian local grid.
"""
from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from lewm.benchmarks.go2_categorical_radial_factorization import (
    ANGULAR_BIN_COUNT,
    CARTESIAN_SHAPE,
    RADIAL_BIN_COUNT,
    RadialFactorization,
    build_radial_factorization,
    gather_polar_logits_to_cartesian,
)

from .encoders import VisionEncoder


IMAGE_SIZE = 112
PATCH_SIZE = 7
TOKEN_SIDE = IMAGE_SIZE // PATCH_SIZE
ENCODER_DIM = 192
ENCODER_DEPTH = 6
ENCODER_HEADS = 6
HORIZONTAL_FOV_DEG = 78.323
VERTICAL_FOV_DEG = 62.8370386364
CAMERA_XYZ_BODY_M = (0.326, 0.0, 0.043)
CAMERA_RPY_BODY_RAD = (0.0, 0.0, 0.0)
CAMERA_NEAR_M = 0.05
VERTICAL_ANCHOR_Z_BODY_M = (-0.333, -0.133, 0.067, 0.267, 0.467)
POLAR_SHAPE = (RADIAL_BIN_COUNT, ANGULAR_BIN_COUNT)
CLASS_COUNT = 3


def _registered_projective_geometry(
    factorization: RadialFactorization,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return per-anchor token grids, validity, and polar coordinates."""

    if CAMERA_RPY_BODY_RAD != (0.0, 0.0, 0.0):
        raise ValueError("registered categorical radial camera RPY changed")
    radial = np.asarray(factorization.radial_centers_m, dtype=np.float64)
    angular = np.asarray(factorization.angular_centers_rad, dtype=np.float64)
    if radial.shape != (RADIAL_BIN_COUNT,) or angular.shape != (
        ANGULAR_BIN_COUNT,
    ):
        raise ValueError("radial factorization uses an unexpected polar lattice")

    radius_grid, angle_grid = np.meshgrid(radial, angular, indexing="ij")
    point_forward = radius_grid * np.cos(angle_grid)
    point_left = radius_grid * np.sin(angle_grid)
    camera_forward = point_forward - CAMERA_XYZ_BODY_M[0]
    camera_left = point_left - CAMERA_XYZ_BODY_M[1]
    tan_horizontal = math.tan(math.radians(HORIZONTAL_FOV_DEG) * 0.5)
    tan_vertical = math.tan(math.radians(VERTICAL_FOV_DEG) * 0.5)
    safe_forward = np.where(
        np.abs(camera_forward) > np.finfo(np.float64).eps,
        camera_forward,
        1.0,
    )

    grids = []
    validity = []
    for anchor_z in VERTICAL_ANCHOR_Z_BODY_M:
        camera_up = float(anchor_z) - CAMERA_XYZ_BODY_M[2]
        normalized_u = -camera_left / (safe_forward * tan_horizontal)
        normalized_v = -camera_up / (safe_forward * tan_vertical)
        valid = (
            (camera_forward >= CAMERA_NEAR_M)
            & (normalized_u >= -1.0)
            & (normalized_u <= 1.0)
            & (normalized_v >= -1.0)
            & (normalized_v <= 1.0)
        )
        grid = np.stack((normalized_u, normalized_v), axis=-1)
        # Invalid queries sample padding while their explicit validity channel
        # lets the context network distinguish absence from a zero feature.
        grid = np.where(valid[..., None], grid, 2.0)
        grids.append(grid)
        validity.append(valid)

    radial_normalized = 2.0 * (radial - radial[0]) / (
        radial[-1] - radial[0]
    ) - 1.0
    bearing_normalized = angular / math.radians(HORIZONTAL_FOV_DEG * 0.5)
    radial_feature, bearing_feature = np.meshgrid(
        radial_normalized,
        bearing_normalized,
        indexing="ij",
    )
    coordinate_features = np.stack(
        (
            radial_feature,
            bearing_feature,
            np.sin(angle_grid),
            np.cos(angle_grid),
        ),
        axis=0,
    )
    return (
        torch.from_numpy(np.stack(grids).astype(np.float32)),
        torch.from_numpy(np.stack(validity)),
        torch.from_numpy(coordinate_features.astype(np.float32)),
    )


class _ContextBlock(nn.Module):
    """One residual, weight-shared context pass along a polar axis."""

    def __init__(self, channels: int, *, kernel_size: tuple[int, int]) -> None:
        super().__init__()
        padding = (kernel_size[0] // 2, kernel_size[1] // 2)
        self.net = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size, padding=padding),
            nn.GroupNorm(8, channels),
            nn.GELU(),
            nn.Conv2d(channels, channels, 1),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return features + self.net(features)


class CategoricalRadialPerception(nn.Module):
    """Registered RGB -> categorical polar -> Cartesian physical evidence model."""

    def __init__(
        self,
        *,
        token_feature_dim: int = 24,
        context_dim: int = 64,
    ) -> None:
        super().__init__()
        if int(token_feature_dim) <= 0:
            raise ValueError("token_feature_dim must be positive")
        if int(context_dim) <= 0 or int(context_dim) % 8:
            raise ValueError("context_dim must be a positive multiple of eight")
        self.token_feature_dim = int(token_feature_dim)
        self.context_dim = int(context_dim)
        self.factorization = build_radial_factorization()
        self.encoder = VisionEncoder(
            image_size=IMAGE_SIZE,
            patch_size=PATCH_SIZE,
            hidden_dim=ENCODER_DIM,
            depth=ENCODER_DEPTH,
            n_heads=ENCODER_HEADS,
        )
        self.token_projection = nn.Conv2d(
            ENCODER_DIM,
            self.token_feature_dim,
            kernel_size=1,
        )

        projective_grid, anchor_validity, coordinates = (
            _registered_projective_geometry(self.factorization)
        )
        anchor_count = len(VERTICAL_ANCHOR_Z_BODY_M)
        self.register_buffer("projective_sample_grid", projective_grid)
        self.register_buffer("projective_anchor_validity", anchor_validity)
        self.register_buffer(
            "anchor_identity",
            torch.eye(anchor_count, dtype=torch.float32),
        )
        self.register_buffer("polar_coordinate_features", coordinates)
        self.register_buffer(
            "cartesian_support_mask",
            torch.from_numpy(
                np.asarray(self.factorization.representable_mask, dtype=bool).copy()
            ),
        )

        per_anchor_channels = self.token_feature_dim + anchor_count + 1
        input_channels = anchor_count * per_anchor_channels + int(
            coordinates.shape[0]
        )
        self.context_stem = nn.Sequential(
            nn.Conv2d(input_channels, self.context_dim, 1),
            nn.GroupNorm(8, self.context_dim),
            nn.GELU(),
        )
        self.radial_context = _ContextBlock(
            self.context_dim,
            kernel_size=(5, 1),
        )
        self.angular_context = _ContextBlock(
            self.context_dim,
            kernel_size=(1, 5),
        )
        self.polar_head = nn.Conv2d(self.context_dim, CLASS_COUNT, 1)

    @staticmethod
    def _validate_image(image: torch.Tensor) -> None:
        if image.ndim != 4 or tuple(image.shape[1:]) != (
            3,
            IMAGE_SIZE,
            IMAGE_SIZE,
        ):
            raise ValueError(
                "image must have shape "
                f"(B, 3, {IMAGE_SIZE}, {IMAGE_SIZE}), got {tuple(image.shape)}"
            )
        if not torch.is_floating_point(image):
            raise ValueError("image must use a floating-point dtype")

    def _projected_token_map(self, image: torch.Tensor) -> torch.Tensor:
        tokens = self.encoder.forward_tokens(image)[:, 1:]
        if tokens.shape[1:] != (TOKEN_SIDE * TOKEN_SIDE, ENCODER_DIM):
            raise RuntimeError("registered encoder returned an unexpected token grid")
        token_map = tokens.transpose(1, 2).reshape(
            image.shape[0],
            ENCODER_DIM,
            TOKEN_SIDE,
            TOKEN_SIDE,
        )
        return self.token_projection(token_map)

    def sample_projective_anchors(
        self,
        projected_tokens: torch.Tensor,
    ) -> torch.Tensor:
        """Sample each registered height anchor without reducing their identity."""

        if projected_tokens.ndim != 4:
            raise ValueError(
                "projected_tokens must have shape "
                f"(B, {self.token_feature_dim}, {TOKEN_SIDE}, {TOKEN_SIDE})"
            )
        expected = (
            projected_tokens.shape[0],
            self.token_feature_dim,
            TOKEN_SIDE,
            TOKEN_SIDE,
        )
        if tuple(projected_tokens.shape) != expected:
            raise ValueError(
                "projected_tokens must have shape "
                f"(B, {self.token_feature_dim}, {TOKEN_SIDE}, {TOKEN_SIDE})"
            )
        batch = projected_tokens.shape[0]
        samples = []
        for anchor_index in range(len(VERTICAL_ANCHOR_Z_BODY_M)):
            grid = self.projective_sample_grid[anchor_index][None].expand(
                batch, -1, -1, -1
            )
            sampled = F.grid_sample(
                projected_tokens,
                grid.to(dtype=projected_tokens.dtype),
                mode="bilinear",
                padding_mode="zeros",
                align_corners=False,
            )
            valid = self.projective_anchor_validity[anchor_index][None, None]
            samples.append(sampled * valid.to(dtype=sampled.dtype))
        return torch.stack(samples, dim=1)

    def polar_logits(self, image: torch.Tensor) -> torch.Tensor:
        """Return ``(B, 3, 64, 256)`` UNKNOWN/FREE/OCCUPIED logits."""

        self._validate_image(image)
        projected = self._projected_token_map(image)
        samples = self.sample_projective_anchors(projected)
        batch, anchor_count, _channels, radial_count, angular_count = samples.shape
        identity = self.anchor_identity[None, :, :, None, None].expand(
            batch,
            -1,
            -1,
            radial_count,
            angular_count,
        )
        validity = self.projective_anchor_validity[
            None, :, None
        ].expand(batch, -1, -1, -1, -1)
        anchored = torch.cat(
            (
                samples,
                identity.to(dtype=samples.dtype),
                validity.to(dtype=samples.dtype),
            ),
            dim=2,
        ).reshape(batch, -1, radial_count, angular_count)
        coordinates = self.polar_coordinate_features[None].expand(
            batch, -1, -1, -1
        )
        features = self.context_stem(
            torch.cat((anchored, coordinates.to(dtype=anchored.dtype)), dim=1)
        )
        features = self.radial_context(features)
        features = self.angular_context(features)
        return self.polar_head(features)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """Return finite ``(B, 3, 64, 64)`` Cartesian categorical logits."""

        polar = self.polar_logits(image)
        cartesian = gather_polar_logits_to_cartesian(
            polar,
            factorization=self.factorization,
        )
        if tuple(cartesian.shape[-3:]) != (CLASS_COUNT, *CARTESIAN_SHAPE):
            raise RuntimeError("radial factorization returned an unexpected shape")
        return cartesian

    def occupancy_logits(self, image: torch.Tensor) -> torch.Tensor:
        """Compatibility surface used by the shared physical-map evaluator."""

        return self.forward(image)


__all__ = [
    "CAMERA_NEAR_M",
    "CAMERA_RPY_BODY_RAD",
    "CAMERA_XYZ_BODY_M",
    "CategoricalRadialPerception",
    "ENCODER_DEPTH",
    "ENCODER_DIM",
    "ENCODER_HEADS",
    "HORIZONTAL_FOV_DEG",
    "IMAGE_SIZE",
    "PATCH_SIZE",
    "POLAR_SHAPE",
    "TOKEN_SIDE",
    "VERTICAL_ANCHOR_Z_BODY_M",
    "VERTICAL_FOV_DEG",
]
