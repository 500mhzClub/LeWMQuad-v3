"""Learned camera evidence head for the observable V4 target.

One patch-7 :class:`VisionEncoder` produces a shared dense image feature map.
The pixel branch predicts an ordered obstacle first-hit distribution and a
bounded within-bin distance offset on the registered stride-2 ray lattice.
The ground branch projects the canonical source-cell supports from each
frame's measured camera calibration, samples the same dense feature map in
bounded chunks, and predicts clear-to-target logits.

The model consumes only normalized RGB and current-frame calibration.  It does
not consume a physical map, collision geometry, morphology, or target labels,
and it does not threshold or repair its learned outputs.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from lewm.benchmarks.go2_observable_camera_ray_evidence_v4 import (
    CAMERA_BASIS_ORTHONORMAL_ATOL,
    CAMERA_HORIZONTAL_FOV_DEG,
    CAMERA_IMAGE_SHAPE,
    CAMERA_NEAR_M,
    CAMERA_VERTICAL_FOV_DEG,
    GROUND_SUPPORT_COUNT,
    GROUND_SUPPORT_OFFSETS_CELL_FRACTION,
    OUTPUT_CELL_SIZE_M,
    PIXEL_RAY_SHAPE,
    SOURCE_CELL_SIZE_M,
    SOURCE_FORWARD_MIN_EDGE_M,
    SOURCE_LEFT_MIN_EDGE_M,
    SOURCE_SHAPE,
)

from .encoders import VisionEncoder


IMAGE_SIZE = 112
PATCH_SIZE = 7
TOKEN_SIDE = IMAGE_SIZE // PATCH_SIZE
ENCODER_DIM = 192
ENCODER_DEPTH = 6
ENCODER_HEADS = 6

DEPTH_BIN_COUNT = 64
DEPTH_BIN_SIZE_M = float(OUTPUT_CELL_SIZE_M)
DEPTH_NEAR_EDGE_M = float(CAMERA_NEAR_M)
DEPTH_FAR_EDGE_M = DEPTH_NEAR_EDGE_M + DEPTH_BIN_COUNT * DEPTH_BIN_SIZE_M

DENSE_FEATURE_DIM = 36
GROUND_HIDDEN_DIM = 64
DEFAULT_QUERY_CHUNK_SIZE = 4096


@dataclass(frozen=True)
class OrderedObstacleFirstHitLogProbabilitiesV4:
    """Normalized ordered obstacle-hit probabilities for each camera ray."""

    hit: torch.Tensor
    no_hit: torch.Tensor


@dataclass(frozen=True)
class GroundQueryGeometryV4:
    """Exact calibrated geometry for one batch of canonical ground supports."""

    in_frustum: torch.Tensor
    uv_px: torch.Tensor
    target_distance_m: torch.Tensor
    sample_grid: torch.Tensor

    def __post_init__(self) -> None:
        if not all(
            isinstance(value, torch.Tensor)
            for value in (
                self.in_frustum,
                self.uv_px,
                self.target_distance_m,
                self.sample_grid,
            )
        ):
            raise TypeError("ground query geometry fields must be tensors")
        if self.in_frustum.ndim != 4:
            raise ValueError("ground query geometry must have shape (B,Sx,Sy,K)")
        query_shape = tuple(self.in_frustum.shape)
        if tuple(self.target_distance_m.shape) != query_shape:
            raise ValueError("ground target distances do not match query validity")
        if tuple(self.uv_px.shape) != (*query_shape, 2):
            raise ValueError("ground UV coordinates do not match query validity")
        if tuple(self.sample_grid.shape) != (*query_shape, 2):
            raise ValueError("ground sample grid does not match query validity")
        if self.in_frustum.dtype != torch.bool:
            raise ValueError("ground in-frustum field must be boolean")


@dataclass(frozen=True)
class ObservableCameraRayEvidenceV4RawOutput:
    """Unthresholded learned evidence and deterministic query calibration."""

    pixel_first_hit_hazard_logits: torch.Tensor
    pixel_within_bin_offset_m: torch.Tensor
    ground_clear_to_target_logits: torch.Tensor
    ground_query_in_frustum: torch.Tensor
    ground_query_uv_px: torch.Tensor
    ground_target_distance_m: torch.Tensor

    def __post_init__(self) -> None:
        hazard = self.pixel_first_hit_hazard_logits
        offset = self.pixel_within_bin_offset_m
        ground = self.ground_clear_to_target_logits
        in_frustum = self.ground_query_in_frustum
        uv = self.ground_query_uv_px
        distance = self.ground_target_distance_m
        if not all(
            isinstance(value, torch.Tensor)
            for value in (hazard, offset, ground, in_frustum, uv, distance)
        ):
            raise TypeError("raw V4 output fields must be tensors")
        if hazard.ndim != 4 or hazard.shape[1] <= 0:
            raise ValueError("pixel hazards must have shape (B,D,Hray,Wray)")
        if tuple(offset.shape) != tuple(hazard.shape):
            raise ValueError("pixel offsets must match pixel hazards")
        if ground.ndim != 4 or ground.shape[0] != hazard.shape[0]:
            raise ValueError("ground logits must have shape (B,Sx,Sy,K)")
        if tuple(in_frustum.shape) != tuple(ground.shape):
            raise ValueError("ground validity must match ground logits")
        if tuple(distance.shape) != tuple(ground.shape):
            raise ValueError("ground distances must match ground logits")
        if tuple(uv.shape) != (*ground.shape, 2):
            raise ValueError("ground UV coordinates must match ground logits")
        if in_frustum.dtype != torch.bool:
            raise ValueError("ground query validity must be boolean")


def ordered_obstacle_first_hit_log_probabilities_v4(
    hazard_logits: torch.Tensor,
) -> OrderedObstacleFirstHitLogProbabilitiesV4:
    """Normalize ordered per-bin hazards into one first hit or no hit.

    ``hazard_logits`` has shape ``(B,D,H,W)``.  Hit bin ``d`` means every
    earlier bin survived and bin ``d`` terminated the ray.  Computation stays
    in log space so long-ray survival remains finite.
    """

    if not isinstance(hazard_logits, torch.Tensor):
        raise TypeError("hazard_logits must be a tensor")
    if hazard_logits.ndim != 4 or hazard_logits.shape[1] <= 0:
        raise ValueError("hazard_logits must have shape (B,D,H,W)")
    if not hazard_logits.is_floating_point():
        raise ValueError("hazard_logits must be floating point")
    log_survive = F.logsigmoid(-hazard_logits)
    exclusive_prefix = torch.cat(
        (
            torch.zeros_like(log_survive[:, :1]),
            torch.cumsum(log_survive, dim=1)[:, :-1],
        ),
        dim=1,
    )
    return OrderedObstacleFirstHitLogProbabilitiesV4(
        hit=exclusive_prefix + F.logsigmoid(hazard_logits),
        no_hit=log_survive.sum(dim=1),
    )


def sample_dense_features_at_ground_queries_v4(
    dense_features: torch.Tensor,
    query_geometry: GroundQueryGeometryV4,
    *,
    query_chunk_size: int | None,
) -> torch.Tensor:
    """Bilinearly sample ``(B,C,H,W)`` features at arbitrary ground queries.

    The result has shape ``(B,Sx,Sy,K,C)``.  Chunking bounds the largest
    temporary grid-sample tensor without changing the per-query operation.
    """

    if not isinstance(dense_features, torch.Tensor) or dense_features.ndim != 4:
        raise ValueError("dense_features must have shape (B,C,H,W)")
    if not dense_features.is_floating_point():
        raise ValueError("dense_features must be floating point")
    if dense_features.shape[0] != query_geometry.in_frustum.shape[0]:
        raise ValueError("dense feature and ground query batches differ")
    if dense_features.device != query_geometry.sample_grid.device:
        raise ValueError("dense features and ground queries must share a device")
    query_shape = tuple(query_geometry.in_frustum.shape[1:])
    batch = dense_features.shape[0]
    channels = dense_features.shape[1]
    flat_grid = query_geometry.sample_grid.reshape(batch, -1, 2)
    query_count = flat_grid.shape[1]
    if query_chunk_size is None:
        chunk_size = query_count
    else:
        chunk_size = int(query_chunk_size)
        if chunk_size <= 0:
            raise ValueError("query_chunk_size must be positive or None")
    sampled_chunks = []
    for start in range(0, query_count, chunk_size):
        grid = flat_grid[:, start : start + chunk_size, None, :]
        sampled = F.grid_sample(
            dense_features,
            grid.to(dtype=dense_features.dtype),
            mode="bilinear",
            padding_mode="zeros",
            align_corners=False,
        )
        sampled_chunks.append(sampled.squeeze(-1).transpose(1, 2))
    sampled_flat = torch.cat(sampled_chunks, dim=1)
    return sampled_flat.reshape(batch, *query_shape, channels)


class ObservableCameraRayEvidenceV4Model(nn.Module):
    """Shared patch-7 RGB encoder with pixel-ray and ground-query heads."""

    def __init__(
        self,
        *,
        encoder_depth: int = ENCODER_DEPTH,
        source_shape: tuple[int, int] = SOURCE_SHAPE,
        pixel_ray_shape: tuple[int, int] = PIXEL_RAY_SHAPE,
        query_chunk_size: int | None = DEFAULT_QUERY_CHUNK_SIZE,
    ) -> None:
        super().__init__()
        self.source_shape = self._positive_shape(source_shape, name="source_shape")
        self.pixel_ray_shape = self._positive_shape(
            pixel_ray_shape, name="pixel_ray_shape"
        )
        self.query_chunk_size = (
            None if query_chunk_size is None else int(query_chunk_size)
        )
        if self.query_chunk_size is not None and self.query_chunk_size <= 0:
            raise ValueError("query_chunk_size must be positive or None")
        if int(encoder_depth) < 0:
            raise ValueError("encoder_depth must be non-negative")

        self.encoder = VisionEncoder(
            image_size=IMAGE_SIZE,
            patch_size=PATCH_SIZE,
            hidden_dim=ENCODER_DIM,
            depth=int(encoder_depth),
            n_heads=ENCODER_HEADS,
        )
        self.dense_decoder = nn.Sequential(
            nn.ConvTranspose2d(
                ENCODER_DIM,
                DENSE_FEATURE_DIM,
                kernel_size=PATCH_SIZE,
                stride=PATCH_SIZE,
            ),
            nn.GroupNorm(4, DENSE_FEATURE_DIM),
            nn.GELU(),
            nn.Conv2d(
                DENSE_FEATURE_DIM,
                DENSE_FEATURE_DIM,
                kernel_size=3,
                padding=1,
            ),
            nn.GroupNorm(4, DENSE_FEATURE_DIM),
            nn.GELU(),
        )
        self.pixel_head = nn.Conv2d(
            DENSE_FEATURE_DIM,
            2 * DEPTH_BIN_COUNT,
            kernel_size=1,
        )
        ground_input_dim = DENSE_FEATURE_DIM + 4
        self.ground_head = nn.Sequential(
            nn.Linear(ground_input_dim, GROUND_HIDDEN_DIM),
            nn.GELU(),
            nn.Linear(GROUND_HIDDEN_DIM, 1),
        )
        self._init_heads()

        support_xy = self._canonical_support_xy(self.source_shape)
        self.register_buffer(
            "canonical_ground_support_xy_body_m",
            support_xy,
            persistent=True,
        )

    @staticmethod
    def _positive_shape(value: tuple[int, int], *, name: str) -> tuple[int, int]:
        if len(value) != 2:
            raise ValueError(f"{name} must contain two dimensions")
        result = tuple(int(item) for item in value)
        if any(item <= 0 for item in result):
            raise ValueError(f"{name} dimensions must be positive")
        return result

    @staticmethod
    def _canonical_support_xy(source_shape: tuple[int, int]) -> torch.Tensor:
        forward = SOURCE_FORWARD_MIN_EDGE_M + (
            torch.arange(source_shape[0], dtype=torch.float64) + 0.5
        ) * SOURCE_CELL_SIZE_M
        left = SOURCE_LEFT_MIN_EDGE_M + (
            torch.arange(source_shape[1], dtype=torch.float64) + 0.5
        ) * SOURCE_CELL_SIZE_M
        forward_grid, left_grid = torch.meshgrid(forward, left, indexing="ij")
        offsets = (
            torch.tensor(
                GROUND_SUPPORT_OFFSETS_CELL_FRACTION,
                dtype=torch.float64,
            )
            * SOURCE_CELL_SIZE_M
        )
        result = torch.empty(
            *source_shape,
            GROUND_SUPPORT_COUNT,
            2,
            dtype=torch.float64,
        )
        result[..., 0] = forward_grid[..., None] + offsets[:, 0]
        result[..., 1] = left_grid[..., None] + offsets[:, 1]
        return result

    def _init_heads(self) -> None:
        nn.init.xavier_uniform_(self.pixel_head.weight)
        nn.init.zeros_(self.pixel_head.bias)
        with torch.no_grad():
            self.pixel_head.bias[:DEPTH_BIN_COUNT] = -math.log(
                float(DEPTH_BIN_COUNT - 1)
            )
        for module in self.ground_head.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)

    @staticmethod
    def _validate_image(image: torch.Tensor) -> None:
        if not isinstance(image, torch.Tensor) or image.ndim != 4:
            raise ValueError("image must have shape (B,3,112,112)")
        if tuple(image.shape[1:]) != (3, IMAGE_SIZE, IMAGE_SIZE):
            raise ValueError("image must have shape (B,3,112,112)")
        if not image.is_floating_point():
            raise ValueError("image must be floating point")
        if not bool(torch.isfinite(image).all().item()):
            raise ValueError("image must be finite")

    @staticmethod
    def _validate_calibration(
        camera_origin_body_m: torch.Tensor,
        camera_basis_body_fru: torch.Tensor,
        ground_plane_z_body_m: torch.Tensor,
        *,
        batch: int,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        values = (
            camera_origin_body_m,
            camera_basis_body_fru,
            ground_plane_z_body_m,
        )
        if not all(isinstance(value, torch.Tensor) for value in values):
            raise TypeError("camera calibration inputs must be tensors")
        if tuple(camera_origin_body_m.shape) != (batch, 3):
            raise ValueError("camera_origin_body_m must have shape (B,3)")
        if tuple(camera_basis_body_fru.shape) != (batch, 3, 3):
            raise ValueError("camera_basis_body_fru must have shape (B,3,3)")
        if tuple(ground_plane_z_body_m.shape) != (batch,):
            raise ValueError("ground_plane_z_body_m must have shape (B,)")
        if not all(value.is_floating_point() for value in values):
            raise ValueError("camera calibration inputs must be floating point")
        if any(value.device != device for value in values):
            raise ValueError("image and camera calibration must share a device")
        if not all(bool(torch.isfinite(value).all().item()) for value in values):
            raise ValueError("camera calibration inputs must be finite")
        origin = camera_origin_body_m.to(dtype=torch.float64)
        basis = camera_basis_body_fru.to(dtype=torch.float64)
        ground_z = ground_plane_z_body_m.to(dtype=torch.float64)
        identity = torch.eye(3, dtype=torch.float64, device=device)[None]
        gram = torch.bmm(basis, basis.transpose(1, 2))
        if not torch.allclose(
            gram,
            identity.expand(batch, -1, -1),
            rtol=0.0,
            atol=CAMERA_BASIS_ORTHONORMAL_ATOL,
        ):
            raise ValueError("camera basis must be orthonormal")
        handed_up = torch.linalg.cross(basis[:, 1], basis[:, 0], dim=1)
        if not torch.allclose(
            handed_up,
            basis[:, 2],
            rtol=0.0,
            atol=CAMERA_BASIS_ORTHONORMAL_ATOL,
        ):
            raise ValueError("camera basis must use forward/right/up handedness")
        return origin, basis, ground_z

    def encode_dense_features(self, image: torch.Tensor) -> torch.Tensor:
        """Run the shared encoder exactly once and return a 112x112 feature map."""

        self._validate_image(image)
        tokens = self.encoder.forward_tokens(image)[:, 1:]
        if tuple(tokens.shape[1:]) != (TOKEN_SIDE * TOKEN_SIDE, ENCODER_DIM):
            raise RuntimeError("patch-7 encoder returned an unexpected token grid")
        token_map = tokens.transpose(1, 2).reshape(
            image.shape[0],
            ENCODER_DIM,
            TOKEN_SIDE,
            TOKEN_SIDE,
        )
        dense = self.dense_decoder(token_map)
        if tuple(dense.shape[1:]) != (
            DENSE_FEATURE_DIM,
            IMAGE_SIZE,
            IMAGE_SIZE,
        ):
            raise RuntimeError("dense decoder returned an unexpected feature map")
        return dense

    def ground_query_geometry(
        self,
        camera_origin_body_m: torch.Tensor,
        camera_basis_body_fru: torch.Tensor,
        ground_plane_z_body_m: torch.Tensor,
    ) -> GroundQueryGeometryV4:
        """Project every canonical support using the supplied frame calibration."""

        if not isinstance(camera_origin_body_m, torch.Tensor):
            raise TypeError("camera_origin_body_m must be a tensor")
        batch = camera_origin_body_m.shape[0] if camera_origin_body_m.ndim else 0
        origin, basis, ground_z = self._validate_calibration(
            camera_origin_body_m,
            camera_basis_body_fru,
            ground_plane_z_body_m,
            batch=batch,
            device=camera_origin_body_m.device,
        )
        support_xy = self.canonical_ground_support_xy_body_m.to(
            device=origin.device,
            dtype=torch.float64,
        )
        points = torch.empty(
            batch,
            *self.source_shape,
            GROUND_SUPPORT_COUNT,
            3,
            dtype=torch.float64,
            device=origin.device,
        )
        points[..., :2] = support_xy[None]
        points[..., 2] = ground_z[:, None, None, None]
        relative = points - origin[:, None, None, None, :]
        camera_coordinates = torch.einsum("bsqki,bji->bsqkj", relative, basis)
        forward = camera_coordinates[..., 0]
        right = camera_coordinates[..., 1]
        up = camera_coordinates[..., 2]
        distance = torch.linalg.vector_norm(relative, dim=-1)
        tan_h = math.tan(math.radians(CAMERA_HORIZONTAL_FOV_DEG) * 0.5)
        tan_v = math.tan(math.radians(CAMERA_VERTICAL_FOV_DEG) * 0.5)
        safe_forward = torch.where(
            forward.abs() > 1e-12,
            forward,
            torch.ones_like(forward),
        )
        normalized_x = right / (safe_forward * tan_h)
        normalized_y = up / (safe_forward * tan_v)
        in_frustum = (
            (forward > CAMERA_NEAR_M)
            & (normalized_x.abs() <= 1.0 + 1e-12)
            & (normalized_y.abs() <= 1.0 + 1e-12)
        )
        image_height, image_width = CAMERA_IMAGE_SHAPE
        uv = torch.stack(
            (
                (normalized_x + 1.0) * (float(image_width) * 0.5),
                (1.0 - normalized_y) * (float(image_height) * 0.5),
            ),
            dim=-1,
        )
        raw_sample_grid = torch.stack((normalized_x, -normalized_y), dim=-1)
        sample_grid = torch.where(
            in_frustum[..., None],
            raw_sample_grid,
            raw_sample_grid.new_full((), 2.0),
        )
        return GroundQueryGeometryV4(
            in_frustum=in_frustum,
            uv_px=uv,
            target_distance_m=distance,
            sample_grid=sample_grid,
        )

    def pixel_branch(
        self,
        dense_features: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if tuple(dense_features.shape[1:]) != (
            DENSE_FEATURE_DIM,
            IMAGE_SIZE,
            IMAGE_SIZE,
        ):
            raise ValueError("dense feature shape changed")
        ray_features = F.interpolate(
            dense_features,
            size=self.pixel_ray_shape,
            mode="bilinear",
            align_corners=False,
        )
        raw = self.pixel_head(ray_features).reshape(
            dense_features.shape[0],
            2,
            DEPTH_BIN_COUNT,
            *self.pixel_ray_shape,
        )
        hazard_logits = raw[:, 0]
        bounded_offset_m = 0.5 * DEPTH_BIN_SIZE_M * torch.tanh(raw[:, 1])
        return hazard_logits, bounded_offset_m

    def ground_branch(
        self,
        dense_features: torch.Tensor,
        query_geometry: GroundQueryGeometryV4,
        *,
        query_chunk_size: int | None = None,
    ) -> torch.Tensor:
        chunk_size = (
            self.query_chunk_size
            if query_chunk_size is None
            else int(query_chunk_size)
        )
        sampled = sample_dense_features_at_ground_queries_v4(
            dense_features,
            query_geometry,
            query_chunk_size=chunk_size,
        )
        sample_dtype = sampled.dtype
        normalized_uv = torch.stack(
            (
                query_geometry.uv_px[..., 0]
                / (float(CAMERA_IMAGE_SHAPE[1]) * 0.5)
                - 1.0,
                query_geometry.uv_px[..., 1]
                / (float(CAMERA_IMAGE_SHAPE[0]) * 0.5)
                - 1.0,
            ),
            dim=-1,
        ).to(dtype=sample_dtype)
        distance_feature = (
            torch.log1p(query_geometry.target_distance_m)
            / math.log1p(DEPTH_FAR_EDGE_M)
        )[..., None].to(dtype=sample_dtype)
        validity_feature = query_geometry.in_frustum[..., None].to(
            dtype=sample_dtype
        )
        features = torch.cat(
            (sampled, normalized_uv, distance_feature, validity_feature),
            dim=-1,
        )
        return self.ground_head(features).squeeze(-1)

    def forward(
        self,
        image: torch.Tensor,
        camera_origin_body_m: torch.Tensor,
        camera_basis_body_fru: torch.Tensor,
        ground_plane_z_body_m: torch.Tensor,
        *,
        query_chunk_size: int | None = None,
    ) -> ObservableCameraRayEvidenceV4RawOutput:
        dense = self.encode_dense_features(image)
        query_geometry = self.ground_query_geometry(
            camera_origin_body_m,
            camera_basis_body_fru,
            ground_plane_z_body_m,
        )
        if query_geometry.in_frustum.shape[0] != image.shape[0]:
            raise ValueError("image and calibration batches differ")
        hazard_logits, offset_m = self.pixel_branch(dense)
        ground_logits = self.ground_branch(
            dense,
            query_geometry,
            query_chunk_size=query_chunk_size,
        )
        return ObservableCameraRayEvidenceV4RawOutput(
            pixel_first_hit_hazard_logits=hazard_logits,
            pixel_within_bin_offset_m=offset_m,
            ground_clear_to_target_logits=ground_logits,
            ground_query_in_frustum=query_geometry.in_frustum,
            ground_query_uv_px=query_geometry.uv_px,
            ground_target_distance_m=query_geometry.target_distance_m,
        )


# This literal is verified by a focused test against the registered default.
REGISTERED_PARAMETER_COUNT = 3_105_513


__all__ = [
    "DEFAULT_QUERY_CHUNK_SIZE",
    "DENSE_FEATURE_DIM",
    "DEPTH_BIN_COUNT",
    "DEPTH_BIN_SIZE_M",
    "DEPTH_FAR_EDGE_M",
    "DEPTH_NEAR_EDGE_M",
    "ENCODER_DEPTH",
    "GroundQueryGeometryV4",
    "IMAGE_SIZE",
    "ObservableCameraRayEvidenceV4Model",
    "ObservableCameraRayEvidenceV4RawOutput",
    "OrderedObstacleFirstHitLogProbabilitiesV4",
    "PATCH_SIZE",
    "REGISTERED_PARAMETER_COUNT",
    "ordered_obstacle_first_hit_log_probabilities_v4",
    "sample_dense_features_at_ground_queries_v4",
]
