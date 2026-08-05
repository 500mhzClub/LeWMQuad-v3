"""Camera-ray ordinal first-surface perception for physical local evidence.

This development-only model keeps the registered patch-7 image encoder, but
predicts an ordered first surface on a dense rectilinear camera-ray lattice.
The predicted floor and obstacle evidence is gathered into the unchanged
64 x 64 yaw-aligned physical grid using the deployment-valid base attitude.

Privileged geometry is not an input. The runtime inputs are normalized RGB,
the timestamp-aligned base quaternion, and its matching stored yaw.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .categorical_radial_perception_full_ray import FullRayRadialContext
from .encoders import VisionEncoder


IMAGE_SIZE = 112
PATCH_SIZE = 7
TOKEN_SIDE = IMAGE_SIZE // PATCH_SIZE
ENCODER_DIM = 192
ENCODER_DEPTH = 6
ENCODER_HEADS = 6

RAY_HEIGHT = 84
RAY_WIDTH = 112
DEPTH_BIN_COUNT = 64
DEPTH_BIN_SIZE_M = 0.10
DEPTH_NEAR_EDGE_M = 0.05
RAY_FEATURE_DIM = 36
DEPTH_CONTEXT_DIM = 8

CARTESIAN_SHAPE = (64, 64)
CARTESIAN_FORWARD_CENTERS_M = (-0.95, 5.35)
CARTESIAN_LEFT_CENTERS_M = (-3.15, 3.15)
HORIZONTAL_FOV_DEG = 78.323
VERTICAL_FOV_DEG = 62.8370386364
CAMERA_XYZ_BODY_M = (0.326, 0.0, 0.043)
CAMERA_RPY_BODY_RAD = (0.0, 0.0, 0.0)
CAMERA_NEAR_M = 0.05
VERTICAL_ANCHOR_Z_BODY_M = (-0.333, -0.133, 0.067, 0.267, 0.467)
CELL_SQUARE_OFFSETS_M = (
    (0.0, 0.0),
    (-0.05, -0.05),
    (-0.05, 0.05),
    (0.05, -0.05),
    (0.05, 0.05),
)

QUATERNION_NORM_TOLERANCE = 1e-5
QUATERNION_YAW_TOLERANCE_RAD = 1e-5
FLOAT32_BOUNDARY_TOLERANCE_ULPS = 8.0

CLASS_COUNT = 3

# Trainable occupancy-path parameters in the frozen dynamic Cartesian model:
# encoder 2,747,520 + decoder 369,856 + occupancy head 195.
DYNAMIC_CARTESIAN_OCCUPANCY_PARAMETER_COUNT = 3_117_571
REGISTERED_PARAMETER_COUNT = 3_118_778
REGISTERED_DECODER_PARAMETER_COUNT = 371_258


@dataclass(frozen=True)
class OrderedRayLogProbabilities:
    """Normalized first-surface probabilities on a camera-ray lattice."""

    obstacle: torch.Tensor
    floor: torch.Tensor
    no_hit: torch.Tensor


@dataclass(frozen=True)
class YawAlignedCameraBasis:
    """Registered camera basis expressed in base-position/stored-yaw axes."""

    origin: torch.Tensor
    forward: torch.Tensor
    left: torch.Tensor
    up: torch.Tensor


def ordered_first_surface_log_probabilities(
    hazard_logits: torch.Tensor,
    obstacle_given_hit_logits: torch.Tensor,
) -> OrderedRayLogProbabilities:
    """Convert ordered hazards into one first hit (floor/obstacle) or no hit.

    Both input tensors have shape (B,D,H,W). Computation stays in log space,
    so late-depth survival remains finite even for confident hazards.
    """

    if (
        not isinstance(hazard_logits, torch.Tensor)
        or not isinstance(obstacle_given_hit_logits, torch.Tensor)
    ):
        raise TypeError("ordered first-surface inputs must be tensors")
    if hazard_logits.shape != obstacle_given_hit_logits.shape:
        raise ValueError("hazard and conditional-surface logits must match")
    if hazard_logits.ndim != 4 or hazard_logits.shape[1] <= 0:
        raise ValueError("ordered first-surface logits must have shape (B,D,H,W)")
    if not hazard_logits.is_floating_point() or not (
        obstacle_given_hit_logits.is_floating_point()
    ):
        raise ValueError("ordered first-surface logits must be floating point")

    log_survive = F.logsigmoid(-hazard_logits)
    zero = torch.zeros_like(log_survive[:, :1])
    exclusive_prefix = torch.cat(
        (zero, torch.cumsum(log_survive, dim=1)[:, :-1]),
        dim=1,
    )
    log_hit = exclusive_prefix + F.logsigmoid(hazard_logits)
    log_obstacle = log_hit + F.logsigmoid(obstacle_given_hit_logits)
    log_floor = log_hit + F.logsigmoid(-obstacle_given_hit_logits)
    log_no_hit = log_survive.sum(dim=1)
    return OrderedRayLogProbabilities(
        obstacle=log_obstacle,
        floor=log_floor,
        no_hit=log_no_hit,
    )


def apply_full_ray_context_per_ray(
    context: nn.Module,
    features: torch.Tensor,
    *,
    ray_chunk_size: int | None,
) -> torch.Tensor:
    """Apply full-ray context independently and equivalently in ray chunks.

    FullRayRadialContext contains GroupNorm. Directly slicing the spatial ray
    axis would change its normalization statistics. Moving rays into the batch
    dimension makes normalization ray-local; chunking that augmented batch is
    then exactly the same mathematical operation as one full call.
    """

    if not isinstance(features, torch.Tensor) or features.ndim != 4:
        raise ValueError("features must have shape (B,C,D,R)")
    batch, channels, depth, rays = features.shape
    if min(batch, channels, depth, rays) <= 0:
        raise ValueError("all full-ray feature dimensions must be positive")
    per_ray = (
        features.permute(0, 3, 1, 2)
        .contiguous()
        .reshape(batch * rays, channels, depth, 1)
    )
    if ray_chunk_size is None:
        mixed = context(per_ray)
    else:
        chunk = int(ray_chunk_size)
        if chunk <= 0:
            raise ValueError("ray_chunk_size must be positive or None")
        mixed = torch.cat(
            tuple(
                context(per_ray[start : start + chunk])
                for start in range(0, per_ray.shape[0], chunk)
            ),
            dim=0,
        )
    if mixed.shape != per_ray.shape:
        raise RuntimeError("full-ray context changed the registered tensor shape")
    return (
        mixed.reshape(batch, rays, channels, depth)
        .permute(0, 2, 3, 1)
        .contiguous()
    )


def yaw_aligned_camera_basis(
    base_quat_world_xyzw: torch.Tensor,
    stored_base_yaw_rad: torch.Tensor,
    *,
    camera_xyz_body_m: tuple[float, float, float] = CAMERA_XYZ_BODY_M,
) -> YawAlignedCameraBasis:
    """Compose the registered camera while preserving base roll and pitch."""

    if (
        not isinstance(base_quat_world_xyzw, torch.Tensor)
        or base_quat_world_xyzw.ndim != 2
        or base_quat_world_xyzw.shape[1] != 4
        or not base_quat_world_xyzw.is_floating_point()
    ):
        raise ValueError("base_quat_world_xyzw must be floating with shape (B,4)")
    if (
        not isinstance(stored_base_yaw_rad, torch.Tensor)
        or stored_base_yaw_rad.ndim != 1
        or stored_base_yaw_rad.shape[0] != base_quat_world_xyzw.shape[0]
        or not stored_base_yaw_rad.is_floating_point()
    ):
        raise ValueError("stored_base_yaw_rad must be floating with shape (B,)")
    if base_quat_world_xyzw.device != stored_base_yaw_rad.device:
        raise ValueError("quaternion and yaw tensors must share a device")
    if not bool(torch.isfinite(base_quat_world_xyzw).all().item()) or not bool(
        torch.isfinite(stored_base_yaw_rad).all().item()
    ):
        raise ValueError("quaternion and yaw tensors must be finite")

    validation_quaternion = base_quat_world_xyzw.to(dtype=torch.float64)
    validation_yaw = stored_base_yaw_rad.to(dtype=torch.float64)
    norm_squared = validation_quaternion[:, 0].square()
    norm_squared = norm_squared + validation_quaternion[:, 1].square()
    norm_squared = norm_squared + validation_quaternion[:, 2].square()
    norm_squared = norm_squared + validation_quaternion[:, 3].square()
    norm = torch.sqrt(norm_squared)
    if bool(((norm - 1.0).abs() > QUATERNION_NORM_TOLERANCE).any().item()):
        raise ValueError("base quaternion norm differs from one")
    qx, qy, qz, qw = validation_quaternion.unbind(dim=1)
    quaternion_yaw = torch.atan2(
        2.0 * (qw * qz + qx * qy),
        1.0 - 2.0 * (qy * qy + qz * qz),
    )
    yaw_difference = torch.atan2(
        torch.sin(validation_yaw - quaternion_yaw),
        torch.cos(validation_yaw - quaternion_yaw),
    )
    if bool(
        (yaw_difference.abs() > QUATERNION_YAW_TOLERANCE_RAD).any().item()
    ):
        raise ValueError("stored base yaw disagrees with the base quaternion")

    quaternion = base_quat_world_xyzw.to(dtype=torch.float32)
    stored_yaw = stored_base_yaw_rad.to(dtype=torch.float32)
    qx, qy, qz, qw = quaternion.unbind(dim=1)
    rotation = torch.stack(
        (
            1.0 - 2.0 * (qy * qy + qz * qz),
            2.0 * (qx * qy - qz * qw),
            2.0 * (qx * qz + qy * qw),
            2.0 * (qx * qy + qz * qw),
            1.0 - 2.0 * (qx * qx + qz * qz),
            2.0 * (qy * qz - qx * qw),
            2.0 * (qx * qz - qy * qw),
            2.0 * (qy * qz + qx * qw),
            1.0 - 2.0 * (qx * qx + qy * qy),
        ),
        dim=1,
    ).reshape(-1, 3, 3)
    cos_yaw = torch.cos(stored_yaw)
    sin_yaw = torch.sin(stored_yaw)
    yaw_from_world = torch.zeros_like(rotation)
    yaw_from_world[:, 0, 0] = cos_yaw
    yaw_from_world[:, 0, 1] = sin_yaw
    yaw_from_world[:, 1, 0] = -sin_yaw
    yaw_from_world[:, 1, 1] = cos_yaw
    yaw_from_world[:, 2, 2] = 1.0
    rotation_yaw_from_body = torch.bmm(yaw_from_world, rotation)
    camera_forward = rotation_yaw_from_body[:, :, 0]
    camera_left = rotation_yaw_from_body[:, :, 1]
    camera_up = rotation_yaw_from_body[:, :, 2]
    mount = quaternion.new_tensor(camera_xyz_body_m)
    camera_origin = (
        mount[0] * camera_forward
        + mount[1] * camera_left
        + mount[2] * camera_up
    )
    return YawAlignedCameraBasis(
        origin=camera_origin,
        forward=camera_forward,
        left=camera_left,
        up=camera_up,
    )


class CameraRayOrdinalPerception(nn.Module):
    """RGB and measured attitude to ray first surfaces and a physical map."""

    def __init__(
        self,
        *,
        encoder_depth: int = ENCODER_DEPTH,
        ray_height: int = RAY_HEIGHT,
        ray_width: int = RAY_WIDTH,
        depth_bin_count: int = DEPTH_BIN_COUNT,
        depth_bin_size_m: float = DEPTH_BIN_SIZE_M,
        ray_chunk_size: int | None = 512,
    ) -> None:
        super().__init__()
        self.ray_height = int(ray_height)
        self.ray_width = int(ray_width)
        self.depth_bin_count = int(depth_bin_count)
        self.depth_bin_size_m = float(depth_bin_size_m)
        self.ray_chunk_size = (
            None if ray_chunk_size is None else int(ray_chunk_size)
        )
        if self.ray_height <= 0 or self.ray_width <= 0:
            raise ValueError("ray lattice dimensions must be positive")
        if self.depth_bin_count < 2 or not (
            math.isfinite(self.depth_bin_size_m) and self.depth_bin_size_m > 0.0
        ):
            raise ValueError("at least two depth bins and a positive bin size are required")
        if self.ray_chunk_size is not None and self.ray_chunk_size <= 0:
            raise ValueError("ray_chunk_size must be positive or None")
        if CAMERA_RPY_BODY_RAD != (0.0, 0.0, 0.0):
            raise ValueError("camera-ray head requires the registered zero mount RPY")

        self.encoder = VisionEncoder(
            image_size=IMAGE_SIZE,
            patch_size=PATCH_SIZE,
            hidden_dim=ENCODER_DIM,
            depth=int(encoder_depth),
            n_heads=ENCODER_HEADS,
        )
        self.ray_upsample = nn.Sequential(
            nn.ConvTranspose2d(
                ENCODER_DIM,
                RAY_FEATURE_DIM,
                kernel_size=PATCH_SIZE,
                stride=PATCH_SIZE,
            ),
            nn.GroupNorm(4, RAY_FEATURE_DIM),
            nn.GELU(),
            nn.Conv2d(RAY_FEATURE_DIM, RAY_FEATURE_DIM, 3, padding=1),
            nn.GroupNorm(4, RAY_FEATURE_DIM),
            nn.GELU(),
        )
        self.depth_seed = nn.Conv2d(
            RAY_FEATURE_DIM,
            DEPTH_CONTEXT_DIM * self.depth_bin_count,
            kernel_size=1,
        )
        self.ray_context = FullRayRadialContext(DEPTH_CONTEXT_DIM)
        self.ray_head = nn.Conv2d(DEPTH_CONTEXT_DIM, 2, kernel_size=1)
        with torch.no_grad():
            # logit(1 / D): an approximately uniform initial hazard over depth.
            self.ray_head.bias[0] = -math.log(float(self.depth_bin_count - 1))
            self.ray_head.bias[1] = 0.0

        ray_u = (
            2.0
            * (torch.arange(self.ray_width, dtype=torch.float32) + 0.5)
            / float(self.ray_width)
            - 1.0
        )
        ray_v = (
            2.0
            * (torch.arange(self.ray_height, dtype=torch.float32) + 0.5)
            / float(self.ray_height)
            - 1.0
        )
        grid_v, grid_u = torch.meshgrid(ray_v, ray_u, indexing="ij")
        self.register_buffer(
            "native_ray_sample_grid",
            torch.stack((grid_u, grid_v), dim=-1)[None],
            persistent=True,
        )

        forward = torch.linspace(
            *CARTESIAN_FORWARD_CENTERS_M,
            CARTESIAN_SHAPE[0],
            dtype=torch.float32,
        )
        left = torch.linspace(
            *CARTESIAN_LEFT_CENTERS_M,
            CARTESIAN_SHAPE[1],
            dtype=torch.float32,
        )
        forward_grid, left_grid = torch.meshgrid(forward, left, indexing="ij")
        centers = torch.stack(
            (forward_grid.reshape(-1), left_grid.reshape(-1)),
            dim=-1,
        )
        offsets = torch.tensor(CELL_SQUARE_OFFSETS_M, dtype=torch.float32)
        anchors = torch.tensor(VERTICAL_ANCHOR_Z_BODY_M, dtype=torch.float32)
        horizontal = centers[:, None, :] + offsets[None, :, :]
        points = torch.empty(
            centers.shape[0],
            len(CELL_SQUARE_OFFSETS_M),
            len(VERTICAL_ANCHOR_Z_BODY_M),
            3,
            dtype=torch.float32,
        )
        points[..., :2] = horizontal[:, :, None, :]
        points[..., 2] = anchors[None, None, :]
        self.register_buffer(
            "registered_support_points",
            points.reshape(centers.shape[0], -1, 3),
            persistent=True,
        )
        self.register_buffer(
            "ground_support_indices",
            torch.arange(
                0,
                len(CELL_SQUARE_OFFSETS_M) * len(VERTICAL_ANCHOR_Z_BODY_M),
                len(VERTICAL_ANCHOR_Z_BODY_M),
                dtype=torch.long,
            ),
            persistent=True,
        )

    @staticmethod
    def _validate_image(image: torch.Tensor) -> None:
        if image.ndim != 4 or tuple(image.shape[1:]) != (
            3,
            IMAGE_SIZE,
            IMAGE_SIZE,
        ):
            raise ValueError(f"image must have shape (B,3,{IMAGE_SIZE},{IMAGE_SIZE})")
        if not image.is_floating_point():
            raise ValueError("image must be floating point")

    def _ray_image_features(self, image: torch.Tensor) -> torch.Tensor:
        tokens = self.encoder.forward_tokens(image)[:, 1:]
        if tokens.shape[1:] != (TOKEN_SIDE * TOKEN_SIDE, ENCODER_DIM):
            raise RuntimeError("encoder returned an unexpected patch-token grid")
        token_map = tokens.transpose(1, 2).reshape(
            image.shape[0],
            ENCODER_DIM,
            TOKEN_SIDE,
            TOKEN_SIDE,
        )
        dense = self.ray_upsample(token_map)
        if tuple(dense.shape[-2:]) != (IMAGE_SIZE, IMAGE_SIZE):
            raise RuntimeError("ray upsampler changed the registered image size")
        grid = self.native_ray_sample_grid.expand(image.shape[0], -1, -1, -1)
        return F.grid_sample(
            dense,
            grid.to(dtype=dense.dtype),
            mode="bilinear",
            padding_mode="zeros",
            align_corners=False,
        )

    def raw_ray_logits(
        self,
        image: torch.Tensor,
        *,
        ray_chunk_size: int | None = None,
    ) -> torch.Tensor:
        """Return hazard/type logits with shape (B,2,D,Hray,Wray)."""

        self._validate_image(image)
        ray_features = self._ray_image_features(image)
        seeded = self.depth_seed(ray_features)
        batch = image.shape[0]
        seeded = seeded.reshape(
            batch,
            DEPTH_CONTEXT_DIM,
            self.depth_bin_count,
            self.ray_height * self.ray_width,
        )
        chunk = self.ray_chunk_size if ray_chunk_size is None else ray_chunk_size
        mixed = apply_full_ray_context_per_ray(
            self.ray_context,
            seeded,
            ray_chunk_size=chunk,
        )
        raw = self.ray_head(mixed)
        return raw.reshape(
            batch,
            2,
            self.depth_bin_count,
            self.ray_height,
            self.ray_width,
        )

    def ray_log_probabilities(
        self,
        image: torch.Tensor,
        *,
        ray_chunk_size: int | None = None,
    ) -> OrderedRayLogProbabilities:
        raw = self.raw_ray_logits(image, ray_chunk_size=ray_chunk_size)
        return ordered_first_surface_log_probabilities(raw[:, 0], raw[:, 1])

    def _project_registered_support(
        self,
        base_quat_world_xyzw: torch.Tensor,
        stored_base_yaw_rad: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        basis = yaw_aligned_camera_basis(
            base_quat_world_xyzw,
            stored_base_yaw_rad,
        )
        points = self.registered_support_points.to(
            device=basis.origin.device,
            dtype=torch.float32,
        )
        delta = points[None] - basis.origin[:, None, None, :]
        forward = (delta * basis.forward[:, None, None, :]).sum(dim=-1)
        left = (delta * basis.left[:, None, None, :]).sum(dim=-1)
        up = (delta * basis.up[:, None, None, :]).sum(dim=-1)
        safe_forward = torch.where(
            forward.abs() > torch.finfo(torch.float32).eps,
            forward,
            torch.ones_like(forward),
        )
        tan_horizontal = math.tan(math.radians(HORIZONTAL_FOV_DEG) * 0.5)
        tan_vertical = math.tan(math.radians(VERTICAL_FOV_DEG) * 0.5)
        normalized_u = -left / (safe_forward * tan_horizontal)
        normalized_v = -up / (safe_forward * tan_vertical)
        ray_length = torch.linalg.vector_norm(delta, dim=-1)
        depth_coordinate = (
            2.0
            * (ray_length - DEPTH_NEAR_EDGE_M)
            / (self.depth_bin_count * self.depth_bin_size_m)
            - 1.0
        )
        boundary_tolerance = (
            FLOAT32_BOUNDARY_TOLERANCE_ULPS * torch.finfo(torch.float32).eps
        )
        depth_far_edge = (
            DEPTH_NEAR_EDGE_M + self.depth_bin_count * self.depth_bin_size_m
        )
        visible = (
            (forward >= CAMERA_NEAR_M - boundary_tolerance)
            & (normalized_u >= -1.0 - boundary_tolerance)
            & (normalized_u <= 1.0 + boundary_tolerance)
            & (normalized_v >= -1.0 - boundary_tolerance)
            & (normalized_v <= 1.0 + boundary_tolerance)
            & (ray_length >= DEPTH_NEAR_EDGE_M - boundary_tolerance)
            & (ray_length <= depth_far_edge + boundary_tolerance)
        )
        grid = torch.stack(
            (normalized_u, normalized_v, depth_coordinate),
            dim=-1,
        )
        grid = torch.where(visible[..., None], grid, grid.new_full((), 2.0))
        return grid, visible

    def registered_support_visibility(
        self,
        base_quat_world_xyzw: torch.Tensor,
        stored_base_yaw_rad: torch.Tensor,
    ) -> torch.Tensor:
        """Return per-cell visibility of any registered 3-D support point."""

        _grid, support_visible = self._project_registered_support(
            base_quat_world_xyzw,
            stored_base_yaw_rad,
        )
        return support_visible.any(dim=-1).reshape(
            base_quat_world_xyzw.shape[0],
            *CARTESIAN_SHAPE,
        )

    def cartesian_log_probabilities(
        self,
        ray_probabilities: OrderedRayLogProbabilities,
        base_quat_world_xyzw: torch.Tensor,
        stored_base_yaw_rad: torch.Tensor,
    ) -> torch.Tensor:
        """Gather ray first surfaces into physical-grid log probabilities."""

        obstacle = ray_probabilities.obstacle
        floor = ray_probabilities.floor
        expected = (
            base_quat_world_xyzw.shape[0],
            self.depth_bin_count,
            self.ray_height,
            self.ray_width,
        )
        if obstacle.shape != expected or floor.shape != expected:
            raise ValueError(f"ray probabilities must have shape {expected}")
        if obstacle.device != base_quat_world_xyzw.device:
            raise ValueError("ray probabilities and attitude must share a device")
        grid, support_visible = self._project_registered_support(
            base_quat_world_xyzw,
            stored_base_yaw_rad,
        )
        volume = torch.stack((obstacle.exp(), floor.exp()), dim=1)
        batch, cell_count, support_count, _xyz = grid.shape
        sampled = F.grid_sample(
            volume,
            grid.reshape(batch, cell_count * support_count, 1, 1, 3).to(
                dtype=volume.dtype
            ),
            mode="bilinear",
            padding_mode="zeros",
            align_corners=False,
        )
        sampled = sampled.reshape(batch, 2, cell_count, support_count)
        sampled = sampled * support_visible[:, None].to(dtype=sampled.dtype)
        epsilon = torch.finfo(sampled.dtype).eps

        obstacle_support = sampled[:, 0].clamp(min=0.0, max=1.0 - epsilon)
        log_no_obstacle = torch.log1p(-obstacle_support).sum(dim=-1)
        occupied_probability = -torch.expm1(log_no_obstacle)

        floor_support = sampled[:, 1].index_select(
            -1,
            self.ground_support_indices,
        )
        ground_visible = support_visible.index_select(
            -1,
            self.ground_support_indices,
        )
        floor_support = floor_support * ground_visible.to(dtype=floor_support.dtype)
        log_all_floor = torch.log(
            floor_support.clamp_min(torch.finfo(floor_support.dtype).tiny)
        ).sum(dim=-1)
        free_support_probability = torch.exp(log_all_floor)

        not_occupied = 1.0 - occupied_probability
        free_probability = not_occupied * free_support_probability
        unknown_probability = not_occupied * (1.0 - free_support_probability)
        probabilities = torch.stack(
            (unknown_probability, free_probability, occupied_probability),
            dim=1,
        ).clamp_min(torch.finfo(sampled.dtype).tiny)
        probabilities = probabilities / probabilities.sum(dim=1, keepdim=True)
        return probabilities.log().reshape(
            batch,
            CLASS_COUNT,
            *CARTESIAN_SHAPE,
        )

    def forward(
        self,
        image: torch.Tensor,
        base_quat_world_xyzw: torch.Tensor,
        stored_base_yaw_rad: torch.Tensor,
        *,
        ray_chunk_size: int | None = None,
    ) -> torch.Tensor:
        self._validate_image(image)
        if base_quat_world_xyzw.shape[0] != image.shape[0]:
            raise ValueError("attitude batch must match image batch")
        if (
            base_quat_world_xyzw.device != image.device
            or stored_base_yaw_rad.device != image.device
        ):
            raise ValueError("image and attitude tensors must share a device")
        ray_probabilities = self.ray_log_probabilities(
            image,
            ray_chunk_size=ray_chunk_size,
        )
        return self.cartesian_log_probabilities(
            ray_probabilities,
            base_quat_world_xyzw,
            stored_base_yaw_rad,
        )

    def occupancy_logits(
        self,
        image: torch.Tensor,
        base_quat_world_xyzw: torch.Tensor,
        stored_base_yaw_rad: torch.Tensor,
    ) -> torch.Tensor:
        """Compatibility entrypoint used by physical-map evaluators."""

        return self.forward(
            image,
            base_quat_world_xyzw,
            stored_base_yaw_rad,
        )


__all__ = [
    "CAMERA_NEAR_M",
    "CAMERA_RPY_BODY_RAD",
    "CAMERA_XYZ_BODY_M",
    "CARTESIAN_SHAPE",
    "CameraRayOrdinalPerception",
    "DEPTH_BIN_COUNT",
    "DEPTH_BIN_SIZE_M",
    "DEPTH_NEAR_EDGE_M",
    "DYNAMIC_CARTESIAN_OCCUPANCY_PARAMETER_COUNT",
    "OrderedRayLogProbabilities",
    "QUATERNION_NORM_TOLERANCE",
    "QUATERNION_YAW_TOLERANCE_RAD",
    "RAY_HEIGHT",
    "RAY_WIDTH",
    "REGISTERED_DECODER_PARAMETER_COUNT",
    "REGISTERED_PARAMETER_COUNT",
    "YawAlignedCameraBasis",
    "apply_full_ray_context_per_ray",
    "ordered_first_surface_log_probabilities",
    "yaw_aligned_camera_basis",
]
