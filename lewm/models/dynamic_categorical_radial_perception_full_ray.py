"""Full-attitude categorical radial perception with explicit full-ray context."""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from lewm.benchmarks.go2_categorical_radial_factorization import (
    CARTESIAN_SHAPE,
    gather_polar_logits_to_cartesian,
)

from .categorical_radial_perception import (
    CAMERA_NEAR_M,
    CAMERA_RPY_BODY_RAD,
    CAMERA_XYZ_BODY_M,
    CLASS_COUNT,
    ENCODER_DIM,
    HORIZONTAL_FOV_DEG,
    TOKEN_SIDE,
    VERTICAL_ANCHOR_Z_BODY_M,
    VERTICAL_FOV_DEG,
)
from .categorical_radial_perception_full_ray import (
    CategoricalRadialPerceptionFullRay,
)
from .egomotion_bev_jepa import (
    PROJECTIVE_FLOAT32_BOUNDARY_TOLERANCE_ULPS,
    PROJECTIVE_QUATERNION_NORM_TOLERANCE,
    PROJECTIVE_QUATERNION_YAW_TOLERANCE_RAD,
)


def _yaw_aligned_camera(
    base_quat_world_xyzw: torch.Tensor,
    stored_base_yaw_rad: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if (
        not isinstance(base_quat_world_xyzw, torch.Tensor)
        or base_quat_world_xyzw.ndim != 2
        or base_quat_world_xyzw.shape[1] != 4
        or not torch.is_floating_point(base_quat_world_xyzw)
    ):
        raise ValueError("base quaternion must be a floating tensor with shape (B,4)")
    if (
        not isinstance(stored_base_yaw_rad, torch.Tensor)
        or stored_base_yaw_rad.ndim != 1
        or stored_base_yaw_rad.shape[0] != base_quat_world_xyzw.shape[0]
        or not torch.is_floating_point(stored_base_yaw_rad)
    ):
        raise ValueError("stored yaw must be a floating tensor with shape (B,)")
    if base_quat_world_xyzw.device != stored_base_yaw_rad.device:
        raise ValueError("base quaternion and stored yaw must share a device")
    if not bool(torch.isfinite(base_quat_world_xyzw).all().item()) or not bool(
        torch.isfinite(stored_base_yaw_rad).all().item()
    ):
        raise ValueError("base quaternion and stored yaw must be finite")

    validation = base_quat_world_xyzw.to(dtype=torch.float64)
    norm_squared = validation[:, 0].square()
    norm_squared = norm_squared + validation[:, 1].square()
    norm_squared = norm_squared + validation[:, 2].square()
    norm_squared = norm_squared + validation[:, 3].square()
    if bool(
        (
            (torch.sqrt(norm_squared) - 1.0).abs()
            > PROJECTIVE_QUATERNION_NORM_TOLERANCE
        )
        .any()
        .item()
    ):
        raise ValueError("base quaternion norm differs from one")
    qx, qy, qz, qw = validation.unbind(dim=1)
    quaternion_yaw = torch.atan2(
        2.0 * (qw * qz + qx * qy),
        1.0 - 2.0 * (qy * qy + qz * qz),
    )
    yaw64 = stored_base_yaw_rad.to(dtype=torch.float64)
    difference = torch.atan2(
        torch.sin(yaw64 - quaternion_yaw), torch.cos(yaw64 - quaternion_yaw)
    )
    if bool(
        (difference.abs() > PROJECTIVE_QUATERNION_YAW_TOLERANCE_RAD).any().item()
    ):
        raise ValueError("stored yaw disagrees with base quaternion")

    quaternion = base_quat_world_xyzw.to(dtype=torch.float32)
    yaw = stored_base_yaw_rad.to(dtype=torch.float32)
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
    cos_yaw = torch.cos(yaw)
    sin_yaw = torch.sin(yaw)
    yaw_from_world = torch.zeros_like(rotation)
    yaw_from_world[:, 0, 0] = cos_yaw
    yaw_from_world[:, 0, 1] = sin_yaw
    yaw_from_world[:, 1, 0] = -sin_yaw
    yaw_from_world[:, 1, 1] = cos_yaw
    yaw_from_world[:, 2, 2] = 1.0
    yaw_from_body = torch.bmm(yaw_from_world, rotation)
    camera_forward = yaw_from_body[:, :, 0]
    camera_left = yaw_from_body[:, :, 1]
    camera_up = yaw_from_body[:, :, 2]
    mount = quaternion.new_tensor(CAMERA_XYZ_BODY_M)
    camera_origin = (
        mount[0] * camera_forward
        + mount[1] * camera_left
        + mount[2] * camera_up
    )
    return camera_origin, camera_forward, camera_left, camera_up


class DynamicCategoricalRadialPerceptionFullRay(CategoricalRadialPerceptionFullRay):
    """Direct projective sampling plus full-ray context at measured attitude."""

    def __init__(self) -> None:
        if CAMERA_RPY_BODY_RAD != (0.0, 0.0, 0.0):
            raise ValueError("dynamic radial head requires the registered zero mount RPY")
        super().__init__()
        radial = torch.tensor(
            self.factorization.radial_centers_m, dtype=torch.float32
        )
        angular = torch.tensor(
            self.factorization.angular_centers_rad, dtype=torch.float32
        )
        radius_grid, angle_grid = torch.meshgrid(radial, angular, indexing="ij")
        self.register_buffer(
            "dynamic_polar_forward_m",
            radius_grid * torch.cos(angle_grid),
            persistent=False,
        )
        self.register_buffer(
            "dynamic_polar_left_m",
            radius_grid * torch.sin(angle_grid),
            persistent=False,
        )

    def dynamic_anchor_geometry(
        self,
        base_quat_world_xyzw: torch.Tensor,
        stored_base_yaw_rad: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        origin, camera_forward, camera_left, camera_up = _yaw_aligned_camera(
            base_quat_world_xyzw, stored_base_yaw_rad
        )
        forward = self.dynamic_polar_forward_m.to(
            device=origin.device, dtype=torch.float32
        )
        left = self.dynamic_polar_left_m.to(device=origin.device, dtype=torch.float32)
        anchors = origin.new_tensor(VERTICAL_ANCHOR_Z_BODY_M)
        points = torch.stack(
            (
                forward[None].expand(anchors.numel(), -1, -1),
                left[None].expand(anchors.numel(), -1, -1),
                anchors[:, None, None].expand(-1, *forward.shape),
            ),
            dim=-1,
        )
        delta = points[None] - origin[:, None, None, None, :]
        camera_x = (delta * camera_forward[:, None, None, None, :]).sum(dim=-1)
        camera_y = (delta * camera_left[:, None, None, None, :]).sum(dim=-1)
        camera_z = (delta * camera_up[:, None, None, None, :]).sum(dim=-1)
        safe_x = torch.where(
            camera_x.abs() > torch.finfo(torch.float32).eps,
            camera_x,
            torch.ones_like(camera_x),
        )
        tan_h = math.tan(math.radians(HORIZONTAL_FOV_DEG) * 0.5)
        tan_v = math.tan(math.radians(VERTICAL_FOV_DEG) * 0.5)
        normalized_u = -camera_y / (safe_x * tan_h)
        normalized_v = -camera_z / (safe_x * tan_v)
        tolerance = (
            PROJECTIVE_FLOAT32_BOUNDARY_TOLERANCE_ULPS
            * torch.finfo(torch.float32).eps
        )
        valid = (
            (camera_x >= CAMERA_NEAR_M - tolerance)
            & (normalized_u >= -1.0 - tolerance)
            & (normalized_u <= 1.0 + tolerance)
            & (normalized_v >= -1.0 - tolerance)
            & (normalized_v <= 1.0 + tolerance)
        )
        grid = torch.stack((normalized_u, normalized_v), dim=-1)
        grid = torch.where(valid[..., None], grid, grid.new_tensor(2.0))
        return grid, valid

    def sample_dynamic_projective_anchors(
        self,
        projected_tokens: torch.Tensor,
        base_quat_world_xyzw: torch.Tensor,
        stored_base_yaw_rad: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if tuple(projected_tokens.shape[1:]) != (
            self.token_feature_dim,
            TOKEN_SIDE,
            TOKEN_SIDE,
        ):
            raise ValueError("projected token map shape changed")
        if base_quat_world_xyzw.shape[0] != projected_tokens.shape[0]:
            raise ValueError("attitude batch does not match projected tokens")
        grid, validity = self.dynamic_anchor_geometry(
            base_quat_world_xyzw, stored_base_yaw_rad
        )
        samples = []
        for anchor_index in range(len(VERTICAL_ANCHOR_Z_BODY_M)):
            sampled = F.grid_sample(
                projected_tokens,
                grid[:, anchor_index].to(dtype=projected_tokens.dtype),
                mode="bilinear",
                padding_mode="zeros",
                align_corners=False,
            )
            samples.append(
                sampled
                * validity[:, anchor_index, None].to(dtype=sampled.dtype)
            )
        return torch.stack(samples, dim=1), validity

    def polar_logits(
        self,
        image: torch.Tensor,
        base_quat_world_xyzw: torch.Tensor,
        stored_base_yaw_rad: torch.Tensor,
    ) -> torch.Tensor:
        self._validate_image(image)
        projected = self._projected_token_map(image)
        samples, validity = self.sample_dynamic_projective_anchors(
            projected, base_quat_world_xyzw, stored_base_yaw_rad
        )
        batch, anchor_count, _channels, radial_count, angular_count = samples.shape
        identity = self.anchor_identity[None, :, :, None, None].expand(
            batch, -1, -1, radial_count, angular_count
        )
        validity_channel = validity[:, :, None]
        anchored = torch.cat(
            (
                samples,
                identity.to(dtype=samples.dtype),
                validity_channel.to(dtype=samples.dtype),
            ),
            dim=2,
        ).reshape(batch, -1, radial_count, angular_count)
        coordinates = self.polar_coordinate_features[None].expand(batch, -1, -1, -1)
        features = self.context_stem(
            torch.cat((anchored, coordinates.to(dtype=anchored.dtype)), dim=1)
        )
        features = self.radial_context(features)
        features = self.angular_context(features)
        return self.polar_head(features)

    def forward(
        self,
        image: torch.Tensor,
        base_quat_world_xyzw: torch.Tensor,
        stored_base_yaw_rad: torch.Tensor,
    ) -> torch.Tensor:
        polar = self.polar_logits(
            image, base_quat_world_xyzw, stored_base_yaw_rad
        )
        cartesian = gather_polar_logits_to_cartesian(
            polar, factorization=self.factorization
        )
        if tuple(cartesian.shape[-3:]) != (CLASS_COUNT, *CARTESIAN_SHAPE):
            raise RuntimeError("dynamic radial factorization returned a wrong shape")
        return cartesian

    def occupancy_logits(
        self,
        image: torch.Tensor,
        base_quat_world_xyzw: torch.Tensor,
        stored_base_yaw_rad: torch.Tensor,
    ) -> torch.Tensor:
        return self.forward(image, base_quat_world_xyzw, stored_base_yaw_rad)


__all__ = ["DynamicCategoricalRadialPerceptionFullRay", "_yaw_aligned_camera"]
