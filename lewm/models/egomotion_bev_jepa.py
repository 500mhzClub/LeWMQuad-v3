"""Egomotion-aligned spatial JEPA with an auxiliary traversability head."""
from __future__ import annotations

import copy
import hashlib
import json
import math
import operator
from typing import Any, Mapping

import torch
import torch.nn as nn
import torch.nn.functional as F

from .encoders import VisionEncoder


UNKNOWN_CLASS = 0
FREE_CLASS = 1
OCCUPIED_CLASS = 2

GLOBAL_CROSS_ATTENTION_LIFT = "global_cross_attention_v1"
PROJECTIVE_COLUMN_ATTENTION_LIFT = "projective_column_attention_v1"
PROJECTIVE_FOOTPRINT_ATTENTION_LIFT = "projective_footprint_attention_v1"
PROJECTIVE_CELL_SQUARE_ATTENTION_LIFT = "projective_cell_square_attention_v1"
DYNAMIC_PROJECTIVE_CELL_SQUARE_ATTENTION_LIFT = (
    "dynamic_projective_cell_square_attention_v1"
)
PROJECTIVE_QUERY_SUPPORT_SCHEMA = "lewm_projective_query_support_v1"
PROJECTIVE_CELL_SQUARE_OUTPUT_CELL_SIZE_M = 0.10
PROJECTIVE_QUATERNION_NORM_TOLERANCE = 1e-5
PROJECTIVE_QUATERNION_YAW_TOLERANCE_RAD = 1e-5
PROJECTIVE_FLOAT32_BOUNDARY_TOLERANCE_ULPS = 8.0
PROJECTIVE_CELL_SQUARE_BIAS_AGGREGATION = (
    "minimum_normalized_image_token_distance_over_output_cell_support_"
    "and_vertical_anchors_v1"
)
PROJECTIVE_CELL_SQUARE_VISIBILITY_AGGREGATION = (
    "any_output_cell_support_and_vertical_anchor_visible_v1"
)


def bev_variance_floor_loss(
    features: torch.Tensor,
    *,
    target_std: float = 1.0,
    eps: float = 1e-4,
) -> torch.Tensor:
    """Penalize input-independent BEV features at every channel/cell."""

    if features.ndim != 4:
        raise ValueError("features must have shape (B, D, H, W)")
    if features.shape[0] < 2:
        raise ValueError("variance loss requires at least two BEV samples")
    std = torch.sqrt(features.float().var(dim=0, unbiased=False) + float(eps))
    return torch.relu(float(target_std) - std).mean().to(features.dtype)


def _coordinate_axis(
    minimum: float,
    maximum: float,
    cells: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    if cells < 2:
        return torch.tensor([(minimum + maximum) * 0.5], device=device, dtype=dtype)
    return torch.linspace(float(minimum), float(maximum), cells, device=device, dtype=dtype)


def _masked_spatial_mean(
    values: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    if values.shape != mask.shape:
        raise ValueError("values and mask must have the same spatial shape")
    weight = mask.to(values.dtype)
    return (values * weight).sum() / weight.sum().clamp_min(1.0)


def _normalized_spatial_error(
    prediction: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    return (
        F.normalize(prediction, dim=1) - F.normalize(target, dim=1)
    ).square().mean(dim=1)


def _validate_bool_mask(
    mask: torch.Tensor | None,
    expected_shape: tuple[int, ...],
    *,
    name: str,
) -> torch.Tensor | None:
    if mask is None:
        return None
    if tuple(mask.shape) != expected_shape:
        raise ValueError(f"{name} must have shape {expected_shape}")
    if mask.dtype is not torch.bool:
        raise ValueError(f"{name} must use boolean dtype")
    return mask


def _validate_class_weights(
    weights: torch.Tensor | None,
    *,
    classes: int,
    name: str,
    reference: torch.Tensor,
) -> torch.Tensor | None:
    if weights is None:
        return None
    if weights.shape != (classes,):
        raise ValueError(f"{name} must have shape ({classes},)")
    if not torch.is_floating_point(weights):
        raise ValueError(f"{name} must be floating point")
    if not bool(torch.isfinite(weights).all().item()):
        raise ValueError(f"{name} must be finite")
    if bool((weights < 0).any().item()) or not bool((weights.sum() > 0).item()):
        raise ValueError(f"{name} must be nonnegative with positive sum")
    return weights.to(device=reference.device, dtype=reference.dtype)


def _weighted_cross_entropy_mean(
    logits: torch.Tensor,
    labels: torch.Tensor,
    mask: torch.Tensor,
    class_weights: torch.Tensor | None,
) -> torch.Tensor:
    """Reduce CE by the sum of weights actually applied to valid cells."""

    loss = F.cross_entropy(logits, labels, reduction="none")
    applied_weights = torch.ones_like(loss)
    if class_weights is not None:
        applied_weights = class_weights[labels]
    applied_weights = applied_weights * mask.to(loss.dtype)
    denominator = applied_weights.sum()
    return (loss * applied_weights).sum() / denominator.clamp_min(
        torch.finfo(loss.dtype).tiny
    )


def _finite_projection_value(value: Any, *, name: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a finite number")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite number") from exc
    if not math.isfinite(result):
        raise ValueError(f"{name} must be a finite number")
    return result


def _projection_triple(value: Any, *, name: str) -> tuple[float, float, float]:
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        raise ValueError(f"{name} must contain exactly three values")
    return tuple(
        _finite_projection_value(item, name=f"{name}[{index}]")
        for index, item in enumerate(value)
    )


def _projection_anchors(value: Any) -> tuple[float, ...]:
    if not isinstance(value, (list, tuple)) or not value:
        raise ValueError(
            "projective_vertical_anchor_z_body_m must be a non-empty sequence"
        )
    anchors = tuple(
        _finite_projection_value(
            item,
            name=f"projective_vertical_anchor_z_body_m[{index}]",
        )
        for index, item in enumerate(value)
    )
    if any(next_value <= value for value, next_value in zip(anchors, anchors[1:])):
        raise ValueError(
            "projective_vertical_anchor_z_body_m must be strictly increasing"
        )
    return anchors


def _projection_integer(value: Any, *, name: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be an integer")
    try:
        return int(operator.index(value))
    except TypeError as exc:
        raise ValueError(f"{name} must be an integer") from exc


def _footprint_horizontal_offsets(
    radius_m: float,
    perimeter_samples: int,
) -> tuple[tuple[float, float], ...]:
    """Return center then a deterministic, cardinal-aligned footprint ring."""

    offsets = [(0.0, 0.0)]
    for index in range(perimeter_samples):
        angle = 2.0 * math.pi * float(index) / float(perimeter_samples)
        offsets.append((radius_m * math.cos(angle), radius_m * math.sin(angle)))
    return tuple(offsets)


def _cell_square_horizontal_offsets(
    output_cell_size_m: float,
) -> tuple[tuple[float, float], ...]:
    """Return center then lexicographically ordered output-cell corners."""

    cell_size = _finite_projection_value(
        output_cell_size_m,
        name="projective_output_cell_size_m",
    )
    if cell_size <= 0.0:
        raise ValueError("projective_output_cell_size_m must be positive")
    if not math.isclose(
        cell_size,
        PROJECTIVE_CELL_SQUARE_OUTPUT_CELL_SIZE_M,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise ValueError(
            "projective_output_cell_size_m differs from registered 0.10 m support"
        )
    half = 0.5 * cell_size
    return (
        (0.0, 0.0),
        (-half, -half),
        (-half, half),
        (half, -half),
        (half, half),
    )


def _canonical_json_sha256(payload: Any) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def build_projective_query_support_contract(
    dataset_manifest: Mapping[str, Any],
    *,
    lift_type: str = PROJECTIVE_CELL_SQUARE_ATTENTION_LIFT,
) -> dict[str, Any]:
    """Derive cell-square query support only from a physical-v3 manifest."""

    lift_type = str(lift_type)
    if lift_type not in (
        PROJECTIVE_CELL_SQUARE_ATTENTION_LIFT,
        DYNAMIC_PROJECTIVE_CELL_SQUARE_ATTENTION_LIFT,
    ):
        raise ValueError("query support requires a cell-square lift type")

    if dataset_manifest.get("schema") != "lewm_go2_paired_navigation_dataset_v3":
        raise ValueError("cell-square query support requires physical dataset v3")
    semantics = dataset_manifest.get("label_semantics")
    grid = dataset_manifest.get("local_grid")
    if not isinstance(semantics, Mapping) or not isinstance(grid, Mapping):
        raise ValueError("physical dataset lacks label or local-grid semantics")
    if (
        semantics.get("label_contract") != "observable_physical_occupancy_v3"
        or semantics.get("target_occupancy_space")
        != "observable_physical_occupancy"
        or semantics.get("per_frame_configuration_classes_supervised") is not False
    ):
        raise ValueError("cell-square query support requires physical occupancy labels")
    aggregation = semantics.get("physical_aggregation")
    if not isinstance(aggregation, Mapping):
        raise ValueError("physical dataset lacks its aggregation contract")
    aggregation_core = dict(aggregation)
    aggregation_sha256 = str(aggregation_core.pop("contract_sha256", ""))
    if (
        aggregation.get("schema") != "lewm_observable_physical_aggregation_v1"
        or _canonical_json_sha256(aggregation_core) != aggregation_sha256
    ):
        raise ValueError("physical aggregation contract hash mismatch")
    output_cell_size_m = _finite_projection_value(
        grid.get("cell_size_m"), name="local_grid.cell_size_m"
    )
    aggregation_cell_size_m = _finite_projection_value(
        aggregation.get("output_cell_size_m"),
        name="physical_aggregation.output_cell_size_m",
    )
    if not math.isclose(
        output_cell_size_m,
        aggregation_cell_size_m,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise ValueError("local-grid and physical-aggregation cell sizes differ")
    offsets = _cell_square_horizontal_offsets(output_cell_size_m)
    core = {
        "schema": PROJECTIVE_QUERY_SUPPORT_SCHEMA,
        "lift_type": lift_type,
        "support_geometry": "output_cell_center_plus_four_corners_v1",
        "support_frame": "base_forward_left_offsets_from_output_cell_center",
        "output_cell_size_m": output_cell_size_m,
        "output_cell_half_extent_m": 0.5 * output_cell_size_m,
        "horizontal_offsets_body_m": [list(value) for value in offsets],
        "support_point_count": len(offsets),
        "uses_body_footprint": False,
        "attention_bias_aggregation": (
            PROJECTIVE_CELL_SQUARE_BIAS_AGGREGATION
        ),
        "query_visibility_aggregation": (
            PROJECTIVE_CELL_SQUARE_VISIBILITY_AGGREGATION
        ),
        "physical_aggregation_contract": {
            "schema": str(aggregation["schema"]),
            "contract_sha256": aggregation_sha256,
            "output_cell_size_m": aggregation_cell_size_m,
        },
    }
    return {**core, "contract_sha256": _canonical_json_sha256(core)}


def validate_projective_query_support_binding(
    *,
    model_config: Mapping[str, Any],
    projective_query_support: Mapping[str, Any] | None,
    dataset_manifest: Mapping[str, Any],
    occupancy_output_contract: Mapping[str, Any] | None = None,
) -> dict[str, Any] | None:
    """Validate new-lift support while preserving old checkpoint contracts."""

    lift_type = str(
        model_config.get("bev_lift_type", GLOBAL_CROSS_ATTENTION_LIFT)
    )
    if lift_type not in (
        PROJECTIVE_CELL_SQUARE_ATTENTION_LIFT,
        DYNAMIC_PROJECTIVE_CELL_SQUARE_ATTENTION_LIFT,
    ):
        if projective_query_support is not None:
            raise ValueError("projective query support is invalid for this lift type")
        if "projective_output_cell_size_m" in model_config:
            raise ValueError("output-cell support size is invalid for this lift type")
        if occupancy_output_contract is not None and occupancy_output_contract.get(
            "projective_query_support_contract_sha256"
        ) is not None:
            raise ValueError("occupancy output binds support for the wrong lift type")
        return None
    if not isinstance(projective_query_support, Mapping):
        raise ValueError("cell-square lift lacks projective query support")
    expected = build_projective_query_support_contract(
        dataset_manifest,
        lift_type=lift_type,
    )
    if dict(projective_query_support) != expected:
        raise ValueError("projective query support differs from the dataset contract")
    configured_size = _finite_projection_value(
        model_config.get("projective_output_cell_size_m"),
        name="model_config.projective_output_cell_size_m",
    )
    if not math.isclose(
        configured_size,
        float(expected["output_cell_size_m"]),
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise ValueError("model output-cell support size differs from its contract")
    for name in (
        "projective_footprint_radius_m",
        "projective_footprint_perimeter_samples",
    ):
        if model_config.get(name) is not None:
            raise ValueError("cell-square support must not use body-footprint arguments")
    if occupancy_output_contract is not None and occupancy_output_contract.get(
        "projective_query_support_contract_sha256"
    ) != expected["contract_sha256"]:
        raise ValueError("occupancy output is not bound to projective query support")
    return expected


def _rpy_body_from_camera(
    rpy_rad: tuple[float, float, float],
) -> torch.Tensor:
    """Return ``R_body_from_camera = Rz(yaw) @ Ry(pitch) @ Rx(roll)``."""

    roll, pitch, yaw = rpy_rad
    cos_roll, sin_roll = math.cos(roll), math.sin(roll)
    cos_pitch, sin_pitch = math.cos(pitch), math.sin(pitch)
    cos_yaw, sin_yaw = math.cos(yaw), math.sin(yaw)
    return torch.tensor(
        (
            (
                cos_yaw * cos_pitch,
                cos_yaw * sin_pitch * sin_roll - sin_yaw * cos_roll,
                cos_yaw * sin_pitch * cos_roll + sin_yaw * sin_roll,
            ),
            (
                sin_yaw * cos_pitch,
                sin_yaw * sin_pitch * sin_roll + cos_yaw * cos_roll,
                sin_yaw * sin_pitch * cos_roll - cos_yaw * sin_roll,
            ),
            (-sin_pitch, cos_pitch * sin_roll, cos_pitch * cos_roll),
        ),
        dtype=torch.float64,
    )


def _projective_column_attention_geometry(
    *,
    metric_forward_grid: torch.Tensor,
    metric_left_grid: torch.Tensor,
    token_side: int,
    horizontal_fov_deg: float,
    vertical_fov_deg: float,
    camera_xyz_body_m: tuple[float, float, float],
    camera_rpy_body_rad: tuple[float, float, float],
    near_m: float,
    vertical_anchor_z_body_m: tuple[float, ...],
    horizontal_offsets_body_m: tuple[tuple[float, float], ...],
    sigma_tokens: float,
    bias_floor: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build a fixed projective token prior for body-frame BEV queries.

    The square input represents the full camera frame. Normalized image x/y span
    ``[-1, 1]`` from left/right and top/bottom respectively. Camera axes are
    ``x_forward_y_left_z_up``; BEV rows increase forward and columns increase left.
    Each query attends near the union of its vertical columns at the supplied
    horizontal support offsets. The reduction is streamed over offsets so the
    temporary tensor shape remains ``(queries, anchors, image_tokens)``.
    """

    query_forward = metric_forward_grid.reshape(-1).to(dtype=torch.float64)
    query_left = metric_left_grid.reshape(-1).to(dtype=torch.float64)
    anchors = torch.tensor(vertical_anchor_z_body_m, dtype=torch.float64)
    query_count = int(query_forward.numel())
    anchor_count = int(anchors.numel())
    camera_origin = torch.tensor(camera_xyz_body_m, dtype=torch.float64)
    rotation_body_from_camera = _rpy_body_from_camera(camera_rpy_body_rad)
    tan_horizontal = math.tan(math.radians(horizontal_fov_deg) * 0.5)
    tan_vertical = math.tan(math.radians(vertical_fov_deg) * 0.5)
    boundary_tolerance = 32.0 * torch.finfo(torch.float64).eps

    token_axis = (
        2.0
        * (torch.arange(token_side, dtype=torch.float64) + 0.5)
        / float(token_side)
        - 1.0
    )
    token_v, token_u = torch.meshgrid(token_axis, token_axis, indexing="ij")
    token_centers = torch.stack((token_u.reshape(-1), token_v.reshape(-1)), dim=-1)
    token_width = 2.0 / float(token_side)
    nearest_distance_squared = torch.full(
        (query_count, token_side * token_side),
        float("inf"),
        dtype=torch.float64,
    )
    query_visible = torch.zeros(query_count, dtype=torch.bool)
    center_only = horizontal_offsets_body_m == ((0.0, 0.0),)
    for offset_forward, offset_left in horizontal_offsets_body_m:
        support_forward = query_forward[:, None].expand(query_count, anchor_count)
        support_left = query_left[:, None].expand(query_count, anchor_count)
        if not center_only:
            support_forward = support_forward + float(offset_forward)
            support_left = support_left + float(offset_left)
        points_body = torch.stack(
            (
                support_forward,
                support_left,
                anchors[None].expand(query_count, anchor_count),
            ),
            dim=-1,
        )
        # Row-vector form of p_camera = R_body_from_camera.T @ (p_body - t).
        points_camera = (points_body - camera_origin) @ rotation_body_from_camera
        camera_forward = points_camera[..., 0]
        safe_forward = torch.where(
            camera_forward.abs() > torch.finfo(torch.float64).eps,
            camera_forward,
            torch.ones_like(camera_forward),
        )
        normalized_u = -points_camera[..., 1] / (safe_forward * tan_horizontal)
        normalized_v = -points_camera[..., 2] / (safe_forward * tan_vertical)
        anchor_visible = (
            (camera_forward >= near_m - boundary_tolerance)
            & (normalized_u >= -1.0 - boundary_tolerance)
            & (normalized_u <= 1.0 + boundary_tolerance)
            & (normalized_v >= -1.0 - boundary_tolerance)
            & (normalized_v <= 1.0 + boundary_tolerance)
        )
        query_visible |= anchor_visible.any(dim=1)
        projected = torch.stack((normalized_u, normalized_v), dim=-1)
        difference_tokens = (
            projected[:, :, None, :] - token_centers[None, None, :, :]
        ) / token_width
        distance_squared = difference_tokens.square().sum(dim=-1)
        distance_squared = distance_squared.masked_fill(
            ~anchor_visible[:, :, None],
            float("inf"),
        )
        support_nearest_distance_squared = distance_squared.amin(dim=1)
        if center_only:
            nearest_distance_squared = support_nearest_distance_squared
        else:
            nearest_distance_squared = torch.minimum(
                nearest_distance_squared,
                support_nearest_distance_squared,
            )
    attention_bias = -0.5 * nearest_distance_squared / (sigma_tokens * sigma_tokens)
    attention_bias = attention_bias.clamp(min=bias_floor, max=0.0)
    attention_bias = torch.where(
        query_visible[:, None],
        attention_bias,
        torch.zeros_like(attention_bias),
    )
    return attention_bias.to(dtype=torch.float32), query_visible


def _dynamic_projective_cell_square_attention_geometry(
    *,
    metric_forward_grid: torch.Tensor,
    metric_left_grid: torch.Tensor,
    token_side: int,
    horizontal_fov_deg: float,
    vertical_fov_deg: float,
    camera_xyz_body_m: tuple[float, float, float],
    camera_rpy_body_rad: tuple[float, float, float],
    near_m: float,
    vertical_anchor_z_body_m: tuple[float, ...],
    horizontal_offsets_body_m: tuple[tuple[float, float], ...],
    sigma_tokens: float,
    bias_floor: float,
    base_quat_world_xyzw: torch.Tensor,
    stored_base_yaw_rad: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build per-frame projective priors from deployment-valid base attitude."""

    if camera_rpy_body_rad != (0.0, 0.0, 0.0):
        raise ValueError("dynamic cell-square lift requires the registered zero camera RPY")
    if horizontal_offsets_body_m != _cell_square_horizontal_offsets(
        PROJECTIVE_CELL_SQUARE_OUTPUT_CELL_SIZE_M
    ):
        raise ValueError("dynamic lift requires registered cell-square support")
    if (
        not isinstance(base_quat_world_xyzw, torch.Tensor)
        or base_quat_world_xyzw.ndim != 2
        or base_quat_world_xyzw.shape[1] != 4
        or not torch.is_floating_point(base_quat_world_xyzw)
    ):
        raise ValueError("base_quat_world_xyzw must be a floating tensor with shape (B, 4)")
    if (
        not isinstance(stored_base_yaw_rad, torch.Tensor)
        or stored_base_yaw_rad.ndim != 1
        or stored_base_yaw_rad.shape[0] != base_quat_world_xyzw.shape[0]
        or not torch.is_floating_point(stored_base_yaw_rad)
    ):
        raise ValueError("stored_base_yaw_rad must be a floating tensor with shape (B,)")
    if base_quat_world_xyzw.device != stored_base_yaw_rad.device:
        raise ValueError("quaternion and yaw tensors must share a device")
    if not bool(torch.isfinite(base_quat_world_xyzw).all().item()) or not bool(
        torch.isfinite(stored_base_yaw_rad).all().item()
    ):
        raise ValueError("quaternion and yaw tensors must be finite")

    # Validate the source values before the bounded float32 geometry cast.
    validation_quaternion = base_quat_world_xyzw.to(dtype=torch.float64)
    validation_yaw = stored_base_yaw_rad.to(dtype=torch.float64)
    norm_squared = validation_quaternion[:, 0].square()
    norm_squared = norm_squared + validation_quaternion[:, 1].square()
    norm_squared = norm_squared + validation_quaternion[:, 2].square()
    norm_squared = norm_squared + validation_quaternion[:, 3].square()
    norm = torch.sqrt(norm_squared)
    if bool(
        ((norm - 1.0).abs() > PROJECTIVE_QUATERNION_NORM_TOLERANCE).any().item()
    ):
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
        (yaw_difference.abs() > PROJECTIVE_QUATERNION_YAW_TOLERANCE_RAD)
        .any()
        .item()
    ):
        raise ValueError("stored base yaw disagrees with the base quaternion")

    # Geometry is parameter-free and evaluated in float32 on the selected device.
    quaternion = base_quat_world_xyzw.to(dtype=torch.float32)
    stored_yaw = stored_base_yaw_rad.to(dtype=torch.float32)
    qx, qy, qz, qw = quaternion.unbind(dim=1)

    # Standard raw-quaternion R_world_from_body, without renormalization.
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

    query_forward = metric_forward_grid.reshape(-1).to(
        device=quaternion.device, dtype=torch.float32
    )
    query_left = metric_left_grid.reshape(-1).to(
        device=quaternion.device, dtype=torch.float32
    )
    anchors = quaternion.new_tensor(vertical_anchor_z_body_m)
    token_axis = (
        2.0
        * (torch.arange(token_side, device=quaternion.device, dtype=torch.float32) + 0.5)
        / float(token_side)
        - 1.0
    )
    token_v, token_u = torch.meshgrid(token_axis, token_axis, indexing="ij")
    token_centers = torch.stack((token_u.reshape(-1), token_v.reshape(-1)), dim=-1)
    token_width = 2.0 / float(token_side)
    batch = quaternion.shape[0]
    query_count = int(query_forward.numel())
    token_count = token_side * token_side
    nearest_distance_squared = torch.full(
        (batch, query_count, token_count),
        float("inf"),
        device=quaternion.device,
        dtype=torch.float32,
    )
    query_visible = torch.zeros(
        (batch, query_count), device=quaternion.device, dtype=torch.bool
    )
    tan_horizontal = math.tan(math.radians(horizontal_fov_deg) * 0.5)
    tan_vertical = math.tan(math.radians(vertical_fov_deg) * 0.5)
    boundary_tolerance = (
        PROJECTIVE_FLOAT32_BOUNDARY_TOLERANCE_ULPS
        * torch.finfo(torch.float32).eps
    )

    for offset_forward, offset_left in horizontal_offsets_body_m:
        support_forward = (
            query_forward[:, None].expand(query_count, anchors.numel())
            + float(offset_forward)
        )
        support_left = (
            query_left[:, None].expand(query_count, anchors.numel())
            + float(offset_left)
        )
        points = torch.stack(
            (
                support_forward,
                support_left,
                anchors[None, :].expand(query_count, -1),
            ),
            dim=-1,
        )
        delta = points[None, :, :, :] - camera_origin[:, None, None, :]
        forward = (delta * camera_forward[:, None, None, :]).sum(dim=-1)
        left = (delta * camera_left[:, None, None, :]).sum(dim=-1)
        up = (delta * camera_up[:, None, None, :]).sum(dim=-1)
        safe_forward = torch.where(
            forward.abs() > torch.finfo(torch.float32).eps,
            forward,
            torch.ones_like(forward),
        )
        normalized_u = -left / (safe_forward * tan_horizontal)
        normalized_v = -up / (safe_forward * tan_vertical)
        visible = (
            (forward >= float(near_m) - boundary_tolerance)
            & (normalized_u >= -1.0 - boundary_tolerance)
            & (normalized_u <= 1.0 + boundary_tolerance)
            & (normalized_v >= -1.0 - boundary_tolerance)
            & (normalized_v <= 1.0 + boundary_tolerance)
        )
        query_visible |= visible.any(dim=2)
        projected = torch.stack((normalized_u, normalized_v), dim=-1)
        difference = (
            projected[:, :, :, None, :] - token_centers[None, None, None, :, :]
        ) / token_width
        distance_squared = difference.square().sum(dim=-1).masked_fill(
            ~visible[:, :, :, None], float("inf")
        )
        nearest_distance_squared = torch.minimum(
            nearest_distance_squared,
            distance_squared.amin(dim=2),
        )

    attention_bias = -0.5 * nearest_distance_squared / float(sigma_tokens**2)
    attention_bias = attention_bias.clamp(min=float(bias_floor), max=0.0)
    attention_bias = torch.where(
        query_visible[:, :, None], attention_bias, torch.zeros_like(attention_bias)
    )
    return attention_bias, query_visible


def warp_bev_current_to_next(
    current: torch.Tensor,
    delta_pose_current: torch.Tensor,
    *,
    forward_range_m: tuple[float, float],
    left_range_m: tuple[float, float],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Warp a current ego BEV into the next base frame using relative SE(2).

    ``delta_pose_current`` is ``(dx_forward, dy_left, dyaw)`` for the next base
    expressed in the current base frame. Rows are forward and columns are left.
    The returned mask marks next-frame cells whose source coordinates lie in the
    represented current grid.
    """

    if current.ndim != 4:
        raise ValueError("current must have shape (B, D, H, W)")
    if delta_pose_current.ndim != 2 or delta_pose_current.shape != (
        current.shape[0],
        3,
    ):
        raise ValueError("delta_pose_current must have shape (B, 3)")
    forward_min, forward_max = map(float, forward_range_m)
    left_min, left_max = map(float, left_range_m)
    if not forward_max > forward_min or not left_max > left_min:
        raise ValueError("BEV coordinate ranges must be increasing")

    batch, _channels, height, width = current.shape
    forward = _coordinate_axis(
        forward_min,
        forward_max,
        height,
        device=current.device,
        dtype=current.dtype,
    )
    left = _coordinate_axis(
        left_min,
        left_max,
        width,
        device=current.device,
        dtype=current.dtype,
    )
    next_forward, next_left = torch.meshgrid(forward, left, indexing="ij")
    next_forward = next_forward[None].expand(batch, -1, -1)
    next_left = next_left[None].expand(batch, -1, -1)

    dx = delta_pose_current[:, 0, None, None]
    dy = delta_pose_current[:, 1, None, None]
    yaw = delta_pose_current[:, 2, None, None]
    cos_yaw = torch.cos(yaw)
    sin_yaw = torch.sin(yaw)
    source_forward = dx + cos_yaw * next_forward - sin_yaw * next_left
    source_left = dy + sin_yaw * next_forward + cos_yaw * next_left

    grid_x = 2.0 * (source_left - left_min) / (left_max - left_min) - 1.0
    grid_y = 2.0 * (source_forward - forward_min) / (
        forward_max - forward_min
    ) - 1.0
    grid = torch.stack((grid_x, grid_y), dim=-1)
    overlap = (
        (grid_x >= -1.0)
        & (grid_x <= 1.0)
        & (grid_y >= -1.0)
        & (grid_y <= 1.0)
    )
    warped = F.grid_sample(
        current,
        grid,
        mode="bilinear",
        padding_mode="zeros",
        align_corners=True,
    )
    return warped, overlap[:, None]


class BevDecoder(nn.Module):
    """Lift image tokens into fixed-calibration metric BEV queries."""

    def __init__(
        self,
        *,
        token_dim: int,
        bev_dim: int,
        token_side: int,
        bev_size: tuple[int, int],
        forward_range_m: tuple[float, float],
        left_range_m: tuple[float, float],
        attention_heads: int,
        lift_type: str = GLOBAL_CROSS_ATTENTION_LIFT,
        projective_horizontal_fov_deg: float | None = None,
        projective_vertical_fov_deg: float | None = None,
        projective_camera_xyz_body_m: tuple[float, float, float] | None = None,
        projective_camera_rpy_body_rad: tuple[float, float, float] | None = None,
        projective_near_m: float | None = None,
        projective_vertical_anchor_z_body_m: tuple[float, ...] | None = None,
        projective_output_cell_size_m: float | None = None,
        projective_footprint_radius_m: float | None = None,
        projective_footprint_perimeter_samples: int | None = None,
        projective_attention_sigma_tokens: float = 1.0,
        projective_attention_bias_floor: float = -6.0,
    ) -> None:
        super().__init__()
        self.token_side = int(token_side)
        self.bev_size = (int(bev_size[0]), int(bev_size[1]))
        self.forward_range_m = tuple(map(float, forward_range_m))
        self.left_range_m = tuple(map(float, left_range_m))
        self.lift_type = str(lift_type)
        self.attention_heads = int(attention_heads)
        if not self.forward_range_m[1] > self.forward_range_m[0]:
            raise ValueError("forward_range_m must be increasing")
        if not self.left_range_m[1] > self.left_range_m[0]:
            raise ValueError("left_range_m must be increasing")
        if int(attention_heads) <= 0:
            raise ValueError("attention_heads must be positive")
        if int(bev_dim) % int(attention_heads) != 0:
            raise ValueError("bev_dim must be divisible by attention_heads")
        forward = torch.linspace(*self.forward_range_m, self.bev_size[0])
        left = torch.linspace(*self.left_range_m, self.bev_size[1])
        forward_grid, left_grid = torch.meshgrid(forward, left, indexing="ij")
        forward_grid = forward_grid / max(abs(value) for value in self.forward_range_m)
        left_grid = left_grid / max(abs(value) for value in self.left_range_m)
        coordinate_features = torch.stack(
            (
                forward_grid,
                left_grid,
                torch.sin(math.pi * forward_grid),
                torch.cos(math.pi * forward_grid),
                torch.sin(math.pi * left_grid),
                torch.cos(math.pi * left_grid),
            ),
            dim=-1,
        ).reshape(-1, 6)
        self.register_buffer("coordinate_features", coordinate_features)
        self.coordinate_query = nn.Sequential(
            nn.Linear(6, int(bev_dim)),
            nn.GELU(),
            nn.Linear(int(bev_dim), int(bev_dim)),
        )
        self.query_bias = nn.Parameter(
            torch.empty(self.bev_size[0] * self.bev_size[1], int(bev_dim))
        )
        nn.init.trunc_normal_(self.query_bias, std=0.02)
        self.token_project = nn.Linear(int(token_dim), int(bev_dim))
        self.cross_attention = nn.MultiheadAttention(
            int(bev_dim),
            int(attention_heads),
            batch_first=True,
        )
        self.query_norm = nn.LayerNorm(int(bev_dim))
        self.refine = nn.Sequential(
            nn.Conv2d(int(bev_dim), int(bev_dim), 3, padding=1),
            nn.GELU(),
            nn.Conv2d(int(bev_dim), int(bev_dim), 3, padding=1),
            nn.GroupNorm(1, int(bev_dim)),
        )
        projective_values = {
            "projective_horizontal_fov_deg": projective_horizontal_fov_deg,
            "projective_vertical_fov_deg": projective_vertical_fov_deg,
            "projective_camera_xyz_body_m": projective_camera_xyz_body_m,
            "projective_camera_rpy_body_rad": projective_camera_rpy_body_rad,
            "projective_near_m": projective_near_m,
            "projective_vertical_anchor_z_body_m": (
                projective_vertical_anchor_z_body_m
            ),
        }
        footprint_values = {
            "projective_footprint_radius_m": projective_footprint_radius_m,
            "projective_footprint_perimeter_samples": (
                projective_footprint_perimeter_samples
            ),
        }
        cell_square_values = {
            "projective_output_cell_size_m": projective_output_cell_size_m,
        }
        self.projective_output_cell_size_m: float | None = None
        self.projective_footprint_radius_m: float | None = None
        self.projective_footprint_perimeter_samples: int | None = None
        self.projective_horizontal_offsets_body_m: tuple[
            tuple[float, float], ...
        ] = ()
        self.projective_horizontal_fov_deg: float | None = None
        self.projective_vertical_fov_deg: float | None = None
        self.projective_camera_xyz_body_m: tuple[float, float, float] | None = None
        self.projective_camera_rpy_body_rad: tuple[float, float, float] | None = None
        self.projective_near_m: float | None = None
        self.projective_vertical_anchor_z_body_m: tuple[float, ...] = ()
        self.projective_attention_sigma_tokens: float | None = None
        self.projective_attention_bias_floor: float | None = None
        self.register_buffer(
            "dynamic_metric_forward_grid", None, persistent=False
        )
        self.register_buffer(
            "dynamic_metric_left_grid", None, persistent=False
        )
        if self.lift_type == GLOBAL_CROSS_ATTENTION_LIFT:
            sigma_tokens = _finite_projection_value(
                projective_attention_sigma_tokens,
                name="projective_attention_sigma_tokens",
            )
            bias_floor = _finite_projection_value(
                projective_attention_bias_floor,
                name="projective_attention_bias_floor",
            )
            if sigma_tokens != 1.0 or bias_floor != -6.0:
                raise ValueError(
                    "projective attention tuning requires "
                    "a projective lift type"
                )
            supplied = sorted(
                name
                for name, value in (projective_values | footprint_values).items()
                if value is not None
            )
            supplied.extend(
                name for name, value in cell_square_values.items() if value is not None
            )
            supplied.sort()
            if supplied:
                raise ValueError(
                    "projective camera parameters require "
                    f"a projective lift type: {supplied}"
                )
            self.register_buffer(
                "projective_attention_bias",
                None,
                persistent=False,
            )
            self.register_buffer(
                "projective_query_visibility",
                None,
                persistent=False,
            )
        elif self.lift_type in (
            PROJECTIVE_COLUMN_ATTENTION_LIFT,
            PROJECTIVE_CELL_SQUARE_ATTENTION_LIFT,
            DYNAMIC_PROJECTIVE_CELL_SQUARE_ATTENTION_LIFT,
            PROJECTIVE_FOOTPRINT_ATTENTION_LIFT,
        ):
            if self.token_side <= 0:
                raise ValueError("token_side must be positive for projective attention")
            missing = sorted(
                name for name, value in projective_values.items() if value is None
            )
            if missing:
                raise ValueError(
                    "projective-column attention requires all fixed camera "
                    f"parameters; missing={missing}"
                )
            supplied_footprint = sorted(
                name for name, value in footprint_values.items() if value is not None
            )
            supplied_cell_square = sorted(
                name for name, value in cell_square_values.items() if value is not None
            )
            if self.lift_type == PROJECTIVE_COLUMN_ATTENTION_LIFT:
                if supplied_footprint:
                    raise ValueError(
                        "footprint projection parameters require "
                        f"lift_type={PROJECTIVE_FOOTPRINT_ATTENTION_LIFT!r}: "
                        f"{supplied_footprint}"
                    )
                if supplied_cell_square:
                    raise ValueError(
                        "output-cell projection parameters require "
                        f"lift_type={PROJECTIVE_CELL_SQUARE_ATTENTION_LIFT!r}: "
                        f"{supplied_cell_square}"
                    )
                horizontal_offsets = ((0.0, 0.0),)
            elif self.lift_type in (
                PROJECTIVE_CELL_SQUARE_ATTENTION_LIFT,
                DYNAMIC_PROJECTIVE_CELL_SQUARE_ATTENTION_LIFT,
            ):
                if supplied_footprint:
                    raise ValueError(
                        "cell-square support must not use body-footprint parameters: "
                        f"{supplied_footprint}"
                    )
                if not supplied_cell_square:
                    raise ValueError(
                        "projective-cell-square attention requires "
                        "projective_output_cell_size_m"
                    )
                output_cell_size_m = _finite_projection_value(
                    projective_output_cell_size_m,
                    name="projective_output_cell_size_m",
                )
                if output_cell_size_m <= 0.0:
                    raise ValueError("projective_output_cell_size_m must be positive")
                self.projective_output_cell_size_m = output_cell_size_m
                horizontal_offsets = _cell_square_horizontal_offsets(
                    output_cell_size_m
                )
            else:
                if supplied_cell_square:
                    raise ValueError(
                        "output-cell projection parameters require "
                        f"lift_type={PROJECTIVE_CELL_SQUARE_ATTENTION_LIFT!r}: "
                        f"{supplied_cell_square}"
                    )
                missing_footprint = sorted(
                    name for name, value in footprint_values.items() if value is None
                )
                if missing_footprint:
                    raise ValueError(
                        "projective-footprint attention requires all footprint "
                        f"parameters; missing={missing_footprint}"
                    )
                footprint_radius = _finite_projection_value(
                    projective_footprint_radius_m,
                    name="projective_footprint_radius_m",
                )
                if footprint_radius <= 0.0:
                    raise ValueError("projective_footprint_radius_m must be positive")
                perimeter_samples = _projection_integer(
                    projective_footprint_perimeter_samples,
                    name="projective_footprint_perimeter_samples",
                )
                if (
                    perimeter_samples < 4
                    or perimeter_samples > 64
                    or perimeter_samples % 4 != 0
                ):
                    raise ValueError(
                        "projective_footprint_perimeter_samples must be a multiple "
                        "of four between four and 64"
                    )
                self.projective_footprint_radius_m = footprint_radius
                self.projective_footprint_perimeter_samples = perimeter_samples
                horizontal_offsets = _footprint_horizontal_offsets(
                    footprint_radius,
                    perimeter_samples,
                )
            self.projective_horizontal_offsets_body_m = horizontal_offsets
            horizontal_fov = _finite_projection_value(
                projective_horizontal_fov_deg,
                name="projective_horizontal_fov_deg",
            )
            vertical_fov = _finite_projection_value(
                projective_vertical_fov_deg,
                name="projective_vertical_fov_deg",
            )
            if not 0.0 < horizontal_fov < 180.0:
                raise ValueError("projective_horizontal_fov_deg must lie in (0, 180)")
            if not 0.0 < vertical_fov < 180.0:
                raise ValueError("projective_vertical_fov_deg must lie in (0, 180)")
            camera_xyz = _projection_triple(
                projective_camera_xyz_body_m,
                name="projective_camera_xyz_body_m",
            )
            camera_rpy = _projection_triple(
                projective_camera_rpy_body_rad,
                name="projective_camera_rpy_body_rad",
            )
            near_m = _finite_projection_value(
                projective_near_m,
                name="projective_near_m",
            )
            if near_m <= 0.0:
                raise ValueError("projective_near_m must be positive")
            anchors = _projection_anchors(projective_vertical_anchor_z_body_m)
            sigma_tokens = _finite_projection_value(
                projective_attention_sigma_tokens,
                name="projective_attention_sigma_tokens",
            )
            if sigma_tokens <= 0.0:
                raise ValueError("projective_attention_sigma_tokens must be positive")
            bias_floor = _finite_projection_value(
                projective_attention_bias_floor,
                name="projective_attention_bias_floor",
            )
            if bias_floor >= 0.0:
                raise ValueError("projective_attention_bias_floor must be negative")
            metric_forward = torch.linspace(
                *self.forward_range_m,
                self.bev_size[0],
                dtype=torch.float64,
            )
            metric_left = torch.linspace(
                *self.left_range_m,
                self.bev_size[1],
                dtype=torch.float64,
            )
            metric_forward_grid, metric_left_grid = torch.meshgrid(
                metric_forward,
                metric_left,
                indexing="ij",
            )
            if self.lift_type == DYNAMIC_PROJECTIVE_CELL_SQUARE_ATTENTION_LIFT:
                if camera_rpy != (0.0, 0.0, 0.0):
                    raise ValueError(
                        "dynamic cell-square lift requires registered zero camera RPY"
                    )
                self.projective_horizontal_fov_deg = horizontal_fov
                self.projective_vertical_fov_deg = vertical_fov
                self.projective_camera_xyz_body_m = camera_xyz
                self.projective_camera_rpy_body_rad = camera_rpy
                self.projective_near_m = near_m
                self.projective_vertical_anchor_z_body_m = anchors
                self.projective_attention_sigma_tokens = sigma_tokens
                self.projective_attention_bias_floor = bias_floor
                self.dynamic_metric_forward_grid = metric_forward_grid.to(
                    dtype=torch.float32
                )
                self.dynamic_metric_left_grid = metric_left_grid.to(
                    dtype=torch.float32
                )
                self.register_buffer(
                    "projective_attention_bias", None, persistent=False
                )
                self.register_buffer(
                    "projective_query_visibility", None, persistent=False
                )
            else:
                attention_bias, query_visibility = (
                    _projective_column_attention_geometry(
                        metric_forward_grid=metric_forward_grid,
                        metric_left_grid=metric_left_grid,
                        token_side=self.token_side,
                        horizontal_fov_deg=horizontal_fov,
                        vertical_fov_deg=vertical_fov,
                        camera_xyz_body_m=camera_xyz,
                        camera_rpy_body_rad=camera_rpy,
                        near_m=near_m,
                        vertical_anchor_z_body_m=anchors,
                        horizontal_offsets_body_m=horizontal_offsets,
                        sigma_tokens=sigma_tokens,
                        bias_floor=bias_floor,
                    )
                )
                if not bool(torch.isfinite(attention_bias).all().item()):
                    raise ValueError(
                        "projective attention geometry produced non-finite bias"
                    )
                self.register_buffer(
                    "projective_attention_bias",
                    attention_bias,
                    persistent=False,
                )
                self.register_buffer(
                    "projective_query_visibility",
                    query_visibility,
                    persistent=False,
                )
        else:
            raise ValueError(
                "lift_type must be one of "
                f"{GLOBAL_CROSS_ATTENTION_LIFT!r}, "
                f"{PROJECTIVE_COLUMN_ATTENTION_LIFT!r}, "
                f"{PROJECTIVE_CELL_SQUARE_ATTENTION_LIFT!r}, "
                f"{DYNAMIC_PROJECTIVE_CELL_SQUARE_ATTENTION_LIFT!r}, "
                f"{PROJECTIVE_FOOTPRINT_ATTENTION_LIFT!r}"
            )

    def forward(
        self,
        patch_tokens: torch.Tensor,
        *,
        base_quat_world_xyzw: torch.Tensor | None = None,
        stored_base_yaw_rad: torch.Tensor | None = None,
    ) -> torch.Tensor:
        expected_tokens = self.token_side * self.token_side
        if patch_tokens.ndim != 3 or patch_tokens.shape[1] != expected_tokens:
            raise ValueError(
                f"patch_tokens must have shape (B, {expected_tokens}, D)"
            )
        attitude_supplied = (
            base_quat_world_xyzw is not None,
            stored_base_yaw_rad is not None,
        )
        if self.lift_type == DYNAMIC_PROJECTIVE_CELL_SQUARE_ATTENTION_LIFT:
            if not all(attitude_supplied):
                raise ValueError(
                    "dynamic cell-square lift requires base quaternion and stored yaw"
                )
            assert base_quat_world_xyzw is not None
            assert stored_base_yaw_rad is not None
            if not isinstance(base_quat_world_xyzw, torch.Tensor) or not isinstance(
                stored_base_yaw_rad, torch.Tensor
            ):
                raise ValueError("dynamic attitude inputs must be tensors")
            if (
                base_quat_world_xyzw.ndim != 2
                or base_quat_world_xyzw.shape != (patch_tokens.shape[0], 4)
                or stored_base_yaw_rad.ndim != 1
                or stored_base_yaw_rad.shape != (patch_tokens.shape[0],)
            ):
                raise ValueError("attitude batch must match patch-token batch")
            if (
                base_quat_world_xyzw.device != patch_tokens.device
                or stored_base_yaw_rad.device != patch_tokens.device
            ):
                raise ValueError("attitude and patch-token tensors must share a device")
        elif any(attitude_supplied):
            raise ValueError("attitude inputs are invalid for legacy lift types")
        tokens = self.token_project(patch_tokens)
        queries = self.coordinate_query(
            self.coordinate_features.to(dtype=patch_tokens.dtype)
        )
        queries = queries + self.query_bias.to(dtype=queries.dtype)
        queries = queries[None].expand(patch_tokens.shape[0], -1, -1)
        if self.lift_type == GLOBAL_CROSS_ATTENTION_LIFT:
            attended, _weights = self.cross_attention(
                queries,
                tokens,
                tokens,
                need_weights=False,
            )
        elif self.lift_type == DYNAMIC_PROJECTIVE_CELL_SQUARE_ATTENTION_LIFT:
            if (
                self.dynamic_metric_forward_grid is None
                or self.dynamic_metric_left_grid is None
                or self.projective_horizontal_fov_deg is None
                or self.projective_vertical_fov_deg is None
                or self.projective_camera_xyz_body_m is None
                or self.projective_camera_rpy_body_rad is None
                or self.projective_near_m is None
                or not self.projective_vertical_anchor_z_body_m
                or not self.projective_horizontal_offsets_body_m
                or self.projective_attention_sigma_tokens is None
                or self.projective_attention_bias_floor is None
            ):
                raise RuntimeError("dynamic projective geometry is incomplete")
            assert base_quat_world_xyzw is not None
            assert stored_base_yaw_rad is not None
            attention_bias, query_visibility = (
                _dynamic_projective_cell_square_attention_geometry(
                    metric_forward_grid=self.dynamic_metric_forward_grid,
                    metric_left_grid=self.dynamic_metric_left_grid,
                    token_side=self.token_side,
                    horizontal_fov_deg=self.projective_horizontal_fov_deg,
                    vertical_fov_deg=self.projective_vertical_fov_deg,
                    camera_xyz_body_m=self.projective_camera_xyz_body_m,
                    camera_rpy_body_rad=self.projective_camera_rpy_body_rad,
                    near_m=self.projective_near_m,
                    vertical_anchor_z_body_m=(
                        self.projective_vertical_anchor_z_body_m
                    ),
                    horizontal_offsets_body_m=(
                        self.projective_horizontal_offsets_body_m
                    ),
                    sigma_tokens=self.projective_attention_sigma_tokens,
                    bias_floor=self.projective_attention_bias_floor,
                    base_quat_world_xyzw=base_quat_world_xyzw,
                    stored_base_yaw_rad=stored_base_yaw_rad,
                )
            )
            attention_bias = (
                attention_bias[:, None]
                .expand(-1, self.attention_heads, -1, -1)
                .reshape(
                    patch_tokens.shape[0] * self.attention_heads,
                    attention_bias.shape[1],
                    attention_bias.shape[2],
                )
            )
            attended, _weights = self.cross_attention(
                queries,
                tokens,
                tokens,
                attn_mask=attention_bias.to(dtype=queries.dtype),
                need_weights=False,
            )
            attended = attended * query_visibility.to(
                dtype=attended.dtype
            )[:, :, None]
        else:
            if (
                self.projective_attention_bias is None
                or self.projective_query_visibility is None
            ):
                raise RuntimeError("projective-column geometry buffers are missing")
            attended, _weights = self.cross_attention(
                queries,
                tokens,
                tokens,
                attn_mask=self.projective_attention_bias.to(
                    device=queries.device,
                    dtype=queries.dtype,
                ),
                need_weights=False,
            )
            attended = attended * self.projective_query_visibility.to(
                device=attended.device,
                dtype=attended.dtype,
            )[None, :, None]
        features = self.query_norm(queries + attended)
        features = features.transpose(1, 2).reshape(
            patch_tokens.shape[0],
            -1,
            self.bev_size[0],
            self.bev_size[1],
        )
        return self.refine(features)


class BevResidualPredictor(nn.Module):
    """Predict newly revealed/changed BEV latent content after geometric warp."""

    def __init__(self, *, bev_dim: int, action_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.condition = nn.Sequential(
            nn.Linear(int(action_dim) + 3, int(hidden_dim)),
            nn.GELU(),
            nn.Linear(int(hidden_dim), int(bev_dim)),
        )
        self.net = nn.Sequential(
            nn.Conv2d(int(bev_dim) * 2, int(hidden_dim), 3, padding=1),
            nn.GELU(),
            nn.Conv2d(int(hidden_dim), int(hidden_dim), 3, padding=1),
            nn.GELU(),
            nn.Conv2d(int(hidden_dim), int(bev_dim), 3, padding=1),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(
        self,
        warped_current: torch.Tensor,
        action: torch.Tensor,
        delta_pose_current: torch.Tensor,
    ) -> torch.Tensor:
        if action.ndim != 2 or action.shape[0] != warped_current.shape[0]:
            raise ValueError("action must have shape (B, action_dim)")
        condition = self.condition(torch.cat((action, delta_pose_current), dim=1))
        condition = condition[:, :, None, None].expand_as(warped_current)
        return warped_current + self.net(torch.cat((warped_current, condition), dim=1))


class EgomotionBevJepa(nn.Module):
    """Predict EMA BEV latents while learning traversability logits.

    The promoted predictive branch uses a commanded primitive and its frozen
    train-set nominal SE(2) delta. Realized future odometry is restricted to an
    auxiliary equivariance loss and must not enter that predictor.
    """

    def __init__(
        self,
        *,
        image_size: int = 128,
        patch_size: int = 16,
        encoder_dim: int = 192,
        encoder_depth: int = 6,
        encoder_heads: int = 6,
        encoder_mlp_ratio: int = 4,
        bev_dim: int = 64,
        bev_size: tuple[int, int] = (64, 64),
        forward_range_m: tuple[float, float] = (-0.95, 5.35),
        left_range_m: tuple[float, float] = (-3.15, 3.15),
        action_dim: int = 9,
        bev_attention_heads: int = 4,
        bev_lift_type: str = GLOBAL_CROSS_ATTENTION_LIFT,
        projective_horizontal_fov_deg: float | None = None,
        projective_vertical_fov_deg: float | None = None,
        projective_camera_xyz_body_m: tuple[float, float, float] | None = None,
        projective_camera_rpy_body_rad: tuple[float, float, float] | None = None,
        projective_near_m: float | None = None,
        projective_vertical_anchor_z_body_m: tuple[float, ...] | None = None,
        projective_output_cell_size_m: float | None = None,
        projective_footprint_radius_m: float | None = None,
        projective_footprint_perimeter_samples: int | None = None,
        projective_attention_sigma_tokens: float = 1.0,
        projective_attention_bias_floor: float = -6.0,
        predictor_hidden_dim: int = 128,
        target_ema_momentum: float = 0.996,
        jepa_weight: float = 1.0,
        occupancy_weight: float = 1.0,
        equivariance_weight: float = 0.25,
        action_contrast_weight: float = 1.0,
        action_margin_fraction: float = 0.1,
        variance_weight: float = 0.1,
        variance_target_std: float = 0.5,
    ) -> None:
        super().__init__()
        if image_size % patch_size != 0:
            raise ValueError("image_size must be divisible by patch_size")
        if not 0.0 <= float(target_ema_momentum) < 1.0:
            raise ValueError("target_ema_momentum must lie in [0, 1)")
        self.image_size = int(image_size)
        self.action_dim = int(action_dim)
        self.bev_size = (int(bev_size[0]), int(bev_size[1]))
        self.forward_range_m = tuple(map(float, forward_range_m))
        self.left_range_m = tuple(map(float, left_range_m))
        self.bev_lift_type = str(bev_lift_type)
        self.target_ema_momentum = float(target_ema_momentum)
        self.jepa_weight = float(jepa_weight)
        self.occupancy_weight = float(occupancy_weight)
        self.equivariance_weight = float(equivariance_weight)
        self.action_contrast_weight = float(action_contrast_weight)
        self.action_margin_fraction = float(action_margin_fraction)
        self.variance_weight = float(variance_weight)
        self.variance_target_std = float(variance_target_std)
        for name in (
            "jepa_weight",
            "occupancy_weight",
            "equivariance_weight",
            "action_contrast_weight",
            "action_margin_fraction",
            "variance_weight",
            "variance_target_std",
        ):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) < 0.0:
                raise ValueError(f"{name} must be finite and nonnegative")

        self.encoder = VisionEncoder(
            image_size=int(image_size),
            patch_size=int(patch_size),
            hidden_dim=int(encoder_dim),
            depth=int(encoder_depth),
            n_heads=int(encoder_heads),
            mlp_ratio=int(encoder_mlp_ratio),
        )
        token_side = int(image_size) // int(patch_size)
        self.bev_decoder = BevDecoder(
            token_dim=int(encoder_dim),
            bev_dim=int(bev_dim),
            token_side=token_side,
            bev_size=self.bev_size,
            forward_range_m=self.forward_range_m,
            left_range_m=self.left_range_m,
            attention_heads=int(bev_attention_heads),
            lift_type=self.bev_lift_type,
            projective_horizontal_fov_deg=projective_horizontal_fov_deg,
            projective_vertical_fov_deg=projective_vertical_fov_deg,
            projective_camera_xyz_body_m=projective_camera_xyz_body_m,
            projective_camera_rpy_body_rad=projective_camera_rpy_body_rad,
            projective_near_m=projective_near_m,
            projective_vertical_anchor_z_body_m=(
                projective_vertical_anchor_z_body_m
            ),
            projective_output_cell_size_m=projective_output_cell_size_m,
            projective_footprint_radius_m=projective_footprint_radius_m,
            projective_footprint_perimeter_samples=(
                projective_footprint_perimeter_samples
            ),
            projective_attention_sigma_tokens=projective_attention_sigma_tokens,
            projective_attention_bias_floor=projective_attention_bias_floor,
        )
        self.occupancy_head = nn.Conv2d(int(bev_dim), 3, kernel_size=1)
        self.predictor = BevResidualPredictor(
            bev_dim=int(bev_dim),
            action_dim=int(action_dim),
            hidden_dim=int(predictor_hidden_dim),
        )
        self.target_encoder = copy.deepcopy(self.encoder)
        self.target_bev_decoder = copy.deepcopy(self.bev_decoder)
        for module in (self.target_encoder, self.target_bev_decoder):
            module.requires_grad_(False)
            module.eval()

    def train(self, mode: bool = True) -> "EgomotionBevJepa":
        super().train(mode)
        self.target_encoder.eval()
        self.target_bev_decoder.eval()
        return self

    def _encode_online(
        self,
        image: torch.Tensor,
        base_quat_world_xyzw: torch.Tensor | None = None,
        stored_base_yaw_rad: torch.Tensor | None = None,
    ) -> torch.Tensor:
        tokens = self.encoder.forward_tokens(image)[:, 1:]
        return self.bev_decoder(
            tokens,
            base_quat_world_xyzw=base_quat_world_xyzw,
            stored_base_yaw_rad=stored_base_yaw_rad,
        )

    @torch.no_grad()
    def _encode_target(
        self,
        image: torch.Tensor,
        base_quat_world_xyzw: torch.Tensor | None = None,
        stored_base_yaw_rad: torch.Tensor | None = None,
    ) -> torch.Tensor:
        tokens = self.target_encoder.forward_tokens(image)[:, 1:]
        return self.target_bev_decoder(
            tokens,
            base_quat_world_xyzw=base_quat_world_xyzw,
            stored_base_yaw_rad=stored_base_yaw_rad,
        )

    def occupancy_logits(
        self,
        image: torch.Tensor,
        base_quat_world_xyzw: torch.Tensor | None = None,
        stored_base_yaw_rad: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return current-frame unknown/free/occupied logits."""

        return self.occupancy_head(
            self._encode_online(
                image,
                base_quat_world_xyzw,
                stored_base_yaw_rad,
            )
        )

    def predict_from_command(
        self,
        current_bev: torch.Tensor,
        action: torch.Tensor,
        commanded_delta_pose_current: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Predict from runtime-available command inputs only."""

        if action.ndim != 2 or action.shape != (
            current_bev.shape[0],
            self.action_dim,
        ):
            raise ValueError(f"action must have shape (B, {self.action_dim})")
        if commanded_delta_pose_current.shape != (current_bev.shape[0], 3):
            raise ValueError(
                "commanded_delta_pose_current must have shape (B, 3)"
            )
        warped, overlap = warp_bev_current_to_next(
            current_bev,
            commanded_delta_pose_current,
            forward_range_m=self.forward_range_m,
            left_range_m=self.left_range_m,
        )
        prediction = self.predictor(
            warped,
            action,
            commanded_delta_pose_current,
        )
        return prediction, warped, overlap

    @torch.no_grad()
    def update_target_encoder(self) -> None:
        momentum = self.target_ema_momentum
        for target_module, online_module in (
            (self.target_encoder, self.encoder),
            (self.target_bev_decoder, self.bev_decoder),
        ):
            for target, online in zip(
                target_module.parameters(),
                online_module.parameters(),
                strict=True,
            ):
                target.mul_(momentum).add_(online, alpha=1.0 - momentum)
            for target, online in zip(
                target_module.buffers(), online_module.buffers(), strict=True
            ):
                target.copy_(online)

    @staticmethod
    def _occupancy_loss(
        logits: torch.Tensor,
        labels: torch.Tensor | None,
        mask: torch.Tensor | None,
        class_weights: torch.Tensor | None,
        *,
        unknown_known_weights: torch.Tensor | None = None,
        free_occupied_weights: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if logits.ndim != 4 or logits.shape[1] != 3:
            raise ValueError("occupancy logits must have shape (B, 3, H, W)")
        decomposed_weights_supplied = (
            unknown_known_weights is not None,
            free_occupied_weights is not None,
        )
        if any(decomposed_weights_supplied) and not all(
            decomposed_weights_supplied
        ):
            raise ValueError(
                "unknown/known and free/occupied class weights must be supplied together"
            )
        if class_weights is not None and all(decomposed_weights_supplied):
            raise ValueError(
                "three-class and decomposed occupancy weights are mutually exclusive"
            )
        if labels is None:
            if mask is not None:
                raise ValueError("occupancy mask requires occupancy labels")
            return logits.sum() * 0.0
        expected_shape = logits.shape[:1] + logits.shape[2:]
        if labels.shape != expected_shape:
            raise ValueError("occupancy labels must have shape (B, H, W)")
        if labels.dtype not in {
            torch.uint8,
            torch.int8,
            torch.int16,
            torch.int32,
            torch.int64,
        }:
            raise ValueError("occupancy labels must use an integer dtype")
        if labels.numel() and (
            int(labels.min().item()) < UNKNOWN_CLASS
            or int(labels.max().item()) > OCCUPIED_CLASS
        ):
            raise ValueError("occupancy labels must be UNKNOWN/FREE/OCCUPIED")
        mask = _validate_bool_mask(
            mask,
            expected_shape,
            name="occupancy mask",
        )
        valid_mask = (
            mask
            if mask is not None
            else torch.ones(expected_shape, dtype=torch.bool, device=logits.device)
        )
        labels_long = labels.long()
        if all(decomposed_weights_supplied):
            unknown_known_weights = _validate_class_weights(
                unknown_known_weights,
                classes=2,
                name="unknown/known class weights",
                reference=logits,
            )
            free_occupied_weights = _validate_class_weights(
                free_occupied_weights,
                classes=2,
                name="free/occupied class weights",
                reference=logits,
            )
            known_logits = torch.logsumexp(logits[:, 1:], dim=1)
            unknown_known_logits = torch.stack(
                (logits[:, UNKNOWN_CLASS], known_logits),
                dim=1,
            )
            unknown_known_labels = (labels_long != UNKNOWN_CLASS).long()
            unknown_known_loss = _weighted_cross_entropy_mean(
                unknown_known_logits,
                unknown_known_labels,
                valid_mask,
                unknown_known_weights,
            )

            known_mask = valid_mask & (labels_long != UNKNOWN_CLASS)
            free_occupied_labels = labels_long - FREE_CLASS
            free_occupied_labels = free_occupied_labels.clamp_min(0)
            free_occupied_loss = _weighted_cross_entropy_mean(
                logits[:, 1:],
                free_occupied_labels,
                known_mask,
                free_occupied_weights,
            )
            return 0.5 * unknown_known_loss + 0.5 * free_occupied_loss

        class_weights = _validate_class_weights(
            class_weights,
            classes=3,
            name="occupancy class weights",
            reference=logits,
        )
        return _weighted_cross_entropy_mean(
            logits,
            labels_long,
            valid_mask,
            class_weights,
        )

    def forward(
        self,
        current_image: torch.Tensor,
        next_image: torch.Tensor,
        action: torch.Tensor,
        realized_delta_pose_current: torch.Tensor,
        *,
        commanded_delta_pose_current: torch.Tensor,
        current_base_quat_world_xyzw: torch.Tensor | None = None,
        current_stored_base_yaw_rad: torch.Tensor | None = None,
        next_base_quat_world_xyzw: torch.Tensor | None = None,
        next_stored_base_yaw_rad: torch.Tensor | None = None,
        current_occupancy: torch.Tensor | None = None,
        next_occupancy: torch.Tensor | None = None,
        current_occupancy_mask: torch.Tensor | None = None,
        next_occupancy_mask: torch.Tensor | None = None,
        next_prediction_mask: torch.Tensor | None = None,
        occupancy_class_weights: torch.Tensor | None = None,
        occupancy_unknown_known_weights: torch.Tensor | None = None,
        occupancy_free_occupied_weights: torch.Tensor | None = None,
        diagnostic_wrong_action: torch.Tensor | None = None,
        diagnostic_wrong_action_delta_pose_current: torch.Tensor | None = None,
        diagnostic_wrong_commanded_delta_pose_current: torch.Tensor | None = None,
    ) -> dict[str, Any]:
        if current_image.shape != next_image.shape:
            raise ValueError("current_image and next_image shapes must match")
        if action.ndim != 2 or action.shape != (
            current_image.shape[0],
            self.action_dim,
        ):
            raise ValueError(
                f"action must have shape (B, {self.action_dim})"
            )
        if realized_delta_pose_current.shape != (current_image.shape[0], 3):
            raise ValueError(
                "realized_delta_pose_current must have shape (B, 3)"
            )
        if commanded_delta_pose_current.shape != (current_image.shape[0], 3):
            raise ValueError(
                "commanded_delta_pose_current must have shape (B, 3)"
            )
        expected_grid_shape = (
            current_image.shape[0],
            self.bev_size[0],
            self.bev_size[1],
        )
        current_occupancy_mask = _validate_bool_mask(
            current_occupancy_mask,
            expected_grid_shape,
            name="current_occupancy_mask",
        )
        next_occupancy_mask = _validate_bool_mask(
            next_occupancy_mask,
            expected_grid_shape,
            name="next_occupancy_mask",
        )
        next_prediction_mask = _validate_bool_mask(
            next_prediction_mask,
            expected_grid_shape,
            name="next_prediction_mask",
        )
        if diagnostic_wrong_action is not None and diagnostic_wrong_action.shape != (
            current_image.shape[0], self.action_dim
        ):
            raise ValueError(
                f"diagnostic_wrong_action must have shape (B, {self.action_dim})"
            )
        if (diagnostic_wrong_action is None) != (
            diagnostic_wrong_action_delta_pose_current is None
        ):
            raise ValueError(
                "diagnostic wrong action and its commanded delta must be supplied together"
            )
        if (
            diagnostic_wrong_action_delta_pose_current is not None
            and diagnostic_wrong_action_delta_pose_current.shape
            != (current_image.shape[0], 3)
        ):
            raise ValueError(
                "diagnostic_wrong_action_delta_pose_current must have shape (B, 3)"
            )
        if (
            diagnostic_wrong_commanded_delta_pose_current is not None
            and diagnostic_wrong_commanded_delta_pose_current.shape
            != (current_image.shape[0], 3)
        ):
            raise ValueError(
                "diagnostic_wrong_commanded_delta_pose_current must have shape (B, 3)"
            )
        online_images = torch.cat((current_image, next_image), dim=0)
        online_quaternion = None
        online_yaw = None
        attitude_values = (
            current_base_quat_world_xyzw,
            current_stored_base_yaw_rad,
            next_base_quat_world_xyzw,
            next_stored_base_yaw_rad,
        )
        if self.bev_lift_type == DYNAMIC_PROJECTIVE_CELL_SQUARE_ATTENTION_LIFT:
            if any(value is None for value in attitude_values):
                raise ValueError(
                    "dynamic lift requires current and next quaternion and yaw"
                )
            assert current_base_quat_world_xyzw is not None
            assert current_stored_base_yaw_rad is not None
            assert next_base_quat_world_xyzw is not None
            assert next_stored_base_yaw_rad is not None
            for name, value, shape in (
                (
                    "current_base_quat_world_xyzw",
                    current_base_quat_world_xyzw,
                    (current_image.shape[0], 4),
                ),
                (
                    "current_stored_base_yaw_rad",
                    current_stored_base_yaw_rad,
                    (current_image.shape[0],),
                ),
                (
                    "next_base_quat_world_xyzw",
                    next_base_quat_world_xyzw,
                    (current_image.shape[0], 4),
                ),
                (
                    "next_stored_base_yaw_rad",
                    next_stored_base_yaw_rad,
                    (current_image.shape[0],),
                ),
            ):
                if not isinstance(value, torch.Tensor) or value.shape != shape:
                    raise ValueError(f"{name} must be a tensor with shape {shape}")
            online_quaternion = torch.cat(
                (current_base_quat_world_xyzw, next_base_quat_world_xyzw), dim=0
            )
            online_yaw = torch.cat(
                (current_stored_base_yaw_rad, next_stored_base_yaw_rad), dim=0
            )
        elif any(value is not None for value in attitude_values):
            raise ValueError("attitude inputs are invalid for legacy lift types")
        online_bev = self._encode_online(
            online_images,
            online_quaternion,
            online_yaw,
        )
        current_bev, next_online_bev = online_bev.chunk(2, dim=0)
        target_next_bev = self._encode_target(
            next_image,
            next_base_quat_world_xyzw,
            next_stored_base_yaw_rad,
        )
        predicted_next_bev, commanded_warped_current, commanded_overlap = (
            self.predict_from_command(
                current_bev,
                action,
                commanded_delta_pose_current,
            )
        )
        realized_warped_current, realized_overlap = warp_bev_current_to_next(
            current_bev,
            realized_delta_pose_current,
            forward_range_m=self.forward_range_m,
            left_range_m=self.left_range_m,
        )
        prediction_error = _normalized_spatial_error(
            predicted_next_bev,
            target_next_bev,
        )
        prediction_mask = commanded_overlap[:, 0]
        if next_prediction_mask is not None:
            prediction_mask = prediction_mask & next_prediction_mask
        jepa_loss = _masked_spatial_mean(prediction_error, prediction_mask)

        realized_prediction_mask = realized_overlap[:, 0]
        if next_prediction_mask is not None:
            realized_prediction_mask = realized_prediction_mask & next_prediction_mask
        equivariance_error = _normalized_spatial_error(
            realized_warped_current,
            target_next_bev,
        )
        equivariance_loss = _masked_spatial_mean(
            equivariance_error,
            realized_prediction_mask,
        )

        with torch.no_grad():
            persistence_error = _normalized_spatial_error(
                commanded_warped_current.detach(),
                target_next_bev,
            )
            persistence_loss = _masked_spatial_mean(
                persistence_error,
                prediction_mask,
            )
            persistence_ratio = jepa_loss.detach() / persistence_loss.clamp_min(1e-8)

        current_logits = self.occupancy_head(current_bev)
        next_logits = self.occupancy_head(next_online_bev)
        current_occ_loss = self._occupancy_loss(
            current_logits,
            current_occupancy,
            current_occupancy_mask,
            occupancy_class_weights,
            unknown_known_weights=occupancy_unknown_known_weights,
            free_occupied_weights=occupancy_free_occupied_weights,
        )
        next_occ_loss = self._occupancy_loss(
            next_logits,
            next_occupancy,
            next_occupancy_mask,
            occupancy_class_weights,
            unknown_known_weights=occupancy_unknown_known_weights,
            free_occupied_weights=occupancy_free_occupied_weights,
        )
        occupancy_terms = []
        if current_occupancy is not None:
            occupancy_terms.append(current_occ_loss)
        if next_occupancy is not None:
            occupancy_terms.append(next_occ_loss)
        occupancy_loss = (
            torch.stack(occupancy_terms).mean()
            if occupancy_terms
            else current_logits.sum() * 0.0
        )
        variance_loss = bev_variance_floor_loss(
            torch.cat((current_bev, next_online_bev), dim=0),
            target_std=self.variance_target_std,
        )
        action_contrast_loss = jepa_loss.new_zeros(())
        wrong_action_results: dict[str, torch.Tensor] = {}
        if diagnostic_wrong_action is not None:
            wrong_prediction, wrong_warped, wrong_overlap = self.predict_from_command(
                current_bev,
                diagnostic_wrong_action,
                diagnostic_wrong_action_delta_pose_current,
            )
            wrong_mask = prediction_mask & wrong_overlap[:, 0]
            real_matched_loss = _masked_spatial_mean(prediction_error, wrong_mask)
            persistence_matched_loss = _masked_spatial_mean(
                persistence_error,
                wrong_mask,
            )
            wrong_loss = _masked_spatial_mean(
                _normalized_spatial_error(wrong_prediction, target_next_bev),
                wrong_mask,
            )
            required_margin = (
                self.action_margin_fraction * persistence_matched_loss.detach()
            )
            wrong_action_contrast_loss = torch.relu(
                real_matched_loss + required_margin - wrong_loss
            )
            zero_action = torch.zeros_like(action)
            zero_delta = torch.zeros_like(commanded_delta_pose_current)
            zero_prediction, zero_warped, zero_overlap = self.predict_from_command(
                current_bev,
                zero_action,
                zero_delta,
            )
            zero_mask = prediction_mask & zero_overlap[:, 0]
            zero_real_loss = _masked_spatial_mean(prediction_error, zero_mask)
            zero_persistence_loss = _masked_spatial_mean(
                persistence_error,
                zero_mask,
            )
            zero_loss = _masked_spatial_mean(
                _normalized_spatial_error(zero_prediction, target_next_bev),
                zero_mask,
            )
            zero_action_contrast_loss = torch.relu(
                zero_real_loss
                + self.action_margin_fraction * zero_persistence_loss.detach()
                - zero_loss
            )
            action_contrast_loss = 0.5 * (
                wrong_action_contrast_loss + zero_action_contrast_loss
            )
            wrong_action_results = {
                "wrong_action_contrast_loss": wrong_action_contrast_loss.detach(),
                "wrong_action_loss": wrong_loss.detach(),
                "wrong_action_matched_real_loss": real_matched_loss.detach(),
                "wrong_action_advantage": (
                    wrong_loss.detach() - real_matched_loss.detach()
                ),
                "wrong_action_advantage_over_target_change": (
                    (wrong_loss.detach() - real_matched_loss.detach())
                    / persistence_matched_loss.clamp_min(1e-8)
                ),
                "wrong_action_prediction_sensitivity": _masked_spatial_mean(
                    _normalized_spatial_error(
                        wrong_prediction.detach(), predicted_next_bev.detach()
                    ),
                    wrong_mask,
                ),
                "wrong_action_valid_cells": wrong_mask.sum(),
                "wrong_action_matched_mask": wrong_mask[:, None],
                "wrong_action_commanded_warped_bev": wrong_warped.detach(),
                "wrong_action_predicted_next_bev": wrong_prediction.detach(),
                "zero_action_contrast_loss": zero_action_contrast_loss.detach(),
                "zero_action_matched_mask": zero_mask[:, None],
                "zero_action_commanded_warped_bev": zero_warped.detach(),
                "zero_action_predicted_next_bev": zero_prediction.detach(),
            }

        total = (
            self.jepa_weight * jepa_loss
            + self.occupancy_weight * occupancy_loss
            + self.equivariance_weight * equivariance_loss
            + self.action_contrast_weight * action_contrast_loss
            + self.variance_weight * variance_loss
        )
        result: dict[str, Any] = {
            "loss": total,
            "jepa_loss": jepa_loss,
            "equivariance_loss": equivariance_loss,
            "action_contrast_loss": action_contrast_loss,
            "warped_persistence_loss": persistence_loss,
            "prediction_to_persistence_ratio": persistence_ratio,
            "prediction_valid_cells": prediction_mask.sum(),
            "occupancy_loss": occupancy_loss,
            "variance_loss": variance_loss,
            "current_occupancy_logits": current_logits,
            "next_occupancy_logits": next_logits,
            "current_bev": current_bev,
            "commanded_warped_current_bev": commanded_warped_current,
            "warped_persistence_bev": commanded_warped_current,
            "realized_warped_current_bev": realized_warped_current,
            "predicted_next_bev": predicted_next_bev,
            "target_next_bev": target_next_bev,
            "prediction_overlap_mask": commanded_overlap,
            "prediction_valid_mask": prediction_mask[:, None],
            "realized_equivariance_overlap_mask": realized_overlap,
            "realized_equivariance_valid_mask": realized_prediction_mask[:, None],
        }
        result.update(wrong_action_results)
        if diagnostic_wrong_commanded_delta_pose_current is not None:
            with torch.no_grad():
                wrong_delta_prediction, _wrong_delta_warp, wrong_delta_overlap = (
                    self.predict_from_command(
                        current_bev.detach(),
                        action.detach(),
                        diagnostic_wrong_commanded_delta_pose_current,
                    )
                )
                wrong_delta_mask = prediction_mask & wrong_delta_overlap[:, 0]
                matched_real_loss = _masked_spatial_mean(
                    prediction_error.detach(),
                    wrong_delta_mask,
                )
                matched_persistence_loss = _masked_spatial_mean(
                    persistence_error,
                    wrong_delta_mask,
                )
                wrong_delta_loss = _masked_spatial_mean(
                    _normalized_spatial_error(
                        wrong_delta_prediction,
                        target_next_bev,
                    ),
                    wrong_delta_mask,
                )
                wrong_delta_advantage = wrong_delta_loss - matched_real_loss
                wrong_delta_sensitivity = _masked_spatial_mean(
                    _normalized_spatial_error(
                        wrong_delta_prediction,
                        predicted_next_bev.detach(),
                    ),
                    wrong_delta_mask,
                )
            result.update(
                {
                    "wrong_delta_loss": wrong_delta_loss,
                    "wrong_delta_matched_real_loss": matched_real_loss,
                    "wrong_delta_advantage": wrong_delta_advantage,
                    "wrong_delta_prediction_sensitivity": wrong_delta_sensitivity,
                    "wrong_delta_advantage_over_target_change": (
                        wrong_delta_advantage
                        / matched_persistence_loss.clamp_min(1e-8)
                    ),
                    "wrong_delta_valid_cells": wrong_delta_mask.sum(),
                    "wrong_delta_matched_mask": wrong_delta_mask[:, None],
                }
            )
        return result


__all__ = [
    "BevDecoder",
    "BevResidualPredictor",
    "DYNAMIC_PROJECTIVE_CELL_SQUARE_ATTENTION_LIFT",
    "EgomotionBevJepa",
    "FREE_CLASS",
    "OCCUPIED_CLASS",
    "UNKNOWN_CLASS",
    "bev_variance_floor_loss",
    "warp_bev_current_to_next",
]
