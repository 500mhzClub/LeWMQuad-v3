"""Pure geometry and summaries for the N32 camera-pose projection audit.

This module performs no file I/O.  The audit runner owns the train-metadata
access boundary and supplies one recorded base/camera pose at a time.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Mapping, Sequence

import numpy as np

RESULT_SCHEMA = "lewm_go2_n32_pose_projection_audit_v1"
GEOMETRY_SCHEMA = "lewm_go2_n32_pose_projection_geometry_v1"
SUMMARY_SCHEMA = "lewm_go2_n32_pose_projection_summary_v1"
DECISION_SCHEMA = "lewm_go2_n32_pose_projection_ordering_decision_v1"
FAMILIES = (
    "open_obstacle_field",
    "rough_local_dynamics",
    "small_enclosed_maze",
    "medium_enclosed_maze",
    "large_enclosed_maze",
)
RADIAL_BIN_COUNT = 64
ANGULAR_BIN_COUNT = 256
RADIAL_BIN_SIZE_M = 0.10
HORIZONTAL_FOV_DEG = 78.323
VERTICAL_FOV_DEG = 62.8370386364
CAMERA_NEAR_M = 0.05
CAMERA_XYZ_BODY_M = (0.326, 0.0, 0.043)
VERTICAL_ANCHOR_Z_BODY_M = (-0.333, -0.133, 0.067, 0.267, 0.467)
TOKEN_SIDE = 16
QUERY_SHAPE = (
    len(VERTICAL_ANCHOR_Z_BODY_M),
    RADIAL_BIN_COUNT,
    ANGULAR_BIN_COUNT,
)
QUERY_COUNT = int(np.prod(QUERY_SHAPE))
REGISTERED_CAMERA_ORIGIN = np.asarray(CAMERA_XYZ_BODY_M, dtype=np.float64)
REGISTERED_FORWARD = np.asarray((1.0, 0.0, 0.0), dtype=np.float64)
REGISTERED_LEFT = np.asarray((0.0, 1.0, 0.0), dtype=np.float64)
REGISTERED_UP = np.asarray((0.0, 0.0, 1.0), dtype=np.float64)


@dataclass(frozen=True)
class CameraGeometry:
    """One camera pose expressed in the yaw-aligned base frame."""

    origin_xyz: np.ndarray
    forward_xyz: np.ndarray
    left_xyz: np.ndarray
    up_xyz: np.ndarray
    forward_pitch_rad: float
    up_roll_rad: float


@dataclass(frozen=True)
class ProjectionComparison:
    """Per-frame scalar report plus valid-in-both displacement samples."""

    metrics: dict[str, Any]
    token_displacements: np.ndarray


def _finite_vector(value: Sequence[float], *, name: str) -> np.ndarray:
    vector = np.asarray(value, dtype=np.float64)
    if vector.shape != (3,) or not np.isfinite(vector).all():
        raise ValueError(f"{name} must contain three finite values")
    return vector


def _unit(value: np.ndarray, *, name: str) -> np.ndarray:
    norm = float(np.linalg.norm(value))
    if not math.isfinite(norm) or norm <= 1e-12:
        raise ValueError(f"{name} cannot be normalized")
    return np.asarray(value, dtype=np.float64) / norm


def registered_anchor_queries() -> np.ndarray:
    """Return the immutable ``[5,64,256,3]`` yaw-base anchor queries."""

    radial = (
        np.arange(RADIAL_BIN_COUNT, dtype=np.float64) + 0.5
    ) * RADIAL_BIN_SIZE_M
    half_fov = math.radians(HORIZONTAL_FOV_DEG) * 0.5
    angular = -half_fov + (
        np.arange(ANGULAR_BIN_COUNT, dtype=np.float64) + 0.5
    ) * (2.0 * half_fov / ANGULAR_BIN_COUNT)
    radius_grid, angle_grid = np.meshgrid(radial, angular, indexing="ij")
    forward = radius_grid * np.cos(angle_grid)
    left = radius_grid * np.sin(angle_grid)
    result = np.empty((*QUERY_SHAPE, 3), dtype=np.float64)
    result[..., 0] = forward[None]
    result[..., 1] = left[None]
    result[..., 2] = np.asarray(
        VERTICAL_ANCHOR_Z_BODY_M, dtype=np.float64
    )[:, None, None]
    return result


def reconstruct_yaw_aligned_camera(
    *,
    base_position_world: Sequence[float],
    base_yaw_rad: float,
    camera_position_world: Sequence[float],
    camera_lookat_world: Sequence[float],
    camera_up_world: Sequence[float],
) -> CameraGeometry:
    """Transform and orthonormalize one recorded camera in yaw-base axes."""

    base = _finite_vector(base_position_world, name="base position")
    camera = _finite_vector(camera_position_world, name="camera position")
    lookat = _finite_vector(camera_lookat_world, name="camera lookat")
    up_hint_world = _finite_vector(camera_up_world, name="camera up")
    yaw = float(base_yaw_rad)
    if not math.isfinite(yaw):
        raise ValueError("base yaw must be finite")
    cos_yaw = math.cos(yaw)
    sin_yaw = math.sin(yaw)
    yaw_axes_world = np.asarray(
        (
            (cos_yaw, -sin_yaw, 0.0),
            (sin_yaw, cos_yaw, 0.0),
            (0.0, 0.0, 1.0),
        ),
        dtype=np.float64,
    )

    def in_yaw_frame(vector_world: np.ndarray) -> np.ndarray:
        return yaw_axes_world.T @ vector_world

    origin = in_yaw_frame(camera - base)
    forward = _unit(
        in_yaw_frame(lookat - camera), name="camera forward"
    )
    up_hint = _unit(in_yaw_frame(up_hint_world), name="camera up")
    left = _unit(np.cross(up_hint, forward), name="camera left")
    up = _unit(np.cross(forward, left), name="orthogonal camera up")
    if float(np.dot(up, up_hint)) < 0.0:
        left = -left
        up = -up

    pitch = math.atan2(
        float(forward[2]),
        math.hypot(float(forward[0]), float(forward[1])),
    )
    level_up = REGISTERED_UP - float(np.dot(REGISTERED_UP, forward)) * forward
    level_up = _unit(level_up, name="level-reference camera up")
    roll = math.atan2(
        float(np.dot(np.cross(level_up, up), forward)),
        float(np.dot(level_up, up)),
    )
    basis = np.stack((forward, left, up))
    if not np.allclose(basis @ basis.T, np.eye(3), rtol=0.0, atol=1e-12):
        raise ValueError("camera basis did not orthonormalize")
    if float(np.linalg.det(basis)) < 1.0 - 1e-12:
        raise ValueError("camera basis is not right-handed forward/left/up")
    return CameraGeometry(
        origin_xyz=origin,
        forward_xyz=forward,
        left_xyz=left,
        up_xyz=up,
        forward_pitch_rad=pitch,
        up_roll_rad=roll,
    )


def project_registered_queries(
    camera: CameraGeometry,
    *,
    queries_xyz: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Project registered queries to normalized grid-sample coordinates."""

    queries = (
        registered_anchor_queries()
        if queries_xyz is None
        else np.asarray(queries_xyz, dtype=np.float64)
    )
    if queries.shape != (*QUERY_SHAPE, 3) or not np.isfinite(queries).all():
        raise ValueError(f"queries must have finite shape {(*QUERY_SHAPE, 3)}")
    delta = queries - np.asarray(camera.origin_xyz, dtype=np.float64)
    depth = np.einsum("...c,c->...", delta, camera.forward_xyz)
    camera_left = np.einsum("...c,c->...", delta, camera.left_xyz)
    camera_up = np.einsum("...c,c->...", delta, camera.up_xyz)
    tan_horizontal = math.tan(math.radians(HORIZONTAL_FOV_DEG) * 0.5)
    tan_vertical = math.tan(math.radians(VERTICAL_FOV_DEG) * 0.5)
    safe_depth = np.where(np.abs(depth) > np.finfo(np.float64).eps, depth, 1.0)
    normalized_u = -camera_left / (safe_depth * tan_horizontal)
    normalized_v = -camera_up / (safe_depth * tan_vertical)
    valid = (
        (depth >= CAMERA_NEAR_M)
        & (normalized_u >= -1.0)
        & (normalized_u <= 1.0)
        & (normalized_v >= -1.0)
        & (normalized_v <= 1.0)
    )
    grid = np.stack((normalized_u, normalized_v), axis=-1)
    grid = np.where(valid[..., None], grid, 2.0)
    return grid, valid


def registered_camera_geometry() -> CameraGeometry:
    """Return the fixed level-camera geometry used by the width-24 head."""

    return CameraGeometry(
        origin_xyz=REGISTERED_CAMERA_ORIGIN.copy(),
        forward_xyz=REGISTERED_FORWARD.copy(),
        left_xyz=REGISTERED_LEFT.copy(),
        up_xyz=REGISTERED_UP.copy(),
        forward_pitch_rad=0.0,
        up_roll_rad=0.0,
    )


def _linear_quantile(values: np.ndarray, quantile: float) -> float:
    data = np.asarray(values, dtype=np.float64)
    if data.ndim != 1 or not data.size or not np.isfinite(data).all():
        raise ValueError("quantiles require a nonempty finite float64 vector")
    return float(np.quantile(data, float(quantile), method="linear"))


def displacement_summary(values: np.ndarray) -> dict[str, float | int]:
    """Summarize valid-in-both token displacements exactly as registered."""

    data = np.asarray(values, dtype=np.float64)
    if data.ndim != 1 or not data.size or not np.isfinite(data).all():
        raise ValueError("token displacement requires nonempty finite values")
    return {
        "count": int(data.size),
        "p50_token": _linear_quantile(data, 0.50),
        "p95_token": _linear_quantile(data, 0.95),
        "maximum_token": float(np.max(data)),
        "fraction_ge_0_5_token": float(np.count_nonzero(data >= 0.5) / data.size),
    }


def compare_projection(camera: CameraGeometry) -> ProjectionComparison:
    """Compare one actual camera pose against the immutable fixed projection."""

    queries = registered_anchor_queries()
    fixed_grid, fixed_valid = project_registered_queries(
        registered_camera_geometry(), queries_xyz=queries
    )
    actual_grid, actual_valid = project_registered_queries(
        camera, queries_xyz=queries
    )
    valid_in_both = fixed_valid & actual_valid
    if not valid_in_both.any():
        raise ValueError("actual and fixed cameras have no jointly valid queries")
    normalized_delta = actual_grid[valid_in_both] - fixed_grid[valid_in_both]
    token_displacement = (
        np.linalg.norm(normalized_delta, axis=-1).astype(np.float64, copy=False)
        * (TOKEN_SIDE / 2.0)
    )
    origin_delta = np.asarray(camera.origin_xyz) - REGISTERED_CAMERA_ORIGIN
    flip_count = int(np.count_nonzero(fixed_valid ^ actual_valid))
    metrics: dict[str, Any] = {
        "camera_origin_xyz_yaw_base_m": [float(value) for value in camera.origin_xyz],
        "camera_origin_delta_xyz_m": [float(value) for value in origin_delta],
        "camera_origin_delta_norm_m": float(np.linalg.norm(origin_delta)),
        "camera_forward_xyz_yaw_base": [float(value) for value in camera.forward_xyz],
        "camera_left_xyz_yaw_base": [float(value) for value in camera.left_xyz],
        "camera_up_xyz_yaw_base": [float(value) for value in camera.up_xyz],
        "camera_forward_pitch_rad": float(camera.forward_pitch_rad),
        "camera_forward_pitch_deg": math.degrees(camera.forward_pitch_rad),
        "camera_up_roll_rad": float(camera.up_roll_rad),
        "camera_up_roll_deg": math.degrees(camera.up_roll_rad),
        "fixed_valid_query_count": int(np.count_nonzero(fixed_valid)),
        "actual_valid_query_count": int(np.count_nonzero(actual_valid)),
        "valid_in_both_query_count": int(np.count_nonzero(valid_in_both)),
        "validity_flip_count": flip_count,
        "validity_flip_rate": float(flip_count / QUERY_COUNT),
        "token_displacement": displacement_summary(token_displacement),
    }
    return ProjectionComparison(metrics=metrics, token_displacements=token_displacement)


def scalar_distribution(values: Sequence[float]) -> dict[str, float | int]:
    """Return a deterministic signed scalar distribution summary."""

    data = np.asarray(tuple(values), dtype=np.float64)
    if data.ndim != 1 or not data.size or not np.isfinite(data).all():
        raise ValueError("scalar distribution requires finite values")
    return {
        "count": int(data.size),
        "minimum": float(np.min(data)),
        "p05": _linear_quantile(data, 0.05),
        "p50": _linear_quantile(data, 0.50),
        "p95": _linear_quantile(data, 0.95),
        "maximum": float(np.max(data)),
        "mean": float(np.mean(data, dtype=np.float64)),
    }


def summarize_frame_comparisons(
    comparisons: Sequence[ProjectionComparison],
) -> dict[str, Any]:
    """Aggregate frame metrics with pooled query-level displacement quantiles."""

    records = tuple(comparisons)
    if not records:
        raise ValueError("projection summary requires at least one frame")
    displacements = np.concatenate(
        [record.token_displacements for record in records]
    ).astype(np.float64, copy=False)
    metrics = [record.metrics for record in records]
    fixed_count = sum(int(record["fixed_valid_query_count"]) for record in metrics)
    actual_count = sum(int(record["actual_valid_query_count"]) for record in metrics)
    joint_count = sum(int(record["valid_in_both_query_count"]) for record in metrics)
    flip_count = sum(int(record["validity_flip_count"]) for record in metrics)
    origin_components = np.asarray(
        [record["camera_origin_delta_xyz_m"] for record in metrics],
        dtype=np.float64,
    )
    return {
        "schema": SUMMARY_SCHEMA,
        "frame_count": len(records),
        "query_count_per_frame": QUERY_COUNT,
        "camera_origin_delta_forward_m": scalar_distribution(origin_components[:, 0]),
        "camera_origin_delta_left_m": scalar_distribution(origin_components[:, 1]),
        "camera_origin_delta_up_m": scalar_distribution(origin_components[:, 2]),
        "camera_origin_delta_norm_m": scalar_distribution(
            [record["camera_origin_delta_norm_m"] for record in metrics]
        ),
        "camera_forward_pitch_deg": scalar_distribution(
            [record["camera_forward_pitch_deg"] for record in metrics]
        ),
        "camera_up_roll_deg": scalar_distribution(
            [record["camera_up_roll_deg"] for record in metrics]
        ),
        "fixed_valid_query_count": fixed_count,
        "actual_valid_query_count": actual_count,
        "valid_in_both_query_count": joint_count,
        "valid_in_both_per_frame": scalar_distribution(
            [record["valid_in_both_query_count"] for record in metrics]
        ),
        "validity_flip_count": flip_count,
        "validity_flip_rate": float(flip_count / (len(records) * QUERY_COUNT)),
        "validity_flip_per_frame": scalar_distribution(
            [record["validity_flip_count"] for record in metrics]
        ),
        "token_displacement": displacement_summary(displacements),
        "per_frame_p50_token_displacement": scalar_distribution(
            [record["token_displacement"]["p50_token"] for record in metrics]
        ),
    }


def ordering_decision(
    comparisons_by_family: Mapping[str, Sequence[ProjectionComparison]],
) -> dict[str, Any]:
    """Apply the frozen dynamic-pose-versus-hierarchy ordering decision."""

    if tuple(comparisons_by_family) != tuple(FAMILIES):
        raise ValueError("projection decision families are not in registered order")
    per_frame_p50 = {
        family: np.asarray(
            [
                record.metrics["token_displacement"]["p50_token"]
                for record in comparisons_by_family[family]
            ],
            dtype=np.float64,
        )
        for family in FAMILIES
    }
    if any(not values.size for values in per_frame_p50.values()):
        raise ValueError("projection decision requires every registered family")
    rough = _linear_quantile(per_frame_p50["rough_local_dynamics"], 0.50)
    non_rough = np.concatenate(
        [
            per_frame_p50[family]
            for family in FAMILIES
            if family != "rough_local_dynamics"
        ]
    )
    non_rough_median = _linear_quantile(non_rough, 0.50)
    difference = rough - non_rough_median
    rough_gate = rough >= 0.5
    contrast_gate = difference >= 0.25
    material = bool(rough_gate and contrast_gate)
    return {
        "schema": DECISION_SCHEMA,
        "estimand": "median_of_per_frame_p50_token_displacement",
        "rough_local_dynamics_median_token": rough,
        "pooled_non_rough_median_token": non_rough_median,
        "rough_minus_non_rough_median_token": difference,
        "rough_threshold_token": 0.5,
        "rough_minus_non_rough_threshold_token": 0.25,
        "rough_threshold_passes": rough_gate,
        "contrast_threshold_passes": contrast_gate,
        "material_dynamic_pose_mismatch": material,
        "next_intervention": (
            "fixed_vs_recorded_pose_projective_sampling_ab"
            if material
            else "explicit_hierarchical_output"
        ),
    }


def geometry_contract() -> dict[str, Any]:
    """Return the exact immutable geometry represented by this audit."""

    radial = (
        np.arange(RADIAL_BIN_COUNT, dtype=np.float64) + 0.5
    ) * RADIAL_BIN_SIZE_M
    half_fov = math.radians(HORIZONTAL_FOV_DEG) * 0.5
    angular = -half_fov + (
        np.arange(ANGULAR_BIN_COUNT, dtype=np.float64) + 0.5
    ) * (2.0 * half_fov / ANGULAR_BIN_COUNT)
    return {
        "schema": GEOMETRY_SCHEMA,
        "query_shape": list(QUERY_SHAPE),
        "query_count": QUERY_COUNT,
        "radial_centers_m": [float(value) for value in radial],
        "angular_centers_rad": [float(value) for value in angular],
        "vertical_anchor_z_yaw_base_m": list(VERTICAL_ANCHOR_Z_BODY_M),
        "horizontal_fov_deg": HORIZONTAL_FOV_DEG,
        "vertical_fov_deg": VERTICAL_FOV_DEG,
        "camera_near_m": CAMERA_NEAR_M,
        "registered_camera_origin_xyz_yaw_base_m": list(CAMERA_XYZ_BODY_M),
        "registered_camera_rpy_yaw_base_rad": [0.0, 0.0, 0.0],
        "token_grid_shape": [TOKEN_SIDE, TOKEN_SIDE],
        "normalized_displacement_to_token_multiplier": TOKEN_SIDE / 2.0,
        "quantile_method": "numpy_linear_float64",
        "family_order": list(FAMILIES),
    }


__all__ = [
    "CameraGeometry",
    "DECISION_SCHEMA",
    "GEOMETRY_SCHEMA",
    "ProjectionComparison",
    "QUERY_COUNT",
    "QUERY_SHAPE",
    "RESULT_SCHEMA",
    "SUMMARY_SCHEMA",
    "compare_projection",
    "displacement_summary",
    "geometry_contract",
    "ordering_decision",
    "project_registered_queries",
    "reconstruct_yaw_aligned_camera",
    "registered_anchor_queries",
    "registered_camera_geometry",
    "scalar_distribution",
    "summarize_frame_comparisons",
]
