"""Parameter-free scoring and loss core for the projective-support JEPA probe.

This module implements only the geometry masks and differentiable Torch
operations frozen by preregistration commit ``8a52adb``.  It opens no dataset,
checkpoint, generated artifact, runner output, held-out, or sealed material.
It defines no learned parameter and deliberately reuses the unchanged geometry-
anchored model's prediction, semantic-decoder, and latent-energy APIs.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import hashlib
import math
import struct
from typing import Any, Final, Mapping, Sequence

import torch
import torch.nn.functional as F

from lewm.models.geometry_anchored_deformable_bev_lift_joint_jepa_v1 import (
    ACTION_VOCABULARY_V1,
    GeometryAnchoredDeformableBevLiftJointJepaV1Config,
    final_class_macro_nll_per_row,
    latent_energy_per_row,
)
from lewm.planning.oriented_footprint import (
    DirectionalSupportFootprint,
    OrientedRectangle,
    Pose2D,
    convex_polygon_intersects_rectangle,
    wrap_angle_pi,
)


PREREGISTRATION_COMMIT: Final = "8a52adb77d30cb98a6dd086037e6f7c296d76d63"
PREREGISTRATION_FILE_SHA256: Final = (
    "fe39daa2ff2f19624d67910d60a6da640f6351a6b5a6135db7b877bbc784e045"
)

ACTION_ORDER: Final = ACTION_VOCABULARY_V1
ACTION_COUNT: Final = 9
HOLD_ACTION_INDEX: Final = 6
NON_HOLD_ACTION_INDICES: Final = tuple(
    index for index in range(ACTION_COUNT) if index != HOLD_ACTION_INDEX
)
STATION_COUNT: Final = 11
BEV_HEIGHT: Final = 64
BEV_WIDTH: Final = 64
BEV_CELL_SIZE_M: Final = 0.10
MICROBATCH_SIZE: Final = 4
SMOOTH_MIN_TEMPERATURE: Final = 8.0
RANKING_TEMPERATURE: Final = 8.0
IMMEDIATE_MAXIMUM_CORNER_STEP_M: Final = 0.025
IMMEDIATE_MAXIMUM_YAW_STEP_RAD: Final = math.radians(5.0)

REMOTE_POSE_OFFSETS: Final = (0, 1, 10, 19, 28, 37, 46, 55, 64, 73, 82, 91)
REMOTE_POSE_SHA256: Final = (
    "df96a4d23e9f2a297467c7384e54e9d7f8eac64609e937392f0db51e3c87abc3"
)
FULL_MASK_COUNTS: Final = (49, 61, 61, 61, 61, 61, 61, 61, 61, 61, 61)
FULL_MASK_TOTAL: Final = 659
FULL_MASK_SHA256: Final = (
    "63648c9c157d032db943b4dea5168879c287c847101606c56c97688f06e69da4"
)

PERSISTENCE_MASK_COUNTS: Final[Mapping[str, tuple[int, ...]]] = {
    "arc_left": (49, 64, 63, 64, 64, 64, 63, 62, 64, 60, 64),
    "arc_right": (50, 63, 61, 63, 62, 66, 62, 64, 63, 64, 63),
    "backward": FULL_MASK_COUNTS,
    "forward_fast": (51, 63, 63, 63, 63, 63, 63, 63, 63, 63, 63),
    "forward_medium": FULL_MASK_COUNTS,
    "forward_slow": FULL_MASK_COUNTS,
    "hold": FULL_MASK_COUNTS,
    "yaw_left": (50, 64, 63, 63, 65, 63, 66, 60, 65, 61, 63),
    "yaw_right": (49, 63, 62, 66, 62, 64, 62, 64, 63, 62, 61),
}
PERSISTENCE_MASK_SHA256: Final[Mapping[str, str]] = {
    "arc_left": "ea6e49053b653dd84250647f6ca51d5aa929df7cf84a214203a6c5822f186740",
    "arc_right": "77bf4e01900e559387a11f36a2c66a9859caee93c139032bb7e74c2296f3a1c2",
    "backward": "dfc0aeac0f6f8b44a8e37c7eac16dcfbd06ee98a7e1e3bf308f78413a472b08f",
    "forward_fast": "17a8e0b66a03c8d0210a7b0bf1665daa71ba8d355df2d344d5bf06feb3f6f773",
    "forward_medium": "4b78889928776d40f0c344d37dd942f91356da333b5a98ebb843fc966bb617d9",
    "forward_slow": "f651df5fead03d200477f1bfc418f17ed3bd613918c77a7615d65fbfdc75853f",
    "hold": FULL_MASK_SHA256,
    "yaw_left": "bcba50e628bd4557840db74e2e47b9a0513d5bd0b454cd3c863d4883e1d1e6f2",
    "yaw_right": "c91dc19501891039bee3d3b9a536de655a243f9b3e4e74a88e9d9da2888f180f",
}
PERSISTENCE_MASK_TOTAL: Final = 6_040
PERSISTENCE_MASK_STACK_SHA256: Final = (
    "983577015f2822bbf60d89cd633baa9958afd624410e1a3e4390422647e59e34"
)

# The frozen observed-max-plus-margin directional polygon.  These are the exact
# serialized vertices from the SHA-bound policy named by the preregistration.
_FOOTPRINT_VERTICES_XY_M: Final = (
    (0.22180186069512248, 0.20559036817997559),
    (-0.006657641607736686, 0.2668059073252429),
    (-0.27409585007903725, 0.2668059073252429),
    (-0.3707941103470017, 0.24089568657694727),
    (-0.39359187560054765, 0.22773339067089063),
    (-0.4032734824761839, 0.2180517837952544),
    (-0.4252483854592046, 0.1799901353372656),
    (-0.4321031310225031, 0.15440787662207844),
    (-0.43210313102250314, -0.12853063446358498),
    (-0.4277240327087361, -0.14487365186190276),
    (-0.41287945563947137, -0.17058521356274112),
    (-0.3990345266614402, -0.1844301425407723),
    (-0.37245460425775256, -0.19977606789558075),
    (-0.2106736071187977, -0.24312515542966545),
    (0.16964537438186286, -0.2431251554296655),
    (0.19572552173487207, -0.23613700100794202),
    (0.24343382811563966, -0.20859259747642472),
    (0.28613377495100734, -0.16589265064105704),
    (0.3647525772485165, -0.02972089063155864),
    (0.3700000000000001, -0.010137242314228814),
    (0.3700000000000001, 0.010137242314228523),
    (0.3647525772485163, 0.02972089063155902),
    (0.298366549067437, 0.14470486435388766),
    (0.25889920964211904, 0.18417220377920562),
)

# Five identical 0.10-second commands per action, in the frozen action order.
_ACTION_COMMANDS: Final[Mapping[str, tuple[float, float, float]]] = {
    "arc_left": (0.20, 0.0, 0.45),
    "arc_right": (0.20, 0.0, -0.45),
    "backward": (-0.20, 0.0, 0.0),
    "forward_fast": (0.30, 0.0, 0.0),
    "forward_medium": (0.25, 0.0, 0.0),
    "forward_slow": (0.20, 0.0, 0.0),
    "hold": (0.0, 0.0, 0.0),
    "yaw_left": (0.0, 0.0, 0.45),
    "yaw_right": (0.0, 0.0, -0.45),
}


@dataclass(frozen=True)
class CorridorMasksV1:
    """Validated CPU mask tensors in the frozen row-major layouts."""

    full: torch.Tensor
    persistence: torch.Tensor
    projective_support: torch.Tensor


@dataclass(frozen=True)
class ImmediateSupportRegressionV1:
    """Exact one-primitive swept-footprint/projective-support witness."""

    action_mask_cell_counts: tuple[int, ...]
    action_sample_counts: tuple[int, ...]
    action_overlap_cell_counts: tuple[int, ...]
    mask_stack_sha256: str
    projective_support_cell_count: int
    overlap_cell_count: int
    passed: bool


@dataclass(frozen=True)
class CorridorScoreTermsV1:
    """Differentiable per-cell, per-station, and prefix scores."""

    free_log_odds: torch.Tensor
    station_logits: torch.Tensor
    station_probabilities: torch.Tensor
    prefix_utility: torch.Tensor


@dataclass(frozen=True)
class SemanticLossTermsV1:
    """The frozen equal-current/next semantic term S."""

    loss: torch.Tensor
    current_per_row: torch.Tensor
    next_per_row: torch.Tensor


@dataclass(frozen=True)
class MicrobatchPersistenceTermsV1:
    """Exact same-microbatch EMA persistence-normalized term P."""

    loss: torch.Tensor
    numerator: torch.Tensor
    baseline: torch.Tensor
    executed_energy_per_row: torch.Tensor
    persistence_energy_per_row: torch.Tensor


@dataclass(frozen=True)
class RankingLossTermsV1:
    """Eligible-pair/eligible-row ranking term and detached counts."""

    loss: torch.Tensor
    predicted_prefix_utility: torch.Tensor
    target_prefix_utility: torch.Tensor
    eligible_row_count: torch.Tensor
    eligible_pair_count: torch.Tensor


@dataclass(frozen=True)
class JointLossTermsV1:
    """Composite ``S + P + Q + R`` terms for one size-four microbatch."""

    loss: torch.Tensor
    semantic: torch.Tensor
    persistence: torch.Tensor
    corridor_binary: torch.Tensor
    prefix_ranking: torch.Tensor
    persistence_terms: MicrobatchPersistenceTermsV1
    ranking_terms: RankingLossTermsV1


def _remote_pose_intervals() -> tuple[tuple[Pose2D, ...], ...]:
    intervals: list[tuple[Pose2D, ...]] = [(Pose2D(290 / 200, 0.0, 0.0),)]
    for station in range(1, STATION_COUNT):
        intervals.append(tuple(
            Pose2D((290 + 40 * (station - 1) + 5 * sample) / 200, 0.0, 0.0)
            for sample in range(9)
        ))
    return tuple(intervals)


def _remote_pose_bytes(intervals: Sequence[Sequence[Pose2D]]) -> bytes:
    offsets = [0]
    flat: list[float] = []
    for interval in intervals:
        for pose in interval:
            flat.extend((pose.x_m, pose.y_m, pose.yaw_rad))
        offsets.append(len(flat) // 3)
    return (
        struct.pack(f"<{len(offsets)}q", *offsets)
        + struct.pack(f"<{len(flat)}d", *flat)
    )


def _tensor_u1_bytes(value: torch.Tensor) -> bytes:
    if value.device.type != "cpu" or value.dtype != torch.uint8:
        raise TypeError("mask hash input must be a CPU uint8 tensor")
    return bytes(value.contiguous().reshape(-1).tolist())


def _sha256_tensor_u1(value: torch.Tensor) -> str:
    return hashlib.sha256(_tensor_u1_bytes(value)).hexdigest()


def _lattice_centers() -> tuple[tuple[float, ...], tuple[float, ...]]:
    config = GeometryAnchoredDeformableBevLiftJointJepaV1Config()
    forward = tuple(torch.linspace(
        config.forward_range_m[0],
        config.forward_range_m[1],
        config.bev_size[0],
        dtype=torch.float64,
    ).tolist())
    left = tuple(torch.linspace(
        config.left_range_m[0],
        config.left_range_m[1],
        config.bev_size[1],
        dtype=torch.float64,
    ).tolist())
    return forward, left


def _rasterize_polygon_union(
    polygons: Sequence[Sequence[tuple[float, float]]],
    *,
    forward_centers: Sequence[float],
    left_centers: Sequence[float],
) -> torch.Tensor:
    """Rasterize closed polygon/cell intersections using the frozen SAT helper."""

    mask = torch.zeros((BEV_HEIGHT, BEV_WIDTH), dtype=torch.uint8)
    half = 0.5 * BEV_CELL_SIZE_M
    # This bound is only a lossless acceleration.  The final decision remains the
    # exact closed-polygon/closed-rectangle SAT predicate with its frozen epsilon.
    padding = half + 1e-9
    for polygon in polygons:
        min_x = min(vertex[0] for vertex in polygon) - padding
        max_x = max(vertex[0] for vertex in polygon) + padding
        min_y = min(vertex[1] for vertex in polygon) - padding
        max_y = max(vertex[1] for vertex in polygon) + padding
        row_indices = tuple(
            index for index, center in enumerate(forward_centers)
            if min_x <= center <= max_x
        )
        column_indices = tuple(
            index for index, center in enumerate(left_centers)
            if min_y <= center <= max_y
        )
        for row in row_indices:
            for column in column_indices:
                if mask[row, column]:
                    continue
                cell = OrientedRectangle(
                    center_xy_m=(forward_centers[row], left_centers[column]),
                    half_extent_x_m=half,
                    half_extent_y_m=half,
                    yaw_rad=0.0,
                )
                if convex_polygon_intersects_rectangle(polygon, cell):
                    mask[row, column] = 1
    return mask


def _integrated_action_poses(action: str) -> tuple[Pose2D, ...]:
    vx, vy, yaw_rate = _ACTION_COMMANDS[action]
    x_m = y_m = yaw_rad = 0.0
    poses = [Pose2D(x_m, y_m, yaw_rad)]
    for _ in range(5):
        x_m = x_m + (
            vx * math.cos(yaw_rad) - vy * math.sin(yaw_rad)
        ) * 0.10
        y_m = y_m + (
            vx * math.sin(yaw_rad) + vy * math.cos(yaw_rad)
        ) * 0.10
        yaw_rad = wrap_angle_pi(yaw_rad + yaw_rate * 0.10)
        poses.append(Pose2D(x_m, y_m, yaw_rad))
    return tuple(poses)


def _integrated_action_endpoint(action: str) -> Pose2D:
    return _integrated_action_poses(action)[-1]


def _interpolated_action_sweep_v1(
    footprint: DirectionalSupportFootprint,
    action: str,
) -> tuple[Pose2D, ...]:
    samples: list[Pose2D] = []
    poses = _integrated_action_poses(action)
    for segment, (start, end) in enumerate(zip(poses[:-1], poses[1:], strict=True)):
        delta_x = end.x_m - start.x_m
        delta_y = end.y_m - start.y_m
        delta_yaw = wrap_angle_pi(end.yaw_rad - start.yaw_rad)
        corner_motion_upper_bound_m = (
            math.hypot(delta_x, delta_y)
            + footprint.maximum_vertex_radius_m * abs(delta_yaw)
        )
        interval_count = max(
            1,
            int(math.ceil(
                corner_motion_upper_bound_m / IMMEDIATE_MAXIMUM_CORNER_STEP_M
            )),
            int(math.ceil(abs(delta_yaw) / IMMEDIATE_MAXIMUM_YAW_STEP_RAD)),
        )
        segment_samples = tuple(
            Pose2D(
                x_m=start.x_m + fraction * delta_x,
                y_m=start.y_m + fraction * delta_y,
                yaw_rad=wrap_angle_pi(start.yaw_rad + fraction * delta_yaw),
            )
            for index in range(interval_count + 1)
            for fraction in (index / interval_count,)
        )
        samples.extend(segment_samples[(1 if segment else 0) :])
    return tuple(samples)


def build_projective_support_mask_v1() -> torch.Tensor:
    """Reconstruct the model's exact fixed 1,964-cell anchor-support mask."""

    config = GeometryAnchoredDeformableBevLiftJointJepaV1Config()
    dtype = torch.float64
    forward = torch.linspace(
        config.forward_range_m[0], config.forward_range_m[1], BEV_HEIGHT, dtype=dtype
    )
    left = torch.linspace(
        config.left_range_m[0], config.left_range_m[1], BEV_WIDTH, dtype=dtype
    )
    forward_grid, left_grid = torch.meshgrid(forward, left, indexing="ij")
    camera_forward = forward_grid - config.camera_origin_xyz_m[0]
    camera_right = -left_grid
    camera_up = config.ground_z_m - config.camera_origin_xyz_m[2]
    safe_forward = camera_forward.clamp_min(torch.finfo(dtype).eps)
    grid_x = camera_right / (
        safe_forward * math.tan(math.radians(config.horizontal_fov_degrees) / 2.0)
    )
    grid_y = -camera_up / (
        safe_forward * math.tan(math.radians(config.vertical_fov_degrees) / 2.0)
    )
    support = (
        (camera_forward > config.camera_near_m)
        & (grid_x >= -1.0)
        & (grid_x <= 1.0)
        & (grid_y >= -1.0)
        & (grid_y <= 1.0)
    )
    if int(support.sum().item()) != 1_964:
        raise RuntimeError("projective-support population changed")
    return support


def validate_immediate_footprint_support_regression_v1(
    immediate_masks: torch.Tensor,
    projective_support: torch.Tensor,
    action_sample_counts: Sequence[int],
) -> ImmediateSupportRegressionV1:
    """Fail closed unless every immediate swept footprint has zero support."""

    if immediate_masks.shape != (ACTION_COUNT, BEV_HEIGHT, BEV_WIDTH):
        raise ValueError("immediate masks must have shape (9,64,64)")
    if immediate_masks.dtype != torch.uint8 or immediate_masks.device.type != "cpu":
        raise TypeError("immediate masks must be CPU uint8")
    if not immediate_masks.is_contiguous():
        raise ValueError("immediate masks must be C-contiguous")
    if not bool(((immediate_masks == 0) | (immediate_masks == 1)).all()):
        raise ValueError("immediate masks must be binary")
    if (
        projective_support.shape != (BEV_HEIGHT, BEV_WIDTH)
        or projective_support.dtype != torch.bool
        or projective_support.device.type != "cpu"
    ):
        raise TypeError("projective support must be CPU bool (64,64)")
    if int(projective_support.sum().item()) != 1_964:
        raise RuntimeError("projective-support population changed")
    sample_counts = tuple(int(value) for value in action_sample_counts)
    if len(sample_counts) != ACTION_COUNT or any(value < 1 for value in sample_counts):
        raise ValueError("every action must have a positive sweep-sample count")
    overlaps = immediate_masks.bool() & projective_support[None]
    overlap_counts = tuple(int(mask.sum().item()) for mask in overlaps)
    overlap_count = int(overlaps.sum().item())
    if overlap_count != 0:
        raise RuntimeError("an immediate swept footprint overlaps projective support")
    return ImmediateSupportRegressionV1(
        action_mask_cell_counts=tuple(
            int(mask.sum().item()) for mask in immediate_masks
        ),
        action_sample_counts=sample_counts,
        action_overlap_cell_counts=overlap_counts,
        mask_stack_sha256=_sha256_tensor_u1(immediate_masks),
        projective_support_cell_count=1_964,
        overlap_cell_count=overlap_count,
        passed=True,
    )


def build_immediate_footprint_support_regression_v1(
) -> ImmediateSupportRegressionV1:
    """Reconstruct and witness the preregistered one-primitive zero support."""

    footprint = DirectionalSupportFootprint(vertices_xy_m=_FOOTPRINT_VERTICES_XY_M)
    forward_centers, left_centers = _lattice_centers()
    action_masks: list[torch.Tensor] = []
    sample_counts: list[int] = []
    for action in ACTION_ORDER:
        samples = _interpolated_action_sweep_v1(footprint, action)
        sample_counts.append(len(samples))
        action_masks.append(_rasterize_polygon_union(
            tuple(footprint.vertices_at(pose) for pose in samples),
            forward_centers=forward_centers,
            left_centers=left_centers,
        ))
    return validate_immediate_footprint_support_regression_v1(
        torch.stack(action_masks).contiguous(),
        build_projective_support_mask_v1(),
        tuple(sample_counts),
    )


def _construct_masks() -> CorridorMasksV1:
    intervals = _remote_pose_intervals()
    pose_sha = hashlib.sha256(_remote_pose_bytes(intervals)).hexdigest()
    if pose_sha != REMOTE_POSE_SHA256:
        raise RuntimeError("remote-pose identity changed")

    footprint = DirectionalSupportFootprint(
        vertices_xy_m=_FOOTPRINT_VERTICES_XY_M
    )
    forward_centers, left_centers = _lattice_centers()
    full_rows = []
    for interval in intervals:
        polygons = tuple(footprint.vertices_at(pose) for pose in interval)
        full_rows.append(_rasterize_polygon_union(
            polygons,
            forward_centers=forward_centers,
            left_centers=left_centers,
        ))
    full = torch.stack(full_rows).contiguous()

    persistence_actions = []
    for action in ACTION_ORDER:
        if action == "hold":
            persistence_actions.append(full.clone())
            continue
        endpoint = _integrated_action_endpoint(action)
        cos_yaw = math.cos(endpoint.yaw_rad)
        sin_yaw = math.sin(endpoint.yaw_rad)
        action_rows = []
        for interval in intervals:
            polygons = []
            for pose in interval:
                next_polygon = footprint.vertices_at(pose)
                polygons.append(tuple(
                    (
                        endpoint.x_m + cos_yaw * x_m - sin_yaw * y_m,
                        endpoint.y_m + sin_yaw * x_m + cos_yaw * y_m,
                    )
                    for x_m, y_m in next_polygon
                ))
            action_rows.append(_rasterize_polygon_union(
                polygons,
                forward_centers=forward_centers,
                left_centers=left_centers,
            ))
        persistence_actions.append(torch.stack(action_rows))
    persistence = torch.stack(persistence_actions).contiguous()
    support = build_projective_support_mask_v1()
    validate_corridor_masks_v1(full, persistence, support)
    return CorridorMasksV1(full=full, persistence=persistence, projective_support=support)


@lru_cache(maxsize=1)
def _cached_masks() -> CorridorMasksV1:
    return _construct_masks()


def build_validated_corridor_masks_v1() -> CorridorMasksV1:
    """Return clones of the exact validated full, persistence, and support masks."""

    cached = _cached_masks()
    return CorridorMasksV1(
        full=cached.full.clone(),
        persistence=cached.persistence.clone(),
        projective_support=cached.projective_support.clone(),
    )


def build_full_corridor_masks_v1() -> torch.Tensor:
    """Return the validated C-contiguous ``uint8 [11,64,64]`` full mask."""

    return build_validated_corridor_masks_v1().full


def build_persistence_corridor_masks_v1() -> torch.Tensor:
    """Return validated current-frame ``uint8 [9,11,64,64]`` masks."""

    return build_validated_corridor_masks_v1().persistence


def validate_corridor_masks_v1(
    full: torch.Tensor,
    persistence: torch.Tensor,
    projective_support: torch.Tensor | None = None,
) -> None:
    """Fail closed unless mask layouts, counts, hashes, and support are exact."""

    if full.shape != (STATION_COUNT, BEV_HEIGHT, BEV_WIDTH):
        raise ValueError("full mask must have shape (11,64,64)")
    if persistence.shape != (ACTION_COUNT, STATION_COUNT, BEV_HEIGHT, BEV_WIDTH):
        raise ValueError("persistence masks must have shape (9,11,64,64)")
    if full.dtype != torch.uint8 or persistence.dtype != torch.uint8:
        raise TypeError("corridor masks must use exact uint8")
    if full.device.type != "cpu" or persistence.device.type != "cpu":
        raise TypeError("corridor masks must be validated on CPU")
    if not full.is_contiguous() or not persistence.is_contiguous():
        raise ValueError("corridor masks must be C-contiguous")
    if not bool(((full == 0) | (full == 1)).all()):
        raise ValueError("full mask is not binary")
    if not bool(((persistence == 0) | (persistence == 1)).all()):
        raise ValueError("persistence mask is not binary")
    full_counts = tuple(int(row.sum().item()) for row in full)
    if (
        full_counts != FULL_MASK_COUNTS
        or int(full.sum().item()) != FULL_MASK_TOTAL
        or _sha256_tensor_u1(full) != FULL_MASK_SHA256
    ):
        raise RuntimeError("full corridor-mask identity changed")
    for action_index, action in enumerate(ACTION_ORDER):
        mask = persistence[action_index]
        counts = tuple(int(row.sum().item()) for row in mask)
        if (
            counts != PERSISTENCE_MASK_COUNTS[action]
            or _sha256_tensor_u1(mask) != PERSISTENCE_MASK_SHA256[action]
        ):
            raise RuntimeError(f"{action} persistence-mask identity changed")
    if (
        int(persistence.sum().item()) != PERSISTENCE_MASK_TOTAL
        or _sha256_tensor_u1(persistence) != PERSISTENCE_MASK_STACK_SHA256
    ):
        raise RuntimeError("persistence mask-stack identity changed")
    support = (
        build_projective_support_mask_v1()
        if projective_support is None
        else projective_support
    )
    if support.shape != (BEV_HEIGHT, BEV_WIDTH) or support.dtype != torch.bool:
        raise TypeError("projective support must be bool (64,64)")
    if int(support.sum().item()) != 1_964:
        raise RuntimeError("projective-support population changed")
    if bool((full.bool() & ~support).any()):
        raise RuntimeError("full corridor mask escaped projective support")
    if bool((persistence.bool() & ~support[None, None]).any()):
        raise RuntimeError("persistence corridor mask escaped projective support")


def predict_and_decode_all_actions_v1(
    model: Any,
    current_latent: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Predict and decode all nine action latents through the shared head."""

    predicted = model.predict_all_actions(current_latent)
    return predicted, decode_all_action_semantic_logits_v1(model, predicted)


def decode_all_action_semantic_logits_v1(
    model: Any,
    predicted_latents: torch.Tensor,
) -> torch.Tensor:
    """Decode ``[B,9,64,64,64]`` latents as ``[B,9,3,64,64]``."""

    if predicted_latents.ndim != 5 or tuple(predicted_latents.shape[1:]) != (
        ACTION_COUNT,
        64,
        BEV_HEIGHT,
        BEV_WIDTH,
    ):
        raise ValueError("predicted latents must have shape (B,9,64,64,64)")
    if predicted_latents.shape[0] < 1:
        raise ValueError("predicted latents must contain at least one row")
    batch = predicted_latents.shape[0]
    flat = predicted_latents.reshape(
        batch * ACTION_COUNT, 64, BEV_HEIGHT, BEV_WIDTH
    )
    decoded = model.semantic_logits_from_latent(flat)
    if decoded.shape != (batch * ACTION_COUNT, 3, BEV_HEIGHT, BEV_WIDTH):
        raise RuntimeError("shared semantic decoder returned an unexpected shape")
    return decoded.reshape(batch, ACTION_COUNT, 3, BEV_HEIGHT, BEV_WIDTH)


def free_log_odds_v1(semantic_logits: torch.Tensor) -> torch.Tensor:
    """Return ``FREE`` versus ``UNKNOWN or OCCUPIED`` log odds per cell."""

    if semantic_logits.ndim not in (4, 5) or semantic_logits.shape[-3] != 3:
        raise ValueError("semantic logits must have shape (B,3,H,W) or (B,A,3,H,W)")
    if not semantic_logits.is_floating_point():
        raise TypeError("semantic logits must use a floating dtype")
    unknown = semantic_logits.select(-3, 0)
    free = semantic_logits.select(-3, 1)
    occupied = semantic_logits.select(-3, 2)
    return free - torch.logaddexp(unknown, occupied)


def _canonical_action_masks(
    masks: torch.Tensor,
    *,
    action_count: int,
    height: int,
    width: int,
    reference: torch.Tensor,
) -> torch.Tensor:
    if masks.ndim == 3:
        if masks.shape != (STATION_COUNT, height, width):
            raise ValueError("shared masks must have shape (11,H,W)")
        masks = masks[None].expand(action_count, -1, -1, -1)
    elif masks.ndim == 4:
        if masks.shape != (action_count, STATION_COUNT, height, width):
            raise ValueError("action masks must have shape (A,11,H,W)")
    else:
        raise ValueError("masks must have shape (11,H,W) or (A,11,H,W)")
    if masks.dtype not in (torch.bool, torch.uint8):
        raise TypeError("masks must use bool or uint8")
    if masks.dtype == torch.uint8 and not bool(((masks == 0) | (masks == 1)).all()):
        raise ValueError("uint8 masks must be binary")
    masks = masks.to(device=reference.device, dtype=torch.bool)
    if bool((masks.sum(dim=(-2, -1)) == 0).any()):
        raise ValueError("every station mask must be nonempty")
    return masks


def smooth_min_station_logits_v1(
    free_log_odds: torch.Tensor,
    masks: torch.Tensor,
) -> torch.Tensor:
    """Apply the frozen temperature-eight normalized smooth minimum."""

    if free_log_odds.ndim != 4:
        raise ValueError("free log odds must have shape (B,A,H,W)")
    if not free_log_odds.is_floating_point():
        raise TypeError("free log odds must use a floating dtype")
    batch, actions, height, width = free_log_odds.shape
    if batch < 1 or actions < 1:
        raise ValueError("free log odds must contain rows and actions")
    action_masks = _canonical_action_masks(
        masks,
        action_count=actions,
        height=height,
        width=width,
        reference=free_log_odds,
    )
    temperature = free_log_odds.new_tensor(SMOOTH_MIN_TEMPERATURE)
    scaled = -temperature * free_log_odds[:, :, None]
    scaled = scaled.masked_fill(~action_masks[None], -torch.inf)
    log_sum = torch.logsumexp(scaled.flatten(-2), dim=-1)
    counts = action_masks.sum(dim=(-2, -1)).to(dtype=free_log_odds.dtype)
    return -log_sum / temperature + torch.log(counts)[None] / temperature


def differentiable_prefix_utility_v1(
    station_probabilities: torch.Tensor,
) -> torch.Tensor:
    """Expected normalized safe prefix ``sum_s prod_(j<=s) p_j / 11``."""

    if station_probabilities.ndim != 3 or station_probabilities.shape[-1] != STATION_COUNT:
        raise ValueError("station probabilities must have shape (B,A,11)")
    if not station_probabilities.is_floating_point():
        raise TypeError("station probabilities must use a floating dtype")
    return station_probabilities.cumprod(dim=-1).sum(dim=-1) / STATION_COUNT


def corridor_scores_from_semantic_logits_v1(
    semantic_logits: torch.Tensor,
    masks: torch.Tensor,
) -> CorridorScoreTermsV1:
    """Compute log odds, interval logits/probabilities, and expected prefix."""

    log_odds = free_log_odds_v1(semantic_logits)
    if log_odds.ndim == 3:
        if masks.ndim != 4:
            raise ValueError("single semantic field requires action-specific masks")
        log_odds = log_odds[:, None].expand(-1, masks.shape[0], -1, -1)
    station_logits = smooth_min_station_logits_v1(log_odds, masks)
    station_probabilities = torch.sigmoid(station_logits)
    return CorridorScoreTermsV1(
        free_log_odds=log_odds,
        station_logits=station_logits,
        station_probabilities=station_probabilities,
        prefix_utility=differentiable_prefix_utility_v1(station_probabilities),
    )


def semantic_loss_v1(
    current_logits: torch.Tensor,
    current_labels: torch.Tensor,
    next_logits: torch.Tensor,
    next_labels: torch.Tensor,
) -> SemanticLossTermsV1:
    """Compute exact equal-row, equal-current/next present-class term ``S``."""

    if current_logits.shape != next_logits.shape or current_labels.shape != next_labels.shape:
        raise ValueError("current and next semantic batches must have matching shapes")
    current_rows = final_class_macro_nll_per_row(current_logits, current_labels)
    next_rows = final_class_macro_nll_per_row(next_logits, next_labels)
    loss = (
        0.5 * current_rows.mean() + 0.5 * next_rows.mean()
    ) / math.log(3.0)
    return SemanticLossTermsV1(
        loss=loss,
        current_per_row=current_rows,
        next_per_row=next_rows,
    )


def corridor_binary_loss_v1(
    station_logits: torch.Tensor,
    station_safe: torch.Tensor,
) -> torch.Tensor:
    """Compute equal-row/action/station binary corridor term ``Q``."""

    if station_logits.ndim != 3 or station_logits.shape[1:] != (
        ACTION_COUNT,
        STATION_COUNT,
    ):
        raise ValueError("station logits must have shape (B,9,11)")
    if station_safe.shape != station_logits.shape:
        raise ValueError("station-safe labels must match station logits")
    labels = station_safe.to(device=station_logits.device)
    if labels.dtype == torch.bool or not labels.is_floating_point():
        labels = labels.to(dtype=station_logits.dtype)
    elif labels.dtype != station_logits.dtype:
        labels = labels.to(dtype=station_logits.dtype)
    if not bool(((labels == 0) | (labels == 1)).all()):
        raise ValueError("station-safe labels must be exactly zero or one")
    return F.binary_cross_entropy_with_logits(
        station_logits, labels, reduction="mean"
    ) / math.log(2.0)


def prefix_ranking_loss_v1(
    station_probabilities: torch.Tensor,
    station_safe: torch.Tensor,
) -> RankingLossTermsV1:
    """Compute pair means within rows, then the mean over eligible rows only."""

    if station_probabilities.shape != station_safe.shape:
        raise ValueError("station probabilities and labels must share a shape")
    if station_probabilities.ndim != 3 or station_probabilities.shape[1:] != (
        ACTION_COUNT,
        STATION_COUNT,
    ):
        raise ValueError("ranking tensors must have shape (B,9,11)")
    labels = station_safe.to(
        device=station_probabilities.device,
        dtype=station_probabilities.dtype,
    )
    if not bool(((labels == 0) | (labels == 1)).all()):
        raise ValueError("station-safe labels must be exactly zero or one")
    predicted = differentiable_prefix_utility_v1(station_probabilities)
    target = differentiable_prefix_utility_v1(labels)
    indices = torch.tensor(
        NON_HOLD_ACTION_INDICES,
        dtype=torch.long,
        device=station_probabilities.device,
    )
    predicted_non_hold = predicted.index_select(1, indices)
    target_non_hold = target.index_select(1, indices)
    eligible = target_non_hold[:, :, None] > target_non_hold[:, None, :]
    margins = predicted_non_hold[:, :, None] - predicted_non_hold[:, None, :]
    pair_losses = F.softplus(-RANKING_TEMPERATURE * margins) / math.log(2.0)
    pair_counts = eligible.sum(dim=(1, 2))
    eligible_rows = pair_counts > 0
    per_row = (
        (pair_losses * eligible.to(dtype=pair_losses.dtype)).sum(dim=(1, 2))
        / pair_counts.clamp_min(1).to(dtype=pair_losses.dtype)
    )
    if bool(eligible_rows.any()):
        loss = per_row[eligible_rows].mean()
    else:
        loss = 0.0 * predicted.sum()
    return RankingLossTermsV1(
        loss=loss,
        predicted_prefix_utility=predicted,
        target_prefix_utility=target,
        eligible_row_count=eligible_rows.sum().detach(),
        eligible_pair_count=eligible.sum().detach(),
    )


def microbatch_persistence_loss_v1(
    predicted_latents: torch.Tensor,
    executed_action_indices: torch.Tensor,
    ema_current_latent: torch.Tensor,
    ema_next_latent: torch.Tensor,
) -> MicrobatchPersistenceTermsV1:
    """Compute the exact detached same-size-four-microbatch term ``P``."""

    if (
        predicted_latents.ndim != 5
        or tuple(predicted_latents.shape[:3])
        != (MICROBATCH_SIZE, ACTION_COUNT, 64)
        or predicted_latents.shape[-2] < 1
        or predicted_latents.shape[-1] < 1
    ):
        raise ValueError("predicted latents must have shape (4,9,64,H,W)")
    expected_target = (
        MICROBATCH_SIZE,
        64,
        predicted_latents.shape[-2],
        predicted_latents.shape[-1],
    )
    if ema_current_latent.shape != expected_target or ema_next_latent.shape != expected_target:
        raise ValueError("EMA latents must have shape (4,64,H,W) matching predictions")
    if predicted_latents.dtype != ema_current_latent.dtype or predicted_latents.dtype != ema_next_latent.dtype:
        raise TypeError("prediction and EMA latent dtypes differ")
    if predicted_latents.device != ema_current_latent.device or predicted_latents.device != ema_next_latent.device:
        raise TypeError("prediction and EMA latents must share a device")
    if executed_action_indices.shape != (MICROBATCH_SIZE,):
        raise ValueError("executed action indices must have shape (4,)")
    if executed_action_indices.is_floating_point() or executed_action_indices.dtype == torch.bool:
        raise TypeError("executed action indices must use an integer dtype")
    action_indices = executed_action_indices.to(
        device=predicted_latents.device, dtype=torch.long
    )
    if bool(((action_indices < 0) | (action_indices >= ACTION_COUNT)).any()):
        raise ValueError("executed action index escaped the frozen vocabulary")
    rows = torch.arange(MICROBATCH_SIZE, device=predicted_latents.device)
    executed = predicted_latents[rows, action_indices]
    target_current = ema_current_latent.detach()
    target_next = ema_next_latent.detach()
    persistence_energy = latent_energy_per_row(target_current, target_next)
    baseline = persistence_energy.mean().detach().clamp_min(1e-6)
    executed_energy = latent_energy_per_row(executed, target_next)
    numerator = executed_energy.mean()
    return MicrobatchPersistenceTermsV1(
        loss=numerator / baseline,
        numerator=numerator,
        baseline=baseline,
        executed_energy_per_row=executed_energy,
        persistence_energy_per_row=persistence_energy,
    )


def joint_microbatch_loss_v1(
    *,
    semantic_loss: torch.Tensor,
    predicted_latents: torch.Tensor,
    executed_action_indices: torch.Tensor,
    ema_current_latent: torch.Tensor,
    ema_next_latent: torch.Tensor,
    station_logits: torch.Tensor,
    station_safe: torch.Tensor,
) -> JointLossTermsV1:
    """Compute the frozen parameter-free composite ``L = S + P + Q + R``."""

    if semantic_loss.ndim != 0 or not semantic_loss.is_floating_point():
        raise ValueError("semantic loss must be a floating scalar")
    if station_logits.shape != (MICROBATCH_SIZE, ACTION_COUNT, STATION_COUNT):
        raise ValueError("joint station logits must have shape (4,9,11)")
    persistence = microbatch_persistence_loss_v1(
        predicted_latents,
        executed_action_indices,
        ema_current_latent,
        ema_next_latent,
    )
    corridor_binary = corridor_binary_loss_v1(station_logits, station_safe)
    ranking = prefix_ranking_loss_v1(torch.sigmoid(station_logits), station_safe)
    total = semantic_loss + persistence.loss + corridor_binary + ranking.loss
    return JointLossTermsV1(
        loss=total,
        semantic=semantic_loss,
        persistence=persistence.loss,
        corridor_binary=corridor_binary,
        prefix_ranking=ranking.loss,
        persistence_terms=persistence,
        ranking_terms=ranking,
    )


__all__ = [
    "ACTION_COUNT",
    "ACTION_ORDER",
    "BEV_HEIGHT",
    "BEV_WIDTH",
    "CorridorMasksV1",
    "CorridorScoreTermsV1",
    "FULL_MASK_COUNTS",
    "FULL_MASK_SHA256",
    "FULL_MASK_TOTAL",
    "HOLD_ACTION_INDEX",
    "IMMEDIATE_MAXIMUM_CORNER_STEP_M",
    "IMMEDIATE_MAXIMUM_YAW_STEP_RAD",
    "ImmediateSupportRegressionV1",
    "JointLossTermsV1",
    "MICROBATCH_SIZE",
    "MicrobatchPersistenceTermsV1",
    "NON_HOLD_ACTION_INDICES",
    "PERSISTENCE_MASK_COUNTS",
    "PERSISTENCE_MASK_SHA256",
    "PERSISTENCE_MASK_STACK_SHA256",
    "PERSISTENCE_MASK_TOTAL",
    "PREREGISTRATION_COMMIT",
    "PREREGISTRATION_FILE_SHA256",
    "REMOTE_POSE_OFFSETS",
    "REMOTE_POSE_SHA256",
    "RankingLossTermsV1",
    "STATION_COUNT",
    "SemanticLossTermsV1",
    "build_full_corridor_masks_v1",
    "build_immediate_footprint_support_regression_v1",
    "build_persistence_corridor_masks_v1",
    "build_projective_support_mask_v1",
    "build_validated_corridor_masks_v1",
    "corridor_binary_loss_v1",
    "corridor_scores_from_semantic_logits_v1",
    "decode_all_action_semantic_logits_v1",
    "differentiable_prefix_utility_v1",
    "free_log_odds_v1",
    "joint_microbatch_loss_v1",
    "microbatch_persistence_loss_v1",
    "predict_and_decode_all_actions_v1",
    "prefix_ranking_loss_v1",
    "semantic_loss_v1",
    "smooth_min_station_logits_v1",
    "validate_corridor_masks_v1",
    "validate_immediate_footprint_support_regression_v1",
]
