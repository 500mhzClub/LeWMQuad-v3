"""Pure-Torch training mechanics for observable camera-ray evidence V4.

This module derives ordered first-hit targets, computes the pixel and ground
losses, and constructs a differentiable physical-evidence raster from raw model
outputs.  It consumes no scene map, collision geometry, or morphology.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence

import torch
import torch.nn.functional as F

from lewm.benchmarks.go2_observable_camera_ray_evidence_v4 import (
    CAMERA_BASIS_ORTHONORMAL_ATOL,
    CAMERA_HORIZONTAL_FOV_DEG,
    CAMERA_NEAR_M,
    CAMERA_VERTICAL_FOV_DEG,
    FREE_CLASS,
    GROUND_SUPPORT_COUNT,
    OCCUPIED_CLASS,
    OUTPUT_CELL_SIZE_M,
    OUTPUT_FORWARD_MIN_EDGE_M,
    OUTPUT_LEFT_MIN_EDGE_M,
    OUTPUT_SHAPE,
    UNKNOWN_CLASS,
)

from .observable_camera_ray_evidence_v4 import (
    DEPTH_BIN_COUNT,
    DEPTH_BIN_SIZE_M,
    DEPTH_FAR_EDGE_M,
    DEPTH_NEAR_EDGE_M,
    ObservableCameraRayEvidenceV4RawOutput,
    ordered_obstacle_first_hit_log_probabilities_v4,
)


DEFAULT_GROUND_DISTANCE_BIN_EDGES_M = (
    0.0,
    1.0,
    2.0,
    3.0,
    4.0,
    5.0,
    float("inf"),
)
DEFAULT_PIXEL_RAY_CHUNK_SIZE = 256


@dataclass(frozen=True)
class ObservableCameraRayEvidenceV4Targets:
    """Canonical ordered-pixel and ground-query training targets."""

    pixel_in_range_hit_mask: torch.Tensor
    pixel_no_hit_mask: torch.Tensor
    pixel_hit_bin_index: torch.Tensor
    pixel_within_bin_offset_m: torch.Tensor
    ground_in_frustum: torch.Tensor
    ground_clear_to_target: torch.Tensor

    def __post_init__(self) -> None:
        pixel_shape = tuple(self.pixel_in_range_hit_mask.shape)
        if len(pixel_shape) != 3:
            raise ValueError("pixel target masks must have shape (B,Hray,Wray)")
        if tuple(self.pixel_no_hit_mask.shape) != pixel_shape:
            raise ValueError("pixel no-hit mask shape changed")
        if tuple(self.pixel_hit_bin_index.shape) != pixel_shape:
            raise ValueError("pixel hit-bin shape changed")
        if tuple(self.pixel_within_bin_offset_m.shape) != pixel_shape:
            raise ValueError("pixel offset target shape changed")
        if self.pixel_in_range_hit_mask.dtype != torch.bool:
            raise ValueError("pixel in-range hit mask must be boolean")
        if self.pixel_no_hit_mask.dtype != torch.bool:
            raise ValueError("pixel no-hit mask must be boolean")
        if self.pixel_hit_bin_index.dtype != torch.long:
            raise ValueError("pixel hit-bin target must use torch.long")
        if not torch.equal(
            self.pixel_no_hit_mask,
            ~self.pixel_in_range_hit_mask,
        ):
            raise ValueError("pixel hit and no-hit masks must partition every ray")
        ground_shape = tuple(self.ground_in_frustum.shape)
        if len(ground_shape) != 4 or ground_shape[-1] != GROUND_SUPPORT_COUNT:
            raise ValueError("ground targets must have shape (B,Sx,Sy,5)")
        if tuple(self.ground_clear_to_target.shape) != ground_shape:
            raise ValueError("ground clear target shape changed")
        if self.ground_in_frustum.dtype != torch.bool:
            raise ValueError("ground in-frustum target must be boolean")
        if self.ground_clear_to_target.dtype != torch.bool:
            raise ValueError("ground clear target must be boolean")


@dataclass(frozen=True)
class SoftObservableCameraRayRasterV4:
    """Differentiable UNKNOWN/FREE/OCCUPIED evidence probabilities."""

    source_free_probability: torch.Tensor
    free_given_not_occupied_probability: torch.Tensor
    occupied_probability: torch.Tensor
    class_probabilities: torch.Tensor

    def __post_init__(self) -> None:
        source = self.source_free_probability
        conditional_free = self.free_given_not_occupied_probability
        occupied = self.occupied_probability
        classes = self.class_probabilities
        if source.ndim != 3:
            raise ValueError("source FREE probability must have shape (B,Sx,Sy)")
        if conditional_free.ndim != 3:
            raise ValueError("conditional FREE probability must have shape (B,H,W)")
        if tuple(occupied.shape) != tuple(conditional_free.shape):
            raise ValueError("occupied and conditional FREE rasters must match")
        if tuple(classes.shape) != (
            conditional_free.shape[0],
            3,
            conditional_free.shape[1],
            conditional_free.shape[2],
        ):
            raise ValueError("class probabilities must have shape (B,3,H,W)")


@dataclass(frozen=True)
class HierarchicalRasterCrossEntropyV4:
    """OCCUPIED-first raster loss and its two Bernoulli components."""

    total: torch.Tensor
    occupied: torch.Tensor
    free_given_not_occupied: torch.Tensor
    occupied_count: int = 0
    rest_count: int = 0
    free_count: int = 0
    unknown_count: int = 0


@dataclass(frozen=True)
class OrderedObstacleFirstHitNLLBreakdownV4:
    """State-balanced ordered NLL and the contributing event counts."""

    total: torch.Tensor
    no_hit_count: int
    hit_distance_bin_counts: tuple[int, ...]
    nonempty_group_count: int


def derive_observable_camera_ray_evidence_v4_targets(
    *,
    pixel_hit_mask: torch.Tensor,
    pixel_first_hit_distance_m: torch.Tensor,
    ground_support_in_frustum: torch.Tensor,
    ground_support_clear_to_target: torch.Tensor,
    depth_bin_count: int = DEPTH_BIN_COUNT,
    depth_near_edge_m: float = DEPTH_NEAR_EDGE_M,
    depth_bin_size_m: float = DEPTH_BIN_SIZE_M,
) -> ObservableCameraRayEvidenceV4Targets:
    """Derive ordered-bin targets without NumPy or privileged geometry."""

    if not all(
        isinstance(value, torch.Tensor)
        for value in (
            pixel_hit_mask,
            pixel_first_hit_distance_m,
            ground_support_in_frustum,
            ground_support_clear_to_target,
        )
    ):
        raise TypeError("V4 target inputs must be tensors")
    if pixel_hit_mask.ndim != 3 or pixel_hit_mask.dtype != torch.bool:
        raise ValueError("pixel_hit_mask must be boolean with shape (B,H,W)")
    if tuple(pixel_first_hit_distance_m.shape) != tuple(pixel_hit_mask.shape):
        raise ValueError("pixel hit range must match pixel hit mask")
    if not pixel_first_hit_distance_m.is_floating_point():
        raise ValueError("pixel hit range must be floating point")
    if pixel_first_hit_distance_m.device != pixel_hit_mask.device:
        raise ValueError("pixel target tensors must share a device")
    if not bool(torch.isfinite(pixel_first_hit_distance_m).all().item()):
        raise ValueError("pixel hit range must be finite")
    if bool((pixel_first_hit_distance_m[~pixel_hit_mask] != 0.0).any().item()):
        raise ValueError("pixel no-hit ranges must be canonical zero")
    near = float(depth_near_edge_m)
    bin_size = float(depth_bin_size_m)
    bin_count = int(depth_bin_count)
    if bin_count <= 0 or not math.isfinite(near) or near < 0.0:
        raise ValueError("depth bins require a non-negative finite near edge")
    if not math.isfinite(bin_size) or bin_size <= 0.0:
        raise ValueError("depth bin size must be positive and finite")
    if bool((pixel_first_hit_distance_m[pixel_hit_mask] < near).any().item()):
        raise ValueError("pixel hit range lies before the represented near edge")
    far = near + bin_count * bin_size
    in_range_hit = pixel_hit_mask & (pixel_first_hit_distance_m < far)
    scaled = (pixel_first_hit_distance_m - near) / bin_size
    scaled_tolerance = 8.0 * torch.finfo(pixel_first_hit_distance_m.dtype).eps
    bin_index = torch.floor(scaled + scaled_tolerance).to(dtype=torch.long)
    bin_index = torch.where(in_range_hit, bin_index, torch.zeros_like(bin_index))
    if bool(
        (
            in_range_hit
            & ((bin_index < 0) | (bin_index >= bin_count))
        )
        .any()
        .item()
    ):
        raise RuntimeError("derived pixel bin lies outside the represented range")
    bin_center = near + (bin_index.to(pixel_first_hit_distance_m.dtype) + 0.5) * bin_size
    offset = pixel_first_hit_distance_m - bin_center
    offset = torch.where(in_range_hit, offset, torch.zeros_like(offset))
    tolerance = 8.0 * torch.finfo(pixel_first_hit_distance_m.dtype).eps
    if bool(
        (offset[in_range_hit].abs() > 0.5 * bin_size + tolerance).any().item()
    ):
        raise RuntimeError("derived within-bin offset exceeds its bin")

    ground_shape = tuple(ground_support_in_frustum.shape)
    if (
        len(ground_shape) != 4
        or ground_shape[-1] != GROUND_SUPPORT_COUNT
        or ground_support_in_frustum.dtype != torch.bool
    ):
        raise ValueError("ground in-frustum labels must be boolean (B,Sx,Sy,5)")
    if tuple(ground_support_clear_to_target.shape) != ground_shape:
        raise ValueError("ground clear labels must match ground in-frustum labels")
    if ground_support_clear_to_target.dtype != torch.bool:
        raise ValueError("ground clear labels must be boolean")
    if ground_support_in_frustum.device != pixel_hit_mask.device or (
        ground_support_clear_to_target.device != pixel_hit_mask.device
    ):
        raise ValueError("pixel and ground targets must share a device")
    if bool(
        (ground_support_clear_to_target & ~ground_support_in_frustum).any().item()
    ):
        raise ValueError("out-of-frustum ground support cannot be clear")
    if ground_shape[0] != pixel_hit_mask.shape[0]:
        raise ValueError("pixel and ground target batches differ")
    return ObservableCameraRayEvidenceV4Targets(
        pixel_in_range_hit_mask=in_range_hit,
        pixel_no_hit_mask=~in_range_hit,
        pixel_hit_bin_index=bin_index,
        pixel_within_bin_offset_m=offset,
        ground_in_frustum=ground_support_in_frustum,
        ground_clear_to_target=ground_support_clear_to_target,
    )


def ordered_obstacle_first_hit_nll_v4(
    hazard_logits: torch.Tensor,
    targets: ObservableCameraRayEvidenceV4Targets,
) -> torch.Tensor:
    """Equally weight nonempty no-hit and represented hit-distance groups."""

    return ordered_obstacle_first_hit_nll_breakdown_v4(
        hazard_logits,
        targets,
    ).total


def ordered_obstacle_first_hit_nll_breakdown_v4(
    hazard_logits: torch.Tensor,
    targets: ObservableCameraRayEvidenceV4Targets,
) -> OrderedObstacleFirstHitNLLBreakdownV4:
    """Return a skew-resistant ordered NLL and its exact group counts.

    The no-hit rays form one group. Each represented depth bin forms another
    group when it contains at least one target hit. Every nonempty group has
    equal weight regardless of its raw ray count.
    """

    probabilities = ordered_obstacle_first_hit_log_probabilities_v4(hazard_logits)
    expected = (
        hazard_logits.shape[0],
        hazard_logits.shape[2],
        hazard_logits.shape[3],
    )
    if tuple(targets.pixel_in_range_hit_mask.shape) != expected:
        raise ValueError("pixel targets do not match hazard logits")
    if hazard_logits.device != targets.pixel_hit_bin_index.device:
        raise ValueError("pixel predictions and targets must share a device")
    if bool(
        (
            targets.pixel_in_range_hit_mask
            & (targets.pixel_hit_bin_index >= hazard_logits.shape[1])
        )
        .any()
        .item()
    ):
        raise ValueError("pixel target bin exceeds hazard depth")
    hit_log_probability = probabilities.hit.gather(
        1,
        targets.pixel_hit_bin_index[:, None],
    ).squeeze(1)
    no_hit_mask = targets.pixel_no_hit_mask
    no_hit_count = int(no_hit_mask.sum().item())
    group_losses = []
    if no_hit_count:
        group_losses.append(-probabilities.no_hit[no_hit_mask].mean())
    hit_counts = []
    for depth_bin in range(hazard_logits.shape[1]):
        mask = (
            targets.pixel_in_range_hit_mask
            & (targets.pixel_hit_bin_index == depth_bin)
        )
        count = int(mask.sum().item())
        hit_counts.append(count)
        if count:
            group_losses.append(-hit_log_probability[mask].mean())
    total = (
        torch.stack(group_losses).mean()
        if group_losses
        else hazard_logits.sum() * 0.0
    )
    return OrderedObstacleFirstHitNLLBreakdownV4(
        total=total,
        no_hit_count=no_hit_count,
        hit_distance_bin_counts=tuple(hit_counts),
        nonempty_group_count=len(group_losses),
    )


def in_range_pixel_offset_smooth_l1_v4(
    predicted_offset_m: torch.Tensor,
    targets: ObservableCameraRayEvidenceV4Targets,
    *,
    beta: float = 0.01,
) -> torch.Tensor:
    """Smooth-L1 offset loss only at the target bin of represented hits."""

    if not isinstance(predicted_offset_m, torch.Tensor):
        raise TypeError("predicted_offset_m must be a tensor")
    if predicted_offset_m.ndim != 4 or not predicted_offset_m.is_floating_point():
        raise ValueError("predicted offsets must have shape (B,D,H,W)")
    expected = (
        predicted_offset_m.shape[0],
        predicted_offset_m.shape[2],
        predicted_offset_m.shape[3],
    )
    if tuple(targets.pixel_in_range_hit_mask.shape) != expected:
        raise ValueError("pixel targets do not match predicted offsets")
    if not math.isfinite(float(beta)) or float(beta) <= 0.0:
        raise ValueError("Smooth-L1 beta must be positive and finite")
    selected = predicted_offset_m.gather(
        1,
        targets.pixel_hit_bin_index[:, None],
    ).squeeze(1)
    mask = targets.pixel_in_range_hit_mask
    if not bool(mask.any().item()):
        return predicted_offset_m.sum() * 0.0
    return F.smooth_l1_loss(
        selected[mask],
        targets.pixel_within_bin_offset_m[mask].to(dtype=selected.dtype),
        beta=float(beta),
        reduction="mean",
    )


def balanced_ground_clear_bce_v4(
    clear_logits: torch.Tensor,
    targets: ObservableCameraRayEvidenceV4Targets,
    target_distance_m: torch.Tensor,
    *,
    distance_bin_edges_m: Sequence[float] = DEFAULT_GROUND_DISTANCE_BIN_EDGES_M,
) -> torch.Tensor:
    """Average BCE equally over nonempty distance-bin x clear-state groups."""

    if tuple(clear_logits.shape) != tuple(targets.ground_in_frustum.shape):
        raise ValueError("ground logits do not match ground targets")
    if tuple(target_distance_m.shape) != tuple(clear_logits.shape):
        raise ValueError("ground distance does not match ground logits")
    if not clear_logits.is_floating_point() or not target_distance_m.is_floating_point():
        raise ValueError("ground logits and distances must be floating point")
    if clear_logits.device != targets.ground_in_frustum.device or (
        target_distance_m.device != clear_logits.device
    ):
        raise ValueError("ground predictions, distances, and targets must share a device")
    if not bool(torch.isfinite(target_distance_m).all().item()):
        raise ValueError("ground target distances must be finite")
    edges = tuple(float(value) for value in distance_bin_edges_m)
    if len(edges) < 2 or any(
        math.isnan(value) for value in edges
    ) or any(right <= left for left, right in zip(edges, edges[1:])):
        raise ValueError("ground distance-bin edges must be strictly increasing")
    if edges[0] > 0.0 or not math.isinf(edges[-1]) or edges[-1] < 0.0:
        raise ValueError("ground distance bins must cover [0,+inf)")
    valid = targets.ground_in_frustum
    if bool((target_distance_m[valid] < edges[0]).any().item()):
        raise ValueError("ground distance lies below the distance-bin domain")
    boundaries = clear_logits.new_tensor(edges[1:-1])
    bin_index = torch.bucketize(
        target_distance_m.to(dtype=clear_logits.dtype).contiguous(),
        boundaries,
        right=True,
    )
    group_losses = []
    for distance_bin in range(len(edges) - 1):
        for clear_state in (False, True):
            mask = (
                valid
                & (bin_index == distance_bin)
                & (targets.ground_clear_to_target == clear_state)
            )
            if bool(mask.any().item()):
                group_losses.append(
                    F.binary_cross_entropy_with_logits(
                        clear_logits[mask],
                        targets.ground_clear_to_target[mask].to(
                            dtype=clear_logits.dtype
                        ),
                        reduction="mean",
                    )
                )
    if not group_losses:
        return clear_logits.sum() * 0.0
    return torch.stack(group_losses).mean()


def _validate_camera_calibration(
    camera_origin_body_m: torch.Tensor,
    camera_basis_body_fru: torch.Tensor,
    *,
    batch: int,
    dtype: torch.dtype,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    if not isinstance(camera_origin_body_m, torch.Tensor) or not isinstance(
        camera_basis_body_fru, torch.Tensor
    ):
        raise TypeError("camera origin and basis must be tensors")
    if tuple(camera_origin_body_m.shape) != (batch, 3):
        raise ValueError("camera origin must have shape (B,3)")
    if tuple(camera_basis_body_fru.shape) != (batch, 3, 3):
        raise ValueError("camera basis must have shape (B,3,3)")
    if not camera_origin_body_m.is_floating_point() or not (
        camera_basis_body_fru.is_floating_point()
    ):
        raise ValueError("camera calibration must be floating point")
    if camera_origin_body_m.device != device or camera_basis_body_fru.device != device:
        raise ValueError("camera calibration and predictions must share a device")
    if not bool(torch.isfinite(camera_origin_body_m).all().item()) or not bool(
        torch.isfinite(camera_basis_body_fru).all().item()
    ):
        raise ValueError("camera calibration must be finite")
    origin64 = camera_origin_body_m.to(dtype=torch.float64)
    basis64 = camera_basis_body_fru.to(dtype=torch.float64)
    identity = torch.eye(3, dtype=torch.float64, device=device)[None]
    if not torch.allclose(
        torch.bmm(basis64, basis64.transpose(1, 2)),
        identity.expand(batch, -1, -1),
        rtol=0.0,
        atol=CAMERA_BASIS_ORTHONORMAL_ATOL,
    ):
        raise ValueError("camera basis must be orthonormal")
    if not torch.allclose(
        torch.linalg.cross(basis64[:, 1], basis64[:, 0], dim=1),
        basis64[:, 2],
        rtol=0.0,
        atol=CAMERA_BASIS_ORTHONORMAL_ATOL,
    ):
        raise ValueError("camera basis must use forward/right/up handedness")
    return camera_origin_body_m.to(dtype=dtype), camera_basis_body_fru.to(dtype=dtype)


def calibrated_pixel_ray_directions_torch_v4(
    camera_basis_body_fru: torch.Tensor,
    *,
    ray_shape: tuple[int, int],
    dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """Return calibrated body-frame ray directions on a uniform pixel lattice."""

    if not isinstance(camera_basis_body_fru, torch.Tensor):
        raise TypeError("camera_basis_body_fru must be a tensor")
    if camera_basis_body_fru.ndim != 3:
        raise ValueError("camera basis must have shape (B,3,3)")
    height, width = (int(ray_shape[0]), int(ray_shape[1]))
    if height <= 0 or width <= 0:
        raise ValueError("ray_shape dimensions must be positive")
    result_dtype = camera_basis_body_fru.dtype if dtype is None else dtype
    basis = camera_basis_body_fru.to(dtype=result_dtype)
    pixel_x = torch.arange(width, dtype=result_dtype, device=basis.device) + 0.5
    pixel_y = torch.arange(height, dtype=result_dtype, device=basis.device) + 0.5
    normalized_x = (2.0 * pixel_x / float(width) - 1.0) * math.tan(
        math.radians(CAMERA_HORIZONTAL_FOV_DEG) * 0.5
    )
    normalized_y = (1.0 - 2.0 * pixel_y / float(height)) * math.tan(
        math.radians(CAMERA_VERTICAL_FOV_DEG) * 0.5
    )
    grid_x, grid_y = torch.meshgrid(normalized_x, normalized_y, indexing="xy")
    directions = (
        basis[:, None, None, 0]
        + grid_x[None, ..., None] * basis[:, None, None, 1]
        + grid_y[None, ..., None] * basis[:, None, None, 2]
    )
    return F.normalize(directions, p=2.0, dim=-1)


def _soft_occupied_probability_v4(
    hazard_logits: torch.Tensor,
    within_bin_offset_m: torch.Tensor,
    camera_origin_body_m: torch.Tensor,
    camera_basis_body_fru: torch.Tensor,
    *,
    output_shape: tuple[int, int],
    pixel_ray_chunk_size: int,
) -> torch.Tensor:
    """Mass-conserving bilinear ray splat followed by a union across rays."""

    if tuple(within_bin_offset_m.shape) != tuple(hazard_logits.shape):
        raise ValueError("pixel offsets must match pixel hazards")
    if hazard_logits.ndim != 4 or not hazard_logits.is_floating_point():
        raise ValueError("pixel hazards must have shape (B,D,Hray,Wray)")
    if not within_bin_offset_m.is_floating_point():
        raise ValueError("pixel offsets must be floating point")
    if hazard_logits.device != within_bin_offset_m.device:
        raise ValueError("pixel hazards and offsets must share a device")
    batch, depth_count, ray_height, ray_width = hazard_logits.shape
    rows, columns = int(output_shape[0]), int(output_shape[1])
    if rows <= 0 or columns <= 0:
        raise ValueError("output_shape dimensions must be positive")
    chunk_size = int(pixel_ray_chunk_size)
    if chunk_size <= 0:
        raise ValueError("pixel_ray_chunk_size must be positive")
    if depth_count != DEPTH_BIN_COUNT:
        raise ValueError("soft V4 raster requires the registered 64 depth bins")
    origin, basis = _validate_camera_calibration(
        camera_origin_body_m,
        camera_basis_body_fru,
        batch=batch,
        dtype=hazard_logits.dtype,
        device=hazard_logits.device,
    )
    directions = calibrated_pixel_ray_directions_torch_v4(
        basis,
        ray_shape=(ray_height, ray_width),
        dtype=hazard_logits.dtype,
    ).reshape(batch, ray_height * ray_width, 3)
    ordered = ordered_obstacle_first_hit_log_probabilities_v4(hazard_logits)
    hit_probability = ordered.hit.exp().reshape(
        batch,
        depth_count,
        ray_height * ray_width,
    )
    offset = within_bin_offset_m.reshape(
        batch,
        depth_count,
        ray_height * ray_width,
    )
    bin_centers = DEPTH_NEAR_EDGE_M + (
        torch.arange(
            depth_count,
            dtype=hazard_logits.dtype,
            device=hazard_logits.device,
        )
        + 0.5
    ) * DEPTH_BIN_SIZE_M
    cell_count = rows * columns
    log_no_occupied = hazard_logits.new_zeros(batch, cell_count)
    epsilon = torch.finfo(hazard_logits.dtype).eps
    ray_count = ray_height * ray_width

    for start in range(0, ray_count, chunk_size):
        stop = min(start + chunk_size, ray_count)
        local_rays = stop - start
        distance = bin_centers[None, :, None] + offset[:, :, start:stop]
        point_xy = origin[:, None, None, :2] + (
            distance[..., None]
            * directions[:, None, start:stop, :2]
        )
        row_coordinate = (
            point_xy[..., 0] - OUTPUT_FORWARD_MIN_EDGE_M
        ) / OUTPUT_CELL_SIZE_M - 0.5
        column_coordinate = (
            point_xy[..., 1] - OUTPUT_LEFT_MIN_EDGE_M
        ) / OUTPUT_CELL_SIZE_M - 0.5
        inside_extent = (
            (row_coordinate >= -0.5)
            & (row_coordinate <= rows - 0.5)
            & (column_coordinate >= -0.5)
            & (column_coordinate <= columns - 0.5)
        )
        row_low = torch.floor(row_coordinate).to(dtype=torch.long)
        column_low = torch.floor(column_coordinate).to(dtype=torch.long)
        row_fraction = row_coordinate - row_low.to(dtype=row_coordinate.dtype)
        column_fraction = (
            column_coordinate - column_low.to(dtype=column_coordinate.dtype)
        )
        candidates: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []
        normalizer = torch.zeros_like(row_coordinate)
        for row_delta in (0, 1):
            row_index = row_low + row_delta
            row_weight = row_fraction if row_delta else 1.0 - row_fraction
            for column_delta in (0, 1):
                column_index = column_low + column_delta
                column_weight = (
                    column_fraction if column_delta else 1.0 - column_fraction
                )
                valid = (
                    inside_extent
                    & (row_index >= 0)
                    & (row_index < rows)
                    & (column_index >= 0)
                    & (column_index < columns)
                )
                raw_weight = row_weight * column_weight
                normalizer = normalizer + torch.where(
                    valid,
                    raw_weight,
                    torch.zeros_like(raw_weight),
                )
                candidates.append((row_index, column_index, valid))

        per_ray_cell = hazard_logits.new_zeros(batch * local_rays * cell_count)
        ray_group = (
            torch.arange(batch, device=hazard_logits.device)[:, None, None]
            * local_rays
            + torch.arange(local_rays, device=hazard_logits.device)[None, None, :]
        ) * cell_count
        candidate_index = 0
        for row_delta in (0, 1):
            row_weight = row_fraction if row_delta else 1.0 - row_fraction
            for column_delta in (0, 1):
                column_weight = (
                    column_fraction if column_delta else 1.0 - column_fraction
                )
                row_index, column_index, valid = candidates[candidate_index]
                candidate_index += 1
                raw_weight = row_weight * column_weight
                weight = torch.where(
                    valid,
                    raw_weight / normalizer.clamp_min(epsilon),
                    torch.zeros_like(raw_weight),
                )
                contribution = hit_probability[:, :, start:stop] * weight
                flat_index = (
                    ray_group
                    + row_index.clamp(0, rows - 1) * columns
                    + column_index.clamp(0, columns - 1)
                )
                per_ray_cell = per_ray_cell.scatter_add(
                    0,
                    flat_index[valid],
                    contribution[valid],
                )
        per_ray_cell = per_ray_cell.reshape(batch, local_rays, cell_count)
        per_ray_cell = per_ray_cell.clamp(min=0.0, max=1.0 - epsilon)
        log_no_occupied = log_no_occupied + torch.log1p(
            -per_ray_cell
        ).sum(dim=1)
    return -torch.expm1(log_no_occupied).reshape(batch, rows, columns)


def soft_rasterize_observable_camera_ray_evidence_v4(
    raw_output: ObservableCameraRayEvidenceV4RawOutput,
    *,
    camera_origin_body_m: torch.Tensor,
    camera_basis_body_fru: torch.Tensor,
    output_shape: tuple[int, int] = OUTPUT_SHAPE,
    pixel_ray_chunk_size: int = DEFAULT_PIXEL_RAY_CHUNK_SIZE,
) -> SoftObservableCameraRayRasterV4:
    """Build a differentiable OCCUPIED-first physical-evidence raster."""

    if not isinstance(raw_output, ObservableCameraRayEvidenceV4RawOutput):
        raise TypeError("raw_output must be ObservableCameraRayEvidenceV4RawOutput")
    rows, columns = int(output_shape[0]), int(output_shape[1])
    ground_logits = raw_output.ground_clear_to_target_logits
    expected_source_shape = (2 * rows, 2 * columns, GROUND_SUPPORT_COUNT)
    if tuple(ground_logits.shape[1:]) != expected_source_shape:
        raise ValueError(
            "ground source shape must be exactly twice the output lattice"
        )
    support_valid = raw_output.ground_query_in_frustum
    source_valid = support_valid.all(dim=-1)
    log_source_free = F.logsigmoid(ground_logits).sum(dim=-1)
    log_source_free = torch.where(
        source_valid,
        log_source_free,
        torch.full_like(log_source_free, -torch.inf),
    )
    source_free_probability = log_source_free.exp()
    grouped_log_source_free = log_source_free.reshape(
        ground_logits.shape[0],
        rows,
        2,
        columns,
        2,
    )
    log_free_given_not_occupied = grouped_log_source_free.sum(dim=4).sum(dim=2)
    free_given_not_occupied = log_free_given_not_occupied.exp()
    occupied_probability = _soft_occupied_probability_v4(
        raw_output.pixel_first_hit_hazard_logits,
        raw_output.pixel_within_bin_offset_m,
        camera_origin_body_m,
        camera_basis_body_fru,
        output_shape=(rows, columns),
        pixel_ray_chunk_size=pixel_ray_chunk_size,
    )
    not_occupied = 1.0 - occupied_probability
    free_probability = not_occupied * free_given_not_occupied
    unknown_probability = not_occupied * (1.0 - free_given_not_occupied)
    class_probabilities = torch.stack(
        (unknown_probability, free_probability, occupied_probability),
        dim=1,
    )
    return SoftObservableCameraRayRasterV4(
        source_free_probability=source_free_probability,
        free_given_not_occupied_probability=free_given_not_occupied,
        occupied_probability=occupied_probability,
        class_probabilities=class_probabilities,
    )


def hierarchical_raster_cross_entropy_v4(
    raster: SoftObservableCameraRayRasterV4,
    target_labels: torch.Tensor,
) -> HierarchicalRasterCrossEntropyV4:
    """Train OCCUPIED first, then FREE versus UNKNOWN where not occupied."""

    if not isinstance(target_labels, torch.Tensor) or target_labels.ndim != 3:
        raise ValueError("target_labels must have shape (B,H,W)")
    if tuple(target_labels.shape) != tuple(raster.occupied_probability.shape):
        raise ValueError("raster labels do not match soft raster shape")
    if target_labels.device != raster.occupied_probability.device:
        raise ValueError("raster labels and probabilities must share a device")
    if target_labels.is_floating_point() or target_labels.dtype == torch.bool:
        raise ValueError("raster labels must use an integer class dtype")
    supported = (
        (target_labels == UNKNOWN_CLASS)
        | (target_labels == FREE_CLASS)
        | (target_labels == OCCUPIED_CLASS)
    )
    if not bool(supported.all().item()):
        raise ValueError("raster labels contain an unsupported class")
    epsilon = torch.finfo(raster.occupied_probability.dtype).eps
    occupied_target = target_labels == OCCUPIED_CLASS
    occupied_probability = raster.occupied_probability.clamp(
        min=epsilon,
        max=1.0 - epsilon,
    )
    occupied_loss, rest_count, occupied_count = _balanced_probability_bce_v4(
        occupied_probability,
        occupied_target,
        torch.ones_like(occupied_target),
    )
    non_occupied = ~occupied_target
    conditional_free = raster.free_given_not_occupied_probability.clamp(
        min=epsilon,
        max=1.0 - epsilon,
    )
    free_loss, unknown_count, free_count = _balanced_probability_bce_v4(
        conditional_free,
        target_labels == FREE_CLASS,
        non_occupied,
    )
    return HierarchicalRasterCrossEntropyV4(
        total=0.5 * occupied_loss + 0.5 * free_loss,
        occupied=occupied_loss,
        free_given_not_occupied=free_loss,
        occupied_count=occupied_count,
        rest_count=rest_count,
        free_count=free_count,
        unknown_count=unknown_count,
    )


def _balanced_probability_bce_v4(
    probability: torch.Tensor,
    positive_target: torch.Tensor,
    valid_mask: torch.Tensor,
) -> tuple[torch.Tensor, int, int]:
    """Average each nonempty binary state once and retain empty gradients."""

    group_losses = []
    counts = []
    for state in (False, True):
        mask = valid_mask & (positive_target == state)
        count = int(mask.sum().item())
        counts.append(count)
        if count:
            group_losses.append(
                F.binary_cross_entropy(
                    probability[mask],
                    positive_target[mask].to(dtype=probability.dtype),
                    reduction="mean",
                )
            )
    loss = (
        torch.stack(group_losses).mean()
        if group_losses
        else probability.sum() * 0.0
    )
    return loss, counts[0], counts[1]


__all__ = [
    "DEFAULT_GROUND_DISTANCE_BIN_EDGES_M",
    "DEFAULT_PIXEL_RAY_CHUNK_SIZE",
    "HierarchicalRasterCrossEntropyV4",
    "ObservableCameraRayEvidenceV4Targets",
    "OrderedObstacleFirstHitNLLBreakdownV4",
    "SoftObservableCameraRayRasterV4",
    "balanced_ground_clear_bce_v4",
    "calibrated_pixel_ray_directions_torch_v4",
    "derive_observable_camera_ray_evidence_v4_targets",
    "hierarchical_raster_cross_entropy_v4",
    "in_range_pixel_offset_smooth_l1_v4",
    "ordered_obstacle_first_hit_nll_v4",
    "ordered_obstacle_first_hit_nll_breakdown_v4",
    "soft_rasterize_observable_camera_ray_evidence_v4",
]
