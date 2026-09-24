"""Pure fit-frame decomposition for perfect-ray target mismatches.

The module consumes already-authorized in-memory geometry.  It performs no
file I/O and imports the V2 perfect-ray implementation under a neutral module
name.  Physical-prior and collision-veto discrepancies are classified by
independent source predicates while preserving the exact V1/V2 raster arms.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import math
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

import numpy as np


V2_CORE_PATH = Path(__file__).with_name("go2_perfect_camera_ray_field_audit_v2.py")
FRAME_SCHEMA = "lewm_go2_perfect_ray_mismatch_decomposition_frame_v1"
PHYSICAL_CATEGORIES = (
    "outside_domain_or_world_boundary",
    "nonfree_terrain_or_surface",
    "matched_collision_rendered_source_footprint",
    "residual",
)
COLLISION_CATEGORIES = (
    "recovered_by_native_stride1_lattice_absent_from_registered_stride2",
    "no_native_pixel_first_surface_witness",
)
TERRAIN_KINDS = frozenset(
    {
        "ramp",
        "step",
        "slick_patch",
        "terrain",
        "slope",
        "platform",
        "raised_platform",
    }
)


def _load_v2_core() -> Any:
    name = "go2_perfect_camera_ray_field_audit_v2_for_decomposition"
    existing = sys.modules.get(name)
    if existing is not None:
        return existing
    spec = importlib.util.spec_from_file_location(name, V2_CORE_PATH)
    if spec is None or spec.loader is None:
        raise ImportError("cannot load the V2 perfect-ray audit core")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


v2 = _load_v2_core()
v1 = v2.v1
UNKNOWN_CLASS = v1.UNKNOWN_CLASS
FREE_CLASS = v1.FREE_CLASS
OCCUPIED_CLASS = v1.OCCUPIED_CLASS


def _canonical_json_sha256(value: Any) -> str:
    raw = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _box_from_mapping(value: Mapping[str, Any]) -> Any:
    required = {"center_xyz_m", "size_xyz_m", "roll_rad", "pitch_rad", "yaw_rad"}
    if not required.issubset(value):
        raise ValueError("box mapping is incomplete")
    return v1.OrientedBox(
        center_xyz_m=tuple(value["center_xyz_m"]),
        size_xyz_m=tuple(value["size_xyz_m"]),
        roll_rad=float(value["roll_rad"]),
        pitch_rad=float(value["pitch_rad"]),
        yaw_rad=float(value["yaw_rad"]),
    )


def _camera_from_mapping(value: Mapping[str, Any]) -> Any:
    return v1.CameraRaySpec(
        position_xyz_m=tuple(value["position_xyz_m"]),
        lookat_xyz_m=tuple(value["lookat_xyz_m"]),
        up_xyz=tuple(value["up_xyz"]),
        horizontal_fov_deg=float(value["horizontal_fov_deg"]),
        vertical_fov_deg=float(value["vertical_fov_deg"]),
        near_m=float(value["near_m"]),
        ground_plane_z_m=float(value["ground_plane_z_m"]),
        image_width_px=int(value["image_width_px"]),
        image_height_px=int(value["image_height_px"]),
        obstacle_ray_stride_px=int(value["obstacle_ray_stride_px"]),
    )


def _output_grid_from_mapping(value: Mapping[str, Any]) -> Any:
    return v1.OutputGridSpec(
        rows=int(value["rows"]),
        cols=int(value["cols"]),
        cell_size_m=float(value["cell_size_m"]),
        forward_min_edge_m=float(value["forward_min_edge_m"]),
        left_min_edge_m=float(value["left_min_edge_m"]),
    )


def _support_indices(
    *,
    output_index: tuple[int, int],
    output_x: np.ndarray,
    output_y: np.ndarray,
    physical_x: np.ndarray,
    physical_y: np.ndarray,
    output_yaw_rad: float,
    physical_cell_size_m: float,
    output_cell_size_m: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return exact source indices admitted by V1 rotated-square aggregation."""

    source_half = 0.5 * float(physical_cell_size_m)
    output_half = 0.5 * float(output_cell_size_m)
    yaw = float(output_yaw_rad)
    cos_yaw = math.cos(yaw)
    sin_yaw = math.sin(yaw)
    output_u = np.asarray((cos_yaw, sin_yaw), dtype=np.float64)
    output_v = np.asarray((-sin_yaw, cos_yaw), dtype=np.float64)
    world_extent = output_half * (abs(cos_yaw) + abs(sin_yaw)) + source_half
    center_x = float(output_x[output_index])
    center_y = float(output_y[output_index])
    x_candidates = np.flatnonzero(
        np.abs(physical_x - center_x) <= world_extent + 1e-12
    )
    y_candidates = np.flatnonzero(
        np.abs(physical_y - center_y) <= world_extent + 1e-12
    )
    if x_candidates.size == 0 or y_candidates.size == 0:
        return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64)
    candidate_x, candidate_y = np.meshgrid(
        physical_x[x_candidates], physical_y[y_candidates], indexing="ij"
    )
    dx = candidate_x - center_x
    dy = candidate_y - center_y
    along_u = dx * output_u[0] + dy * output_u[1]
    along_v = dx * output_v[0] + dy * output_v[1]
    intersects = (
        (
            np.abs(along_u)
            <= output_half
            + source_half * (abs(output_u[0]) + abs(output_u[1]))
            + 1e-12
        )
        & (
            np.abs(along_v)
            <= output_half
            + source_half * (abs(output_v[0]) + abs(output_v[1]))
            + 1e-12
        )
        & (np.abs(dx) <= world_extent + 1e-12)
        & (np.abs(dy) <= world_extent + 1e-12)
    )
    local_rows, local_cols = np.nonzero(intersects)
    return x_candidates[local_rows], y_candidates[local_cols]


def _points_inside_planning_box_xy(
    x: np.ndarray,
    y: np.ndarray,
    box: Any,
) -> np.ndarray:
    """Match the zero-inflation planning grid's yaw-oriented 2D box test."""

    center_x, center_y, _center_z = box.center_xyz_m
    size_x, size_y, _size_z = box.size_xyz_m
    dx = np.asarray(x, dtype=np.float64) - float(center_x)
    dy = np.asarray(y, dtype=np.float64) - float(center_y)
    cos_yaw = math.cos(-float(box.yaw_rad))
    sin_yaw = math.sin(-float(box.yaw_rad))
    local_x = cos_yaw * dx - sin_yaw * dy
    local_y = sin_yaw * dx + cos_yaw * dy
    return (
        (np.abs(local_x) <= 0.5 * float(size_x) + 1e-7)
        & (np.abs(local_y) <= 0.5 * float(size_y) + 1e-7)
    )


def _physical_blocker_flags(
    *,
    blocker_x: np.ndarray,
    blocker_y: np.ndarray,
    blocker_inside_grid: np.ndarray,
    physical_cell_size_m: float,
    world_bounds_xy_m: Sequence[Sequence[float]],
    collision_records: Sequence[Mapping[str, Any]],
) -> dict[str, bool]:
    if len(world_bounds_xy_m) != 2 or any(len(pair) != 2 for pair in world_bounds_xy_m):
        raise ValueError("world bounds must contain low/high XY pairs")
    (x_low, y_low), (x_high, y_high) = (
        tuple(map(float, pair)) for pair in world_bounds_xy_m
    )
    half_cell = 0.5 * float(physical_cell_size_m)
    square_outside_world = (
        (blocker_x - half_cell < x_low - 1e-12)
        | (blocker_x + half_cell > x_high + 1e-12)
        | (blocker_y - half_cell < y_low - 1e-12)
        | (blocker_y + half_cell > y_high + 1e-12)
    )
    flags = {
        "outside_grid": bool(np.any(~blocker_inside_grid)),
        "source_square_outside_world_bounds": bool(np.any(square_outside_world)),
        "wall_source_footprint": False,
        "terrain_surface": False,
        "collision_or_rendered": False,
        "matched_rendered": False,
        "collision_only": False,
        "unattributed_inside_nonfree": False,
    }
    inside_indices = np.flatnonzero(blocker_inside_grid)
    attributed = np.zeros(inside_indices.size, dtype=bool)
    if inside_indices.size:
        inside_x = blocker_x[inside_indices]
        inside_y = blocker_y[inside_indices]
        for record in collision_records:
            box = _box_from_mapping(record["box"])
            contains = _points_inside_planning_box_xy(inside_x, inside_y, box)
            if not np.any(contains):
                continue
            attributed |= contains
            group = str(record.get("group", ""))
            kind = str(record.get("kind", "")).lower()
            if group == "wall":
                flags["wall_source_footprint"] = True
                flags["collision_or_rendered"] = True
            elif kind in TERRAIN_KINDS:
                flags["terrain_surface"] = True
            else:
                flags["collision_or_rendered"] = True
            if record.get("rendered_index") is None:
                flags["collision_only"] = True
            else:
                flags["matched_rendered"] = True
        flags["unattributed_inside_nonfree"] = bool(np.any(~attributed))
    return flags


def _physical_category(flags: Mapping[str, bool]) -> str:
    if flags["outside_grid"] or flags["source_square_outside_world_bounds"]:
        return "outside_domain_or_world_boundary"
    if flags["terrain_surface"]:
        return "nonfree_terrain_or_surface"
    if (
        flags["collision_or_rendered"]
        and flags["matched_rendered"]
        and not flags["collision_only"]
    ):
        return "matched_collision_rendered_source_footprint"
    return "residual"


def _native_pixel_first_hits(
    camera: Any,
    rendered_boxes: Sequence[Any],
) -> tuple[np.ndarray, np.ndarray]:
    """Return stride-1 native first-hit XY and nearest rendered-box indices."""

    position, forward, right, up, tan_h, tan_v = v1._camera_basis(camera)
    width = int(camera.image_width_px)
    height = int(camera.image_height_px)
    pixel_x = np.arange(width, dtype=np.float64) + 0.5
    pixel_y = np.arange(height, dtype=np.float64) + 0.5
    normalized_x = (2.0 * pixel_x / float(width) - 1.0) * tan_h
    normalized_y = (1.0 - 2.0 * pixel_y / float(height)) * tan_v
    grid_x, grid_y = np.meshgrid(normalized_x, normalized_y, indexing="xy")
    directions = (
        forward[None, None, :]
        + grid_x[:, :, None] * right[None, None, :]
        + grid_y[:, :, None] * up[None, None, :]
    )
    directions /= np.linalg.norm(directions, axis=2, keepdims=True)
    flat = directions.reshape(-1, 3)
    nearest = np.full(flat.shape[0], np.inf, dtype=np.float64)
    nearest_index = np.full(flat.shape[0], -1, dtype=np.int32)
    for box_index, box in enumerate(rendered_boxes):
        entry = v1._ray_box_entry_distances(position, flat, box)
        closer = entry < nearest - 1e-10
        nearest[closer] = entry[closer]
        nearest_index[closer] = box_index
    valid = np.isfinite(nearest) & (nearest > float(camera.near_m))
    hits = position[None, :] + flat[valid] * nearest[valid, None]
    return hits[:, :2], nearest_index[valid]


def _hits_in_output_cell(
    hits_xy: np.ndarray,
    *,
    center_x: float,
    center_y: float,
    output_yaw_rad: float,
    output_cell_size_m: float,
) -> np.ndarray:
    if hits_xy.size == 0:
        return np.zeros(0, dtype=bool)
    dx = hits_xy[:, 0] - float(center_x)
    dy = hits_xy[:, 1] - float(center_y)
    cos_yaw = math.cos(float(output_yaw_rad))
    sin_yaw = math.sin(float(output_yaw_rad))
    forward = cos_yaw * dx + sin_yaw * dy
    left = -sin_yaw * dx + cos_yaw * dy
    half = 0.5 * float(output_cell_size_m)
    return (
        (np.abs(forward) <= half + 1e-12)
        & (np.abs(left) <= half + 1e-12)
    )


def _mask_record(coords: Sequence[Sequence[int]]) -> dict[str, Any]:
    canonical = [[int(row), int(column)] for row, column in coords]
    return {
        "count": len(canonical),
        "row_column_identities_sha256": _canonical_json_sha256(canonical),
        "sample": canonical[:32],
    }


def decompose_frame(
    *,
    authoritative_labels: np.ndarray,
    supervision_mask: np.ndarray,
    frame_key: Mapping[str, Any],
    camera: Mapping[str, Any],
    rendered_boxes: Sequence[Mapping[str, Any]],
    collision_records: Sequence[Mapping[str, Any]],
    base_xy_yaw: Sequence[float],
    physical_free_mask: np.ndarray,
    physical_origin_xy_m: Sequence[float],
    physical_cell_size_m: float,
    world_bounds_xy_m: Sequence[Sequence[float]],
    rendered_collision_parity_complete: bool,
    output_grid: Mapping[str, Any],
) -> dict[str, Any]:
    """Decompose one frame while reproducing all three V2 arms exactly."""

    target = np.asarray(authoritative_labels)
    supervision = np.asarray(supervision_mask)
    if target.shape != (64, 64) or target.dtype != np.dtype(np.uint8):
        raise ValueError("authoritative labels must be uint8 [64, 64]")
    if supervision.shape != (64, 64) or supervision.dtype != np.dtype(bool):
        raise ValueError("supervision must be bool [64, 64]")
    if not np.all(supervision):
        raise ValueError("decomposition requires full-grid supervision")
    if rendered_collision_parity_complete is not True:
        raise ValueError("decomposition requires exact rendered/collision box parity")
    camera_spec = _camera_from_mapping(camera)
    rendered = tuple(_box_from_mapping(box) for box in rendered_boxes)
    collision = tuple(_box_from_mapping(record["box"]) for record in collision_records)
    grid_spec = _output_grid_from_mapping(output_grid)
    reconstruction = v2.reconstruct_frame_from_perfect_rays(
        camera=camera_spec,
        rendered_obstacle_boxes=rendered,
        collision_obstacle_boxes=collision,
        base_xy_yaw=base_xy_yaw,
        physical_free_mask=physical_free_mask,
        physical_origin_xy_m=physical_origin_xy_m,
        physical_cell_size_m=physical_cell_size_m,
        output_grid=grid_spec,
    )
    if not np.array_equal(reconstruction.contract_labels, target):
        raise ValueError("contract-assisted reconstruction is not authoritative")
    legacy = reconstruction.collision_vetoed_ray_only_labels
    observable = reconstruction.observable_ray_only_labels
    physical_mask = (target == UNKNOWN_CLASS) & (legacy == FREE_CLASS)
    collision_mask = (observable == FREE_CLASS) & (legacy == UNKNOWN_CLASS)
    if np.any(physical_mask & reconstruction.collision_overlap):
        raise ValueError("physical-prior mask unexpectedly overlaps output collision veto")
    expected_collision_mask = (
        (reconstruction.v1_rasterization.ray_only_pre_veto_labels == FREE_CLASS)
        & reconstruction.collision_overlap
    )
    if not np.array_equal(collision_mask, expected_collision_mask):
        raise ValueError("collision-veto delta does not equal pre-veto FREE overlap")
    if np.any(collision_mask & (target != UNKNOWN_CLASS)):
        raise ValueError("collision veto changed a non-UNKNOWN authoritative cell")
    if int(np.count_nonzero(target != observable)) != int(
        np.count_nonzero(physical_mask) + np.count_nonzero(collision_mask)
    ):
        raise ValueError("V2 mismatch does not partition into prior and veto masks")

    output_x, output_y = v1.output_world_centers(base_xy_yaw, grid_spec)
    window = v1.physical_window_for_output(
        physical_free_mask=physical_free_mask,
        physical_origin_xy_m=physical_origin_xy_m,
        physical_cell_size_m=physical_cell_size_m,
        output_world_x_m=output_x,
        output_world_y_m=output_y,
        output_cell_size_m=grid_spec.cell_size_m,
    )
    field = v1.build_perfect_camera_ray_field(
        camera=camera_spec,
        rendered_obstacle_boxes=rendered,
        physical_x_centers_m=window.x_centers_m,
        physical_y_centers_m=window.y_centers_m,
        physical_cell_size_m=window.cell_size_m,
    )
    visible_source = np.all(field.ground_support_visible, axis=2)
    origin_x, origin_y = map(float, physical_origin_xy_m)
    cell_size = float(physical_cell_size_m)
    full_free = np.asarray(physical_free_mask, dtype=bool)

    physical_identities: dict[str, list[list[int]]] = {
        category: [] for category in PHYSICAL_CATEGORIES
    }
    physical_flag_counts = {
        name: 0
        for name in (
            "outside_grid",
            "source_square_outside_world_bounds",
            "wall_source_footprint",
            "terrain_surface",
            "collision_or_rendered",
            "matched_rendered",
            "collision_only",
            "unattributed_inside_nonfree",
        )
    }
    for raw_row, raw_column in np.argwhere(physical_mask):
        row, column = int(raw_row), int(raw_column)
        source_rows, source_cols = _support_indices(
            output_index=(row, column),
            output_x=output_x,
            output_y=output_y,
            physical_x=window.x_centers_m,
            physical_y=window.y_centers_m,
            output_yaw_rad=float(base_xy_yaw[2]),
            physical_cell_size_m=window.cell_size_m,
            output_cell_size_m=grid_spec.cell_size_m,
        )
        blockers = visible_source[source_rows, source_cols] & ~window.physical_free_mask[
            source_rows, source_cols
        ]
        blocker_rows = source_rows[blockers]
        blocker_cols = source_cols[blockers]
        blocker_x = window.x_centers_m[blocker_rows]
        blocker_y = window.y_centers_m[blocker_cols]
        full_ix = np.rint((blocker_x - origin_x) / cell_size - 0.5).astype(np.int64)
        full_iy = np.rint((blocker_y - origin_y) / cell_size - 0.5).astype(np.int64)
        aligned_x = origin_x + (full_ix.astype(np.float64) + 0.5) * cell_size
        aligned_y = origin_y + (full_iy.astype(np.float64) + 0.5) * cell_size
        if not (
            np.allclose(blocker_x, aligned_x, rtol=0.0, atol=1e-10)
            and np.allclose(blocker_y, aligned_y, rtol=0.0, atol=1e-10)
        ):
            raise ValueError("physical source axes are not integer-aligned to the grid")
        inside = (
            (full_ix >= 0)
            & (full_ix < full_free.shape[0])
            & (full_iy >= 0)
            & (full_iy < full_free.shape[1])
        )
        if np.any(
            full_free[
                full_ix[inside].astype(np.int64),
                full_iy[inside].astype(np.int64),
            ]
        ):
            raise ValueError("causal inside-grid source support is unexpectedly FREE")
        if blocker_rows.size == 0:
            raise ValueError("physical-prior mismatch has no causal source support")
        flags = _physical_blocker_flags(
            blocker_x=blocker_x,
            blocker_y=blocker_y,
            blocker_inside_grid=inside,
            physical_cell_size_m=cell_size,
            world_bounds_xy_m=world_bounds_xy_m,
            collision_records=collision_records,
        )
        for name, active in flags.items():
            physical_flag_counts[name] += int(bool(active))
        physical_identities[_physical_category(flags)].append([row, column])

    stride1_hits_xy, _stride1_box_indices = _native_pixel_first_hits(
        camera_spec, rendered
    )
    collision_identities: dict[str, list[list[int]]] = {
        category: [] for category in COLLISION_CATEGORIES
    }
    collision_flag_counts = {
        "stride1_first_hit_in_cell": 0,
    }
    for raw_row, raw_column in np.argwhere(collision_mask):
        row, column = int(raw_row), int(raw_column)
        in_cell = _hits_in_output_cell(
            stride1_hits_xy,
            center_x=float(output_x[row, column]),
            center_y=float(output_y[row, column]),
            output_yaw_rad=float(base_xy_yaw[2]),
            output_cell_size_m=grid_spec.cell_size_m,
        )
        if bool(np.any(in_cell)):
            category = (
                "recovered_by_native_stride1_lattice_absent_from_registered_stride2"
            )
        else:
            category = "no_native_pixel_first_surface_witness"
        collision_flag_counts["stride1_first_hit_in_cell"] += int(
            bool(np.any(in_cell))
        )
        collision_identities[category].append([row, column])

    if sum(map(len, physical_identities.values())) != int(np.count_nonzero(physical_mask)):
        raise AssertionError("physical categories do not partition their mask")
    if sum(map(len, collision_identities.values())) != int(np.count_nonzero(collision_mask)):
        raise AssertionError("collision categories do not partition their mask")
    return {
        "schema": FRAME_SCHEMA,
        "frame_key": dict(frame_key),
        "authoritative_labels_sha256": hashlib.sha256(
            target.tobytes(order="C")
        ).hexdigest(),
        "physical_prior_mismatch_cell_count": int(np.count_nonzero(physical_mask)),
        "collision_veto_delta_cell_count": int(np.count_nonzero(collision_mask)),
        "observable_ray_only_mismatch_cell_count": int(
            np.count_nonzero(target != observable)
        ),
        "physical_prior_categories": {
            name: _mask_record(physical_identities[name]) for name in PHYSICAL_CATEGORIES
        },
        "physical_prior_overlapping_evidence_flags": physical_flag_counts,
        "collision_veto_categories": {
            name: _mask_record(collision_identities[name])
            for name in COLLISION_CATEGORIES
        },
        "collision_veto_overlapping_evidence_flags": collision_flag_counts,
        "stride1_first_hit_count": int(stride1_hits_xy.shape[0]),
        "_private_identities": {
            "physical_prior": physical_identities,
            "collision_veto": collision_identities,
        },
    }


__all__ = [
    "COLLISION_CATEGORIES",
    "FRAME_SCHEMA",
    "PHYSICAL_CATEGORIES",
    "TERRAIN_KINDS",
    "V2_CORE_PATH",
    "decompose_frame",
]
