"""Rolling local occupancy built from ego-camera depth and odometry."""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from lewm.planning.local_obstacles import LocalObstacleContract, XY


@dataclass(frozen=True)
class DepthLocalObstacleConfig:
    cell_size_m: float = 0.05
    # Genesis camera ``fov`` is the VERTICAL field of view; horizontal is derived
    # from the image aspect. (Treating it as horizontal underestimates vertical
    # ray angles and lifts floor returns into the obstacle height band.)
    vertical_fov_deg: float = 78.323
    max_range_m: float = 3.0
    min_range_m: float = 0.08
    sample_stride_px: int = 8
    occupied_inflation_m: float = 0.22
    free_query_radius_m: float = 0.08
    robot_free_radius_m: float = 0.20
    min_obstacle_height_m: float = 0.08
    max_obstacle_height_m: float = 1.50
    max_age_frames: int = 12
    unknown_is_free: bool = False
    debug_capture: bool = False


class DepthLocalObstacleModel:
    """Conservative rolling 2-D obstacle map from depth observations.

    Queries and camera poses share a controller/odometry frame. The model does
    not access scene geometry. Call :meth:`update` once per fresh ego-depth
    observation before asking local trajectory-safety questions.
    """

    def __init__(
        self,
        config: DepthLocalObstacleConfig | None = None,
        *,
        odometry_detail: str = "onboard odometry",
        odometry_deployment_valid: bool = True,
    ) -> None:
        self.config = config or DepthLocalObstacleConfig()
        self._contract = LocalObstacleContract(
            source="ego_depth_local_occupancy_v0",
            deployment_valid=bool(odometry_deployment_valid),
            query_frame="controller_odometry_xy",
            detail=(
                "ego-camera depth projected into rolling local occupancy using "
                f"{odometry_detail}; unknown space is "
                f"{'free' if self.config.unknown_is_free else 'blocked'}"
            ),
        )
        self.reset()

    @property
    def contract(self) -> LocalObstacleContract:
        return self._contract

    def reset(self) -> None:
        self._frame_id = 0
        self._free: dict[tuple[int, int], int] = {}
        self._occupied: dict[tuple[int, int], int] = {}
        self._robot_xy: XY | None = None
        self._valid_depth_points = 0
        self._obstacle_points = 0
        self._low_points = 0
        self._high_points = 0
        self._depth_range_m: tuple[float, float] | None = None
        self._endpoint_height_range_m: tuple[float, float] | None = None
        # Diagnostic-only is_free rejection tallies (no behaviour effect).
        self._query_total = 0
        self._free_occupied_reject = 0
        self._free_unknown_reject = 0
        self._free_robot_hit = 0
        self._free_cell_hit = 0
        # Diagnostic per-point projection capture (row, col, distance, x, y, height, is_obstacle).
        self._debug_points: list[tuple[int, int, float, float, float, float, bool]] = []

    def _cell(self, xy: XY) -> tuple[int, int]:
        scale = float(self.config.cell_size_m)
        return (
            int(math.floor(float(xy[0]) / scale)),
            int(math.floor(float(xy[1]) / scale)),
        )

    def _cell_center(self, cell: tuple[int, int]) -> XY:
        scale = float(self.config.cell_size_m)
        return ((cell[0] + 0.5) * scale, (cell[1] + 0.5) * scale)

    def _mark_disk(self, store: dict[tuple[int, int], int], xy: XY, radius_m: float) -> None:
        cell = self._cell(xy)
        radius_cells = int(math.ceil(float(radius_m) / float(self.config.cell_size_m)))
        radius_sq = float(radius_m) ** 2
        for di in range(-radius_cells, radius_cells + 1):
            for dj in range(-radius_cells, radius_cells + 1):
                candidate = (cell[0] + di, cell[1] + dj)
                center = self._cell_center(candidate)
                if (center[0] - xy[0]) ** 2 + (center[1] - xy[1]) ** 2 <= radius_sq:
                    store[candidate] = self._frame_id

    def _mark_ray_free(self, origin_xy: XY, endpoint_xy: XY, stop_short_m: float) -> None:
        dx = float(endpoint_xy[0] - origin_xy[0])
        dy = float(endpoint_xy[1] - origin_xy[1])
        length = math.hypot(dx, dy)
        free_length = max(0.0, length - float(stop_short_m))
        if free_length <= 0.0:
            return
        steps = max(1, int(math.ceil(free_length / (0.5 * self.config.cell_size_m))))
        for index in range(steps + 1):
            distance = free_length * index / steps
            xy = (
                float(origin_xy[0] + dx * distance / max(length, 1e-8)),
                float(origin_xy[1] + dy * distance / max(length, 1e-8)),
            )
            cell = self._cell(xy)
            self._free[cell] = self._frame_id
            # A ray seen THROUGH a cell clears a stale occupied mark there
            # (standard occupancy semantics). Without this, a spurious close
            # return during a scan rotation stays occupied forever and intrudes
            # into a corridor that later frames observe as free. Current-frame
            # occupied endpoints are preserved (only older marks are cleared).
            if self._occupied.get(cell, self._frame_id) != self._frame_id:
                del self._occupied[cell]

    def _prune(self) -> None:
        oldest = self._frame_id - int(self.config.max_age_frames)
        self._free = {cell: age for cell, age in self._free.items() if age >= oldest}
        self._occupied = {
            cell: age for cell, age in self._occupied.items() if age >= oldest
        }

    def update(
        self,
        depth_m: np.ndarray,
        *,
        camera_position_xyz: np.ndarray,
        camera_rotation: np.ndarray,
        robot_xy: XY,
    ) -> None:
        """Fuse one depth image into the rolling controller-frame occupancy."""

        depth = np.asarray(depth_m, dtype=np.float32)
        depth = np.squeeze(depth)
        if depth.ndim != 2:
            raise ValueError(f"depth_m must squeeze to (H, W), got {depth.shape}")
        camera_position = np.asarray(camera_position_xyz, dtype=np.float32).reshape(3)
        rotation = np.asarray(camera_rotation, dtype=np.float32).reshape(3, 3)
        self._frame_id += 1
        self._robot_xy = (float(robot_xy[0]), float(robot_xy[1]))
        if self.config.debug_capture:
            self._debug_points = []
        self._prune()
        self._mark_disk(self._free, self._robot_xy, self.config.robot_free_radius_m)

        height, width = depth.shape
        tan_v = math.tan(math.radians(float(self.config.vertical_fov_deg)) * 0.5)
        tan_h = tan_v * float(width) / max(float(height), 1.0)
        stride = max(1, int(self.config.sample_stride_px))
        valid_count = 0
        obstacle_count = 0
        low_count = 0
        high_count = 0
        valid_depths: list[float] = []
        endpoint_heights: list[float] = []
        for row in range(stride // 2, height, stride):
            vertical = (2.0 * (row + 0.5) / height - 1.0) * tan_v
            for col in range(stride // 2, width, stride):
                distance = float(depth[row, col])
                if (
                    not math.isfinite(distance)
                    or distance < float(self.config.min_range_m)
                    or distance > float(self.config.max_range_m)
                ):
                    continue
                horizontal = (2.0 * (col + 0.5) / width - 1.0) * tan_h
                # Genesis depth is forward-axis camera depth, so use the
                # unnormalised pinhole ray [1, x/z, y/z]. Normalising it would
                # incorrectly lift floor returns and classify them as walls.
                local_direction = np.asarray(
                    [1.0, horizontal, -vertical],
                    dtype=np.float32,
                )
                endpoint = camera_position + distance * (rotation @ local_direction)
                endpoint_xy = (float(endpoint[0]), float(endpoint[1]))
                endpoint_height = float(endpoint[2])
                valid_depths.append(distance)
                endpoint_heights.append(endpoint_height)
                is_obstacle = (
                    float(self.config.min_obstacle_height_m)
                    <= endpoint_height
                    <= float(self.config.max_obstacle_height_m)
                )
                if self.config.debug_capture:
                    self._debug_points.append(
                        (row, col, distance, endpoint_xy[0], endpoint_xy[1],
                         endpoint_height, bool(is_obstacle)))
                self._mark_ray_free(
                    (float(camera_position[0]), float(camera_position[1])),
                    endpoint_xy,
                    self.config.occupied_inflation_m if is_obstacle else 0.0,
                )
                valid_count += 1
                if is_obstacle:
                    self._occupied[self._cell(endpoint_xy)] = self._frame_id
                    obstacle_count += 1
                elif endpoint_height < float(self.config.min_obstacle_height_m):
                    low_count += 1
                else:
                    high_count += 1
        self._valid_depth_points = valid_count
        self._obstacle_points = obstacle_count
        self._low_points = low_count
        self._high_points = high_count
        self._depth_range_m = (
            (min(valid_depths), max(valid_depths)) if valid_depths else None
        )
        self._endpoint_height_range_m = (
            (min(endpoint_heights), max(endpoint_heights)) if endpoint_heights else None
        )

    def _has_recent_cell_within(
        self,
        store: dict[tuple[int, int], int],
        xy: XY,
        radius_m: float,
    ) -> bool:
        cell = self._cell(xy)
        radius_cells = int(math.ceil(float(radius_m) / float(self.config.cell_size_m)))
        radius_sq = float(radius_m) ** 2
        for di in range(-radius_cells, radius_cells + 1):
            for dj in range(-radius_cells, radius_cells + 1):
                candidate = (cell[0] + di, cell[1] + dj)
                if candidate not in store:
                    continue
                center = self._cell_center(candidate)
                if (center[0] - xy[0]) ** 2 + (center[1] - xy[1]) ** 2 <= radius_sq:
                    return True
        return False

    def is_free(self, xy: XY) -> bool:
        point = (float(xy[0]), float(xy[1]))
        self._query_total += 1
        if self._has_recent_cell_within(
            self._occupied,
            point,
            self.config.occupied_inflation_m,
        ):
            self._free_occupied_reject += 1
            return False
        if (
            self._robot_xy is not None
            and math.dist(point, self._robot_xy) <= float(self.config.robot_free_radius_m)
        ):
            self._free_robot_hit += 1
            return True
        if self._has_recent_cell_within(
            self._free,
            point,
            self.config.free_query_radius_m,
        ):
            self._free_cell_hit += 1
            return True
        if not self.config.unknown_is_free:
            self._free_unknown_reject += 1
        return bool(self.config.unknown_is_free)

    def nearest_free(self, xy: XY, *, max_radius_m: float | None = None) -> XY | None:
        point = (float(xy[0]), float(xy[1]))
        if self.is_free(point):
            return point
        maximum = 0.5 if max_radius_m is None else float(max_radius_m)
        step = max(float(self.config.cell_size_m), 0.05)
        radius = step
        while radius <= maximum + 1e-9:
            samples = max(16, int(math.ceil(2.0 * math.pi * radius / step)))
            for index in range(samples):
                angle = 2.0 * math.pi * index / samples
                candidate = (
                    point[0] + radius * math.cos(angle),
                    point[1] + radius * math.sin(angle),
                )
                if self.is_free(candidate):
                    return candidate
            radius += step
        return None

    def debug_points(self) -> list[tuple[int, int, float, float, float, float, bool]]:
        return list(self._debug_points)

    def diagnostics(self) -> dict[str, int | bool | float | None]:
        depth_range = self._depth_range_m
        height_range = self._endpoint_height_range_m
        return {
            "frames_fused": int(self._frame_id),
            "recent_free_cells": len(self._free),
            "recent_occupied_cells": len(self._occupied),
            "last_valid_depth_points": int(self._valid_depth_points),
            "last_obstacle_points": int(self._obstacle_points),
            "last_low_points": int(self._low_points),
            "last_high_points": int(self._high_points),
            "last_depth_min_m": None if depth_range is None else float(depth_range[0]),
            "last_depth_max_m": None if depth_range is None else float(depth_range[1]),
            "last_endpoint_height_min_m": (
                None if height_range is None else float(height_range[0])
            ),
            "last_endpoint_height_max_m": (
                None if height_range is None else float(height_range[1])
            ),
            "unknown_is_free": bool(self.config.unknown_is_free),
            "query_total": int(self._query_total),
            "reject_occupied": int(self._free_occupied_reject),
            "reject_unknown": int(self._free_unknown_reject),
            "hit_robot_footprint": int(self._free_robot_hit),
            "hit_free_cell": int(self._free_cell_hit),
        }
