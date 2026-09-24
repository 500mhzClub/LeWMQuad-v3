"""Orientation-aware 2D footprint feasibility over scene manifests.

The existing oracle grid inflates obstacles by a circular body radius.  This
module preserves the calibrated asymmetric Go2 footprint instead: the body is
represented by forward, rear, left, and right extents about the base frame and
tested against the yawed boxes in a :class:`SceneManifest`.

Collision tests at an individual pose use the separating-axis theorem (SAT)
for exact rectangle-versus-oriented-rectangle intersection.  Swept tests use
adaptive interpolation along the shortest yaw arc.  The interpolation density
is bounded by both base translation and the arc travelled by the footprint's
furthest corner, so an in-place turn cannot silently skip a corner collision.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Mapping

from lewm_worlds.manifest import BoxObject, SceneManifest


DEFAULT_MAXIMUM_CORNER_STEP_M = 0.05
DEFAULT_MAXIMUM_YAW_STEP_RAD = math.radians(5.0)
DEFAULT_GEOMETRY_EPSILON_M = 1e-9


def _require_finite(value: float, *, name: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed):
        raise ValueError(f"{name} must be finite")
    return parsed


def _require_positive(value: float, *, name: str) -> float:
    parsed = _require_finite(value, name=name)
    if parsed <= 0.0:
        raise ValueError(f"{name} must be positive")
    return parsed


def wrap_angle_pi(angle_rad: float) -> float:
    """Wrap an angle to ``[-pi, pi)``."""

    angle = _require_finite(angle_rad, name="angle_rad")
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


@dataclass(frozen=True)
class Pose2D:
    """World-frame planar pose of the footprint's base reference point."""

    x_m: float
    y_m: float
    yaw_rad: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "x_m", _require_finite(self.x_m, name="x_m"))
        object.__setattr__(self, "y_m", _require_finite(self.y_m, name="y_m"))
        object.__setattr__(
            self,
            "yaw_rad",
            _require_finite(self.yaw_rad, name="yaw_rad"),
        )


@dataclass(frozen=True)
class OrientedRectangle:
    """A planar oriented rectangle represented by its geometric centre."""

    center_xy_m: tuple[float, float]
    half_extent_x_m: float
    half_extent_y_m: float
    yaw_rad: float
    object_id: str | None = None

    def __post_init__(self) -> None:
        if len(self.center_xy_m) != 2:
            raise ValueError("center_xy_m must contain exactly two values")
        center = (
            _require_finite(self.center_xy_m[0], name="center_xy_m[0]"),
            _require_finite(self.center_xy_m[1], name="center_xy_m[1]"),
        )
        object.__setattr__(self, "center_xy_m", center)
        object.__setattr__(
            self,
            "half_extent_x_m",
            _require_positive(self.half_extent_x_m, name="half_extent_x_m"),
        )
        object.__setattr__(
            self,
            "half_extent_y_m",
            _require_positive(self.half_extent_y_m, name="half_extent_y_m"),
        )
        object.__setattr__(
            self,
            "yaw_rad",
            _require_finite(self.yaw_rad, name="yaw_rad"),
        )

    @classmethod
    def from_box_object(cls, box: BoxObject) -> "OrientedRectangle":
        """Project a manifest box onto the world ``xy`` plane."""

        return cls(
            center_xy_m=(float(box.center_xyz_m[0]), float(box.center_xyz_m[1])),
            half_extent_x_m=0.5 * float(box.size_xyz_m[0]),
            half_extent_y_m=0.5 * float(box.size_xyz_m[1]),
            yaw_rad=float(box.yaw_rad),
            object_id=str(box.object_id),
        )

    @property
    def axes(self) -> tuple[tuple[float, float], tuple[float, float]]:
        """Return unit vectors for the rectangle's local x and y axes."""

        cos_yaw = math.cos(self.yaw_rad)
        sin_yaw = math.sin(self.yaw_rad)
        return ((cos_yaw, sin_yaw), (-sin_yaw, cos_yaw))

    @property
    def corners_xy_m(self) -> tuple[tuple[float, float], ...]:
        """Return the four world-frame corners."""

        axis_x, axis_y = self.axes
        center_x, center_y = self.center_xy_m
        corners: list[tuple[float, float]] = []
        for sign_x, sign_y in ((1.0, 1.0), (-1.0, 1.0), (-1.0, -1.0), (1.0, -1.0)):
            corners.append(
                (
                    center_x
                    + sign_x * self.half_extent_x_m * axis_x[0]
                    + sign_y * self.half_extent_y_m * axis_y[0],
                    center_y
                    + sign_x * self.half_extent_x_m * axis_x[1]
                    + sign_y * self.half_extent_y_m * axis_y[1],
                )
            )
        return tuple(corners)


@dataclass(frozen=True)
class AsymmetricFootprint:
    """Body extents from the base reference point in body coordinates.

    Body ``+x`` is forward and body ``+y`` is left.  Every extent is a
    positive distance from the reference point to the respective body edge.
    """

    forward_m: float
    rear_m: float
    left_m: float
    right_m: float

    def __post_init__(self) -> None:
        for name in ("forward_m", "rear_m", "left_m", "right_m"):
            object.__setattr__(
                self,
                name,
                _require_positive(getattr(self, name), name=name),
            )

    @classmethod
    def with_half_width(
        cls,
        *,
        forward_m: float,
        rear_m: float,
        half_width_m: float,
    ) -> "AsymmetricFootprint":
        """Construct a longitudinally asymmetric, laterally symmetric body."""

        return cls(
            forward_m=forward_m,
            rear_m=rear_m,
            left_m=half_width_m,
            right_m=half_width_m,
        )

    @property
    def local_center_xy_m(self) -> tuple[float, float]:
        return (
            0.5 * (self.forward_m - self.rear_m),
            0.5 * (self.left_m - self.right_m),
        )

    @property
    def half_extent_xy_m(self) -> tuple[float, float]:
        return (
            0.5 * (self.forward_m + self.rear_m),
            0.5 * (self.left_m + self.right_m),
        )

    @property
    def maximum_corner_radius_m(self) -> float:
        """Maximum distance from the base reference point to any corner."""

        return max(
            math.hypot(x_m, y_m)
            for x_m in (self.forward_m, -self.rear_m)
            for y_m in (self.left_m, -self.right_m)
        )

    def rectangle_at(self, pose: Pose2D) -> OrientedRectangle:
        """Return the footprint rectangle at ``pose``."""

        local_x, local_y = self.local_center_xy_m
        cos_yaw = math.cos(pose.yaw_rad)
        sin_yaw = math.sin(pose.yaw_rad)
        half_x, half_y = self.half_extent_xy_m
        return OrientedRectangle(
            center_xy_m=(
                pose.x_m + cos_yaw * local_x - sin_yaw * local_y,
                pose.y_m + sin_yaw * local_x + cos_yaw * local_y,
            ),
            half_extent_x_m=half_x,
            half_extent_y_m=half_y,
            yaw_rad=pose.yaw_rad,
        )

    def corners_at(self, pose: Pose2D) -> tuple[tuple[float, float], ...]:
        return self.rectangle_at(pose).corners_xy_m


def _polygon_signed_area(vertices: tuple[tuple[float, float], ...]) -> float:
    return 0.5 * sum(
        x0 * y1 - x1 * y0
        for (x0, y0), (x1, y1) in zip(
            vertices,
            (*vertices[1:], vertices[0]),
            strict=True,
        )
    )


def _clip_polygon_to_support_plane(
    vertices: tuple[tuple[float, float], ...],
    *,
    normal_xy: tuple[float, float],
    support_m: float,
) -> tuple[tuple[float, float], ...]:
    if not vertices:
        return ()
    nx, ny = normal_xy
    clipped: list[tuple[float, float]] = []
    previous = vertices[-1]
    previous_distance = previous[0] * nx + previous[1] * ny - support_m
    previous_inside = previous_distance <= 0.0
    for current in vertices:
        current_distance = current[0] * nx + current[1] * ny - support_m
        current_inside = current_distance <= 0.0
        if current_inside != previous_inside:
            denominator = previous_distance - current_distance
            fraction = previous_distance / denominator
            clipped.append(
                (
                    previous[0] + fraction * (current[0] - previous[0]),
                    previous[1] + fraction * (current[1] - previous[1]),
                )
            )
        if current_inside:
            clipped.append(current)
        previous = current
        previous_distance = current_distance
        previous_inside = current_inside
    return tuple(clipped)


def _simplify_convex_vertices(
    vertices: tuple[tuple[float, float], ...],
    *,
    tolerance_m: float = 1e-10,
) -> tuple[tuple[float, float], ...]:
    if len(vertices) < 3:
        return vertices
    deduplicated: list[tuple[float, float]] = []
    for vertex in vertices:
        if not deduplicated or math.dist(vertex, deduplicated[-1]) > tolerance_m:
            deduplicated.append(vertex)
    if len(deduplicated) > 1 and math.dist(
        deduplicated[0], deduplicated[-1]
    ) <= tolerance_m:
        deduplicated.pop()
    changed = True
    while changed and len(deduplicated) >= 3:
        changed = False
        reduced: list[tuple[float, float]] = []
        count = len(deduplicated)
        for index, current in enumerate(deduplicated):
            previous = deduplicated[(index - 1) % count]
            following = deduplicated[(index + 1) % count]
            cross = (
                (current[0] - previous[0]) * (following[1] - current[1])
                - (current[1] - previous[1]) * (following[0] - current[0])
            )
            if abs(cross) <= tolerance_m:
                changed = True
                continue
            reduced.append(current)
        deduplicated = reduced
    return tuple(deduplicated)


@dataclass(frozen=True)
class DirectionalSupportFootprint:
    """Convex body footprint reconstructed from directional support planes."""

    vertices_xy_m: tuple[tuple[float, float], ...]
    support_angles_deg: tuple[float, ...] = ()
    support_values_m: tuple[float, ...] = ()
    margin_m: float = 0.0

    def __post_init__(self) -> None:
        vertices = tuple(
            (
                _require_finite(vertex[0], name="vertices_xy_m.x"),
                _require_finite(vertex[1], name="vertices_xy_m.y"),
            )
            for vertex in self.vertices_xy_m
        )
        if len(vertices) < 3:
            raise ValueError("a directional footprint requires at least three vertices")
        if _polygon_signed_area(vertices) <= 0.0:
            raise ValueError("vertices_xy_m must be counter-clockwise and non-degenerate")
        for first, second, third in zip(
            vertices,
            (*vertices[1:], vertices[0]),
            (*vertices[2:], vertices[0], vertices[1]),
            strict=True,
        ):
            cross = (
                (second[0] - first[0]) * (third[1] - second[1])
                - (second[1] - first[1]) * (third[0] - second[0])
            )
            if cross <= 0.0:
                raise ValueError("vertices_xy_m must describe a strictly convex polygon")
        angles = tuple(
            _require_finite(value, name="support_angles_deg")
            for value in self.support_angles_deg
        )
        supports = tuple(
            _require_positive(value, name="support_values_m")
            for value in self.support_values_m
        )
        if len(angles) != len(supports):
            raise ValueError("support angle and value counts must match")
        margin = _require_finite(self.margin_m, name="margin_m")
        if margin < 0.0:
            raise ValueError("margin_m must be non-negative")
        object.__setattr__(self, "vertices_xy_m", vertices)
        object.__setattr__(self, "support_angles_deg", angles)
        object.__setattr__(self, "support_values_m", supports)
        object.__setattr__(self, "margin_m", margin)

    @classmethod
    def from_directional_support(
        cls,
        support_by_angle_deg: Mapping[float | int | str, float]
        | Iterable[tuple[float, float]],
        *,
        margin_m: float = 0.0,
    ) -> "DirectionalSupportFootprint":
        """Intersect support half-spaces into a convex body-frame polygon."""

        margin = _require_finite(margin_m, name="margin_m")
        if margin < 0.0:
            raise ValueError("margin_m must be non-negative")
        items = (
            support_by_angle_deg.items()
            if isinstance(support_by_angle_deg, Mapping)
            else support_by_angle_deg
        )
        normalized: dict[float, float] = {}
        for raw_angle, raw_support in items:
            angle = _require_finite(float(raw_angle), name="support angle") % 360.0
            support = _require_positive(raw_support, name="directional support")
            if angle in normalized:
                raise ValueError(f"duplicate support angle: {angle}")
            normalized[angle] = support + margin
        if len(normalized) < 4:
            raise ValueError("at least four directional supports are required")
        angles = tuple(sorted(normalized))
        gaps = tuple(
            (following - angle) % 360.0
            for angle, following in zip(
                angles,
                (*angles[1:], angles[0]),
                strict=True,
            )
        )
        if max(gaps) >= 180.0:
            raise ValueError("support directions do not bound a finite polygon")
        supports = tuple(normalized[angle] for angle in angles)
        initial_radius = 4.0 * max(supports)
        vertices: tuple[tuple[float, float], ...] = (
            (-initial_radius, -initial_radius),
            (initial_radius, -initial_radius),
            (initial_radius, initial_radius),
            (-initial_radius, initial_radius),
        )
        for angle, support in zip(angles, supports, strict=True):
            angle_rad = math.radians(angle)
            vertices = _clip_polygon_to_support_plane(
                vertices,
                normal_xy=(math.cos(angle_rad), math.sin(angle_rad)),
                support_m=support,
            )
            if len(vertices) < 3:
                raise ValueError("directional supports have an empty intersection")
        vertices = _simplify_convex_vertices(vertices)
        return cls(
            vertices_xy_m=vertices,
            support_angles_deg=angles,
            support_values_m=supports,
            margin_m=margin,
        )

    @property
    def maximum_vertex_radius_m(self) -> float:
        return max(math.hypot(x_m, y_m) for x_m, y_m in self.vertices_xy_m)

    def support_m(self, angle_deg: float) -> float:
        angle_rad = math.radians(float(angle_deg))
        normal_x, normal_y = math.cos(angle_rad), math.sin(angle_rad)
        return max(
            x_m * normal_x + y_m * normal_y
            for x_m, y_m in self.vertices_xy_m
        )

    def vertices_at(self, pose: Pose2D) -> tuple[tuple[float, float], ...]:
        cos_yaw = math.cos(pose.yaw_rad)
        sin_yaw = math.sin(pose.yaw_rad)
        return tuple(
            (
                pose.x_m + cos_yaw * body_x - sin_yaw * body_y,
                pose.y_m + sin_yaw * body_x + cos_yaw * body_y,
            )
            for body_x, body_y in self.vertices_xy_m
        )


def oriented_rectangles_intersect(
    first: OrientedRectangle,
    second: OrientedRectangle,
    *,
    epsilon_m: float = DEFAULT_GEOMETRY_EPSILON_M,
) -> bool:
    """Return whether two closed oriented rectangles intersect.

    Edge and corner contact count as intersection.  ``epsilon_m`` absorbs
    floating-point noise conservatively: gaps no larger than the epsilon also
    count as contact.
    """

    epsilon = _require_finite(epsilon_m, name="epsilon_m")
    if epsilon < 0.0:
        raise ValueError("epsilon_m must be non-negative")

    first_axes = first.axes
    second_axes = second.axes
    center_delta = (
        second.center_xy_m[0] - first.center_xy_m[0],
        second.center_xy_m[1] - first.center_xy_m[1],
    )
    for axis in (*first_axes, *second_axes):
        center_distance = abs(center_delta[0] * axis[0] + center_delta[1] * axis[1])
        first_radius = (
            first.half_extent_x_m
            * abs(first_axes[0][0] * axis[0] + first_axes[0][1] * axis[1])
            + first.half_extent_y_m
            * abs(first_axes[1][0] * axis[0] + first_axes[1][1] * axis[1])
        )
        second_radius = (
            second.half_extent_x_m
            * abs(second_axes[0][0] * axis[0] + second_axes[0][1] * axis[1])
            + second.half_extent_y_m
            * abs(second_axes[1][0] * axis[0] + second_axes[1][1] * axis[1])
        )
        if center_distance > first_radius + second_radius + epsilon:
            return False
    return True


def convex_polygon_intersects_rectangle(
    polygon_vertices_xy_m: Iterable[tuple[float, float]],
    rectangle: OrientedRectangle,
    *,
    epsilon_m: float = DEFAULT_GEOMETRY_EPSILON_M,
) -> bool:
    """Return whether a closed convex polygon intersects an oriented rectangle."""

    epsilon = _require_finite(epsilon_m, name="epsilon_m")
    if epsilon < 0.0:
        raise ValueError("epsilon_m must be non-negative")
    vertices = tuple(
        (
            _require_finite(vertex[0], name="polygon vertex x"),
            _require_finite(vertex[1], name="polygon vertex y"),
        )
        for vertex in polygon_vertices_xy_m
    )
    if len(vertices) < 3:
        raise ValueError("polygon must contain at least three vertices")
    polygon_axes: list[tuple[float, float]] = []
    for first, second in zip(
        vertices,
        (*vertices[1:], vertices[0]),
        strict=True,
    ):
        edge_x = second[0] - first[0]
        edge_y = second[1] - first[1]
        length = math.hypot(edge_x, edge_y)
        if length <= 0.0:
            raise ValueError("polygon contains a zero-length edge")
        polygon_axes.append((-edge_y / length, edge_x / length))
    rectangle_vertices = rectangle.corners_xy_m
    for axis_x, axis_y in (*polygon_axes, *rectangle.axes):
        polygon_projection = tuple(
            x_m * axis_x + y_m * axis_y for x_m, y_m in vertices
        )
        rectangle_projection = tuple(
            x_m * axis_x + y_m * axis_y for x_m, y_m in rectangle_vertices
        )
        if (
            max(polygon_projection) < min(rectangle_projection) - epsilon
            or max(rectangle_projection) < min(polygon_projection) - epsilon
        ):
            return False
    return True


@dataclass(frozen=True)
class PoseFeasibility:
    """Diagnostic result for one footprint pose."""

    pose: Pose2D
    inside_world_bounds: bool
    colliding_object_ids: tuple[str, ...]

    @property
    def feasible(self) -> bool:
        return self.inside_world_bounds and not self.colliding_object_ids


@dataclass(frozen=True)
class SweptPoseFeasibility:
    """Diagnostic result for an adaptively sampled pose sweep."""

    feasible: bool
    sample_count: int
    samples_evaluated: int
    first_infeasible_fraction: float | None
    first_infeasible_pose: PoseFeasibility | None


class ManifestFootprintFeasibility:
    """Check an asymmetric footprint against all manifest collision boxes."""

    def __init__(
        self,
        manifest: SceneManifest,
        footprint: AsymmetricFootprint,
        *,
        geometry_epsilon_m: float = DEFAULT_GEOMETRY_EPSILON_M,
    ) -> None:
        epsilon = _require_finite(geometry_epsilon_m, name="geometry_epsilon_m")
        if epsilon < 0.0:
            raise ValueError("geometry_epsilon_m must be non-negative")
        (x_min, y_min), (x_max, y_max) = manifest.world_bounds_xy_m
        bounds = tuple(
            _require_finite(value, name="world_bounds_xy_m")
            for value in (x_min, y_min, x_max, y_max)
        )
        if bounds[0] >= bounds[2] or bounds[1] >= bounds[3]:
            raise ValueError("world_bounds_xy_m must have positive area")

        self.manifest = manifest
        self.footprint = footprint
        self.geometry_epsilon_m = epsilon
        self.world_bounds_xy_m = ((bounds[0], bounds[1]), (bounds[2], bounds[3]))
        self.collision_boxes = tuple(
            OrientedRectangle.from_box_object(box)
            for box in manifest.static_objects
        )

    def pose_feasibility(self, pose: Pose2D) -> PoseFeasibility:
        """Return collision and boundary diagnostics for ``pose``."""

        footprint_rectangle = self.footprint.rectangle_at(pose)
        (x_min, y_min), (x_max, y_max) = self.world_bounds_xy_m
        epsilon = self.geometry_epsilon_m
        inside_bounds = all(
            x_min - epsilon <= x_m <= x_max + epsilon
            and y_min - epsilon <= y_m <= y_max + epsilon
            for x_m, y_m in footprint_rectangle.corners_xy_m
        )
        colliding_ids = tuple(
            box.object_id or ""
            for box in self.collision_boxes
            if oriented_rectangles_intersect(
                footprint_rectangle,
                box,
                epsilon_m=epsilon,
            )
        )
        return PoseFeasibility(
            pose=pose,
            inside_world_bounds=inside_bounds,
            colliding_object_ids=colliding_ids,
        )

    def is_pose_feasible(self, pose: Pose2D) -> bool:
        return self.pose_feasibility(pose).feasible

    def interpolated_sweep(
        self,
        start: Pose2D,
        end: Pose2D,
        *,
        maximum_corner_step_m: float = DEFAULT_MAXIMUM_CORNER_STEP_M,
        maximum_yaw_step_rad: float = DEFAULT_MAXIMUM_YAW_STEP_RAD,
    ) -> tuple[tuple[float, Pose2D], ...]:
        """Return adaptive samples, including both sweep endpoints.

        Yaw follows the shortest signed arc.  The number of intervals bounds
        the sum of base translation and maximum corner arc length, while also
        applying an explicit yaw-step bound.
        """

        corner_step = _require_positive(
            maximum_corner_step_m,
            name="maximum_corner_step_m",
        )
        yaw_step = _require_positive(
            maximum_yaw_step_rad,
            name="maximum_yaw_step_rad",
        )
        delta_x = end.x_m - start.x_m
        delta_y = end.y_m - start.y_m
        delta_yaw = wrap_angle_pi(end.yaw_rad - start.yaw_rad)
        translation_m = math.hypot(delta_x, delta_y)
        corner_motion_upper_bound_m = (
            translation_m
            + self.footprint.maximum_corner_radius_m * abs(delta_yaw)
        )
        interval_count = max(
            1,
            int(math.ceil(corner_motion_upper_bound_m / corner_step)),
            int(math.ceil(abs(delta_yaw) / yaw_step)),
        )
        return tuple(
            (
                fraction,
                Pose2D(
                    x_m=start.x_m + fraction * delta_x,
                    y_m=start.y_m + fraction * delta_y,
                    yaw_rad=wrap_angle_pi(start.yaw_rad + fraction * delta_yaw),
                ),
            )
            for index in range(interval_count + 1)
            for fraction in (index / interval_count,)
        )

    def swept_pose_feasibility(
        self,
        start: Pose2D,
        end: Pose2D,
        *,
        maximum_corner_step_m: float = DEFAULT_MAXIMUM_CORNER_STEP_M,
        maximum_yaw_step_rad: float = DEFAULT_MAXIMUM_YAW_STEP_RAD,
    ) -> SweptPoseFeasibility:
        """Check every adaptive sample from ``start`` through ``end``."""

        samples = self.interpolated_sweep(
            start,
            end,
            maximum_corner_step_m=maximum_corner_step_m,
            maximum_yaw_step_rad=maximum_yaw_step_rad,
        )
        for sample_index, (fraction, pose) in enumerate(samples):
            report = self.pose_feasibility(pose)
            if not report.feasible:
                return SweptPoseFeasibility(
                    feasible=False,
                    sample_count=len(samples),
                    samples_evaluated=sample_index + 1,
                    first_infeasible_fraction=fraction,
                    first_infeasible_pose=report,
                )
        return SweptPoseFeasibility(
            feasible=True,
            sample_count=len(samples),
            samples_evaluated=len(samples),
            first_infeasible_fraction=None,
            first_infeasible_pose=None,
        )

    def is_swept_pose_feasible(
        self,
        start: Pose2D,
        end: Pose2D,
        *,
        maximum_corner_step_m: float = DEFAULT_MAXIMUM_CORNER_STEP_M,
        maximum_yaw_step_rad: float = DEFAULT_MAXIMUM_YAW_STEP_RAD,
    ) -> bool:
        return self.swept_pose_feasibility(
            start,
            end,
            maximum_corner_step_m=maximum_corner_step_m,
            maximum_yaw_step_rad=maximum_yaw_step_rad,
        ).feasible


class ManifestDirectionalFootprintFeasibility:
    """Check a directional-support polygon against manifest collision boxes."""

    def __init__(
        self,
        manifest: SceneManifest,
        footprint: DirectionalSupportFootprint,
        *,
        geometry_epsilon_m: float = DEFAULT_GEOMETRY_EPSILON_M,
    ) -> None:
        epsilon = _require_finite(geometry_epsilon_m, name="geometry_epsilon_m")
        if epsilon < 0.0:
            raise ValueError("geometry_epsilon_m must be non-negative")
        (x_min, y_min), (x_max, y_max) = manifest.world_bounds_xy_m
        bounds = tuple(
            _require_finite(value, name="world_bounds_xy_m")
            for value in (x_min, y_min, x_max, y_max)
        )
        if bounds[0] >= bounds[2] or bounds[1] >= bounds[3]:
            raise ValueError("world_bounds_xy_m must have positive area")
        self.manifest = manifest
        self.footprint = footprint
        self.geometry_epsilon_m = epsilon
        self.world_bounds_xy_m = ((bounds[0], bounds[1]), (bounds[2], bounds[3]))
        self.collision_boxes = tuple(
            OrientedRectangle.from_box_object(box) for box in manifest.static_objects
        )

    def pose_feasibility(self, pose: Pose2D) -> PoseFeasibility:
        vertices = self.footprint.vertices_at(pose)
        (x_min, y_min), (x_max, y_max) = self.world_bounds_xy_m
        epsilon = self.geometry_epsilon_m
        inside_bounds = all(
            x_min - epsilon <= x_m <= x_max + epsilon
            and y_min - epsilon <= y_m <= y_max + epsilon
            for x_m, y_m in vertices
        )
        colliding_ids = tuple(
            box.object_id or ""
            for box in self.collision_boxes
            if convex_polygon_intersects_rectangle(
                vertices,
                box,
                epsilon_m=epsilon,
            )
        )
        return PoseFeasibility(
            pose=pose,
            inside_world_bounds=inside_bounds,
            colliding_object_ids=colliding_ids,
        )

    def is_pose_feasible(self, pose: Pose2D) -> bool:
        return self.pose_feasibility(pose).feasible

    def interpolated_sweep(
        self,
        start: Pose2D,
        end: Pose2D,
        *,
        maximum_corner_step_m: float = DEFAULT_MAXIMUM_CORNER_STEP_M,
        maximum_yaw_step_rad: float = DEFAULT_MAXIMUM_YAW_STEP_RAD,
    ) -> tuple[tuple[float, Pose2D], ...]:
        corner_step = _require_positive(
            maximum_corner_step_m,
            name="maximum_corner_step_m",
        )
        yaw_step = _require_positive(
            maximum_yaw_step_rad,
            name="maximum_yaw_step_rad",
        )
        delta_x = end.x_m - start.x_m
        delta_y = end.y_m - start.y_m
        delta_yaw = wrap_angle_pi(end.yaw_rad - start.yaw_rad)
        translation_m = math.hypot(delta_x, delta_y)
        corner_motion_upper_bound_m = (
            translation_m
            + self.footprint.maximum_vertex_radius_m * abs(delta_yaw)
        )
        interval_count = max(
            1,
            int(math.ceil(corner_motion_upper_bound_m / corner_step)),
            int(math.ceil(abs(delta_yaw) / yaw_step)),
        )
        return tuple(
            (
                fraction,
                Pose2D(
                    x_m=start.x_m + fraction * delta_x,
                    y_m=start.y_m + fraction * delta_y,
                    yaw_rad=wrap_angle_pi(start.yaw_rad + fraction * delta_yaw),
                ),
            )
            for index in range(interval_count + 1)
            for fraction in (index / interval_count,)
        )

    def swept_pose_feasibility(
        self,
        start: Pose2D,
        end: Pose2D,
        *,
        maximum_corner_step_m: float = DEFAULT_MAXIMUM_CORNER_STEP_M,
        maximum_yaw_step_rad: float = DEFAULT_MAXIMUM_YAW_STEP_RAD,
    ) -> SweptPoseFeasibility:
        samples = self.interpolated_sweep(
            start,
            end,
            maximum_corner_step_m=maximum_corner_step_m,
            maximum_yaw_step_rad=maximum_yaw_step_rad,
        )
        for sample_index, (fraction, pose) in enumerate(samples):
            report = self.pose_feasibility(pose)
            if not report.feasible:
                return SweptPoseFeasibility(
                    feasible=False,
                    sample_count=len(samples),
                    samples_evaluated=sample_index + 1,
                    first_infeasible_fraction=fraction,
                    first_infeasible_pose=report,
                )
        return SweptPoseFeasibility(
            feasible=True,
            sample_count=len(samples),
            samples_evaluated=len(samples),
            first_infeasible_fraction=None,
            first_infeasible_pose=None,
        )

    def is_swept_pose_feasible(
        self,
        start: Pose2D,
        end: Pose2D,
        *,
        maximum_corner_step_m: float = DEFAULT_MAXIMUM_CORNER_STEP_M,
        maximum_yaw_step_rad: float = DEFAULT_MAXIMUM_YAW_STEP_RAD,
    ) -> bool:
        return self.swept_pose_feasibility(
            start,
            end,
            maximum_corner_step_m=maximum_corner_step_m,
            maximum_yaw_step_rad=maximum_yaw_step_rad,
        ).feasible
