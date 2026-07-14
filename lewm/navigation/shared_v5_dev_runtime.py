"""Small closed-loop Shared V5 controller for development mazes.

The runtime deliberately owns very little.  It calls the existing Shared V5
frame API once, hands the resulting feature objects to the existing target and
optional G4 heads, and delegates learned-evidence admission to the configured
physical fuser.  Mapping and routing stay in the existing revisioned physical
memory and two-resolution projection/planner.

Evaluation is intentionally absent from this module.  A caller may score the
returned immutable run only after :meth:`run_controller` has sealed the loop.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
import re
from typing import Any, Mapping, Sequence

import torch


_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_TARGET_COLORS = ("red", "yellow", "blue", "green")


class SharedV5DevRuntimeError(RuntimeError):
    """Base error for the development controller."""


class SharedV5DevRuntimeConfigurationError(SharedV5DevRuntimeError):
    """A required trained artifact or runtime component is unavailable."""


class SharedV5DevRuntimeOrderError(SharedV5DevRuntimeError):
    """Reset, tick, or controller-seal ordering was violated."""


def _sha256(value: object, *, name: str) -> str:
    if type(value) is not str or _SHA256.fullmatch(value) is None:
        raise SharedV5DevRuntimeConfigurationError(
            f"{name} must be an explicit lowercase SHA-256"
        )
    return value


def _finite(value: object, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise SharedV5DevRuntimeConfigurationError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise SharedV5DevRuntimeConfigurationError(f"{name} must be finite")
    return result


def _callable_attr(value: object, name: str) -> Any:
    result = getattr(value, name, None)
    if not callable(result):
        raise SharedV5DevRuntimeConfigurationError(
            f"{type(value).__name__} is missing callable {name}"
        )
    return result


def _cell(value: object, *, name: str) -> tuple[int, int]:
    if (
        not isinstance(value, Sequence)
        or isinstance(value, (str, bytes))
        or len(value) != 2
        or any(isinstance(item, bool) or not isinstance(item, int) for item in value)
    ):
        raise SharedV5DevRuntimeError(f"{name} must contain two integer indices")
    return (int(value[0]), int(value[1]))


@dataclass(frozen=True)
class RuntimeArtifactBindings:
    """Artifact identities required before a controller can be constructed."""

    shared_checkpoint_sha256: str
    g2_report_sha256: str
    physical_calibration_sha256: str
    target_head_checkpoint_sha256: str
    target_calibration_sha256: str
    g4_head_checkpoint_sha256: str | None = None
    g4_calibration_sha256: str | None = None

    def __post_init__(self) -> None:
        for name in (
            "shared_checkpoint_sha256",
            "g2_report_sha256",
            "physical_calibration_sha256",
            "target_head_checkpoint_sha256",
            "target_calibration_sha256",
        ):
            _sha256(getattr(self, name), name=name)
        g4_values = (self.g4_head_checkpoint_sha256, self.g4_calibration_sha256)
        if (g4_values[0] is None) != (g4_values[1] is None):
            raise SharedV5DevRuntimeConfigurationError(
                "G4 checkpoint and calibration must be supplied together"
            )
        if g4_values[0] is not None:
            _sha256(g4_values[0], name="g4_head_checkpoint_sha256")
            _sha256(g4_values[1], name="g4_calibration_sha256")

    @property
    def has_trained_g4(self) -> bool:
        return self.g4_head_checkpoint_sha256 is not None


@dataclass(frozen=True)
class TargetConfirmationCalibration:
    """Development thresholds selected without maze-evaluator feedback."""

    minimum_presence_probability: float
    minimum_quality: float
    maximum_uncertainty: float
    maximum_range_m: float

    def __post_init__(self) -> None:
        presence = _finite(
            self.minimum_presence_probability,
            name="minimum_presence_probability",
        )
        quality = _finite(self.minimum_quality, name="minimum_quality")
        uncertainty = _finite(self.maximum_uncertainty, name="maximum_uncertainty")
        maximum_range = _finite(self.maximum_range_m, name="maximum_range_m")
        if not 0.0 <= presence <= 1.0:
            raise SharedV5DevRuntimeConfigurationError(
                "minimum_presence_probability must lie in [0,1]"
            )
        if not 0.0 <= quality <= 1.0:
            raise SharedV5DevRuntimeConfigurationError(
                "minimum_quality must lie in [0,1]"
            )
        if uncertainty <= 0.0 or maximum_range <= 0.0:
            raise SharedV5DevRuntimeConfigurationError(
                "uncertainty and maximum range must be positive"
            )

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> "TargetConfirmationCalibration":
        required = {
            "minimum_presence_probability",
            "minimum_quality",
            "maximum_uncertainty",
            "maximum_range_m",
        }
        if not isinstance(value, Mapping) or not required <= set(value):
            missing = sorted(required - set(value)) if isinstance(value, Mapping) else sorted(required)
            raise SharedV5DevRuntimeConfigurationError(
                "target calibration is missing: " + ", ".join(missing)
            )
        return cls(**{name: value[name] for name in required})  # type: ignore[arg-type]


@dataclass(frozen=True)
class Pose2D:
    x_m: float
    y_m: float
    yaw_rad: float

    def __post_init__(self) -> None:
        for name in ("x_m", "y_m", "yaw_rad"):
            object.__setattr__(self, name, _finite(getattr(self, name), name=name))


@dataclass(frozen=True)
class DevelopmentPhysicalFuseReceipt:
    """Exact development-memory revision produced by one visual frame.

    This is a local integration receipt, not G2 qualification or production
    evidence authority.
    """

    memory: object = field(repr=False, compare=False)
    physical_map_frame_sha256: str
    revision_before: int
    revision_after: int
    physical_content_sha256: str

    def __post_init__(self) -> None:
        _sha256(self.physical_map_frame_sha256, name="physical_map_frame_sha256")
        _sha256(self.physical_content_sha256, name="physical_content_sha256")
        for name in ("revision_before", "revision_after"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise SharedV5DevRuntimeError(f"{name} must be nonnegative")
        if self.revision_after <= self.revision_before:
            raise SharedV5DevRuntimeError(
                "physical fuse must advance the supplied memory revision"
            )


@dataclass(frozen=True)
class MotionCommand:
    """One conservative velocity command tick."""

    primitive: str
    vx_body_mps: float
    vy_body_mps: float
    yaw_rate_radps: float

    def __post_init__(self) -> None:
        allowed = {
            "yaw_left",
            "yaw_right",
            "arc_left",
            "arc_right",
            "forward",
            "hold",
        }
        if self.primitive not in allowed:
            raise ValueError(f"unknown development primitive: {self.primitive}")
        for name in ("vx_body_mps", "vy_body_mps", "yaw_rate_radps"):
            object.__setattr__(self, name, _finite(getattr(self, name), name=name))

    @property
    def velocity_xyz(self) -> tuple[float, float, float]:
        return (self.vx_body_mps, self.vy_body_mps, self.yaw_rate_radps)


@dataclass(frozen=True)
class G4CandidateBatch:
    """Existing learned-head batch plus its exact configuration-cell rows."""

    cells: tuple[tuple[int, int], ...]
    head_batch: object

    def __post_init__(self) -> None:
        cells = tuple(_cell(item, name="G4 candidate cell") for item in self.cells)
        if not cells or len(set(cells)) != len(cells):
            raise SharedV5DevRuntimeError("G4 candidate cells must be nonempty and unique")
        object.__setattr__(self, "cells", cells)


@dataclass(frozen=True)
class TickDecision:
    tick_index: int
    pose: Pose2D
    current_configuration_cell: tuple[int, int]
    route_kind: str
    goal_configuration_cell: tuple[int, int] | None
    path_cells: tuple[tuple[int, int], ...]
    target_confirmed: bool
    target_presence_probability: float
    target_quality: float
    target_uncertainty: float
    command: MotionCommand


@dataclass(frozen=True)
class RuntimeCounters:
    visual_ticks: int
    rgb_renders: int
    rgb_preprocesses: int
    shared_forward_frames: int
    target_head_calls: int
    g4_head_calls: int
    physical_fusions: int
    commands_applied: int

    def assert_one_frame_per_tick(self) -> None:
        expected = self.visual_ticks
        if not (
            self.rgb_renders
            == self.rgb_preprocesses
            == self.shared_forward_frames
            == self.target_head_calls
            == self.physical_fusions
            == self.commands_applied
            == expected
        ):
            raise SharedV5DevRuntimeError(
                "one-render/one-preprocess/one-forward tick invariant failed"
            )
        if self.g4_head_calls > expected:
            raise SharedV5DevRuntimeError("G4 ran more than once per visual tick")


@dataclass(frozen=True)
class ControllerRun:
    decisions: tuple[TickDecision, ...]
    counters: RuntimeCounters

    def __post_init__(self) -> None:
        self.counters.assert_one_frame_per_tick()
        if len(self.decisions) != self.counters.visual_ticks:
            raise SharedV5DevRuntimeError("decision count differs from visual ticks")


class SharedV5DevMazeRuntime:
    """One reset-local development controller with fixed-duration execution."""

    def __init__(
        self,
        *,
        model: object,
        target_head: object,
        physical_fuser: object,
        physical_memory: object,
        projection: object,
        planner: object,
        target_calibration: TargetConfirmationCalibration,
        artifacts: RuntimeArtifactBindings,
        target_color: str,
        g4_head: object | None = None,
        g4_candidate_builder: object | None = None,
        frontier_cap: int = 16,
    ) -> None:
        _callable_attr(model, "forward_frame")
        if not callable(target_head):
            raise SharedV5DevRuntimeConfigurationError("target head must be callable")
        _callable_attr(physical_fuser, "fuse")
        _callable_attr(projection, "project")
        for method in ("connected_component", "frontier_cells", "astar"):
            _callable_attr(planner, method)
        if type(target_calibration) is not TargetConfirmationCalibration:
            raise TypeError("target_calibration must be TargetConfirmationCalibration")
        if type(artifacts) is not RuntimeArtifactBindings:
            raise TypeError("artifacts must be RuntimeArtifactBindings")
        if target_color not in _TARGET_COLORS:
            raise SharedV5DevRuntimeConfigurationError(
                f"target_color must be one of {_TARGET_COLORS}"
            )
        if type(frontier_cap) is not int or not 1 <= frontier_cap <= 64:
            raise SharedV5DevRuntimeConfigurationError(
                "frontier_cap must be an integer in [1,64]"
            )
        map_frame = getattr(physical_memory, "map_frame", None)
        physical_cell_size = getattr(map_frame, "cell_size_m", None)
        if physical_cell_size is None or not math.isclose(
            float(physical_cell_size), 0.05, rel_tol=0.0, abs_tol=1e-12
        ):
            raise SharedV5DevRuntimeConfigurationError(
                "development navigation requires the existing 0.05 m physical memory"
            )
        projection_memory = getattr(projection, "memory", None)
        if projection_memory is None:
            projection_memory = getattr(projection, "_memory", None)
        if projection_memory is not physical_memory:
            raise SharedV5DevRuntimeConfigurationError(
                "projection must consume the exact supplied physical memory"
            )
        if artifacts.has_trained_g4:
            if g4_head is None or g4_candidate_builder is None:
                raise SharedV5DevRuntimeConfigurationError(
                    "trained G4 requires both its existing head and candidate builder"
                )
            if not callable(g4_head) or not callable(g4_candidate_builder):
                raise SharedV5DevRuntimeConfigurationError(
                    "G4 head and candidate builder must be callable"
                )
        elif g4_head is not None or g4_candidate_builder is not None:
            raise SharedV5DevRuntimeConfigurationError(
                "G4 cannot run without explicit trained checkpoint and calibration"
            )

        self.model = model
        self.target_head = target_head
        self.physical_fuser = physical_fuser
        self.physical_memory = physical_memory
        self.projection = projection
        self.planner = planner
        self.target_calibration = target_calibration
        self.artifacts = artifacts
        self.target_color = target_color
        self.g4_head = g4_head
        self.g4_candidate_builder = g4_candidate_builder
        self.frontier_cap = frontier_cap
        self._reset_done = False
        self._sealed = False
        self._tick_index = 0
        self._counts = {
            "rgb_renders": 0,
            "rgb_preprocesses": 0,
            "shared_forward_frames": 0,
            "target_head_calls": 0,
            "g4_head_calls": 0,
            "physical_fusions": 0,
            "commands_applied": 0,
        }

    def reset(self, backend: object) -> None:
        if self._sealed:
            raise SharedV5DevRuntimeOrderError("sealed controller cannot be reset")
        if self._reset_done:
            raise SharedV5DevRuntimeOrderError("controller was already reset")
        _callable_attr(backend, "reset")()
        self._reset_done = True
        self._tick_index = 0

    def tick(self, backend: object) -> TickDecision:
        if self._sealed:
            raise SharedV5DevRuntimeOrderError("controller is sealed")
        if not self._reset_done:
            raise SharedV5DevRuntimeOrderError("reset must precede the first tick")

        raw_rgb = _callable_attr(backend, "render_rgb")()
        self._counts["rgb_renders"] += 1
        image = _callable_attr(backend, "preprocess_rgb")(raw_rgb)
        self._counts["rgb_preprocesses"] += 1
        if not isinstance(image, torch.Tensor) or image.ndim != 4 or image.shape[0] != 1:
            raise SharedV5DevRuntimeError(
                "preprocess_rgb must return one batched image tensor"
            )
        camera = _callable_attr(backend, "camera_calibration_tensors")(image)
        if not isinstance(camera, Sequence) or len(camera) != 3:
            raise SharedV5DevRuntimeError(
                "camera_calibration_tensors must return origin, basis, and ground"
            )

        with torch.inference_mode():
            frame = self.model.forward_frame(image, camera[0], camera[1], camera[2])
            self._counts["shared_forward_frames"] += 1
            patch_tokens = getattr(frame, "patch_tokens", None)
            bev = getattr(frame, "bev", None)
            evidence = getattr(frame, "evidence", None)
            if patch_tokens is None or bev is None or evidence is None:
                raise SharedV5DevRuntimeError(
                    "Shared V5 forward_frame returned an incomplete frame"
                )
            target_output = self.target_head(patch_tokens, bev)
            self._counts["target_head_calls"] += 1

        pose_value = _callable_attr(backend, "pose_xy_yaw")()
        pose = pose_value if type(pose_value) is Pose2D else Pose2D(*pose_value)
        revision_before = getattr(self.physical_memory, "revision", None)
        receipt = self.physical_fuser.fuse(
            evidence=evidence,
            pose=pose,
            tick_index=self._tick_index,
            memory=self.physical_memory,
            camera_origin_body_m=camera[0],
            camera_basis_body_fru=camera[1],
            ground_plane_z_body_m=camera[2],
        )
        self._counts["physical_fusions"] += 1
        if type(receipt) is not DevelopmentPhysicalFuseReceipt:
            raise SharedV5DevRuntimeError(
                "physical fuser must return DevelopmentPhysicalFuseReceipt"
            )
        map_frame = getattr(self.physical_memory, "map_frame", None)
        map_frame_sha = getattr(map_frame, "content_sha256", None)
        if (
            receipt.memory is not self.physical_memory
            or receipt.revision_before != revision_before
            or receipt.revision_after != getattr(self.physical_memory, "revision", None)
            or receipt.physical_map_frame_sha256 != map_frame_sha
            or receipt.physical_content_sha256
            != getattr(self.physical_memory, "physical_content_sha256", None)
        ):
            raise SharedV5DevRuntimeError(
                "physical fuse receipt does not bind the exact updated memory"
            )
        snapshot = self.projection.project()
        if (
            getattr(snapshot, "physical_revision", None) != receipt.revision_after
            or getattr(snapshot, "physical_map_frame_sha256", None)
            != receipt.physical_map_frame_sha256
            or getattr(snapshot, "physical_content_sha256", None)
            != receipt.physical_content_sha256
        ):
            raise SharedV5DevRuntimeError(
                "projection snapshot is stale or bound to a different physical memory"
            )
        configuration_frame = getattr(snapshot, "configuration_map_frame", None)
        if configuration_frame is None:
            raise SharedV5DevRuntimeError("projection snapshot has no configuration frame")
        current = _cell(
            _callable_attr(configuration_frame, "world_to_cell")((pose.x_m, pose.y_m)),
            name="current configuration cell",
        )

        target = self._target_reading(target_output)
        confirmed = self._target_is_confirmed(target)
        route_kind = "hold"
        goal: tuple[int, int] | None = None
        path_cells: tuple[tuple[int, int], ...] = ()
        free_cells = getattr(snapshot, "free_cells", frozenset())
        if current in free_cells:
            component = self.planner.connected_component(snapshot, current)
            component_cells = tuple(
                sorted(_cell(item, name="component cell") for item in component.cells)
            )
            if confirmed and component_cells:
                target_xy = self._target_world_xy(pose, target)
                goal = min(
                    component_cells,
                    key=lambda item: (
                        self._distance_sq(
                            configuration_frame.cell_center(item), target_xy
                        ),
                        item,
                    ),
                )
                route_kind = "target"
            else:
                frontiers = self.planner.frontier_cells(snapshot, component)
                nearest = tuple(
                    sorted(
                        (_cell(item, name="frontier cell") for item in frontiers.cells),
                        key=lambda item: (
                            self._distance_sq(
                                configuration_frame.cell_center(item),
                                (pose.x_m, pose.y_m),
                            ),
                            item,
                        ),
                    )[: self.frontier_cap]
                )
                if nearest:
                    goal = self._select_frontier(
                        snapshot=snapshot,
                        component=component,
                        frontiers=frontiers,
                        nearest=nearest,
                        current=current,
                        pose=pose,
                        patch_tokens=patch_tokens,
                        bev=bev,
                    )
                    route_kind = "frontier"
            if goal is not None:
                path = self.planner.astar(snapshot, current, goal)
                if path is not None:
                    path_cells = tuple(
                        _cell(item, name="path cell") for item in path.cells
                    )

        command = self._command_for_path(
            pose=pose,
            route_kind=route_kind,
            path_cells=path_cells,
            configuration_frame=configuration_frame,
        )
        _callable_attr(backend, "apply_command")(command)
        self._counts["commands_applied"] += 1
        decision = TickDecision(
            tick_index=self._tick_index,
            pose=pose,
            current_configuration_cell=current,
            route_kind=route_kind,
            goal_configuration_cell=goal,
            path_cells=path_cells,
            target_confirmed=confirmed,
            target_presence_probability=target["presence"],
            target_quality=target["quality"],
            target_uncertainty=target["uncertainty"],
            command=command,
        )
        self._tick_index += 1
        return decision

    def run_controller(self, backend: object, *, visual_ticks: int) -> ControllerRun:
        """Reset once, run a fixed number of visual ticks, then seal forever."""

        primary_error: BaseException | None = None
        try:
            if type(visual_ticks) is not int or visual_ticks <= 0:
                raise ValueError("visual_ticks must be a positive exact integer")
            self.reset(backend)
            decisions = tuple(self.tick(backend) for _ in range(visual_ticks))
            counters = RuntimeCounters(
                visual_ticks=visual_ticks,
                **self._counts,
            )
            counters.assert_one_frame_per_tick()
            return ControllerRun(decisions=decisions, counters=counters)
        except BaseException as exc:
            primary_error = exc
            raise
        finally:
            self._sealed = True
            try:
                _callable_attr(backend, "stop")()
            except BaseException as stop_error:
                if primary_error is None:
                    raise
                primary_error.add_note(
                    "backend.stop() also failed: "
                    f"{type(stop_error).__name__}: {stop_error}"
                )

    def _target_reading(self, output: object) -> dict[str, float]:
        colors = tuple(getattr(output, "colors", ()))
        if colors != _TARGET_COLORS:
            raise SharedV5DevRuntimeError("target head color order changed")
        index = colors.index(self.target_color)

        def scalar(name: str) -> float:
            tensor = getattr(output, name, None)
            if not isinstance(tensor, torch.Tensor) or tuple(tensor.shape) != (1, 4):
                raise SharedV5DevRuntimeError(
                    f"target head {name} must have shape [1,4]"
                )
            result = float(tensor[0, index].detach().item())
            if not math.isfinite(result):
                raise SharedV5DevRuntimeError(f"target head {name} is nonfinite")
            return result

        return {
            "presence": scalar("presence_probability"),
            "quality": scalar("quality"),
            "uncertainty": scalar("uncertainty"),
            "bearing": scalar("bearing_mean_rad"),
            "range": scalar("range_mean_m"),
        }

    def _target_is_confirmed(self, target: Mapping[str, float]) -> bool:
        calibration = self.target_calibration
        return bool(
            target["presence"] >= calibration.minimum_presence_probability
            and target["quality"] >= calibration.minimum_quality
            and target["uncertainty"] <= calibration.maximum_uncertainty
            and 0.0 < target["range"] <= calibration.maximum_range_m
        )

    @staticmethod
    def _target_world_xy(
        pose: Pose2D, target: Mapping[str, float]
    ) -> tuple[float, float]:
        body_angle = pose.yaw_rad + target["bearing"]
        return (
            pose.x_m + target["range"] * math.cos(body_angle),
            pose.y_m + target["range"] * math.sin(body_angle),
        )

    def _select_frontier(
        self,
        *,
        snapshot: object,
        component: object,
        frontiers: object,
        nearest: tuple[tuple[int, int], ...],
        current: tuple[int, int],
        pose: Pose2D,
        patch_tokens: object,
        bev: object,
    ) -> tuple[int, int]:
        if self.g4_head is None:
            return nearest[0]
        built = self.g4_candidate_builder(
            snapshot=snapshot,
            component=component,
            frontiers=frontiers,
            nearest_cells=nearest,
            start_cell=current,
            pose=pose,
            cap=self.frontier_cap,
        )
        if type(built) is not G4CandidateBatch:
            raise SharedV5DevRuntimeError(
                "G4 candidate builder must return G4CandidateBatch"
            )
        if len(built.cells) > self.frontier_cap or not set(built.cells) <= set(nearest):
            raise SharedV5DevRuntimeError(
                "G4 candidate rows left the capped nearest-frontier set"
            )
        with torch.inference_mode():
            scores = self.g4_head(patch_tokens, bev, built.head_batch)
        self._counts["g4_head_calls"] += 1
        selected = _callable_attr(scores, "selected_row_indices")()
        if not isinstance(selected, Sequence) or len(selected) != 1:
            raise SharedV5DevRuntimeError("G4 must select one row for batch size one")
        index = selected[0]
        if isinstance(index, bool) or not isinstance(index, int) or not 0 <= index < len(built.cells):
            raise SharedV5DevRuntimeError("G4 selected an invalid candidate row")
        return built.cells[index]

    @staticmethod
    def _distance_sq(left: Sequence[float], right: Sequence[float]) -> float:
        return (float(left[0]) - float(right[0])) ** 2 + (
            float(left[1]) - float(right[1])
        ) ** 2

    @staticmethod
    def _wrap_angle(value: float) -> float:
        return (value + math.pi) % (2.0 * math.pi) - math.pi

    def _command_for_path(
        self,
        *,
        pose: Pose2D,
        route_kind: str,
        path_cells: tuple[tuple[int, int], ...],
        configuration_frame: object,
    ) -> MotionCommand:
        if len(path_cells) < 2:
            if route_kind == "frontier" and path_cells:
                return MotionCommand("yaw_left", 0.0, 0.0, 0.35)
            return MotionCommand("hold", 0.0, 0.0, 0.0)
        waypoint = _callable_attr(configuration_frame, "cell_center")(path_cells[1])
        desired = math.atan2(
            float(waypoint[1]) - pose.y_m,
            float(waypoint[0]) - pose.x_m,
        )
        error = self._wrap_angle(desired - pose.yaw_rad)
        if abs(error) > 0.35:
            return MotionCommand(
                "yaw_left" if error > 0.0 else "yaw_right",
                0.0,
                0.0,
                0.45 if error > 0.0 else -0.45,
            )
        if abs(error) > 0.12:
            return MotionCommand(
                "arc_left" if error > 0.0 else "arc_right",
                0.18,
                0.0,
                0.30 if error > 0.0 else -0.30,
            )
        return MotionCommand("forward", 0.25, 0.0, 0.0)


__all__ = [
    "ControllerRun",
    "DevelopmentPhysicalFuseReceipt",
    "G4CandidateBatch",
    "MotionCommand",
    "Pose2D",
    "RuntimeArtifactBindings",
    "RuntimeCounters",
    "SharedV5DevMazeRuntime",
    "SharedV5DevRuntimeConfigurationError",
    "SharedV5DevRuntimeError",
    "SharedV5DevRuntimeOrderError",
    "TargetConfirmationCalibration",
    "TickDecision",
]
