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

from dataclasses import asdict, dataclass, field
import hashlib
import json
import math
import re
from typing import Any, Mapping, Sequence

import torch


_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_TARGET_COLORS = ("red", "yellow", "blue", "green")
_CHAIN_GENESIS = hashlib.sha256(b"shared-v5-dev-tick-chain-v1").hexdigest()
def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


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
        if self.revision_after != self.revision_before + 1:
            raise SharedV5DevRuntimeError(
                "physical fuse must advance the supplied memory revision by exactly one"
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
    claim_intent_sha256: str | None
    claim_receipt: Mapping[str, object] | None
    command: MotionCommand


@dataclass(frozen=True)
class TickEvidence:
    tick_index: int; decision_sha256: str
    physical_revision: int; physical_content_sha256: str
    coverage_pose_xy_yaw: tuple[float, float, float]; visibility_opportunity_sha256: str
    collision_observed: bool; fall_observed: bool
    backend_evidence_sha256: str; claim_attempt_sha256: str | None
    previous_chain_sha256: str; content_sha256: str = field(init=False)
    chain_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        core = {"schema": "lewm_shared_v5_dev_tick_evidence_v1", **self.__dict__}; content = _canonical_sha256(core)
        object.__setattr__(self, "content_sha256", content)
        object.__setattr__(self, "chain_sha256", _canonical_sha256({"previous": self.previous_chain_sha256, "content": content}))
@dataclass(frozen=True)
class TerminalFaultRecord:
    tick_index: int; stage: str; exception_type: str
    physical_revision: int; physical_content_sha256: str; post_fuse_mutation: bool
    claim_journal: tuple[Mapping[str, object], ...]; last_claim_intent_sha256: str | None
    last_command: MotionCommand | None; counters: Mapping[str, int]; previous_chain_sha256: str
    schema: str = field(init=False, default="lewm_shared_v5_dev_terminal_fault_v1")
    content_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        core = {**self.__dict__, "last_command": asdict(self.last_command) if self.last_command else None}
        object.__setattr__(self, "content_sha256", _canonical_sha256({"schema": "lewm_shared_v5_dev_terminal_fault_v1", **core}))


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
    artifacts: RuntimeArtifactBindings; decisions: tuple[TickDecision, ...]
    tick_evidence: tuple[TickEvidence, ...]; claim_journal: tuple[Mapping[str, object], ...]
    counters: RuntimeCounters; reset_state_sha256: str
    content_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        self.counters.assert_one_frame_per_tick()
        if len(self.decisions) != self.counters.visual_ticks:
            raise SharedV5DevRuntimeError("decision count differs from visual ticks")
        if len(self.tick_evidence) != len(self.decisions): raise SharedV5DevRuntimeError("tick evidence count differs from decisions")
        previous = _CHAIN_GENESIS
        projected_claims = []
        for decision, evidence in zip(self.decisions, self.tick_evidence, strict=True):
            if evidence.previous_chain_sha256 != previous: raise SharedV5DevRuntimeError("tick evidence hash chain changed")
            if evidence.decision_sha256 != _canonical_sha256(asdict(decision)): raise SharedV5DevRuntimeError("tick evidence decision binding changed")
            if decision.claim_receipt is not None: projected_claims.append(decision.claim_receipt)
            previous = evidence.chain_sha256
        if tuple(projected_claims) != self.claim_journal: raise SharedV5DevRuntimeError("claim journal projection changed")
        object.__setattr__(self, "content_sha256", _canonical_sha256(
            {"schema": "lewm_shared_v5_dev_controller_run_v1", "artifacts": asdict(self.artifacts), "decisions": [asdict(x) for x in self.decisions],
             "tick_evidence": [asdict(x) for x in self.tick_evidence], "claims": list(self.claim_journal),
             "counters": asdict(self.counters), "reset_state_sha256": self.reset_state_sha256}))


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
        initial_state = _callable_attr(physical_memory, "to_dict")()
        empty_fields = ("seen_observation_ids", "seen_transaction_keys", "seen_semantic_transaction_keys",
                        "transactions", "active_observations", "traversals", "execution_blocks")
        if (type(initial_state) is not dict or initial_state.get("revision") != 0
                or initial_state.get("physical_content_sha256") != getattr(physical_memory, "physical_content_sha256", None)
                or any(initial_state.get(name) for name in empty_fields)):
            raise SharedV5DevRuntimeConfigurationError("controller requires canonical fresh revision-zero physical memory")
        self._initial_physical_state_sha256 = _canonical_sha256(initial_state)
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
        self._decisions: list[TickDecision] = []; self._tick_evidence: list[TickEvidence] = []
        self._claim_attempts: dict[str, Mapping[str, object]] = {}
        self._last_claim_intent_sha256: str | None = None; self._last_command: MotionCommand | None = None
        self._previous_chain_sha256 = _CHAIN_GENESIS
        self._stage = "constructed"
        self.terminal_fault: TerminalFaultRecord | None = None
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
        if (_canonical_sha256(_callable_attr(self.physical_memory, "to_dict")())
                != self._initial_physical_state_sha256 or self._claim_attempts):
            raise SharedV5DevRuntimeOrderError("reset requires canonical revision zero and empty controller journals")
        _callable_attr(backend, "reset")()
        self._reset_done = True
        self._tick_index = 0

    def tick(self, backend: object) -> TickDecision:
        if self._sealed:
            raise SharedV5DevRuntimeOrderError("controller is sealed")
        if not self._reset_done:
            raise SharedV5DevRuntimeOrderError("reset must precede the first tick")

        self._stage = "render"
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
        self._stage = "physical_fuse"
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
        self._stage = "post_fuse"
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

        claim_sha = None
        claim_ready = confirmed and goal == current and self.target_color not in self._claim_attempts
        if claim_ready:
            intent = _canonical_sha256({"tick": self._tick_index, "color": self.target_color, "goal": current})
            self._last_claim_intent_sha256 = intent
            receipt_value = _callable_attr(backend, "attempt_claim")(
                tick_index=self._tick_index, color=self.target_color, intent_sha256=intent)
            if type(receipt_value) is not dict or set(receipt_value) != {
                    "schema", "tick_index", "color", "intent_sha256", "content_sha256"}:
                raise SharedV5DevRuntimeError("claim receipt fields changed")
            core = dict(receipt_value)
            claim_sha = core.pop("content_sha256", None)
            if (receipt_value["schema"], receipt_value["tick_index"], receipt_value["color"], receipt_value["intent_sha256"], claim_sha) != (
                    "lewm_shared_v5_dev_claim_attempt_v1", self._tick_index, self.target_color, intent, _canonical_sha256(core)):
                raise SharedV5DevRuntimeError("claim receipt binding changed")
            self._claim_attempts[self.target_color] = receipt_value
        command = MotionCommand("hold", 0.0, 0.0, 0.0) if claim_ready else self._command_for_path(
            pose=pose, route_kind=route_kind, path_cells=path_cells, configuration_frame=configuration_frame
        )
        self._last_command = command
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
            claim_intent_sha256=intent if claim_ready else None,
            claim_receipt=receipt_value if claim_ready else None,
            command=command,
        )
        evidence = _callable_attr(backend, "navigation_evidence")(tick_index=self._tick_index)
        if type(evidence) is not dict or set(evidence) != {
            "schema", "tick_index", "coverage_pose_xy_yaw", "visibility_opportunity_sha256",
            "collision_observed", "fall_observed", "source_sha256", "content_sha256"
        }:
            raise SharedV5DevRuntimeError("backend navigation evidence fields changed")
        evidence_core = dict(evidence)
        evidence_sha = evidence_core.pop("content_sha256", None)
        if evidence["schema"] != "lewm_shared_v5_dev_backend_evidence_v1" or evidence["tick_index"] != self._tick_index or evidence_sha != _canonical_sha256(evidence_core):
            raise SharedV5DevRuntimeError("backend navigation evidence binding changed")
        coverage_pose = Pose2D(*evidence["coverage_pose_xy_yaw"])
        if type(evidence["collision_observed"]) is not bool or type(evidence["fall_observed"]) is not bool: raise SharedV5DevRuntimeError("backend collision/fall evidence type changed")
        _sha256(evidence["visibility_opportunity_sha256"], name="visibility opportunity")
        _sha256(evidence["source_sha256"], name="backend evidence source")
        tick_evidence = TickEvidence(
            tick_index=self._tick_index, decision_sha256=_canonical_sha256(asdict(decision)),
            physical_revision=receipt.revision_after, physical_content_sha256=receipt.physical_content_sha256,
            coverage_pose_xy_yaw=(coverage_pose.x_m, coverage_pose.y_m, coverage_pose.yaw_rad),
            visibility_opportunity_sha256=evidence["visibility_opportunity_sha256"],
            collision_observed=evidence["collision_observed"], fall_observed=evidence["fall_observed"],
            backend_evidence_sha256=evidence_sha, claim_attempt_sha256=claim_sha,
            previous_chain_sha256=self._previous_chain_sha256,
        )
        self._decisions.append(decision)
        self._tick_evidence.append(tick_evidence)
        self._previous_chain_sha256 = tick_evidence.chain_sha256
        self._tick_index += 1
        self._stage = "tick_complete"
        return decision

    def run_controller(self, backend: object, *, visual_ticks: int) -> ControllerRun:
        """Reset once, run a fixed number of visual ticks, then seal forever."""

        primary_error: BaseException | None = None
        try:
            if type(visual_ticks) is not int or visual_ticks <= 0:
                raise ValueError("visual_ticks must be a positive exact integer")
            self.reset(backend)
            reset_state_sha256 = _canonical_sha256({"physical_state_sha256": self._initial_physical_state_sha256, "claim_journal": []})
            for _ in range(visual_ticks):
                self.tick(backend)
            counters = RuntimeCounters(
                visual_ticks=visual_ticks,
                **self._counts,
            )
            counters.assert_one_frame_per_tick()
            return ControllerRun(
                artifacts=self.artifacts, decisions=tuple(self._decisions), tick_evidence=tuple(self._tick_evidence),
                claim_journal=tuple(self._claim_attempts.values()), counters=counters, reset_state_sha256=reset_state_sha256)
        except BaseException as exc:
            primary_error = exc
            if self._reset_done:
                self._record_fault(exc)
            raise
        finally:
            self._sealed = True
            try:
                _callable_attr(backend, "stop")()
            except BaseException as stop_error:
                if primary_error is None:
                    self._stage = "stop"
                    self._record_fault(stop_error)
                    raise
                primary_error.add_note(
                    "backend.stop() also failed: "
                    f"{type(stop_error).__name__}: {stop_error}"
                )

    def _record_fault(self, exc: BaseException) -> None:
        self.terminal_fault = TerminalFaultRecord(
            tick_index=self._tick_index, stage=self._stage, exception_type=type(exc).__name__,
            physical_revision=getattr(self.physical_memory, "revision", 0),
            physical_content_sha256=getattr(self.physical_memory, "physical_content_sha256", ""),
            post_fuse_mutation=getattr(self.physical_memory, "revision", 0) > len(self._decisions),
            claim_journal=tuple(self._claim_attempts.values()), last_claim_intent_sha256=self._last_claim_intent_sha256,
            last_command=self._last_command, counters={"visual_ticks_completed": len(self._decisions), **self._counts},
            previous_chain_sha256=self._previous_chain_sha256,
        )
        setattr(exc, "lewm_terminal_fault_record", self.terminal_fault)

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
    "TerminalFaultRecord",
    "TickDecision",
    "TickEvidence",
]
