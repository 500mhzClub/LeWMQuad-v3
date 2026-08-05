"""Deterministic safe target routing over exact G3 V2 snapshots."""
from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
import math
from typing import Iterable, Sequence

from lewm.planning.revisioned_physical_configuration_memory import (
    PhysicalLabel,
    SnapshotBindingError,
)
from lewm.planning.two_resolution_configuration_projection_v2 import (
    CONFIGURATION_CELL_SIZE_M,
    ConfigurationComponentV2,
    ConfigurationPathV2,
    TwoResolutionConfigurationPlannerV2,
    TwoResolutionConfigurationProjectionV2,
    TwoResolutionConfigurationSnapshotV2,
)
from lewm.planning.two_resolution_reversible_target_belief_v1 import (
    TwoResolutionReversibleTargetBeliefMemoryV1,
    TwoResolutionTargetHypothesisV1,
    TwoResolutionTargetPosteriorSnapshotV1,
)


Cell = tuple[int, int]

PRODUCTION_TWO_RESOLUTION_TARGET_ROUTER = None


class TwoResolutionTargetRouterError(ValueError):
    """Base target-router error."""


class TwoResolutionTargetRouteBindingError(
    TwoResolutionTargetRouterError, SnapshotBindingError
):
    """A route plan changed or no longer binds live navigation state."""


class NoSafeTargetRouteError(TwoResolutionTargetRouterError):
    """No confirmed-free reacquisition or claim-view route exists."""


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def _sha256(value: object) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _require_sha256(value: object, name: str) -> str:
    if not (
        type(value) is str
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256")
    return value


def _cell(value: Sequence[int], name: str) -> Cell:
    if (
        isinstance(value, (str, bytes))
        or len(value) != 2
        or any(isinstance(item, bool) or not isinstance(item, int) for item in value)
    ):
        raise TypeError(f"{name} must contain two integer indices")
    return int(value[0]), int(value[1])


def _finite(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _wrap(angle: float) -> float:
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def _cells_json(cells: Iterable[Cell]) -> list[list[int]]:
    return [[cell[0], cell[1]] for cell in sorted(cells)]


def _path_sha256(path: ConfigurationPathV2) -> str:
    return _sha256(
        {
            "schema": "lewm_g3_v2_configuration_path_for_target_router_v1",
            "snapshot_sha256": path.snapshot_sha256,
            "physical_map_frame_sha256": path.physical_map_frame_sha256,
            "configuration_map_frame_sha256": path.configuration_map_frame_sha256,
            "physical_revision": path.physical_revision,
            "configuration_revision": path.configuration_revision,
            "free_support_sha256": path.free_support_sha256,
            "occupied_support_sha256": path.occupied_support_sha256,
            "cells": [list(cell) for cell in path.cells],
            "cost": path.cost,
        }
    )


@dataclass(frozen=True)
class TwoResolutionTargetRouterConfigV1:
    minimum_view_distance_m: float = 0.10
    maximum_claim_distance_m: float = 1.20
    claim_bearing_tolerance_rad: float = 0.25
    content_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        minimum = _finite(self.minimum_view_distance_m, "minimum_view_distance_m")
        maximum = _finite(self.maximum_claim_distance_m, "maximum_claim_distance_m")
        bearing = _finite(
            self.claim_bearing_tolerance_rad,
            "claim_bearing_tolerance_rad",
        )
        if not 0.0 < minimum < maximum or not math.isclose(
            maximum, 1.20, rel_tol=0.0, abs_tol=1e-12
        ):
            raise ValueError("target route distance contract changed")
        if not math.isclose(bearing, 0.25, rel_tol=0.0, abs_tol=1e-12):
            raise ValueError("target route bearing contract changed")
        object.__setattr__(self, "minimum_view_distance_m", minimum)
        object.__setattr__(self, "maximum_claim_distance_m", maximum)
        object.__setattr__(self, "claim_bearing_tolerance_rad", bearing)
        object.__setattr__(self, "content_sha256", _sha256(self.to_dict(False)))

    def to_dict(self, include_hash: bool = True) -> dict[str, object]:
        result: dict[str, object] = {
            "schema": "lewm_g5_two_resolution_target_router_config_v1",
            "minimum_view_distance_m": self.minimum_view_distance_m,
            "maximum_claim_distance_m": self.maximum_claim_distance_m,
            "claim_bearing_tolerance_rad": self.claim_bearing_tolerance_rad,
            "route_cell_size_m": CONFIGURATION_CELL_SIZE_M,
            "production_promotion_authorized": False,
            "hardware_execution_authorized": False,
        }
        if include_hash:
            result["content_sha256"] = self.content_sha256
        return result


@dataclass(frozen=True)
class TwoResolutionTargetRouteReceiptV1:
    router_config_sha256: str
    target_id: str
    target_posterior_sha256: str
    target_memory_instance_sha256: str
    target_memory_revision: int
    target_evidence_chain_sha256: str
    snapshot_sha256: str
    component_sha256: str
    physical_map_frame_sha256: str
    configuration_map_frame_sha256: str
    physical_revision: int
    configuration_revision: int
    profile_sha256: str
    free_support_sha256: str
    occupied_support_sha256: str
    start_cell: Cell
    start_yaw_rad: float
    selected_hypothesis_cells: frozenset[Cell]
    selected_hypothesis_mass: float
    selected_hypothesis_peak_cell: Cell
    candidate_set_sha256: str
    candidate_count: int
    goal_cell: Cell
    target_world_xy_m: tuple[float, float]
    target_distance_m: float
    terminal_yaw_rad: float
    initial_heading_error_rad: float
    path_sha256: str
    path_cells: tuple[Cell, ...]
    path_cost_configuration_steps: float
    path_cost_m: float
    score_key: tuple[float, ...]
    exact_sim_tainted: bool
    production_promotion_authorized: bool
    hardware_execution_authorized: bool
    _issuance_capability: object = field(repr=False, compare=False)
    content_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        for name in (
            "router_config_sha256",
            "target_posterior_sha256",
            "target_memory_instance_sha256",
            "target_evidence_chain_sha256",
            "snapshot_sha256",
            "component_sha256",
            "physical_map_frame_sha256",
            "configuration_map_frame_sha256",
            "profile_sha256",
            "free_support_sha256",
            "occupied_support_sha256",
            "candidate_set_sha256",
            "path_sha256",
        ):
            _require_sha256(getattr(self, name), name)
        if type(self.target_id) is not str or not self.target_id:
            raise ValueError("target_id must be nonempty")
        for name in (
            "target_memory_revision",
            "physical_revision",
            "configuration_revision",
            "candidate_count",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")
        if self.candidate_count == 0:
            raise ValueError("target route requires at least one candidate")
        start = _cell(self.start_cell, "start_cell")
        goal = _cell(self.goal_cell, "goal_cell")
        peak = _cell(self.selected_hypothesis_peak_cell, "hypothesis peak")
        hypothesis = frozenset(
            _cell(cell, "hypothesis cell") for cell in self.selected_hypothesis_cells
        )
        path = tuple(_cell(cell, "path cell") for cell in self.path_cells)
        if not hypothesis or peak not in hypothesis:
            raise ValueError("selected hypothesis is malformed")
        if not path or path[0] != start or path[-1] != goal:
            raise ValueError("target path endpoints changed")
        if goal in hypothesis or any(cell in hypothesis for cell in path[-1:]):
            raise ValueError("target route cannot terminate in a hypothesis cell")
        if any(
            abs(a[0] - b[0]) + abs(a[1] - b[1]) != 1
            for a, b in zip(path, path[1:])
        ):
            raise ValueError("target path is not four-connected")
        target_xy = tuple(float(value) for value in self.target_world_xy_m)
        if len(target_xy) != 2 or any(not math.isfinite(value) for value in target_xy):
            raise ValueError("target_world_xy_m must be finite")
        for name in (
            "start_yaw_rad",
            "selected_hypothesis_mass",
            "target_distance_m",
            "terminal_yaw_rad",
            "initial_heading_error_rad",
            "path_cost_configuration_steps",
            "path_cost_m",
        ):
            _finite(getattr(self, name), name)
        if not 0.0 < self.selected_hypothesis_mass <= 1.0:
            raise ValueError("hypothesis mass is invalid")
        if not 0.10 - 1e-12 <= self.target_distance_m <= 1.20 + 1e-12:
            raise ValueError("target view distance is outside the claim contract")
        expected_steps = float(len(path) - 1)
        if not math.isclose(
            self.path_cost_configuration_steps,
            expected_steps,
            rel_tol=0.0,
            abs_tol=1e-12,
        ) or not math.isclose(
            self.path_cost_m,
            expected_steps * CONFIGURATION_CELL_SIZE_M,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise ValueError("target route path cost changed")
        score = tuple(_finite(value, "score_key") for value in self.score_key)
        if len(score) != 8:
            raise ValueError("target route score key changed")
        if type(self.exact_sim_tainted) is not bool:
            raise TypeError("exact_sim_tainted must be boolean")
        if self.production_promotion_authorized is not False:
            raise PermissionError("target route cannot authorize production promotion")
        if self.hardware_execution_authorized is not False:
            raise PermissionError("target route cannot authorize hardware execution")
        if self._issuance_capability is None:
            raise TypeError("target route receipt requires an issuance capability")
        object.__setattr__(self, "start_cell", start)
        object.__setattr__(self, "goal_cell", goal)
        object.__setattr__(self, "selected_hypothesis_peak_cell", peak)
        object.__setattr__(self, "selected_hypothesis_cells", hypothesis)
        object.__setattr__(self, "path_cells", path)
        object.__setattr__(self, "target_world_xy_m", target_xy)
        object.__setattr__(self, "score_key", score)
        object.__setattr__(self, "content_sha256", _sha256(self.to_dict(False)))

    def to_dict(self, include_hash: bool = True) -> dict[str, object]:
        result: dict[str, object] = {
            "schema": "lewm_g5_two_resolution_target_route_receipt_v1",
            "router_config_sha256": self.router_config_sha256,
            "target_id": self.target_id,
            "target_posterior_sha256": self.target_posterior_sha256,
            "target_memory_instance_sha256": self.target_memory_instance_sha256,
            "target_memory_revision": self.target_memory_revision,
            "target_evidence_chain_sha256": self.target_evidence_chain_sha256,
            "snapshot_sha256": self.snapshot_sha256,
            "component_sha256": self.component_sha256,
            "physical_map_frame_sha256": self.physical_map_frame_sha256,
            "configuration_map_frame_sha256": self.configuration_map_frame_sha256,
            "physical_revision": self.physical_revision,
            "configuration_revision": self.configuration_revision,
            "profile_sha256": self.profile_sha256,
            "free_support_sha256": self.free_support_sha256,
            "occupied_support_sha256": self.occupied_support_sha256,
            "start_cell": list(self.start_cell),
            "start_yaw_rad": self.start_yaw_rad,
            "selected_hypothesis_cells": _cells_json(
                self.selected_hypothesis_cells
            ),
            "selected_hypothesis_mass": self.selected_hypothesis_mass,
            "selected_hypothesis_peak_cell": list(
                self.selected_hypothesis_peak_cell
            ),
            "candidate_set_sha256": self.candidate_set_sha256,
            "candidate_count": self.candidate_count,
            "goal_cell": list(self.goal_cell),
            "target_world_xy_m": list(self.target_world_xy_m),
            "target_distance_m": self.target_distance_m,
            "terminal_yaw_rad": self.terminal_yaw_rad,
            "initial_heading_error_rad": self.initial_heading_error_rad,
            "path_sha256": self.path_sha256,
            "path_cells": [list(cell) for cell in self.path_cells],
            "path_cost_configuration_steps": self.path_cost_configuration_steps,
            "path_cost_m": self.path_cost_m,
            "score_key": list(self.score_key),
            "exact_sim_tainted": self.exact_sim_tainted,
            "production_promotion_authorized": self.production_promotion_authorized,
            "hardware_execution_authorized": self.hardware_execution_authorized,
        }
        if include_hash:
            result["content_sha256"] = self.content_sha256
        return result

    def assert_integrity(self) -> None:
        if self.content_sha256 != _sha256(self.to_dict(False)):
            raise TwoResolutionTargetRouteBindingError("target route receipt was mutated")

    def __copy__(self) -> "TwoResolutionTargetRouteReceiptV1":
        raise TypeError("target route receipts are non-copyable")

    def __deepcopy__(self, memo: object) -> "TwoResolutionTargetRouteReceiptV1":
        del memo
        raise TypeError("target route receipts are non-copyable")


@dataclass(frozen=True)
class TwoResolutionTargetRoutePlanV1:
    path: ConfigurationPathV2
    receipt: TwoResolutionTargetRouteReceiptV1

    def __post_init__(self) -> None:
        if type(self.path) is not ConfigurationPathV2:
            raise TypeError("route plan path has the wrong exact type")
        if type(self.receipt) is not TwoResolutionTargetRouteReceiptV1:
            raise TypeError("route plan receipt has the wrong exact type")
        if _path_sha256(self.path) != self.receipt.path_sha256:
            raise TwoResolutionTargetRouteBindingError(
                "route plan path differs from its receipt"
            )

    @property
    def production_promotion_authorized(self) -> bool:
        return False

    @property
    def hardware_execution_authorized(self) -> bool:
        return False


class TwoResolutionDeterministicTargetRouterV1:
    """Select a confirmed-free target viewing pose and retain its exact path."""

    def __init__(
        self,
        *,
        projection: TwoResolutionConfigurationProjectionV2,
        planner: TwoResolutionConfigurationPlannerV2,
        target_memory: TwoResolutionReversibleTargetBeliefMemoryV1,
        config: TwoResolutionTargetRouterConfigV1 | None = None,
    ) -> None:
        if type(projection) is not TwoResolutionConfigurationProjectionV2:
            raise TypeError("projection has the wrong exact type")
        if type(planner) is not TwoResolutionConfigurationPlannerV2:
            raise TypeError("planner has the wrong exact type")
        if getattr(planner, "_projection", None) is not projection:
            raise TwoResolutionTargetRouteBindingError(
                "planner and projection instances differ"
            )
        if type(target_memory) is not TwoResolutionReversibleTargetBeliefMemoryV1:
            raise TypeError("target_memory has the wrong exact type")
        supplied = config or TwoResolutionTargetRouterConfigV1()
        if type(supplied) is not TwoResolutionTargetRouterConfigV1:
            raise TypeError("config has the wrong exact type")
        self._projection = projection
        self._planner = planner
        self._memory = target_memory
        self._config = supplied
        self._capability = object()
        self._plans: dict[int, TwoResolutionTargetRoutePlanV1] = {}
        self._consumed: set[int] = set()

    def __copy__(self) -> "TwoResolutionDeterministicTargetRouterV1":
        raise TypeError("target routers are non-copyable")

    def __deepcopy__(self, memo: object) -> "TwoResolutionDeterministicTargetRouterV1":
        del memo
        raise TypeError("target routers are non-copyable")

    @staticmethod
    def _bindings_match(
        snapshot: TwoResolutionConfigurationSnapshotV2,
        component: ConfigurationComponentV2,
        posterior: TwoResolutionTargetPosteriorSnapshotV1,
    ) -> bool:
        return bool(
            posterior.configuration_snapshot_sha256 == snapshot.content_sha256
            and posterior.configuration_component_sha256 == component.content_sha256
            and posterior.physical_map_frame.content_sha256
            == snapshot.physical_map_frame_sha256
            and posterior.configuration_map_frame.content_sha256
            == snapshot.configuration_map_frame_sha256
            and posterior.physical_revision == snapshot.physical_revision
            and posterior.configuration_revision == snapshot.configuration_revision
            and posterior.profile_sha256 == snapshot.profile_sha256
            and posterior.free_support_sha256 == snapshot.free_support_sha256
            and posterior.occupied_support_sha256 == snapshot.occupied_support_sha256
            and posterior.projection_source_sha256 == snapshot.projection_source_sha256
            and posterior.candidate_domain == component.cells
            - posterior.excluded_target_configuration_cells
        )

    def _candidates(
        self,
        snapshot: TwoResolutionConfigurationSnapshotV2,
        component: ConfigurationComponentV2,
        hypothesis: TwoResolutionTargetHypothesisV1,
        excluded: frozenset[Cell],
    ) -> tuple[tuple[Cell, float, float], ...]:
        target_x, target_y = hypothesis.mean_world_xy_m
        rows: list[tuple[Cell, float, float]] = []
        forbidden = set(hypothesis.cells) | set(excluded)
        for cell in sorted(component.cells):
            if cell in forbidden or snapshot.state(cell) is not PhysicalLabel.FREE:
                continue
            x, y = snapshot.configuration_map_frame.cell_center(cell)
            distance = math.hypot(target_x - x, target_y - y)
            if (
                self._config.minimum_view_distance_m - 1e-12
                <= distance
                <= self._config.maximum_claim_distance_m + 1e-12
            ):
                yaw = math.atan2(target_y - y, target_x - x)
                rows.append((cell, distance, yaw))
        return tuple(rows)

    def issue(
        self,
        *,
        snapshot: TwoResolutionConfigurationSnapshotV2,
        component: ConfigurationComponentV2,
        posterior: TwoResolutionTargetPosteriorSnapshotV1,
        start_cell: Sequence[int],
        start_yaw_rad: float,
    ) -> TwoResolutionTargetRoutePlanV1:
        self._projection.assert_current_snapshot(snapshot)
        self._planner.validate_component(snapshot, component)
        self._memory.assert_current_snapshot(posterior)
        if not self._bindings_match(snapshot, component, posterior):
            raise TwoResolutionTargetRouteBindingError(
                "posterior does not bind the exact current G3 V2 component"
            )
        start = _cell(start_cell, "start_cell")
        start_yaw = _wrap(_finite(start_yaw_rad, "start_yaw_rad"))
        if start not in component.cells or snapshot.state(start) is not PhysicalLabel.FREE:
            raise NoSafeTargetRouteError("route start is not confirmed FREE")
        hypotheses = self._memory.hypotheses(posterior)
        if not hypotheses:
            raise NoSafeTargetRouteError("posterior has no localized target hypothesis")

        options: list[
            tuple[
                tuple[float, ...],
                TwoResolutionTargetHypothesisV1,
                tuple[tuple[Cell, float, float], ...],
                Cell,
                float,
                float,
                ConfigurationPathV2,
            ]
        ] = []
        for hypothesis_index, hypothesis in enumerate(hypotheses):
            candidates = self._candidates(
                snapshot,
                component,
                hypothesis,
                posterior.excluded_target_configuration_cells,
            )
            for cell, distance, terminal_yaw in candidates:
                path = self._planner.astar(snapshot, start, cell)
                if path is None:
                    continue
                self._planner.validate_path(snapshot, path)
                if any(
                    path_cell in hypothesis.cells
                    or path_cell in posterior.excluded_target_configuration_cells
                    for path_cell in path.cells
                ):
                    continue
                heading_error = abs(_wrap(terminal_yaw - start_yaw))
                score = (
                    float(hypothesis_index),
                    -hypothesis.mass,
                    path.cost,
                    distance,
                    heading_error,
                    float(cell[0]),
                    float(cell[1]),
                    terminal_yaw,
                )
                options.append(
                    (
                        score,
                        hypothesis,
                        candidates,
                        cell,
                        distance,
                        terminal_yaw,
                        path,
                    )
                )
        if not options:
            raise NoSafeTargetRouteError("no confirmed-free target viewing route exists")
        score, hypothesis, candidates, goal, distance, terminal_yaw, path = min(
            options, key=lambda row: row[0]
        )
        candidate_hash = _sha256(
            {
                "schema": "lewm_g5_two_resolution_target_route_candidates_v1",
                "snapshot_sha256": snapshot.content_sha256,
                "posterior_sha256": posterior.content_sha256,
                "hypothesis_cells": _cells_json(hypothesis.cells),
                "rows": [
                    {
                        "cell": list(cell),
                        "target_distance_m": candidate_distance,
                        "terminal_yaw_rad": yaw,
                    }
                    for cell, candidate_distance, yaw in candidates
                ],
            }
        )
        receipt = TwoResolutionTargetRouteReceiptV1(
            router_config_sha256=self._config.content_sha256,
            target_id=posterior.target_id,
            target_posterior_sha256=posterior.content_sha256,
            target_memory_instance_sha256=posterior.target_memory_instance_sha256,
            target_memory_revision=posterior.target_memory_revision,
            target_evidence_chain_sha256=posterior.evidence_chain_sha256,
            snapshot_sha256=snapshot.content_sha256,
            component_sha256=component.content_sha256,
            physical_map_frame_sha256=snapshot.physical_map_frame_sha256,
            configuration_map_frame_sha256=snapshot.configuration_map_frame_sha256,
            physical_revision=snapshot.physical_revision,
            configuration_revision=snapshot.configuration_revision,
            profile_sha256=snapshot.profile_sha256,
            free_support_sha256=snapshot.free_support_sha256,
            occupied_support_sha256=snapshot.occupied_support_sha256,
            start_cell=start,
            start_yaw_rad=start_yaw,
            selected_hypothesis_cells=hypothesis.cells,
            selected_hypothesis_mass=hypothesis.mass,
            selected_hypothesis_peak_cell=hypothesis.peak_cell,
            candidate_set_sha256=candidate_hash,
            candidate_count=len(candidates),
            goal_cell=goal,
            target_world_xy_m=hypothesis.mean_world_xy_m,
            target_distance_m=distance,
            terminal_yaw_rad=terminal_yaw,
            initial_heading_error_rad=abs(_wrap(terminal_yaw - start_yaw)),
            path_sha256=_path_sha256(path),
            path_cells=path.cells,
            path_cost_configuration_steps=path.cost,
            path_cost_m=path.cost * CONFIGURATION_CELL_SIZE_M,
            score_key=score,
            exact_sim_tainted=posterior.exact_sim_tainted,
            production_promotion_authorized=False,
            hardware_execution_authorized=False,
            _issuance_capability=self._capability,
        )
        plan = TwoResolutionTargetRoutePlanV1(path=path, receipt=receipt)
        self._plans[id(plan)] = plan
        return plan

    def validate(
        self,
        *,
        snapshot: TwoResolutionConfigurationSnapshotV2,
        component: ConfigurationComponentV2,
        posterior: TwoResolutionTargetPosteriorSnapshotV1,
        plan: TwoResolutionTargetRoutePlanV1,
        consume: bool = False,
    ) -> None:
        if type(plan) is not TwoResolutionTargetRoutePlanV1:
            raise TypeError("plan has the wrong exact type")
        self._projection.assert_current_snapshot(snapshot)
        self._planner.validate_component(snapshot, component)
        self._memory.assert_current_snapshot(posterior)
        if self._plans.get(id(plan)) is not plan:
            raise TwoResolutionTargetRouteBindingError(
                "route plan is not the exact live object issued here"
            )
        if id(plan) in self._consumed:
            raise TwoResolutionTargetRouteBindingError("route plan was already consumed")
        plan.receipt.assert_integrity()
        self._planner.validate_path(snapshot, plan.path)
        if (
            not self._bindings_match(snapshot, component, posterior)
            or plan.receipt.target_posterior_sha256 != posterior.content_sha256
            or plan.receipt.snapshot_sha256 != snapshot.content_sha256
            or plan.receipt.component_sha256 != component.content_sha256
            or _path_sha256(plan.path) != plan.receipt.path_sha256
            or any(snapshot.state(cell) is not PhysicalLabel.FREE for cell in plan.path.cells)
            or any(
                cell in plan.receipt.selected_hypothesis_cells
                or cell in posterior.excluded_target_configuration_cells
                for cell in plan.path.cells
            )
            or plan.receipt.goal_cell in plan.receipt.selected_hypothesis_cells
            or plan.receipt.goal_cell
            in posterior.excluded_target_configuration_cells
        ):
            raise TwoResolutionTargetRouteBindingError(
                "route plan differs from current posterior or map state"
            )
        if consume:
            self._consumed.add(id(plan))


def require_production_two_resolution_target_router() -> object:
    if PRODUCTION_TWO_RESOLUTION_TARGET_ROUTER is None:
        raise PermissionError("production two-resolution target router is unset")
    return PRODUCTION_TWO_RESOLUTION_TARGET_ROUTER


__all__ = [
    "NoSafeTargetRouteError",
    "PRODUCTION_TWO_RESOLUTION_TARGET_ROUTER",
    "TwoResolutionDeterministicTargetRouterV1",
    "TwoResolutionTargetRouteBindingError",
    "TwoResolutionTargetRoutePlanV1",
    "TwoResolutionTargetRouteReceiptV1",
    "TwoResolutionTargetRouterConfigV1",
    "TwoResolutionTargetRouterError",
    "require_production_two_resolution_target_router",
]
