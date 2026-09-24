"""All-hypothesis-safe successor to the deterministic target router V1.

V1 excluded only the selected posterior mode from its retained path.  This
additive successor binds the union of every current posterior hypothesis and
rejects any terminal or path cell in that union.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
import math
from typing import Iterable, Sequence

from lewm.planning.revisioned_physical_configuration_memory import PhysicalLabel
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
from lewm.planning.two_resolution_target_router_v1 import (
    NoSafeTargetRouteError,
    TwoResolutionTargetRouteBindingError,
    TwoResolutionTargetRoutePlanV1,
    TwoResolutionTargetRouteReceiptV1,
)


Cell = tuple[int, int]

PRODUCTION_TWO_RESOLUTION_TARGET_ROUTER_V2 = None


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
class TwoResolutionTargetRouterConfigV2:
    minimum_view_distance_m: float = 0.10
    maximum_claim_distance_m: float = 1.20
    claim_bearing_tolerance_rad: float = 0.25
    forbid_all_posterior_hypothesis_cells: bool = True
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
        if self.forbid_all_posterior_hypothesis_cells is not True:
            raise PermissionError("V2 must forbid every posterior hypothesis cell")
        object.__setattr__(self, "minimum_view_distance_m", minimum)
        object.__setattr__(self, "maximum_claim_distance_m", maximum)
        object.__setattr__(self, "claim_bearing_tolerance_rad", bearing)
        object.__setattr__(self, "content_sha256", _sha256(self.to_dict(False)))

    def to_dict(self, include_hash: bool = True) -> dict[str, object]:
        result: dict[str, object] = {
            "schema": "lewm_g5_two_resolution_target_router_config_v2",
            "minimum_view_distance_m": self.minimum_view_distance_m,
            "maximum_claim_distance_m": self.maximum_claim_distance_m,
            "claim_bearing_tolerance_rad": self.claim_bearing_tolerance_rad,
            "route_cell_size_m": CONFIGURATION_CELL_SIZE_M,
            "forbid_all_posterior_hypothesis_cells": (
                self.forbid_all_posterior_hypothesis_cells
            ),
            "production_promotion_authorized": False,
            "hardware_execution_authorized": False,
        }
        if include_hash:
            result["content_sha256"] = self.content_sha256
        return result


@dataclass(frozen=True)
class TwoResolutionTargetRoutePlanV2:
    retained_v1_plan: TwoResolutionTargetRoutePlanV1
    router_config_sha256: str
    target_posterior_sha256: str
    all_hypothesis_cells: frozenset[Cell]
    all_hypothesis_cells_sha256: str
    production_promotion_authorized: bool
    hardware_execution_authorized: bool
    _issuance_capability: object = field(repr=False, compare=False)
    content_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        if type(self.retained_v1_plan) is not TwoResolutionTargetRoutePlanV1:
            raise TypeError("V2 requires an exact retained V1 path/receipt container")
        for value, name in (
            (self.router_config_sha256, "router_config_sha256"),
            (self.target_posterior_sha256, "target_posterior_sha256"),
            (self.all_hypothesis_cells_sha256, "all_hypothesis_cells_sha256"),
        ):
            if not (
                type(value) is str
                and len(value) == 64
                and all(character in "0123456789abcdef" for character in value)
            ):
                raise ValueError(f"{name} must be a lowercase SHA-256")
        cells = frozenset(
            _cell(cell, "all hypothesis cell") for cell in self.all_hypothesis_cells
        )
        if not cells:
            raise ValueError("V2 route requires at least one hypothesis cell")
        expected = _sha256(
            {
                "schema": "lewm_g5_two_resolution_all_hypothesis_cells_v2",
                "target_posterior_sha256": self.target_posterior_sha256,
                "cells": _cells_json(cells),
            }
        )
        if expected != self.all_hypothesis_cells_sha256:
            raise TwoResolutionTargetRouteBindingError(
                "V2 all-hypothesis commitment changed"
            )
        path_cells = set(self.retained_v1_plan.path.cells)
        if path_cells & cells:
            raise TwoResolutionTargetRouteBindingError(
                "V2 path crosses a posterior hypothesis cell"
            )
        if self.production_promotion_authorized is not False:
            raise PermissionError("V2 route cannot authorize production promotion")
        if self.hardware_execution_authorized is not False:
            raise PermissionError("V2 route cannot authorize hardware execution")
        if self._issuance_capability is None:
            raise TypeError("V2 route requires an issuance capability")
        object.__setattr__(self, "all_hypothesis_cells", cells)
        object.__setattr__(self, "content_sha256", _sha256(self.to_dict(False)))

    @property
    def path(self) -> ConfigurationPathV2:
        return self.retained_v1_plan.path

    @property
    def receipt(self) -> TwoResolutionTargetRouteReceiptV1:
        return self.retained_v1_plan.receipt

    def to_dict(self, include_hash: bool = True) -> dict[str, object]:
        result: dict[str, object] = {
            "schema": "lewm_g5_two_resolution_target_route_plan_v2",
            "retained_v1_receipt": self.receipt.to_dict(),
            "router_config_sha256": self.router_config_sha256,
            "target_posterior_sha256": self.target_posterior_sha256,
            "all_hypothesis_cells": _cells_json(self.all_hypothesis_cells),
            "all_hypothesis_cells_sha256": self.all_hypothesis_cells_sha256,
            "forbid_all_posterior_hypothesis_cells": True,
            "production_promotion_authorized": self.production_promotion_authorized,
            "hardware_execution_authorized": self.hardware_execution_authorized,
        }
        if include_hash:
            result["content_sha256"] = self.content_sha256
        return result

    def assert_integrity(self) -> None:
        self.receipt.assert_integrity()
        if (
            self.production_promotion_authorized is not False
            or self.hardware_execution_authorized is not False
            or self.receipt.production_promotion_authorized is not False
            or self.receipt.hardware_execution_authorized is not False
        ):
            raise TwoResolutionTargetRouteBindingError(
                "V2 route authority denial was changed"
            )
        if (
            self.receipt.router_config_sha256 != self.router_config_sha256
            or self.receipt.target_posterior_sha256
            != self.target_posterior_sha256
            or set(self.path.cells) & self.all_hypothesis_cells
        ):
            raise TwoResolutionTargetRouteBindingError(
                "V2 retained route binding changed"
            )
        if self.content_sha256 != _sha256(self.to_dict(False)):
            raise TwoResolutionTargetRouteBindingError("V2 route plan was mutated")

    def __copy__(self) -> "TwoResolutionTargetRoutePlanV2":
        raise TypeError("V2 target route plans are non-copyable")

    def __deepcopy__(self, memo: object) -> "TwoResolutionTargetRoutePlanV2":
        del memo
        raise TypeError("V2 target route plans are non-copyable")


class TwoResolutionDeterministicTargetRouterV2:
    """Retain exact G3 paths that avoid every live posterior hypothesis."""

    def __init__(
        self,
        *,
        projection: TwoResolutionConfigurationProjectionV2,
        planner: TwoResolutionConfigurationPlannerV2,
        target_memory: TwoResolutionReversibleTargetBeliefMemoryV1,
        config: TwoResolutionTargetRouterConfigV2 | None = None,
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
        supplied = config or TwoResolutionTargetRouterConfigV2()
        if type(supplied) is not TwoResolutionTargetRouterConfigV2:
            raise TypeError("config has the wrong exact type")
        self._projection = projection
        self._planner = planner
        self._memory = target_memory
        self._config = supplied
        self._capability = object()
        self._plans: dict[int, TwoResolutionTargetRoutePlanV2] = {}
        self._issued_content_sha256: dict[int, str] = {}
        self._consumed: set[int] = set()

    def __copy__(self) -> "TwoResolutionDeterministicTargetRouterV2":
        raise TypeError("V2 target routers are non-copyable")

    def __deepcopy__(self, memo: object) -> "TwoResolutionDeterministicTargetRouterV2":
        del memo
        raise TypeError("V2 target routers are non-copyable")

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
            and posterior.candidate_domain
            == component.cells - posterior.excluded_target_configuration_cells
        )

    def _candidates(
        self,
        snapshot: TwoResolutionConfigurationSnapshotV2,
        component: ConfigurationComponentV2,
        hypothesis: TwoResolutionTargetHypothesisV1,
        forbidden: frozenset[Cell],
    ) -> tuple[tuple[Cell, float, float], ...]:
        target_x, target_y = hypothesis.mean_world_xy_m
        rows: list[tuple[Cell, float, float]] = []
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
                rows.append((cell, distance, math.atan2(target_y - y, target_x - x)))
        return tuple(rows)

    def issue(
        self,
        *,
        snapshot: TwoResolutionConfigurationSnapshotV2,
        component: ConfigurationComponentV2,
        posterior: TwoResolutionTargetPosteriorSnapshotV1,
        start_cell: Sequence[int],
        start_yaw_rad: float,
    ) -> TwoResolutionTargetRoutePlanV2:
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
        all_hypothesis_cells = frozenset(
            cell for hypothesis in hypotheses for cell in hypothesis.cells
        )
        forbidden = all_hypothesis_cells | posterior.excluded_target_configuration_cells
        union_hash = _sha256(
            {
                "schema": "lewm_g5_two_resolution_all_hypothesis_cells_v2",
                "target_posterior_sha256": posterior.content_sha256,
                "cells": _cells_json(all_hypothesis_cells),
            }
        )

        options: list[tuple[tuple[float, ...], object, tuple, Cell, float, float, ConfigurationPathV2]] = []
        for hypothesis_index, hypothesis in enumerate(hypotheses):
            candidates = self._candidates(snapshot, component, hypothesis, forbidden)
            for cell, distance, terminal_yaw in candidates:
                path = self._planner.astar(snapshot, start, cell)
                if path is None:
                    continue
                self._planner.validate_path(snapshot, path)
                if set(path.cells) & forbidden:
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
                    (score, hypothesis, candidates, cell, distance, terminal_yaw, path)
                )
        if not options:
            raise NoSafeTargetRouteError(
                "no confirmed-free route avoids every posterior hypothesis"
            )
        score, selected, candidates, goal, distance, terminal_yaw, path = min(
            options, key=lambda row: row[0]
        )
        if not isinstance(selected, TwoResolutionTargetHypothesisV1):
            raise AssertionError("selected hypothesis type changed")
        candidate_hash = _sha256(
            {
                "schema": "lewm_g5_two_resolution_target_route_candidates_v2",
                "snapshot_sha256": snapshot.content_sha256,
                "posterior_sha256": posterior.content_sha256,
                "selected_hypothesis_cells": _cells_json(selected.cells),
                "all_hypothesis_cells_sha256": union_hash,
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
            selected_hypothesis_cells=selected.cells,
            selected_hypothesis_mass=selected.mass,
            selected_hypothesis_peak_cell=selected.peak_cell,
            candidate_set_sha256=candidate_hash,
            candidate_count=len(candidates),
            goal_cell=goal,
            target_world_xy_m=selected.mean_world_xy_m,
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
        retained = TwoResolutionTargetRoutePlanV1(path=path, receipt=receipt)
        plan = TwoResolutionTargetRoutePlanV2(
            retained_v1_plan=retained,
            router_config_sha256=self._config.content_sha256,
            target_posterior_sha256=posterior.content_sha256,
            all_hypothesis_cells=all_hypothesis_cells,
            all_hypothesis_cells_sha256=union_hash,
            production_promotion_authorized=False,
            hardware_execution_authorized=False,
            _issuance_capability=self._capability,
        )
        self._plans[id(plan)] = plan
        self._issued_content_sha256[id(plan)] = plan.content_sha256
        return plan

    def validate(
        self,
        *,
        snapshot: TwoResolutionConfigurationSnapshotV2,
        component: ConfigurationComponentV2,
        posterior: TwoResolutionTargetPosteriorSnapshotV1,
        plan: TwoResolutionTargetRoutePlanV2,
        consume: bool = False,
    ) -> None:
        if type(plan) is not TwoResolutionTargetRoutePlanV2:
            raise TypeError("plan has the wrong exact V2 type")
        self._projection.assert_current_snapshot(snapshot)
        self._planner.validate_component(snapshot, component)
        self._memory.assert_current_snapshot(posterior)
        if self._plans.get(id(plan)) is not plan:
            raise TwoResolutionTargetRouteBindingError(
                "V2 route plan is not the exact live object issued here"
            )
        if id(plan) in self._consumed:
            raise TwoResolutionTargetRouteBindingError("V2 route was already consumed")
        plan.assert_integrity()
        if (
            self._issued_content_sha256.get(id(plan)) != plan.content_sha256
            or plan.production_promotion_authorized is not False
            or plan.hardware_execution_authorized is not False
            or plan.receipt.production_promotion_authorized is not False
            or plan.receipt.hardware_execution_authorized is not False
        ):
            raise TwoResolutionTargetRouteBindingError(
                "V2 issued route content or authority denial changed"
            )
        self._planner.validate_path(snapshot, plan.path)
        hypotheses = self._memory.hypotheses(posterior)
        expected_union = frozenset(
            cell for hypothesis in hypotheses for cell in hypothesis.cells
        )
        if (
            not self._bindings_match(snapshot, component, posterior)
            or plan.target_posterior_sha256 != posterior.content_sha256
            or plan.all_hypothesis_cells != expected_union
            or plan.receipt.target_posterior_sha256 != posterior.content_sha256
            or plan.receipt.snapshot_sha256 != snapshot.content_sha256
            or plan.receipt.component_sha256 != component.content_sha256
            or _path_sha256(plan.path) != plan.receipt.path_sha256
            or any(snapshot.state(cell) is not PhysicalLabel.FREE for cell in plan.path.cells)
            or set(plan.path.cells) & expected_union
            or set(plan.path.cells)
            & posterior.excluded_target_configuration_cells
        ):
            raise TwoResolutionTargetRouteBindingError(
                "V2 route differs from current all-hypothesis-safe state"
            )
        if consume:
            self._consumed.add(id(plan))


def require_production_two_resolution_target_router_v2() -> object:
    if PRODUCTION_TWO_RESOLUTION_TARGET_ROUTER_V2 is None:
        raise PermissionError("production two-resolution target router V2 is unset")
    return PRODUCTION_TWO_RESOLUTION_TARGET_ROUTER_V2


__all__ = [
    "PRODUCTION_TWO_RESOLUTION_TARGET_ROUTER_V2",
    "TwoResolutionDeterministicTargetRouterV2",
    "TwoResolutionTargetRoutePlanV2",
    "TwoResolutionTargetRouterConfigV2",
    "require_production_two_resolution_target_router_v2",
]
