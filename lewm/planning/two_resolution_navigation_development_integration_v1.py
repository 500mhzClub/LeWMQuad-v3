"""Development-only composition of the passed two-resolution navigation APIs.

This module adds no planning, projection, evidence, posterior, or claim-scoring
logic.  It seals the exact receipts from those owners into one controller-side
claim trace, then permits one observer-only evaluation after controller work is
complete.  The observer import is deliberately lazy so issuing a controller
trace cannot load or call the canonical evaluator.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
import math
from typing import Any, Mapping, Sequence

from lewm.benchmarks.go2_physical_claim_trace import (
    build_claim_attempt,
    build_claim_trace,
    object_id_reference,
)
from lewm.planning.two_resolution_configuration_projection_v2 import (
    ConfigurationComponentV2,
    TwoResolutionConfigurationPlannerV2,
    TwoResolutionConfigurationProjectionV2,
    TwoResolutionConfigurationSnapshotV2,
)
from lewm.planning.two_resolution_frontier_viewpoint_v2 import (
    TwoResolutionFrontierViewpointPlannerV2,
    TwoResolutionPhysicalViewStateIssuerV2,
)
from lewm.planning.two_resolution_reversible_target_belief_v1 import (
    TwoResolutionReversibleTargetBeliefMemoryV1,
)
from lewm.planning.two_resolution_target_evidence_v1 import (
    SyntheticV5TargetOutcomeV1,
    TwoResolutionTargetEvidenceIssuerV1,
)
from lewm.planning.two_resolution_target_router_v2 import (
    TwoResolutionDeterministicTargetRouterV2,
)
from lewm.planning.two_resolution_world_waypoint_adapter_v2 import (
    ConfigurationPathWorldWaypointIssuerV2,
)
from lewm_worlds.manifest import SceneManifest, manifest_sha256


PRODUCTION_TWO_RESOLUTION_NAVIGATION_INTEGRATION_V1 = None


class TwoResolutionNavigationIntegrationError(ValueError):
    """Base failure for the development integration boundary."""


class TwoResolutionNavigationIntegrationBindingError(
    TwoResolutionNavigationIntegrationError
):
    """Raised when an exact object, hash, or owner binding changed."""


class TwoResolutionNavigationIntegrationReplayError(
    TwoResolutionNavigationIntegrationError
):
    """Raised when a sealed controller trace is observed more than once."""


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


def _plain_json(value: object) -> object:
    return json.loads(_canonical_bytes(value).decode("ascii"))


def _plain_mapping(value: object, name: str) -> dict[str, object]:
    result = _plain_json(value)
    if type(result) is not dict:
        raise TypeError(f"{name} must be a JSON object")
    return result


def _require_sha256(value: object, name: str) -> str:
    if not (
        type(value) is str
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256")
    return value


def _nonempty(value: object, name: str) -> str:
    if type(value) is not str or not value:
        raise ValueError(f"{name} must be a nonempty string")
    return value


def _nonnegative_int(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return value


def _finite(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _assert_authority_denials(value: Mapping[str, object], name: str) -> None:
    if (
        value.get("production_promotion_authorized") is not False
        or value.get("hardware_execution_authorized") is not False
    ):
        raise TwoResolutionNavigationIntegrationBindingError(
            f"{name} authority denial changed"
        )


@dataclass(frozen=True)
class TwoResolutionDevelopmentControllerClaimTraceV1:
    """Sealed controller output with hashes for every retained authority."""

    controller_attempt_sequence: int
    scene_id: str
    physical_manifest_sha256: str
    snapshot_sha256: str
    component_sha256: str
    physical_map_frame_sha256: str
    configuration_map_frame_sha256: str
    physical_revision: int
    configuration_revision: int
    physical_view_state_sha256: str
    frontier_candidate_set_sha256: str
    selected_frontier_candidate_sha256: str
    raw_outcome_content_sha256: str
    target_context_sha256: str
    target_evidence_sha256: str
    target_posterior_sha256: str
    target_route_plan_sha256: str
    target_route_receipt_sha256: str
    world_waypoint_receipt_sha256: str
    target_id: str
    task_object_id: str
    target_route_plan: Mapping[str, object]
    world_waypoint_receipt: Mapping[str, object]
    controller_claim_attempt: Mapping[str, Any]
    raw_claim_trace: Mapping[str, Any]
    task_object_ids: tuple[str, ...]
    task_object_set_sha256: str
    production_promotion_authorized: bool
    hardware_execution_authorized: bool
    _issuance_capability: object = field(repr=False, compare=False)
    content_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        if _nonnegative_int(
            self.controller_attempt_sequence, "controller_attempt_sequence"
        ) == 0:
            raise ValueError("controller_attempt_sequence must be positive")
        _nonempty(self.scene_id, "scene_id")
        _nonempty(self.target_id, "target_id")
        _nonempty(self.task_object_id, "task_object_id")
        for name in (
            "physical_manifest_sha256",
            "snapshot_sha256",
            "component_sha256",
            "physical_map_frame_sha256",
            "configuration_map_frame_sha256",
            "physical_view_state_sha256",
            "frontier_candidate_set_sha256",
            "selected_frontier_candidate_sha256",
            "raw_outcome_content_sha256",
            "target_context_sha256",
            "target_evidence_sha256",
            "target_posterior_sha256",
            "target_route_plan_sha256",
            "target_route_receipt_sha256",
            "world_waypoint_receipt_sha256",
            "task_object_set_sha256",
        ):
            _require_sha256(getattr(self, name), name)
        _nonnegative_int(self.physical_revision, "physical_revision")
        _nonnegative_int(self.configuration_revision, "configuration_revision")
        if (
            self.production_promotion_authorized is not False
            or self.hardware_execution_authorized is not False
        ):
            raise PermissionError(
                "development controller traces cannot authorize production or hardware"
            )
        if self._issuance_capability is None:
            raise TypeError("controller claim trace requires an issuance capability")

        task_ids = tuple(self.task_object_ids)
        if (
            not task_ids
            or any(type(value) is not str or not value for value in task_ids)
            or tuple(sorted(task_ids, key=lambda value: value.encode("utf-8")))
            != task_ids
            or len(set(task_ids)) != len(task_ids)
            or self.task_object_id not in task_ids
        ):
            raise ValueError("task_object_ids are not a canonical bound task set")

        route = _plain_mapping(self.target_route_plan, "target_route_plan")
        waypoint = _plain_mapping(
            self.world_waypoint_receipt, "world_waypoint_receipt"
        )
        attempt = _plain_mapping(
            self.controller_claim_attempt, "controller_claim_attempt"
        )
        trace = _plain_mapping(self.raw_claim_trace, "raw_claim_trace")

        retained = route.get("retained_v1_receipt")
        if type(retained) is not dict:
            raise TwoResolutionNavigationIntegrationBindingError(
                "V2 route omitted its retained V1 receipt"
            )
        _assert_authority_denials(route, "target route")
        _assert_authority_denials(retained, "retained target route receipt")
        _assert_authority_denials(waypoint, "world waypoint receipt")
        if (
            route.get("schema") != "lewm_g5_two_resolution_target_route_plan_v2"
            or route.get("content_sha256") != self.target_route_plan_sha256
            or retained.get("content_sha256") != self.target_route_receipt_sha256
            or retained.get("target_id") != self.target_id
            or retained.get("target_posterior_sha256")
            != self.target_posterior_sha256
            or retained.get("snapshot_sha256") != self.snapshot_sha256
            or retained.get("component_sha256") != self.component_sha256
            or waypoint.get("schema")
            != "lewm_g3_v2_configuration_path_world_waypoint_receipt_v2"
            or waypoint.get("content_sha256")
            != self.world_waypoint_receipt_sha256
            or waypoint.get("snapshot_sha256") != self.snapshot_sha256
            or waypoint.get("physical_map_frame_sha256")
            != self.physical_map_frame_sha256
            or waypoint.get("configuration_map_frame_sha256")
            != self.configuration_map_frame_sha256
            or waypoint.get("physical_revision") != self.physical_revision
            or waypoint.get("configuration_revision")
            != self.configuration_revision
        ):
            raise TwoResolutionNavigationIntegrationBindingError(
                "route or waypoint serialization changed its retained binding"
            )

        requested = attempt.get("requested_target")
        claimed = attempt.get("claimed_target")
        expected_reference = {
            "namespace": "object_id",
            "value": self.task_object_id,
        }
        if (
            attempt.get("scene_id") != self.scene_id
            or attempt.get("physical_manifest_sha256")
            != self.physical_manifest_sha256
            or requested != expected_reference
            or claimed != expected_reference
            or trace.get("scene_id") != self.scene_id
            or trace.get("physical_manifest_sha256")
            != self.physical_manifest_sha256
            or trace.get("task_object_ids") != list(task_ids)
            or trace.get("task_object_set_sha256")
            != self.task_object_set_sha256
            or trace.get("controller_claim_attempts") != [attempt]
            or trace.get("evaluator_feedback_to_controller") != []
        ):
            raise TwoResolutionNavigationIntegrationBindingError(
                "raw controller claim trace changed its sealed target or "
                "observer boundary"
            )

        object.__setattr__(self, "task_object_ids", task_ids)
        object.__setattr__(self, "target_route_plan", route)
        object.__setattr__(self, "world_waypoint_receipt", waypoint)
        object.__setattr__(self, "controller_claim_attempt", attempt)
        object.__setattr__(self, "raw_claim_trace", trace)
        object.__setattr__(self, "content_sha256", _sha256(self.to_dict(False)))

    @property
    def development_execution_eligible(self) -> bool:
        return True

    def to_dict(self, include_hash: bool = True) -> dict[str, object]:
        result: dict[str, object] = {
            "schema": "lewm_two_resolution_development_controller_claim_trace_v1",
            "controller_attempt_sequence": self.controller_attempt_sequence,
            "scene_id": self.scene_id,
            "physical_manifest_sha256": self.physical_manifest_sha256,
            "g3": {
                "snapshot_sha256": self.snapshot_sha256,
                "component_sha256": self.component_sha256,
                "physical_map_frame_sha256": self.physical_map_frame_sha256,
                "configuration_map_frame_sha256": (
                    self.configuration_map_frame_sha256
                ),
                "physical_revision": self.physical_revision,
                "configuration_revision": self.configuration_revision,
            },
            "g4": {
                "physical_view_state_sha256": self.physical_view_state_sha256,
                "frontier_candidate_set_sha256": (
                    self.frontier_candidate_set_sha256
                ),
                "selected_frontier_candidate_sha256": (
                    self.selected_frontier_candidate_sha256
                ),
            },
            "g5": {
                "raw_outcome_content_sha256": self.raw_outcome_content_sha256,
                "target_context_sha256": self.target_context_sha256,
                "target_evidence_sha256": self.target_evidence_sha256,
                "target_posterior_sha256": self.target_posterior_sha256,
                "target_id": self.target_id,
            },
            "task_object_id": self.task_object_id,
            "target_route_plan_sha256": self.target_route_plan_sha256,
            "target_route_receipt_sha256": self.target_route_receipt_sha256,
            "world_waypoint_receipt_sha256": (
                self.world_waypoint_receipt_sha256
            ),
            "target_route_plan": _plain_json(self.target_route_plan),
            "world_waypoint_receipt": _plain_json(
                self.world_waypoint_receipt
            ),
            "controller_claim_attempt": _plain_json(
                self.controller_claim_attempt
            ),
            "raw_claim_trace": _plain_json(self.raw_claim_trace),
            "task_object_ids": list(self.task_object_ids),
            "task_object_set_sha256": self.task_object_set_sha256,
            "development_execution_eligible": True,
            "production_promotion_authorized": (
                self.production_promotion_authorized
            ),
            "hardware_execution_authorized": self.hardware_execution_authorized,
        }
        if include_hash:
            result["content_sha256"] = self.content_sha256
        return result

    def assert_integrity(self) -> None:
        if (
            self.production_promotion_authorized is not False
            or self.hardware_execution_authorized is not False
        ):
            raise TwoResolutionNavigationIntegrationBindingError(
                "controller claim trace authority denial changed"
            )
        if self.content_sha256 != _sha256(self.to_dict(False)):
            raise TwoResolutionNavigationIntegrationBindingError(
                "controller claim trace was mutated"
            )

    def __copy__(self) -> "TwoResolutionDevelopmentControllerClaimTraceV1":
        raise TypeError("development controller claim traces are non-copyable")

    def __deepcopy__(
        self,
        memo: object,
    ) -> "TwoResolutionDevelopmentControllerClaimTraceV1":
        del memo
        raise TypeError("development controller claim traces are non-copyable")

    def __reduce_ex__(self, protocol: int) -> object:
        del protocol
        raise TypeError("development controller claim traces are non-serializable")


@dataclass(frozen=True)
class TwoResolutionObserverClaimEvaluationV1:
    """Observer-owned result; it has no controller callback or control token."""

    controller_claim_trace_sha256: str
    evaluator_access_ledger: Mapping[str, int]
    evaluated_claim_trace: Mapping[str, Any]
    production_promotion_authorized: bool
    hardware_execution_authorized: bool
    _issuance_capability: object = field(repr=False, compare=False)
    content_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        _require_sha256(
            self.controller_claim_trace_sha256,
            "controller_claim_trace_sha256",
        )
        ledger = _plain_mapping(
            self.evaluator_access_ledger, "evaluator_access_ledger"
        )
        expected = {
            "evaluator_output_reads_by_controller": 0,
            "evaluator_callbacks_into_controller": 0,
            "evaluator_derived_termination_signals": 0,
        }
        if ledger != expected or any(
            type(value) is not int for value in ledger.values()
        ):
            raise TwoResolutionNavigationIntegrationBindingError(
                "evaluator access ledger is not actually empty"
            )
        evaluated = _plain_mapping(
            self.evaluated_claim_trace, "evaluated_claim_trace"
        )
        if (
            self.production_promotion_authorized is not False
            or self.hardware_execution_authorized is not False
        ):
            raise PermissionError(
                "observer evaluation cannot authorize production or hardware"
            )
        if self._issuance_capability is None:
            raise TypeError("observer evaluation requires an issuance capability")
        object.__setattr__(self, "evaluator_access_ledger", ledger)
        object.__setattr__(self, "evaluated_claim_trace", evaluated)
        object.__setattr__(self, "content_sha256", _sha256(self.to_dict(False)))

    def to_dict(self, include_hash: bool = True) -> dict[str, object]:
        result: dict[str, object] = {
            "schema": "lewm_two_resolution_observer_claim_evaluation_v1",
            "controller_claim_trace_sha256": self.controller_claim_trace_sha256,
            "observer_only": True,
            "evaluator_access_ledger": _plain_json(self.evaluator_access_ledger),
            "evaluated_claim_trace": _plain_json(self.evaluated_claim_trace),
            "controller_callback": None,
            "production_promotion_authorized": (
                self.production_promotion_authorized
            ),
            "hardware_execution_authorized": self.hardware_execution_authorized,
        }
        if include_hash:
            result["content_sha256"] = self.content_sha256
        return result

    def assert_integrity(self) -> None:
        if (
            self.production_promotion_authorized is not False
            or self.hardware_execution_authorized is not False
            or self.evaluator_access_ledger
            != {
                "evaluator_output_reads_by_controller": 0,
                "evaluator_callbacks_into_controller": 0,
                "evaluator_derived_termination_signals": 0,
            }
        ):
            raise TwoResolutionNavigationIntegrationBindingError(
                "observer boundary or authority denial changed"
            )
        if self.content_sha256 != _sha256(self.to_dict(False)):
            raise TwoResolutionNavigationIntegrationBindingError(
                "observer evaluation was mutated"
            )

    def __copy__(self) -> "TwoResolutionObserverClaimEvaluationV1":
        raise TypeError("observer claim evaluations are non-copyable")

    def __deepcopy__(
        self,
        memo: object,
    ) -> "TwoResolutionObserverClaimEvaluationV1":
        del memo
        raise TypeError("observer claim evaluations are non-copyable")

    def __reduce_ex__(self, protocol: int) -> object:
        del protocol
        raise TypeError("observer claim evaluations are non-serializable")


@dataclass(frozen=True)
class _IssuedControllerTrace:
    artifact: TwoResolutionDevelopmentControllerClaimTraceV1
    original_content_sha256: str
    physical_manifest: SceneManifest


class TwoResolutionDevelopmentNavigationIntegrationV1:
    """One-shot development composition over exact passed authority objects."""

    def __init__(
        self,
        *,
        projection: TwoResolutionConfigurationProjectionV2,
        planner: TwoResolutionConfigurationPlannerV2,
        physical_view_state_issuer: TwoResolutionPhysicalViewStateIssuerV2,
        frontier_viewpoint_planner: TwoResolutionFrontierViewpointPlannerV2,
        target_evidence_issuer: TwoResolutionTargetEvidenceIssuerV1,
        target_memory: TwoResolutionReversibleTargetBeliefMemoryV1,
        target_router: TwoResolutionDeterministicTargetRouterV2,
        world_waypoint_issuer: ConfigurationPathWorldWaypointIssuerV2,
        _synthetic_development_fixture: bool = False,
    ) -> None:
        if _synthetic_development_fixture is not True:
            raise PermissionError(
                "no production two-resolution navigation integration is configured"
            )
        exact_types = (
            (projection, TwoResolutionConfigurationProjectionV2, "projection"),
            (planner, TwoResolutionConfigurationPlannerV2, "planner"),
            (
                physical_view_state_issuer,
                TwoResolutionPhysicalViewStateIssuerV2,
                "physical_view_state_issuer",
            ),
            (
                frontier_viewpoint_planner,
                TwoResolutionFrontierViewpointPlannerV2,
                "frontier_viewpoint_planner",
            ),
            (
                target_evidence_issuer,
                TwoResolutionTargetEvidenceIssuerV1,
                "target_evidence_issuer",
            ),
            (
                target_memory,
                TwoResolutionReversibleTargetBeliefMemoryV1,
                "target_memory",
            ),
            (
                target_router,
                TwoResolutionDeterministicTargetRouterV2,
                "target_router",
            ),
            (
                world_waypoint_issuer,
                ConfigurationPathWorldWaypointIssuerV2,
                "world_waypoint_issuer",
            ),
        )
        for value, expected, name in exact_types:
            if type(value) is not expected:
                raise TypeError(f"{name} has the wrong exact type")
        if not (
            getattr(planner, "_projection", None) is projection
            and physical_view_state_issuer.projection is projection
            and getattr(frontier_viewpoint_planner, "_planner", None) is planner
            and getattr(frontier_viewpoint_planner, "_issuer", None)
            is physical_view_state_issuer
            and getattr(target_evidence_issuer, "_projection", None) is projection
            and getattr(target_evidence_issuer, "_planner", None) is planner
            and getattr(target_memory, "_issuer", None) is target_evidence_issuer
            and getattr(target_router, "_projection", None) is projection
            and getattr(target_router, "_planner", None) is planner
            and getattr(target_router, "_memory", None) is target_memory
            and getattr(world_waypoint_issuer, "_projection", None) is projection
            and getattr(world_waypoint_issuer, "_planner", None) is planner
        ):
            raise TwoResolutionNavigationIntegrationBindingError(
                "navigation authorities do not share the exact passed owner chain"
            )
        self._projection = projection
        self._planner = planner
        self._view_issuer = physical_view_state_issuer
        self._frontier = frontier_viewpoint_planner
        self._evidence_issuer = target_evidence_issuer
        self._target_memory = target_memory
        self._target_router = target_router
        self._waypoint_issuer = world_waypoint_issuer
        self._capability = object()
        self._sequence = 0
        self._issued: dict[int, _IssuedControllerTrace] = {}
        self._observed: set[int] = set()

    @property
    def production_eligible(self) -> bool:
        return False

    @property
    def hardware_execution_authorized(self) -> bool:
        return False

    def __copy__(self) -> "TwoResolutionDevelopmentNavigationIntegrationV1":
        raise TypeError("development navigation integrations are non-copyable")

    def __deepcopy__(
        self,
        memo: object,
    ) -> "TwoResolutionDevelopmentNavigationIntegrationV1":
        del memo
        raise TypeError("development navigation integrations are non-copyable")

    def __reduce_ex__(self, protocol: int) -> object:
        del protocol
        raise TypeError("development navigation integrations are non-serializable")

    def _assert_exact_controller_trace(
        self,
        artifact: TwoResolutionDevelopmentControllerClaimTraceV1,
    ) -> _IssuedControllerTrace:
        if type(artifact) is not TwoResolutionDevelopmentControllerClaimTraceV1:
            raise TypeError(
                "artifact must be TwoResolutionDevelopmentControllerClaimTraceV1"
            )
        row = self._issued.get(id(artifact))
        if (
            row is None
            or row.artifact is not artifact
            or artifact._issuance_capability is not self._capability
        ):
            raise TwoResolutionNavigationIntegrationBindingError(
                "controller claim trace is not the exact live object issued here"
            )
        artifact.assert_integrity()
        if artifact.content_sha256 != row.original_content_sha256:
            raise TwoResolutionNavigationIntegrationBindingError(
                "controller claim trace differs from its original issuance"
            )
        return row

    def assert_controller_claim_trace(
        self,
        artifact: TwoResolutionDevelopmentControllerClaimTraceV1,
    ) -> None:
        self._assert_exact_controller_trace(artifact)

    def issue_controller_claim_trace(
        self,
        *,
        snapshot: TwoResolutionConfigurationSnapshotV2,
        component: ConfigurationComponentV2,
        outcome: SyntheticV5TargetOutcomeV1,
        start_configuration_cell: Sequence[int],
        current_yaw_rad: float,
        physical_manifest: SceneManifest,
        trace_id: str,
        episode_id: str,
        event_id: str,
        tick: int,
        event_index: int,
        robot_pose_world_xy_yaw: Sequence[float],
        task_object_id: str,
        task_object_ids: Sequence[str] | None = None,
        pose_xy_variance_m2: float = 0.0,
        pose_yaw_variance_rad2: float = 0.0,
    ) -> TwoResolutionDevelopmentControllerClaimTraceV1:
        if type(snapshot) is not TwoResolutionConfigurationSnapshotV2:
            raise TypeError("snapshot has the wrong exact type")
        if type(component) is not ConfigurationComponentV2:
            raise TypeError("component has the wrong exact type")
        if type(outcome) is not SyntheticV5TargetOutcomeV1:
            raise TypeError("outcome has the wrong exact type")
        if type(physical_manifest) is not SceneManifest:
            raise TypeError("physical_manifest has the wrong exact type")
        task_object_id = _nonempty(task_object_id, "task_object_id")
        yaw = _finite(current_yaw_rad, "current_yaw_rad")
        xy_variance = _finite(pose_xy_variance_m2, "pose_xy_variance_m2")
        yaw_variance = _finite(
            pose_yaw_variance_rad2, "pose_yaw_variance_rad2"
        )
        if xy_variance < 0.0 or yaw_variance < 0.0:
            raise ValueError("pose variances must be non-negative")

        self._projection.assert_current_snapshot(snapshot)
        self._planner.validate_component(snapshot, component)
        state = self._view_issuer.issue(
            snapshot,
            pose_xy_variance_m2=xy_variance,
            pose_yaw_variance_rad2=yaw_variance,
        )
        candidates = self._frontier.generate(
            snapshot,
            state,
            start_configuration_cell=start_configuration_cell,
            current_yaw_rad=yaw,
        )
        selected = self._frontier.select(snapshot, state, candidates)
        if selected is None:
            raise TwoResolutionNavigationIntegrationError(
                "G4 produced no development frontier/viewpoint candidate"
            )
        self._frontier.validate_candidate(snapshot, state, candidates, selected)

        context = self._evidence_issuer.issue_context(
            snapshot,
            component,
            outcome,
        )
        writer = self._evidence_issuer.open_writer(context)
        evidence = (
            writer.issue_positive()
            if outcome.outcome_kind == "positive"
            else writer.issue_negative()
        )
        posterior = self._target_memory.apply(context, evidence)
        self._target_memory.assert_current_snapshot(posterior)

        route = self._target_router.issue(
            snapshot=snapshot,
            component=component,
            posterior=posterior,
            start_cell=start_configuration_cell,
            start_yaw_rad=yaw,
        )
        waypoint = self._waypoint_issuer.issue(snapshot, route.path)
        self._target_router.validate(
            snapshot=snapshot,
            component=component,
            posterior=posterior,
            plan=route,
        )
        self._waypoint_issuer.validate(snapshot, route.path, waypoint)
        if (
            route.production_promotion_authorized is not False
            or route.hardware_execution_authorized is not False
            or route.receipt.production_promotion_authorized is not False
            or route.receipt.hardware_execution_authorized is not False
            or waypoint.production_promotion_authorized is not False
            or waypoint.hardware_execution_authorized is not False
        ):
            raise TwoResolutionNavigationIntegrationBindingError(
                "a retained route or waypoint gained forbidden authority"
            )

        reference = object_id_reference(task_object_id)
        attempt = build_claim_attempt(
            manifest=physical_manifest,
            trace_id=trace_id,
            episode_id=episode_id,
            event_id=event_id,
            tick=tick,
            event_index=event_index,
            requested_target=reference,
            claimed_target=reference,
            robot_pose_world_xy_yaw=robot_pose_world_xy_yaw,
            pose_provenance="runtime_full_precision",
        )
        trace, task_ids, task_hash = build_claim_trace(
            manifest=physical_manifest,
            trace_id=trace_id,
            episode_id=episode_id,
            controller_claim_attempts=[attempt],
            task_object_ids=task_object_ids,
        )

        self._sequence += 1
        artifact = TwoResolutionDevelopmentControllerClaimTraceV1(
            controller_attempt_sequence=self._sequence,
            scene_id=physical_manifest.scene_id,
            physical_manifest_sha256=manifest_sha256(physical_manifest),
            snapshot_sha256=snapshot.content_sha256,
            component_sha256=component.content_sha256,
            physical_map_frame_sha256=snapshot.physical_map_frame_sha256,
            configuration_map_frame_sha256=(
                snapshot.configuration_map_frame_sha256
            ),
            physical_revision=snapshot.physical_revision,
            configuration_revision=snapshot.configuration_revision,
            physical_view_state_sha256=state.content_sha256,
            frontier_candidate_set_sha256=candidates.content_sha256,
            selected_frontier_candidate_sha256=selected.content_sha256,
            raw_outcome_content_sha256=outcome.raw_outcome_content_sha256,
            target_context_sha256=context.content_sha256,
            target_evidence_sha256=evidence.content_sha256,
            target_posterior_sha256=posterior.content_sha256,
            target_route_plan_sha256=route.content_sha256,
            target_route_receipt_sha256=route.receipt.content_sha256,
            world_waypoint_receipt_sha256=waypoint.content_sha256,
            target_id=outcome.target_id,
            task_object_id=task_object_id,
            target_route_plan=route.to_dict(),
            world_waypoint_receipt=waypoint.to_dict(),
            controller_claim_attempt=attempt,
            raw_claim_trace=trace,
            task_object_ids=task_ids,
            task_object_set_sha256=task_hash,
            production_promotion_authorized=False,
            hardware_execution_authorized=False,
            _issuance_capability=self._capability,
        )

        # Seal both downstream authorities before exposing the controller trace.
        self._waypoint_issuer.validate(
            snapshot,
            route.path,
            waypoint,
            consume=True,
        )
        self._target_router.validate(
            snapshot=snapshot,
            component=component,
            posterior=posterior,
            plan=route,
            consume=True,
        )
        self._issued[id(artifact)] = _IssuedControllerTrace(
            artifact=artifact,
            original_content_sha256=artifact.content_sha256,
            physical_manifest=physical_manifest,
        )
        return artifact

    def evaluate_observer_only(
        self,
        artifact: TwoResolutionDevelopmentControllerClaimTraceV1,
        *,
        evaluator_access_ledger: Mapping[str, int],
    ) -> TwoResolutionObserverClaimEvaluationV1:
        row = self._assert_exact_controller_trace(artifact)
        if id(artifact) in self._observed:
            raise TwoResolutionNavigationIntegrationReplayError(
                "controller claim trace was already evaluated"
            )

        # Loading evaluator code is permitted only after the controller trace is sealed.
        from lewm.benchmarks.go2_physical_claim_observer import (
            empty_evaluator_access_ledger,
            evaluate_runtime_claim_trace,
        )

        expected_ledger = empty_evaluator_access_ledger()
        supplied_ledger = _plain_mapping(
            evaluator_access_ledger, "evaluator_access_ledger"
        )
        if (
            supplied_ledger != expected_ledger
            or set(supplied_ledger) != set(expected_ledger)
            or any(type(value) is not int for value in supplied_ledger.values())
        ):
            raise TwoResolutionNavigationIntegrationBindingError(
                "evaluator access ledger is not actually empty"
            )
        if artifact.raw_claim_trace.get("evaluator_feedback_to_controller") != []:
            raise TwoResolutionNavigationIntegrationBindingError(
                "controller trace contains evaluator feedback"
            )
        evaluated = evaluate_runtime_claim_trace(
            artifact.raw_claim_trace,
            row.physical_manifest,
            artifact.task_object_ids,
            artifact.task_object_set_sha256,
        )
        result = TwoResolutionObserverClaimEvaluationV1(
            controller_claim_trace_sha256=artifact.content_sha256,
            evaluator_access_ledger=supplied_ledger,
            evaluated_claim_trace=evaluated,
            production_promotion_authorized=False,
            hardware_execution_authorized=False,
            _issuance_capability=self._capability,
        )
        self._observed.add(id(artifact))
        return result


def require_production_two_resolution_navigation_integration_v1() -> object:
    if PRODUCTION_TWO_RESOLUTION_NAVIGATION_INTEGRATION_V1 is None:
        raise PermissionError(
            "no production two-resolution navigation integration is configured"
        )
    return PRODUCTION_TWO_RESOLUTION_NAVIGATION_INTEGRATION_V1


__all__ = [
    "TwoResolutionDevelopmentControllerClaimTraceV1",
    "TwoResolutionDevelopmentNavigationIntegrationV1",
    "TwoResolutionNavigationIntegrationBindingError",
    "TwoResolutionNavigationIntegrationError",
    "TwoResolutionNavigationIntegrationReplayError",
    "TwoResolutionObserverClaimEvaluationV1",
    "require_production_two_resolution_navigation_integration_v1",
]
