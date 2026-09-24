"""Atomic successor for the frozen development-integration V2 block.

V3 preserves V1 and V2 unchanged and retains no predecessor integration engine.
It extends V2's downstream-owner transaction across controller-record creation,
registry insertion, and coordinator-seal assignment so every failed call is
exactly retry-safe.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
import math
from typing import Any, Mapping, Sequence
from weakref import WeakKeyDictionary

from lewm.benchmarks.go2_physical_claim_trace import (
    build_claim_attempt,
    build_claim_trace,
    canonical_task_object_ids,
    object_id_reference,
    task_object_set_sha256,
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
    SyntheticV5TargetOutcomeIssuerV1,
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


PRODUCTION_TWO_RESOLUTION_NAVIGATION_INTEGRATION_V3 = None


class TwoResolutionNavigationIntegrationV3Error(ValueError):
    """Base error for the additive V3 coordinator."""


class TwoResolutionNavigationIntegrationV3BindingError(
    TwoResolutionNavigationIntegrationV3Error
):
    """An episode, controller, observer, owner, or state binding changed."""


class TwoResolutionNavigationIntegrationV3ReplayError(
    TwoResolutionNavigationIntegrationV3Error
):
    """An exact single-use V3 authority was replayed."""


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


def _nonempty(value: object, name: str) -> str:
    if type(value) is not str or not value:
        raise ValueError(f"{name} must be a nonempty string")
    return value


def _require_sha256(value: object, name: str) -> str:
    if not (
        type(value) is str
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256")
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


def _configuration_cell(value: object, name: str) -> tuple[int, int]:
    if (
        not isinstance(value, Sequence)
        or isinstance(value, (str, bytes))
        or len(value) != 2
        or any(isinstance(item, bool) or not isinstance(item, int) for item in value)
    ):
        raise ValueError(f"{name} must contain exactly two integer indices")
    return (int(value[0]), int(value[1]))


def _empty_ledger() -> dict[str, int]:
    return {
        "evaluator_output_reads_by_controller": 0,
        "evaluator_callbacks_into_controller": 0,
        "evaluator_derived_termination_signals": 0,
    }


def _validate_empty_ledger(value: object) -> dict[str, int]:
    ledger = _plain_mapping(value, "evaluator_access_ledger")
    if ledger != _empty_ledger() or any(
        type(item) is not int for item in ledger.values()
    ):
        raise TwoResolutionNavigationIntegrationV3BindingError(
            "evaluator access ledger is not actually empty"
        )
    return ledger  # type: ignore[return-value]


def _object_rows(values: Mapping[int, object]) -> list[dict[str, object]]:
    rows = []
    for identity, value in sorted(values.items()):
        content = getattr(value, "content_sha256", None)
        rows.append(
            {
                "object_identity": identity,
                "exact_object_identity": id(value),
                "content_sha256": content,
            }
        )
    return rows


def _consumed_hashes(
    consumed: set[int],
    issued: Mapping[int, object],
) -> list[dict[str, object]]:
    return [
        {
            "object_identity": identity,
            "content_sha256": getattr(issued.get(identity), "content_sha256", None),
        }
        for identity in sorted(consumed)
    ]


@dataclass(frozen=True, order=True)
class TwoResolutionNavigationTargetObjectBindingV3:
    semantic_target_id: str
    task_object_id: str

    def __post_init__(self) -> None:
        _nonempty(self.semantic_target_id, "semantic_target_id")
        _nonempty(self.task_object_id, "task_object_id")

    def to_dict(self) -> dict[str, str]:
        return {
            "semantic_target_id": self.semantic_target_id,
            "task_object_id": self.task_object_id,
        }


@dataclass(frozen=True)
class TwoResolutionNavigationEpisodeAuthorityV3:
    """Exact episode-wide scene, task bijection, and G3 snapshot authority."""

    authority_sequence: int
    episode_id: str
    scene_id: str
    physical_manifest_sha256: str
    task_object_ids: tuple[str, ...]
    task_object_set_sha256: str
    target_object_bindings: tuple[TwoResolutionNavigationTargetObjectBindingV3, ...]
    snapshot_sha256: str
    component_sha256: str
    physical_session_id: str
    configuration_session_id: str
    physical_frame_id: str
    configuration_frame_id: str
    physical_map_frame_sha256: str
    configuration_map_frame_sha256: str
    physical_revision: int
    configuration_revision: int
    projection_source_sha256: str
    _issuer: object = field(repr=False, compare=False)
    content_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        if _nonnegative_int(self.authority_sequence, "authority_sequence") == 0:
            raise ValueError("authority_sequence must be positive")
        for name in (
            "episode_id",
            "scene_id",
            "physical_session_id",
            "configuration_session_id",
            "physical_frame_id",
            "configuration_frame_id",
        ):
            _nonempty(getattr(self, name), name)
        for name in (
            "physical_manifest_sha256",
            "task_object_set_sha256",
            "snapshot_sha256",
            "component_sha256",
            "physical_map_frame_sha256",
            "configuration_map_frame_sha256",
            "projection_source_sha256",
        ):
            _require_sha256(getattr(self, name), name)
        _nonnegative_int(self.physical_revision, "physical_revision")
        _nonnegative_int(self.configuration_revision, "configuration_revision")
        task_ids = tuple(self.task_object_ids)
        bindings = tuple(self.target_object_bindings)
        if (
            not task_ids
            or tuple(sorted(task_ids, key=lambda value: value.encode("utf-8")))
            != task_ids
            or len(set(task_ids)) != len(task_ids)
            or any(type(value) is not str or not value for value in task_ids)
        ):
            raise ValueError("episode task object IDs are not canonical")
        if (
            not bindings
            or any(
                type(binding) is not TwoResolutionNavigationTargetObjectBindingV3
                for binding in bindings
            )
            or tuple(sorted(bindings)) != bindings
            or len({row.semantic_target_id for row in bindings}) != len(bindings)
            or len({row.task_object_id for row in bindings}) != len(bindings)
            or {row.task_object_id for row in bindings} != set(task_ids)
        ):
            raise ValueError(
                "target-to-object mapping must be complete, one-to-one, and canonical"
            )
        if (
            self.physical_session_id == self.configuration_session_id
            or self.physical_frame_id == self.configuration_frame_id
            or self.physical_map_frame_sha256
            == self.configuration_map_frame_sha256
        ):
            raise ValueError("episode G3 physical/configuration identities collapsed")
        if self._issuer is None:
            raise TypeError("episode authority requires its exact live issuer")
        object.__setattr__(self, "task_object_ids", task_ids)
        object.__setattr__(self, "target_object_bindings", bindings)
        object.__setattr__(self, "content_sha256", _sha256(self.to_dict(False)))

    @property
    def production_promotion_authorized(self) -> bool:
        return False

    @property
    def hardware_execution_authorized(self) -> bool:
        return False

    def target_object_map(self) -> dict[str, str]:
        return {
            row.semantic_target_id: row.task_object_id
            for row in self.target_object_bindings
        }

    def to_dict(self, include_hash: bool = True) -> dict[str, object]:
        result: dict[str, object] = {
            "schema": "lewm_two_resolution_navigation_episode_authority_v3",
            "authority_sequence": self.authority_sequence,
            "episode_id": self.episode_id,
            "scene_id": self.scene_id,
            "physical_manifest_sha256": self.physical_manifest_sha256,
            "task_object_ids": list(self.task_object_ids),
            "task_object_set_sha256": self.task_object_set_sha256,
            "target_object_bindings": [
                row.to_dict() for row in self.target_object_bindings
            ],
            "g3": {
                "snapshot_sha256": self.snapshot_sha256,
                "component_sha256": self.component_sha256,
                "physical_session_id": self.physical_session_id,
                "configuration_session_id": self.configuration_session_id,
                "physical_frame_id": self.physical_frame_id,
                "configuration_frame_id": self.configuration_frame_id,
                "physical_map_frame_sha256": self.physical_map_frame_sha256,
                "configuration_map_frame_sha256": (
                    self.configuration_map_frame_sha256
                ),
                "physical_revision": self.physical_revision,
                "configuration_revision": self.configuration_revision,
                "projection_source_sha256": self.projection_source_sha256,
            },
            "development_only": True,
            "production_promotion_authorized": False,
            "hardware_execution_authorized": False,
        }
        if include_hash:
            result["content_sha256"] = self.content_sha256
        return result

    def _assert_structural_integrity(self) -> None:
        if self.content_sha256 != _sha256(self.to_dict(False)):
            raise TwoResolutionNavigationIntegrationV3BindingError(
                "episode authority was mutated"
            )

    def assert_integrity(self) -> None:
        if type(self._issuer) is not TwoResolutionNavigationEpisodeAuthorityIssuerV3:
            raise TwoResolutionNavigationIntegrationV3BindingError(
                "episode authority lost its exact issuer"
            )
        self._issuer.assert_episode_authority(self)

    def __copy__(self) -> "TwoResolutionNavigationEpisodeAuthorityV3":
        raise TypeError("navigation episode authorities are non-copyable")

    def __deepcopy__(
        self,
        memo: object,
    ) -> "TwoResolutionNavigationEpisodeAuthorityV3":
        del memo
        raise TypeError("navigation episode authorities are non-copyable")

    def __reduce_ex__(self, protocol: int) -> object:
        del protocol
        raise TypeError("navigation episode authorities are non-serializable")


@dataclass(frozen=True)
class _EpisodeAuthorityRecord:
    authority: TwoResolutionNavigationEpisodeAuthorityV3
    original_content_sha256: str
    snapshot: TwoResolutionConfigurationSnapshotV2
    component: ConfigurationComponentV2
    physical_manifest: SceneManifest
    original_manifest_sha256: str


@dataclass
class _EpisodeIssuerState:
    projection: TwoResolutionConfigurationProjectionV2
    planner: TwoResolutionConfigurationPlannerV2
    manifest: SceneManifest
    manifest_sha256: str
    episode_id: str
    task_ids: tuple[str, ...]
    task_hash: str
    bindings: tuple[TwoResolutionNavigationTargetObjectBindingV3, ...]
    record: _EpisodeAuthorityRecord | None = None


def _episode_state_accessors():
    states: WeakKeyDictionary[object, _EpisodeIssuerState] = WeakKeyDictionary()

    def install(owner: object, state: _EpisodeIssuerState) -> None:
        if owner in states:
            raise RuntimeError("episode issuer state already installed")
        states[owner] = state

    def resolve(owner: object) -> _EpisodeIssuerState:
        try:
            return states[owner]
        except KeyError as exc:
            raise TwoResolutionNavigationIntegrationV3BindingError(
                "episode issuer has no exact live state"
            ) from exc

    return install, resolve


_install_episode_state, _episode_state = _episode_state_accessors()


class TwoResolutionNavigationEpisodeAuthorityIssuerV3:
    """Own one exact manifest, task bijection, and G3 episode snapshot."""

    __slots__ = ("__weakref__",)

    def __init__(
        self,
        *,
        projection: TwoResolutionConfigurationProjectionV2,
        planner: TwoResolutionConfigurationPlannerV2,
        physical_manifest: SceneManifest,
        episode_id: str,
        target_object_mapping: Mapping[str, str],
        _synthetic_development_fixture: bool = False,
    ) -> None:
        if _synthetic_development_fixture is not True:
            raise PermissionError(
                "no production navigation episode authority is configured"
            )
        if type(projection) is not TwoResolutionConfigurationProjectionV2:
            raise TypeError("projection has the wrong exact type")
        if type(planner) is not TwoResolutionConfigurationPlannerV2:
            raise TypeError("planner has the wrong exact type")
        if getattr(planner, "_projection", None) is not projection:
            raise TwoResolutionNavigationIntegrationV3BindingError(
                "episode planner and projection instances differ"
            )
        if type(physical_manifest) is not SceneManifest:
            raise TypeError("physical_manifest has the wrong exact type")
        task_ids = canonical_task_object_ids(physical_manifest)
        if not isinstance(target_object_mapping, Mapping):
            raise TypeError("target_object_mapping must be a mapping")
        bindings = tuple(
            sorted(
                TwoResolutionNavigationTargetObjectBindingV3(semantic, task_object)
                for semantic, task_object in target_object_mapping.items()
            )
        )
        if (
            len(bindings) != len(task_ids)
            or len({row.semantic_target_id for row in bindings}) != len(bindings)
            or len({row.task_object_id for row in bindings}) != len(bindings)
            or {row.task_object_id for row in bindings} != set(task_ids)
        ):
            raise ValueError(
                "target-to-object mapping must be a complete one-to-one task mapping"
            )
        manifest_hash = manifest_sha256(physical_manifest)
        _install_episode_state(
            self,
            _EpisodeIssuerState(
                projection=projection,
                planner=planner,
                manifest=physical_manifest,
                manifest_sha256=manifest_hash,
                episode_id=_nonempty(episode_id, "episode_id"),
                task_ids=task_ids,
                task_hash=task_object_set_sha256(physical_manifest, task_ids),
                bindings=bindings,
            ),
        )

    @property
    def projection(self) -> TwoResolutionConfigurationProjectionV2:
        return _episode_state(self).projection

    @property
    def planner(self) -> TwoResolutionConfigurationPlannerV2:
        return _episode_state(self).planner

    def __copy__(self) -> "TwoResolutionNavigationEpisodeAuthorityIssuerV3":
        raise TypeError("navigation episode authority issuers are non-copyable")

    def __deepcopy__(
        self,
        memo: object,
    ) -> "TwoResolutionNavigationEpisodeAuthorityIssuerV3":
        del memo
        raise TypeError("navigation episode authority issuers are non-copyable")

    def __reduce_ex__(self, protocol: int) -> object:
        del protocol
        raise TypeError("navigation episode authority issuers are non-serializable")

    def issue(
        self,
        snapshot: TwoResolutionConfigurationSnapshotV2,
        component: ConfigurationComponentV2,
    ) -> TwoResolutionNavigationEpisodeAuthorityV3:
        state = _episode_state(self)
        if state.record is not None:
            raise TwoResolutionNavigationIntegrationV3ReplayError(
                "this issuer already issued its one episode authority"
            )
        state.projection.assert_current_snapshot(snapshot)
        state.planner.validate_component(snapshot, component)
        if manifest_sha256(state.manifest) != state.manifest_sha256:
            raise TwoResolutionNavigationIntegrationV3BindingError(
                "episode physical manifest changed before authority issuance"
            )
        expected_physical_session_id = f"{state.manifest.scene_id}:g3-v2:physical"
        expected_configuration_session_id = (
            f"{state.manifest.scene_id}:g3-v2:configuration"
        )
        if (
            snapshot.physical_map_frame.session_id
            != expected_physical_session_id
            or snapshot.configuration_map_frame.session_id
            != expected_configuration_session_id
        ):
            raise TwoResolutionNavigationIntegrationV3BindingError(
                "G3 physical/configuration sessions do not name the exact "
                "episode manifest scene"
            )
        authority = TwoResolutionNavigationEpisodeAuthorityV3(
            authority_sequence=1,
            episode_id=state.episode_id,
            scene_id=state.manifest.scene_id,
            physical_manifest_sha256=state.manifest_sha256,
            task_object_ids=state.task_ids,
            task_object_set_sha256=state.task_hash,
            target_object_bindings=state.bindings,
            snapshot_sha256=snapshot.content_sha256,
            component_sha256=component.content_sha256,
            physical_session_id=snapshot.physical_map_frame.session_id,
            configuration_session_id=(
                snapshot.configuration_map_frame.session_id
            ),
            physical_frame_id=snapshot.physical_map_frame.frame_id,
            configuration_frame_id=snapshot.configuration_map_frame.frame_id,
            physical_map_frame_sha256=snapshot.physical_map_frame_sha256,
            configuration_map_frame_sha256=(
                snapshot.configuration_map_frame_sha256
            ),
            physical_revision=snapshot.physical_revision,
            configuration_revision=snapshot.configuration_revision,
            projection_source_sha256=snapshot.projection_source_sha256,
            _issuer=self,
        )
        state.record = _EpisodeAuthorityRecord(
            authority=authority,
            original_content_sha256=authority.content_sha256,
            snapshot=snapshot,
            component=component,
            physical_manifest=state.manifest,
            original_manifest_sha256=state.manifest_sha256,
        )
        return authority

    def _record(
        self,
        authority: TwoResolutionNavigationEpisodeAuthorityV3,
    ) -> _EpisodeAuthorityRecord:
        if type(authority) is not TwoResolutionNavigationEpisodeAuthorityV3:
            raise TypeError("authority has the wrong exact type")
        state = _episode_state(self)
        row = state.record
        if (
            row is None
            or row.authority is not authority
            or authority._issuer is not self
        ):
            raise TwoResolutionNavigationIntegrationV3BindingError(
                "episode authority is not the exact live object issued here"
            )
        authority._assert_structural_integrity()
        if authority.content_sha256 != row.original_content_sha256:
            raise TwoResolutionNavigationIntegrationV3BindingError(
                "episode authority differs from its original issuance"
            )
        if (
            row.physical_manifest is not state.manifest
            or manifest_sha256(row.physical_manifest)
            != row.original_manifest_sha256
            or row.original_manifest_sha256 != state.manifest_sha256
            or row.physical_manifest.scene_id != authority.scene_id
            or canonical_task_object_ids(row.physical_manifest)
            != authority.task_object_ids
            or task_object_set_sha256(
                row.physical_manifest,
                authority.task_object_ids,
            )
            != authority.task_object_set_sha256
        ):
            raise TwoResolutionNavigationIntegrationV3BindingError(
                "episode physical manifest differs from its original issuance"
            )
        return row

    def assert_episode_authority(
        self,
        authority: TwoResolutionNavigationEpisodeAuthorityV3,
    ) -> None:
        self._record(authority)

    def resolve_for_controller(
        self,
        *,
        authority: TwoResolutionNavigationEpisodeAuthorityV3,
        snapshot: TwoResolutionConfigurationSnapshotV2,
        component: ConfigurationComponentV2,
        physical_manifest: SceneManifest,
    ) -> _EpisodeAuthorityRecord:
        row = self._record(authority)
        state = _episode_state(self)
        state.projection.assert_current_snapshot(snapshot)
        state.planner.validate_component(snapshot, component)
        if (
            row.snapshot is not snapshot
            or row.component is not component
            or row.physical_manifest is not physical_manifest
            or manifest_sha256(physical_manifest) != row.original_manifest_sha256
            or physical_manifest.scene_id != authority.scene_id
            or snapshot.content_sha256 != authority.snapshot_sha256
            or component.content_sha256 != authority.component_sha256
            or snapshot.physical_map_frame.session_id
            != authority.physical_session_id
            or snapshot.configuration_map_frame.session_id
            != authority.configuration_session_id
            or snapshot.physical_map_frame.frame_id != authority.physical_frame_id
            or snapshot.configuration_map_frame.frame_id
            != authority.configuration_frame_id
            or snapshot.physical_map_frame_sha256
            != authority.physical_map_frame_sha256
            or snapshot.configuration_map_frame_sha256
            != authority.configuration_map_frame_sha256
            or snapshot.projection_source_sha256
            != authority.projection_source_sha256
        ):
            raise TwoResolutionNavigationIntegrationV3BindingError(
                "G3 scene/session/snapshot/frame differs from episode manifest "
                "authority"
            )
        return row


@dataclass(frozen=True)
class TwoResolutionDevelopmentControllerClaimTraceV3:
    episode_authority_sha256: str
    semantic_target_id: str
    task_object_id: str
    episode_authority: Mapping[str, object]
    controller_payload: Mapping[str, object]
    _integration: object = field(repr=False, compare=False)
    content_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        _require_sha256(self.episode_authority_sha256, "episode_authority_sha256")
        _nonempty(self.semantic_target_id, "semantic_target_id")
        _nonempty(self.task_object_id, "task_object_id")
        authority = _plain_mapping(self.episode_authority, "episode_authority")
        payload = _plain_mapping(self.controller_payload, "controller_payload")
        authority_g3 = authority.get("g3")
        payload_g3 = payload.get("g3")
        payload_g5 = payload.get("g5")
        route = payload.get("target_route_plan")
        waypoint = payload.get("world_waypoint_receipt")
        trace = payload.get("raw_claim_trace")
        if not all(
            type(value) is dict
            for value in (authority_g3, payload_g3, payload_g5, route, waypoint, trace)
        ):
            raise TwoResolutionNavigationIntegrationV3BindingError(
                "V3 controller payload is structurally incomplete"
            )
        retained_route = route.get("retained_v1_receipt")
        mapping = {
            "semantic_target_id": self.semantic_target_id,
            "task_object_id": self.task_object_id,
        }
        if (
            type(retained_route) is not dict
            or authority.get("content_sha256") != self.episode_authority_sha256
            or mapping not in authority.get("target_object_bindings", [])
            or payload.get("scene_id") != authority.get("scene_id")
            or payload.get("physical_manifest_sha256")
            != authority.get("physical_manifest_sha256")
            or payload_g3.get("snapshot_sha256")
            != authority_g3.get("snapshot_sha256")
            or payload_g3.get("component_sha256")
            != authority_g3.get("component_sha256")
            or payload_g5.get("target_id") != self.semantic_target_id
            or payload.get("task_object_id") != self.task_object_id
            or trace.get("evaluator_feedback_to_controller") != []
            or route.get("production_promotion_authorized") is not False
            or route.get("hardware_execution_authorized") is not False
            or retained_route.get("production_promotion_authorized") is not False
            or retained_route.get("hardware_execution_authorized") is not False
            or waypoint.get("production_promotion_authorized") is not False
            or waypoint.get("hardware_execution_authorized") is not False
            or payload.get("production_promotion_authorized") is not False
            or payload.get("hardware_execution_authorized") is not False
            or authority.get("production_promotion_authorized") is not False
            or authority.get("hardware_execution_authorized") is not False
        ):
            raise TwoResolutionNavigationIntegrationV3BindingError(
                "V3 controller payload differs from exact episode authority"
            )
        if self._integration is None:
            raise TypeError("V3 controller trace requires its exact integration")
        object.__setattr__(self, "episode_authority", authority)
        object.__setattr__(self, "controller_payload", payload)
        object.__setattr__(self, "content_sha256", _sha256(self.to_dict(False)))

    @property
    def production_promotion_authorized(self) -> bool:
        return False

    @property
    def hardware_execution_authorized(self) -> bool:
        return False

    @property
    def raw_claim_trace(self) -> Mapping[str, object]:
        value = self.controller_payload.get("raw_claim_trace")
        if type(value) is not dict:
            raise TwoResolutionNavigationIntegrationV3BindingError(
                "V3 controller payload lost its raw claim trace"
            )
        return value

    def to_dict(self, include_hash: bool = True) -> dict[str, object]:
        result: dict[str, object] = {
            "schema": "lewm_two_resolution_development_controller_claim_trace_v3",
            "episode_authority_sha256": self.episode_authority_sha256,
            "semantic_target_id": self.semantic_target_id,
            "task_object_id": self.task_object_id,
            "episode_authority": _plain_json(self.episode_authority),
            "controller_payload": _plain_json(self.controller_payload),
            "development_execution_eligible": True,
            "production_promotion_authorized": False,
            "hardware_execution_authorized": False,
        }
        if include_hash:
            result["content_sha256"] = self.content_sha256
        return result

    def _assert_structural_integrity(self) -> None:
        if self.content_sha256 != _sha256(self.to_dict(False)):
            raise TwoResolutionNavigationIntegrationV3BindingError(
                "V3 controller claim trace was mutated"
            )

    def assert_integrity(self) -> None:
        if type(self._integration) is not (
            TwoResolutionDevelopmentNavigationIntegrationV3
        ):
            raise TwoResolutionNavigationIntegrationV3BindingError(
                "V3 controller trace lost its exact integration"
            )
        self._integration.assert_controller_claim_trace(self)

    def __copy__(self) -> "TwoResolutionDevelopmentControllerClaimTraceV3":
        raise TypeError("V3 controller claim traces are non-copyable")

    def __deepcopy__(
        self,
        memo: object,
    ) -> "TwoResolutionDevelopmentControllerClaimTraceV3":
        del memo
        raise TypeError("V3 controller claim traces are non-copyable")

    def __reduce_ex__(self, protocol: int) -> object:
        del protocol
        raise TypeError("V3 controller claim traces are non-serializable")


@dataclass(frozen=True)
class TwoResolutionObserverClaimEvaluationV3:
    controller_claim_trace_sha256: str
    episode_authority_sha256: str
    evaluator_access_ledger: Mapping[str, int]
    evaluated_claim_trace: Mapping[str, object]
    _integration: object = field(repr=False, compare=False)
    content_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        _require_sha256(
            self.controller_claim_trace_sha256,
            "controller_claim_trace_sha256",
        )
        _require_sha256(self.episode_authority_sha256, "episode_authority_sha256")
        ledger = _validate_empty_ledger(self.evaluator_access_ledger)
        evaluated = _plain_mapping(self.evaluated_claim_trace, "evaluated_claim_trace")
        if self._integration is None:
            raise TypeError("V3 observer result requires its exact integration")
        object.__setattr__(self, "evaluator_access_ledger", ledger)
        object.__setattr__(self, "evaluated_claim_trace", evaluated)
        object.__setattr__(self, "content_sha256", _sha256(self.to_dict(False)))

    @property
    def production_promotion_authorized(self) -> bool:
        return False

    @property
    def hardware_execution_authorized(self) -> bool:
        return False

    def to_dict(self, include_hash: bool = True) -> dict[str, object]:
        result: dict[str, object] = {
            "schema": "lewm_two_resolution_observer_claim_evaluation_v3",
            "controller_claim_trace_sha256": self.controller_claim_trace_sha256,
            "episode_authority_sha256": self.episode_authority_sha256,
            "observer_only": True,
            "evaluator_access_ledger": _plain_json(self.evaluator_access_ledger),
            "evaluated_claim_trace": _plain_json(self.evaluated_claim_trace),
            "controller_callback": None,
            "production_promotion_authorized": False,
            "hardware_execution_authorized": False,
        }
        if include_hash:
            result["content_sha256"] = self.content_sha256
        return result

    def _assert_structural_integrity(self) -> None:
        if self.content_sha256 != _sha256(self.to_dict(False)):
            raise TwoResolutionNavigationIntegrationV3BindingError(
                "V3 observer evaluation was mutated"
            )

    def assert_integrity(self) -> None:
        if type(self._integration) is not (
            TwoResolutionDevelopmentNavigationIntegrationV3
        ):
            raise TwoResolutionNavigationIntegrationV3BindingError(
                "V3 observer evaluation lost its exact integration"
            )
        self._integration.assert_observer_evaluation(self)

    def __copy__(self) -> "TwoResolutionObserverClaimEvaluationV3":
        raise TypeError("V3 observer claim evaluations are non-copyable")

    def __deepcopy__(
        self,
        memo: object,
    ) -> "TwoResolutionObserverClaimEvaluationV3":
        del memo
        raise TypeError("V3 observer claim evaluations are non-copyable")

    def __reduce_ex__(self, protocol: int) -> object:
        del protocol
        raise TypeError("V3 observer claim evaluations are non-serializable")


@dataclass(frozen=True)
class _ControllerRecordV3:
    artifact: TwoResolutionDevelopmentControllerClaimTraceV3
    original_content_sha256: str
    episode_authority: TwoResolutionNavigationEpisodeAuthorityV3


@dataclass(frozen=True)
class _ObserverRecordV3:
    result: TwoResolutionObserverClaimEvaluationV3
    original_content_sha256: str


@dataclass
class _IntegrationState:
    projection: TwoResolutionConfigurationProjectionV2
    planner: TwoResolutionConfigurationPlannerV2
    view_issuer: TwoResolutionPhysicalViewStateIssuerV2
    frontier: TwoResolutionFrontierViewpointPlannerV2
    evidence_issuer: TwoResolutionTargetEvidenceIssuerV1
    target_memory: TwoResolutionReversibleTargetBeliefMemoryV1
    target_router: TwoResolutionDeterministicTargetRouterV2
    waypoint_issuer: ConfigurationPathWorldWaypointIssuerV2
    episode_issuer: TwoResolutionNavigationEpisodeAuthorityIssuerV3
    owner_state_sha256: str = ""
    issued: dict[int, _ControllerRecordV3] = field(default_factory=dict)
    observed_controller_ids: set[int] = field(default_factory=set)
    observer_results: dict[int, _ObserverRecordV3] = field(default_factory=dict)
    consumed_observer_result_ids: set[int] = field(default_factory=set)
    known_outcomes: dict[int, tuple[SyntheticV5TargetOutcomeV1, str, int]] = field(
        default_factory=dict
    )
    known_outcome_sequence: int = 0
    synthetic_fault_after_stage: str | None = None


def _integration_state_accessors():
    states: WeakKeyDictionary[object, _IntegrationState] = WeakKeyDictionary()

    def install(owner: object, state: _IntegrationState) -> None:
        if owner in states:
            raise RuntimeError("integration state already installed")
        states[owner] = state

    def resolve(owner: object) -> _IntegrationState:
        try:
            return states[owner]
        except KeyError as exc:
            raise TwoResolutionNavigationIntegrationV3BindingError(
                "V3 integration has no exact live state"
            ) from exc

    return install, resolve


_install_integration_state, _integration_state = _integration_state_accessors()


_SYNTHETIC_TRANSACTION_FAULT_STAGES = frozenset(
    {
        "g4_view_issue",
        "g4_frontier_generate",
        "g4_frontier_select",
        "g4_frontier_validate",
        "g5_context_issue",
        "g5_writer_open",
        "g5_evidence_issue",
        "g5_posterior_apply",
        "g5_posterior_validate",
        "router_issue",
        "waypoint_issue",
        "router_validate",
        "waypoint_validate",
        "artifact_construct",
        "controller_payload_build",
        "waypoint_consume",
        "router_consume",
        "owner_state_seal",
        "controller_record_construct",
        "controller_registry_insert",
        "coordinator_seal_assign",
    }
)


def _synthetic_transaction_fault(state: _IntegrationState, stage: str) -> None:
    if state.synthetic_fault_after_stage == stage:
        state.synthetic_fault_after_stage = None
        raise RuntimeError(f"synthetic V3 transaction fault after {stage}")


def _assert_append_only_outcome_issuance(state: _IntegrationState) -> None:
    """Accept new source-owned outcomes, but reject rewriting prior ingress."""

    source = getattr(state.evidence_issuer, "_outcome_source")
    if type(source) is not SyntheticV5TargetOutcomeIssuerV1:
        raise TwoResolutionNavigationIntegrationV3BindingError(
            "G5 evidence issuer lost its exact outcome source"
        )
    issued = getattr(source, "_issued")
    sequence = getattr(source, "_sequence")
    if not isinstance(issued, dict) or isinstance(sequence, bool) or not isinstance(
        sequence, int
    ):
        raise TwoResolutionNavigationIntegrationV3BindingError(
            "G5 outcome source registry changed type"
        )
    if sequence < state.known_outcome_sequence:
        raise TwoResolutionNavigationIntegrationV3BindingError(
            "G5 outcome source sequence moved backwards"
        )
    for identity, (original, original_hash, original_sequence) in (
        state.known_outcomes.items()
    ):
        current = issued.get(identity)
        if (
            current is not original
            or current.raw_outcome_content_sha256 != original_hash
            or current.outcome_sequence != original_sequence
        ):
            raise TwoResolutionNavigationIntegrationV3BindingError(
                "G5 outcome source rewrote an existing exact issuance"
            )
        current.assert_integrity()
    rows: dict[int, tuple[SyntheticV5TargetOutcomeV1, str, int]] = {}
    sequences: list[int] = []
    capability = getattr(source, "_capability")
    for identity, outcome in issued.items():
        if (
            type(identity) is not int
            or type(outcome) is not SyntheticV5TargetOutcomeV1
            or id(outcome) != identity
            or getattr(outcome, "_issuance_capability", None) is not capability
        ):
            raise TwoResolutionNavigationIntegrationV3BindingError(
                "G5 outcome source contains a foreign exact object"
            )
        outcome.assert_integrity()
        sequences.append(outcome.outcome_sequence)
        rows[identity] = (
            outcome,
            outcome.raw_outcome_content_sha256,
            outcome.outcome_sequence,
        )
    if sorted(sequences) != list(range(1, sequence + 1)):
        raise TwoResolutionNavigationIntegrationV3BindingError(
            "G5 outcome source issuance sequence is not complete and append-only"
        )
    state.known_outcomes = rows
    state.known_outcome_sequence = sequence


def _owner_state_sha256(state: _IntegrationState) -> str:
    evidence = state.evidence_issuer
    outcome_source = getattr(evidence, "_outcome_source")
    memory = state.target_memory
    router = state.target_router
    waypoint = state.waypoint_issuer
    view = state.view_issuer
    frontier = state.frontier
    planner = state.planner
    evidence_objects = getattr(evidence, "_evidence")
    outcome_objects = getattr(outcome_source, "_issued")
    router_plans = getattr(router, "_plans")
    waypoint_objects = getattr(waypoint, "_issued")
    return _sha256(
        {
            "schema": "lewm_two_resolution_navigation_owner_state_seal_v3",
            "g3_planner": {
                "issued_components": _object_rows(
                    getattr(planner, "_issued_components")
                ),
                "issued_frontiers": _object_rows(
                    getattr(planner, "_issued_frontiers")
                ),
                "issued_paths": _object_rows(getattr(planner, "_issued_paths")),
            },
            "g4": {
                "view_content_sha256": view.view_content_sha256,
                "issued_states": _object_rows(getattr(view, "_issued_states")),
                "issued_candidate_sets": [
                    {
                        "object_identity": identity,
                        "content_sha256": row.candidate_set.content_sha256,
                    }
                    for identity, row in sorted(
                        getattr(frontier, "_issued_sets").items()
                    )
                ],
                "issued_candidates": _object_rows(
                    getattr(frontier, "_issued_candidates")
                ),
                "score_cache": [
                    {
                        "candidate_set_identity": identity,
                        "score_sha256s": [row.content_sha256 for row in rows],
                    }
                    for identity, rows in sorted(
                        getattr(frontier, "_score_cache").items()
                    )
                ],
            },
            "outcome_source": {
                # New source-owned outcomes are legitimate runtime ingress.  Their
                # append-only registry is checked separately; consumption is part
                # of the downstream transaction seal.
                "exact_source_identity": id(outcome_source),
                "consumed": [
                    {
                        "object_identity": identity,
                        "content_sha256": getattr(
                            outcome_objects.get(identity),
                            "raw_outcome_content_sha256",
                            None,
                        ),
                    }
                    for identity in sorted(getattr(outcome_source, "_consumed"))
                ],
            },
            "evidence_issuer": {
                "sequence": getattr(evidence, "_sequence"),
                "contexts": [
                    {
                        "object_identity": identity,
                        "content_sha256": row[0].content_sha256,
                        "writer_identity": (
                            None if row[1].writer is None else id(row[1].writer)
                        ),
                        "writer_used": (
                            None
                            if row[1].writer is None
                            else bool(row[1].writer._used)
                        ),
                        "evidence_identity": (
                            None if row[1].evidence is None else id(row[1].evidence)
                        ),
                        "evidence_sha256": getattr(
                            row[1].evidence,
                            "content_sha256",
                            None,
                        ),
                    }
                    for identity, row in sorted(
                        getattr(evidence, "_contexts").items()
                    )
                ],
                "evidence": _object_rows(evidence_objects),
                "consumed": _consumed_hashes(
                    getattr(evidence, "_consumed_evidence"),
                    evidence_objects,
                ),
            },
            "target_memory": {
                "revision": getattr(memory, "_revision"),
                "last_context_sequence": getattr(
                    memory, "_last_context_sequence"
                ),
                "last_pose_timestamp_ns": getattr(
                    memory, "_last_pose_timestamp_ns"
                ),
                "immutable_binding": (
                    None
                    if getattr(memory, "_immutable_binding") is None
                    else [
                        [
                            key,
                            (
                                value.content_sha256
                                if hasattr(value, "content_sha256")
                                else list(value)
                                if isinstance(value, tuple)
                                else value
                            ),
                        ]
                        for key, value in sorted(
                            getattr(memory, "_immutable_binding").items()
                        )
                    ]
                ),
                "mass": [
                    [
                        target,
                        [
                            [list(cell), value]
                            for cell, value in sorted(rows.items())
                        ],
                    ]
                    for target, rows in sorted(getattr(memory, "_mass").items())
                ],
                "unlocalized": sorted(getattr(memory, "_unlocalized").items()),
                "positive_count": sorted(
                    getattr(memory, "_positive_count").items()
                ),
                "negative_count": sorted(
                    getattr(memory, "_negative_count").items()
                ),
                "contexts": [
                    [target, context.content_sha256]
                    for target, context in sorted(
                        getattr(memory, "_contexts").items()
                    )
                ],
                "chains": sorted(getattr(memory, "_chains").items()),
                "seen_context_ids": sorted(getattr(memory, "_seen_context_ids")),
                "seen_evidence_hashes": sorted(
                    getattr(memory, "_seen_evidence_hashes")
                ),
                "seen_raw_outcomes": sorted(
                    getattr(memory, "_seen_raw_outcomes")
                ),
                "issued_snapshots": _object_rows(
                    getattr(memory, "_issued_snapshots")
                ),
            },
            "router": {
                "plans": _object_rows(router_plans),
                "issued_content": sorted(
                    getattr(router, "_issued_content_sha256").items()
                ),
                "consumed": _consumed_hashes(
                    getattr(router, "_consumed"),
                    router_plans,
                ),
            },
            "waypoint": {
                "issued": _object_rows(waypoint_objects),
                "consumed": _consumed_hashes(
                    getattr(waypoint, "_consumed"),
                    waypoint_objects,
                ),
            },
        }
    )


def _transaction_owner_state_sha256(state: _IntegrationState) -> str:
    """Hash downstream ingress plus coordinator registry and seal state."""

    source = getattr(state.evidence_issuer, "_outcome_source")
    return _sha256(
        {
            "schema": "lewm_two_resolution_navigation_transaction_state_v3",
            "downstream_owner_state_sha256": _owner_state_sha256(state),
            "outcome_sequence": getattr(source, "_sequence"),
            "outcome_issued": [
                {
                    "object_identity": identity,
                    "exact_object_identity": id(outcome),
                    "content_sha256": outcome.raw_outcome_content_sha256,
                    "outcome_sequence": outcome.outcome_sequence,
                }
                for identity, outcome in sorted(
                    getattr(source, "_issued").items()
                )
            ],
            "coordinator": {
                "stored_owner_state_sha256": state.owner_state_sha256,
                "controller_records": [
                    {
                        "object_identity": identity,
                        "artifact_identity": id(row.artifact),
                        "artifact_sha256": row.original_content_sha256,
                        "episode_authority_identity": id(row.episode_authority),
                        "episode_authority_sha256": (
                            row.episode_authority.content_sha256
                        ),
                    }
                    for identity, row in sorted(state.issued.items())
                ],
            },
        }
    )


@dataclass(frozen=True)
class _OwnerTransactionSnapshotV3:
    mapping_rows: tuple[tuple[object, str, object, dict[object, object]], ...]
    set_rows: tuple[tuple[object, str, object, set[object]], ...]
    scalar_rows: tuple[tuple[object, str, object], ...]
    nested_mapping_rows: tuple[tuple[object, dict[object, object]], ...]
    context_state_rows: tuple[tuple[object, object, object, object, object], ...]
    original_state_sha256: str

    @classmethod
    def capture(cls, state: _IntegrationState) -> "_OwnerTransactionSnapshotV3":
        evidence = state.evidence_issuer
        source = getattr(evidence, "_outcome_source")
        memory = state.target_memory
        view = state.view_issuer
        frontier = state.frontier
        router = state.target_router
        waypoint = state.waypoint_issuer
        mapping_specs = (
            (state, "issued"),
            (state.planner, "_issued_components"),
            (state.planner, "_issued_frontiers"),
            (state.planner, "_issued_paths"),
            (view, "_configuration_history"),
            (view, "_records"),
            (view, "_issued_states"),
            (frontier, "_issued_sets"),
            (frontier, "_issued_candidates"),
            (frontier, "_score_cache"),
            (source, "_issued"),
            (evidence, "_contexts"),
            (evidence, "_evidence"),
            (memory, "_mass"),
            (memory, "_unlocalized"),
            (memory, "_contexts"),
            (memory, "_positive_count"),
            (memory, "_negative_count"),
            (memory, "_chains"),
            (memory, "_issued_snapshots"),
            (router, "_plans"),
            (router, "_issued_content_sha256"),
            (waypoint, "_issued"),
        )
        set_specs = (
            (view, "_swept_physical_cells"),
            (source, "_consumed"),
            (evidence, "_consumed_evidence"),
            (memory, "_seen_context_ids"),
            (memory, "_seen_evidence_hashes"),
            (memory, "_seen_raw_outcomes"),
            (router, "_consumed"),
            (waypoint, "_consumed"),
        )
        scalar_specs = (
            (state, "owner_state_sha256"),
            (view, "_view_revision"),
            (view, "_view_step"),
            (source, "_sequence"),
            (evidence, "_sequence"),
            (memory, "_revision"),
            (memory, "_last_context_sequence"),
            (memory, "_last_pose_timestamp_ns"),
            (memory, "_immutable_binding"),
        )
        mapping_rows = []
        for owner, name in mapping_specs:
            container = getattr(owner, name)
            if type(container) is not dict:
                raise TwoResolutionNavigationIntegrationV3BindingError(
                    f"transaction owner {name} is not an exact dictionary"
                )
            mapping_rows.append((owner, name, container, dict(container)))
        set_rows = []
        for owner, name in set_specs:
            container = getattr(owner, name)
            if type(container) is not set:
                raise TwoResolutionNavigationIntegrationV3BindingError(
                    f"transaction owner {name} is not an exact set"
                )
            set_rows.append((owner, name, container, set(container)))
        mass = getattr(memory, "_mass")
        nested_mapping_rows = tuple(
            (rows, dict(rows)) for rows in mass.values()
        )
        context_state_rows = tuple(
            (
                context_state,
                context_state.writer,
                context_state.evidence,
                (
                    None
                    if context_state.writer is None
                    else context_state.writer._used
                ),
                context,
            )
            for context, context_state in getattr(evidence, "_contexts").values()
        )
        return cls(
            mapping_rows=tuple(mapping_rows),
            set_rows=tuple(set_rows),
            scalar_rows=tuple(
                (owner, name, getattr(owner, name))
                for owner, name in scalar_specs
            ),
            nested_mapping_rows=nested_mapping_rows,
            context_state_rows=context_state_rows,
            original_state_sha256=_transaction_owner_state_sha256(state),
        )

    def restore(self, state: _IntegrationState) -> None:
        for owner, name, value in self.scalar_rows:
            setattr(owner, name, value)
        for container, values in self.nested_mapping_rows:
            container.clear()
            container.update(values)
        for owner, name, container, values in self.mapping_rows:
            if getattr(owner, name) is not container:
                setattr(owner, name, container)
            container.clear()
            container.update(values)
        for owner, name, container, values in self.set_rows:
            if getattr(owner, name) is not container:
                setattr(owner, name, container)
            container.clear()
            container.update(values)
        for context_state, writer, evidence, writer_used, _context in (
            self.context_state_rows
        ):
            context_state.writer = writer
            context_state.evidence = evidence
            if writer is not None:
                writer._used = writer_used
        restored = _transaction_owner_state_sha256(state)
        if restored != self.original_state_sha256:
            raise TwoResolutionNavigationIntegrationV3BindingError(
                "V3 transaction rollback did not restore exact coordinator "
                "and owner state"
            )


def _assert_owner_state(state: _IntegrationState) -> None:
    _assert_append_only_outcome_issuance(state)
    current = _owner_state_sha256(state)
    if current != state.owner_state_sha256:
        raise TwoResolutionNavigationIntegrationV3BindingError(
            "shared downstream owner state changed outside this V3 coordinator"
        )


class TwoResolutionDevelopmentNavigationIntegrationV3:
    """Standalone preflight and lifecycle successor over passed owner APIs."""

    __slots__ = ("__weakref__",)

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
        episode_authority_issuer: TwoResolutionNavigationEpisodeAuthorityIssuerV3,
        _synthetic_development_fixture: bool = False,
        _synthetic_fault_after_stage: str | None = None,
    ) -> None:
        if _synthetic_development_fixture is not True:
            raise PermissionError(
                "no production two-resolution navigation integration V3 is configured"
            )
        if type(episode_authority_issuer) is not (
            TwoResolutionNavigationEpisodeAuthorityIssuerV3
        ):
            raise TypeError("episode_authority_issuer has the wrong exact type")
        if (
            _synthetic_fault_after_stage is not None
            and _synthetic_fault_after_stage
            not in _SYNTHETIC_TRANSACTION_FAULT_STAGES
        ):
            raise ValueError("unknown synthetic V3 transaction fault stage")
        if (
            episode_authority_issuer.projection is not projection
            or episode_authority_issuer.planner is not planner
        ):
            raise TwoResolutionNavigationIntegrationV3BindingError(
                "episode authority and downstream G3 owners differ"
            )
        exact = (
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
        for value, expected, name in exact:
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
            raise TwoResolutionNavigationIntegrationV3BindingError(
                "navigation authorities do not share the exact passed owner chain"
            )
        state = _IntegrationState(
            projection=projection,
            planner=planner,
            view_issuer=physical_view_state_issuer,
            frontier=frontier_viewpoint_planner,
            evidence_issuer=target_evidence_issuer,
            target_memory=target_memory,
            target_router=target_router,
            waypoint_issuer=world_waypoint_issuer,
            episode_issuer=episode_authority_issuer,
            synthetic_fault_after_stage=_synthetic_fault_after_stage,
        )
        _assert_append_only_outcome_issuance(state)
        state.owner_state_sha256 = _owner_state_sha256(state)
        _install_integration_state(self, state)

    @property
    def production_eligible(self) -> bool:
        return False

    @property
    def hardware_execution_authorized(self) -> bool:
        return False

    @property
    def issued_controller_count(self) -> int:
        return len(_integration_state(self).issued)

    @property
    def issued_observer_result_count(self) -> int:
        return len(_integration_state(self).observer_results)

    @property
    def development_owner_state_audit_sha256(self) -> str:
        state = _integration_state(self)
        _assert_owner_state(state)
        return _transaction_owner_state_sha256(state)

    def __copy__(self) -> "TwoResolutionDevelopmentNavigationIntegrationV3":
        raise TypeError("V3 development integrations are non-copyable")

    def __deepcopy__(
        self,
        memo: object,
    ) -> "TwoResolutionDevelopmentNavigationIntegrationV3":
        del memo
        raise TypeError("V3 development integrations are non-copyable")

    def __reduce_ex__(self, protocol: int) -> object:
        del protocol
        raise TypeError("V3 development integrations are non-serializable")

    def _controller_record(
        self,
        artifact: TwoResolutionDevelopmentControllerClaimTraceV3,
    ) -> _ControllerRecordV3:
        state = _integration_state(self)
        _assert_owner_state(state)
        if type(artifact) is not TwoResolutionDevelopmentControllerClaimTraceV3:
            raise TypeError("artifact has the wrong exact type")
        row = state.issued.get(id(artifact))
        if (
            row is None
            or row.artifact is not artifact
            or artifact._integration is not self
        ):
            raise TwoResolutionNavigationIntegrationV3BindingError(
                "V3 controller claim trace is not the exact live object issued here"
            )
        artifact._assert_structural_integrity()
        if artifact.content_sha256 != row.original_content_sha256:
            raise TwoResolutionNavigationIntegrationV3BindingError(
                "V3 controller claim trace differs from its original issuance"
            )
        state.episode_issuer.assert_episode_authority(row.episode_authority)
        if row.episode_authority.content_sha256 != artifact.episode_authority_sha256:
            raise TwoResolutionNavigationIntegrationV3BindingError(
                "V3 episode authority changed after controller issuance"
            )
        return row

    def assert_controller_claim_trace(
        self,
        artifact: TwoResolutionDevelopmentControllerClaimTraceV3,
    ) -> None:
        self._controller_record(artifact)

    def issue_controller_claim_trace(
        self,
        *,
        episode_authority: TwoResolutionNavigationEpisodeAuthorityV3,
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
    ) -> TwoResolutionDevelopmentControllerClaimTraceV3:
        state = _integration_state(self)
        _assert_owner_state(state)
        authority_record = state.episode_issuer.resolve_for_controller(
            authority=episode_authority,
            snapshot=snapshot,
            component=component,
            physical_manifest=physical_manifest,
        )
        if type(outcome) is not SyntheticV5TargetOutcomeV1:
            raise TypeError("outcome has the wrong exact type")
        outcome_source = getattr(state.evidence_issuer, "_outcome_source")
        if type(outcome_source) is not SyntheticV5TargetOutcomeIssuerV1:
            raise TwoResolutionNavigationIntegrationV3BindingError(
                "G5 evidence issuer lost its exact outcome source"
            )
        outcome_source.assert_issued(outcome)
        expected_object = episode_authority.target_object_map().get(outcome.target_id)
        if (
            expected_object is None
            or _nonempty(task_object_id, "task_object_id") != expected_object
        ):
            raise TwoResolutionNavigationIntegrationV3BindingError(
                "semantic target does not map to the requested task object"
            )
        if episode_id != episode_authority.episode_id:
            raise TwoResolutionNavigationIntegrationV3BindingError(
                "episode identity differs from episode authority"
            )
        supplied_task_ids = (
            episode_authority.task_object_ids
            if task_object_ids is None
            else tuple(task_object_ids)
        )
        if supplied_task_ids != episode_authority.task_object_ids:
            raise ValueError("task_object_ids are not the canonical bound task set")
        start_cell = _configuration_cell(
            start_configuration_cell,
            "start_configuration_cell",
        )
        if start_cell not in component.cells or start_cell not in snapshot.free_cells:
            raise ValueError(
                "start_configuration_cell must be current component-confirmed FREE"
            )
        yaw = _finite(current_yaw_rad, "current_yaw_rad")
        xy_variance = _finite(pose_xy_variance_m2, "pose_xy_variance_m2")
        yaw_variance = _finite(
            pose_yaw_variance_rad2,
            "pose_yaw_variance_rad2",
        )
        if xy_variance < 0.0 or yaw_variance < 0.0:
            raise ValueError("pose variances must be non-negative")
        reference = object_id_reference(expected_object)
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
            task_object_ids=supplied_task_ids,
        )
        if (
            task_ids != episode_authority.task_object_ids
            or task_hash != episode_authority.task_object_set_sha256
            or manifest_sha256(physical_manifest)
            != authority_record.original_manifest_sha256
        ):
            raise TwoResolutionNavigationIntegrationV3BindingError(
                "preflight task or physical manifest differs from episode authority"
            )

        transaction = _OwnerTransactionSnapshotV3.capture(state)
        try:
            view_state = state.view_issuer.issue(
                snapshot,
                pose_xy_variance_m2=xy_variance,
                pose_yaw_variance_rad2=yaw_variance,
            )
            _synthetic_transaction_fault(state, "g4_view_issue")
            candidate_set = state.frontier.generate(
                snapshot,
                view_state,
                start_configuration_cell=start_cell,
                current_yaw_rad=yaw,
            )
            _synthetic_transaction_fault(state, "g4_frontier_generate")
            selected = state.frontier.select(snapshot, view_state, candidate_set)
            _synthetic_transaction_fault(state, "g4_frontier_select")
            if selected is None:
                raise TwoResolutionNavigationIntegrationV3Error(
                    "G4 produced no development frontier/viewpoint candidate"
                )
            state.frontier.validate_candidate(
                snapshot,
                view_state,
                candidate_set,
                selected,
            )
            _synthetic_transaction_fault(state, "g4_frontier_validate")
            context = state.evidence_issuer.issue_context(
                snapshot,
                component,
                outcome,
            )
            _synthetic_transaction_fault(state, "g5_context_issue")
            writer = state.evidence_issuer.open_writer(context)
            _synthetic_transaction_fault(state, "g5_writer_open")
            evidence = (
                writer.issue_positive()
                if outcome.outcome_kind == "positive"
                else writer.issue_negative()
            )
            _synthetic_transaction_fault(state, "g5_evidence_issue")
            posterior = state.target_memory.apply(context, evidence)
            _synthetic_transaction_fault(state, "g5_posterior_apply")
            state.target_memory.assert_current_snapshot(posterior)
            _synthetic_transaction_fault(state, "g5_posterior_validate")
            route = state.target_router.issue(
                snapshot=snapshot,
                component=component,
                posterior=posterior,
                start_cell=start_cell,
                start_yaw_rad=yaw,
            )
            _synthetic_transaction_fault(state, "router_issue")
            waypoint = state.waypoint_issuer.issue(snapshot, route.path)
            _synthetic_transaction_fault(state, "waypoint_issue")
            state.target_router.validate(
                snapshot=snapshot,
                component=component,
                posterior=posterior,
                plan=route,
            )
            _synthetic_transaction_fault(state, "router_validate")
            state.waypoint_issuer.validate(snapshot, route.path, waypoint)
            _synthetic_transaction_fault(state, "waypoint_validate")
            if (
                route.production_promotion_authorized is not False
                or route.hardware_execution_authorized is not False
                or route.receipt.production_promotion_authorized is not False
                or route.receipt.hardware_execution_authorized is not False
                or waypoint.production_promotion_authorized is not False
                or waypoint.hardware_execution_authorized is not False
            ):
                raise TwoResolutionNavigationIntegrationV3BindingError(
                    "retained route or waypoint gained forbidden authority"
                )
            payload = {
                "schema": "lewm_two_resolution_navigation_controller_payload_v3",
                "scene_id": physical_manifest.scene_id,
                "physical_manifest_sha256": manifest_sha256(physical_manifest),
                "g3": {
                    "snapshot_sha256": snapshot.content_sha256,
                    "component_sha256": component.content_sha256,
                    "physical_map_frame_sha256": (
                        snapshot.physical_map_frame_sha256
                    ),
                    "configuration_map_frame_sha256": (
                        snapshot.configuration_map_frame_sha256
                    ),
                    "physical_revision": snapshot.physical_revision,
                    "configuration_revision": snapshot.configuration_revision,
                },
                "g4": {
                    "physical_view_state_sha256": view_state.content_sha256,
                    "frontier_candidate_set_sha256": candidate_set.content_sha256,
                    "selected_frontier_candidate_sha256": selected.content_sha256,
                },
                "g5": {
                    "raw_outcome_content_sha256": (
                        outcome.raw_outcome_content_sha256
                    ),
                    "target_context_sha256": context.content_sha256,
                    "target_evidence_sha256": evidence.content_sha256,
                    "target_posterior_sha256": posterior.content_sha256,
                    "target_id": outcome.target_id,
                },
                "task_object_id": expected_object,
                "target_route_plan": route.to_dict(),
                "world_waypoint_receipt": waypoint.to_dict(),
                "controller_claim_attempt": attempt,
                "raw_claim_trace": trace,
                "task_object_ids": list(task_ids),
                "task_object_set_sha256": task_hash,
                "development_execution_eligible": True,
                "production_promotion_authorized": False,
                "hardware_execution_authorized": False,
            }
            _synthetic_transaction_fault(state, "controller_payload_build")
            artifact = TwoResolutionDevelopmentControllerClaimTraceV3(
                episode_authority_sha256=episode_authority.content_sha256,
                semantic_target_id=outcome.target_id,
                task_object_id=expected_object,
                episode_authority=episode_authority.to_dict(),
                controller_payload=payload,
                _integration=self,
            )
            _synthetic_transaction_fault(state, "artifact_construct")
            state.waypoint_issuer.validate(
                snapshot,
                route.path,
                waypoint,
                consume=True,
            )
            _synthetic_transaction_fault(state, "waypoint_consume")
            state.target_router.validate(
                snapshot=snapshot,
                component=component,
                posterior=posterior,
                plan=route,
                consume=True,
            )
            _synthetic_transaction_fault(state, "router_consume")
            committed_owner_state_sha256 = _owner_state_sha256(state)
            _synthetic_transaction_fault(state, "owner_state_seal")
            record = _ControllerRecordV3(
                artifact=artifact,
                original_content_sha256=artifact.content_sha256,
                episode_authority=episode_authority,
            )
            _synthetic_transaction_fault(state, "controller_record_construct")
            if id(artifact) in state.issued:
                raise TwoResolutionNavigationIntegrationV3BindingError(
                    "V3 controller artifact identity is already registered"
                )
            state.issued[id(artifact)] = record
            _synthetic_transaction_fault(state, "controller_registry_insert")
            state.owner_state_sha256 = committed_owner_state_sha256
            _synthetic_transaction_fault(state, "coordinator_seal_assign")
        except BaseException:
            try:
                transaction.restore(state)
            except BaseException as rollback_error:
                raise TwoResolutionNavigationIntegrationV3BindingError(
                    "V3 transaction failed and exact rollback could not be proven"
                ) from rollback_error
            raise
        return artifact

    def evaluate_observer_only(
        self,
        artifact: TwoResolutionDevelopmentControllerClaimTraceV3,
        *,
        evaluator_access_ledger: Mapping[str, int],
    ) -> TwoResolutionObserverClaimEvaluationV3:
        state = _integration_state(self)
        row = self._controller_record(artifact)
        if id(artifact) in state.observed_controller_ids:
            raise TwoResolutionNavigationIntegrationV3ReplayError(
                "V3 controller claim trace was already evaluated"
            )
        ledger = _validate_empty_ledger(evaluator_access_ledger)
        from lewm.benchmarks.go2_physical_claim_observer import (
            evaluate_runtime_claim_trace,
        )

        authority_record = state.episode_issuer._record(row.episode_authority)
        evaluated = evaluate_runtime_claim_trace(
            artifact.raw_claim_trace,
            authority_record.physical_manifest,
            row.episode_authority.task_object_ids,
            row.episode_authority.task_object_set_sha256,
        )
        result = TwoResolutionObserverClaimEvaluationV3(
            controller_claim_trace_sha256=artifact.content_sha256,
            episode_authority_sha256=artifact.episode_authority_sha256,
            evaluator_access_ledger=ledger,
            evaluated_claim_trace=evaluated,
            _integration=self,
        )
        state.observer_results[id(result)] = _ObserverRecordV3(
            result=result,
            original_content_sha256=result.content_sha256,
        )
        state.observed_controller_ids.add(id(artifact))
        return result

    def _observer_record(
        self,
        result: TwoResolutionObserverClaimEvaluationV3,
        *,
        consume: bool = False,
    ) -> _ObserverRecordV3:
        state = _integration_state(self)
        _assert_owner_state(state)
        if type(result) is not TwoResolutionObserverClaimEvaluationV3:
            raise TypeError("observer result has the wrong exact type")
        row = state.observer_results.get(id(result))
        if row is None or row.result is not result or result._integration is not self:
            raise TwoResolutionNavigationIntegrationV3BindingError(
                "V3 observer result is not the exact live object issued here"
            )
        result._assert_structural_integrity()
        if result.content_sha256 != row.original_content_sha256:
            raise TwoResolutionNavigationIntegrationV3BindingError(
                "V3 observer result differs from its original issuance"
            )
        if id(result) in state.consumed_observer_result_ids:
            raise TwoResolutionNavigationIntegrationV3ReplayError(
                "V3 observer result was already consumed"
            )
        if consume:
            state.consumed_observer_result_ids.add(id(result))
        return row

    def assert_observer_evaluation(
        self,
        result: TwoResolutionObserverClaimEvaluationV3,
    ) -> None:
        self._observer_record(result)

    def consume_observer_evaluation(
        self,
        result: TwoResolutionObserverClaimEvaluationV3,
    ) -> None:
        self._observer_record(result, consume=True)


def require_production_two_resolution_navigation_integration_v3() -> object:
    if PRODUCTION_TWO_RESOLUTION_NAVIGATION_INTEGRATION_V3 is None:
        raise PermissionError(
            "no production two-resolution navigation integration V3 is configured"
        )
    return PRODUCTION_TWO_RESOLUTION_NAVIGATION_INTEGRATION_V3


__all__ = [
    "TwoResolutionDevelopmentControllerClaimTraceV3",
    "TwoResolutionDevelopmentNavigationIntegrationV3",
    "TwoResolutionNavigationEpisodeAuthorityIssuerV3",
    "TwoResolutionNavigationEpisodeAuthorityV3",
    "TwoResolutionNavigationIntegrationV3BindingError",
    "TwoResolutionNavigationIntegrationV3Error",
    "TwoResolutionNavigationIntegrationV3ReplayError",
    "TwoResolutionNavigationTargetObjectBindingV3",
    "TwoResolutionObserverClaimEvaluationV3",
    "require_production_two_resolution_navigation_integration_v3",
]
