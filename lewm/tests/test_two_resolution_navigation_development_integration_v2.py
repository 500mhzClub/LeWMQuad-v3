from __future__ import annotations

import copy
from dataclasses import replace
import math

import pytest

from lewm.benchmarks.go2_physical_claim_observer import (
    empty_evaluator_access_ledger,
)
from lewm.planning import two_resolution_navigation_development_integration_v2 as module
from lewm.planning.revisioned_physical_configuration_memory import (
    SnapshotBindingError,
)
from lewm.planning.two_resolution_navigation_development_integration_v1 import (
    TwoResolutionDevelopmentNavigationIntegrationV1,
)
from lewm.planning.two_resolution_navigation_development_integration_v2 import (
    TwoResolutionDevelopmentNavigationIntegrationV2,
    TwoResolutionNavigationEpisodeAuthorityIssuerV2,
    TwoResolutionNavigationIntegrationV2BindingError,
    TwoResolutionNavigationIntegrationV2ReplayError,
    require_production_two_resolution_navigation_integration_v2,
)
from lewm.tests.test_two_resolution_navigation_development_integration_v1 import (
    CALIBRATION,
    CHECKPOINT,
    G5_TARGET_ID,
    RUNNER,
    START,
    TARGET_ID,
    _bundle,
    _hash,
    _manifest,
)


EPISODE_ID = "integration-v2-episode"


def _v2_bundle(*, fault_after_stage: str | None = None):
    bundle = _bundle()
    authority_issuer = TwoResolutionNavigationEpisodeAuthorityIssuerV2(
        projection=bundle.projection,
        planner=bundle.planner,
        physical_manifest=bundle.manifest,
        episode_id=EPISODE_ID,
        target_object_mapping={G5_TARGET_ID: TARGET_ID},
        _synthetic_development_fixture=True,
    )
    authority = authority_issuer.issue(bundle.snapshot, bundle.component)
    retained_v1 = bundle.integration
    integration = TwoResolutionDevelopmentNavigationIntegrationV2(
        projection=bundle.projection,
        planner=bundle.planner,
        physical_view_state_issuer=retained_v1._view_issuer,
        frontier_viewpoint_planner=retained_v1._frontier,
        target_evidence_issuer=retained_v1._evidence_issuer,
        target_memory=bundle.target_memory,
        target_router=bundle.router,
        world_waypoint_issuer=bundle.waypoint_issuer,
        episode_authority_issuer=authority_issuer,
        _synthetic_development_fixture=True,
        _synthetic_fault_after_stage=fault_after_stage,
    )
    bundle.authority_issuer = authority_issuer
    bundle.authority = authority
    bundle.integration_v2 = integration
    return bundle


def _issue_v2(bundle, **overrides):
    values = {
        "episode_authority": bundle.authority,
        "snapshot": bundle.snapshot,
        "component": bundle.component,
        "outcome": bundle.outcome,
        "start_configuration_cell": START,
        "current_yaw_rad": -math.pi / 2.0,
        "physical_manifest": bundle.manifest,
        "trace_id": "integration-v2-trace",
        "episode_id": EPISODE_ID,
        "event_id": "integration-v2-claim",
        "tick": 25,
        "event_index": 0,
        "robot_pose_world_xy_yaw": (16.75, -5.75, -math.pi / 2.0),
        "task_object_id": TARGET_ID,
        "pose_xy_variance_m2": 0.0025,
        "pose_yaw_variance_rad2": 0.0004,
    }
    values.update(overrides)
    return bundle.integration_v2.issue_controller_claim_trace(**values)


def _blue_outcome(bundle):
    original = bundle.outcome
    return bundle.outcome_source.issue(
        snapshot=bundle.snapshot,
        outcome_kind="positive",
        target_id="blue",
        pose_timestamp_ns=101,
        runner_execution_identity=RUNNER,
        checkpoint_file_sha256=CHECKPOINT,
        raw_outcome_file_sha256=_hash("integration-v2-blue-outcome"),
        camera_calibration_sha256=CALIBRATION,
        pose_provenance_sha256=_hash("integration-v2-blue-pose"),
        free_physical_cells=original.free_physical_cells,
        unknown_physical_cells=original.unknown_physical_cells,
        target_physical_cells=original.target_physical_cells,
        visible_physical_cells=original.visible_physical_cells,
        physical_probability=original.physical_probability,
        unlocalized_probability=original.unlocalized_probability,
        confidence=original.confidence,
    )


def _next_red_outcome(bundle):
    original = bundle.outcome
    return bundle.outcome_source.issue(
        snapshot=bundle.snapshot,
        outcome_kind="positive",
        target_id=G5_TARGET_ID,
        pose_timestamp_ns=original.pose_timestamp_ns + 1,
        runner_execution_identity=RUNNER,
        checkpoint_file_sha256=CHECKPOINT,
        raw_outcome_file_sha256=_hash("integration-v2-next-red-outcome"),
        camera_calibration_sha256=CALIBRATION,
        pose_provenance_sha256=_hash("integration-v2-next-red-pose"),
        free_physical_cells=original.free_physical_cells,
        unknown_physical_cells=original.unknown_physical_cells,
        target_physical_cells=original.target_physical_cells,
        visible_physical_cells=original.visible_physical_cells,
        physical_probability=original.physical_probability,
        unlocalized_probability=original.unlocalized_probability,
        confidence=original.confidence,
    )


def _assert_uncommitted(bundle, outcome=None) -> None:
    checked = bundle.outcome if outcome is None else outcome
    assert checked.raw_outcome_content_sha256 not in (
        bundle.target_memory._seen_raw_outcomes
    )
    assert id(checked) not in bundle.outcome_source._consumed
    assert bundle.integration_v2.issued_controller_count == 0
    assert bundle.integration_v2.issued_observer_result_count == 0
    assert bundle.integration._evidence_issuer._contexts == {}
    assert bundle.router._plans == {}
    assert bundle.waypoint_issuer._issued == {}
    assert checked.target_id not in bundle.target_memory._contexts


def test_v2_happy_path_binds_episode_and_observer_original_digest() -> None:
    bundle = _v2_bundle()
    artifact = _issue_v2(bundle)
    artifact.assert_integrity()
    encoded = artifact.to_dict()
    assert encoded["semantic_target_id"] == G5_TARGET_ID
    assert encoded["task_object_id"] == TARGET_ID
    assert encoded["episode_authority"]["scene_id"] == bundle.manifest.scene_id
    assert encoded["episode_authority"]["g3"]["physical_session_id"] == (
        bundle.snapshot.physical_map_frame.session_id
    )
    assert artifact.production_promotion_authorized is False
    assert artifact.hardware_execution_authorized is False

    result = bundle.integration_v2.evaluate_observer_only(
        artifact,
        evaluator_access_ledger=empty_evaluator_access_ledger(),
    )
    result.assert_integrity()
    assert result.evaluated_claim_trace["physical_claim_summary"][
        "all_targets_claimed"
    ] is True
    assert result.production_promotion_authorized is False
    assert result.hardware_execution_authorized is False

    direct = module.TwoResolutionObserverClaimEvaluationV2(
        controller_claim_trace_sha256=artifact.content_sha256,
        episode_authority_sha256=artifact.episode_authority_sha256,
        evaluator_access_ledger=empty_evaluator_access_ledger(),
        evaluated_claim_trace=result.evaluated_claim_trace,
        _integration=bundle.integration_v2,
    )
    with pytest.raises(
        TwoResolutionNavigationIntegrationV2BindingError,
        match="exact live",
    ):
        direct.assert_integrity()
    clone = replace(result)
    with pytest.raises(
        TwoResolutionNavigationIntegrationV2BindingError,
        match="exact live",
    ):
        clone.assert_integrity()
    foreign = _v2_bundle()
    with pytest.raises(
        TwoResolutionNavigationIntegrationV2BindingError,
        match="exact live",
    ):
        foreign.integration_v2.assert_observer_evaluation(result)

    object.__setattr__(
        result,
        "evaluated_claim_trace",
        {
            "schema": "forged-v2-observer-result",
            "physical_claim_summary": {"all_targets_claimed": False},
        },
    )
    object.__setattr__(result, "content_sha256", module._sha256(result.to_dict(False)))
    with pytest.raises(
        TwoResolutionNavigationIntegrationV2BindingError,
        match="original issuance",
    ):
        result.assert_integrity()


def test_v2_all_claim_preflight_failures_are_retry_safe() -> None:
    bundle = _v2_bundle()
    invalid_calls = (
        ({"task_object_id": "not-a-manifest-task"}, "target.*object"),
        ({"physical_manifest": _manifest("foreign-v2-scene")}, "scene|manifest"),
        ({"episode_id": "foreign-episode"}, "episode identity"),
        ({"task_object_ids": ("not-a-manifest-task",)}, "canonical bound"),
        ({"event_id": ""}, "nonempty"),
        ({"tick": -1}, "nonnegative"),
        ({"robot_pose_world_xy_yaw": (float("nan"), 0.0, 0.0)}, "finite"),
        (
            {"start_configuration_cell": (999, 999)},
            "component-confirmed FREE",
        ),
    )
    for overrides, message in invalid_calls:
        with pytest.raises(
            (ValueError, TwoResolutionNavigationIntegrationV2BindingError),
            match=message,
        ):
            _issue_v2(bundle, **overrides)
        _assert_uncommitted(bundle)

    artifact = _issue_v2(bundle)
    bundle.integration_v2.assert_controller_claim_trace(artifact)
    clone = replace(artifact)
    with pytest.raises(
        TwoResolutionNavigationIntegrationV2BindingError,
        match="exact live",
    ):
        clone.assert_integrity()

    result = bundle.integration_v2.evaluate_observer_only(
        artifact,
        evaluator_access_ledger=empty_evaluator_access_ledger(),
    )
    with pytest.raises(TwoResolutionNavigationIntegrationV2ReplayError):
        bundle.integration_v2.evaluate_observer_only(
            artifact,
            evaluator_access_ledger=empty_evaluator_access_ledger(),
        )
    bundle.integration_v2.consume_observer_evaluation(result)
    with pytest.raises(TwoResolutionNavigationIntegrationV2ReplayError):
        bundle.integration_v2.consume_observer_evaluation(result)

    artifact.controller_payload["semantic_splice"] = "forged"
    object.__setattr__(
        artifact,
        "content_sha256",
        module._sha256(artifact.to_dict(False)),
    )
    with pytest.raises(
        TwoResolutionNavigationIntegrationV2BindingError,
        match="original issuance",
    ):
        bundle.integration_v2.assert_controller_claim_trace(artifact)


def test_v2_cross_target_splice_rejects_without_consuming_outcome() -> None:
    bundle = _v2_bundle()
    blue = _blue_outcome(bundle)
    with pytest.raises(
        TwoResolutionNavigationIntegrationV2BindingError,
        match="target.*object",
    ):
        _issue_v2(bundle, outcome=blue)
    _assert_uncommitted(bundle, blue)
    assert bundle.outcome.raw_outcome_content_sha256 not in (
        bundle.target_memory._seen_raw_outcomes
    )


def test_v2_episode_authority_is_complete_exact_live_and_session_bound() -> None:
    bundle = _v2_bundle()
    authority = bundle.authority
    assert authority.target_object_map() == {G5_TARGET_ID: TARGET_ID}
    authority.assert_integrity()
    with pytest.raises(TwoResolutionNavigationIntegrationV2ReplayError):
        bundle.authority_issuer.issue(bundle.snapshot, bundle.component)
    with pytest.raises(TypeError, match="non-copyable"):
        copy.copy(authority)
    clone = replace(authority)
    with pytest.raises(
        TwoResolutionNavigationIntegrationV2BindingError,
        match="exact live",
    ):
        clone.assert_integrity()

    foreign = _v2_bundle()
    with pytest.raises(
        TwoResolutionNavigationIntegrationV2BindingError,
        match="exact live",
    ):
        _issue_v2(bundle, episode_authority=foreign.authority)
    _assert_uncommitted(bundle)

    object.__setattr__(authority, "physical_session_id", "foreign-session")
    object.__setattr__(
        authority,
        "content_sha256",
        module._sha256(authority.to_dict(False)),
    )
    with pytest.raises(
        TwoResolutionNavigationIntegrationV2BindingError,
        match="original issuance",
    ):
        authority.assert_integrity()

    duplicate = _bundle()
    with pytest.raises(ValueError, match="complete one-to-one"):
        TwoResolutionNavigationEpisodeAuthorityIssuerV2(
            projection=duplicate.projection,
            planner=duplicate.planner,
            physical_manifest=duplicate.manifest,
            episode_id=EPISODE_ID,
            target_object_mapping={"red": TARGET_ID, "blue": TARGET_ID},
            _synthetic_development_fixture=True,
        )

    scene_splice = _bundle()
    scene_splice_issuer = TwoResolutionNavigationEpisodeAuthorityIssuerV2(
        projection=scene_splice.projection,
        planner=scene_splice.planner,
        physical_manifest=_manifest("foreign-v2-episode-scene"),
        episode_id=EPISODE_ID,
        target_object_mapping={G5_TARGET_ID: TARGET_ID},
        _synthetic_development_fixture=True,
    )
    with pytest.raises(
        TwoResolutionNavigationIntegrationV2BindingError,
        match="sessions.*manifest scene",
    ):
        scene_splice_issuer.issue(
            scene_splice.snapshot,
            scene_splice.component,
        )

    stale = _v2_bundle()
    stale.projection.project()
    with pytest.raises(SnapshotBindingError):
        _issue_v2(stale)
    _assert_uncommitted(stale)
    with pytest.raises(ValueError, match="complete one-to-one"):
        TwoResolutionNavigationEpisodeAuthorityIssuerV2(
            projection=duplicate.projection,
            planner=duplicate.planner,
            physical_manifest=duplicate.manifest,
            episode_id=EPISODE_ID,
            target_object_mapping={},
            _synthetic_development_fixture=True,
        )


def test_v2_ledger_and_production_authority_fail_closed_before_observer() -> None:
    bundle = _v2_bundle()
    artifact = _issue_v2(bundle)
    for ledger in (
        {
            "evaluator_output_reads_by_controller": 1,
            "evaluator_callbacks_into_controller": 0,
            "evaluator_derived_termination_signals": 0,
        },
        {
            "evaluator_output_reads_by_controller": 0,
            "evaluator_callbacks_into_controller": 0,
        },
        {
            "evaluator_output_reads_by_controller": False,
            "evaluator_callbacks_into_controller": 0,
            "evaluator_derived_termination_signals": 0,
        },
    ):
        with pytest.raises(
            TwoResolutionNavigationIntegrationV2BindingError,
            match="actually empty",
        ):
            bundle.integration_v2.evaluate_observer_only(
                artifact,
                evaluator_access_ledger=ledger,
            )
    result = bundle.integration_v2.evaluate_observer_only(
        artifact,
        evaluator_access_ledger=empty_evaluator_access_ledger(),
    )
    bundle.integration_v2.assert_observer_evaluation(result)

    with pytest.raises(PermissionError):
        require_production_two_resolution_navigation_integration_v2()
    retained_v1 = bundle.integration
    with pytest.raises(PermissionError):
        TwoResolutionDevelopmentNavigationIntegrationV2(
            projection=bundle.projection,
            planner=bundle.planner,
            physical_view_state_issuer=retained_v1._view_issuer,
            frontier_viewpoint_planner=retained_v1._frontier,
            target_evidence_issuer=retained_v1._evidence_issuer,
            target_memory=bundle.target_memory,
            target_router=bundle.router,
            world_waypoint_issuer=bundle.waypoint_issuer,
            episode_authority_issuer=bundle.authority_issuer,
        )
    with pytest.raises(PermissionError):
        TwoResolutionNavigationEpisodeAuthorityIssuerV2(
            projection=bundle.projection,
            planner=bundle.planner,
            physical_manifest=bundle.manifest,
            episode_id=EPISODE_ID,
            target_object_mapping={G5_TARGET_ID: TARGET_ID},
        )


@pytest.mark.parametrize(
    "fault_after_stage",
    sorted(module._SYNTHETIC_TRANSACTION_FAULT_STAGES),
)
def test_v2_every_post_mutation_failure_rolls_back_and_retries(
    fault_after_stage: str,
) -> None:
    bundle = _v2_bundle(fault_after_stage=fault_after_stage)
    before = bundle.integration_v2.development_owner_state_audit_sha256
    with pytest.raises(
        RuntimeError,
        match=f"synthetic V2 transaction fault after {fault_after_stage}",
    ):
        _issue_v2(bundle)
    assert bundle.integration_v2.development_owner_state_audit_sha256 == before
    _assert_uncommitted(bundle)

    artifact = _issue_v2(bundle)
    artifact.assert_integrity()
    assert bundle.integration_v2.issued_controller_count == 1


def test_v2_has_no_reachable_v1_engine_and_rejects_shared_owner_bypass() -> None:
    bundle = _v2_bundle()
    integration = bundle.integration_v2
    assert not hasattr(integration, "__dict__")
    for forbidden in (
        "_v1",
        "_outcome_source",
        "_evidence_issuer",
        "_target_memory",
        "_target_router",
        "_waypoint_issuer",
        "_episode_issuer",
        "_issued",
        "_capability",
    ):
        assert not hasattr(integration, forbidden)

    retained = bundle.integration
    unbound_v1 = TwoResolutionDevelopmentNavigationIntegrationV1(
        projection=bundle.projection,
        planner=bundle.planner,
        physical_view_state_issuer=retained._view_issuer,
        frontier_viewpoint_planner=retained._frontier,
        target_evidence_issuer=retained._evidence_issuer,
        target_memory=bundle.target_memory,
        target_router=bundle.router,
        world_waypoint_issuer=bundle.waypoint_issuer,
        _synthetic_development_fixture=True,
    )
    with pytest.raises(ValueError):
        unbound_v1.issue_controller_claim_trace(
            snapshot=bundle.snapshot,
            component=bundle.component,
            outcome=bundle.outcome,
            start_configuration_cell=START,
            current_yaw_rad=-math.pi / 2.0,
            physical_manifest=bundle.manifest,
            trace_id="unbound-v1-trace",
            episode_id="unbound-v1-episode",
            event_id="unbound-v1-event",
            tick=25,
            event_index=0,
            robot_pose_world_xy_yaw=(16.75, -5.75, -math.pi / 2.0),
            task_object_id="not-a-manifest-task",
            pose_xy_variance_m2=0.0025,
            pose_yaw_variance_rad2=0.0004,
        )
    next_outcome = _next_red_outcome(bundle)
    with pytest.raises(
        TwoResolutionNavigationIntegrationV2BindingError,
        match="shared downstream owner state changed",
    ):
        _issue_v2(bundle, outcome=next_outcome)
    assert id(next_outcome) not in bundle.outcome_source._consumed
    assert integration.issued_controller_count == 0


def test_v2_rechecks_exact_manifest_before_observer_scoring() -> None:
    bundle = _v2_bundle()
    artifact = _issue_v2(bundle)
    object.__setattr__(bundle.manifest, "scene_id", "post-controller-splice")

    with pytest.raises(
        TwoResolutionNavigationIntegrationV2BindingError,
        match="manifest.*original issuance",
    ):
        artifact.assert_integrity()
    with pytest.raises(
        TwoResolutionNavigationIntegrationV2BindingError,
        match="manifest.*original issuance",
    ):
        bundle.integration_v2.evaluate_observer_only(
            artifact,
            evaluator_access_ledger=empty_evaluator_access_ledger(),
        )
    assert bundle.integration_v2.issued_observer_result_count == 0
