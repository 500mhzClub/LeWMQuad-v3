from __future__ import annotations

import copy
from dataclasses import replace
import hashlib
import math
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from lewm.benchmarks.go2_g3_exact_physical_equivalence_v2 import (
    _build_projected_snapshot,
)
from lewm.planning.revisioned_physical_configuration_memory import (
    PhysicalLabel,
    SnapshotBindingError,
)
from lewm.planning.two_resolution_frontier_viewpoint_v2 import (
    TwoResolutionFrontierViewpointPlannerV2,
    TwoResolutionPhysicalViewStateIssuerV2,
)
from lewm.planning.two_resolution_navigation_development_integration_v1 import (
    TwoResolutionDevelopmentNavigationIntegrationV1,
    TwoResolutionNavigationIntegrationBindingError,
    TwoResolutionNavigationIntegrationReplayError,
    require_production_two_resolution_navigation_integration_v1,
)
from lewm.planning.two_resolution_reversible_target_belief_v1 import (
    TwoResolutionReversibleTargetBeliefMemoryV1,
)
from lewm.planning.two_resolution_target_evidence_v1 import (
    PhysicalCellProbabilityV1,
    SyntheticV5TargetOutcomeIssuerV1,
    TwoResolutionTargetEvidenceBindingError,
    TwoResolutionTargetEvidenceIssuerV1,
    V5RunnerExecutionIdentityV1,
)
from lewm.planning.two_resolution_target_router_v2 import (
    TwoResolutionDeterministicTargetRouterV2,
)
from lewm.planning.two_resolution_world_waypoint_adapter_v2 import (
    ConfigurationPathWorldWaypointIssuerV2,
)
from lewm_worlds.manifest import (
    BoxObject,
    CameraValidityConstraints,
    SceneManifest,
    SpawnSpec,
)


ORIGIN = (13.7, -9.3)
PHYSICAL_SHAPE = (84, 96)
CONFIGURATION_SHAPE = (42, 48)
START = (30, 35)
TARGET_ID = "beacon_red"
G5_TARGET_ID = "red"


def _hash(label: str) -> str:
    return hashlib.sha256(label.encode("ascii")).hexdigest()


RUNNER = V5RunnerExecutionIdentityV1(
    entrypoint_wrapper_file_sha256=_hash("integration-wrapper"),
    captured_launcher_file_sha256=_hash("integration-launcher"),
    captured_core_file_sha256=_hash("integration-core"),
)
CHECKPOINT = _hash("integration-checkpoint")
CALIBRATION = _hash("integration-calibration")


def _manifest(
    scene_id: str = "two-resolution-development-integration",
) -> SceneManifest:
    target = BoxObject(
        object_id=TARGET_ID,
        kind="landmark",
        center_xyz_m=(16.75, -6.25, 0.5),
        size_xyz_m=(0.2, 0.2, 1.0),
        yaw_rad=0.0,
        material_id="landmark_red",
    )
    return SceneManifest(
        scene_id=scene_id,
        family="two-resolution-development-integration",
        difficulty_tier="unit",
        topology_seed=1,
        visual_seed=2,
        physics_seed=3,
        world_bounds_xy_m=((10.0, -12.0), (20.0, 0.0)),
        spawn=SpawnSpec(
            xyz_m=(16.75, -5.75, 0.35),
            quat_wxyz=(1.0, 0.0, 0.0, 0.0),
        ),
        graph_nodes=(),
        graph_edges=(),
        obstacles=(),
        landmarks=(target,),
        camera_constraints=CameraValidityConstraints(
            min_wall_thickness_m=0.08,
            near_m=0.05,
            far_m=200.0,
            min_camera_clearance_m=0.10,
        ),
        split="train",
    )


def _bundle() -> SimpleNamespace:
    physical = np.full(PHYSICAL_SHAPE, int(PhysicalLabel.UNKNOWN), dtype=np.uint8)
    physical[28:76, 36:88] = int(PhysicalLabel.FREE)
    manifest = _manifest()
    memory, snapshot, planner = _build_projected_snapshot(
        manifest,
        physical,
        origin_xy_m=ORIGIN,
        configuration_shape=CONFIGURATION_SHAPE,
    )
    projection = planner._projection
    component = planner.connected_component(snapshot, START)
    view_issuer = TwoResolutionPhysicalViewStateIssuerV2(memory, projection)
    frontier = TwoResolutionFrontierViewpointPlannerV2(planner, view_issuer)
    outcome_source = SyntheticV5TargetOutcomeIssuerV1(
        _synthetic_test_fixture=True
    )
    evidence_issuer = TwoResolutionTargetEvidenceIssuerV1(
        projection=projection,
        planner=planner,
        outcome_source=outcome_source,
        runner_execution_identity=RUNNER,
        checkpoint_file_sha256=CHECKPOINT,
        camera_calibration_sha256=CALIBRATION,
        _synthetic_test_fixture=True,
    )
    target_memory = TwoResolutionReversibleTargetBeliefMemoryV1(
        evidence_issuer=evidence_issuer,
        _synthetic_development_fixture=True,
    )
    router = TwoResolutionDeterministicTargetRouterV2(
        projection=projection,
        planner=planner,
        target_memory=target_memory,
    )
    waypoint_issuer = ConfigurationPathWorldWaypointIssuerV2(projection, planner)
    integration = TwoResolutionDevelopmentNavigationIntegrationV1(
        projection=projection,
        planner=planner,
        physical_view_state_issuer=view_issuer,
        frontier_viewpoint_planner=frontier,
        target_evidence_issuer=evidence_issuer,
        target_memory=target_memory,
        target_router=router,
        world_waypoint_issuer=waypoint_issuer,
        _synthetic_development_fixture=True,
    )
    rows = (
        PhysicalCellProbabilityV1((60, 60), 0.4),
        PhysicalCellProbabilityV1((61, 60), 0.3),
        PhysicalCellProbabilityV1((62, 60), 0.2),
    )
    outcome = outcome_source.issue(
        snapshot=snapshot,
        outcome_kind="positive",
        target_id=G5_TARGET_ID,
        pose_timestamp_ns=100,
        runner_execution_identity=RUNNER,
        checkpoint_file_sha256=CHECKPOINT,
        raw_outcome_file_sha256=_hash("integration-raw-outcome"),
        camera_calibration_sha256=CALIBRATION,
        pose_provenance_sha256=_hash("integration-pose"),
        free_physical_cells=(row.cell for row in rows),
        unknown_physical_cells=(),
        target_physical_cells=((70, 80),),
        visible_physical_cells=(row.cell for row in rows),
        physical_probability=rows,
        unlocalized_probability=0.1,
        confidence=0.9,
    )
    return SimpleNamespace(
        manifest=manifest,
        memory=memory,
        projection=projection,
        snapshot=snapshot,
        planner=planner,
        component=component,
        outcome_source=outcome_source,
        outcome=outcome,
        target_memory=target_memory,
        router=router,
        waypoint_issuer=waypoint_issuer,
        integration=integration,
    )


def _issue(bundle: SimpleNamespace):
    return bundle.integration.issue_controller_claim_trace(
        snapshot=bundle.snapshot,
        component=bundle.component,
        outcome=bundle.outcome,
        start_configuration_cell=START,
        current_yaw_rad=-math.pi / 2.0,
        physical_manifest=bundle.manifest,
        trace_id="integration-trace",
        episode_id="integration-episode",
        event_id="claim-0",
        tick=25,
        event_index=0,
        robot_pose_world_xy_yaw=(16.75, -5.75, -math.pi / 2.0),
        task_object_id=TARGET_ID,
        pose_xy_variance_m2=0.0025,
        pose_yaw_variance_rad2=0.0004,
    )


def test_complete_chain_is_sealed_before_observer_import_and_claims() -> None:
    old_observer = sys.modules.pop(
        "lewm.benchmarks.go2_physical_claim_observer", None
    )
    old_evaluator = sys.modules.pop(
        "lewm.benchmarks.go2_physical_claim_evaluator", None
    )
    try:
        bundle = _bundle()
        artifact = _issue(bundle)
        assert "lewm.benchmarks.go2_physical_claim_observer" not in sys.modules
        assert "lewm.benchmarks.go2_physical_claim_evaluator" not in sys.modules
        bundle.integration.assert_controller_claim_trace(artifact)
        encoded = artifact.to_dict()

        assert encoded["g3"]["snapshot_sha256"] == bundle.snapshot.content_sha256
        assert encoded["g4"]["selected_frontier_candidate_sha256"]
        assert encoded["g5"]["target_id"] == G5_TARGET_ID
        assert encoded["task_object_id"] == TARGET_ID
        assert artifact.raw_claim_trace["evaluator_feedback_to_controller"] == []
        assert artifact.production_promotion_authorized is False
        assert artifact.hardware_execution_authorized is False
        assert len(bundle.router._consumed) == 1
        assert len(bundle.waypoint_issuer._consumed) == 1

        from lewm.benchmarks.go2_physical_claim_observer import (
            empty_evaluator_access_ledger,
        )

        observed = bundle.integration.evaluate_observer_only(
            artifact,
            evaluator_access_ledger=empty_evaluator_access_ledger(),
        )
        observed.assert_integrity()
        assert observed.evaluator_access_ledger == empty_evaluator_access_ledger()
        assert observed.evaluated_claim_trace["physical_claim_summary"][
            "all_targets_claimed"
        ] is True
        assert observed.to_dict()["controller_callback"] is None
        assert observed.production_promotion_authorized is False
        assert observed.hardware_execution_authorized is False
    finally:
        if old_observer is not None:
            sys.modules["lewm.benchmarks.go2_physical_claim_observer"] = old_observer
        if old_evaluator is not None:
            sys.modules["lewm.benchmarks.go2_physical_claim_evaluator"] = old_evaluator


def test_copied_stale_and_replayed_authorities_fail_closed() -> None:
    bundle = _bundle()
    artifact = _issue(bundle)
    with pytest.raises(TypeError, match="non-copyable"):
        copy.copy(artifact)
    copied = replace(artifact)
    assert copied.content_sha256 == artifact.content_sha256
    with pytest.raises(
        TwoResolutionNavigationIntegrationBindingError,
        match="exact live",
    ):
        bundle.integration.assert_controller_claim_trace(copied)

    from lewm.benchmarks.go2_physical_claim_observer import (
        empty_evaluator_access_ledger,
    )

    bundle.integration.evaluate_observer_only(
        artifact,
        evaluator_access_ledger=empty_evaluator_access_ledger(),
    )
    with pytest.raises(TwoResolutionNavigationIntegrationReplayError):
        bundle.integration.evaluate_observer_only(
            artifact,
            evaluator_access_ledger=empty_evaluator_access_ledger(),
        )

    stale = _bundle()
    stale.projection.project()
    with pytest.raises(SnapshotBindingError):
        _issue(stale)

    copied_outcome = _bundle()
    copied_outcome.outcome = copy.copy(copied_outcome.outcome)
    with pytest.raises(TwoResolutionTargetEvidenceBindingError, match="exact live"):
        _issue(copied_outcome)


def test_authority_mutation_and_evaluator_ledger_leakage_are_rejected() -> None:
    authority = _bundle()
    artifact = _issue(authority)
    object.__setattr__(artifact, "hardware_execution_authorized", True)
    with pytest.raises(
        TwoResolutionNavigationIntegrationBindingError,
        match="authority denial",
    ):
        authority.integration.assert_controller_claim_trace(artifact)

    feedback = _bundle()
    feedback_artifact = _issue(feedback)
    feedback_artifact.raw_claim_trace["evaluator_feedback_to_controller"].append(
        {"claimed": True}
    )
    with pytest.raises(
        TwoResolutionNavigationIntegrationBindingError,
        match="mutated",
    ):
        feedback.integration.evaluate_observer_only(
            feedback_artifact,
            evaluator_access_ledger={
                "evaluator_output_reads_by_controller": 0,
                "evaluator_callbacks_into_controller": 0,
                "evaluator_derived_termination_signals": 0,
            },
        )

    ledger_bundle = _bundle()
    ledger_artifact = _issue(ledger_bundle)
    for leaked in (
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
            "evaluator_output_reads_by_controller": 0,
            "evaluator_callbacks_into_controller": 0,
            "evaluator_derived_termination_signals": 0,
            "unreviewed_access": 0,
        },
        {
            "evaluator_output_reads_by_controller": False,
            "evaluator_callbacks_into_controller": 0,
            "evaluator_derived_termination_signals": 0,
        },
    ):
        with pytest.raises(
            TwoResolutionNavigationIntegrationBindingError,
            match="actually empty",
        ):
            ledger_bundle.integration.evaluate_observer_only(
                ledger_artifact,
                evaluator_access_ledger=leaked,
            )


def test_wrong_owner_chain_and_production_entrypoints_fail_closed() -> None:
    bundle = _bundle()
    foreign = _bundle()
    with pytest.raises(
        TwoResolutionNavigationIntegrationBindingError,
        match="owner chain",
    ):
        TwoResolutionDevelopmentNavigationIntegrationV1(
            projection=bundle.projection,
            planner=bundle.planner,
            physical_view_state_issuer=bundle.integration._view_issuer,
            frontier_viewpoint_planner=bundle.integration._frontier,
            target_evidence_issuer=bundle.integration._evidence_issuer,
            target_memory=bundle.target_memory,
            target_router=bundle.router,
            world_waypoint_issuer=foreign.waypoint_issuer,
            _synthetic_development_fixture=True,
        )
    with pytest.raises(PermissionError):
        TwoResolutionDevelopmentNavigationIntegrationV1(
            projection=bundle.projection,
            planner=bundle.planner,
            physical_view_state_issuer=bundle.integration._view_issuer,
            frontier_viewpoint_planner=bundle.integration._frontier,
            target_evidence_issuer=bundle.integration._evidence_issuer,
            target_memory=bundle.target_memory,
            target_router=bundle.router,
            world_waypoint_issuer=bundle.waypoint_issuer,
        )
    with pytest.raises(PermissionError):
        require_production_two_resolution_navigation_integration_v1()
