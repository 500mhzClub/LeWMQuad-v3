from __future__ import annotations

import copy
from dataclasses import replace
import hashlib

import numpy as np
import pytest

from lewm.benchmarks.go2_g3_exact_physical_equivalence_v2 import (
    _build_projected_snapshot,
)
from lewm.planning.revisioned_physical_configuration_memory import (
    PhysicalLabel,
    StaleSnapshotError,
)
from lewm.planning.two_resolution_reversible_target_belief_v1 import (
    TwoResolutionReversibleTargetBeliefMemoryV1,
)
from lewm.planning.two_resolution_target_evidence_v1 import (
    PhysicalCellProbabilityV1,
    SyntheticV5TargetOutcomeIssuerV1,
    TwoResolutionTargetEvidenceIssuerV1,
    V5RunnerExecutionIdentityV1,
)
from lewm.planning.two_resolution_target_router_v1 import (
    NoSafeTargetRouteError,
    TwoResolutionDeterministicTargetRouterV1,
    TwoResolutionTargetRouteBindingError,
    require_production_two_resolution_target_router,
)
from lewm_worlds.manifest import CameraValidityConstraints, SceneManifest, SpawnSpec


def _hash(label: str) -> str:
    return hashlib.sha256(label.encode("ascii")).hexdigest()


RUNNER = V5RunnerExecutionIdentityV1(
    entrypoint_wrapper_file_sha256=_hash("router-wrapper"),
    captured_launcher_file_sha256=_hash("router-launcher"),
    captured_core_file_sha256=_hash("router-core"),
)
CHECKPOINT = _hash("router-checkpoint")
CALIBRATION = _hash("router-calibration")


def _manifest() -> SceneManifest:
    return SceneManifest(
        scene_id="two-resolution-router",
        family="unit",
        difficulty_tier="unit",
        topology_seed=1,
        visual_seed=2,
        physics_seed=3,
        world_bounds_xy_m=((-1.6, -1.6), (1.6, 1.6)),
        spawn=SpawnSpec(
            xyz_m=(0.0, 0.0, 0.35),
            quat_wxyz=(1.0, 0.0, 0.0, 0.0),
        ),
        graph_nodes=(),
        graph_edges=(),
        obstacles=(),
        landmarks=(),
        camera_constraints=CameraValidityConstraints(
            min_wall_thickness_m=0.08,
            near_m=0.05,
            far_m=200.0,
            min_camera_clearance_m=0.10,
        ),
        split="train",
    )


def _stack(*, rows: tuple[PhysicalCellProbabilityV1, ...] | None = None):
    labels = np.full((64, 64), int(PhysicalLabel.FREE), dtype=np.uint8)
    memory_g3, snapshot, planner = _build_projected_snapshot(
        _manifest(),
        labels,
        origin_xy_m=(-1.6, -1.6),
        configuration_shape=(32, 32),
    )
    del memory_g3
    projection = getattr(planner, "_projection")
    component = planner.connected_component(snapshot, (5, 5))
    source = SyntheticV5TargetOutcomeIssuerV1(_synthetic_test_fixture=True)
    evidence_issuer = TwoResolutionTargetEvidenceIssuerV1(
        projection=projection,
        planner=planner,
        outcome_source=source,
        runner_execution_identity=RUNNER,
        checkpoint_file_sha256=CHECKPOINT,
        camera_calibration_sha256=CALIBRATION,
        _synthetic_test_fixture=True,
    )
    target_memory = TwoResolutionReversibleTargetBeliefMemoryV1(
        evidence_issuer=evidence_issuer,
        _synthetic_development_fixture=True,
    )
    if rows is None:
        rows = (
            PhysicalCellProbabilityV1((30, 30), 0.4),
            PhysicalCellProbabilityV1((31, 30), 0.3),
            PhysicalCellProbabilityV1((32, 30), 0.2),
        )
    visible = frozenset(row.cell for row in rows)
    outcome = source.issue(
        snapshot=snapshot,
        outcome_kind="positive",
        target_id="blue",
        pose_timestamp_ns=18,
        runner_execution_identity=RUNNER,
        checkpoint_file_sha256=CHECKPOINT,
        raw_outcome_file_sha256=_hash("router-raw-1"),
        camera_calibration_sha256=CALIBRATION,
        pose_provenance_sha256=_hash("router-pose-1"),
        free_physical_cells=visible,
        unknown_physical_cells=frozenset(),
        target_physical_cells=frozenset({(40, 40)}),
        visible_physical_cells=visible,
        physical_probability=rows,
        unlocalized_probability=0.1,
        confidence=0.85,
    )
    context = evidence_issuer.issue_context(snapshot, component, outcome)
    evidence = evidence_issuer.open_writer(context).issue_positive()
    posterior = target_memory.apply(context, evidence)
    router = TwoResolutionDeterministicTargetRouterV1(
        projection=projection,
        planner=planner,
        target_memory=target_memory,
    )
    return (
        projection,
        snapshot,
        planner,
        component,
        source,
        evidence_issuer,
        target_memory,
        posterior,
        router,
    )


def test_router_retains_safe_free_path_faces_mode_and_excludes_target_cells() -> None:
    _, snapshot, planner, component, _, _, _, posterior, router = _stack()
    plan = router.issue(
        snapshot=snapshot,
        component=component,
        posterior=posterior,
        start_cell=(5, 5),
        start_yaw_rad=0.0,
    )
    router.validate(
        snapshot=snapshot,
        component=component,
        posterior=posterior,
        plan=plan,
    )
    planner.validate_path(snapshot, plan.path)
    receipt = plan.receipt
    assert receipt.goal_cell not in receipt.selected_hypothesis_cells
    assert receipt.goal_cell not in posterior.excluded_target_configuration_cells
    assert not set(receipt.path_cells) & set(receipt.selected_hypothesis_cells)
    assert not set(receipt.path_cells) & set(
        posterior.excluded_target_configuration_cells
    )
    assert all(snapshot.state(cell) is PhysicalLabel.FREE for cell in receipt.path_cells)
    assert 0.10 - 1e-12 <= receipt.target_distance_m <= 1.20 + 1e-12
    goal_xy = snapshot.configuration_map_frame.cell_center(receipt.goal_cell)
    expected_yaw = pytest.approx(
        np.arctan2(
            receipt.target_world_xy_m[1] - goal_xy[1],
            receipt.target_world_xy_m[0] - goal_xy[0],
        )
    )
    assert receipt.terminal_yaw_rad == expected_yaw
    assert receipt.path_cost_m == pytest.approx(
        0.10 * receipt.path_cost_configuration_steps
    )
    assert receipt.production_promotion_authorized is False
    assert receipt.hardware_execution_authorized is False


def test_router_is_deterministic_and_high_indices_are_configuration_not_physical() -> None:
    rows = (
        PhysicalCellProbabilityV1((50, 50), 0.4),
        PhysicalCellProbabilityV1((51, 50), 0.3),
        PhysicalCellProbabilityV1((50, 51), 0.2),
    )
    _, snapshot, _, component, _, _, _, posterior, router = _stack(rows=rows)
    first = router.issue(
        snapshot=snapshot,
        component=component,
        posterior=posterior,
        start_cell=(5, 5),
        start_yaw_rad=-0.3,
    )
    second = router.issue(
        snapshot=snapshot,
        component=component,
        posterior=posterior,
        start_cell=(5, 5),
        start_yaw_rad=-0.3,
    )
    assert first.receipt.selected_hypothesis_peak_cell == (25, 25)
    assert first.receipt.selected_hypothesis_peak_cell != (50, 50)
    assert first.receipt.to_dict() == second.receipt.to_dict()


def test_route_plan_is_exact_instance_single_use_and_snapshot_stale_safe() -> None:
    _, snapshot, _, component, source, issuer, memory, posterior, router = _stack()
    plan = router.issue(
        snapshot=snapshot,
        component=component,
        posterior=posterior,
        start_cell=(5, 5),
        start_yaw_rad=0.0,
    )
    with pytest.raises(TypeError):
        copy.copy(plan.receipt)
    with pytest.raises(TwoResolutionTargetRouteBindingError):
        router.validate(
            snapshot=snapshot,
            component=component,
            posterior=posterior,
            plan=replace(plan),
        )
    router.validate(
        snapshot=snapshot,
        component=component,
        posterior=posterior,
        plan=plan,
        consume=True,
    )
    with pytest.raises(TwoResolutionTargetRouteBindingError):
        router.validate(
            snapshot=snapshot,
            component=component,
            posterior=posterior,
            plan=plan,
        )

    negative_rows = (
        PhysicalCellProbabilityV1((30, 30), 0.9),
        PhysicalCellProbabilityV1((30, 31), 0.8),
        PhysicalCellProbabilityV1((31, 30), 0.7),
        PhysicalCellProbabilityV1((31, 31), 0.6),
    )
    visible = frozenset(row.cell for row in negative_rows)
    outcome = source.issue(
        snapshot=snapshot,
        outcome_kind="negative",
        target_id="blue",
        pose_timestamp_ns=19,
        runner_execution_identity=RUNNER,
        checkpoint_file_sha256=CHECKPOINT,
        raw_outcome_file_sha256=_hash("router-raw-2"),
        camera_calibration_sha256=CALIBRATION,
        pose_provenance_sha256=_hash("router-pose-2"),
        free_physical_cells=visible,
        unknown_physical_cells=frozenset(),
        target_physical_cells=frozenset({(40, 40)}),
        visible_physical_cells=visible,
        physical_probability=negative_rows,
        confidence=0.85,
    )
    context = issuer.issue_context(snapshot, component, outcome)
    evidence = issuer.open_writer(context).issue_negative()
    memory.apply(context, evidence)
    with pytest.raises(StaleSnapshotError):
        router.validate(
            snapshot=snapshot,
            component=component,
            posterior=posterior,
            plan=plan,
        )


def test_router_rejects_hypothesis_start_and_has_no_production_authority() -> None:
    _, snapshot, _, component, _, _, _, posterior, router = _stack()
    with pytest.raises(NoSafeTargetRouteError):
        router.issue(
            snapshot=snapshot,
            component=component,
            posterior=posterior,
            start_cell=(15, 15),
            start_yaw_rad=0.0,
        )
    with pytest.raises(PermissionError):
        require_production_two_resolution_target_router()

