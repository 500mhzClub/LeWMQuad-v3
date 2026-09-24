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
from lewm.planning import two_resolution_target_router_v1 as router_v1_module
from lewm.planning import two_resolution_target_router_v2 as router_v2_module
from lewm.planning.two_resolution_target_router_v1 import (
    TwoResolutionTargetRouteBindingError,
)
from lewm.planning.two_resolution_target_router_v2 import (
    TwoResolutionDeterministicTargetRouterV2,
    require_production_two_resolution_target_router_v2,
)
from lewm.planning.two_resolution_world_waypoint_adapter_v2 import (
    ConfigurationPathWorldWaypointIssuerV2,
)
from lewm_worlds.manifest import CameraValidityConstraints, SceneManifest, SpawnSpec


def _hash(label: str) -> str:
    return hashlib.sha256(label.encode("ascii")).hexdigest()


RUNNER = V5RunnerExecutionIdentityV1(
    entrypoint_wrapper_file_sha256=_hash("router-v2-wrapper"),
    captured_launcher_file_sha256=_hash("router-v2-launcher"),
    captured_core_file_sha256=_hash("router-v2-core"),
)
CHECKPOINT = _hash("router-v2-checkpoint")
CALIBRATION = _hash("router-v2-calibration")


def _manifest() -> SceneManifest:
    return SceneManifest(
        scene_id="two-resolution-router-v2",
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


def _stack(rows: tuple[PhysicalCellProbabilityV1, ...]):
    labels = np.full((64, 64), int(PhysicalLabel.FREE), dtype=np.uint8)
    _, snapshot, planner = _build_projected_snapshot(
        _manifest(),
        labels,
        origin_xy_m=(-1.6, -1.6),
        configuration_shape=(32, 32),
    )
    projection = getattr(planner, "_projection")
    component = planner.connected_component(snapshot, (5, 5))
    source = SyntheticV5TargetOutcomeIssuerV1(_synthetic_test_fixture=True)
    issuer = TwoResolutionTargetEvidenceIssuerV1(
        projection=projection,
        planner=planner,
        outcome_source=source,
        runner_execution_identity=RUNNER,
        checkpoint_file_sha256=CHECKPOINT,
        camera_calibration_sha256=CALIBRATION,
        _synthetic_test_fixture=True,
    )
    memory = TwoResolutionReversibleTargetBeliefMemoryV1(
        evidence_issuer=issuer,
        _synthetic_development_fixture=True,
    )
    visible = frozenset(row.cell for row in rows)
    outcome = source.issue(
        snapshot=snapshot,
        outcome_kind="positive",
        target_id="blue",
        pose_timestamp_ns=18,
        runner_execution_identity=RUNNER,
        checkpoint_file_sha256=CHECKPOINT,
        raw_outcome_file_sha256=_hash("router-v2-raw"),
        camera_calibration_sha256=CALIBRATION,
        pose_provenance_sha256=_hash("router-v2-pose"),
        free_physical_cells=visible,
        unknown_physical_cells=frozenset(),
        target_physical_cells=frozenset({(40, 40)}),
        visible_physical_cells=visible,
        physical_probability=rows,
        unlocalized_probability=0.1,
        confidence=0.85,
    )
    context = issuer.issue_context(snapshot, component, outcome)
    evidence = issuer.open_writer(context).issue_positive()
    posterior = memory.apply(context, evidence)
    router = TwoResolutionDeterministicTargetRouterV2(
        projection=projection,
        planner=planner,
        target_memory=memory,
    )
    return projection, snapshot, planner, component, memory, posterior, router


def _separated_rows() -> tuple[PhysicalCellProbabilityV1, ...]:
    return (
        PhysicalCellProbabilityV1((50, 10), 0.4),
        PhysicalCellProbabilityV1((51, 10), 0.2),
        PhysicalCellProbabilityV1((20, 10), 0.3),
    )


def test_v2_exact_counterexample_avoids_unselected_posterior_mode() -> None:
    _, snapshot, planner, component, memory, posterior, router = _stack(
        _separated_rows()
    )
    hypotheses = memory.hypotheses(posterior)
    assert [row.peak_cell for row in hypotheses] == [(25, 5), (10, 5)]
    plan = router.issue(
        snapshot=snapshot,
        component=component,
        posterior=posterior,
        start_cell=(5, 5),
        start_yaw_rad=0.0,
    )
    all_cells = {cell for hypothesis in hypotheses for cell in hypothesis.cells}
    assert plan.all_hypothesis_cells == all_cells
    assert not set(plan.path.cells) & all_cells
    assert (10, 5) not in plan.path.cells
    assert (25, 5) not in plan.path.cells
    planner.validate_path(snapshot, plan.path)
    router.validate(
        snapshot=snapshot,
        component=component,
        posterior=posterior,
        plan=plan,
    )


@pytest.mark.parametrize(
    "object_name,field_name",
    (
        ("plan", "production_promotion_authorized"),
        ("plan", "hardware_execution_authorized"),
        ("receipt", "production_promotion_authorized"),
        ("receipt", "hardware_execution_authorized"),
    ),
)
def test_v2_rejects_rehashed_authority_tampering(
    object_name: str,
    field_name: str,
) -> None:
    rows = (
        PhysicalCellProbabilityV1((30, 30), 0.4),
        PhysicalCellProbabilityV1((31, 30), 0.3),
        PhysicalCellProbabilityV1((32, 30), 0.2),
    )
    _, snapshot, _, component, _, posterior, router = _stack(rows)
    plan = router.issue(
        snapshot=snapshot,
        component=component,
        posterior=posterior,
        start_cell=(5, 5),
        start_yaw_rad=0.0,
    )
    target = plan if object_name == "plan" else plan.receipt
    object.__setattr__(target, field_name, True)
    if object_name == "receipt":
        object.__setattr__(
            plan.receipt,
            "content_sha256",
            router_v1_module._sha256(plan.receipt.to_dict(False)),
        )
    object.__setattr__(
        plan,
        "content_sha256",
        router_v2_module._sha256(plan.to_dict(False)),
    )
    with pytest.raises(TwoResolutionTargetRouteBindingError):
        router.validate(
            snapshot=snapshot,
            component=component,
            posterior=posterior,
            plan=plan,
        )


def test_v2_stores_original_issued_content_and_rejects_rehashed_semantic_change() -> None:
    rows = (
        PhysicalCellProbabilityV1((30, 30), 0.4),
        PhysicalCellProbabilityV1((31, 30), 0.3),
        PhysicalCellProbabilityV1((32, 30), 0.2),
    )
    _, snapshot, _, component, _, posterior, router = _stack(rows)
    plan = router.issue(
        snapshot=snapshot,
        component=component,
        posterior=posterior,
        start_cell=(5, 5),
        start_yaw_rad=0.0,
    )
    original = plan.content_sha256
    object.__setattr__(plan.receipt, "initial_heading_error_rad", 0.0)
    object.__setattr__(
        plan.receipt,
        "content_sha256",
        router_v1_module._sha256(plan.receipt.to_dict(False)),
    )
    object.__setattr__(
        plan,
        "content_sha256",
        router_v2_module._sha256(plan.to_dict(False)),
    )
    assert plan.content_sha256 != original
    with pytest.raises(TwoResolutionTargetRouteBindingError):
        router.validate(
            snapshot=snapshot,
            component=component,
            posterior=posterior,
            plan=plan,
        )


def test_v2_exact_single_use_and_world_waypoint_v2_composition() -> None:
    rows = (
        PhysicalCellProbabilityV1((30, 30), 0.4),
        PhysicalCellProbabilityV1((31, 30), 0.3),
        PhysicalCellProbabilityV1((32, 30), 0.2),
    )
    projection, snapshot, planner, component, _, posterior, router = _stack(rows)
    plan = router.issue(
        snapshot=snapshot,
        component=component,
        posterior=posterior,
        start_cell=(5, 5),
        start_yaw_rad=0.0,
    )
    with pytest.raises(TypeError):
        copy.copy(plan)
    with pytest.raises(TwoResolutionTargetRouteBindingError):
        router.validate(
            snapshot=snapshot,
            component=component,
            posterior=posterior,
            plan=replace(plan),
        )
    waypoint_issuer = ConfigurationPathWorldWaypointIssuerV2(projection, planner)
    waypoint = waypoint_issuer.issue(snapshot, plan.path)
    assert waypoint.production_promotion_authorized is False
    assert waypoint.hardware_execution_authorized is False
    assert waypoint.goal_configuration_cell == plan.receipt.goal_cell
    waypoint_issuer.validate(snapshot, plan.path, waypoint, consume=True)
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


def test_v2_production_authority_remains_unset() -> None:
    with pytest.raises(PermissionError):
        require_production_two_resolution_target_router_v2()

