from __future__ import annotations

import copy

import numpy as np
import pytest

from lewm.benchmarks.go2_g3_exact_physical_equivalence_v2 import (
    _build_projected_snapshot,
)
from lewm.planning.revisioned_physical_configuration_memory import (
    PhysicalLabel,
    SnapshotBindingError,
)
from lewm.planning.two_resolution_configuration_projection_v2 import (
    TwoResolutionConfigurationPlannerV2,
    TwoResolutionConfigurationProjectionV2,
)
from lewm.planning.two_resolution_frontier_viewpoint_v2 import (
    configuration_center_in_physical_grid_v2,
)
from lewm.planning.two_resolution_world_waypoint_adapter_v1 import (
    ConfigurationPathWorldWaypointIssuerV1,
    ConfigurationPathWorldWaypointReceiptV1,
    WorldWaypointBindingError,
)
from lewm_worlds.manifest import (
    CameraValidityConstraints,
    SceneManifest,
    SpawnSpec,
)


def _manifest(scene_id: str = "world-waypoint-unit") -> SceneManifest:
    return SceneManifest(
        scene_id=scene_id,
        family="unit",
        difficulty_tier="unit",
        topology_seed=1,
        visual_seed=2,
        physics_seed=3,
        world_bounds_xy_m=((10.0, -12.0), (18.0, -4.0)),
        spawn=SpawnSpec(
            xyz_m=(12.5, -8.0, 0.35),
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


def _bundle():
    physical = np.full((88, 92), int(PhysicalLabel.FREE), dtype=np.uint8)
    memory, snapshot, planner = _build_projected_snapshot(
        _manifest(),
        physical,
        origin_xy_m=(12.37, -8.91),
        configuration_shape=(44, 46),
    )
    projection = getattr(planner, "_projection")
    assert type(projection) is TwoResolutionConfigurationProjectionV2
    path = planner.astar(snapshot, (30, 35), (32, 35))
    assert path is not None
    return memory, projection, snapshot, planner, path


@pytest.fixture(scope="module")
def shared_bundle():
    return _bundle()


def test_exact_high_index_world_centres_and_metric_receipt(shared_bundle) -> None:
    _memory, projection, snapshot, planner, path = shared_bundle
    issuer = ConfigurationPathWorldWaypointIssuerV1(projection, planner)
    receipt = issuer.issue(snapshot, path)

    assert [row.configuration_cell for row in receipt.waypoints] == [
        (30, 35),
        (31, 35),
        (32, 35),
    ]
    assert receipt.waypoints[0].world_xy_m == pytest.approx((15.42, -5.36))
    assert receipt.waypoints[-1].world_xy_m == pytest.approx((15.62, -5.36))
    assert receipt.path_cost_configuration_steps == 2.0
    assert receipt.path_cost_m == pytest.approx(0.20)
    assert receipt.start_configuration_cell == (30, 35)
    assert receipt.goal_configuration_cell == (32, 35)
    assert receipt.hardware_execution_authorized is False
    assert receipt.to_dict()["development_execution_eligible"] is True
    assert receipt.to_dict()["hardware_execution_authorized"] is False
    assert configuration_center_in_physical_grid_v2(
        snapshot,
        receipt.start_configuration_cell,
    ) == pytest.approx((61.0, 71.0))

    issuer.validate(snapshot, path, receipt)
    assert receipt.content_sha256 == receipt.to_dict()["content_sha256"]


def test_receipt_is_exact_live_single_use_and_noncopyable(shared_bundle) -> None:
    _memory, projection, snapshot, planner, path = shared_bundle
    issuer = ConfigurationPathWorldWaypointIssuerV1(projection, planner)
    receipt = issuer.issue(snapshot, path)

    with pytest.raises(TypeError, match="non-copyable"):
        copy.copy(receipt)
    with pytest.raises(TypeError, match="non-copyable"):
        copy.deepcopy(receipt)
    with pytest.raises(TypeError, match="non-copyable"):
        copy.copy(issuer)

    issuer.validate(snapshot, path, receipt, consume=True)
    with pytest.raises(WorldWaypointBindingError, match="already consumed"):
        issuer.validate(snapshot, path, receipt)


def test_reconstruction_forgery_and_mutation_reject(shared_bundle) -> None:
    _memory, projection, snapshot, planner, path = shared_bundle
    issuer = ConfigurationPathWorldWaypointIssuerV1(projection, planner)
    receipt = issuer.issue(snapshot, path)

    fields = {
        name: getattr(receipt, name)
        for name in ConfigurationPathWorldWaypointReceiptV1.__dataclass_fields__
        if name != "content_sha256"
    }
    forged = ConfigurationPathWorldWaypointReceiptV1(**fields)
    assert forged.content_sha256 == receipt.content_sha256
    with pytest.raises(WorldWaypointBindingError, match="exact live"):
        issuer.validate(snapshot, path, forged)

    object.__setattr__(receipt, "path_cost_m", 99.0)
    with pytest.raises(WorldWaypointBindingError, match="mutated"):
        issuer.validate(snapshot, path, receipt)


def test_wrong_projection_planner_pair_and_stale_snapshot_reject() -> None:
    memory, projection, snapshot, planner, path = _bundle()
    second_projection = TwoResolutionConfigurationProjectionV2(
        memory,
        configuration_map_frame=snapshot.configuration_map_frame,
        physical_shape=snapshot.physical_shape,
        configuration_shape=snapshot.configuration_shape,
    )
    second_planner = TwoResolutionConfigurationPlannerV2(second_projection)
    with pytest.raises(SnapshotBindingError, match="instances differ"):
        ConfigurationPathWorldWaypointIssuerV1(projection, second_planner)

    issuer = ConfigurationPathWorldWaypointIssuerV1(projection, planner)
    receipt = issuer.issue(snapshot, path)
    next_snapshot = projection.project()
    assert next_snapshot.configuration_revision == snapshot.configuration_revision + 1
    with pytest.raises(SnapshotBindingError):
        issuer.validate(snapshot, path, receipt)


def test_world_waypoint_receipt_rejects_wrong_ratio_or_cost(shared_bundle) -> None:
    _memory, projection, snapshot, planner, path = shared_bundle
    receipt = ConfigurationPathWorldWaypointIssuerV1(projection, planner).issue(
        snapshot,
        path,
    )
    fields = {
        name: getattr(receipt, name)
        for name in ConfigurationPathWorldWaypointReceiptV1.__dataclass_fields__
        if name != "content_sha256"
    }
    with pytest.raises(ValueError, match="2:1"):
        ConfigurationPathWorldWaypointReceiptV1(
            **{**fields, "physical_shape": (159, 160)}
        )
    with pytest.raises(ValueError, match="0.10 m"):
        ConfigurationPathWorldWaypointReceiptV1(
            **{**fields, "path_cost_m": receipt.path_cost_m + 0.05}
        )
