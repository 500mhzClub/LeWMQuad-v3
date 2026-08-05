from __future__ import annotations

import copy
from dataclasses import replace
from fractions import Fraction
import hashlib
import json

import numpy as np
import pytest

from lewm.benchmarks.go2_g3_exact_physical_equivalence_v2 import (
    _build_projected_snapshot,
    independent_cross_grid_configuration_labels,
)
from lewm.planning import revisioned_physical_configuration_memory as memory_module
from lewm.planning.revisioned_physical_configuration_memory import (
    EvidenceAuthority,
    ExecutionBlock,
    ExecutionBlockKind,
    ExecutionEvidenceAdmission,
    ExecutionEvidenceKind,
    MapFrameIdentity,
    ObservationIdentity,
    PhysicalCellEvidence,
    PhysicalEvidenceTransaction,
    PhysicalLabel,
    PoseProvenance,
    PoseSource,
    RevisionedPhysicalMemory,
    SnapshotBindingError,
    StalePathError,
    StaleSnapshotError,
)
from lewm.planning.two_resolution_configuration_projection_v2 import (
    ConfigurationFrontiersV2,
    ConfigurationPathV2,
    FIXED_PROFILE_V2,
    FREE_SUPPORT_COUNT,
    FREE_SUPPORT_SHA256,
    OCCUPIED_SUPPORT_COUNT,
    OCCUPIED_SUPPORT_SHA256,
    PROFILE_SHA256,
    TwoResolutionConfigurationPlannerV2,
    TwoResolutionConfigurationProjectionV2,
    TwoResolutionConfigurationSnapshotV2,
    assert_fixed_profile_integrity,
    physical_index_for_configuration_offset,
)
from lewm.planning.zero_inflation_exact_physical_adapter_v1 import (
    ZeroInflationExactPhysicalAdapterV1,
    exact_physical_cells_content_sha256,
)
from lewm_worlds.manifest import CameraValidityConstraints, SceneManifest, SpawnSpec


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
    ).hexdigest()


def _manifest(scene_id: str = "g3-v2-projection-unit") -> SceneManifest:
    return SceneManifest(
        scene_id=scene_id,
        family="unit",
        difficulty_tier="unit",
        topology_seed=1,
        visual_seed=2,
        physics_seed=3,
        world_bounds_xy_m=((-2.0, -2.0), (2.0, 2.0)),
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


def _exact_rational_supports() -> tuple[tuple[tuple[int, int], ...], tuple[tuple[int, int], ...]]:
    radius_squared = Fraction(47, 100) ** 2
    half_physical = Fraction(1, 40)
    free: list[tuple[int, int]] = []
    occupied: list[tuple[int, int]] = []
    for dx in range(-12, 13):
        for dy in range(-12, 13):
            delta_x = Fraction(2 * dx - 1, 40)
            delta_y = Fraction(2 * dy - 1, 40)
            near_x = max(abs(delta_x) - half_physical, Fraction(0))
            near_y = max(abs(delta_y) - half_physical, Fraction(0))
            if near_x**2 + near_y**2 <= radius_squared:
                free.append((dx, dy))
            if delta_x**2 + delta_y**2 <= radius_squared:
                occupied.append((dx, dy))
    return tuple(sorted(free)), tuple(sorted(occupied))


def _kernel_core(schema: str, offsets: tuple[tuple[int, int], ...]) -> dict[str, object]:
    return {
        "configuration_cell_size_m": 0.1,
        "footprint_radius_m": 0.47,
        "inclusive_boundary": True,
        "offsets": [[x, y] for x, y in offsets],
        "physical_cell_size_m": 0.05,
        "physical_index_rule": "(2*cx+dx,2*cy+dy)",
        "schema": schema,
        "shared_origin_cell_boundaries": True,
    }


def _projection(memory, snapshot) -> TwoResolutionConfigurationProjectionV2:
    return TwoResolutionConfigurationProjectionV2(
        memory,
        configuration_map_frame=snapshot.configuration_map_frame,
        physical_shape=snapshot.physical_shape,
        configuration_shape=snapshot.configuration_shape,
    )


def _physical() -> np.ndarray:
    labels = np.full((60, 64), int(PhysicalLabel.FREE), dtype=np.uint8)
    labels[30, 32] = int(PhysicalLabel.OCCUPIED)
    labels[10:13, 50:53] = int(PhysicalLabel.UNKNOWN)
    return labels


def _learned_transaction(
    memory: RevisionedPhysicalMemory,
    observation_id: str,
    *,
    retractions: tuple[str, ...] = (),
) -> PhysicalEvidenceTransaction:
    camera = memory.bound_camera_transform_sha256
    assert camera is not None
    return PhysicalEvidenceTransaction(
        observation=ObservationIdentity(
            observation_id=observation_id,
            payload_sha256=hashlib.sha256(f"payload:{observation_id}".encode()).hexdigest(),
            producer_sha256=hashlib.sha256(b"learned-test-producer").hexdigest(),
            authority=EvidenceAuthority.LEARNED_PHYSICAL,
        ),
        map_frame=memory.map_frame,
        pose=PoseProvenance(
            source=PoseSource.DEPLOYMENT_ODOMETRY,
            frame_id=memory.map_frame.frame_id,
            mean_xy_yaw=(0.0, 0.0, 0.0),
            covariance_xy_yaw=((0.0, 0.0, 0.0),) * 3,
            timestamp_ns=memory.revision + 1,
            synchronization_id=f"sync:{observation_id}",
            camera_transform_sha256=camera,
        ),
        physical_evidence=(
            ()
            if retractions
            else (
                PhysicalCellEvidence(
                    cell=(100, 100),
                    label=PhysicalLabel.OCCUPIED,
                ),
            )
        ),
        retract_learned_observation_ids=retractions,
        projection_contract_sha256=PROFILE_SHA256,
    )


def _contact_transaction(
    memory: RevisionedPhysicalMemory,
    *,
    physical_cell: tuple[int, int],
) -> PhysicalEvidenceTransaction:
    receipt = hashlib.sha256(b"contact-receipt").hexdigest()
    camera = memory.bound_camera_transform_sha256
    assert camera is not None
    observation = ObservationIdentity(
        observation_id="admitted-contact-v2",
        payload_sha256=receipt,
        producer_sha256=hashlib.sha256(b"development-contact-runner").hexdigest(),
        authority=EvidenceAuthority.EXECUTOR_OUTCOME,
    )
    pose = PoseProvenance(
        source=PoseSource.DEPLOYMENT_ODOMETRY,
        frame_id=memory.map_frame.frame_id,
        mean_xy_yaw=(*memory.map_frame.cell_center(physical_cell), 0.0),
        covariance_xy_yaw=((0.0, 0.0, 0.0),) * 3,
        timestamp_ns=memory.revision + 1,
        synchronization_id="admitted-contact-v2",
        camera_transform_sha256=camera,
    )
    block = ExecutionBlock(
        block_id="contact-v2",
        body_center_xy_m=memory.map_frame.cell_center(physical_cell),
        kind=ExecutionBlockKind.CONTACT,
        outcome_sha256=receipt,
    )
    evidence_hash = memory_module._execution_evidence_content_sha256((), (block,))
    admission = ExecutionEvidenceAdmission(
        admission_id_sha256=hashlib.sha256(b"contact-admission").hexdigest(),
        adapter_instance_sha256=hashlib.sha256(b"development-adapter").hexdigest(),
        source_memory_instance_sha256=hashlib.sha256(b"development-memory").hexdigest(),
        receipt_content_sha256=receipt,
        adapter_contract_sha256=hashlib.sha256(b"development-contract").hexdigest(),
        body_support_contract_sha256=hashlib.sha256(b"development-body").hexdigest(),
        map_frame_sha256=memory.map_frame.content_sha256,
        observation_sha256=_canonical_sha256(observation.to_dict()),
        pose_sha256=pose.content_sha256,
        evidence_content_sha256=evidence_hash,
        memory_revision_before=memory.revision,
        evidence_kind=ExecutionEvidenceKind.CONTACT,
    )
    return PhysicalEvidenceTransaction(
        observation=observation,
        map_frame=memory.map_frame,
        pose=pose,
        execution_blocks=(block,),
        execution_admission=admission,
        projection_contract_sha256=PROFILE_SHA256,
    )


def test_preregistered_exact_rational_support_records_and_alignment() -> None:
    free, occupied = _exact_rational_supports()
    free_core = _kernel_core(
        "lewm_g3_v2_cross_grid_free_closed_square_intersection_kernel_v1",
        free,
    )
    occupied_core = _kernel_core(
        "lewm_g3_v2_cross_grid_occupied_center_inside_disc_kernel_v1",
        occupied,
    )
    projection_core = {
        "configuration_cell_size_m": 0.1,
        "footprint_radius_m": 0.47,
        "free_support_count": 316,
        "free_support_sha256": _canonical_sha256(free_core),
        "occupied_precedes_free": True,
        "occupied_support_count": 276,
        "occupied_support_sha256": _canonical_sha256(occupied_core),
        "otherwise": "unknown",
        "out_of_domain_support": "occupied",
        "physical_cell_size_m": 0.05,
        "physical_shape_per_configuration_cell": [2, 2],
        "schema": "lewm_g3_v2_two_resolution_configuration_projection_v1",
        "shared_origin_cell_boundaries": True,
    }
    assert len(free) == FREE_SUPPORT_COUNT == 316
    assert len(occupied) == OCCUPIED_SUPPORT_COUNT == 276
    assert _canonical_sha256(free_core) == FREE_SUPPORT_SHA256
    assert _canonical_sha256(occupied_core) == OCCUPIED_SUPPORT_SHA256
    assert _canonical_sha256(projection_core) == PROFILE_SHA256
    assert free == FIXED_PROFILE_V2.free_support_offsets
    assert occupied == FIXED_PROFILE_V2.occupied_support_offsets
    assert set(occupied) < set(free)
    assert (min(x for x, _ in free), max(x for x, _ in free)) == (-9, 10)
    assert (min(x for x, _ in occupied), max(x for x, _ in occupied)) == (-8, 9)
    for offsets in (free, occupied):
        support = set(offsets)
        assert {(1 - x, y) for x, y in support} == support
        assert {(x, 1 - y) for x, y in support} == support
    assert (10, 3) in free and (10, 4) not in free
    assert (9, 4) in occupied and (9, 5) not in occupied
    assert (10, 0) in free and (10, 0) not in occupied
    assert_fixed_profile_integrity()


def test_shared_boundary_origin_transform_is_exact_and_translation_invariant() -> None:
    configuration = (3, 4)
    for origin in ((0.0, 0.0), (12.37, -8.91)):
        for offset in ((0, 0), (1, 1), (-2, 5)):
            physical = physical_index_for_configuration_offset(configuration, offset)
            assert physical == (
                2 * configuration[0] + offset[0],
                2 * configuration[1] + offset[1],
            )
            configuration_center = (
                origin[0] + configuration[0] * 0.10 + 0.05,
                origin[1] + configuration[1] * 0.10 + 0.05,
            )
            physical_center = (
                origin[0] + physical[0] * 0.05 + 0.025,
                origin[1] + physical[1] * 0.05 + 0.025,
            )
            assert physical_center[0] - configuration_center[0] == pytest.approx(
                (offset[0] - 0.5) * 0.05
            )
            assert physical_center[1] - configuration_center[1] == pytest.approx(
                (offset[1] - 0.5) * 0.05
            )


def test_projection_matches_independent_oracle_and_binds_complete_rasters() -> None:
    physical = _physical()
    memory, snapshot, planner = _build_projected_snapshot(
        _manifest(),
        physical,
        origin_xy_m=(-1.5, -1.6),
        configuration_shape=(30, 32),
    )
    expected = independent_cross_grid_configuration_labels(
        physical,
        configuration_shape=(30, 32),
    )
    actual = np.full((30, 32), int(PhysicalLabel.UNKNOWN), dtype=np.uint8)
    for cell in snapshot.free_cells:
        actual[cell] = int(PhysicalLabel.FREE)
    for cell in snapshot.occupied_cells:
        actual[cell] = int(PhysicalLabel.OCCUPIED)
    np.testing.assert_array_equal(actual, expected)
    assert len(snapshot.evaluated_cells) == 30 * 32
    assert snapshot.physical_shape == physical.shape
    assert snapshot.physical_map_frame_sha256 != snapshot.configuration_map_frame_sha256
    assert snapshot.physical_revision == memory.revision == 1
    assert snapshot.configuration_revision == 1
    assert snapshot.production_promotion_authorized is False
    assert snapshot.profile_sha256 == PROFILE_SHA256

    component = planner.connected_component(snapshot, min(snapshot.free_cells))
    goal = max(component.cells)
    path = planner.astar(snapshot, component.start_cell, goal)
    assert path is not None
    planner.validate_path(snapshot, path)


@pytest.mark.parametrize(
    ("origin_delta", "cell_size", "frame_id", "physical_shape"),
    [
        ((0.025, 0.025), 0.10, "g3_v2_configuration_planning", (60, 64)),
        ((0.0, 0.0), 0.05, "g3_v2_configuration_planning", (60, 64)),
        ((0.0, 0.0), 0.10, "g3_v2_physical_evidence", (60, 64)),
        ((0.0, 0.0), 0.10, "g3_v2_configuration_planning", (60, 63)),
    ],
)
def test_projection_rejects_centre_alignment_same_grid_wrong_frame_or_ratio(
    origin_delta: tuple[float, float],
    cell_size: float,
    frame_id: str,
    physical_shape: tuple[int, int],
) -> None:
    memory, snapshot, _planner = _build_projected_snapshot(
        _manifest(),
        _physical(),
        origin_xy_m=(-1.5, -1.6),
        configuration_shape=(30, 32),
    )
    origin = memory.map_frame.origin_xy_m
    configuration_frame = MapFrameIdentity(
        session_id="adversarial-configuration",
        origin_xy_m=(origin[0] + origin_delta[0], origin[1] + origin_delta[1]),
        cell_size_m=cell_size,
        frame_id=frame_id,
    )
    with pytest.raises(ValueError, match="alignment|2x"):
        TwoResolutionConfigurationProjectionV2(
            memory,
            configuration_map_frame=configuration_frame,
            physical_shape=physical_shape,
            configuration_shape=snapshot.configuration_shape,
        )


def test_snapshot_rejects_bad_types_hashes_revisions_origins_and_incomplete_domain() -> None:
    _memory, snapshot, _planner = _build_projected_snapshot(
        _manifest(),
        _physical(),
        origin_xy_m=(-1.5, -1.6),
        configuration_shape=(30, 32),
    )
    values = {
        name: getattr(snapshot, name)
        for name in TwoResolutionConfigurationSnapshotV2.__dataclass_fields__
        if name != "content_sha256"
    }
    for name, value, error in (
        ("physical_revision", -7, ValueError),
        ("configuration_revision", True, TypeError),
        ("physical_content_sha256", "not-a-hash", ValueError),
        ("exact_sim_tainted", 1, TypeError),
        ("configuration_shape", (30, 31), ValueError),
        (
            "unknown_cells",
            frozenset(set(snapshot.unknown_cells) - {next(iter(snapshot.unknown_cells))}),
            ValueError,
        ),
    ):
        forged = {**values, name: value}
        with pytest.raises(error):
            TwoResolutionConfigurationSnapshotV2(**forged)
    with pytest.raises(ValueError, match="finite"):
        MapFrameIdentity(
            session_id="bad",
            origin_xy_m=(float("nan"), float("inf")),
            cell_size_m=0.10,
            frame_id="bad",
        )


def test_planner_rejects_forged_unissued_and_stale_snapshots() -> None:
    memory, snapshot, planner = _build_projected_snapshot(
        _manifest(),
        _physical(),
        origin_xy_m=(-1.5, -1.6),
        configuration_shape=(30, 32),
    )
    moved = next(iter(snapshot.unknown_cells))
    forged = replace(
        snapshot,
        free_cells=frozenset(set(snapshot.free_cells) | {moved}),
        unknown_cells=frozenset(set(snapshot.unknown_cells) - {moved}),
    )
    with pytest.raises(SnapshotBindingError, match="exact live"):
        planner.connected_component(forged, moved)

    memory.apply_transaction(_learned_transaction(memory, "learned-v2"))
    with pytest.raises(StaleSnapshotError, match="physical revision"):
        planner.connected_component(snapshot, min(snapshot.free_cells))


def test_support_mutation_is_rejected_but_independent_oracle_is_unchanged() -> None:
    physical = _physical()
    _memory, snapshot, planner = _build_projected_snapshot(
        _manifest(),
        physical,
        origin_xy_m=(-1.5, -1.6),
        configuration_shape=(30, 32),
    )
    baseline = independent_cross_grid_configuration_labels(
        physical,
        configuration_shape=(30, 32),
    )
    original = FIXED_PROFILE_V2.free_support_offsets
    object.__setattr__(
        FIXED_PROFILE_V2,
        "free_support_offsets",
        tuple(cell for cell in original if cell != (10, 3)),
    )
    try:
        np.testing.assert_array_equal(
            independent_cross_grid_configuration_labels(
                physical,
                configuration_shape=(30, 32),
            ),
            baseline,
        )
        with pytest.raises(SnapshotBindingError, match="mutated"):
            planner.connected_component(snapshot, min(snapshot.free_cells))
    finally:
        object.__setattr__(FIXED_PROFILE_V2, "free_support_offsets", original)
    assert_fixed_profile_integrity()


def test_exact_object_issuance_rejects_snapshot_component_frontier_and_path_replay() -> None:
    physical = _physical()
    memory, snapshot, planner = _build_projected_snapshot(
        _manifest(),
        physical,
        origin_xy_m=(-1.5, -1.6),
        configuration_shape=(30, 32),
    )
    restored_snapshot = TwoResolutionConfigurationSnapshotV2.deserialize(
        snapshot.serialize()
    )
    assert restored_snapshot.to_dict() == snapshot.to_dict()
    for replay in (
        copy.copy(snapshot),
        copy.deepcopy(snapshot),
        restored_snapshot,
        replace(snapshot),
    ):
        with pytest.raises(SnapshotBindingError, match="exact live"):
            planner.connected_component(replay, min(snapshot.free_cells))

    component = planner.connected_component(snapshot, min(snapshot.free_cells))
    planner.validate_component(snapshot, component)
    for replay in (copy.copy(component), copy.deepcopy(component), replace(component)):
        with pytest.raises(SnapshotBindingError, match="exact live"):
            planner.frontier_cells(snapshot, replay)

    first_frontiers = planner.frontier_cells(snapshot, component)
    second_frontiers = planner.frontier_cells(snapshot, component)
    assert first_frontiers.cells == tuple(sorted(first_frontiers.cells))
    assert first_frontiers.cells == second_frontiers.cells
    planner.validate_frontiers(snapshot, component, first_frontiers)
    planner.validate_frontiers(snapshot, component, second_frontiers)
    for replay in (
        copy.copy(first_frontiers),
        copy.deepcopy(first_frontiers),
        replace(first_frontiers),
        ConfigurationFrontiersV2(
            snapshot_sha256=first_frontiers.snapshot_sha256,
            component_sha256=first_frontiers.component_sha256,
            physical_map_frame_sha256=(
                first_frontiers.physical_map_frame_sha256
            ),
            configuration_map_frame_sha256=(
                first_frontiers.configuration_map_frame_sha256
            ),
            physical_revision=first_frontiers.physical_revision,
            configuration_revision=first_frontiers.configuration_revision,
            free_support_sha256=first_frontiers.free_support_sha256,
            occupied_support_sha256=first_frontiers.occupied_support_sha256,
            cells=first_frontiers.cells,
        ),
    ):
        with pytest.raises(SnapshotBindingError, match="exact live"):
            planner.validate_frontiers(snapshot, component, replay)

    goal = max(component.cells)
    path = planner.astar(snapshot, component.start_cell, goal)
    assert path is not None
    planner.validate_path(snapshot, path)
    for replay in (
        copy.copy(path),
        copy.deepcopy(path),
        replace(path),
        ConfigurationPathV2(
            snapshot_sha256=path.snapshot_sha256,
            physical_map_frame_sha256=path.physical_map_frame_sha256,
            configuration_map_frame_sha256=(
                path.configuration_map_frame_sha256
            ),
            physical_revision=path.physical_revision,
            configuration_revision=path.configuration_revision,
            free_support_sha256=path.free_support_sha256,
            occupied_support_sha256=path.occupied_support_sha256,
            cells=path.cells,
            cost=path.cost,
        ),
    ):
        with pytest.raises(StalePathError, match="exact live"):
            planner.validate_path(snapshot, replay)
    planner.validate_component(snapshot, component)
    planner.validate_frontiers(snapshot, component, first_frontiers)
    planner.validate_path(snapshot, path)


def test_memory_serialization_transaction_retraction_and_reset_binding() -> None:
    physical = _physical()
    memory, snapshot, _planner = _build_projected_snapshot(
        _manifest(),
        physical,
        origin_xy_m=(-1.5, -1.6),
        configuration_shape=(30, 32),
    )

    restored_memory = RevisionedPhysicalMemory.deserialize(memory.serialize())
    restored_projection = TwoResolutionConfigurationProjectionV2(
        restored_memory,
        configuration_map_frame=snapshot.configuration_map_frame,
        physical_shape=snapshot.physical_shape,
        configuration_shape=snapshot.configuration_shape,
    )
    restored = restored_projection.project()
    assert restored.to_dict() == snapshot.to_dict()
    restored_planner = TwoResolutionConfigurationPlannerV2(restored_projection)
    restored_planner.connected_component(restored, min(restored.free_cells))

    memory.apply_transaction(_learned_transaction(memory, "learned-retractable"))
    assert memory.learned_observation_ids == frozenset({"learned-retractable"})
    memory.apply_transaction(
        _learned_transaction(
            memory,
            "learned-retraction",
            retractions=("learned-retractable",),
        )
    )
    assert not memory.learned_observation_ids
    current_projection = _projection(memory, snapshot)
    current = current_projection.project()
    assert current.physical_revision == 3
    assert current.configuration_revision == 1

    _reset_memory, reset_snapshot, reset_planner = _build_projected_snapshot(
        _manifest("g3-v2-after-reset"),
        physical,
        origin_xy_m=(-1.5, -1.6),
        configuration_shape=(30, 32),
    )
    assert reset_snapshot.physical_map_frame_sha256 != snapshot.physical_map_frame_sha256
    with pytest.raises(SnapshotBindingError):
        reset_planner.connected_component(snapshot, min(snapshot.free_cells))


def test_admitted_contact_projects_one_exact_configuration_block_without_dilation() -> None:
    physical = np.full((60, 64), int(PhysicalLabel.FREE), dtype=np.uint8)
    memory, snapshot, planner = _build_projected_snapshot(
        _manifest(),
        physical,
        origin_xy_m=(-1.5, -1.6),
        configuration_shape=(30, 32),
    )
    blocked_configuration = (15, 16)
    blocked_physical = (30, 32)
    assert snapshot.state(blocked_configuration) is PhysicalLabel.FREE
    memory.apply_transaction(
        _contact_transaction(memory, physical_cell=blocked_physical)
    )
    with pytest.raises(StaleSnapshotError):
        planner.connected_component(snapshot, blocked_configuration)

    projection = _projection(memory, snapshot)
    current = projection.project()
    assert memory.execution_block_cells == frozenset({blocked_physical})
    assert current.physical_execution_block_cells == frozenset({blocked_physical})
    assert current.configuration_execution_block_cells == frozenset(
        {blocked_configuration}
    )
    assert current.state(blocked_configuration) is PhysicalLabel.OCCUPIED
    for neighbor in ((14, 16), (16, 16), (15, 15), (15, 17)):
        assert current.state(neighbor) is PhysicalLabel.FREE
    assert current.physical_revision == memory.revision == 2
    assert len(current.execution_block_receipt_sha256) == 64


def test_exact_control_adapter_rejects_caller_learned_authority() -> None:
    physical = np.full((20, 20), int(PhysicalLabel.FREE), dtype=np.uint8)
    memory, snapshot, _planner = _build_projected_snapshot(
        _manifest(),
        physical,
        origin_xy_m=(-0.5, -0.5),
        configuration_shape=(10, 10),
    )
    labels = {(0, 0): PhysicalLabel.FREE}
    camera = memory.bound_camera_transform_sha256
    assert camera is not None
    with pytest.raises(ValueError, match="authority"):
        ZeroInflationExactPhysicalAdapterV1(memory).fuse_cells(
            labels,
            observation=ObservationIdentity(
                observation_id="bad-learned-exact-adapter",
                payload_sha256=exact_physical_cells_content_sha256(labels),
                producer_sha256=hashlib.sha256(b"bad-learned").hexdigest(),
                authority=EvidenceAuthority.LEARNED_PHYSICAL,
            ),
            pose=PoseProvenance(
                source=PoseSource.DEPLOYMENT_ODOMETRY,
                frame_id=memory.map_frame.frame_id,
                mean_xy_yaw=(0.0, 0.0, 0.0),
                covariance_xy_yaw=((0.0, 0.0, 0.0),) * 3,
                timestamp_ns=2,
                synchronization_id="bad-learned",
                camera_transform_sha256=camera,
            ),
            label_inflation_radius_m=0.0,
        )
    assert snapshot.physical_revision == memory.revision == 1
