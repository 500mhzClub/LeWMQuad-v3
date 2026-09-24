from __future__ import annotations

from dataclasses import FrozenInstanceError, replace
import copy
import hashlib
import json
import math

import numpy as np
import pytest

from lewm.datasets.go2_paired_navigation import (
    derive_configuration_labels_from_fused_physical_raster,
)
from lewm.planning.geometry_contract import (
    DEPLOYMENT_GEOMETRY_CONTRACT,
    load_geometry_contract,
)
from lewm.planning.revisioned_physical_configuration_memory import (
    ConfigurationMorphology,
    ConfigurationPlanner,
    EvidenceAuthority,
    ExecutionBlock,
    ExecutionBlockKind,
    FusionMode,
    MapFrameIdentity,
    ObservationIdentity,
    PhysicalCellEvidence,
    PhysicalEvidenceTransaction,
    PhysicalLabel,
    PhysicalMemoryConfig,
    PoseProvenance,
    PoseSource,
    REGISTERED_FREE_SUPPORT_COUNT,
    REGISTERED_OCCUPIED_SUPPORT_COUNT,
    RevisionedPhysicalMemory,
    SnapshotBindingError,
    StalePathError,
    StaleSnapshotError,
    TransactionRejectedError,
    VerifiedTraversalPolygon,
)
from lewm.planning.zero_inflation_exact_physical_adapter_v1 import (
    ZeroInflationExactPhysicalAdapterV1,
    exact_physical_cells_content_sha256,
    exact_physical_raster_cells,
)


def _hash(value: str) -> str:
    return hashlib.sha256(value.encode("ascii")).hexdigest()


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def _refresh_memory_hash(payload: dict) -> None:
    core = dict(payload)
    core.pop("physical_content_sha256", None)
    payload["physical_content_sha256"] = hashlib.sha256(
        _canonical_bytes(core)
    ).hexdigest()


def _frame(
    *,
    session_id: str = "test-reset",
    origin_xy_m: tuple[float, float] = (0.0, 0.0),
) -> MapFrameIdentity:
    return MapFrameIdentity(session_id=session_id, origin_xy_m=origin_xy_m)


def _pose(
    frame: MapFrameIdentity,
    *,
    timestamp_ns: int = 1,
    source: PoseSource = PoseSource.DEPLOYMENT_ODOMETRY,
    mean_xy_yaw: tuple[float, float, float] = (0.0, 0.0, 0.0),
    covariance_xy_yaw: tuple[
        tuple[float, float, float],
        tuple[float, float, float],
        tuple[float, float, float],
    ] = (
        (0.01, 0.0, 0.0),
        (0.0, 0.01, 0.0),
        (0.0, 0.0, 0.001),
    ),
    camera_transform_sha256: str | None = None,
) -> PoseProvenance:
    return PoseProvenance(
        source=source,
        frame_id=frame.frame_id,
        mean_xy_yaw=mean_xy_yaw,
        covariance_xy_yaw=covariance_xy_yaw,
        timestamp_ns=timestamp_ns,
        synchronization_id=f"sync-{timestamp_ns}",
        camera_transform_sha256=(
            camera_transform_sha256 or _hash("camera-transform")
        ),
    )


def _observation(
    observation_id: str,
    *,
    authority: EvidenceAuthority = EvidenceAuthority.LEARNED_PHYSICAL,
    payload_sha256: str | None = None,
) -> ObservationIdentity:
    return ObservationIdentity(
        observation_id=observation_id,
        payload_sha256=payload_sha256 or _hash(f"payload:{observation_id}"),
        producer_sha256=_hash(f"producer:{authority.value}"),
        authority=authority,
    )


def _memory(
    frame: MapFrameIdentity | None = None,
    *,
    fusion_mode: FusionMode = FusionMode.PERSISTENT,
    connectivity: int = 4,
    allow_exact_sim: bool = False,
    promoted_runtime: bool = False,
) -> RevisionedPhysicalMemory:
    return RevisionedPhysicalMemory(
        PhysicalMemoryConfig(
            map_frame=frame or _frame(),
            fusion_mode=fusion_mode,
            planning_connectivity=connectivity,
            allow_exact_sim_odometry_ablation=allow_exact_sim,
            expected_camera_transform_sha256=_hash("camera-transform"),
            promoted_runtime=promoted_runtime,
        )
    )


def _transaction(
    memory: RevisionedPhysicalMemory,
    observation_id: str,
    *,
    evidence: dict[tuple[int, int], PhysicalLabel] | None = None,
    authority: EvidenceAuthority = EvidenceAuthority.LEARNED_PHYSICAL,
    traversals: tuple[VerifiedTraversalPolygon, ...] = (),
    blocks: tuple[ExecutionBlock, ...] = (),
    retractions: tuple[str, ...] = (),
    observed_unknown_cells: tuple[tuple[int, int], ...] = (),
    frame: MapFrameIdentity | None = None,
    pose: PoseProvenance | None = None,
) -> PhysicalEvidenceTransaction:
    selected_frame = frame or memory.map_frame
    return PhysicalEvidenceTransaction(
        observation=_observation(observation_id, authority=authority),
        map_frame=selected_frame,
        pose=pose or _pose(selected_frame, timestamp_ns=memory.revision + 1),
        physical_evidence=tuple(
            PhysicalCellEvidence(cell=cell, label=label)
            for cell, label in sorted((evidence or {}).items())
        ),
        verified_traversals=traversals,
        execution_blocks=blocks,
        retract_learned_observation_ids=retractions,
        observed_unknown_cells=observed_unknown_cells,
    )


def _exact_fuse(
    memory: RevisionedPhysicalMemory,
    labels: dict[tuple[int, int], PhysicalLabel],
    *,
    observation_id: str = "exact",
) -> None:
    adapter = ZeroInflationExactPhysicalAdapterV1(memory)
    adapter.fuse_cells(
        labels,
        observation=_observation(
            observation_id,
            authority=EvidenceAuthority.EXACT_PHYSICAL,
            payload_sha256=exact_physical_cells_content_sha256(labels),
        ),
        pose=_pose(memory.map_frame),
        label_inflation_radius_m=0.0,
    )


def _support_cells(
    morphology: ConfigurationMorphology,
    configuration_cells: set[tuple[int, int]],
) -> set[tuple[int, int]]:
    return {
        (center[0] + offset[0], center[1] + offset[1])
        for center in configuration_cells
        for offset in morphology.free_support_offsets
    }


def test_registered_morphology_has_exact_independent_89_and_69_kernels() -> None:
    morphology = ConfigurationMorphology()
    assert len(morphology.free_support_offsets) == REGISTERED_FREE_SUPPORT_COUNT == 89
    assert (
        len(morphology.occupied_support_offsets)
        == REGISTERED_OCCUPIED_SUPPORT_COUNT
        == 69
    )
    brute_free: set[tuple[int, int]] = set()
    brute_occupied: set[tuple[int, int]] = set()
    for dx in range(-8, 9):
        for dy in range(-8, 9):
            square_dx = max(abs(dx) * 0.10 - 0.05, 0.0)
            square_dy = max(abs(dy) * 0.10 - 0.05, 0.0)
            if square_dx**2 + square_dy**2 <= 0.47**2 + 1e-12:
                brute_free.add((dx, dy))
            if (dx * 0.10) ** 2 + (dy * 0.10) ** 2 <= 0.47**2 + 1e-12:
                brute_occupied.add((dx, dy))
    assert set(morphology.free_support_offsets) == brute_free
    assert set(morphology.occupied_support_offsets) == brute_occupied
    assert brute_occupied < brute_free
    assert ConfigurationMorphology() == morphology


def test_registered_geometry_values_match_but_promotion_calibration_is_open() -> None:
    contract = load_geometry_contract(
        DEPLOYMENT_GEOMETRY_CONTRACT,
        verify_sources=False,
    )
    assert contract.configuration_space.online_cell_size_m == pytest.approx(0.10)
    assert contract.configuration_space.body_inflation_radius_m == pytest.approx(0.47)
    assert contract.configuration_space.connectivity == 4
    assert contract.configuration_space.allow_diagonal_corner_cutting is False
    assert contract.swept_footprint.calibration_required_for_physical_promotion is True
    assert contract.physical_promotion_ready is False


def test_sparse_snapshot_matches_existing_dense_brute_force_oracle() -> None:
    rng = np.random.default_rng(20260712)
    labels = rng.choice(
        np.asarray([0, 1, 2], dtype=np.uint8),
        size=(31, 31),
        p=(0.18, 0.72, 0.10),
    )
    frame = _frame(origin_xy_m=(-1.55, -1.55))
    memory = _memory(frame)
    cells = exact_physical_raster_cells(labels, min_cell=(0, 0))
    _exact_fuse(memory, cells)
    morphology = ConfigurationMorphology()
    candidates = {
        (x, y) for x in range(6, 25, 3) for y in range(6, 25, 3)
    }
    snapshot = memory.create_configuration_snapshot(
        morphology,
        candidate_cells=candidates,
    )

    x_centers = np.asarray([frame.cell_center((x, 0))[0] for x in range(31)])
    y_centers = np.asarray([frame.cell_center((0, y))[1] for y in range(31)])
    ordered = sorted(candidates)
    config_x = np.asarray([frame.cell_center(cell)[0] for cell in ordered])
    config_y = np.asarray([frame.cell_center(cell)[1] for cell in ordered])
    dense = derive_configuration_labels_from_fused_physical_raster(
        labels,
        physical_x_centers_m=x_centers,
        physical_y_centers_m=y_centers,
        configuration_world_x_m=config_x,
        configuration_world_y_m=config_y,
        footprint_radius_m=0.47,
        physical_cell_size_m=0.10,
    )
    sparse = np.asarray([int(snapshot.state(cell)) for cell in ordered])
    np.testing.assert_array_equal(sparse, dense)


def test_free_requires_all_89_and_free_only_boundary_occupied_stays_unknown() -> None:
    morphology = ConfigurationMorphology()
    support = set(morphology.free_support_offsets)
    memory = _memory()
    _exact_fuse(memory, {cell: PhysicalLabel.FREE for cell in support})
    snapshot = memory.create_configuration_snapshot(
        morphology, candidate_cells={(0, 0)}
    )
    assert snapshot.state((0, 0)) is PhysicalLabel.FREE

    boundary = min(
        set(morphology.free_support_offsets)
        - set(morphology.occupied_support_offsets)
    )
    memory = _memory(_frame(session_id="boundary"))
    labels = {cell: PhysicalLabel.FREE for cell in support}
    labels[boundary] = PhysicalLabel.OCCUPIED
    _exact_fuse(memory, labels)
    snapshot = memory.create_configuration_snapshot(
        morphology, candidate_cells={(0, 0)}
    )
    assert snapshot.state((0, 0)) is PhysicalLabel.UNKNOWN


def test_complementary_learned_views_certify_free_only_after_persistent_fusion() -> None:
    morphology = ConfigurationMorphology()
    support = tuple(morphology.free_support_offsets)
    midpoint = len(support) // 2
    memory = _memory()
    memory.apply_transaction(
        _transaction(
            memory,
            "view-a",
            evidence={cell: PhysicalLabel.FREE for cell in support[:midpoint]},
        )
    )
    first = memory.create_configuration_snapshot(
        morphology, candidate_cells={(0, 0)}
    )
    assert first.state((0, 0)) is PhysicalLabel.UNKNOWN
    memory.apply_transaction(
        _transaction(
            memory,
            "view-b",
            evidence={cell: PhysicalLabel.FREE for cell in support[midpoint:]},
        )
    )
    second = memory.create_configuration_snapshot(
        morphology, candidate_cells={(0, 0)}
    )
    assert second.state((0, 0)) is PhysicalLabel.FREE


def test_occupied_center_support_precedes_free_and_execution_block_is_not_dilated() -> None:
    morphology = ConfigurationMorphology()
    candidates = {(0, 0), (1, 0)}
    physical = _support_cells(morphology, candidates)
    memory = _memory()
    labels = {cell: PhysicalLabel.FREE for cell in physical}
    labels[(0, 0)] = PhysicalLabel.OCCUPIED
    _exact_fuse(memory, labels)
    snapshot = memory.create_configuration_snapshot(
        morphology, candidate_cells=candidates
    )
    assert snapshot.state((0, 0)) is PhysicalLabel.OCCUPIED

    block_frame = _frame(session_id="block")
    memory = _memory(block_frame)
    _exact_fuse(
        memory,
        {cell: PhysicalLabel.FREE for cell in physical},
        observation_id="block-background",
    )
    block = ExecutionBlock(
        block_id="contact-1",
        body_center_xy_m=block_frame.cell_center((0, 0)),
        kind=ExecutionBlockKind.CONTACT,
        outcome_sha256=_hash("contact-1"),
    )
    memory.apply_transaction(
        _transaction(memory, "block-event", blocks=(block,))
    )
    snapshot = memory.create_configuration_snapshot(
        morphology, candidate_cells=candidates
    )
    assert snapshot.state((0, 0)) is PhysicalLabel.OCCUPIED
    assert snapshot.state((1, 0)) is PhysicalLabel.FREE


def test_learned_evidence_is_reversible_but_exact_evidence_is_not() -> None:
    memory = _memory()
    memory.apply_transaction(
        _transaction(memory, "learned-free", evidence={(0, 0): PhysicalLabel.FREE})
    )
    memory.apply_transaction(
        _transaction(
            memory,
            "learned-occupied",
            evidence={(0, 0): PhysicalLabel.OCCUPIED},
        )
    )
    assert memory.physical_state((0, 0)) is PhysicalLabel.UNKNOWN
    receipt = memory.apply_transaction(
        _transaction(
            memory,
            "correction",
            retractions=("learned-occupied",),
        )
    )
    assert receipt.learned_observations_retracted == 1
    assert memory.physical_state((0, 0)) is PhysicalLabel.FREE

    exact = {(2, 2): PhysicalLabel.OCCUPIED}
    _exact_fuse(memory, exact, observation_id="exact-indelible")
    revision = memory.revision
    with pytest.raises(TransactionRejectedError, match="not retractable"):
        memory.apply_transaction(
            _transaction(
                memory,
                "bad-exact-retraction",
                retractions=("exact-indelible",),
            )
        )
    assert memory.revision == revision
    assert memory.physical_state((2, 2)) is PhysicalLabel.OCCUPIED


def test_verified_traversal_overrides_learned_only_and_blocks_remain_indelible() -> None:
    frame = _frame()
    memory = _memory(frame)
    memory.apply_transaction(
        _transaction(
            memory,
            "learned-wall",
            evidence={(0, 0): PhysicalLabel.OCCUPIED},
        )
    )
    traversal = VerifiedTraversalPolygon(
        traversal_id="sweep-1",
        vertices_xy_m=((-0.01, -0.01), (0.11, -0.01), (0.11, 0.11), (-0.01, 0.11)),
        outcome_sha256=_hash("sweep-1"),
    )
    block = ExecutionBlock(
        block_id="stall-1",
        body_center_xy_m=frame.cell_center((0, 0)),
        kind=ExecutionBlockKind.STALL,
        outcome_sha256=_hash("stall-1"),
    )
    memory.apply_transaction(
        _transaction(
            memory,
            "execution-outcome",
            traversals=(traversal,),
            blocks=(block,),
        )
    )
    assert memory.physical_state((0, 0)) is PhysicalLabel.FREE
    assert (0, 0) in memory.execution_block_cells

    _exact_fuse(
        memory,
        {(0, 0): PhysicalLabel.OCCUPIED},
        observation_id="exact-wall",
    )
    assert memory.physical_state((0, 0)) is PhysicalLabel.OCCUPIED
    snapshot = memory.create_configuration_snapshot(
        ConfigurationMorphology(), candidate_cells={(0, 0)}
    )
    assert snapshot.state((0, 0)) is PhysicalLabel.OCCUPIED


def test_current_frame_only_replaces_learned_evidence_not_traversal_or_blocks() -> None:
    frame = _frame()
    memory = _memory(frame, fusion_mode=FusionMode.CURRENT_FRAME_ONLY)
    traversal = VerifiedTraversalPolygon(
        traversal_id="persistent-sweep",
        vertices_xy_m=((-0.01, -0.01), (0.11, -0.01), (0.11, 0.11), (-0.01, 0.11)),
        outcome_sha256=_hash("persistent-sweep"),
    )
    block = ExecutionBlock(
        block_id="persistent-block",
        body_center_xy_m=frame.cell_center((3, 3)),
        kind=ExecutionBlockKind.EXECUTION_VETO,
        outcome_sha256=_hash("persistent-block"),
    )
    memory.apply_transaction(
        _transaction(
            memory,
            "frame-one",
            evidence={(5, 5): PhysicalLabel.FREE},
            traversals=(traversal,),
            blocks=(block,),
        )
    )
    memory.apply_transaction(
        _transaction(
            memory,
            "frame-two",
            evidence={(6, 6): PhysicalLabel.FREE},
        )
    )
    assert memory.learned_observation_ids == frozenset({"frame-two"})
    assert memory.physical_state((5, 5)) is PhysicalLabel.UNKNOWN
    assert memory.physical_state((6, 6)) is PhysicalLabel.FREE
    assert memory.physical_state((0, 0)) is PhysicalLabel.FREE
    assert memory.traversal_ids == frozenset({"persistent-sweep"})
    assert memory.execution_block_ids == frozenset({"persistent-block"})


def test_transaction_failure_is_atomic_and_revision_is_strictly_monotonic() -> None:
    memory = _memory()
    first_block = ExecutionBlock(
        block_id="block",
        body_center_xy_m=(0.05, 0.05),
        kind=ExecutionBlockKind.CONTACT,
        outcome_sha256=_hash("block"),
    )
    first = memory.apply_transaction(
        _transaction(memory, "first", blocks=(first_block,))
    )
    assert (first.revision_before, first.revision_after, memory.revision) == (0, 1, 1)
    content_before = memory.physical_content_sha256
    with pytest.raises(TransactionRejectedError, match="indelible"):
        memory.apply_transaction(
            _transaction(
                memory,
                "would-be-partial",
                evidence={(9, 9): PhysicalLabel.FREE},
                blocks=(first_block,),
            )
        )
    assert memory.revision == 1
    assert memory.physical_content_sha256 == content_before
    assert memory.physical_state((9, 9)) is PhysicalLabel.UNKNOWN

    wrong_frame = _frame(session_id="wrong", origin_xy_m=(1.0, 0.0))
    with pytest.raises(TransactionRejectedError, match="frame/origin"):
        memory.apply_transaction(
            _transaction(
                memory,
                "wrong-frame",
                evidence={(1, 1): PhysicalLabel.FREE},
                frame=wrong_frame,
                pose=_pose(wrong_frame),
            )
        )
    assert memory.revision == 1

    second = memory.apply_transaction(
        _transaction(memory, "second", evidence={(1, 1): PhysicalLabel.FREE})
    )
    assert (second.revision_before, second.revision_after, memory.revision) == (1, 2, 2)
    with pytest.raises(TransactionRejectedError, match="duplicate observation"):
        memory.apply_transaction(
            _transaction(memory, "second", evidence={(2, 2): PhysicalLabel.FREE})
        )
    assert memory.revision == 2


def test_transaction_key_binds_observation_map_origin_and_pose_provenance() -> None:
    memory = _memory()
    base = _transaction(memory, "key", evidence={(0, 0): PhysicalLabel.FREE})
    moved_pose = _transaction(
        memory,
        "key",
        evidence={(0, 0): PhysicalLabel.FREE},
        pose=_pose(memory.map_frame, mean_xy_yaw=(0.1, 0.0, 0.0)),
    )
    shifted_frame = _frame(session_id="shifted", origin_xy_m=(0.1, 0.0))
    shifted = _transaction(
        memory,
        "key",
        evidence={(0, 0): PhysicalLabel.FREE},
        frame=shifted_frame,
        pose=_pose(shifted_frame),
    )
    assert len(
        {
            base.transaction_key_sha256,
            moved_pose.transaction_key_sha256,
            shifted.transaction_key_sha256,
        }
    ) == 3


def test_exact_sim_pose_is_rejected_unless_explicit_ablation_is_constructed() -> None:
    memory = _memory()
    transaction = _transaction(
        memory,
        "sim-pose",
        evidence={(0, 0): PhysicalLabel.FREE},
        pose=_pose(
            memory.map_frame,
            source=PoseSource.EXACT_SIM_ODOMETRY_ABLATION,
        ),
    )
    with pytest.raises(TransactionRejectedError, match="simulator odometry"):
        memory.apply_transaction(transaction)
    assert memory.revision == 0

    ablation = _memory(_frame(session_id="ablation"), allow_exact_sim=True)
    receipt = ablation.apply_transaction(
        _transaction(
            ablation,
            "sim-pose",
            evidence={(0, 0): PhysicalLabel.FREE},
            pose=_pose(
                ablation.map_frame,
                source=PoseSource.EXACT_SIM_ODOMETRY_ABLATION,
            ),
        )
    )
    assert receipt.revision_after == 1


def test_raw_physical_memory_has_no_route_frontier_or_component_api() -> None:
    memory = _memory()
    forbidden = {
        "astar",
        "shortest_path",
        "route",
        "frontier_cells",
        "connected_component",
        "connected_confirmed_free",
    }
    assert all(not hasattr(memory, name) for name in forbidden)


def test_snapshot_component_frontier_astar_are_deterministic_and_stale_fail_closed() -> None:
    morphology = ConfigurationMorphology()
    configuration_cells = {(x, y) for x in range(3) for y in range(3)}
    memory = _memory()
    physical = _support_cells(morphology, configuration_cells)
    _exact_fuse(memory, {cell: PhysicalLabel.FREE for cell in physical})
    snapshot = memory.create_configuration_snapshot(
        morphology, candidate_cells=configuration_cells
    )
    assert snapshot.free_cells == frozenset(configuration_cells)
    planner = ConfigurationPlanner(memory, morphology)
    component = planner.connected_component(snapshot, (0, 0))
    assert component.cells == frozenset(configuration_cells)
    frontiers = planner.frontier_cells(snapshot, component)
    assert frontiers.cells == ()
    first = planner.astar(snapshot, (0, 0), (2, 2))
    second = planner.astar(snapshot, (0, 0), (2, 2))
    assert first is not None and first == second
    assert first.cells[0] == (0, 0) and first.cells[-1] == (2, 2)
    assert first.cost == pytest.approx(4.0)
    planner.validate_path(snapshot, first)

    with pytest.raises(FrozenInstanceError):
        snapshot.physical_revision = 99  # type: ignore[misc]

    memory.apply_transaction(
        _transaction(memory, "revision-advance", evidence={(100, 100): PhysicalLabel.FREE})
    )
    with pytest.raises(StaleSnapshotError):
        planner.connected_component(snapshot, (0, 0))
    fresh = memory.create_configuration_snapshot(
        morphology, candidate_cells=configuration_cells
    )
    with pytest.raises(StalePathError):
        planner.validate_path(fresh, first)


def test_eight_connected_astar_uses_octile_cost_without_corner_cutting() -> None:
    morphology = ConfigurationMorphology()
    memory = _memory(connectivity=8)
    physical = _support_cells(morphology, {(0, 0), (1, 1), (1, 0), (0, 1)})
    _exact_fuse(memory, {cell: PhysicalLabel.FREE for cell in physical})
    planner = ConfigurationPlanner(memory, morphology)
    corner_only = memory.create_configuration_snapshot(
        morphology, candidate_cells={(0, 0), (1, 1)}
    )
    assert planner.astar(corner_only, (0, 0), (1, 1)) is None

    supported = memory.create_configuration_snapshot(
        morphology,
        candidate_cells={(0, 0), (1, 0), (0, 1), (1, 1)},
    )
    path = planner.astar(supported, (0, 0), (1, 1))
    assert path is not None
    assert path.cells == ((0, 0), (1, 1))
    assert path.cost == pytest.approx(math.sqrt(2.0))


def test_planner_rejects_snapshot_with_wrong_morphology_hash() -> None:
    morphology = ConfigurationMorphology()
    memory = _memory()
    physical = _support_cells(morphology, {(0, 0)})
    _exact_fuse(memory, {cell: PhysicalLabel.FREE for cell in physical})
    snapshot = memory.create_configuration_snapshot(
        morphology, candidate_cells={(0, 0)}
    )
    malformed = replace(snapshot, free_support_sha256="0" * 64)
    with pytest.raises(SnapshotBindingError, match="morphology"):
        ConfigurationPlanner(memory, morphology).connected_component(
            malformed, (0, 0)
        )


def test_registered_memory_rejects_weakened_radius_and_forged_snapshot() -> None:
    morphology = ConfigurationMorphology()
    memory = _memory()
    physical = _support_cells(morphology, {(0, 0)})
    _exact_fuse(memory, {cell: PhysicalLabel.FREE for cell in physical})
    with pytest.raises(ValueError, match="exact 0.47 m 89/69"):
        memory.create_configuration_snapshot(
            ConfigurationMorphology(footprint_radius_m=0.10),
            candidate_cells={(0, 0)},
        )

    issued = memory.create_configuration_snapshot(
        morphology, candidate_cells={(0, 0)}
    )
    forged = replace(
        issued,
        free_cells=frozenset({(0, 0), (100, 100)}),
    )
    with pytest.raises(SnapshotBindingError, match="not derived"):
        ConfigurationPlanner(memory, morphology).connected_component(
            forged, (0, 0)
        )


class _RecordingMemory(RevisionedPhysicalMemory):
    def __init__(self, config: PhysicalMemoryConfig) -> None:
        super().__init__(config)
        self.apply_calls = 0

    def apply_transaction(
        self, transaction: PhysicalEvidenceTransaction
    ):  # type: ignore[no-untyped-def]
        self.apply_calls += 1
        return super().apply_transaction(transaction)


def test_exact_adapter_verifies_zero_inflation_and_uses_shared_transaction_once() -> None:
    frame = _frame()
    memory = _RecordingMemory(PhysicalMemoryConfig(map_frame=frame))
    adapter = ZeroInflationExactPhysicalAdapterV1(memory)
    raster = np.asarray([[1, 0], [2, 1]], dtype=np.uint8)
    cells = exact_physical_raster_cells(raster, min_cell=(-1, 3))
    observation = _observation(
        "raster",
        authority=EvidenceAuthority.EXACT_PHYSICAL,
        payload_sha256=exact_physical_cells_content_sha256(cells),
    )
    receipt = adapter.fuse_raster(
        raster,
        min_cell=(-1, 3),
        observation=observation,
        pose=_pose(frame),
        label_inflation_radius_m=0.0,
    )
    assert memory.apply_calls == 1
    assert receipt.transaction_sha256
    assert receipt.revision_after == memory.revision == 1
    assert memory.exact_observation_ids == frozenset({"raster"})
    assert memory.physical_state((-1, 3)) is PhysicalLabel.FREE
    assert memory.physical_state((-1, 4)) is PhysicalLabel.UNKNOWN
    assert memory.physical_state((0, 3)) is PhysicalLabel.OCCUPIED


@pytest.mark.parametrize(
    ("inflation", "semantics"),
    [(0.01, "zero_inflation_physical_occupancy"), (0.0, "inflated")],
)
def test_exact_adapter_rejects_nonzero_inflation_or_wrong_semantics_atomically(
    inflation: float,
    semantics: str,
) -> None:
    memory = _memory()
    adapter = ZeroInflationExactPhysicalAdapterV1(memory)
    labels = {(0, 0): PhysicalLabel.FREE}
    observation = _observation(
        "bad-semantics",
        authority=EvidenceAuthority.EXACT_PHYSICAL,
        payload_sha256=exact_physical_cells_content_sha256(labels),
    )
    with pytest.raises(ValueError):
        adapter.fuse_cells(
            labels,
            observation=observation,
            pose=_pose(memory.map_frame),
            label_inflation_radius_m=inflation,
            source_semantics=semantics,
        )
    assert memory.revision == 0


def test_exact_adapter_rejects_payload_hash_or_authority_mismatch() -> None:
    labels = {(0, 0): PhysicalLabel.FREE}
    memory = _memory()
    adapter = ZeroInflationExactPhysicalAdapterV1(memory)
    with pytest.raises(ValueError, match="payload"):
        adapter.build_transaction_from_cells(
            labels,
            observation=_observation(
                "bad-hash",
                authority=EvidenceAuthority.EXACT_PHYSICAL,
            ),
            pose=_pose(memory.map_frame),
            label_inflation_radius_m=0.0,
        )
    with pytest.raises(ValueError, match="authority"):
        adapter.build_transaction_from_cells(
            labels,
            observation=_observation(
                "bad-authority",
                authority=EvidenceAuthority.LEARNED_PHYSICAL,
                payload_sha256=exact_physical_cells_content_sha256(labels),
            ),
            pose=_pose(memory.map_frame),
            label_inflation_radius_m=0.0,
        )
    assert memory.revision == 0


def test_exact_adapter_rejects_fractional_or_boolean_labels() -> None:
    with pytest.raises(ValueError, match="unsupported physical label"):
        exact_physical_cells_content_sha256({(0, 0): 1.5})  # type: ignore[dict-item]
    with pytest.raises(ValueError, match="boolean"):
        exact_physical_cells_content_sha256({(0, 0): True})  # type: ignore[dict-item]
    with pytest.raises(ValueError, match="unsupported physical label"):
        exact_physical_cells_content_sha256({(0, 0): 1.0})  # type: ignore[dict-item]


def test_exact_authority_cannot_bypass_adapter_and_contracts_fail_atomically() -> None:
    memory = _memory()
    wrong_projection = _transaction(
        memory,
        "wrong-projection",
        evidence={(3, 3): PhysicalLabel.FREE},
    )
    object.__setattr__(
        wrong_projection,
        "projection_contract_sha256",
        _hash("wrong-projection"),
    )
    with pytest.raises(TransactionRejectedError, match="projection contract"):
        memory.apply_transaction(wrong_projection)
    assert memory.revision == 0

    exact_observation = _observation(
        "direct-exact",
        authority=EvidenceAuthority.EXACT_PHYSICAL,
    )
    with pytest.raises(TypeError, match="adapter admission"):
        PhysicalEvidenceTransaction(
            observation=exact_observation,
            map_frame=memory.map_frame,
            pose=_pose(memory.map_frame),
            physical_evidence=(
                PhysicalCellEvidence(cell=(0, 0), label=PhysicalLabel.FREE),
            ),
        )

    labels = {(0, 0): PhysicalLabel.FREE}
    observation = _observation(
        "uncertain-exact",
        authority=EvidenceAuthority.EXACT_PHYSICAL,
        payload_sha256=exact_physical_cells_content_sha256(labels),
    )
    uncertain_pose = _pose(
        memory.map_frame,
        covariance_xy_yaw=(
            (0.05, 0.0, 0.0),
            (0.0, 0.01, 0.0),
            (0.0, 0.0, 0.001),
        ),
    )
    with pytest.raises(TransactionRejectedError, match="covariance"):
        ZeroInflationExactPhysicalAdapterV1(memory).fuse_cells(
            labels,
            observation=observation,
            pose=uncertain_pose,
            label_inflation_radius_m=0.0,
        )
    assert memory.revision == 0

    transaction = ZeroInflationExactPhysicalAdapterV1(
        memory
    ).build_transaction_from_cells(
        labels,
        observation=observation,
        pose=_pose(memory.map_frame),
        label_inflation_radius_m=0.0,
    )
    with pytest.raises(ValueError, match="admitted labels"):
        replace(
            transaction,
            physical_evidence=(
                PhysicalCellEvidence(
                    cell=(0, 0), label=PhysicalLabel.OCCUPIED
                ),
            ),
        )
    assert transaction.exact_admission is not None
    object.__setattr__(
        transaction.exact_admission,
        "projection_contract_sha256",
        _hash("forged-projection"),
    )
    with pytest.raises(TransactionRejectedError, match="admission contract"):
        memory.apply_transaction(transaction)
    assert memory.revision == 0


def test_calibration_binding_and_exact_sim_taint_are_explicit() -> None:
    memory = _memory()
    memory.apply_transaction(
        _transaction(memory, "bind-camera", evidence={(0, 0): PhysicalLabel.FREE})
    )
    labels = {(2, 2): PhysicalLabel.FREE}
    observation = _observation(
        "other-camera",
        authority=EvidenceAuthority.EXACT_PHYSICAL,
        payload_sha256=exact_physical_cells_content_sha256(labels),
    )
    with pytest.raises(TransactionRejectedError, match="calibration"):
        ZeroInflationExactPhysicalAdapterV1(memory).fuse_cells(
            labels,
            observation=observation,
            pose=_pose(
                memory.map_frame,
                camera_transform_sha256=_hash("other-camera"),
            ),
            label_inflation_radius_m=0.0,
        )
    assert memory.revision == 1

    with pytest.raises(ValueError, match="cannot be a promoted runtime"):
        PhysicalMemoryConfig(
            map_frame=_frame(session_id="bad-promoted-sim"),
            allow_exact_sim_odometry_ablation=True,
            expected_camera_transform_sha256=_hash("camera-transform"),
            promoted_runtime=True,
        )
    with pytest.raises(ValueError, match="frozen projection, calibration"):
        PhysicalMemoryConfig(
            map_frame=_frame(session_id="missing-calibration"),
            promoted_runtime=True,
        )
    with pytest.raises(ValueError, match="frozen projection, calibration"):
        PhysicalMemoryConfig(
            map_frame=_frame(session_id="loose-uncertainty"),
            expected_camera_transform_sha256=_hash("camera-transform"),
            pose_covariance_diagonal_limits=(1.0, 1.0, 1.0),
            promoted_runtime=True,
        )

    ablation = _memory(_frame(session_id="tainted"), allow_exact_sim=True)
    tainted_labels = {(0, 0): PhysicalLabel.FREE}
    ZeroInflationExactPhysicalAdapterV1(ablation).fuse_cells(
        tainted_labels,
        observation=_observation(
            "tainted-exact",
            authority=EvidenceAuthority.EXACT_PHYSICAL,
            payload_sha256=exact_physical_cells_content_sha256(tainted_labels),
        ),
        pose=_pose(
            ablation.map_frame,
            source=PoseSource.EXACT_SIM_ODOMETRY_ABLATION,
        ),
        label_inflation_radius_m=0.0,
    )
    assert ablation.exact_sim_tainted is True
    snapshot = ablation.create_configuration_snapshot(
        ConfigurationMorphology(), candidate_cells={(0, 0)}
    )
    assert snapshot.exact_sim_tainted is True


def test_promoted_runtime_rejects_unissued_learned_execution_and_exact_evidence() -> None:
    frame = _frame(session_id="promoted-admission")
    memory = _memory(frame, promoted_runtime=True)
    learned = _transaction(
        memory,
        "caller-learned",
        evidence={(100, 100): PhysicalLabel.FREE},
    )
    with pytest.raises(TransactionRejectedError, match="qualified projection adapter"):
        memory.apply_transaction(learned)

    traversal = VerifiedTraversalPolygon(
        traversal_id="caller-traversal",
        vertices_xy_m=((-5.0, -5.0), (5.0, -5.0), (5.0, 5.0), (-5.0, 5.0)),
        outcome_sha256=_hash("not-an-executor-receipt"),
    )
    with pytest.raises(TransactionRejectedError, match="issued outcome adapter"):
        memory.apply_transaction(
            _transaction(memory, "caller-traversal", traversals=(traversal,))
        )

    block = ExecutionBlock(
        block_id="caller-block",
        body_center_xy_m=frame.cell_center((50, 50)),
        kind=ExecutionBlockKind.EXECUTION_VETO,
        outcome_sha256=_hash("not-an-issued-block-receipt"),
    )
    with pytest.raises(TransactionRejectedError, match="issued outcome adapter"):
        memory.apply_transaction(
            _transaction(memory, "caller-block", blocks=(block,))
        )

    exact_labels = {(999, 999): PhysicalLabel.FREE}
    with pytest.raises(PermissionError, match="development-only"):
        ZeroInflationExactPhysicalAdapterV1(memory).fuse_cells(
            exact_labels,
            observation=_observation(
                "caller-exact",
                authority=EvidenceAuthority.EXACT_PHYSICAL,
                payload_sha256=exact_physical_cells_content_sha256(exact_labels),
            ),
            pose=_pose(frame),
            label_inflation_radius_m=0.0,
        )
    assert memory.revision == 0
    assert memory.physical_state((100, 100)) is PhysicalLabel.UNKNOWN
    assert not memory.verified_traversal_cells
    assert not memory.execution_block_ids


def test_all_exact_evidence_marks_development_snapshot_privileged() -> None:
    memory = _memory(_frame(session_id="privileged-exact"))
    _exact_fuse(memory, {(0, 0): PhysicalLabel.FREE})
    assert memory.exact_sim_tainted is True
    snapshot = memory.create_configuration_snapshot(
        ConfigurationMorphology(), candidate_cells={(0, 0)}
    )
    assert snapshot.exact_sim_tainted is True


def test_object_forged_morphology_snapshot_and_component_fail_closed() -> None:
    morphology = ConfigurationMorphology()
    memory = _memory()
    configuration_cells = {(0, 0), (1, 0)}
    _exact_fuse(
        memory,
        {
            cell: PhysicalLabel.FREE
            for cell in _support_cells(morphology, configuration_cells)
        },
    )
    snapshot = memory.create_configuration_snapshot(
        morphology, candidate_cells=configuration_cells
    )
    planner = ConfigurationPlanner(memory, morphology)
    component = planner.connected_component(snapshot, (0, 0))

    forged_component = replace(component, cells=frozenset({(0, 0)}))
    with pytest.raises(SnapshotBindingError, match="complete connected"):
        planner.frontier_cells(snapshot, forged_component)

    object.__setattr__(snapshot, "free_cells", frozenset({(0, 0), (99, 99)}))
    with pytest.raises(SnapshotBindingError, match="snapshot was mutated"):
        planner.connected_component(snapshot, (0, 0))

    fresh_morphology = ConfigurationMorphology()
    forged_offsets = list(fresh_morphology.free_support_offsets)
    forged_offsets[0] = (999, 999)
    object.__setattr__(
        fresh_morphology,
        "free_support_offsets",
        tuple(forged_offsets),
    )
    with pytest.raises(SnapshotBindingError, match="morphology"):
        memory.create_configuration_snapshot(
            fresh_morphology, candidate_cells={(0, 0)}
        )


def test_semantic_duplicate_with_new_identity_is_rejected_atomically() -> None:
    memory = _memory()
    first = _transaction(
        memory,
        "semantic-a",
        evidence={(0, 0): PhysicalLabel.FREE},
    )
    memory.apply_transaction(first)
    content_before = memory.physical_content_sha256
    duplicate = replace(
        first,
        observation=_observation("semantic-b"),
        pose=_pose(memory.map_frame, timestamp_ns=999),
    )
    with pytest.raises(TransactionRejectedError, match="semantic duplicate"):
        memory.apply_transaction(duplicate)
    assert memory.revision == 1
    assert memory.physical_content_sha256 == content_before

    exact_memory = _memory(_frame(session_id="semantic-exact"))
    exact_labels = {(0, 0): PhysicalLabel.FREE}
    _exact_fuse(
        exact_memory,
        exact_labels,
        observation_id="semantic-exact-a",
    )
    with pytest.raises(TransactionRejectedError, match="semantic duplicate"):
        _exact_fuse(
            exact_memory,
            exact_labels,
            observation_id="semantic-exact-b",
        )
    assert exact_memory.revision == 1


def test_current_frame_all_unknown_and_block_only_clear_stale_learned() -> None:
    memory = _memory(fusion_mode=FusionMode.CURRENT_FRAME_ONLY)
    memory.apply_transaction(
        _transaction(memory, "visible", evidence={(5, 5): PhysicalLabel.FREE})
    )
    memory.apply_transaction(
        _transaction(
            memory,
            "all-unknown",
            observed_unknown_cells=((5, 5), (6, 6)),
        )
    )
    assert memory.learned_observation_ids == frozenset()
    assert memory.physical_state((5, 5)) is PhysicalLabel.UNKNOWN

    memory.apply_transaction(
        _transaction(memory, "visible-again", evidence={(7, 7): PhysicalLabel.FREE})
    )
    block = ExecutionBlock(
        block_id="frame-block",
        body_center_xy_m=memory.map_frame.cell_center((1, 1)),
        kind=ExecutionBlockKind.CONTACT,
        outcome_sha256=_hash("frame-block"),
    )
    memory.apply_transaction(_transaction(memory, "block-only", blocks=(block,)))
    assert memory.learned_observation_ids == frozenset()
    assert memory.physical_state((7, 7)) is PhysicalLabel.UNKNOWN
    assert memory.execution_block_ids == frozenset({"frame-block"})


def test_truncated_candidate_domain_cannot_manufacture_frontier() -> None:
    morphology = ConfigurationMorphology()
    memory = _memory()
    _exact_fuse(
        memory,
        {
            cell: PhysicalLabel.FREE
            for cell in _support_cells(morphology, {(0, 0)})
        },
    )
    planner = ConfigurationPlanner(memory, morphology)
    truncated = memory.create_configuration_snapshot(
        morphology, candidate_cells={(0, 0)}
    )
    component = planner.connected_component(truncated, (0, 0))
    assert planner.frontier_cells(truncated, component).cells == ()

    explicit = memory.create_configuration_snapshot(
        morphology, candidate_cells={(0, 0), (1, 0)}
    )
    explicit_component = planner.connected_component(explicit, (0, 0))
    assert explicit.state((1, 0)) is PhysicalLabel.UNKNOWN
    assert planner.frontier_cells(explicit, explicit_component).cells == ((0, 0),)


def test_conflicting_exact_evidence_is_rejected_not_collapsed_to_unknown() -> None:
    memory = _memory()
    _exact_fuse(memory, {(0, 0): PhysicalLabel.FREE}, observation_id="exact-free")
    revision = memory.revision
    content = memory.physical_content_sha256
    with pytest.raises(TransactionRejectedError, match="conflicting exact"):
        _exact_fuse(
            memory,
            {(0, 0): PhysicalLabel.OCCUPIED},
            observation_id="exact-occupied",
        )
    assert memory.revision == revision
    assert memory.physical_content_sha256 == content
    assert memory.physical_state((0, 0)) is PhysicalLabel.FREE


def test_strict_state_roundtrip_replays_full_pose_config_keys_and_taint() -> None:
    memory = _memory(_frame(session_id="serialize"), allow_exact_sim=True)
    first = _transaction(
        memory,
        "learned",
        evidence={(0, 0): PhysicalLabel.FREE},
    )
    first_pose_dict = first.pose.to_dict()
    memory.apply_transaction(first)
    committed_hash = memory.physical_content_sha256
    object.__setattr__(first.pose, "mean_xy_yaw", (99.0, 99.0, 0.0))
    assert memory.physical_content_sha256 == committed_hash
    exact = {(2, 2): PhysicalLabel.OCCUPIED}
    ZeroInflationExactPhysicalAdapterV1(memory).fuse_cells(
        exact,
        observation=_observation(
            "sim-exact",
            authority=EvidenceAuthority.EXACT_PHYSICAL,
            payload_sha256=exact_physical_cells_content_sha256(exact),
        ),
        pose=_pose(
            memory.map_frame,
            timestamp_ns=2,
            source=PoseSource.EXACT_SIM_ODOMETRY_ABLATION,
        ),
        label_inflation_radius_m=0.0,
    )
    payload = memory.to_dict()
    assert payload["config"] == memory.config.to_dict()
    assert payload["exact_sim_tainted"] is True
    assert payload["seen_transaction_keys"]
    assert payload["seen_semantic_transaction_keys"]
    assert payload["transactions"][0]["pose"] == first_pose_dict

    encoded = memory.serialize()
    restored = RevisionedPhysicalMemory.deserialize(encoded)
    assert restored.to_dict() == payload
    assert restored.serialize() == encoded
    assert restored.physical_content_sha256 == memory.physical_content_sha256

    with pytest.raises(ValueError, match="canonical JSON"):
        RevisionedPhysicalMemory.deserialize(
            json.dumps(payload, indent=2, sort_keys=True).encode("utf-8")
        )

    mutations = []
    damaged = copy.deepcopy(payload)
    damaged["seen_semantic_transaction_keys"] = []
    mutations.append(damaged)
    damaged = copy.deepcopy(payload)
    damaged["transactions"][0]["pose"]["timestamp_ns"] = 999
    mutations.append(damaged)
    damaged = copy.deepcopy(payload)
    damaged["config"]["planning_connectivity"] = 8
    mutations.append(damaged)
    damaged = copy.deepcopy(payload)
    damaged["exact_sim_tainted"] = False
    mutations.append(damaged)
    damaged = copy.deepcopy(payload)
    damaged["unexpected"] = None
    mutations.append(damaged)
    for damaged in mutations:
        _refresh_memory_hash(damaged)
        with pytest.raises((ValueError, TransactionRejectedError), match="."):
            RevisionedPhysicalMemory.from_mapping(damaged)
