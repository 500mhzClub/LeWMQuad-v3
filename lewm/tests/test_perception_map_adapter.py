from __future__ import annotations

from dataclasses import replace
import json
import math

import numpy as np
import pytest

from lewm.planning.online_belief_map import (
    BeliefMapConfig,
    CellState,
    OnlineBeliefMap,
    PoseBelief,
    four_neighbors,
)
from lewm.planning.perception_map_adapter import (
    CameraGeometry,
    EgocentricOccupancyGeometry,
    GridReferenceFrame,
    ModelArtifact,
    ObservationProvenance,
    OccupancyObservation,
    OccupancyValueKind,
    PerceptionMapAdapterConfig,
    PerceptionMapContract,
    PerceptionToBeliefMapAdapter,
    supercover_project_cell,
)

BACKBONE = ModelArtifact("jepa-spatial-v1", "a" * 64)
HEAD = ModelArtifact("body-occupancy-v1", "b" * 64)


def _camera(
    *,
    mount_xyz_m: tuple[float, float, float] = (0.0, 0.0, 0.4),
) -> CameraGeometry:
    return CameraGeometry(
        calibration_id="go2-front-v1",
        image_height_px=128,
        image_width_px=160,
        horizontal_fov_deg=90.0,
        vertical_fov_deg=78.0,
        mount_xyz_m=mount_xyz_m,
        mount_rpy_rad=(0.0, 0.0, 0.0),
    )


def _geometry(
    *,
    height: int = 3,
    width: int = 1,
    cell_size_m: float = 1.0,
    reference_frame: GridReferenceFrame = GridReferenceFrame.BODY,
) -> EgocentricOccupancyGeometry:
    return EgocentricOccupancyGeometry(
        geometry_id="body-grid-v1",
        height=height,
        width=width,
        cell_size_m=cell_size_m,
        forward_min_m=0.0,
        left_min_m=0.0,
        reference_frame=reference_frame,
        body_inflation_radius_m=0.25,
    )


def _contract(
    *,
    camera: CameraGeometry | None = None,
    geometry: EgocentricOccupancyGeometry | None = None,
) -> PerceptionMapContract:
    return PerceptionMapContract(
        backbone=BACKBONE,
        occupancy_head=HEAD,
        probability_calibration_id="temperature-heldout-v1",
        camera=camera or _camera(),
        occupancy_geometry=geometry or _geometry(),
        map_cell_size_m=1.0,
    )


def _pose(
    tick: int,
    *,
    yaw: float = 0.0,
    covariance_xy: float = 0.0,
    frame: str = "odometry",
) -> PoseBelief:
    return PoseBelief(
        mean=(0.0, 0.0, yaw),
        covariance=(
            (covariance_xy, 0.0, 0.0),
            (0.0, covariance_xy, 0.0),
            (0.0, 0.0, 0.01),
        ),
        tick=tick,
        frame=frame,
    )


def _observation(
    contract: PerceptionMapContract,
    values: np.ndarray,
    *,
    observation_id: str,
    tick: int,
    value_kind: OccupancyValueKind = OccupancyValueKind.PROBABILITY,
    valid_mask: np.ndarray | None = None,
    pose: PoseBelief | None = None,
    camera: CameraGeometry | None = None,
    geometry: EgocentricOccupancyGeometry | None = None,
    provenance: ObservationProvenance | None = None,
    confidence: float = 1.0,
) -> OccupancyObservation:
    return OccupancyObservation(
        values=values,
        value_kind=value_kind,
        pose=pose or _pose(tick),
        camera=camera or contract.camera,
        geometry=geometry or contract.occupancy_geometry,
        provenance=provenance
        or ObservationProvenance.create(observation_id, contract=contract),
        valid_mask=valid_mask,
        observation_confidence=confidence,
    )


def test_probability_grid_fuses_connected_free_cells_and_provenance() -> None:
    belief_map = OnlineBeliefMap(BeliefMapConfig(cell_size_m=1.0))
    contract = _contract()
    adapter = PerceptionToBeliefMapAdapter(belief_map, contract)
    observation = _observation(
        contract,
        np.zeros((3, 1), dtype=np.float32),
        observation_id="frame-0001",
        tick=1,
    )

    record = adapter.fuse(observation)

    corridor = ((0, 0), (1, 0), (2, 0))
    assert all(belief_map.is_confirmed_free(cell) for cell in corridor)
    assert belief_map.shortest_path((0, 0), (2, 0)) == corridor
    assert record.free_source_cells == 3
    assert record.projected_free_cells == 3
    assert record.projected_occupied_cells == 0
    assert record.backbone_checkpoint_sha256 == BACKBONE.checkpoint_sha256
    assert record.occupancy_head_checkpoint_sha256 == HEAD.checkpoint_sha256
    assert record.camera_geometry_sha256 == contract.camera.fingerprint
    assert record.occupancy_geometry_sha256 == contract.occupancy_geometry.fingerprint
    assert record.map_cell_size_m == 1.0
    assert record.contract_sha256 == contract.fingerprint
    assert len(record.observation_payload_sha256) == 64
    assert len(record.pose_registration_sha256) == 64
    assert record.pose_frame == "odometry"
    assert record.pose_mean == (0.0, 0.0, 0.0)
    assert not hasattr(adapter, "shortest_path")
    json.dumps(adapter.provenance_state_dict())


def test_logits_and_valid_mask_preserve_unknown_cells() -> None:
    belief_map = OnlineBeliefMap(BeliefMapConfig(cell_size_m=1.0))
    contract = _contract()
    adapter = PerceptionToBeliefMapAdapter(belief_map, contract)
    observation = _observation(
        contract,
        np.asarray([[-20.0], [0.0], [20.0]], dtype=np.float32),
        observation_id="logits-0001",
        tick=1,
        value_kind=OccupancyValueKind.LOGIT,
        valid_mask=np.asarray([[True], [False], [True]], dtype=np.bool_),
    )

    record = adapter.fuse(observation)

    assert belief_map.cell_state((0, 0)) is CellState.CONFIRMED_FREE
    assert belief_map.cell_state((1, 0)) is CellState.UNKNOWN
    assert belief_map.cell_state((2, 0)) is CellState.CONFIRMED_OCCUPIED
    assert record.free_source_cells == 1
    assert record.occupied_source_cells == 1
    assert belief_map.shortest_path((0, 0), (2, 0)) is None


def test_categorical_unknown_is_not_misclassified_as_free() -> None:
    belief_map = OnlineBeliefMap(BeliefMapConfig(cell_size_m=1.0))
    contract = _contract()
    adapter = PerceptionToBeliefMapAdapter(
        belief_map,
        contract,
        PerceptionMapAdapterConfig(
            planner_free_probability_min=0.70,
            planner_occupied_probability_max=0.10,
            planner_unknown_probability_max=0.10,
            occupied_class_probability_min=0.80,
        ),
    )
    # Channels are UNKNOWN, FREE, OCCUPIED. The middle cell has very low
    # occupied probability but remains unknown and must not enter planning.
    values = np.asarray(
        [
            [[0.02], [0.96], [0.02]],
            [[0.96], [0.03], [0.01]],
            [[0.02], [0.01], [0.97]],
        ],
        dtype=np.float32,
    )

    record = adapter.fuse(
        _observation(
            contract,
            values,
            observation_id="categorical-0001",
            tick=1,
            value_kind=OccupancyValueKind.CATEGORICAL_PROBABILITY,
        )
    )

    assert belief_map.cell_state((0, 0)) is CellState.CONFIRMED_FREE
    assert belief_map.cell_state((1, 0)) is CellState.UNKNOWN
    assert belief_map.cell_state((2, 0)) is CellState.CONFIRMED_OCCUPIED
    assert record.free_source_cells == 1
    assert record.occupied_source_cells == 1
    assert belief_map.shortest_path((0, 0), (2, 0)) is None


def test_categorical_logits_use_softmax_and_reject_bad_simplex() -> None:
    belief_map = OnlineBeliefMap(BeliefMapConfig(cell_size_m=1.0))
    contract = _contract(geometry=_geometry(height=1))
    adapter = PerceptionToBeliefMapAdapter(belief_map, contract)
    logits = np.asarray([[[-8.0]], [[8.0]], [[-8.0]]], dtype=np.float32)
    adapter.fuse(
        _observation(
            contract,
            logits,
            observation_id="categorical-logit",
            tick=1,
            value_kind=OccupancyValueKind.CATEGORICAL_LOGIT,
        )
    )
    assert belief_map.cell_state((0, 0)) is CellState.CONFIRMED_FREE

    bad_map = OnlineBeliefMap(BeliefMapConfig(cell_size_m=1.0))
    bad_adapter = PerceptionToBeliefMapAdapter(bad_map, contract)
    with pytest.raises(ValueError, match="sum to one"):
        bad_adapter.fuse(
            _observation(
                contract,
                np.ones((3, 1, 1), dtype=np.float32),
                observation_id="bad-simplex",
                tick=1,
                value_kind=OccupancyValueKind.CATEGORICAL_PROBABILITY,
            )
        )
    assert bad_map.known_cells == frozenset()


def test_unknown_ceiling_only_gates_free_not_strong_occupied_evidence() -> None:
    belief_map = OnlineBeliefMap(BeliefMapConfig(cell_size_m=1.0))
    contract = _contract(geometry=_geometry(height=1))
    adapter = PerceptionToBeliefMapAdapter(
        belief_map,
        contract,
        PerceptionMapAdapterConfig(planner_unknown_probability_max=0.05),
    )
    adapter.fuse(
        _observation(
            contract,
            np.asarray([[[0.14]], [[0.01]], [[0.85]]], dtype=np.float32),
            observation_id="occupied-despite-unknown",
            tick=1,
            value_kind=OccupancyValueKind.CATEGORICAL_PROBABILITY,
        )
    )
    assert belief_map.cell_state((0, 0)) is CellState.CONFIRMED_OCCUPIED


def test_camera_ground_grid_uses_camera_mount_geometry() -> None:
    camera = _camera(mount_xyz_m=(1.0, 0.0, 0.4))
    geometry = _geometry(
        height=1,
        reference_frame=GridReferenceFrame.CAMERA_GROUND,
    )
    contract = _contract(camera=camera, geometry=geometry)
    belief_map = OnlineBeliefMap(BeliefMapConfig(cell_size_m=1.0))
    adapter = PerceptionToBeliefMapAdapter(belief_map, contract)

    adapter.fuse(
        _observation(
            contract,
            np.zeros((1, 1), dtype=np.float32),
            observation_id="camera-offset",
            tick=1,
        )
    )

    assert belief_map.cell_state((0, 0)) is CellState.UNKNOWN
    assert belief_map.cell_state((1, 0)) is CellState.CONFIRMED_FREE


def test_rotated_cell_supercover_is_four_connected() -> None:
    geometry = _geometry(height=1, width=1)
    projected = supercover_project_cell(
        0,
        0,
        geometry=geometry,
        camera=_camera(),
        pose=_pose(1, yaw=math.pi / 4.0),
        map_cell_size_m=0.5,
    )

    assert len(projected) > 1
    reached = {next(iter(projected))}
    pending = list(reached)
    while pending:
        cell = pending.pop()
        for neighbor in four_neighbors(cell):
            if neighbor in projected and neighbor not in reached:
                reached.add(neighbor)
                pending.append(neighbor)
    assert reached == set(projected)


def test_false_free_evidence_reverses_and_immediately_blocks_routing() -> None:
    belief_map = OnlineBeliefMap(BeliefMapConfig(cell_size_m=1.0))
    contract = _contract()
    adapter = PerceptionToBeliefMapAdapter(belief_map, contract)
    free = np.zeros((3, 1), dtype=np.float32)
    middle_occupied = np.asarray([[0.5], [1.0], [0.5]], dtype=np.float32)
    middle_free = np.asarray([[0.5], [0.0], [0.5]], dtype=np.float32)

    adapter.fuse(
        _observation(
            contract,
            free,
            observation_id="free-1",
            tick=1,
        )
    )
    assert belief_map.shortest_path((0, 0), (2, 0)) is not None

    adapter.fuse(
        _observation(
            contract,
            middle_occupied,
            observation_id="occupied-1",
            tick=2,
        )
    )
    assert belief_map.cell_state((1, 0)) is CellState.CONFLICTED
    assert belief_map.shortest_path((0, 0), (2, 0)) is None

    adapter.fuse(
        _observation(
            contract,
            middle_occupied,
            observation_id="occupied-2",
            tick=3,
        )
    )
    assert belief_map.cell_state((1, 0)) is CellState.CONFIRMED_OCCUPIED

    adapter.fuse(
        _observation(
            contract,
            middle_free,
            observation_id="free-2",
            tick=4,
        )
    )
    assert belief_map.cell_state((1, 0)) is CellState.CONFLICTED
    adapter.fuse(
        _observation(
            contract,
            middle_free,
            observation_id="free-3",
            tick=5,
        )
    )
    assert belief_map.cell_state((1, 0)) is CellState.CONFIRMED_FREE
    assert belief_map.shortest_path((0, 0), (2, 0)) is not None


@pytest.mark.parametrize(
    ("values", "kind", "valid_mask", "message"),
    [
        (
            np.zeros((2, 1), dtype=np.float32),
            OccupancyValueKind.PROBABILITY,
            None,
            "shape",
        ),
        (
            np.asarray([[0.0], [1.1], [0.0]], dtype=np.float32),
            OccupancyValueKind.PROBABILITY,
            None,
            "probabilities",
        ),
        (
            np.asarray([[0.0], [np.nan], [0.0]], dtype=np.float32),
            OccupancyValueKind.LOGIT,
            None,
            "finite",
        ),
        (
            np.zeros((3, 1), dtype=np.bool_),
            OccupancyValueKind.PROBABILITY,
            None,
            "non-boolean",
        ),
        (
            np.zeros((3, 1), dtype=np.float32),
            OccupancyValueKind.PROBABILITY,
            np.ones((3, 1), dtype=np.int64),
            "boolean dtype",
        ),
    ],
)
def test_malformed_observations_are_rejected_before_map_mutation(
    values: np.ndarray,
    kind: OccupancyValueKind,
    valid_mask: np.ndarray | None,
    message: str,
) -> None:
    belief_map = OnlineBeliefMap(BeliefMapConfig(cell_size_m=1.0))
    contract = _contract()
    adapter = PerceptionToBeliefMapAdapter(belief_map, contract)
    before = belief_map.state_dict()

    with pytest.raises(ValueError, match=message):
        adapter.fuse(
            _observation(
                contract,
                values,
                observation_id="malformed",
                tick=1,
                value_kind=kind,
                valid_mask=valid_mask,
            )
        )

    assert belief_map.state_dict() == before
    assert adapter.records == ()


def test_geometry_and_backbone_mismatches_are_rejected_atomically() -> None:
    contract = _contract()
    values = np.zeros((3, 1), dtype=np.float32)

    changed_camera = replace(contract.camera, horizontal_fov_deg=91.0)
    changed_geometry = replace(contract.occupancy_geometry, cell_size_m=0.9)
    changed_provenance = replace(
        ObservationProvenance.create("wrong-backbone", contract=contract),
        backbone=ModelArtifact("other-backbone", "c" * 64),
    )
    cases = (
        _observation(
            contract,
            values,
            observation_id="camera-mismatch",
            tick=1,
            camera=changed_camera,
        ),
        _observation(
            contract,
            values,
            observation_id="geometry-mismatch",
            tick=1,
            geometry=changed_geometry,
        ),
        _observation(
            contract,
            values,
            observation_id="wrong-backbone",
            tick=1,
            provenance=changed_provenance,
        ),
    )

    for observation in cases:
        belief_map = OnlineBeliefMap(BeliefMapConfig(cell_size_m=1.0))
        adapter = PerceptionToBeliefMapAdapter(belief_map, contract)
        before = belief_map.state_dict()
        with pytest.raises(ValueError, match="geometry|backbone"):
            adapter.fuse(observation)
        assert belief_map.state_dict() == before
        assert adapter.records == ()


def test_pose_uncertainty_weights_evidence_and_enforces_limit() -> None:
    belief_map = OnlineBeliefMap(BeliefMapConfig(cell_size_m=1.0))
    contract = _contract(geometry=_geometry(height=1))
    adapter = PerceptionToBeliefMapAdapter(
        belief_map,
        contract,
        PerceptionMapAdapterConfig(maximum_pose_position_std_m=2.0),
    )
    pose = _pose(1, covariance_xy=0.5)

    record = adapter.fuse(
        _observation(
            contract,
            np.zeros((1, 1), dtype=np.float32),
            observation_id="uncertain-pose",
            tick=1,
            pose=pose,
        )
    )

    assert record.pose_position_std_m == pytest.approx(1.0)
    assert record.pose_confidence_weight == pytest.approx(0.2)
    assert belief_map.cell_state((0, 0)) is CellState.UNCERTAIN

    rejecting_map = OnlineBeliefMap(BeliefMapConfig(cell_size_m=1.0))
    rejecting_adapter = PerceptionToBeliefMapAdapter(
        rejecting_map,
        contract,
        PerceptionMapAdapterConfig(maximum_pose_position_std_m=0.9),
    )
    with pytest.raises(ValueError, match="pose position uncertainty"):
        rejecting_adapter.fuse(
            _observation(
                contract,
                np.zeros((1, 1), dtype=np.float32),
                observation_id="too-uncertain",
                tick=1,
                pose=pose,
            )
        )
    assert rejecting_map.known_cells == frozenset()


def test_duplicate_observations_and_stale_ticks_cannot_double_fuse() -> None:
    belief_map = OnlineBeliefMap(BeliefMapConfig(cell_size_m=1.0))
    contract = _contract()
    adapter = PerceptionToBeliefMapAdapter(belief_map, contract)
    observation = _observation(
        contract,
        np.zeros((3, 1), dtype=np.float32),
        observation_id="unique-frame",
        tick=2,
    )
    adapter.fuse(observation)
    after_first = belief_map.state_dict()

    with pytest.raises(ValueError, match="duplicate"):
        adapter.fuse(observation)
    assert belief_map.state_dict() == after_first
    assert len(adapter.records) == 1

    stale = _observation(
        contract,
        np.zeros((3, 1), dtype=np.float32),
        observation_id="stale-frame",
        tick=1,
    )
    with pytest.raises(ValueError, match="older"):
        adapter.fuse(stale)
    assert belief_map.state_dict() == after_first


def test_underinflated_geometry_is_rejected_at_adapter_construction() -> None:
    geometry = replace(_geometry(), body_inflation_radius_m=0.05)
    contract = _contract(geometry=geometry)
    belief_map = OnlineBeliefMap(BeliefMapConfig(cell_size_m=1.0))

    with pytest.raises(ValueError, match="inflated"):
        PerceptionToBeliefMapAdapter(
            belief_map,
            contract,
            PerceptionMapAdapterConfig(
                minimum_body_inflation_radius_m=0.20,
            ),
        )


def test_contract_rejects_mismatched_global_map_resolution() -> None:
    contract = _contract()
    belief_map = OnlineBeliefMap(BeliefMapConfig(cell_size_m=0.5))

    with pytest.raises(ValueError, match="cell size"):
        PerceptionToBeliefMapAdapter(belief_map, contract)


def test_malformed_typed_wrapper_is_rejected_before_dereference() -> None:
    contract = _contract()
    belief_map = OnlineBeliefMap(BeliefMapConfig(cell_size_m=1.0))
    adapter = PerceptionToBeliefMapAdapter(belief_map, contract)
    valid = _observation(
        contract,
        np.zeros((3, 1), dtype=np.float32),
        observation_id="missing-pose",
        tick=1,
    )
    malformed = replace(valid, pose=None)  # type: ignore[arg-type]

    with pytest.raises(TypeError, match="PoseBelief"):
        adapter.fuse(malformed)
    assert belief_map.known_cells == frozenset()
