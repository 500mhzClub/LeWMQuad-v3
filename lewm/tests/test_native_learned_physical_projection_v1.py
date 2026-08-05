from __future__ import annotations

import copy
import hashlib
import math
from pathlib import Path

import pytest

from lewm.planning import native_learned_physical_projection_v1 as module
from lewm.planning.native_learned_physical_projection_v1 import (
    FROZEN_SYNTHETIC_NATIVE_CALIBRATION_V1,
    NativeLearnedPhysicalProjectionAdapterV1,
    NativeLearnedProjectionBindingError,
    NativeLearnedProjectionRejectedError,
    NativeLearnedProjectionReplayError,
    NativeV4SourceGeometryV1,
    RawGroundClearCellQueriesV1,
    RawOrderedRayHitDepthV1,
    SyntheticNativeV4RunnerV1,
    canonical_ground_query_xy_body_v1,
    require_production_native_learned_projection_adapter,
)
from lewm.planning.revisioned_physical_configuration_memory import (
    MapFrameIdentity,
    PhysicalLabel,
    PhysicalMemoryConfig,
    PoseProvenance,
    PoseSource,
    RevisionedPhysicalMemory,
    SnapshotBindingError,
)
from lewm.planning.two_resolution_configuration_projection_v2 import (
    PROFILE_SHA256,
    TwoResolutionConfigurationProjectionV2,
)


def _hash(label: str) -> str:
    return hashlib.sha256(label.encode("ascii")).hexdigest()


IDENTITIES = {
    "runner_execution_identity_sha256": _hash("synthetic-v4-runner"),
    "inference_implementation_sha256": _hash("synthetic-v4-inference"),
    "projection_implementation_sha256": _hash("synthetic-v4-projection"),
    "access_ledger_source_sha256": _hash("synthetic-v4-ledger"),
    "checkpoint_file_sha256": _hash("synthetic-v5-checkpoint"),
    "g2_report_file_sha256": _hash("synthetic-passed-g2-report"),
}
CAMERA_TRANSFORM_SHA256 = _hash("synthetic-camera-transform")
CALIBRATION = FROZEN_SYNTHETIC_NATIVE_CALIBRATION_V1
GEOMETRY = NativeV4SourceGeometryV1(
    shape=CALIBRATION.source_shape,
    origin_forward_left_m=CALIBRATION.source_origin_forward_left_m,
    cell_size_m=CALIBRATION.source_cell_size_m,
)
SOURCE_CELL = (20, 64)  # exact local square [0,.05] x [0,.05]


def _stack(*, origin: tuple[float, float] = (-1.6, -1.6)):
    physical_frame = MapFrameIdentity(
        session_id=f"native-physical:{origin}",
        origin_xy_m=origin,
        cell_size_m=0.05,
        frame_id="native_physical",
    )
    configuration_frame = MapFrameIdentity(
        session_id=f"native-configuration:{origin}",
        origin_xy_m=origin,
        cell_size_m=0.10,
        frame_id="native_configuration",
    )
    memory = RevisionedPhysicalMemory(
        PhysicalMemoryConfig(
            map_frame=physical_frame,
            require_registered_lattice=False,
            physical_projection_contract_sha256=PROFILE_SHA256,
            expected_camera_transform_sha256=CAMERA_TRANSFORM_SHA256,
            pose_covariance_diagonal_limits=CALIBRATION.covariance_diagonal_max,
            promoted_runtime=False,
        )
    )
    projection = TwoResolutionConfigurationProjectionV2(
        memory,
        configuration_map_frame=configuration_frame,
        physical_shape=(64, 64),
        configuration_shape=(32, 32),
    )
    snapshot = projection.project()
    runner = SyntheticNativeV4RunnerV1(
        **IDENTITIES,
        calibration_sha256=CALIBRATION.content_sha256,
        _synthetic_test_fixture=True,
    )
    adapter = NativeLearnedPhysicalProjectionAdapterV1(
        memory=memory,
        projection=projection,
        runner=runner,
        calibration=CALIBRATION,
        **IDENTITIES,
        camera_transform_sha256=CAMERA_TRANSFORM_SHA256,
        _synthetic_test_fixture=True,
    )
    return memory, projection, snapshot, runner, adapter


def _pose(
    *,
    yaw: float = 0.0,
    translation: tuple[float, float] = (0.0, 0.0),
    covariance: tuple[float, float, float] = (0.0, 0.0, 0.0),
    timestamp: int = 1,
) -> PoseProvenance:
    return PoseProvenance(
        source=PoseSource.DEPLOYMENT_ODOMETRY,
        frame_id="native_physical",
        mean_xy_yaw=(translation[0], translation[1], yaw),
        covariance_xy_yaw=(
            (covariance[0], 0.0, 0.0),
            (0.0, covariance[1], 0.0),
            (0.0, 0.0, covariance[2]),
        ),
        timestamp_ns=timestamp,
        synchronization_id=f"sync:{timestamp}",
        camera_transform_sha256=CAMERA_TRANSFORM_SHA256,
    )


def _ground_row(*, logit: float = 10.0) -> RawGroundClearCellQueriesV1:
    query = canonical_ground_query_xy_body_v1(GEOMETRY, SOURCE_CELL)
    return RawGroundClearCellQueriesV1(
        source_cell=SOURCE_CELL,
        clear_to_target_logits=(logit,) * 5,
        query_in_frustum=(True,) * 5,
        query_xy_body_m=query,
        query_uv_px=((10.0, 10.0),) * 5,
        target_distance_m=(1.0,) * 5,
    )


def _center_hit() -> RawOrderedRayHitDepthV1:
    return RawOrderedRayHitDepthV1(
        ray_index=0,
        ray_origin_xy_body_m=(-0.025, 0.025),
        ray_direction_xy_body=(1.0, 0.0),
        ordered_hit_logits=(10.0,),
        ordered_depth_m=(0.05,),
    )


def _boundary_hit() -> RawOrderedRayHitDepthV1:
    return RawOrderedRayHitDepthV1(
        ray_index=0,
        ray_origin_xy_body_m=(0.0, 0.025),
        ray_direction_xy_body=(1.0, 0.0),
        ordered_hit_logits=(10.0,),
        ordered_depth_m=(0.05,),
    )


def _outcome(
    runner: SyntheticNativeV4RunnerV1,
    snapshot,
    *,
    pose: PoseProvenance | None = None,
    ground: tuple[RawGroundClearCellQueriesV1, ...] = (),
    rays: tuple[RawOrderedRayHitDepthV1, ...] = (),
    geometry: NativeV4SourceGeometryV1 = GEOMETRY,
    source_derivation: str = "native_raw_v4_0p05",
    sequence: int = 1,
):
    return runner.issue(
        snapshot=snapshot,
        pose=pose or _pose(timestamp=sequence),
        source_geometry=geometry,
        ground_clear_query_tensor=ground,
        ordered_ray_hit_depth_tensor=rays,
        rgb_frame_id=f"rgb:{sequence}",
        rgb_frame_sha256=_hash(f"rgb:{sequence}"),
        raw_outcome_file_sha256=_hash(f"raw-outcome:{sequence}"),
        source_derivation=source_derivation,
    )


def test_native_free_closed_square_commit_and_authority_receipts() -> None:
    memory, _projection, snapshot, runner, adapter = _stack()
    outcome = _outcome(runner, snapshot, ground=(_ground_row(),))
    package = adapter.issue(snapshot, outcome)

    assert package.projection_receipt.free_cells == frozenset({(32, 32)})
    assert not package.projection_receipt.occupied_cells
    assert package.development_only is True
    assert package.hardware_execution_authorized is False
    assert package.production_promotion_authorized is False
    assert package.admission.checkpoint_file_sha256 == IDENTITIES["checkpoint_file_sha256"]
    assert package.admission.g2_report_file_sha256 == IDENTITIES["g2_report_file_sha256"]
    encoded = package.to_dict()
    assert encoded["development_only"] is True
    assert encoded["hardware_execution_authorized"] is False
    assert encoded["production_promotion_authorized"] is False
    assert not hasattr(package, "transaction")

    receipt = adapter.commit(package)
    assert receipt.revision_before == 0 and receipt.revision_after == 1
    assert memory.physical_state((32, 32)) is PhysicalLabel.FREE
    assert memory.learned_observation_ids == frozenset({package.observation_id})


def test_translation_and_rotation_are_origin_aware_and_not_center_upsampling() -> None:
    _memory, _projection, snapshot, runner, adapter = _stack()
    rotated = _outcome(
        runner,
        snapshot,
        pose=_pose(yaw=math.pi / 2.0),
        ground=(_ground_row(),),
        sequence=1,
    )
    rotated_package = adapter.issue(snapshot, rotated)
    assert rotated_package.projection_receipt.free_cells == frozenset({(31, 32)})

    memory2, _projection2, snapshot2, runner2, adapter2 = _stack()
    translated = _outcome(
        runner2,
        snapshot2,
        pose=_pose(translation=(0.025, 0.0)),
        ground=(_ground_row(),),
        sequence=2,
    )
    translated_package = adapter2.issue(snapshot2, translated)
    assert translated_package.projection_receipt.free_cells == frozenset()
    assert (32, 32) in translated_package.projection_receipt.unknown_cells
    adapter2.commit(translated_package)
    assert memory2.physical_state((32, 32)) is PhysicalLabel.UNKNOWN


def test_occupied_hit_uses_closed_union_supercover_for_boundary_and_rotation() -> None:
    _memory, _projection, snapshot, runner, adapter = _stack()
    boundary = _outcome(runner, snapshot, rays=(_boundary_hit(),))
    package = adapter.issue(snapshot, boundary)
    assert package.projection_receipt.occupied_cells == frozenset(
        {(32, 32), (33, 32)}
    )

    _memory2, _projection2, snapshot2, runner2, adapter2 = _stack()
    rotated = _outcome(
        runner2,
        snapshot2,
        pose=_pose(yaw=math.pi / 2.0),
        rays=(_boundary_hit(),),
        sequence=2,
    )
    rotated_package = adapter2.issue(snapshot2, rotated)
    assert rotated_package.projection_receipt.occupied_cells == frozenset(
        {(31, 32), (31, 33)}
    )


def test_occupied_precedes_free() -> None:
    _memory, _projection, snapshot, runner, adapter = _stack()
    outcome = _outcome(
        runner,
        snapshot,
        ground=(_ground_row(),),
        rays=(_center_hit(),),
    )
    package = adapter.issue(snapshot, outcome)
    assert package.projection_receipt.occupied_cells == frozenset({(32, 32)})
    assert (32, 32) not in package.projection_receipt.free_cells


def test_covariance_limit_is_conservative_and_excess_rejects() -> None:
    _memory, _projection, snapshot, runner, adapter = _stack()
    at_limit = _outcome(
        runner,
        snapshot,
        pose=_pose(covariance=(0.0025, 0.0, 0.0)),
        ground=(_ground_row(),),
    )
    package = adapter.issue(snapshot, at_limit)
    assert len(package.projection_receipt.transform_uncertainty_set) == 3
    assert not package.projection_receipt.free_cells

    _memory2, _projection2, snapshot2, runner2, adapter2 = _stack()
    excessive = _outcome(
        runner2,
        snapshot2,
        pose=_pose(covariance=(0.0025001, 0.0, 0.0)),
        ground=(_ground_row(),),
    )
    with pytest.raises(NativeLearnedProjectionRejectedError, match="covariance"):
        adapter2.issue(snapshot2, excessive)


@pytest.mark.parametrize(
    ("cell_size", "derivation"),
    [(0.10, "native_raw_v4_0p05"), (0.05, "upsampled_from_v4_0p10")],
)
def test_derived_point_one_metre_source_or_upsampling_rejects(
    cell_size: float,
    derivation: str,
) -> None:
    _memory, _projection, snapshot, runner, adapter = _stack()
    geometry = NativeV4SourceGeometryV1(
        shape=CALIBRATION.source_shape,
        origin_forward_left_m=CALIBRATION.source_origin_forward_left_m,
        cell_size_m=cell_size,
    )
    query = canonical_ground_query_xy_body_v1(geometry, SOURCE_CELL)
    row = RawGroundClearCellQueriesV1(
        source_cell=SOURCE_CELL,
        clear_to_target_logits=(10.0,) * 5,
        query_in_frustum=(True,) * 5,
        query_xy_body_m=query,
        query_uv_px=((1.0, 1.0),) * 5,
        target_distance_m=(1.0,) * 5,
    )
    outcome = _outcome(
        runner,
        snapshot,
        ground=(row,),
        geometry=geometry,
        source_derivation=derivation,
    )
    with pytest.raises(NativeLearnedProjectionRejectedError, match="0.10 m/upsampling"):
        adapter.issue(snapshot, outcome)


def test_wrong_query_geometry_rejects_without_caller_labels() -> None:
    _memory, _projection, snapshot, runner, adapter = _stack()
    row = _ground_row()
    bad = RawGroundClearCellQueriesV1(
        source_cell=row.source_cell,
        clear_to_target_logits=row.clear_to_target_logits,
        query_in_frustum=row.query_in_frustum,
        query_xy_body_m=((0.001, 0.0), *row.query_xy_body_m[1:]),
        query_uv_px=row.query_uv_px,
        target_distance_m=row.target_distance_m,
    )
    outcome = _outcome(runner, snapshot, ground=(bad,))
    with pytest.raises(NativeLearnedProjectionRejectedError, match="query geometry"):
        adapter.issue(snapshot, outcome)
    assert "labels" not in outcome.to_dict()
    assert "metrics" not in outcome.to_dict()


def test_wrong_origin_rejects_even_for_a_legitimately_issued_outcome() -> None:
    _memory, _projection, snapshot, runner, adapter = _stack()
    _other_memory, _other_projection, other_snapshot, _, _ = _stack(
        origin=(-1.55, -1.6)
    )
    wrong_origin = _outcome(runner, other_snapshot, ground=(_ground_row(),))
    with pytest.raises(NativeLearnedProjectionBindingError, match="origin/shape"):
        adapter.issue(snapshot, wrong_origin)


@pytest.mark.parametrize(
    "field",
    (
        "source_derivation",
        "runner_execution_identity_sha256",
        "inference_implementation_sha256",
        "projection_implementation_sha256",
        "access_ledger_source_sha256",
        "checkpoint_file_sha256",
        "g2_report_file_sha256",
        "calibration_sha256",
        "rgb_frame_id",
        "rgb_frame_sha256",
        "raw_outcome_file_sha256",
        "pose",
        "physical_map_frame",
        "configuration_map_frame",
        "physical_shape",
        "configuration_shape",
        "physical_revision",
        "configuration_revision",
        "physical_content_sha256",
        "configuration_snapshot_sha256",
        "projection_source_sha256",
    ),
)
def test_instance_issued_outcome_content_is_immutable(field: str) -> None:
    _memory, _projection, snapshot, runner, adapter = _stack()
    outcome = _outcome(runner, snapshot, ground=(_ground_row(),))
    replacements: dict[str, object] = {
        "source_derivation": "native_raw_v4_rewritten",
        "runner_execution_identity_sha256": _hash("wrong-runner"),
        "inference_implementation_sha256": _hash("wrong-inference"),
        "projection_implementation_sha256": _hash("wrong-projection"),
        "access_ledger_source_sha256": _hash("wrong-ledger"),
        "checkpoint_file_sha256": _hash("wrong-checkpoint"),
        "g2_report_file_sha256": _hash("wrong-g2-report"),
        "calibration_sha256": _hash("wrong-calibration"),
        "rgb_frame_id": "rgb:rewritten",
        "rgb_frame_sha256": _hash("wrong-rgb"),
        "raw_outcome_file_sha256": _hash("wrong-raw-outcome"),
        "pose": _pose(translation=(0.05, 0.0), timestamp=99),
        "physical_map_frame": MapFrameIdentity(
            session_id="wrong-physical-frame",
            origin_xy_m=(-1.6, -1.6),
            cell_size_m=0.05,
            frame_id="native_physical",
        ),
        "configuration_map_frame": MapFrameIdentity(
            session_id="wrong-configuration-frame",
            origin_xy_m=(-1.6, -1.6),
            cell_size_m=0.10,
            frame_id="native_configuration",
        ),
        "physical_shape": (63, 64),
        "configuration_shape": (31, 32),
        "physical_revision": 99,
        "configuration_revision": 99,
        "physical_content_sha256": _hash("wrong-physical-content"),
        "configuration_snapshot_sha256": _hash("wrong-configuration-snapshot"),
        "projection_source_sha256": _hash("wrong-projection-source"),
    }
    object.__setattr__(outcome, field, replacements[field])
    object.__setattr__(
        outcome,
        "content_sha256",
        module._sha256(outcome.to_dict(False)),
    )

    with pytest.raises(
        NativeLearnedProjectionBindingError,
        match="instance-issued content",
    ):
        adapter.issue(snapshot, outcome)


def test_rehashed_mutated_development_transaction_rejects() -> None:
    _memory, _projection, snapshot, runner, adapter = _stack()
    outcome = _outcome(runner, snapshot, ground=(_ground_row(),))
    package = adapter.issue(snapshot, outcome)
    object.__setattr__(package, "pose", _pose(timestamp=99))
    object.__setattr__(
        package,
        "content_sha256",
        module._sha256(package.to_dict(False)),
    )

    with pytest.raises(
        NativeLearnedProjectionBindingError,
        match="differs from its issued content",
    ):
        adapter.commit(package)


def test_copy_replay_stale_and_transaction_transfer_reject() -> None:
    _memory, projection, snapshot, runner, adapter = _stack()
    outcome = _outcome(runner, snapshot, ground=(_ground_row(),))
    with pytest.raises(NativeLearnedProjectionBindingError, match="exact live"):
        adapter.issue(snapshot, copy.copy(outcome))
    package = adapter.issue(snapshot, outcome)
    with pytest.raises(NativeLearnedProjectionReplayError, match="already consumed"):
        adapter.issue(snapshot, outcome)
    with pytest.raises(TypeError, match="non-copyable"):
        copy.copy(package)
    with pytest.raises(TypeError, match="non-copyable"):
        copy.copy(adapter)

    foreign = NativeLearnedPhysicalProjectionAdapterV1(
        memory=_memory,
        projection=projection,
        runner=runner,
        calibration=CALIBRATION,
        **IDENTITIES,
        camera_transform_sha256=CAMERA_TRANSFORM_SHA256,
        _synthetic_test_fixture=True,
    )
    with pytest.raises(NativeLearnedProjectionBindingError, match="exact live"):
        foreign.commit(package)

    projection.project()
    with pytest.raises(SnapshotBindingError):
        adapter.commit(package)


def test_contradiction_becomes_unknown_and_exact_retraction_restores_free() -> None:
    memory, projection, snapshot, runner, adapter = _stack()
    free_outcome = _outcome(runner, snapshot, ground=(_ground_row(),), sequence=1)
    free_package = adapter.issue(snapshot, free_outcome)
    adapter.commit(free_package)
    assert memory.physical_state((32, 32)) is PhysicalLabel.FREE

    snapshot2 = projection.project()
    occupied_outcome = _outcome(
        runner,
        snapshot2,
        pose=_pose(timestamp=2),
        rays=(_center_hit(),),
        sequence=2,
    )
    occupied_package = adapter.issue(snapshot2, occupied_outcome)
    adapter.commit(occupied_package)
    assert memory.physical_state((32, 32)) is PhysicalLabel.UNKNOWN

    snapshot3 = projection.project()
    retraction = adapter.issue_retraction(snapshot3, occupied_package)
    assert retraction.retracts_observation_id == occupied_package.observation_id
    receipt = adapter.commit(retraction)
    assert receipt.learned_observations_retracted == 1
    assert memory.physical_state((32, 32)) is PhysicalLabel.FREE
    assert occupied_package.observation_id not in memory.learned_observation_ids
    assert free_package.observation_id in memory.learned_observation_ids
    with pytest.raises(NativeLearnedProjectionReplayError):
        adapter.commit(retraction)


def test_production_is_unconfigured_and_source_has_no_accelerator_or_io_surface() -> None:
    assert (
        module.PRODUCTION_NATIVE_V4_RUNNER,
        module.PRODUCTION_V4_CHECKPOINT_FILE_SHA256,
        module.PRODUCTION_G2_REPORT_FILE_SHA256,
        module.PRODUCTION_V4_CALIBRATION_SHA256,
        module.PRODUCTION_NATIVE_LEARNED_PROJECTION_ADAPTER,
    ) == (None,) * 5
    with pytest.raises(PermissionError, match="identities are unset"):
        require_production_native_learned_projection_adapter()
    with pytest.raises(PermissionError, match="synthetic-only"):
        SyntheticNativeV4RunnerV1(
            **IDENTITIES,
            calibration_sha256=CALIBRATION.content_sha256,
        )
    source = Path(module.__file__).read_text(encoding="utf-8").lower()
    for forbidden in (
        "import torch",
        "import numpy",
        "cuda",
        "rocm",
        "heldout",
        "open(",
        "read_bytes",
        "load_state_dict",
    ):
        assert forbidden not in source
