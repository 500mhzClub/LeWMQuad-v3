"""Focused lifecycle and preserved-contract tests for the additive V2 adapter."""
from __future__ import annotations

import copy
import math
from pathlib import Path
import pickle

import pytest

from lewm.planning import native_learned_physical_projection_v1 as v1_module
from lewm.planning import native_learned_physical_projection_v2 as v2_module
from lewm.planning.native_learned_physical_projection_v1 import (
    NativeLearnedPhysicalProjectionAdapterV1,
    NativeLearnedProjectionBindingError,
    NativeLearnedProjectionRejectedError,
    NativeLearnedProjectionReplayError,
    NativeV4SourceGeometryV1,
)
from lewm.planning.native_learned_physical_projection_v2 import (
    NativeLearnedPhysicalProjectionAdapterV2,
    require_production_native_learned_projection_adapter_v2,
)
from lewm.planning.revisioned_physical_configuration_memory import (
    PhysicalLabel,
    SnapshotBindingError,
)
from lewm.tests.test_native_learned_physical_projection_v1 import (
    CALIBRATION,
    CAMERA_TRANSFORM_SHA256,
    IDENTITIES,
    _boundary_hit,
    _center_hit,
    _ground_row,
    _outcome,
    _pose,
    _stack,
)


def _stack_v2(*, origin: tuple[float, float] = (-1.6, -1.6)):
    memory, projection, snapshot, runner, _v1_adapter = _stack(origin=origin)
    adapter = NativeLearnedPhysicalProjectionAdapterV2(
        memory=memory,
        projection=projection,
        runner=runner,
        calibration=CALIBRATION,
        **IDENTITIES,
        camera_transform_sha256=CAMERA_TRANSFORM_SHA256,
        _synthetic_test_fixture=True,
    )
    return memory, projection, snapshot, runner, adapter


def _commit_free_target():
    memory, projection, snapshot, runner, adapter = _stack_v2()
    outcome = _outcome(runner, snapshot, ground=(_ground_row(),))
    target = adapter.issue(snapshot, outcome)
    adapter.commit(target)
    return memory, projection, runner, adapter, target


def _mutate_and_rehash_committed_target(target) -> None:
    replacement_pose = _pose(timestamp=99)
    object.__setattr__(target, "pose", replacement_pose)
    object.__setattr__(
        target.admission,
        "pose_sha256",
        replacement_pose.content_sha256,
    )
    object.__setattr__(
        target.admission,
        "content_sha256",
        v1_module._sha256(target.admission.to_dict(False)),
    )
    object.__setattr__(
        target,
        "content_sha256",
        v1_module._sha256(target.to_dict(False)),
    )


def _advance_memory(
    projection,
    snapshot,
    runner,
    adapter,
    *,
    sequence: int = 2,
) -> None:
    outcome = _outcome(
        runner,
        snapshot,
        pose=_pose(timestamp=sequence),
        rays=(_center_hit(),),
        sequence=sequence,
    )
    package = adapter.issue(snapshot, outcome)
    adapter.commit(package)


def test_v2_retraction_rejects_rehashed_post_commit_target_mutation() -> None:
    _memory, projection, _runner, adapter, target = _commit_free_target()
    _mutate_and_rehash_committed_target(target)

    current = projection.project()
    with pytest.raises(
        NativeLearnedProjectionBindingError,
        match="issued content",
    ):
        adapter.issue_retraction(current, target)


def test_v2_rechecks_target_digest_immediately_before_removal() -> None:
    memory, projection, _runner, adapter, target = _commit_free_target()
    current = projection.project()
    retraction = adapter.issue_retraction(current, target)
    _mutate_and_rehash_committed_target(target)

    with pytest.raises(
        NativeLearnedProjectionBindingError,
        match="issued content",
    ):
        adapter.commit(retraction)
    assert target.observation_id in memory.learned_observation_ids


def test_v2_rejects_a_second_concurrent_live_retraction() -> None:
    _memory, projection, _runner, adapter, target = _commit_free_target()
    current = projection.project()
    first = adapter.issue_retraction(current, target)

    with pytest.raises(
        NativeLearnedProjectionReplayError,
        match="exact live retraction",
    ):
        adapter.issue_retraction(current, target)

    receipt = adapter.commit(first)
    assert receipt.learned_observations_retracted == 1
    with pytest.raises(
        NativeLearnedProjectionReplayError,
        match="already consumed",
    ):
        adapter.commit(first)


def test_v2_proactively_invalidates_stale_package_and_allows_fresh_retry() -> None:
    memory, projection, runner, adapter, target = _commit_free_target()
    shared = projection.project()
    stale = adapter.issue_retraction(shared, target)
    _advance_memory(projection, shared, runner, adapter)

    current = projection.project()
    replacement = adapter.issue_retraction(current, target)
    with pytest.raises(
        NativeLearnedProjectionReplayError,
        match="terminally stale",
    ):
        adapter.commit(stale)

    receipt = adapter.commit(replacement)
    assert receipt.learned_observations_retracted == 1
    assert target.observation_id not in memory.learned_observation_ids
    with pytest.raises(NativeLearnedProjectionReplayError):
        adapter.commit(stale)


def test_v2_failed_stale_commit_releases_target_for_exact_retry() -> None:
    memory, projection, runner, adapter, target = _commit_free_target()
    shared = projection.project()
    stale = adapter.issue_retraction(shared, target)
    _advance_memory(projection, shared, runner, adapter)

    with pytest.raises(SnapshotBindingError):
        adapter.commit(stale)
    assert target.observation_id in memory.learned_observation_ids

    current = projection.project()
    replacement = adapter.issue_retraction(current, target)
    receipt = adapter.commit(replacement)
    assert receipt.learned_observations_retracted == 1
    assert target.observation_id not in memory.learned_observation_ids
    with pytest.raises(
        NativeLearnedProjectionReplayError,
        match="terminally stale",
    ):
        adapter.commit(stale)


def test_v2_rejects_rehashed_mutation_of_live_retraction_package() -> None:
    memory, projection, _runner, adapter, target = _commit_free_target()
    current = projection.project()
    retraction = adapter.issue_retraction(current, target)
    object.__setattr__(retraction, "pose", _pose(timestamp=77))
    object.__setattr__(
        retraction,
        "content_sha256",
        v1_module._sha256(retraction.to_dict(False)),
    )

    with pytest.raises(
        NativeLearnedProjectionBindingError,
        match="issued content",
    ):
        adapter.commit(retraction)
    assert target.observation_id in memory.learned_observation_ids


def test_v2_retraction_copy_reload_and_transfer_do_not_consume_live_package() -> None:
    memory, projection, runner, adapter, target = _commit_free_target()
    current = projection.project()
    retraction = adapter.issue_retraction(current, target)

    with pytest.raises(TypeError, match="non-copyable"):
        copy.copy(retraction)
    reloaded = pickle.loads(pickle.dumps(retraction))
    with pytest.raises(
        NativeLearnedProjectionBindingError,
        match="no exact V2 reservation",
    ):
        adapter.commit(reloaded)

    foreign = NativeLearnedPhysicalProjectionAdapterV2(
        memory=memory,
        projection=projection,
        runner=runner,
        calibration=CALIBRATION,
        **IDENTITIES,
        camera_transform_sha256=CAMERA_TRANSFORM_SHA256,
        _synthetic_test_fixture=True,
    )
    with pytest.raises(
        NativeLearnedProjectionBindingError,
        match="no exact V2 reservation",
    ):
        foreign.commit(retraction)

    receipt = adapter.commit(retraction)
    assert receipt.learned_observations_retracted == 1
    assert target.observation_id not in memory.learned_observation_ids


def test_v2_preserves_free_geometry_hidden_transaction_and_false_authority() -> None:
    memory, _projection, snapshot, runner, adapter = _stack_v2()
    outcome = _outcome(runner, snapshot, ground=(_ground_row(),))
    package = adapter.issue(snapshot, outcome)

    assert package.projection_receipt.free_cells == frozenset({(32, 32)})
    assert not package.projection_receipt.occupied_cells
    assert not hasattr(package, "transaction")
    for surface in (adapter, package.admission, package.projection_receipt, package):
        assert surface.development_only is True
        assert surface.hardware_execution_authorized is False
        assert surface.production_promotion_authorized is False

    adapter.commit(package)
    assert memory.physical_state((32, 32)) is PhysicalLabel.FREE


def test_v2_preserves_origin_aware_rotation_and_closed_occupied_supercover() -> None:
    _memory, _projection, snapshot, runner, adapter = _stack_v2()
    rotated = _outcome(
        runner,
        snapshot,
        pose=_pose(yaw=math.pi / 2.0),
        ground=(_ground_row(),),
        sequence=1,
    )
    assert adapter.issue(
        snapshot,
        rotated,
    ).projection_receipt.free_cells == frozenset({(31, 32)})

    _memory2, _projection2, snapshot2, runner2, adapter2 = _stack_v2()
    boundary = _outcome(
        runner2,
        snapshot2,
        rays=(_boundary_hit(),),
        sequence=2,
    )
    assert adapter2.issue(
        snapshot2,
        boundary,
    ).projection_receipt.occupied_cells == frozenset({(32, 32), (33, 32)})


def test_v2_preserves_covariance_and_native_resolution_rejection() -> None:
    _memory, _projection, snapshot, runner, adapter = _stack_v2()
    excessive = _outcome(
        runner,
        snapshot,
        pose=_pose(covariance=(0.0026, 0.0, 0.0)),
        ground=(_ground_row(),),
        sequence=1,
    )
    with pytest.raises(NativeLearnedProjectionRejectedError, match="covariance"):
        adapter.issue(snapshot, excessive)

    _memory2, _projection2, snapshot2, runner2, adapter2 = _stack_v2()
    derived = NativeV4SourceGeometryV1(
        shape=(64, 64),
        origin_forward_left_m=CALIBRATION.source_origin_forward_left_m,
        cell_size_m=0.10,
    )
    outcome = _outcome(
        runner2,
        snapshot2,
        ground=(_ground_row(),),
        geometry=derived,
        source_derivation="derived_0p10_upsampled",
        sequence=2,
    )
    with pytest.raises(
        NativeLearnedProjectionRejectedError,
        match="0.10 m/upsampling",
    ):
        adapter2.issue(snapshot2, outcome)


def test_v2_preserves_copy_transfer_replay_stale_and_forgery_rejection() -> None:
    _memory, projection, snapshot, runner, adapter = _stack_v2()
    outcome = _outcome(runner, snapshot, ground=(_ground_row(),))
    reloaded_outcome = pickle.loads(pickle.dumps(outcome))
    with pytest.raises(NativeLearnedProjectionBindingError, match="exact live"):
        adapter.issue(snapshot, reloaded_outcome)

    package = adapter.issue(snapshot, outcome)
    with pytest.raises(TypeError, match="non-copyable"):
        copy.copy(package)
    with pytest.raises(TypeError, match="non-copyable"):
        copy.copy(adapter)

    reloaded_package = pickle.loads(pickle.dumps(package))
    with pytest.raises(NativeLearnedProjectionBindingError, match="exact live"):
        adapter.commit(reloaded_package)
    forged = object.__new__(type(package))
    for name, value in package.__dict__.items():
        object.__setattr__(forged, name, value)
    with pytest.raises(NativeLearnedProjectionBindingError, match="exact live"):
        adapter.commit(forged)

    foreign = NativeLearnedPhysicalProjectionAdapterV2(
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


def test_v2_production_is_unset_and_source_has_no_accelerator_or_io_surface() -> None:
    _memory, _projection, _snapshot, _runner, adapter = _stack_v2()
    assert not isinstance(adapter, NativeLearnedPhysicalProjectionAdapterV1)
    assert (
        v2_module.PRODUCTION_NATIVE_V4_RUNNER,
        v2_module.PRODUCTION_V4_CHECKPOINT_FILE_SHA256,
        v2_module.PRODUCTION_G2_REPORT_FILE_SHA256,
        v2_module.PRODUCTION_V4_CALIBRATION_SHA256,
        v2_module.PRODUCTION_NATIVE_LEARNED_PROJECTION_ADAPTER_V2,
    ) == (None,) * 5
    with pytest.raises(PermissionError, match="identities are unset"):
        require_production_native_learned_projection_adapter_v2()

    source = Path(v2_module.__file__).read_text(encoding="utf-8").lower()
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
