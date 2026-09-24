"""Focused standalone-authority and lifecycle tests for projection V4."""
from __future__ import annotations

import copy
import gc
import math
from pathlib import Path
import pickle

import pytest

from lewm.planning import native_learned_physical_projection_v1 as v1_module
from lewm.planning import native_learned_physical_projection_v4 as v4_module
from lewm.planning.native_learned_physical_projection_v1 import (
    NativeLearnedPhysicalProjectionAdapterV1,
    NativeLearnedProjectionBindingError,
    NativeLearnedProjectionRejectedError,
    NativeLearnedProjectionReplayError,
    NativeV4SourceGeometryV1,
    SyntheticNativeV4RunnerV1,
)
from lewm.planning.native_learned_physical_projection_v2 import (
    NativeLearnedPhysicalProjectionAdapterV2,
)
from lewm.planning.native_learned_physical_projection_v3 import (
    NativeLearnedPhysicalProjectionAdapterV3,
)
from lewm.planning.native_learned_physical_projection_v4 import (
    NativeLearnedPhysicalProjectionAdapterV4,
    QualifiedLearnedPhysicalDevelopmentTransactionV4,
    require_production_native_learned_projection_adapter_v4,
)
from lewm.planning.revisioned_physical_configuration_memory import (
    EvidenceAuthority,
    MapFrameIdentity,
    ObservationIdentity,
    PhysicalCellEvidence,
    PhysicalEvidenceTransaction,
    PhysicalLabel,
    PhysicalMemoryConfig,
    RevisionedPhysicalMemory,
    SnapshotBindingError,
    TransactionRejectedError,
)
from lewm.planning.two_resolution_configuration_projection_v2 import (
    PROFILE_SHA256,
    TwoResolutionConfigurationProjectionV2,
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
)


def _stack_v4(*, origin: tuple[float, float] = (-1.6, -1.6)):
    physical_frame = MapFrameIdentity(
        session_id=f"native-v4-physical:{origin}",
        origin_xy_m=origin,
        cell_size_m=0.05,
        frame_id="native_physical",
    )
    configuration_frame = MapFrameIdentity(
        session_id=f"native-v4-configuration:{origin}",
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
    adapter = NativeLearnedPhysicalProjectionAdapterV4(
        memory=memory,
        projection=projection,
        runner=runner,
        calibration=CALIBRATION,
        **IDENTITIES,
        camera_transform_sha256=CAMERA_TRANSFORM_SHA256,
        _synthetic_test_fixture=True,
    )
    return memory, projection, snapshot, runner, adapter


def _committed_projection_v4():
    memory, projection, snapshot, runner, adapter = _stack_v4()
    package = adapter.issue(
        snapshot,
        _outcome(runner, snapshot, ground=(_ground_row(),)),
    )
    adapter.commit(package)
    return memory, projection, runner, adapter, package


def _mutate_and_rehash_v4(package, *, timestamp: int) -> None:
    replacement_pose = _pose(timestamp=timestamp)
    object.__setattr__(package, "pose", replacement_pose)
    object.__setattr__(
        package.admission,
        "pose_sha256",
        replacement_pose.content_sha256,
    )
    object.__setattr__(
        package.admission,
        "content_sha256",
        v1_module._sha256(package.admission.to_dict(False)),
    )
    object.__setattr__(
        package,
        "content_sha256",
        v1_module._sha256(package.to_dict(False)),
    )


def _seed_retraction_identity_collision_v4(
    memory: RevisionedPhysicalMemory,
    observation_id: str,
) -> None:
    payload_sha256 = v1_module._sha256(
        {
            "schema": "v4-retraction-identity-collision",
            "observation_id": observation_id,
        }
    )
    memory.apply_transaction(
        PhysicalEvidenceTransaction(
            observation=ObservationIdentity(
                observation_id=observation_id,
                payload_sha256=payload_sha256,
                producer_sha256="9" * 64,
                authority=EvidenceAuthority.LEARNED_PHYSICAL,
            ),
            map_frame=memory.map_frame,
            pose=_pose(timestamp=700),
            physical_evidence=(
                PhysicalCellEvidence(cell=(1, 1), label=PhysicalLabel.FREE),
            ),
            projection_contract_sha256=PROFILE_SHA256,
        )
    )


def _committed_target_v4():
    memory, projection, snapshot, runner, adapter = _stack_v4()
    target = adapter.issue(
        snapshot,
        _outcome(runner, snapshot, ground=(_ground_row(),)),
    )
    adapter.commit(target)
    return memory, projection, runner, adapter, target


def _mutate_and_rehash_v4(package, *, timestamp: int) -> None:
    replacement_pose = _pose(timestamp=timestamp)
    object.__setattr__(package, "pose", replacement_pose)
    object.__setattr__(
        package.admission,
        "pose_sha256",
        replacement_pose.content_sha256,
    )
    object.__setattr__(
        package.admission,
        "content_sha256",
        v1_module._sha256(package.admission.to_dict(False)),
    )
    object.__setattr__(
        package,
        "content_sha256",
        v1_module._sha256(package.to_dict(False)),
    )


def _advance_v4(projection, snapshot, runner, adapter, *, sequence: int):
    package = adapter.issue(
        snapshot,
        _outcome(
            runner,
            snapshot,
            pose=_pose(timestamp=sequence),
            rays=(_center_hit(),),
            sequence=sequence,
        ),
    )
    adapter.commit(package)
    return package


def _older_adapters(memory, projection, runner):
    kwargs = {
        "memory": memory,
        "projection": projection,
        "runner": runner,
        "calibration": CALIBRATION,
        **IDENTITIES,
        "camera_transform_sha256": CAMERA_TRANSFORM_SHA256,
        "_synthetic_test_fixture": True,
    }
    return (
        NativeLearnedPhysicalProjectionAdapterV1(**kwargs),
        NativeLearnedPhysicalProjectionAdapterV2(**kwargs),
        NativeLearnedPhysicalProjectionAdapterV3(**kwargs),
    )


def test_v4_is_standalone_and_old_bound_or_unbound_methods_reject() -> None:
    memory, projection, snapshot, runner, adapter = _stack_v4()
    package = adapter.issue(
        snapshot,
        _outcome(runner, snapshot, ground=(_ground_row(),)),
    )
    adapter.commit(package)
    current = projection.project()

    assert not isinstance(
        adapter,
        (
            NativeLearnedPhysicalProjectionAdapterV1,
            NativeLearnedPhysicalProjectionAdapterV2,
            NativeLearnedPhysicalProjectionAdapterV3,
        ),
    )
    assert not any(
        isinstance(
            value,
            (
                NativeLearnedPhysicalProjectionAdapterV1,
                NativeLearnedPhysicalProjectionAdapterV2,
                NativeLearnedPhysicalProjectionAdapterV3,
            ),
        )
        for value in gc.get_referents(adapter)
    )

    for older in _older_adapters(memory, projection, runner):
        with pytest.raises((TypeError, NativeLearnedProjectionBindingError)):
            older.commit(package)
        with pytest.raises((TypeError, NativeLearnedProjectionBindingError)):
            older.issue_retraction(current, package)

    with pytest.raises((AttributeError, TypeError)):
        NativeLearnedPhysicalProjectionAdapterV1.commit(adapter, package)
    with pytest.raises((AttributeError, TypeError)):
        NativeLearnedPhysicalProjectionAdapterV1.issue_retraction(
            adapter,
            current,
            package,
        )
    with pytest.raises((AttributeError, TypeError)):
        NativeLearnedPhysicalProjectionAdapterV2.commit(adapter, package)
    with pytest.raises((AttributeError, TypeError)):
        NativeLearnedPhysicalProjectionAdapterV2.issue_retraction(
            adapter,
            current,
            package,
        )
    with pytest.raises((AttributeError, TypeError)):
        NativeLearnedPhysicalProjectionAdapterV3.commit(adapter, package)
    with pytest.raises((AttributeError, TypeError)):
        NativeLearnedPhysicalProjectionAdapterV3.issue_retraction(
            adapter,
            current,
            package,
        )
    unused_outcome = _outcome(
        runner,
        current,
        rays=(_center_hit(),),
        sequence=22,
    )
    with pytest.raises((AttributeError, TypeError)):
        NativeLearnedPhysicalProjectionAdapterV1.issue(
            adapter,
            current,
            unused_outcome,
        )
    with pytest.raises((AttributeError, TypeError)):
        NativeLearnedPhysicalProjectionAdapterV2.issue(
            adapter,
            current,
            unused_outcome,
        )
    with pytest.raises((AttributeError, TypeError)):
        NativeLearnedPhysicalProjectionAdapterV3.issue(
            adapter,
            current,
            unused_outcome,
        )
    assert package.observation_id in memory.learned_observation_ids


def test_v4_closes_reachable_engine_mutated_target_bypass() -> None:
    memory, projection, _runner, adapter, target = _committed_target_v4()
    _mutate_and_rehash_v4(target, timestamp=101)
    current = projection.project()

    with pytest.raises(
        NativeLearnedProjectionBindingError,
        match="issued content",
    ):
        adapter.issue_retraction(current, target)
    with pytest.raises(AttributeError):
        object.__getattribute__(
            adapter,
            "_NativeLearnedPhysicalProjectionAdapterV2__inner",
        )
    with pytest.raises(AttributeError):
        object.__getattribute__(
            adapter,
            "_NativeLearnedPhysicalProjectionAdapterV3__inner",
        )
    assert target.observation_id in memory.learned_observation_ids


def test_v4_target_digest_is_checked_at_issue_and_final_commit() -> None:
    memory, projection, _runner, adapter, target = _committed_target_v4()
    _mutate_and_rehash_v4(target, timestamp=102)
    with pytest.raises(
        NativeLearnedProjectionBindingError,
        match="issued content",
    ):
        adapter.issue_retraction(projection.project(), target)
    assert target.observation_id in memory.learned_observation_ids

    memory2, projection2, _runner2, adapter2, target2 = _committed_target_v4()
    retraction = adapter2.issue_retraction(projection2.project(), target2)
    _mutate_and_rehash_v4(target2, timestamp=103)
    with pytest.raises(
        NativeLearnedProjectionBindingError,
        match="issued content",
    ):
        adapter2.commit(retraction)
    assert target2.observation_id in memory2.learned_observation_ids


def test_v4_copy_forgery_transfer_mutation_and_replay_cannot_remove_target() -> None:
    memory, projection, runner, adapter, target = _committed_target_v4()
    retraction = adapter.issue_retraction(projection.project(), target)

    with pytest.raises(TypeError, match="non-copyable"):
        copy.copy(retraction)
    reloaded = pickle.loads(pickle.dumps(retraction))
    with pytest.raises(NativeLearnedProjectionBindingError):
        adapter.commit(reloaded)
    forged = object.__new__(type(retraction))
    for name, value in retraction.__dict__.items():
        object.__setattr__(forged, name, value)
    with pytest.raises(NativeLearnedProjectionBindingError):
        adapter.commit(forged)

    foreign = NativeLearnedPhysicalProjectionAdapterV4(
        memory=memory,
        projection=projection,
        runner=runner,
        calibration=CALIBRATION,
        **IDENTITIES,
        camera_transform_sha256=CAMERA_TRANSFORM_SHA256,
        _synthetic_test_fixture=True,
    )
    with pytest.raises(NativeLearnedProjectionBindingError):
        foreign.commit(retraction)

    receipt = adapter.commit(retraction)
    assert receipt.learned_observations_retracted == 1
    with pytest.raises(NativeLearnedProjectionReplayError):
        adapter.commit(retraction)
    assert target.observation_id not in memory.learned_observation_ids

    memory2, projection2, _runner2, adapter2, target2 = _committed_target_v4()
    mutated = adapter2.issue_retraction(projection2.project(), target2)
    _mutate_and_rehash_v4(mutated, timestamp=104)
    with pytest.raises(NativeLearnedProjectionBindingError):
        adapter2.commit(mutated)
    assert target2.observation_id in memory2.learned_observation_ids


def test_v4_failed_stale_commit_releases_for_retry_and_old_stays_terminal() -> None:
    memory, projection, runner, adapter, target = _committed_target_v4()
    shared = projection.project()
    stale = adapter.issue_retraction(shared, target)
    _advance_v4(projection, shared, runner, adapter, sequence=2)

    with pytest.raises(SnapshotBindingError):
        adapter.commit(stale)
    assert target.observation_id in memory.learned_observation_ids

    replacement = adapter.issue_retraction(projection.project(), target)
    with pytest.raises(
        NativeLearnedProjectionReplayError,
        match="terminally stale",
    ):
        adapter.commit(stale)
    receipt = adapter.commit(replacement)
    assert receipt.learned_observations_retracted == 1
    assert target.observation_id not in memory.learned_observation_ids
    for _ in range(2):
        projection.project()
        with pytest.raises(
            NativeLearnedProjectionReplayError,
            match="terminally stale",
        ):
            adapter.commit(stale)


def test_v4_proactive_stale_retry_allows_exactly_one_live_retraction() -> None:
    memory, projection, _runner, adapter, target = _committed_target_v4()
    stale = adapter.issue_retraction(projection.project(), target)
    current = projection.project()
    replacement = adapter.issue_retraction(current, target)

    with pytest.raises(
        NativeLearnedProjectionReplayError,
        match="exact live V4 retraction",
    ):
        adapter.issue_retraction(current, target)
    with pytest.raises(
        NativeLearnedProjectionReplayError,
        match="terminally stale",
    ):
        adapter.commit(stale)
    assert target.observation_id in memory.learned_observation_ids
    adapter.commit(replacement)
    assert target.observation_id not in memory.learned_observation_ids


def test_v4_removal_is_atomic_and_preserves_other_active_identity() -> None:
    memory, projection, runner, adapter, target = _committed_target_v4()
    other = _advance_v4(
        projection,
        projection.project(),
        runner,
        adapter,
        sequence=3,
    )
    assert memory.learned_observation_ids == frozenset(
        {target.observation_id, other.observation_id}
    )
    revision_before = memory.revision

    retraction = adapter.issue_retraction(projection.project(), target)
    receipt = adapter.commit(retraction)
    assert receipt.revision_before == revision_before
    assert receipt.revision_after == revision_before + 1
    assert receipt.learned_observations_retracted == 1
    assert memory.learned_observation_ids == frozenset({other.observation_id})
    with pytest.raises(NativeLearnedProjectionBindingError, match="not an exact active"):
        adapter.issue_retraction(projection.project(), target)


def test_v4_preserves_free_rotation_boundary_and_occupied_precedence() -> None:
    memory, _projection, snapshot, runner, adapter = _stack_v4()
    free = adapter.issue(
        snapshot,
        _outcome(runner, snapshot, ground=(_ground_row(),), sequence=1),
    )
    assert free.projection_receipt.free_cells == frozenset({(32, 32)})
    assert not free.projection_receipt.occupied_cells
    assert not hasattr(free, "transaction")
    adapter.commit(free)
    assert memory.physical_state((32, 32)) is PhysicalLabel.FREE

    _m2, _p2, s2, r2, a2 = _stack_v4()
    rotated = a2.issue(
        s2,
        _outcome(
            r2,
            s2,
            pose=_pose(yaw=math.pi / 2.0),
            ground=(_ground_row(),),
            sequence=2,
        ),
    )
    assert rotated.projection_receipt.free_cells == frozenset({(31, 32)})

    _m3, _p3, s3, r3, a3 = _stack_v4()
    boundary = a3.issue(
        s3,
        _outcome(r3, s3, rays=(_boundary_hit(),), sequence=3),
    )
    assert boundary.projection_receipt.occupied_cells == frozenset(
        {(32, 32), (33, 32)}
    )

    _m4, _p4, s4, r4, a4 = _stack_v4()
    conflict = a4.issue(
        s4,
        _outcome(
            r4,
            s4,
            ground=(_ground_row(),),
            rays=(_center_hit(),),
            sequence=4,
        ),
    )
    assert (32, 32) in conflict.projection_receipt.occupied_cells
    assert (32, 32) not in conflict.projection_receipt.free_cells


def test_v4_preserves_translation_covariance_and_native_resolution_rules() -> None:
    _memory, _projection, snapshot, runner, adapter = _stack_v4()
    translated = adapter.issue(
        snapshot,
        _outcome(
            runner,
            snapshot,
            pose=_pose(translation=(0.025, 0.0)),
            ground=(_ground_row(),),
            sequence=1,
        ),
    )
    assert not translated.projection_receipt.free_cells
    assert (32, 32) in translated.projection_receipt.unknown_cells

    _m2, _p2, s2, r2, a2 = _stack_v4()
    excessive = _outcome(
        r2,
        s2,
        pose=_pose(covariance=(0.0026, 0.0, 0.0)),
        ground=(_ground_row(),),
        sequence=2,
    )
    with pytest.raises(NativeLearnedProjectionRejectedError, match="covariance"):
        a2.issue(s2, excessive)

    _m3, _p3, s3, r3, a3 = _stack_v4()
    derived = NativeV4SourceGeometryV1(
        shape=(64, 64),
        origin_forward_left_m=CALIBRATION.source_origin_forward_left_m,
        cell_size_m=0.10,
    )
    derived_outcome = _outcome(
        r3,
        s3,
        ground=(_ground_row(),),
        geometry=derived,
        source_derivation="derived_0p10_upsampled",
        sequence=3,
    )
    with pytest.raises(
        NativeLearnedProjectionRejectedError,
        match="0.10 m/upsampling",
    ):
        a3.issue(s3, derived_outcome)


def test_v4_contradiction_is_unknown_and_exact_retraction_restores_free() -> None:
    memory, projection, snapshot, runner, adapter = _stack_v4()
    free = adapter.issue(
        snapshot,
        _outcome(runner, snapshot, ground=(_ground_row(),), sequence=1),
    )
    adapter.commit(free)

    occupied_snapshot = projection.project()
    occupied = adapter.issue(
        occupied_snapshot,
        _outcome(
            runner,
            occupied_snapshot,
            rays=(_center_hit(),),
            sequence=2,
        ),
    )
    adapter.commit(occupied)
    assert memory.physical_state((32, 32)) is PhysicalLabel.UNKNOWN

    retraction = adapter.issue_retraction(projection.project(), occupied)
    receipt = adapter.commit(retraction)
    assert receipt.learned_observations_retracted == 1
    assert memory.physical_state((32, 32)) is PhysicalLabel.FREE
    assert free.observation_id in memory.learned_observation_ids
    assert occupied.observation_id not in memory.learned_observation_ids


def test_v4_every_callable_commit_path_rechecks_final_target_binding() -> None:
    memory, projection, _runner, adapter, target = _committed_projection_v4()
    retraction = adapter.issue_retraction(projection.project(), target)
    _mutate_and_rehash_v4(target, timestamp=703)

    with pytest.raises(NativeLearnedProjectionBindingError, match="issued content"):
        adapter.commit(retraction)
    assert target.observation_id in memory.learned_observation_ids

    with pytest.raises(NativeLearnedProjectionBindingError, match="issued content"):
        adapter._commit_core_v4(retraction)
    assert target.observation_id in memory.learned_observation_ids


def test_v4_permanent_late_rejection_releases_target_slot() -> None:
    memory, projection, _runner, adapter, target = _committed_projection_v4()
    colliding_id = (
        "qualified-native-v4-v4-retract:1:" f"{target.observation_id}"
    )
    _seed_retraction_identity_collision_v4(memory, colliding_id)

    rejected = adapter.issue_retraction(projection.project(), target)
    revision_before = memory.revision
    with pytest.raises(TransactionRejectedError, match="duplicate observation identity"):
        adapter.commit(rejected)
    assert memory.revision == revision_before
    assert target.observation_id in memory.learned_observation_ids
    with pytest.raises(NativeLearnedProjectionReplayError, match="terminally stale"):
        adapter.commit(rejected)

    replacement = adapter.issue_retraction(projection.project(), target)
    receipt = adapter.commit(replacement)
    assert receipt.learned_observations_retracted == 1
    assert target.observation_id not in memory.learned_observation_ids


def test_v4_transient_late_rejection_keeps_exact_package_retryable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    memory, projection, _runner, adapter, target = _committed_projection_v4()
    retraction = adapter.issue_retraction(projection.project(), target)
    revision_before = memory.revision

    original_apply = RevisionedPhysicalMemory.apply_transaction

    def reject_once(
        instance: RevisionedPhysicalMemory,
        transaction: object,
    ) -> object:
        if instance is memory:
            raise TransactionRejectedError("injected transient rejection")
        return original_apply(instance, transaction)  # type: ignore[arg-type]

    with monkeypatch.context() as patch:
        patch.setattr(RevisionedPhysicalMemory, "apply_transaction", reject_once)
        with pytest.raises(TransactionRejectedError, match="transient"):
            adapter.commit(retraction)
    assert memory.revision == revision_before
    assert target.observation_id in memory.learned_observation_ids

    receipt = adapter.commit(retraction)
    assert receipt.learned_observations_retracted == 1
    assert target.observation_id not in memory.learned_observation_ids


def test_v4_contract_authority_copy_serialization_and_production_fail_closed() -> None:
    _memory, _projection, snapshot, runner, adapter = _stack_v4()
    package = adapter.issue(
        snapshot,
        _outcome(runner, snapshot, ground=(_ground_row(),)),
    )
    assert type(package) is QualifiedLearnedPhysicalDevelopmentTransactionV4
    assert package.admission.adapter_contract_sha256 == adapter.adapter_contract_sha256
    for surface in (adapter, package, package.admission, package.projection_receipt):
        assert surface.development_only is True
        assert surface.hardware_execution_authorized is False
        assert surface.production_promotion_authorized is False
    with pytest.raises(TypeError, match="non-copyable"):
        copy.copy(adapter)
    with pytest.raises(TypeError, match="non-serializable"):
        pickle.dumps(adapter)
    with pytest.raises(TypeError, match="non-copyable"):
        copy.copy(package)

    assert (
        v4_module.PRODUCTION_NATIVE_V4_RUNNER,
        v4_module.PRODUCTION_V4_CHECKPOINT_FILE_SHA256,
        v4_module.PRODUCTION_G2_REPORT_FILE_SHA256,
        v4_module.PRODUCTION_V4_CALIBRATION_SHA256,
        v4_module.PRODUCTION_NATIVE_LEARNED_PROJECTION_ADAPTER_V4,
    ) == (None,) * 5
    with pytest.raises(PermissionError, match="identities are unset"):
        require_production_native_learned_projection_adapter_v4()


def test_v4_source_has_no_old_adapter_or_runtime_input_surface() -> None:
    source = Path(v4_module.__file__).read_text(encoding="utf-8").lower()
    for forbidden in (
        "nativelearnedphysicalprojectionadapterv1",
        "nativelearnedphysicalprojectionadapterv2",
        "nativelearnedphysicalprojectionadapterv3",
        "native_learned_physical_projection_v2",
        "native_learned_physical_projection_v3",
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
