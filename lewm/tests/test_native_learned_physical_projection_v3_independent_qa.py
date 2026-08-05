"""Independent exact-object and lifecycle QA for native learned projection V3.

Only synthetic development fixtures are used.  The pinned author implementation
and author tests are inputs to this review and are not modified here.
"""
from __future__ import annotations

import ast
import copy
import gc
import hashlib
from pathlib import Path
import pickle

import pytest

from lewm.planning import native_learned_physical_projection_v1 as v1_module
from lewm.planning.native_learned_physical_projection_v1 import (
    NativeLearnedPhysicalProjectionAdapterV1,
    NativeLearnedProjectionBindingError,
    NativeLearnedProjectionReplayError,
)
from lewm.planning.native_learned_physical_projection_v2 import (
    NativeLearnedPhysicalProjectionAdapterV2,
)
from lewm.planning.native_learned_physical_projection_v3 import (
    NativeLearnedPhysicalProjectionAdapterV3,
    QualifiedLearnedPhysicalDevelopmentTransactionV3,
)
from lewm.planning.revisioned_physical_configuration_memory import (
    EvidenceAuthority,
    ObservationIdentity,
    PhysicalCellEvidence,
    PhysicalEvidenceTransaction,
    PhysicalLabel,
    RevisionedPhysicalMemory,
    SnapshotBindingError,
    TransactionRejectedError,
)
from lewm.planning.two_resolution_configuration_projection_v2 import (
    PROFILE_SHA256,
    TwoResolutionConfigurationSnapshotV2,
)
from lewm.tests.test_native_learned_physical_projection_v1 import (
    CALIBRATION,
    CAMERA_TRANSFORM_SHA256,
    IDENTITIES,
    _ground_row,
    _outcome,
    _pose,
)
from lewm.tests.test_native_learned_physical_projection_v3 import _stack_v3


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "lewm/planning/native_learned_physical_projection_v3.py"
AUTHOR_TEST = ROOT / "lewm/tests/test_native_learned_physical_projection_v3.py"
HANDOFF = (
    ROOT
    / "docs/lewm_go2_g3_native_learned_physical_projection_v3_author_handoff_2026-07-13.md"
)
FROZEN_SHA256 = {
    SOURCE: "c472b4792279a20fd7085189ea53d3a6c7d2c33343d86cc9063c73eea42f136f",
    AUTHOR_TEST: "d5113b9c98ad88f42315ce326cc8bb2b12933b3fc37471419282886f32f19129",
    HANDOFF: "93cd66b03001abbf465053c1ae2277fa3c9daba8ee3332cd212e8d990a74722b",
}


def _committed_projection():
    memory, projection, snapshot, runner, adapter = _stack_v3()
    package = adapter.issue(
        snapshot,
        _outcome(runner, snapshot, ground=(_ground_row(),)),
    )
    adapter.commit(package)
    return memory, projection, runner, adapter, package


def _legacy_adapters(memory, projection, runner):
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
    )


def _foreign_v3(memory, projection, runner):
    return NativeLearnedPhysicalProjectionAdapterV3(
        memory=memory,
        projection=projection,
        runner=runner,
        calibration=CALIBRATION,
        **IDENTITIES,
        camera_transform_sha256=CAMERA_TRANSFORM_SHA256,
        _synthetic_test_fixture=True,
    )


def _mutate_and_rehash_target(package, *, timestamp: int) -> None:
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


def _seed_retraction_observation_collision(memory, observation_id: str) -> None:
    payload_sha256 = v1_module._sha256(
        {
            "schema": "v3-independent-qa-retraction-collision",
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
            pose=_pose(timestamp=910),
            physical_evidence=(
                PhysicalCellEvidence(cell=(1, 1), label=PhysicalLabel.FREE),
            ),
            projection_contract_sha256=PROFILE_SHA256,
        )
    )


def test_v3_qa_frozen_bytes_and_standalone_authority_shape() -> None:
    for path, expected in FROZEN_SHA256.items():
        assert hashlib.sha256(path.read_bytes()).hexdigest() == expected

    tree = ast.parse(SOURCE.read_text(encoding="utf-8"))
    imported_modules = {
        node.module
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom)
    }
    imported_names = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom)
        for alias in node.names
    }
    assert "lewm.planning.native_learned_physical_projection_v2" not in imported_modules
    assert "NativeLearnedPhysicalProjectionAdapterV1" not in imported_names
    assert "NativeLearnedPhysicalProjectionAdapterV2" not in imported_names

    _memory, _projection, _snapshot, _runner, adapter = _stack_v3()
    assert type(adapter).__mro__ == (NativeLearnedPhysicalProjectionAdapterV3, object)
    assert all(slot.endswith("_v3") for slot in type(adapter).__slots__)
    assert not isinstance(
        adapter,
        (
            NativeLearnedPhysicalProjectionAdapterV1,
            NativeLearnedPhysicalProjectionAdapterV2,
        ),
    )
    assert not any(
        isinstance(
            value,
            (
                NativeLearnedPhysicalProjectionAdapterV1,
                NativeLearnedPhysicalProjectionAdapterV2,
            ),
        )
        for value in gc.get_referents(adapter)
    )
    with pytest.raises(AttributeError):
        object.__getattribute__(
            adapter,
            "_NativeLearnedPhysicalProjectionAdapterV2__inner",
        )


def test_v3_qa_bound_and_unbound_v1_v2_methods_cannot_operate_v3() -> None:
    memory, projection, runner, adapter, target = _committed_projection()
    current = projection.project()
    retraction = adapter.issue_retraction(current, target)

    for legacy in _legacy_adapters(memory, projection, runner):
        with pytest.raises((TypeError, NativeLearnedProjectionBindingError)):
            legacy.commit(target)
        with pytest.raises((TypeError, NativeLearnedProjectionBindingError)):
            legacy.commit(retraction)
        with pytest.raises((TypeError, NativeLearnedProjectionBindingError)):
            legacy.issue_retraction(current, target)

    unused_outcome = _outcome(
        runner,
        current,
        ground=(_ground_row(),),
        sequence=41,
    )
    for legacy_type in (
        NativeLearnedPhysicalProjectionAdapterV1,
        NativeLearnedPhysicalProjectionAdapterV2,
    ):
        with pytest.raises((AttributeError, TypeError)):
            legacy_type.issue(adapter, current, unused_outcome)
        with pytest.raises((AttributeError, TypeError)):
            legacy_type.issue_retraction(adapter, current, target)
        with pytest.raises((AttributeError, TypeError)):
            legacy_type.commit(adapter, retraction)

    issued_after_rejections = adapter.issue(current, unused_outcome)
    assert type(issued_after_rejections) is QualifiedLearnedPhysicalDevelopmentTransactionV3
    assert target.observation_id in memory.learned_observation_ids


def test_v3_qa_exact_snapshot_outcome_and_one_use_bindings() -> None:
    _memory, _projection, snapshot, runner, adapter = _stack_v3()
    outcome = _outcome(runner, snapshot, ground=(_ground_row(),))
    reloaded_snapshot = TwoResolutionConfigurationSnapshotV2.deserialize(
        snapshot.serialize()
    )
    with pytest.raises(SnapshotBindingError, match="exact live object"):
        adapter.issue(reloaded_snapshot, outcome)
    with pytest.raises(NativeLearnedProjectionBindingError, match="exact live object"):
        adapter.issue(snapshot, copy.copy(outcome))

    package = adapter.issue(snapshot, outcome)
    with pytest.raises(NativeLearnedProjectionReplayError, match="already consumed"):
        adapter.issue(snapshot, outcome)
    adapter.commit(package)
    with pytest.raises(NativeLearnedProjectionReplayError, match="already consumed"):
        adapter.commit(package)


def test_v3_qa_copy_pickle_forgery_transfer_and_failed_attempts_preserve_live() -> None:
    memory, projection, runner, adapter, target = _committed_projection()
    retraction = adapter.issue_retraction(projection.project(), target)

    with pytest.raises(TypeError, match="non-copyable"):
        copy.copy(adapter)
    with pytest.raises(TypeError, match="non-copyable"):
        copy.deepcopy(adapter)
    with pytest.raises(TypeError, match="non-serializable"):
        pickle.dumps(adapter)
    with pytest.raises(TypeError, match="non-copyable"):
        copy.copy(retraction)
    with pytest.raises(TypeError, match="non-copyable"):
        copy.deepcopy(retraction)

    reloaded = pickle.loads(pickle.dumps(retraction))
    with pytest.raises(NativeLearnedProjectionBindingError):
        adapter.commit(reloaded)
    forged = object.__new__(type(retraction))
    for name, value in retraction.__dict__.items():
        object.__setattr__(forged, name, value)
    with pytest.raises(NativeLearnedProjectionBindingError):
        adapter.commit(forged)
    with pytest.raises(NativeLearnedProjectionBindingError):
        _foreign_v3(memory, projection, runner).commit(retraction)

    receipt = adapter.commit(retraction)
    assert receipt.learned_observations_retracted == 1
    assert target.observation_id not in memory.learned_observation_ids
    with pytest.raises(NativeLearnedProjectionReplayError, match="already consumed"):
        adapter.commit(retraction)


def test_v3_qa_validation_error_restores_exact_retraction_for_retry() -> None:
    memory, projection, _runner, adapter, target = _committed_projection()
    retraction = adapter.issue_retraction(projection.project(), target)
    issued_digest = retraction.content_sha256
    object.__setattr__(retraction, "content_sha256", "0" * 64)

    with pytest.raises(NativeLearnedProjectionBindingError, match="mutated"):
        adapter.commit(retraction)
    assert target.observation_id in memory.learned_observation_ids

    object.__setattr__(retraction, "content_sha256", issued_digest)
    receipt = adapter.commit(retraction)
    assert receipt.learned_observations_retracted == 1
    assert target.observation_id not in memory.learned_observation_ids


def test_v3_qa_transient_memory_error_keeps_exact_retraction_retryable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    memory, projection, _runner, adapter, target = _committed_projection()
    retraction = adapter.issue_retraction(projection.project(), target)
    revision_before = memory.revision

    original_apply = RevisionedPhysicalMemory.apply_transaction

    def reject_once(
        candidate_memory: RevisionedPhysicalMemory,
        transaction: object,
    ) -> object:
        if candidate_memory is memory:
            raise TransactionRejectedError("independent injected rejection")
        return original_apply(candidate_memory, transaction)

    with monkeypatch.context() as patch:
        patch.setattr(RevisionedPhysicalMemory, "apply_transaction", reject_once)
        with pytest.raises(TransactionRejectedError, match="injected rejection"):
            adapter.commit(retraction)

    assert memory.revision == revision_before
    assert target.observation_id in memory.learned_observation_ids
    receipt = adapter.commit(retraction)
    assert receipt.learned_observations_retracted == 1
    assert target.observation_id not in memory.learned_observation_ids


def test_v3_qa_terminal_memory_rejection_restores_reservation_state() -> None:
    memory, projection, _runner, adapter, target = _committed_projection()
    collision_id = f"qualified-native-v4-v3-retract:1:{target.observation_id}"
    _seed_retraction_observation_collision(memory, collision_id)
    snapshot = projection.project()
    rejected = adapter.issue_retraction(snapshot, target)
    revision_before = memory.revision

    with pytest.raises(TransactionRejectedError, match="duplicate observation identity"):
        adapter.commit(rejected)
    assert memory.revision == revision_before
    assert target.observation_id in memory.learned_observation_ids

    try:
        adapter.commit(rejected)
    except Exception as error:  # noqa: BLE001 - exact failure is asserted below
        old_package_error: Exception | None = error
    else:
        old_package_error = None
    try:
        replacement = adapter.issue_retraction(snapshot, target)
    except Exception as error:  # noqa: BLE001 - exact failure is asserted below
        replacement_error: Exception | None = error
    else:
        replacement_error = None

    assert isinstance(old_package_error, NativeLearnedProjectionReplayError) and (
        replacement_error is None
    ), (
        "terminal memory rejection left reservation state unrestored: "
        f"old retry={type(old_package_error).__name__}, "
        f"fresh issue={type(replacement_error).__name__}"
    )
    assert type(replacement) is QualifiedLearnedPhysicalDevelopmentTransactionV3


def test_v3_qa_stale_reservation_releases_but_old_package_stays_terminal() -> None:
    memory, projection, _runner, adapter, target = _committed_projection()
    stale = adapter.issue_retraction(projection.project(), target)
    current = projection.project()
    replacement = adapter.issue_retraction(current, target)

    with pytest.raises(NativeLearnedProjectionReplayError, match="terminally stale"):
        adapter.commit(stale)
    adapter.commit(replacement)
    assert target.observation_id not in memory.learned_observation_ids


def test_v3_qa_every_callable_commit_path_enforces_final_target_binding() -> None:
    """A reachable helper must not bypass the public final-target checks."""

    memory, projection, _runner, adapter, target = _committed_projection()
    retraction = adapter.issue_retraction(projection.project(), target)
    _mutate_and_rehash_target(target, timestamp=909)

    with pytest.raises(NativeLearnedProjectionBindingError, match="issued content"):
        adapter.commit(retraction)
    assert target.observation_id in memory.learned_observation_ids

    try:
        receipt = adapter._commit_core_v3(retraction)
    except (AttributeError, PermissionError, NativeLearnedProjectionBindingError):
        return
    assert target.observation_id in memory.learned_observation_ids, (
        "reachable _commit_core_v3 bypassed the final target binding and removed "
        f"active evidence with transaction {receipt.transaction_sha256}"
    )
