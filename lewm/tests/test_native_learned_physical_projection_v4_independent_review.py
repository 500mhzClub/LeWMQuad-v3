"""Independent hostile lifecycle review for native learned projection V4.

Only synthetic development fixtures are used. Candidate and author-test bytes
are frozen and never edited by this review.
"""
from __future__ import annotations

import copy
import hashlib
from pathlib import Path
import pickle

import pytest

from lewm.planning import native_learned_physical_projection_v1 as v1_module
from lewm.planning import native_learned_physical_projection_v4 as v4_module
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
)
from lewm.planning.native_learned_physical_projection_v4 import (
    NativeLearnedPhysicalProjectionAdapterV4,
    QualifiedLearnedPhysicalDevelopmentTransactionV4,
    require_production_native_learned_projection_adapter_v4,
)
from lewm.planning.revisioned_physical_configuration_memory import (
    EvidenceAuthority,
    ObservationIdentity,
    PhysicalEvidenceTransaction,
    RevisionedPhysicalMemory,
    TransactionRejectedError,
)
from lewm.planning.two_resolution_configuration_projection_v2 import PROFILE_SHA256
from lewm.tests.test_native_learned_physical_projection_v1 import (
    CALIBRATION,
    CAMERA_TRANSFORM_SHA256,
    IDENTITIES,
    _center_hit,
    _ground_row,
    _outcome,
    _pose,
)
from lewm.tests.test_native_learned_physical_projection_v4 import (
    _committed_projection_v4,
    _mutate_and_rehash_v4,
    _seed_retraction_identity_collision_v4,
    _stack_v4,
)


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "lewm/planning/native_learned_physical_projection_v4.py"
AUTHOR_TEST = ROOT / "lewm/tests/test_native_learned_physical_projection_v4.py"
HANDOFF = ROOT / "docs/lewm_go2_g3_native_learned_physical_projection_v4_author_handoff_2026-07-13.md"
FROZEN_SHA256 = {
    SOURCE: "66486f70f0998502f36e16e496f1c76d11cd117176046e6de433db911473f16a",
    AUTHOR_TEST: "df9b89778adea21da70b89004da41a01354b7086dd25eab5961f3a5bb1e0abb2",
    HANDOFF: "79407230f17714634ec0cb492fbf822131bf4aa958b140ffa49e0b95d027cbce",
}


def _older_adapters(
    memory: RevisionedPhysicalMemory,
    projection: object,
    runner: object,
) -> tuple[object, ...]:
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


def _retire_collision(memory: RevisionedPhysicalMemory, collision_id: str) -> None:
    payload = v1_module._sha256(
        {
            "schema": "v4-independent-retire-collision",
            "collision_id": collision_id,
        }
    )
    memory.apply_transaction(
        PhysicalEvidenceTransaction(
            observation=ObservationIdentity(
                observation_id=f"retire:{collision_id}",
                payload_sha256=payload,
                producer_sha256="8" * 64,
                authority=EvidenceAuthority.LEARNED_PHYSICAL,
            ),
            map_frame=memory.map_frame,
            pose=_pose(timestamp=909),
            retract_learned_observation_ids=(collision_id,),
            projection_contract_sha256=PROFILE_SHA256,
        )
    )


def test_frozen_candidate_author_tests_and_handoff_match() -> None:
    for path, expected in FROZEN_SHA256.items():
        assert hashlib.sha256(path.read_bytes()).hexdigest() == expected


def test_every_bound_and_unbound_v4_commit_path_rechecks_final_target() -> None:
    for call in (
        lambda adapter, package: adapter.commit(package),
        lambda adapter, package: adapter._commit_core_v4(package),
        lambda adapter, package: NativeLearnedPhysicalProjectionAdapterV4.commit(
            adapter,
            package,
        ),
        lambda adapter, package: NativeLearnedPhysicalProjectionAdapterV4._commit_core_v4(
            adapter,
            package,
        ),
    ):
        memory, projection, _runner, adapter, target = _committed_projection_v4()
        retraction = adapter.issue_retraction(projection.project(), target)
        _mutate_and_rehash_v4(target, timestamp=901)
        with pytest.raises(NativeLearnedProjectionBindingError, match="issued content"):
            call(adapter, retraction)
        assert target.observation_id in memory.learned_observation_ids


def test_tombstoned_duplicate_identity_is_permanent_and_must_release_slot() -> None:
    """An indelible but inactive ID must not be mislabeled transient."""

    memory, projection, _runner, adapter, target = _committed_projection_v4()
    collision_id = f"qualified-native-v4-v4-retract:1:{target.observation_id}"
    _seed_retraction_identity_collision_v4(memory, collision_id)
    assert collision_id in memory.learned_observation_ids
    _retire_collision(memory, collision_id)
    assert collision_id not in memory.learned_observation_ids

    current = projection.project()
    rejected = adapter.issue_retraction(current, target)
    revision_before = memory.revision
    with pytest.raises(TransactionRejectedError, match="duplicate observation identity"):
        adapter.commit(rejected)
    assert memory.revision == revision_before
    assert target.observation_id in memory.learned_observation_ids

    with pytest.raises(NativeLearnedProjectionReplayError, match="terminally stale"):
        adapter.commit(rejected)
    replacement = adapter.issue_retraction(current, target)
    assert type(replacement) is QualifiedLearnedPhysicalDevelopmentTransactionV4


def test_transient_rejection_keeps_only_the_exact_package_retryable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    memory, projection, _runner, adapter, target = _committed_projection_v4()
    current = projection.project()
    package = adapter.issue_retraction(current, target)
    original = RevisionedPhysicalMemory.apply_transaction

    def transient(instance: RevisionedPhysicalMemory, transaction: object) -> object:
        if instance is memory:
            raise TransactionRejectedError("independent transient rejection")
        return original(instance, transaction)  # type: ignore[arg-type]

    with monkeypatch.context() as patch:
        patch.setattr(RevisionedPhysicalMemory, "apply_transaction", transient)
        with pytest.raises(TransactionRejectedError, match="independent transient"):
            adapter._commit_core_v4(package)
    with pytest.raises(NativeLearnedProjectionReplayError, match="exact live"):
        adapter.issue_retraction(current, target)
    receipt = adapter._commit_core_v4(package)
    assert receipt.learned_observations_retracted == 1
    assert target.observation_id not in memory.learned_observation_ids


def test_copy_pickle_reconstruction_transfer_replay_and_legacy_cross_calls_reject() -> None:
    memory, projection, runner, adapter, target = _committed_projection_v4()
    package = adapter.issue_retraction(projection.project(), target)
    with pytest.raises(TypeError, match="non-copyable"):
        copy.copy(package)
    with pytest.raises(TypeError, match="non-copyable"):
        copy.deepcopy(package)
    reloaded = pickle.loads(pickle.dumps(package))
    with pytest.raises(NativeLearnedProjectionBindingError):
        adapter.commit(reloaded)
    forged = object.__new__(type(package))
    for name, value in package.__dict__.items():
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
        foreign.commit(package)

    for older in _older_adapters(memory, projection, runner):
        with pytest.raises((TypeError, NativeLearnedProjectionBindingError)):
            older.commit(package)  # type: ignore[attr-defined]
    for legacy in (
        NativeLearnedPhysicalProjectionAdapterV1,
        NativeLearnedPhysicalProjectionAdapterV2,
        NativeLearnedPhysicalProjectionAdapterV3,
    ):
        with pytest.raises((AttributeError, TypeError)):
            legacy.commit(adapter, package)

    adapter.commit(package)
    with pytest.raises(NativeLearnedProjectionReplayError):
        adapter.commit(package)


def test_successful_retraction_is_atomic_and_preserves_unrelated_identity() -> None:
    memory, projection, snapshot, runner, adapter = _stack_v4()
    first = adapter.issue(
        snapshot,
        _outcome(runner, snapshot, ground=(_ground_row(),), sequence=1),
    )
    adapter.commit(first)
    current = projection.project()
    second = adapter.issue(
        current,
        _outcome(runner, current, rays=(_center_hit(),), sequence=2),
    )
    adapter.commit(second)
    before = memory.revision
    receipt = adapter.commit(adapter.issue_retraction(projection.project(), first))
    assert receipt.revision_before == before
    assert receipt.revision_after == before + 1
    assert receipt.learned_observations_retracted == 1
    assert first.observation_id not in memory.learned_observation_ids
    assert second.observation_id in memory.learned_observation_ids


def test_production_and_promotion_remain_fail_closed() -> None:
    assert (
        v4_module.PRODUCTION_NATIVE_V4_RUNNER,
        v4_module.PRODUCTION_V4_CHECKPOINT_FILE_SHA256,
        v4_module.PRODUCTION_G2_REPORT_FILE_SHA256,
        v4_module.PRODUCTION_V4_CALIBRATION_SHA256,
        v4_module.PRODUCTION_NATIVE_LEARNED_PROJECTION_ADAPTER_V4,
    ) == (None,) * 5
    with pytest.raises(PermissionError, match="identities are unset"):
        require_production_native_learned_projection_adapter_v4()
    _memory, _projection, snapshot, runner, adapter = _stack_v4()
    package = adapter.issue(
        snapshot,
        _outcome(runner, snapshot, ground=(_ground_row(),)),
    )
    for surface in (adapter, package, package.admission, package.projection_receipt):
        assert surface.development_only is True
        assert surface.hardware_execution_authorized is False
        assert surface.production_promotion_authorized is False
