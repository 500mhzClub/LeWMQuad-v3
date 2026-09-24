"""Independent lifecycle probes for native learned projection V3.

The candidate source and its author tests remain frozen.  These tests exercise
the component through synthetic development fixtures only.
"""
from __future__ import annotations

import copy
import hashlib
from pathlib import Path
import pickle

import pytest

from lewm.planning import native_learned_physical_projection_v1 as v1_module
from lewm.planning import native_learned_physical_projection_v3 as v3_module
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
    SnapshotBindingError,
    TransactionRejectedError,
)
from lewm.planning.two_resolution_configuration_projection_v2 import (
    PROFILE_SHA256,
)
from lewm.tests.test_native_learned_physical_projection_v1 import (
    CALIBRATION,
    CAMERA_TRANSFORM_SHA256,
    IDENTITIES,
    _center_hit,
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


def _committed_target():
    memory, projection, snapshot, runner, adapter = _stack_v3()
    target = adapter.issue(
        snapshot,
        _outcome(runner, snapshot, ground=(_ground_row(),)),
    )
    adapter.commit(target)
    return memory, projection, runner, adapter, target


def _mutate_and_rehash(package, *, timestamp: int) -> None:
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
    )


def _seed_retraction_observation_collision(memory, observation_id: str) -> None:
    payload = v1_module._sha256(
        {
            "schema": "v3-independent-review-collision",
            "observation_id": observation_id,
        }
    )
    transaction = PhysicalEvidenceTransaction(
        observation=ObservationIdentity(
            observation_id=observation_id,
            payload_sha256=payload,
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
    memory.apply_transaction(transaction)


def test_frozen_candidate_hashes_and_standalone_source_shape() -> None:
    for path, expected in FROZEN_SHA256.items():
        assert hashlib.sha256(path.read_bytes()).hexdigest() == expected

    source = SOURCE.read_text(encoding="utf-8")
    assert "NativeLearnedPhysicalProjectionAdapterV1" not in source
    assert "NativeLearnedPhysicalProjectionAdapterV2" not in source
    assert "native_learned_physical_projection_v2" not in source

    _memory, _projection, _snapshot, _runner, adapter = _stack_v3()
    assert not isinstance(
        adapter,
        (
            NativeLearnedPhysicalProjectionAdapterV1,
            NativeLearnedPhysicalProjectionAdapterV2,
        ),
    )
    for slot in type(adapter).__slots__:
        value = object.__getattribute__(adapter, slot)
        assert not isinstance(
            value,
            (
                NativeLearnedPhysicalProjectionAdapterV1,
                NativeLearnedPhysicalProjectionAdapterV2,
            ),
        )


def test_bound_and_unbound_legacy_entry_points_reject_v3_packages() -> None:
    memory, projection, snapshot, runner, adapter = _stack_v3()
    package = adapter.issue(
        snapshot,
        _outcome(runner, snapshot, ground=(_ground_row(),)),
    )
    adapter.commit(package)
    current = projection.project()

    for older in _older_adapters(memory, projection, runner):
        with pytest.raises((TypeError, NativeLearnedProjectionBindingError)):
            older.commit(package)
        with pytest.raises((TypeError, NativeLearnedProjectionBindingError)):
            older.issue_retraction(current, package)

    for legacy_type in (
        NativeLearnedPhysicalProjectionAdapterV1,
        NativeLearnedPhysicalProjectionAdapterV2,
    ):
        with pytest.raises((AttributeError, TypeError)):
            legacy_type.commit(adapter, package)
        with pytest.raises((AttributeError, TypeError)):
            legacy_type.issue_retraction(adapter, current, package)


def test_object_identity_copy_pickle_transfer_mutation_and_replay_reject() -> None:
    memory, projection, runner, adapter, target = _committed_target()
    retraction = adapter.issue_retraction(projection.project(), target)

    with pytest.raises(TypeError, match="non-copyable"):
        copy.copy(retraction)
    with pytest.raises(TypeError, match="non-copyable"):
        copy.deepcopy(retraction)
    reloaded = pickle.loads(pickle.dumps(retraction))
    with pytest.raises(NativeLearnedProjectionBindingError):
        adapter.commit(reloaded)

    foreign = NativeLearnedPhysicalProjectionAdapterV3(
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

    _mutate_and_rehash(retraction, timestamp=701)
    with pytest.raises(NativeLearnedProjectionBindingError):
        adapter.commit(retraction)
    assert target.observation_id in memory.learned_observation_ids


def test_stale_failure_restores_target_slot_and_old_package_stays_terminal() -> None:
    memory, projection, runner, adapter, target = _committed_target()
    shared = projection.project()
    stale = adapter.issue_retraction(shared, target)
    intervening = adapter.issue(
        shared,
        _outcome(
            runner,
            shared,
            pose=_pose(timestamp=702),
            rays=(_center_hit(),),
            sequence=702,
        ),
    )
    adapter.commit(intervening)

    with pytest.raises(SnapshotBindingError):
        adapter.commit(stale)
    replacement = adapter.issue_retraction(projection.project(), target)
    with pytest.raises(NativeLearnedProjectionReplayError, match="terminally stale"):
        adapter.commit(stale)
    adapter.commit(replacement)
    assert target.observation_id not in memory.learned_observation_ids


def test_internal_commit_entry_cannot_skip_final_target_digest_check() -> None:
    """Every callable commit entry must preserve the final target check."""

    memory, projection, _runner, adapter, target = _committed_target()
    retraction = adapter.issue_retraction(projection.project(), target)
    _mutate_and_rehash(target, timestamp=703)

    with pytest.raises(NativeLearnedProjectionBindingError, match="issued content"):
        adapter.commit(retraction)
    assert target.observation_id in memory.learned_observation_ids

    try:
        receipt = adapter._commit_core_v3(retraction)
    except (AttributeError, PermissionError, NativeLearnedProjectionBindingError):
        return
    assert target.observation_id in memory.learned_observation_ids, (
        "the internal commit entry skipped the final target digest check and "
        f"removed active evidence with receipt {receipt.transaction_sha256}"
    )


def test_late_memory_rejection_releases_target_for_fresh_retraction() -> None:
    """A terminal late rejection must not leave the target slot occupied."""

    memory, projection, _runner, adapter, target = _committed_target()
    colliding_id = (
        "qualified-native-v4-v3-retract:1:" f"{target.observation_id}"
    )
    _seed_retraction_observation_collision(memory, colliding_id)

    retraction = adapter.issue_retraction(projection.project(), target)
    revision_before = memory.revision
    with pytest.raises(TransactionRejectedError, match="duplicate observation identity"):
        adapter.commit(retraction)
    assert memory.revision == revision_before
    assert target.observation_id in memory.learned_observation_ids

    replacement = adapter.issue_retraction(projection.project(), target)
    assert type(replacement) is QualifiedLearnedPhysicalDevelopmentTransactionV3


def test_authority_surfaces_remain_explicitly_development_only() -> None:
    _memory, _projection, snapshot, runner, adapter = _stack_v3()
    package = adapter.issue(
        snapshot,
        _outcome(runner, snapshot, ground=(_ground_row(),)),
    )
    for surface in (adapter, package, package.admission, package.projection_receipt):
        assert surface.development_only is True
        assert surface.hardware_execution_authorized is False
        assert surface.production_promotion_authorized is False
    assert (
        v3_module.PRODUCTION_NATIVE_V4_RUNNER,
        v3_module.PRODUCTION_V4_CHECKPOINT_FILE_SHA256,
        v3_module.PRODUCTION_G2_REPORT_FILE_SHA256,
        v3_module.PRODUCTION_V4_CALIBRATION_SHA256,
        v3_module.PRODUCTION_NATIVE_LEARNED_PROJECTION_ADAPTER_V3,
    ) == (None,) * 5
