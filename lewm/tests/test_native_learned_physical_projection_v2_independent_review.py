"""Independent adversarial review probes for the additive V2 candidate.

These tests do not edit or bless the candidate.  They exercise only synthetic
fixtures and state the authority and lifecycle properties required before a
native learned projection can be integrated downstream.
"""
from __future__ import annotations

import copy
import pickle

import pytest

from lewm.planning import native_learned_physical_projection_v1 as v1_module
from lewm.planning import native_learned_physical_projection_v2 as v2_module
from lewm.planning.native_learned_physical_projection_v1 import (
    NativeLearnedProjectionBindingError,
    NativeLearnedProjectionReplayError,
)
from lewm.planning.native_learned_physical_projection_v2 import (
    NativeLearnedPhysicalProjectionAdapterV2,
    require_production_native_learned_projection_adapter_v2,
)
from lewm.planning.revisioned_physical_configuration_memory import (
    SnapshotBindingError,
)
from lewm.tests.test_native_learned_physical_projection_v1 import (
    CALIBRATION,
    CAMERA_TRANSFORM_SHA256,
    IDENTITIES,
    _center_hit,
    _ground_row,
    _outcome,
    _pose,
    _stack,
)


def _stack_v2():
    memory, projection, snapshot, runner, _v1_adapter = _stack()
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


def _committed_target():
    memory, projection, snapshot, runner, adapter = _stack_v2()
    target = adapter.issue(
        snapshot,
        _outcome(runner, snapshot, ground=(_ground_row(),)),
    )
    adapter.commit(target)
    return memory, projection, runner, adapter, target


def _mutate_and_rehash_target(target) -> None:
    replacement_pose = _pose(timestamp=101)
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


def _advance_with_distinct_observation(
    projection,
    snapshot,
    runner,
    adapter,
    *,
    sequence: int,
):
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


def test_original_target_digest_is_enforced_at_issue_and_commit() -> None:
    memory, projection, _runner, adapter, target = _committed_target()
    issued_digest = target.content_sha256
    _mutate_and_rehash_target(target)
    assert target.content_sha256 != issued_digest

    with pytest.raises(
        NativeLearnedProjectionBindingError,
        match="issued content",
    ):
        adapter.issue_retraction(projection.project(), target)
    assert target.observation_id in memory.learned_observation_ids

    memory2, projection2, _runner2, adapter2, target2 = _committed_target()
    retraction = adapter2.issue_retraction(projection2.project(), target2)
    _mutate_and_rehash_target(target2)
    with pytest.raises(
        NativeLearnedProjectionBindingError,
        match="issued content",
    ):
        adapter2.commit(retraction)
    assert target2.observation_id in memory2.learned_observation_ids


def test_copy_forgery_rehash_transfer_and_replay_cannot_remove_target() -> None:
    memory, projection, runner, adapter, target = _committed_target()
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

    foreign = NativeLearnedPhysicalProjectionAdapterV2(
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


def test_stale_commit_release_allows_retry_and_old_package_never_resurrects() -> None:
    memory, projection, runner, adapter, target = _committed_target()
    shared = projection.project()
    stale = adapter.issue_retraction(shared, target)
    _advance_with_distinct_observation(
        projection,
        shared,
        runner,
        adapter,
        sequence=2,
    )

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
    for _ in range(2):
        projection.project()
        with pytest.raises(
            NativeLearnedProjectionReplayError,
            match="terminally stale",
        ):
            adapter.commit(stale)


def test_proactive_stale_release_preserves_one_live_retraction() -> None:
    memory, projection, _runner, adapter, target = _committed_target()
    first_snapshot = projection.project()
    stale = adapter.issue_retraction(first_snapshot, target)
    current = projection.project()
    replacement = adapter.issue_retraction(current, target)

    with pytest.raises(
        NativeLearnedProjectionReplayError,
        match="exact live retraction",
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


def test_target_removal_is_atomic_and_preserves_other_active_identity() -> None:
    memory, projection, runner, adapter, target = _committed_target()
    other_snapshot = projection.project()
    other = _advance_with_distinct_observation(
        projection,
        other_snapshot,
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


def test_v2_contract_is_distinct_bound_and_has_no_production_authority() -> None:
    _memory, _projection, snapshot, runner, adapter = _stack_v2()
    package = adapter.issue(
        snapshot,
        _outcome(runner, snapshot, ground=(_ground_row(),)),
    )
    _m1, _p1, _s1, _r1, v1_adapter = _stack()

    assert adapter.adapter_contract_sha256 != v1_adapter.adapter_contract_sha256
    assert package.admission.adapter_contract_sha256 == adapter.adapter_contract_sha256
    for surface in (adapter, package, package.admission, package.projection_receipt):
        assert surface.development_only is True
        assert surface.hardware_execution_authorized is False
        assert surface.production_promotion_authorized is False
    assert (
        v2_module.PRODUCTION_NATIVE_V4_RUNNER,
        v2_module.PRODUCTION_V4_CHECKPOINT_FILE_SHA256,
        v2_module.PRODUCTION_G2_REPORT_FILE_SHA256,
        v2_module.PRODUCTION_V4_CALIBRATION_SHA256,
        v2_module.PRODUCTION_NATIVE_LEARNED_PROJECTION_ADAPTER_V2,
    ) == (None,) * 5
    with pytest.raises(PermissionError, match="identities are unset"):
        require_production_native_learned_projection_adapter_v2()


def test_composed_v1_engine_cannot_bypass_v2_retraction_authority() -> None:
    """A reachable V1 engine must not provide a second public commit path."""

    memory, projection, _runner, adapter, target = _committed_target()
    _mutate_and_rehash_target(target)
    current = projection.project()
    with pytest.raises(NativeLearnedProjectionBindingError, match="issued content"):
        adapter.issue_retraction(current, target)
    assert target.observation_id in memory.learned_observation_ids

    try:
        inner = object.__getattribute__(
            adapter,
            "_NativeLearnedPhysicalProjectionAdapterV2__inner",
        )
    except AttributeError:
        return

    try:
        bypass = inner.issue_retraction(current, target)
        receipt = inner.commit(bypass)
    except (AttributeError, PermissionError, NativeLearnedProjectionBindingError):
        return

    assert target.observation_id in memory.learned_observation_ids, (
        "reachable composed V1 engine bypassed V2 and removed the mutated "
        f"target with receipt {receipt.transaction_sha256}"
    )


def test_adapter_and_packages_deny_copy_serialization_authority() -> None:
    _memory, _projection, snapshot, runner, adapter = _stack_v2()
    package = adapter.issue(
        snapshot,
        _outcome(runner, snapshot, ground=(_ground_row(),)),
    )
    with pytest.raises(TypeError, match="non-copyable"):
        copy.copy(adapter)
    with pytest.raises(TypeError, match="non-serializable"):
        pickle.dumps(adapter)
    with pytest.raises(TypeError, match="non-copyable"):
        copy.copy(package)
