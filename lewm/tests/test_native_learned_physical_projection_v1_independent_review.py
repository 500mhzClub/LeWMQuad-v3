"""Independent adversarial review probes for the frozen V1 candidate.

These tests are intentionally outside the frozen candidate test artifact. They
state the lifecycle guarantees required by the preregistered G3 projection
plan; a failure is a review finding rather than a modification of V1.
"""
from __future__ import annotations

import pickle

import pytest

from lewm.planning import native_learned_physical_projection_v1 as module
from lewm.planning.native_learned_physical_projection_v1 import (
    NativeLearnedProjectionBindingError,
)
from lewm.planning.revisioned_physical_configuration_memory import (
    SnapshotBindingError,
)
from lewm.tests.test_native_learned_physical_projection_v1 import (
    _center_hit,
    _ground_row,
    _outcome,
    _pose,
    _stack,
)


def test_serialized_reload_and_object_new_forgery_are_rejected() -> None:
    _memory, _projection, snapshot, runner, adapter = _stack()
    outcome = _outcome(runner, snapshot, ground=(_ground_row(),))

    reloaded_outcome = pickle.loads(pickle.dumps(outcome))
    with pytest.raises(NativeLearnedProjectionBindingError, match="exact live"):
        adapter.issue(snapshot, reloaded_outcome)

    package = adapter.issue(snapshot, outcome)
    reloaded_package = pickle.loads(pickle.dumps(package))
    with pytest.raises(NativeLearnedProjectionBindingError, match="exact live"):
        adapter.commit(reloaded_package)

    forged = object.__new__(type(package))
    for name, value in package.__dict__.items():
        object.__setattr__(forged, name, value)
    with pytest.raises(NativeLearnedProjectionBindingError, match="exact live"):
        adapter.commit(forged)


def test_retraction_rejects_rehashed_post_commit_package_mutation() -> None:
    _memory, projection, snapshot, runner, adapter = _stack()
    outcome = _outcome(runner, snapshot, ground=(_ground_row(),))
    committed = adapter.issue(snapshot, outcome)
    adapter.commit(committed)

    replacement_pose = _pose(timestamp=99)
    object.__setattr__(committed, "pose", replacement_pose)
    object.__setattr__(
        committed.admission,
        "pose_sha256",
        replacement_pose.content_sha256,
    )
    object.__setattr__(
        committed.admission,
        "content_sha256",
        module._sha256(committed.admission.to_dict(False)),
    )
    object.__setattr__(
        committed,
        "content_sha256",
        module._sha256(committed.to_dict(False)),
    )

    current = projection.project()
    with pytest.raises(
        NativeLearnedProjectionBindingError,
        match="issued content",
    ):
        adapter.issue_retraction(current, committed)


def test_stale_retraction_can_be_reissued_for_active_evidence() -> None:
    memory, projection, snapshot, runner, adapter = _stack()
    original_outcome = _outcome(
        runner,
        snapshot,
        ground=(_ground_row(),),
        sequence=1,
    )
    original = adapter.issue(snapshot, original_outcome)
    adapter.commit(original)

    shared_snapshot = projection.project()
    stale_retraction = adapter.issue_retraction(shared_snapshot, original)

    intervening_outcome = _outcome(
        runner,
        shared_snapshot,
        pose=_pose(timestamp=2),
        rays=(_center_hit(),),
        sequence=2,
    )
    intervening = adapter.issue(shared_snapshot, intervening_outcome)
    adapter.commit(intervening)

    with pytest.raises(SnapshotBindingError):
        adapter.commit(stale_retraction)
    assert original.observation_id in memory.learned_observation_ids

    current = projection.project()
    replacement = adapter.issue_retraction(current, original)
    receipt = adapter.commit(replacement)
    assert receipt.learned_observations_retracted == 1
    assert original.observation_id not in memory.learned_observation_ids
