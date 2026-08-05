"""Independent lifecycle review for native learned physical projection V5.

The review uses only synthetic development fixtures. Candidate source, author
tests, and handoff bytes are frozen and remain unmodified.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from lewm.planning import native_learned_physical_projection_v5 as v5_module
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
)
from lewm.planning.native_learned_physical_projection_v5 import (
    NativeLearnedPhysicalProjectionAdapterV5,
    QualifiedLearnedPhysicalDevelopmentTransactionV5,
    require_production_native_learned_projection_adapter_v5,
)
from lewm.planning.revisioned_physical_configuration_memory import (
    RevisionedPhysicalMemory,
    TransactionRejectedError,
)
from lewm.tests.test_native_learned_physical_projection_v1 import (
    CALIBRATION,
    CAMERA_TRANSFORM_SHA256,
    IDENTITIES,
    _center_hit,
    _ground_row,
    _outcome,
)
from lewm.tests.test_native_learned_physical_projection_v5 import (
    _committed_projection_v5,
    _mutate_and_rehash_v5,
    _retract_seeded_identity_v5,
    _seed_retraction_identity_collision_v5,
    _stack_v5,
)


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "lewm/planning/native_learned_physical_projection_v5.py"
MEMORY_SOURCE = ROOT / "lewm/planning/revisioned_physical_configuration_memory.py"
AUTHOR_TEST = ROOT / "lewm/tests/test_native_learned_physical_projection_v5.py"
HISTORY_TEST = (
    ROOT / "lewm/tests/test_revisioned_physical_memory_seen_observation_ids.py"
)
HANDOFF = (
    ROOT
    / "docs/lewm_go2_g3_native_learned_physical_projection_v5_author_handoff_2026-07-13.md"
)
FROZEN_SHA256 = {
    SOURCE: "5ccd22e83c83a4c41db11286d31d417fe7af5615ebd7e62e51d7719d5378eca1",
    MEMORY_SOURCE: "bb05f957e0443e0c1e8405042b97c61948746a66040e84690e12b0a10887d483",
    AUTHOR_TEST: "e5f0d30b96d1da525ac004ded1eac6bcca96330657d92571594914c548a6d077",
    HISTORY_TEST: "20860a1abca8848a5951481ce167da501420ce27ad21fba1c9821bc092459fa4",
    HANDOFF: "6fb25e5af95b5794a45e67c1167c7c618fe9fd5e9aab22fbd83d37bd2da661cc",
}


def _legacy_adapters(memory: object, projection: object, runner: object) -> tuple[object, ...]:
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
        NativeLearnedPhysicalProjectionAdapterV4(**kwargs),
    )


def test_v5_frozen_candidate_author_surfaces_and_handoff_match() -> None:
    for path, expected in FROZEN_SHA256.items():
        assert hashlib.sha256(path.read_bytes()).hexdigest() == expected


def test_history_query_is_immutable_nonmutating_and_roundtrips_exactly() -> None:
    memory, projection, _runner, adapter, target = _committed_projection_v5()
    encoded_before = memory.serialize()
    old_view = memory.seen_observation_ids

    assert type(old_view) is frozenset
    assert old_view == frozenset({target.observation_id})
    with pytest.raises(AttributeError):
        old_view.add("caller-forgery")  # type: ignore[attr-defined]
    assert memory.seen_observation_ids is not old_view
    assert memory.serialize() == encoded_before

    retraction = adapter.issue_retraction(projection.project(), target)
    adapter.commit(retraction)
    assert old_view == frozenset({target.observation_id})
    assert memory.seen_observation_ids == frozenset(
        {target.observation_id, retraction.observation_id}
    )
    encoded_after = memory.serialize()
    restored = RevisionedPhysicalMemory.deserialize(encoded_after)
    assert restored.serialize() == encoded_after
    assert restored.seen_observation_ids == memory.seen_observation_ids
    assert target.observation_id not in restored.learned_observation_ids


@pytest.mark.parametrize("retire_collision", [False, True])
def test_active_and_retired_duplicate_ids_are_terminal_and_release_target(
    retire_collision: bool,
) -> None:
    memory, projection, _runner, adapter, target = _committed_projection_v5()
    collision_id = f"qualified-native-v4-v5-retract:1:{target.observation_id}"
    _seed_retraction_identity_collision_v5(memory, collision_id)
    if retire_collision:
        _retract_seeded_identity_v5(memory, collision_id)

    assert collision_id in memory.seen_observation_ids
    assert (collision_id in memory.learned_observation_ids) is not retire_collision
    package = adapter.issue_retraction(projection.project(), target)
    state_before = memory.serialize()
    revision_before = memory.revision

    with pytest.raises(TransactionRejectedError, match="duplicate observation identity"):
        adapter.commit(package)
    assert memory.revision == revision_before
    assert memory.serialize() == state_before
    assert target.observation_id in memory.learned_observation_ids
    with pytest.raises(NativeLearnedProjectionReplayError, match="terminally stale"):
        adapter._commit_core_v5(package)

    replacement = adapter.issue_retraction(projection.project(), target)
    receipt = adapter.commit(replacement)
    assert receipt.revision_before == revision_before
    assert receipt.revision_after == revision_before + 1
    assert receipt.learned_observations_retracted == 1
    assert target.observation_id not in memory.learned_observation_ids


def test_unrelated_rejection_preserves_exact_retry_and_exclusive_reservation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    memory, projection, _runner, adapter, target = _committed_projection_v5()
    snapshot = projection.project()
    package = adapter.issue_retraction(snapshot, target)
    assert package.observation_id not in memory.seen_observation_ids
    state_before = memory.serialize()
    original_apply = RevisionedPhysicalMemory.apply_transaction

    def reject_only_this_memory(
        instance: RevisionedPhysicalMemory,
        transaction: object,
    ) -> object:
        if instance is memory:
            raise TransactionRejectedError("independent unrelated rejection")
        return original_apply(instance, transaction)  # type: ignore[arg-type]

    with monkeypatch.context() as patch:
        patch.setattr(
            RevisionedPhysicalMemory,
            "apply_transaction",
            reject_only_this_memory,
        )
        with pytest.raises(TransactionRejectedError, match="independent unrelated"):
            adapter._commit_core_v5(package)

    assert memory.serialize() == state_before
    with pytest.raises(NativeLearnedProjectionReplayError, match="exact live"):
        adapter.issue_retraction(snapshot, target)
    receipt = adapter.commit(package)
    assert receipt.learned_observations_retracted == 1
    assert target.observation_id not in memory.learned_observation_ids


def test_all_v5_commit_entry_points_recheck_final_target_binding() -> None:
    calls = (
        lambda adapter, package: adapter.commit(package),
        lambda adapter, package: adapter._commit_core_v5(package),
        lambda adapter, package: NativeLearnedPhysicalProjectionAdapterV5.commit(
            adapter,
            package,
        ),
        lambda adapter, package: (
            NativeLearnedPhysicalProjectionAdapterV5._commit_core_v5(
                adapter,
                package,
            )
        ),
    )
    for call in calls:
        memory, projection, _runner, adapter, target = _committed_projection_v5()
        package = adapter.issue_retraction(projection.project(), target)
        revision_before = memory.revision
        _mutate_and_rehash_v5(target, timestamp=911)

        with pytest.raises(NativeLearnedProjectionBindingError, match="issued content"):
            call(adapter, package)
        assert memory.revision == revision_before
        assert target.observation_id in memory.learned_observation_ids


def test_success_removes_one_target_in_one_revision_and_preserves_other() -> None:
    memory, projection, snapshot, runner, adapter = _stack_v5()
    first = adapter.issue(
        snapshot,
        _outcome(runner, snapshot, ground=(_ground_row(),), sequence=21),
    )
    adapter.commit(first)
    second_snapshot = projection.project()
    second = adapter.issue(
        second_snapshot,
        _outcome(runner, second_snapshot, rays=(_center_hit(),), sequence=22),
    )
    adapter.commit(second)
    active_before = memory.learned_observation_ids
    seen_before = memory.seen_observation_ids
    revision_before = memory.revision

    package = adapter.issue_retraction(projection.project(), first)
    receipt = adapter.commit(package)
    assert active_before == frozenset({first.observation_id, second.observation_id})
    assert receipt.revision_before == revision_before
    assert receipt.revision_after == revision_before + 1
    assert memory.revision == revision_before + 1
    assert receipt.learned_observations_retracted == 1
    assert memory.learned_observation_ids == frozenset({second.observation_id})
    assert memory.seen_observation_ids == seen_before | {package.observation_id}


def test_legacy_versions_cannot_issue_or_commit_v5_authority() -> None:
    memory, projection, runner, adapter, target = _committed_projection_v5()
    package = adapter.issue_retraction(projection.project(), target)
    revision_before = memory.revision
    legacy_instances = _legacy_adapters(memory, projection, runner)
    legacy_classes = (
        NativeLearnedPhysicalProjectionAdapterV1,
        NativeLearnedPhysicalProjectionAdapterV2,
        NativeLearnedPhysicalProjectionAdapterV3,
        NativeLearnedPhysicalProjectionAdapterV4,
    )

    for legacy in legacy_instances:
        with pytest.raises((TypeError, NativeLearnedProjectionBindingError)):
            legacy.commit(package)  # type: ignore[attr-defined]
        with pytest.raises((TypeError, NativeLearnedProjectionBindingError)):
            legacy.issue_retraction(  # type: ignore[attr-defined]
                projection.project(),
                target,
            )
    for legacy_class in legacy_classes:
        with pytest.raises((AttributeError, TypeError)):
            legacy_class.commit(adapter, package)
        with pytest.raises((AttributeError, TypeError)):
            legacy_class.issue_retraction(adapter, projection.project(), target)

    assert memory.revision == revision_before
    assert target.observation_id in memory.learned_observation_ids
    assert type(package) is QualifiedLearnedPhysicalDevelopmentTransactionV5


def test_v5_source_is_standalone_and_production_remains_unset() -> None:
    source = SOURCE.read_text(encoding="utf-8").lower()
    for forbidden in (
        "nativelearnedphysicalprojectionadapterv1",
        "nativelearnedphysicalprojectionadapterv2",
        "nativelearnedphysicalprojectionadapterv3",
        "nativelearnedphysicalprojectionadapterv4",
        "native_learned_physical_projection_v2",
        "native_learned_physical_projection_v3",
        "native_learned_physical_projection_v4",
    ):
        assert forbidden not in source
    assert (
        v5_module.PRODUCTION_NATIVE_V5_RUNNER,
        v5_module.PRODUCTION_V5_CHECKPOINT_FILE_SHA256,
        v5_module.PRODUCTION_G2_REPORT_FILE_SHA256,
        v5_module.PRODUCTION_V5_CALIBRATION_SHA256,
        v5_module.PRODUCTION_NATIVE_LEARNED_PROJECTION_ADAPTER_V5,
    ) == (None,) * 5
    with pytest.raises(PermissionError, match="identities are unset"):
        require_production_native_learned_projection_adapter_v5()
