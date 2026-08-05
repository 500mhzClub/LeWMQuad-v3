"""Focused tests for the immutable append-only observation identity query."""
from __future__ import annotations

import pytest

from lewm.planning.revisioned_physical_configuration_memory import (
    PhysicalLabel,
    RevisionedPhysicalMemory,
    TransactionRejectedError,
)
from lewm.tests.test_revisioned_physical_configuration_memory import (
    _memory,
    _transaction,
)


def _accept_then_retract() -> RevisionedPhysicalMemory:
    memory = _memory()
    memory.apply_transaction(
        _transaction(
            memory,
            "accepted",
            evidence={(0, 0): PhysicalLabel.FREE},
        )
    )
    memory.apply_transaction(
        _transaction(
            memory,
            "accepted-retraction",
            retractions=("accepted",),
        )
    )
    return memory


def test_seen_observation_ids_is_immutable_and_append_only() -> None:
    memory = _memory()
    initial = memory.seen_observation_ids
    assert initial == frozenset()
    with pytest.raises(AttributeError):
        initial.add("forged")  # type: ignore[attr-defined]

    memory.apply_transaction(
        _transaction(memory, "first", evidence={(0, 0): PhysicalLabel.FREE})
    )
    first_view = memory.seen_observation_ids
    memory.apply_transaction(
        _transaction(memory, "second", evidence={(1, 0): PhysicalLabel.FREE})
    )

    assert initial == frozenset()
    assert first_view == frozenset({"first"})
    assert memory.seen_observation_ids == frozenset({"first", "second"})


def test_retracted_identity_remains_seen_and_duplicate_rejection_is_atomic() -> None:
    memory = _accept_then_retract()
    assert "accepted" not in memory.learned_observation_ids
    assert memory.seen_observation_ids == frozenset(
        {"accepted", "accepted-retraction"}
    )

    revision_before = memory.revision
    history_before = memory.seen_observation_ids
    with pytest.raises(TransactionRejectedError, match="duplicate observation"):
        memory.apply_transaction(
            _transaction(
                memory,
                "accepted",
                evidence={(2, 0): PhysicalLabel.FREE},
            )
        )
    assert memory.revision == revision_before
    assert memory.seen_observation_ids == history_before


def test_seen_observation_ids_survives_strict_serialization_roundtrip() -> None:
    memory = _accept_then_retract()
    restored = RevisionedPhysicalMemory.deserialize(memory.serialize())

    assert restored.seen_observation_ids == memory.seen_observation_ids
    assert restored.learned_observation_ids == memory.learned_observation_ids
    assert restored.serialize() == memory.serialize()
