"""Independent adversarial review for the frozen navigation coordinator V3."""
from __future__ import annotations

import copy
import hashlib
from pathlib import Path
import pickle

import pytest

from lewm.benchmarks.go2_physical_claim_observer import (
    empty_evaluator_access_ledger,
)
from lewm.planning import (
    two_resolution_navigation_development_integration_v3 as module,
)
from lewm.planning.two_resolution_navigation_development_integration_v1 import (
    TwoResolutionDevelopmentNavigationIntegrationV1,
)
from lewm.planning.two_resolution_navigation_development_integration_v2 import (
    TwoResolutionDevelopmentNavigationIntegrationV2,
)
from lewm.planning.two_resolution_navigation_development_integration_v3 import (
    TwoResolutionNavigationIntegrationV3BindingError,
    require_production_two_resolution_navigation_integration_v3,
)
from lewm.tests.test_two_resolution_navigation_development_integration_v3 import (
    _assert_uncommitted,
    _issue_v3,
    _next_red_outcome,
    _v3_bundle,
)


SOURCE_SHA256 = "6d8b00aa8ffaa0117efc01baa218cadd299a871732e86d2751e51463520d6523"
AUTHOR_TEST_SHA256 = (
    "d2af0e5a798ff6d186813d6054588e460cda37bb7989697261125a64d0265a54"
)
AUTHOR_HANDOFF_SHA256 = (
    "df7c9234edc06b53b43a395632887dd258102beb8fd7f3776bc0a50ef8c6abe6"
)


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _mapping_specs(state):
    evidence = state.evidence_issuer
    source = getattr(evidence, "_outcome_source")
    memory = state.target_memory
    view = state.view_issuer
    frontier = state.frontier
    router = state.target_router
    waypoint = state.waypoint_issuer
    return (
        (state, "issued"),
        (state, "observer_results"),
        (state, "known_outcomes"),
        (state.planner, "_issued_components"),
        (state.planner, "_issued_frontiers"),
        (state.planner, "_issued_paths"),
        (view, "_configuration_history"),
        (view, "_records"),
        (view, "_issued_states"),
        (frontier, "_issued_sets"),
        (frontier, "_issued_candidates"),
        (frontier, "_score_cache"),
        (source, "_issued"),
        (evidence, "_contexts"),
        (evidence, "_evidence"),
        (memory, "_mass"),
        (memory, "_unlocalized"),
        (memory, "_contexts"),
        (memory, "_positive_count"),
        (memory, "_negative_count"),
        (memory, "_chains"),
        (memory, "_issued_snapshots"),
        (router, "_plans"),
        (router, "_issued_content_sha256"),
        (waypoint, "_issued"),
    )


def _set_specs(state):
    evidence = state.evidence_issuer
    source = getattr(evidence, "_outcome_source")
    memory = state.target_memory
    return (
        (state, "observed_controller_ids"),
        (state, "consumed_observer_result_ids"),
        (state.view_issuer, "_swept_physical_cells"),
        (source, "_consumed"),
        (evidence, "_consumed_evidence"),
        (memory, "_seen_context_ids"),
        (memory, "_seen_evidence_hashes"),
        (memory, "_seen_raw_outcomes"),
        (state.target_router, "_consumed"),
        (state.waypoint_issuer, "_consumed"),
    )


def _scalar_specs(state):
    evidence = state.evidence_issuer
    source = getattr(evidence, "_outcome_source")
    memory = state.target_memory
    view = state.view_issuer
    return (
        (state, "owner_state_sha256"),
        (state, "known_outcome_sequence"),
        (view, "_view_revision"),
        (view, "_view_step"),
        (source, "_sequence"),
        (evidence, "_sequence"),
        (memory, "_revision"),
        (memory, "_last_context_sequence"),
        (memory, "_last_pose_timestamp_ns"),
        (memory, "_immutable_binding"),
    )


def _capture_independent_state(state) -> dict[str, object]:
    mappings = tuple(
        (owner, name, getattr(owner, name), dict(getattr(owner, name)))
        for owner, name in _mapping_specs(state)
    )
    sets = tuple(
        (owner, name, getattr(owner, name), set(getattr(owner, name)))
        for owner, name in _set_specs(state)
    )
    mass = getattr(state.target_memory, "_mass")
    nested_mass = tuple((rows, dict(rows)) for rows in mass.values())
    contexts = tuple(
        (
            row,
            row.writer,
            row.evidence,
            None if row.writer is None else row.writer._used,
        )
        for _context, row in getattr(state.evidence_issuer, "_contexts").values()
    )
    return {
        "mappings": mappings,
        "sets": sets,
        "scalars": tuple(
            (owner, name, getattr(owner, name))
            for owner, name in _scalar_specs(state)
        ),
        "nested_mass": nested_mass,
        "contexts": contexts,
        "fingerprint": module._transaction_owner_state_sha256(state),
    }


def _assert_mapping_values_exact(
    current: dict[object, object],
    expected: dict[object, object],
) -> None:
    assert current.keys() == expected.keys()
    immutable = (str, bytes, int, float, bool, tuple, frozenset, type(None))
    for key, value in expected.items():
        if isinstance(value, immutable):
            assert current[key] == value
        else:
            assert current[key] is value


def _assert_independent_state_restored(state, captured: dict[str, object]) -> None:
    for owner, name, container, values in captured["mappings"]:
        current = getattr(owner, name)
        # The append-only verifier deliberately refreshes this redundant cache
        # before the transaction snapshot. Its exact semantic rows, not its
        # private container identity, are the invariant under review.
        if owner is state and name == "known_outcomes":
            assert type(current) is dict
        else:
            assert current is container
        _assert_mapping_values_exact(current, values)
    for owner, name, container, values in captured["sets"]:
        assert getattr(owner, name) is container
        assert container == values
    for owner, name, value in captured["scalars"]:
        current = getattr(owner, name)
        if isinstance(value, (dict, list, set)):
            assert current is value
        else:
            assert current == value
    for container, values in captured["nested_mass"]:
        assert container == values
    for row, writer, evidence, writer_used in captured["contexts"]:
        assert row.writer is writer
        assert row.evidence is evidence
        if writer is not None:
            assert writer._used is writer_used
    assert module._transaction_owner_state_sha256(state) == captured["fingerprint"]


def test_independent_review_freezes_exact_candidate_identities() -> None:
    root = Path(__file__).resolve().parents[2]
    assert _file_sha256(Path(module.__file__)) == SOURCE_SHA256
    assert _file_sha256(
        root / "lewm/tests/test_two_resolution_navigation_development_integration_v3.py"
    ) == AUTHOR_TEST_SHA256
    assert _file_sha256(
        root
        / "docs/lewm_go2_two_resolution_navigation_development_integration_v3_handoff_2026-07-13.md"
    ) == AUTHOR_HANDOFF_SHA256


@pytest.mark.parametrize(
    "fault_after_stage",
    sorted(module._SYNTHETIC_TRANSACTION_FAULT_STAGES),
)
def test_independent_all_21_stages_restore_exact_state_and_retry(
    fault_after_stage: str,
) -> None:
    bundle = _v3_bundle(fault_after_stage=fault_after_stage)
    state = module._integration_state(bundle.integration_v3)
    captured = _capture_independent_state(state)

    with pytest.raises(
        RuntimeError,
        match=f"synthetic V3 transaction fault after {fault_after_stage}",
    ):
        _issue_v3(bundle)

    _assert_independent_state_restored(state, captured)
    _assert_uncommitted(bundle)
    artifact = _issue_v3(bundle)
    artifact.assert_integrity()
    assert bundle.integration_v3.issued_controller_count == 1


@pytest.mark.parametrize(
    "fault_after_stage",
    (
        "controller_record_construct",
        "controller_registry_insert",
        "coordinator_seal_assign",
    ),
)
def test_independent_final_stage_rollback_preserves_nonempty_registry_and_retry(
    fault_after_stage: str,
) -> None:
    bundle = _v3_bundle()
    first = _issue_v3(bundle)
    state = module._integration_state(bundle.integration_v3)
    first_row = state.issued[id(first)]
    registry = state.issued
    next_outcome = _next_red_outcome(bundle)
    bundle.integration_v3.development_owner_state_audit_sha256
    captured = _capture_independent_state(state)
    state.synthetic_fault_after_stage = fault_after_stage

    with pytest.raises(
        RuntimeError,
        match=f"synthetic V3 transaction fault after {fault_after_stage}",
    ):
        _issue_v3(
            bundle,
            outcome=next_outcome,
            trace_id="independent-second-fault-trace",
            event_id="independent-second-fault-claim",
            tick=26,
        )

    _assert_independent_state_restored(state, captured)
    assert state.issued is registry
    assert state.issued[id(first)] is first_row
    assert id(next_outcome) not in bundle.outcome_source._consumed
    first.assert_integrity()

    second = _issue_v3(
        bundle,
        outcome=next_outcome,
        trace_id="independent-second-retry-trace",
        event_id="independent-second-retry-claim",
        tick=26,
    )
    second.assert_integrity()
    assert bundle.integration_v3.issued_controller_count == 2


def test_independent_v2_constructor_blocker_is_inside_rollback_and_retry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bundle = _v3_bundle()
    state = module._integration_state(bundle.integration_v3)
    captured = _capture_independent_state(state)
    original_record_type = module._ControllerRecordV3

    def fail_record(**_values: object) -> object:
        raise RuntimeError("independent late controller-record failure")

    monkeypatch.setattr(module, "_ControllerRecordV3", fail_record)
    with pytest.raises(RuntimeError, match="late controller-record failure"):
        _issue_v3(bundle)

    _assert_independent_state_restored(state, captured)
    _assert_uncommitted(bundle)
    monkeypatch.setattr(module, "_ControllerRecordV3", original_record_type)
    artifact = _issue_v3(bundle)
    artifact.assert_integrity()


def test_independent_transaction_fingerprint_covers_registry_and_stored_seal() -> None:
    bundle = _v3_bundle()
    artifact = _issue_v3(bundle)
    state = module._integration_state(bundle.integration_v3)
    baseline = module._transaction_owner_state_sha256(state)
    row = state.issued[id(artifact)]

    state.issued.pop(id(artifact))
    assert module._transaction_owner_state_sha256(state) != baseline
    state.issued[id(artifact)] = row
    assert module._transaction_owner_state_sha256(state) == baseline

    original_seal = state.owner_state_sha256
    state.owner_state_sha256 = "f" * 64
    assert module._transaction_owner_state_sha256(state) != baseline
    with pytest.raises(
        TwoResolutionNavigationIntegrationV3BindingError,
        match="shared downstream owner state changed",
    ):
        bundle.integration_v3.development_owner_state_audit_sha256
    state.owner_state_sha256 = original_seal
    assert module._transaction_owner_state_sha256(state) == baseline

    state.issued[id(artifact)] = module._ControllerRecordV3(
        artifact=artifact,
        original_content_sha256="e" * 64,
        episode_authority=row.episode_authority,
    )
    assert module._transaction_owner_state_sha256(state) != baseline
    state.issued[id(artifact)] = row
    assert module._transaction_owner_state_sha256(state) == baseline


def test_independent_legacy_isolation_and_production_fail_closed() -> None:
    source = Path(module.__file__).read_text(encoding="utf-8")
    for forbidden in (
        "two_resolution_navigation_development_integration_v1",
        "two_resolution_navigation_development_integration_v2",
        "TwoResolutionDevelopmentNavigationIntegrationV1",
        "TwoResolutionDevelopmentNavigationIntegrationV2",
    ):
        assert forbidden not in source

    bundle = _v3_bundle()
    artifact = _issue_v3(bundle)
    assert not isinstance(
        bundle.integration_v3,
        (
            TwoResolutionDevelopmentNavigationIntegrationV1,
            TwoResolutionDevelopmentNavigationIntegrationV2,
        ),
    )
    with pytest.raises((AttributeError, TypeError, ValueError)):
        TwoResolutionDevelopmentNavigationIntegrationV1.assert_controller_claim_trace(
            bundle.integration_v3,
            artifact,
        )
    with pytest.raises((AttributeError, TypeError, ValueError)):
        TwoResolutionDevelopmentNavigationIntegrationV2._controller_record(
            bundle.integration_v3,
            artifact,
        )
    with pytest.raises((AttributeError, TypeError, ValueError)):
        TwoResolutionDevelopmentNavigationIntegrationV2.evaluate_observer_only(
            bundle.integration_v3,
            artifact,
            evaluator_access_ledger=empty_evaluator_access_ledger(),
        )

    assert bundle.integration_v3.production_eligible is False
    assert bundle.integration_v3.hardware_execution_authorized is False
    assert module.PRODUCTION_TWO_RESOLUTION_NAVIGATION_INTEGRATION_V3 is None
    with pytest.raises(PermissionError):
        require_production_two_resolution_navigation_integration_v3()
    with pytest.raises(TypeError, match="non-copyable"):
        copy.copy(bundle.integration_v3)
    with pytest.raises(TypeError, match="non-copyable"):
        copy.deepcopy(bundle.integration_v3)
    with pytest.raises(TypeError, match="non-serializable"):
        pickle.dumps(bundle.integration_v3)
