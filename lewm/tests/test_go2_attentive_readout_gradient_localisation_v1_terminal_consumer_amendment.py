"""Focused tests for the read-only gradient terminal consumer amendment."""
from __future__ import annotations

import copy
from pathlib import Path

import pytest

from lewm.oracle import (
    go2_attentive_readout_gradient_localisation_v1_contract as C,
)
from lewm.oracle import (
    go2_attentive_readout_gradient_localisation_v1_terminal_consumer_amendment
    as A,
)
from scripts import diagnose_go2_attentive_readout_gradient_localisation_v1 as D


ROOT = Path(__file__).resolve().parents[2]


def _source_closure() -> dict:
    payload = {
        "schema": A.SOURCE_CLOSURE_SCHEMA,
        "source_repository_commit": "a" * 40,
        "source_repository_clean": True,
        "base_source_commit": A.SOURCE_BASE_COMMIT,
        "exact_committed_additive_path_diff": list(A.NEW_SOURCE_PATHS),
        "frozen_gradient_localisation_files": {},
        "additive_terminal_consumer_files": {},
    }
    return {**payload, A.SOURCE_CLOSURE_SELF_KEY: A.digest(payload)}


def test_count_projection_is_exact_and_matches_frozen_terminal() -> None:
    frozen = A.validate_frozen_runtime_bytes(ROOT)
    planned = frozen["contract"]["execution_counts"]
    terminal_counts = frozen["artifacts"]["terminal.json"][
        "execution_counts"]
    projected = A.translate_planned_execution_counts(planned)
    assert projected == terminal_counts
    assert set(projected) == {
        "backward_attempts", "batch_presentations", "completed_backwards",
        "completed_forwards", "examples_presented",
        "fixture_validation_latent_shard_opens",
        "fixture_validation_row_record_opens", "forward_attempts",
        "fresh_model_constructions", "gradient_clips",
        "optimizer_constructions", "optimizer_steps",
        "pass_latent_shard_loads", "unique_fit_latent_shard_files",
        "unique_fit_row_record_files",
    }


def test_count_projection_rejects_schema_or_budget_changes() -> None:
    planned = dict(C.EXECUTION_COUNTS)
    changed = dict(planned)
    changed["forwards"] = 9
    with pytest.raises(A.TerminalConsumerAmendmentError,
                       match="internally inconsistent"):
        A.translate_planned_execution_counts(changed)
    changed = dict(planned)
    changed["unregistered_counter"] = 0
    with pytest.raises(A.TerminalConsumerAmendmentError,
                       match="schema changed"):
        A.translate_planned_execution_counts(changed)


def test_equality_adapter_accepts_only_the_exact_equivalent_pair() -> None:
    planned = dict(C.EXECUTION_COUNTS)
    dynamic = A.translate_planned_execution_counts(planned)
    adapter = A.EquivalentExecutionCounts(planned)
    assert planned == adapter
    assert adapter == planned
    assert dynamic == adapter
    assert adapter == dynamic
    near = dict(dynamic)
    near["completed_backwards"] -= 1
    assert not near == adapter
    assert not adapter == near


def test_all_eight_runtime_artifacts_remain_exact_and_read_only() -> None:
    frozen = A.validate_frozen_runtime_bytes(ROOT)
    assert set(frozen["artifacts"]) == set(A.FROZEN_ARTIFACTS)
    assert frozen["contract"][C.CONTRACT_SELF_KEY] == (
        A.FROZEN_ARTIFACTS["contract.json"]["self_digest"])
    runtime = A.runtime_root(ROOT)
    for name, expected in A.FROZEN_ARTIFACTS.items():
        path = runtime / name
        assert path.stat().st_size == expected["byte_count"]
        assert A.file_sha256(path) == expected["sha256"]
        assert path.stat().st_mode & 0o222 == 0


def test_frozen_validator_reproduces_only_the_consumer_exception(
        monkeypatch: pytest.MonkeyPatch) -> None:
    frozen = A.validate_frozen_runtime_bytes(ROOT)
    A.validate_frozen_consumer_defect(ROOT, frozen["contract"])
    original_loader = D.load_installed_contract
    original_counts = C.EXECUTION_COUNTS
    with pytest.raises(RuntimeError, match="sentinel"):
        with A._translated_frozen_validator(
                frozen["contract"], frozen["contract"]["execution_counts"]):
            raise RuntimeError("sentinel")
    assert D.load_installed_contract is original_loader
    assert C.EXECUTION_COUNTS is original_counts


def test_complete_frozen_validator_recomputes_terminal_after_projection() -> None:
    terminal = A.validate_completed_terminal(ROOT)
    assert terminal[D.TERMINAL_SELF_KEY] == (
        "7ec0c9d5cd01c965568f38ca7c5e119e0f7fb74b65dc0f909bdba09f98b26187")
    assert terminal["terminal_kind"] == "COMPLETED_MECHANISM_CLASSIFICATION"
    assert terminal["mechanism_classification"] == (
        "ARCHITECTURE_OR_OBJECTIVE_CHANGE_REQUIRED")
    assert terminal["completed_passes"] == list(D.PASS_ORDER)
    assert terminal["later_repair_gate"] == {
        "automatic_repair_or_training": False,
        "classification_can_support_separate_repair_decision": False,
        "repair_authorised_now": False,
        "training_authorised_now": False,
    }


def test_in_memory_consumer_receipt_binds_defect_and_grants_no_authority() -> None:
    terminal = A.validate_completed_terminal(ROOT)
    receipt = A.build_consumer_receipt(_source_closure(), terminal)
    assert A.validate_signed(receipt, A.RECEIPT_SELF_KEY, "receipt") == receipt
    predicate = receipt["consumer_predicate_amendment"]
    assert predicate["frozen_validator_exception_type"] == (
        "GradientLocalisationError")
    assert predicate["frozen_validator_exception_message"] == (
        "completed matrix SDPA ledger changed")
    assert predicate["scientific_calculation_changed"] is False
    assert predicate["runtime_artifact_changed"] is False
    assert set(receipt["authority"].values()) == {False}


def test_source_scope_is_two_additive_files_and_contains_no_write_route() -> None:
    assert len(A.NEW_SOURCE_PATHS) == 2
    assert Path(__file__).relative_to(ROOT).as_posix() in A.NEW_SOURCE_PATHS
    source = (ROOT / A.NEW_SOURCE_PATHS[0]).read_text(encoding="utf-8")
    assert "write_text(" not in source
    assert "publish_json" not in source
    assert "torch." not in source
    assert "model(" not in source
    assert "repair_authorised_now\": True" not in source
    assert "training_authorised_now\": True" not in source


def test_signed_receipt_rejects_terminal_or_source_tampering() -> None:
    terminal = A.validate_completed_terminal(ROOT)
    changed_terminal = copy.deepcopy(terminal)
    changed_terminal["mechanism_classification"] = (
        "BACKEND_NUMERICAL_DEFECT_CONTRACT_PRESERVING")
    with pytest.raises(A.TerminalConsumerAmendmentError,
                       match="terminal binding changed"):
        A.build_consumer_receipt(_source_closure(), changed_terminal)
    changed_source = _source_closure()
    changed_source["source_repository_clean"] = False
    with pytest.raises(A.TerminalConsumerAmendmentError,
                       match="source closure is invalid"):
        A.build_consumer_receipt(changed_source, terminal)
