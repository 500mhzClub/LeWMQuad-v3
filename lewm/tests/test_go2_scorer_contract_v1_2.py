"""Focused source-only tests for the current scorer-contract lineage."""
from __future__ import annotations

import copy
from pathlib import Path

import pytest

from lewm.oracle import go2_scorer_contract_v1_2 as C


def _artifact_inputs(monkeypatch):
    source = {
        "source_repository_commit": "c" * 40,
        "source_repository_clean": True,
        "bound_implementations_digest": "b" * 64,
    }
    feasibility = {
        "state_selector_feasibility_receipt_digest": "f" * 64,
    }
    disposition = {
        "mixed_precontract_disposition_receipt_digest": "d" * 64,
    }
    fixed_reissue = {
        "path": str(C.FIXED_REISSUE_INTERRUPTION.RECEIPT_RELATIVE_PATH),
        "receipt_digest": "1" * 64,
        "raw_sha256": "2" * 64,
        "byte_count": 101,
        "status": C.FIXED_REISSUE_INTERRUPTION.STATUS,
    }
    projection = {
        "path": str(C.INTERRUPTION.RECEIPT_RELATIVE_PATH),
        "receipt_digest": "3" * 64,
        "raw_sha256": "4" * 64,
        "byte_count": 202,
        "status": C.INTERRUPTION.STATUS,
    }
    performance = {
        "path": str(C.PERFORMANCE_INTERRUPTION.V2_RECEIPT_RELATIVE_PATH),
        "receipt_digest": "5" * 64,
        "raw_sha256": "6" * 64,
        "byte_count": 303,
        "status": C.PERFORMANCE_INTERRUPTION.V2_STATUS,
    }
    monkeypatch.setattr(
        C.STATE_SELECTOR, "validate_frozen_reachability_feasibility_pass",
        lambda **_kwargs: feasibility,
    )
    monkeypatch.setattr(
        C.STATE_SELECTOR,
        "validate_preserved_state_mixed_precontract_disposition_receipt",
        lambda *_args, **_kwargs: None,
    )
    return [
        source, feasibility, disposition, fixed_reissue, projection, performance,
    ]


def test_corpus_contract_digest_and_static_interruption_lineage_are_exact():
    assert C._digest(C.CORPUS_SELECTION_CONTRACT) == \
        C.CORPUS_SELECTION_CONTRACT_DIGEST

    static = C.contract()
    transition_lineage = static[
        "preoutcome_fixed_reissue_validation_interruption_lineage"]
    assert transition_lineage == \
        C.FIXED_REISSUE_INTERRUPTION.lineage_contract()
    assert transition_lineage["receipt_path"] == str(
        C.FIXED_REISSUE_INTERRUPTION.RECEIPT_RELATIVE_PATH)
    assert transition_lineage["status"] == C.FIXED_REISSUE_INTERRUPTION.STATUS
    performance_lineage = static[
        "preoutcome_small_search_performance_interruption_lineage"]
    assert performance_lineage == C.PERFORMANCE_INTERRUPTION.lineage_contract_v2()
    assert performance_lineage["receipt_path"] == str(
        C.PERFORMANCE_INTERRUPTION.V2_RECEIPT_RELATIVE_PATH)
    assert performance_lineage["status"] == C.PERFORMANCE_INTERRUPTION.V2_STATUS
    assert static["bound_implementations"][
        "fixed_reissue_validation_interruption_lineage"]["path"] == (
            "lewm/oracle/"
            "go2_scorer_fixed_reissue_validation_interruption_v1.py"
        )


def test_contract_artifact_strictly_embeds_transition_projection_and_v2(
        monkeypatch):
    arguments = _artifact_inputs(monkeypatch)
    artifact = C._contract_artifact_payload(*arguments)

    assert artifact[
        "preoutcome_fixed_reissue_validation_interruption_verified"] is True
    assert artifact["preoutcome_fixed_reissue_validation_interruption"] == \
        arguments[3]
    assert artifact["preoutcome_projection_fix_interruption_verified"] is True
    assert artifact["preoutcome_projection_fix_interruption"] == arguments[4]
    assert artifact[
        "preoutcome_small_search_performance_interruption_verified"] is True
    assert artifact["preoutcome_small_search_performance_interruption"] == \
        arguments[5]

    invalid_cases = []
    extra_transition = copy.deepcopy(arguments)
    extra_transition[3]["unexpected"] = False
    invalid_cases.append(extra_transition)
    uppercase_projection = copy.deepcopy(arguments)
    uppercase_projection[4]["raw_sha256"] = "A" * 64
    invalid_cases.append(uppercase_projection)
    boolean_byte_count = copy.deepcopy(arguments)
    boolean_byte_count[3]["byte_count"] = True
    invalid_cases.append(boolean_byte_count)
    v1_performance_path = copy.deepcopy(arguments)
    v1_performance_path[5]["path"] = str(
        C.PERFORMANCE_INTERRUPTION.V1_RECEIPT_RELATIVE_PATH)
    invalid_cases.append(v1_performance_path)

    for invalid in invalid_cases:
        with pytest.raises(RuntimeError, match="interruption binding is invalid"):
            C._contract_artifact_payload(*invalid)


def test_issue_contract_validates_current_transition_projection_then_v2(
        monkeypatch, tmp_path):
    source = {
        "source_repository_commit": "c" * 40,
        "source_repository_clean": True,
        "bound_implementations_digest": "b" * 64,
    }
    source_digest = C._digest(source)
    feasibility = {"state_selector_feasibility_receipt_digest": "f" * 64}
    disposition = {"mixed_precontract_disposition_receipt_digest": "d" * 64}
    transition_receipt = {"receipt": "transition"}
    transition_binding = {"binding": "transition"}
    projection_receipt = {"receipt": "projection"}
    projection_binding = {"binding": "projection"}
    performance_receipt = {"receipt": "performance-v2"}
    performance_binding = {"binding": "performance-v2"}
    calls = []

    monkeypatch.setattr(
        C, "_managed_scorer_contract_output_path", lambda path: path)
    monkeypatch.setattr(C, "clean_source_binding", lambda: source)
    monkeypatch.setattr(
        C.STATE_SELECTOR, "validate_authority_artifacts",
        lambda: calls.append("selector-authorities"))
    monkeypatch.setattr(
        C.STATE_SELECTOR, "validate_frozen_reachability_feasibility_pass",
        lambda **_kwargs: calls.append("selector-feasibility") or feasibility)

    def load_disposition(**kwargs):
        calls.append("selector-disposition")
        assert kwargs["expected_source_commit"] == \
            source["source_repository_commit"]
        assert kwargs["expected_clean_source_binding_digest"] == source_digest
        return disposition

    monkeypatch.setattr(
        C.STATE_SELECTOR,
        "load_and_validate_preserved_state_mixed_precontract_disposition_receipt",
        load_disposition,
    )

    def load_transition(**kwargs):
        calls.append("transition-load")
        assert kwargs == {
            "expected_source_repository_commit":
                source["source_repository_commit"],
            "expected_clean_source_binding_digest": source_digest,
            "expected_bound_implementations_digest":
                source["bound_implementations_digest"],
            "root": C.ROOT,
        }
        return transition_receipt

    monkeypatch.setattr(
        C.FIXED_REISSUE_INTERRUPTION,
        "load_and_validate_interruption_receipt", load_transition)
    monkeypatch.setattr(
        C.FIXED_REISSUE_INTERRUPTION, "receipt_binding",
        lambda receipt, **_kwargs:
            calls.append("transition-binding") or transition_binding
            if receipt is transition_receipt else None,
    )

    def load_projection(**kwargs):
        calls.append("projection-load")
        assert kwargs["expected_source_repository_commit"] == \
            source["source_repository_commit"]
        assert kwargs["expected_clean_source_binding_digest"] == source_digest
        return projection_receipt

    monkeypatch.setattr(
        C.INTERRUPTION, "load_and_validate_interruption_receipt",
        load_projection)
    monkeypatch.setattr(
        C.INTERRUPTION, "receipt_binding",
        lambda receipt, **_kwargs:
            calls.append("projection-binding") or projection_binding
            if receipt is projection_receipt else None,
    )

    def load_performance_v2(**kwargs):
        calls.append("performance-v2-load")
        assert kwargs["expected_source_repository_commit"] == \
            source["source_repository_commit"]
        assert kwargs["expected_clean_source_binding_digest"] == source_digest
        assert kwargs["expected_source_transition_receipt_binding"] == \
            transition_binding
        return performance_receipt

    monkeypatch.setattr(
        C.PERFORMANCE_INTERRUPTION,
        "load_and_validate_performance_interruption_receipt_v2",
        load_performance_v2,
    )
    monkeypatch.setattr(
        C.PERFORMANCE_INTERRUPTION,
        "performance_interruption_receipt_binding_v2",
        lambda receipt, **_kwargs:
            calls.append("performance-v2-binding") or performance_binding
            if receipt is performance_receipt else None,
    )

    class StopBeforeRuntimeBindings(Exception):
        pass

    def stop_before_runtime_bindings():
        calls.append("invalid-index")
        raise StopBeforeRuntimeBindings

    monkeypatch.setattr(
        C.INVALID_IDS, "load_invalid_identity_index",
        stop_before_runtime_bindings)

    with pytest.raises(StopBeforeRuntimeBindings):
        C.issue_contract(tmp_path / "scorer_contract_v1_2.json")
    assert calls == [
        "selector-authorities",
        "selector-feasibility",
        "selector-disposition",
        "transition-load",
        "transition-binding",
        "projection-load",
        "projection-binding",
        "performance-v2-load",
        "performance-v2-binding",
        "invalid-index",
    ]
