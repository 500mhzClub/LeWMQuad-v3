from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from lewm.safety import protected_contact_scope_requirements_review_v1 as stage_a
from scripts import complete_protected_contact_scope_requirements_review_v1 as complete


def _redigest(value: dict[str, object]) -> None:
    value.pop("content_digest", None)
    value["content_digest"] = complete.canonical_digest(value)


def test_completion_vocabulary_paths_and_acyclic_source_members() -> None:
    assert complete.EXPERIMENT_ID == "PROTECTED_CONTACT_SCOPE_REQUIREMENTS_REVIEW_V1"
    assert complete.PRIMARY_CLASSIFICATION == (
        "PROTECTED_CONTACT_SCOPE_REQUIREMENTS_UNRESOLVED"
    )
    assert complete.STAGE_B_EXECUTION == "NOT_RUN"
    assert complete.NEXT_DECISION == "REQUIREMENTS_ACQUISITION_REQUIRED"
    assert complete.EXPECTED_DIAGNOSTIC_CAUSES == (
        "CONTACT_CRITICAL_PATCH_UNOBSERVED",
        "POINT_TO_PRIMITIVE_DISTANCE_MISMATCH",
        "GLOBAL_THRESHOLD_HETEROGENEITY",
        "SELF_OCCLUSION_AT_CONTACT_CRITICAL_REGION",
        "UNRESOLVED_GEOMETRIC_MISMATCH",
    )
    assert len(complete.RESULT_SOURCE_ADDITIONS) == 6
    additions = [path for path, _role in complete.RESULT_SOURCE_ADDITIONS]
    assert len(additions) == len(set(additions))
    assert not set(additions).intersection(complete.EXCLUDED_SELF_REFERENTIAL_PATHS)
    assert complete.RESULT_PATH.name.endswith("_result.json")
    assert complete.RESULT_SOURCE_CLOSURE_PATH.name.endswith(
        "_result_source_closure.json"
    )


def test_frozen_inputs_and_authoritative_diagnostic_binding() -> None:
    contract, stage_closure, diagnostic = complete.load_and_validate_inputs()
    assert contract["content_digest"] == (
        "9d763926cfe711893db86db5571a79d0af6dee8ea71706658b663c93df9a8f51"
    )
    assert complete.sha256_file(complete.STAGE_A_CONTRACT_PATH) == (
        "f51f69f5c48bd2b4b256bb229c0156d8c801dbfdeb04eb9a52ab9249e212c5cb"
    )
    assert stage_closure["file_count"] == 40
    assert stage_closure["content_digest"] == (
        "fab24ef46c9c98b5c53e425ee31954f688367a7bd4a4d7c3ee58556d4ec78e1e"
    )
    assert complete.sha256_file(complete.DIAGNOSTIC_JSON_PATH) == (
        "a48f2fc8840ae68acab2220654e11795851607b4beb3ad0b198dec29d10368ec"
    )
    assert diagnostic["content_digest"] == (
        "43e12c8aabad9f3d813a4f6bf4a0df016f3a89d62f3fd3fee6188db0e5045b12"
    )
    assert diagnostic["runtime_s"] == 74.89513779900153
    assert diagnostic["workers"] == 32
    assert tuple(diagnostic["supported_diagnostic_cause_ids"]) == (
        complete.EXPECTED_DIAGNOSTIC_CAUSES
    )


def test_heldout_support_denominators_and_support_only_residuals() -> None:
    _contract, _stage_closure, diagnostic = complete.load_and_validate_inputs()
    heldout = complete.extract_heldout_true_future(diagnostic)
    assert len(heldout) == 11
    assert {row["contact_events"] for row in heldout.values()} == {579}
    for row in heldout.values():
        assert row["patch_supported_events"] + row["patch_unobserved_events"] == 579
        assert row["observed_minus_exact_statistics_m"]["count"] == row[
            "patch_supported_events"
        ]
    assert heldout["THREE_DENSE_SPHERICAL_ORIGIN_UPPER_BOUND"][
        "patch_supported_events"
    ] == 577
    assert heldout["CURRENT_SPARSE_RANGE_BASELINE"][
        "observed_minus_exact_statistics_m"
    ]["count"] == 0


def test_calibration_q95_is_complete_descriptive_and_never_reselected() -> None:
    _contract, _stage_closure, diagnostic = complete.load_and_validate_inputs()
    calibration = complete.extract_calibration_q95(diagnostic)
    assert calibration["selection_authority"] is False
    assert calibration["heldout_used"] is False
    assert calibration["interpolation_used"] is False
    assert len(calibration["per_link"]) == 231
    assert len(calibration["per_collision_component"]) == 273
    for group in ("per_link", "per_collision_component"):
        for row in calibration[group].values():
            assert row["threshold_reselected"] is False
            assert (
                row["finite_attributable_link_minimum_scores"]
                + row["unsupported_attributable_link_minimum_scores"]
                == row["positives"]
            )


def test_input_validation_rejects_stage_b_and_diagnostic_counter_tamper(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    contract = json.loads(complete.STAGE_A_CONTRACT_PATH.read_text())
    contract["stage_b_gate"]["authorized"] = True
    _redigest(contract["stage_b_gate"])
    _redigest(contract)
    tampered_contract = tmp_path / "contract.json"
    tampered_contract.write_bytes(complete.canonical_file_bytes(contract))
    monkeypatch.setattr(complete, "STAGE_A_CONTRACT_PATH", tampered_contract)
    with pytest.raises((complete.CompletionError, stage_a.ContractValidationError)):
        complete.load_and_validate_inputs()

    monkeypatch.undo()
    diagnostic = json.loads(complete.DIAGNOSTIC_JSON_PATH.read_text())
    diagnostic["prohibited_action_counters"]["raycasts"] = 1
    _redigest(diagnostic)
    tampered_diagnostic = tmp_path / "diagnostic.json"
    tampered_diagnostic.write_bytes(complete.canonical_file_bytes(diagnostic))
    monkeypatch.setattr(complete, "DIAGNOSTIC_JSON_PATH", tampered_diagnostic)
    with pytest.raises(complete.CompletionError):
        complete.load_and_validate_inputs()


def test_generated_source_closure_preserves_forty_rows_and_adds_six() -> None:
    _contract, stage_closure, _diagnostic = complete.load_and_validate_inputs()
    closure = json.loads(complete.RESULT_SOURCE_CLOSURE_PATH.read_text())
    complete.validate_result_source_closure(closure, stage_closure=stage_closure)
    assert closure["file_count"] == 46
    assert closure["files"][:40] == stage_closure["files"]
    assert [row["path"] for row in closure["files"][40:]] == [
        path for path, _role in complete.RESULT_SOURCE_ADDITIONS
    ]
    assert not set(complete.EXCLUDED_SELF_REFERENTIAL_PATHS).intersection(
        row["path"] for row in closure["files"]
    )


def test_generated_result_preserves_stage_a_and_reports_not_run() -> None:
    contract, stage_closure, diagnostic = complete.load_and_validate_inputs()
    closure = json.loads(complete.RESULT_SOURCE_CLOSURE_PATH.read_text())
    result = json.loads(complete.RESULT_PATH.read_text())
    complete.validate_result(
        result,
        contract=contract,
        diagnostic=diagnostic,
        result_source_closure=closure,
    )
    assert result["stage_a_result"]["primary_classification"] == (
        complete.PRIMARY_CLASSIFICATION
    )
    assert tuple(result["stage_a_result"]["secondary_classifications"]) == (
        stage_a.ALLOWED_SECONDARY_CLASSIFICATIONS
    )
    assert result["stage_a_result"]["stage_b"]["execution"] == "NOT_RUN"
    assert result["stage_a_result"]["stage_b"]["authorized"] is False
    assert result["commit_bindings"]["future_result_commit_bound"] is False
    assert result["commit_bindings"]["future_result_commit"] is None
    assert result["next_decision"]["classification"] == (
        "REQUIREMENTS_ACQUISITION_REQUIRED"
    )
    assert all(value == 0 for value in result["prohibited_action_counters"].values())
    attempts = result["custody"]["diagnostic_development_attempts"]
    assert [attempt["count"] for attempt in attempts["attempts"]] == [2, 2]
    assert attempts["attempts_used_existing_evidence_only"] is True


def test_markdown_and_receipts_regenerate_byte_identically() -> None:
    result = complete.check()
    report = complete.DIAGNOSTIC_MARKDOWN_PATH.read_text()
    assert complete.PRIMARY_CLASSIFICATION in report
    assert all(cause in report for cause in complete.EXPECTED_DIAGNOSTIC_CAUSES)
    assert "Stage B is `NOT_RUN`" in report
    assert complete.NEXT_DECISION in report
    assert result["content_digest"] == complete.canonical_digest(
        {key: value for key, value in result.items() if key != "content_digest"}
    )
