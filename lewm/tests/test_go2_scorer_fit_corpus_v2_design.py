from __future__ import annotations

import copy
import json
from pathlib import Path
import stat

import pytest

from lewm.oracle import go2_scorer_fit_corpus_v2_design as design


COMMIT = "a" * 40


def _sources() -> list[dict[str, object]]:
    return [
        {
            "path": path,
            "role": role,
            "byte_count": index + 1,
            "sha256": f"{index + 1:064x}",
        }
        for index, (path, role) in enumerate(design.SOURCE_SPECS)
    ]


def _classification() -> dict[str, object]:
    return design.build_rotation_mask_classification(
        source_repository_commit=COMMIT,
        source_bindings=_sources(),
        predecessor_validation=design.PREDECESSOR_VALIDATION_PROJECTION,
    )


def _classification_binding() -> dict[str, object]:
    payload = _classification()
    raw = (json.dumps(payload, sort_keys=True, indent=2) + "\n").encode()
    return design.rotation_mask_classification_artifact_binding(payload, raw)


def _amendment() -> dict[str, object]:
    return design.build_design_amendment(
        source_repository_commit=COMMIT,
        source_bindings=_sources(),
        rotation_mask_classification_binding=_classification_binding(),
        predecessor_validation=design.PREDECESSOR_VALIDATION_PROJECTION,
    )


def _issued_design_authority() -> dict[str, object]:
    sources = _sources()
    classification = design.build_rotation_mask_classification(
        source_repository_commit=
            design.ISSUED_FULL_BANK_V2_SOURCE_REPOSITORY_COMMIT,
        source_bindings=sources,
        predecessor_validation=design.PREDECESSOR_VALIDATION_PROJECTION,
    )
    classification_raw = (
        json.dumps(classification, sort_keys=True, indent=2) + "\n").encode()
    classification_binding = (
        design.rotation_mask_classification_artifact_binding(
            classification, classification_raw))
    amendment = design.build_design_amendment(
        source_repository_commit=
            design.ISSUED_FULL_BANK_V2_SOURCE_REPOSITORY_COMMIT,
        source_bindings=sources,
        rotation_mask_classification_binding=classification_binding,
        predecessor_validation=design.PREDECESSOR_VALIDATION_PROJECTION,
    )
    amendment_raw = (
        json.dumps(amendment, sort_keys=True, indent=2) + "\n").encode()
    return design.validate_immutable_issued_design_authority({
        "rotation_mask_classification_payload": classification,
        "rotation_mask_classification_binding": classification_binding,
        "design_amendment_payload": amendment,
        "design_amendment_binding": design.design_amendment_artifact_binding(
            amendment, amendment_raw),
    })


def _corrected_sources() -> list[dict[str, object]]:
    rows = _sources()
    changed = set(design.SOURCE_CORRECTION_ALLOWED_CHANGED_SOURCE_PATHS)
    for index, row in enumerate(rows):
        if row["path"] in changed:
            row["byte_count"] = int(row["byte_count"]) + 10_000
            row["sha256"] = f"{index + 10_000:064x}"
    return rows


def _source_correction() -> dict[str, object]:
    return design.build_preselection_source_correction(
        source_repository_commit="b" * 40,
        source_bindings=_corrected_sources(),
        immutable_issued_design_authority=_issued_design_authority(),
        runtime_outputs_absent_at_issue=design._expected_absence_rows(
            phase="design"),
    )


def test_rotation_inventory_is_closed_and_allocation_only() -> None:
    payload = _classification()
    assert design.validate_rotation_mask_classification(payload) == payload
    assert [row["constraint_id"] for row in payload["conditions"]] == list(
        design.EXPECTED_ROTATION_CONSTRAINT_IDS)
    assert {row["classification"] for row in payload["conditions"]} == {
        "PARTIAL_SUBSET_ALLOCATION_ONLY"}
    assert payload["counts"] == {
        "old_rotation_related_condition_count": 18,
        "partial_subset_allocation_only_count": 18,
        "true_branch_execution_requirement_count": 0,
    }
    assert payload["true_branch_execution_test"][
        "matching_old_rotation_condition_ids"] == []


def test_subset_lmax_is_retired_but_completion_science_is_retained() -> None:
    payload = _classification()
    rows = {row["constraint_id"]: row for row in payload["conditions"]}
    for key in (
        "COMPLETION_ASSIGNED_ROTATION_ELIGIBILITY",
        "ALL_40_COMPLETION_MASKS_PASS",
    ):
        assert rows[key]["v2_disposition"] == (
            "REPLACED_BY_FULL_BANK_L_MAX_STATE_REVALIDATION")
        assert "does not establish branch executability" in rows[key]["rationale"]
    retained = payload["retained_non_rotation_completion_requirements"]
    assert retained["full_bank_l_max_candidate_indices"] == list(range(12))
    assert retained["completion_radius_m"] == 0.75
    assert retained["horizon_ticks"] == 20
    assert retained["branch_execution_used_for_revalidation"] is False


def test_classification_is_self_bound_and_tamper_evident() -> None:
    payload = _classification()
    assert payload[design.MASK_CLASSIFICATION_SELF_KEY] == design.canonical_digest({
        key: value for key, value in payload.items()
        if key != design.MASK_CLASSIFICATION_SELF_KEY
    })
    tampered = copy.deepcopy(payload)
    tampered["conditions"][0]["classification"] = (
        "TRUE_BRANCH_EXECUTION_REQUIREMENT")
    with pytest.raises(design.ScorerFitCorpusV2DesignError):
        design.validate_rotation_mask_classification(tampered)


def test_full_bank_design_freezes_exact_algebra_and_only_one_supersession() -> None:
    payload = _amendment()
    assert design.validate_design_amendment(payload) == payload
    counts = payload["count_contract"]
    assert counts["state_count"] == 120
    assert counts["candidate_indices"] == list(range(12))
    assert counts["assignments_total"] == 1_440
    assert counts["per_candidate"] == {
        "overall": 120,
        "fit": 96,
        "calibration": 24,
        "per_stratum": 40,
        "per_family": 15,
        "fit_per_family": 12,
        "calibration_per_family": 3,
        "per_family_stratum": 5,
    }
    assert counts["unordered_candidate_pair_cooccurrence"] == 120
    assert payload["supersession"]["status"] == design.SIX_OF_TWELVE_SUPERSESSION
    assert payload["supersession"]["selector_superseded"] is False
    assert payload["supersession"]["oracle_superseded"] is False
    assert payload["issuance_boundary"]["milp_or_cp_sat_run"] is False


def test_design_binds_exact_terminal_and_all_prior_failure_lineage() -> None:
    lineage = _amendment()["preoutcome_lineage"]
    assert lineage["terminal_source_repository_commit"] == (
        design.TERMINAL_SOURCE_REPOSITORY_COMMIT)
    assert lineage["active_global_amendment_digest"] == (
        design.ACTIVE_GLOBAL_AMENDMENT_DIGEST)
    assert lineage["global_exact_model_digest"] == design.GLOBAL_EXACT_MODEL_DIGEST
    assert lineage["exact_infeasibility_digest"] == design.EXACT_INFEASIBILITY_DIGEST
    assert lineage["terminal_receipt_digest"] == design.TERMINAL_RECEIPT_DIGEST
    assert lineage["candidate_outcomes_consumed_at_proof"] is False
    assert len(lineage["immutable_v1_v2_failure_bindings"]) == 4
    assert len(lineage["prior_preoutcome_failure_bindings"]) == len(
        design.PRIOR_PREOUTCOME_FAILURE_BINDINGS)
    assert lineage["frozen_predictor_qualification"]["modified_or_rerun"] is False


def test_order_key_is_canonical_deterministic_and_domain_separated() -> None:
    structural_a = {"scene_id": "scene-a", "source_step": 20, "split": "fit"}
    structural_b = {"split": "fit", "source_step": 20, "scene_id": "scene-a"}
    goal = {"landmark_id": "g", "landmark_cell": 7, "material_id": 2}
    key_a = design.completion_order_key(structural_a, goal)
    key_b = design.completion_order_key(structural_b, goal)
    assert key_a == key_b
    assert len(key_a[0]) == 64
    assert key_a != design.completion_order_key(
        structural_a, {**goal, "landmark_cell": 8})
    with pytest.raises(design.ScorerFitCorpusV2DesignError):
        design.completion_order_key(
            structural_a, goal, active_selector_digest="b" * 64)


def test_design_preserves_prospective_final_eval_disjointness_not_reservation() -> None:
    science = _amendment()["preserved_nonallocation_science"]
    assert science["final_200_state_corpus_authorized_in_this_pass"] is False
    assert science["preexisting_reserved_final_evaluation_scene_set"] is False
    assert science["final_evaluation_manifest_absent_at_issue"] is True
    assert "excludes all 120 scenes" in science["future_final_evaluation_rule"]


def test_phase_aware_absence_audit(tmp_path: Path) -> None:
    scorer_fit = tmp_path / design.SCORER_FIT_RELATIVE_PATH
    scorer_fit.mkdir(parents=True)
    assert design.audit_v2_runtime_outputs_absent(
        root=tmp_path, phase="design")
    preoutcome = tmp_path / design.V2_PREOUTCOME_ARTIFACT_PATHS[0]
    preoutcome.write_text("source-only")
    with pytest.raises(design.ScorerFitCorpusV2DesignError):
        design.audit_v2_runtime_outputs_absent(root=tmp_path, phase="design")
    assert design.audit_v2_runtime_outputs_absent(
        root=tmp_path, phase="successor_contract")
    runtime = tmp_path / design.V2_RUNTIME_OUTPUT_PATHS[0]
    runtime.write_text("outcome")
    with pytest.raises(design.ScorerFitCorpusV2DesignError):
        design.audit_v2_runtime_outputs_absent(
            root=tmp_path, phase="successor_contract")


def test_issue_is_exclusive_read_only_and_validates_predecessors_twice(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    scorer_fit = tmp_path / design.SCORER_FIT_RELATIVE_PATH
    scorer_fit.mkdir(parents=True)
    calls: list[str] = []

    monkeypatch.setattr(
        design, "clean_source_authority", lambda *, root: (COMMIT, _sources()))

    def predecessors(*, root: Path) -> dict[str, object]:
        calls.append(str(root))
        return copy.deepcopy(design.PREDECESSOR_VALIDATION_PROJECTION)

    monkeypatch.setattr(
        design, "validate_historical_predecessor_artifacts", predecessors)
    classification = design.issue_rotation_mask_classification(root=tmp_path)
    assert len(calls) == 2
    class_path = tmp_path / design.MASK_CLASSIFICATION_RELATIVE_PATH
    assert stat.S_IMODE(class_path.stat().st_mode) == 0o444
    assert design.issue_rotation_mask_classification(root=tmp_path) == classification
    assert len(calls) == 2  # reopening an issued classification is source-only.

    amendment = design.issue_design_amendment(root=tmp_path)
    assert len(calls) == 4
    design_path = tmp_path / design.DESIGN_RELATIVE_PATH
    assert stat.S_IMODE(design_path.stat().st_mode) == 0o444
    assert design.load_design_amendment(root=tmp_path) == amendment


def test_artifact_binding_rejects_noncanonical_bytes() -> None:
    payload = _classification()
    compact = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    with pytest.raises(design.ScorerFitCorpusV2DesignError):
        design.rotation_mask_classification_artifact_binding(payload, compact)


def test_builders_are_pure_and_do_not_require_repository_or_generated_files(
        tmp_path: Path) -> None:
    # Both builders operate entirely on supplied source identities and literal
    # frozen lineage.  The empty path proves no repository fixture is needed.
    assert not list(tmp_path.iterdir())
    assert _classification()["outcome_access"][
        "historical_receipts_used_for_classification"] is False
    assert _amendment()["selection_field_policy"][
        "historical_receipts_used_for_selection"] is False
    assert not list(tmp_path.iterdir())


def test_preselection_source_correction_preserves_science_and_exact_failure() -> None:
    correction = _source_correction()
    assert design.validate_preselection_source_correction(
        correction, validate_live_authorities=False) == correction
    issued = _issued_design_authority()
    assert correction["preserved_scientific_design_digest"] == issued[
        "design_amendment_payload"][design.DESIGN_SELF_KEY]
    assert correction["preserved_rotation_mask_classification_digest"] == issued[
        "rotation_mask_classification_payload"][
            design.MASK_CLASSIFICATION_SELF_KEY]
    assert correction["source_correction"][
        "observed_changed_source_paths"] == sorted(
            design.SOURCE_CORRECTION_ALLOWED_CHANGED_SOURCE_PATHS)
    failure = correction["preselection_alias_failure_boundary"]
    assert failure == design.PRESELECTION_ALIAS_FAILURE_BOUNDARY
    assert failure["predecessor_fixed_state_count_validated"] == 115
    assert failure["eligible_small_completion_scene_count_validated"] == 17
    assert failure["exclusion_authority_returned"] is False
    assert failure["small_completion_selection_started"] is False
    assert failure["preoutcome_manifest_or_selection_artifact_issued"] is False
    assert failure["candidate_outcome_or_branch_label_read"] is False
    assert failure["solver_or_optimisation_invoked"] is False


def test_preselection_source_correction_is_closed_and_tamper_evident() -> None:
    correction = _source_correction()
    raw = (json.dumps(correction, sort_keys=True, indent=2) + "\n").encode()
    binding = design.preselection_source_correction_artifact_binding(
        correction, raw)
    assert binding["self_digest"] == correction[
        design.SOURCE_CORRECTION_SELF_KEY]
    assert set(binding) == {
        "path", "schema", "self_digest_key", "self_digest", "raw_sha256",
        "byte_count", "source_repository_commit",
    }
    tampered = copy.deepcopy(correction)
    tampered["preselection_alias_failure_boundary"][
        "exclusion_authority_returned"] = True
    with pytest.raises(design.ScorerFitCorpusV2DesignError):
        design.validate_preselection_source_correction(
            tampered, validate_live_authorities=False)
    extra = copy.deepcopy(correction)
    extra["unregistered"] = True
    with pytest.raises(design.ScorerFitCorpusV2DesignError):
        design.validate_preselection_source_correction(
            extra, validate_live_authorities=False)


def test_preselection_source_correction_rejects_wrong_source_delta() -> None:
    sources = _corrected_sources()
    unchanged_path = next(
        row for row in sources
        if row["path"] not in design.SOURCE_CORRECTION_ALLOWED_CHANGED_SOURCE_PATHS)
    unchanged_path["byte_count"] = int(unchanged_path["byte_count"]) + 1
    unchanged_path["sha256"] = "f" * 64
    with pytest.raises(design.ScorerFitCorpusV2DesignError):
        design.build_preselection_source_correction(
            source_repository_commit="b" * 40,
            source_bindings=sources,
            immutable_issued_design_authority=_issued_design_authority(),
            runtime_outputs_absent_at_issue=design._expected_absence_rows(
                phase="design"),
        )


def test_source_correction_issue_is_exclusive_read_only_and_double_audited(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    scorer_fit = tmp_path / design.SCORER_FIT_RELATIVE_PATH
    scorer_fit.mkdir(parents=True)
    issued = _issued_design_authority()
    sources = _corrected_sources()
    absence = design._expected_absence_rows(phase="design")
    absence_calls: list[int] = []

    monkeypatch.setattr(
        design, "clean_source_authority",
        lambda *, root: ("b" * 40, copy.deepcopy(sources)))
    monkeypatch.setattr(
        design, "_load_issued_design_authority_for_source_correction",
        lambda *, root: copy.deepcopy(issued))

    def audit(*, root: Path, phase: str) -> list[dict[str, object]]:
        assert root == tmp_path
        assert phase == "design"
        absence_calls.append(1)
        return copy.deepcopy(absence)

    monkeypatch.setattr(design, "audit_v2_runtime_outputs_absent", audit)
    correction = design.issue_preselection_source_correction(root=tmp_path)
    assert len(absence_calls) == 2
    path = tmp_path / design.SOURCE_CORRECTION_RELATIVE_PATH
    assert stat.S_IMODE(path.stat().st_mode) == 0o444
    assert correction[design.SOURCE_CORRECTION_SELF_KEY]
    active = design.load_active_design_authority(root=tmp_path)
    assert active["source_correction"] == correction
    assert active["source_correction_digest"] == correction[
        design.SOURCE_CORRECTION_SELF_KEY]
    assert active["design_amendment"] == issued["design_amendment_payload"]
    assert design.issue_preselection_source_correction(
        root=tmp_path) == correction
    assert len(absence_calls) == 2
