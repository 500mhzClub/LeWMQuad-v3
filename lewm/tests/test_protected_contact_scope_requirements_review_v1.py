from __future__ import annotations

import copy
from dataclasses import asdict, replace
import inspect
import json

import pytest

from lewm.safety import protected_contact_scope_requirements_review_v1 as contract


def _redigest(value: dict[str, object]) -> None:
    value.pop("content_digest", None)
    value["content_digest"] = contract.canonical_digest(value)


def _context(**changes: object) -> contract.RequirementContext:
    values: dict[str, object] = {
        "robot_link_name": "base",
        "environment_class": "FIXED_WALL",
        "operating_mode": "ANY_OPERATION",
        "contact_type": "EXTERNAL_ENVIRONMENT_CONTACT",
        "duration_status": "UNRESOLVED",
        "repetition_status": "UNRESOLVED",
        "stability_consequence_status": "UNRESOLVED",
        "task_consequence_status": "UNRESOLVED",
        "uncertainty_status": "UNRESOLVED",
    }
    values.update(changes)
    return contract.RequirementContext(**values)


def test_exact_stage_a_literals_are_frozen() -> None:
    assert contract.START_COMMIT == "99cfa17cddb2aaddde69b8bdb6c3ea4a8e5ca849"
    assert contract.PRIMARY_CLASSIFICATION == "PROTECTED_CONTACT_SCOPE_REQUIREMENTS_UNRESOLVED"
    assert contract.REQUIREMENT_CATEGORIES == (
        "HARD_PREACTION_SEPARATION",
        "CONDITIONAL_RECOVERABLE_CONTACT",
        "MONITOR_AND_RECOVER",
        "PERMITTED_SUPPORT_OR_SELF_CONTACT",
        "SEVERITY_OR_REQUIREMENT_UNRESOLVED",
    )
    assert contract.ALLOWED_SECONDARY_CLASSIFICATIONS == (
        "SIMULATED_CONTACT_PROXY_SCOPE_ONLY",
        "DEPLOYMENT_MATERIAL_HAZARD_SCOPE_UNRESOLVED",
        "PERSON_AND_FRAGILE_ASSET_HAZARDS_NOT_REPRESENTED",
        "RECOVERABILITY_REQUIREMENTS_UNRESOLVED",
        "MISSION_PROGRESS_REQUIREMENTS_PRESENT",
        "DISTRIBUTED_BODY_SENSING_CANDIDATE",
        "FULL_BODY_EXTERNAL_RANGE_SENSING_IMPLAUSIBLE",
    )
    assert contract.STAGE_B_STATUS == "STAGE_B_NOT_AUTHORIZED"
    assert contract.PRESERVED_PREDECESSOR_CLASSIFICATIONS == (
        "MULTI_ORIGIN_UP_TO_THREE_RANGE_COVERAGE_NO_GO",
        "SINGLE_ORIGIN_RANGE_COVERAGE_NO_GO",
        "SENSOR_COVERAGE_MICRO_VIABILITY_NO_GO",
        "TRUE_SUCCESSOR_SET_VIABILITY_NO_SIGNAL",
        "DEPLOYABLE_MICRO_ACTION_CONTRACT_ALIGNED",
        "STRUCTURED_GEOMETRY_SET_REDUCTION_COMPUTE_SIGNAL",
        "MULTI_ORIGIN_SET_REDUCTION_COMPUTE_SIGNAL",
        "REPLANNING_INTERFACE_UNRESOLVED",
        "GO2_PLATFORM_STOPPING_MODE_PARITY_PENDING",
        "ASSUMED_SENSOR_CONTRACT",
    )
    assert contract.PRESERVED_EXACT_GEOMETRY_FINDING.startswith(
        "Exact Genesis per-link geometry reconstructs"
    )
    assert contract.PRESERVED_RANGE_FINDING.startswith(
        "Under the current full-body, 13-link protected-contact scope"
    )


def test_component_inventory_is_exact_and_exhaustive() -> None:
    rows = contract.protected_collision_components()
    assert len(rows) == 27
    assert [row.geom_index for row in rows] == list(range(27))
    assert tuple(dict.fromkeys(row.link_name for row in rows)) == contract.PROTECTED_LINK_NAMES
    assert [sum(row.link_name == link for row in rows) for link in contract.PROTECTED_LINK_NAMES] == [
        3, 1, 1, 1, 1, 1, 1, 1, 1, 4, 4, 4, 4
    ]
    assert [(row.component_id, row.primitive, row.data) for row in rows[:3]] == [
        ("TRUNK_CHASSIS", "box", (0.3762, 0.0935, 0.114)),
        ("HEAD_UPPER", "capsule", (0.05, 0.09)),
        ("HEAD_LOWER", "sphere", (0.047,)),
    ]
    assert rows[11].data == (0.012, 0.12)
    assert rows[15].data == rows[19].data == rows[23].data == (0.013, 0.12)
    assert [row.component_id.rsplit("_", 1)[-1] for row in rows if row.component_id.endswith("_FOOT")] == [
        "FOOT", "FOOT", "FOOT", "FOOT"
    ]
    assert all(
        row.support_role == "HISTORICAL_CALF_LINK_PLANE_EXCLUSION_IS_LINK_GRANULAR"
        for row in rows[11:]
    )


def test_component_inventory_validator_fails_closed() -> None:
    rows = [asdict(row) for row in contract.protected_collision_components()]
    contract.validate_component_inventory(rows)
    with pytest.raises(contract.ContractValidationError, match="count"):
        contract.validate_component_inventory(rows[:-1])
    changed = copy.deepcopy(rows)
    changed[0]["geom_index"] = 8
    with pytest.raises(contract.ContractValidationError, match="indices"):
        contract.validate_component_inventory(changed)
    changed = copy.deepcopy(rows)
    changed[0]["data"] = [float("nan"), 1.0, 1.0]
    with pytest.raises(contract.ContractValidationError, match="finite"):
        contract.validate_component_inventory(changed)


def test_historical_h1_exclusions_are_reproduced_without_reinterpretation() -> None:
    assert contract.is_preserved_historical_exclusion(
        robot_link_name="FL_calf", environment_class="Plane"
    )
    assert contract.is_preserved_historical_exclusion(
        robot_link_name="RR_calf", environment_class="ground-plane"
    )
    assert contract.is_preserved_historical_exclusion(
        robot_link_name="base", environment_class="wall", self_contact=True
    )
    assert not contract.is_preserved_historical_exclusion(
        robot_link_name="base", environment_class="Plane"
    )
    h1 = contract.build_historical_h1_contract()
    assert h1["calf_plane_exclusion_granularity"] == "LINK_LEVEL_NOT_FOOT_SHAPE_ONLY"
    assert h1["reinterpretation_authorized"] is False
    assert h1["label_change_authorized"] is False


def test_prospective_support_requires_complete_explicit_context() -> None:
    ordinary = _context(
        robot_link_name="FL_calf",
        environment_class="GROUND_PLANE",
        operating_mode="ORDINARY_STANCE_OR_LOCOMOTION",
        contact_type="INTENDED_GROUND_SUPPORT",
        duration_status="ORDINARY_SUPPORT",
        repetition_status="ORDINARY_SUPPORT",
        stability_consequence_status="STABLE_SUPPORT",
        task_consequence_status="MISSION_SUPPORT",
        uncertainty_status="REQUIREMENTS_CONTEXT_RESOLVED",
        ordinary_support_contact=True,
    )
    decision = contract.decide_requirement(ordinary)
    assert decision.category == contract.PERMITTED_SUPPORT_OR_SELF_CONTACT
    assert decision.historical_h1_exclusion_preserved is True

    for field_name in (
        "operating_mode", "contact_type", "duration_status", "repetition_status",
        "stability_consequence_status", "task_consequence_status", "uncertainty_status",
    ):
        incomplete = replace(ordinary, **{field_name: "UNRESOLVED"})
        result = contract.decide_requirement(incomplete)
        assert result.category == contract.SEVERITY_OR_REQUIREMENT_UNRESOLVED
        assert result.historical_h1_exclusion_preserved is True
    abnormal = replace(ordinary, ordinary_support_contact=False, operating_mode="ABNORMAL_OR_UNKNOWN")
    assert contract.decide_requirement(abnormal).category == contract.SEVERITY_OR_REQUIREMENT_UNRESOLVED
    with pytest.raises(contract.ContractValidationError, match="only for a calf-link/ground"):
        contract.decide_requirement(replace(ordinary, robot_link_name="base"))


def test_context_first_decision_rules_cover_all_five_categories() -> None:
    assert contract.decide_requirement(_context(self_contact=True)).category == (
        contract.PERMITTED_SUPPORT_OR_SELF_CONTACT
    )
    assert contract.decide_requirement(_context(
        categorical_person_fragile_safety_critical_or_prohibited=True
    )).category == contract.HARD_PREACTION_SEPARATION
    assert contract.decide_requirement(_context(
        approved_conditional_recoverability_requirement=True,
        object_consequence_evidence_complete=True,
        physical_severity_evidence_complete=True,
        recovery_evidence_complete=True,
        approved_acceptance_criteria=True,
    )).category == contract.CONDITIONAL_RECOVERABLE_CONTACT
    assert contract.decide_requirement(_context(
        approved_monitor_and_recover_requirement=True,
        recovery_evidence_complete=True,
        approved_acceptance_criteria=True,
    )).category == contract.MONITOR_AND_RECOVER
    assert contract.decide_requirement(_context()).category == (
        contract.SEVERITY_OR_REQUIREMENT_UNRESOLVED
    )


def test_context_rules_fail_closed_on_conflicts_unknowns_and_incomplete_recovery() -> None:
    with pytest.raises(contract.ContractValidationError, match="conflicting"):
        contract.decide_requirement(_context(
            approved_hard_separation_requirement=True,
            approved_monitor_and_recover_requirement=True,
        ))
    with pytest.raises(contract.ContractValidationError, match="unknown protected link"):
        contract.decide_requirement(_context(robot_link_name="foot"))
    with pytest.raises(contract.ContractValidationError, match="non-empty"):
        contract.decide_requirement(_context(contact_type=""))
    conditional = contract.decide_requirement(_context(
        approved_conditional_recoverability_requirement=True,
        object_consequence_evidence_complete=True,
    ))
    monitored = contract.decide_requirement(_context(
        approved_monitor_and_recover_requirement=True,
        recovery_evidence_complete=True,
    ))
    assert conditional.category == monitored.category == contract.SEVERITY_OR_REQUIREMENT_UNRESOLVED
    assert conditional.scope_change_authorized is monitored.scope_change_authorized is False


def test_requirements_only_barrier_rejects_nested_outcome_fields() -> None:
    contract.assert_requirements_only_payload({"requirement": {"severity_limit": "MISSING"}})
    for key in (
        "heldout_metrics", "current_contact_auc", "safe_action_count",
        "h3_route_progress", "calibration_threshold", "sensor_coverage",
    ):
        with pytest.raises(contract.ContractValidationError, match="forbidden"):
            contract.assert_requirements_only_payload({"nested": [{key: 1}]})
    assert "outcome_fields_used" not in inspect.signature(contract.decide_requirement).parameters
    assert "heldout_metrics" not in inspect.signature(contract.build_contract_receipt).parameters


def test_source_and_assumption_builders_are_deterministic_and_outcome_blind() -> None:
    sources = contract.build_source_inventory()
    assumptions = contract.build_assumption_inventory()
    assert sources == contract.build_source_inventory()
    assert assumptions == contract.build_assumption_inventory()
    contract.validate_content_digest(sources)
    contract.validate_content_digest(assumptions)
    assert len(sources["sources"]) == 13
    assert all(row["scientific_outcomes_accessed"] is False for row in sources["sources"])
    assert all(len(row["sha256"]) == 64 for row in sources["sources"])
    assert sources["outcome_fields_used"] == assumptions["outcome_fields_used"] == []
    assert all(item["scope_change_authority"] is False for item in assumptions["assumptions"])
    assert assumptions["unresolved_evidence_count"] == len(
        assumptions["unresolved_evidence"]
    ) == 9
    assert {item["evidence_id"] for item in assumptions["unresolved_evidence"]} >= {
        "PCS-UE-ODD-001",
        "PCS-UE-OBJECT-001",
        "PCS-UE-CONSEQUENCE-001",
        "PCS-UE-STOP-001",
        "PCS-UE-RECOVERY-001",
        "PCS-UE-TASK-001",
    }


def test_traceability_exhausts_components_without_authorizing_scope_change() -> None:
    trace = contract.build_traceability_matrix()
    contract.validate_content_digest(trace)
    contract.validate_traceability_matrix(trace)
    assert trace["row_count"] == 27 == len(trace["rows"])
    assert trace["hazard_row_count"] == 11 == len(trace["hazard_rows"])
    assert {row["hazard"] for row in trace["hazard_rows"]} >= {
        "person collision",
        "fragile or safety-critical infrastructure contact",
        "trunk or payload impact",
        "proximal limb entrapment",
        "distal limb or calf contact",
        "destabilisation or fall",
        "repeated contact",
        "contact followed by stuck",
        "loss of task progress",
        "repeated abstention",
        "failure to complete inspection or search",
    }
    assert all(
        set(row) == {
            "hazard_id", "hazard", "safety_requirement", "robot_region",
            "object_context", "required_treatment", "evidence", "assumption",
            "residual_limitation",
        }
        for row in trace["hazard_rows"]
    )
    assert [row["geom_index"] for row in trace["rows"]] == list(range(27))
    assert all(row["context_decision_default"] == contract.SEVERITY_OR_REQUIREMENT_UNRESOLVED
               for row in trace["rows"])
    assert all(row["scope_change_authorized"] is False for row in trace["rows"])
    assert all(row["stage_b_status"] == contract.STAGE_B_STATUS for row in trace["rows"])


def test_link_object_context_matrix_is_exhaustive_and_outcome_free() -> None:
    matrix = contract.build_link_object_context_matrix()
    contract.validate_link_object_context_matrix(matrix)
    assert matrix["component_count"] == 27
    assert matrix["context_count"] == 9
    assert matrix["expected_row_count"] == len(matrix["rows"]) == 243
    assert len({(row["geom_index"], row["context_id"]) for row in matrix["rows"]}) == 243
    assert set(matrix["category_counts"]) == set(contract.REQUIREMENT_CATEGORIES)
    assert sum(matrix["category_counts"].values()) == 243
    assert matrix["category_counts"][contract.CONDITIONAL_RECOVERABLE_CONTACT] == 0
    assert matrix["category_counts"][contract.MONITOR_AND_RECOVER] == 0
    assert all(row["category"] in contract.REQUIREMENT_CATEGORIES for row in matrix["rows"])
    assert all(row["perfect_sensing_counterfactual"] is True for row in matrix["rows"])
    assert all(row["scope_change_authorized"] is False for row in matrix["rows"])
    contract.assert_requirements_only_payload(matrix)


def test_ground_matrix_separates_ordinary_from_abnormal_calf_contexts() -> None:
    matrix = contract.build_link_object_context_matrix()
    rows = matrix["rows"]
    ordinary_calf = [row for row in rows if row["context_id"] == "GROUND_ORDINARY_SUPPORT"
                     and row["link_name"] in contract.CALF_LINK_NAMES]
    abnormal_calf = [row for row in rows if row["context_id"] == "GROUND_ABNORMAL_OR_UNKNOWN"
                     and row["link_name"] in contract.CALF_LINK_NAMES]
    assert len(ordinary_calf) == len(abnormal_calf) == 16
    assert {row["category"] for row in ordinary_calf} == {
        contract.PERMITTED_SUPPORT_OR_SELF_CONTACT
    }
    assert {row["category"] for row in abnormal_calf} == {
        contract.SEVERITY_OR_REQUIREMENT_UNRESOLVED
    }
    assert all(row["historical_h1_exclusion_preserved"] for row in ordinary_calf + abnormal_calf)


def test_matrix_validator_detects_missing_and_semantically_changed_rows() -> None:
    matrix = contract.build_link_object_context_matrix()
    changed = copy.deepcopy(matrix)
    changed["rows"].pop()
    _redigest(changed)
    with pytest.raises(contract.ContractValidationError, match="cardinality"):
        contract.validate_link_object_context_matrix(changed)
    changed = copy.deepcopy(matrix)
    changed["rows"][0]["category"] = contract.MONITOR_AND_RECOVER
    _redigest(changed)
    with pytest.raises(contract.ContractValidationError, match="does not reproduce"):
        contract.validate_link_object_context_matrix(changed)


def test_perfect_sensing_counterfactual_does_not_resolve_requirements() -> None:
    receipt = contract.build_perfect_sensing_counterfactual()
    assert receipt["primary_classification"] == contract.PRIMARY_CLASSIFICATION
    assert receipt["protected_scope_change_authorized"] is False
    assert receipt["stage_b_status"] == contract.STAGE_B_STATUS
    assert "simulated contact occurrence" in receipt["could_establish"]
    assert any("material" in item for item in receipt["cannot_establish"])
    assert any("mission progress" in item for item in receipt["cannot_establish"])


def test_stage_b_gate_is_unauthorized_and_complete() -> None:
    gate = contract.build_stage_b_gate()
    assert gate["status"] == contract.STAGE_B_STATUS
    assert gate["authorized"] is False
    assert len(gate["blockers"]) == 8
    assert any("operational design domain" in item for item in gate["blockers"])
    assert any("stopping envelope" in item for item in gate["blockers"])
    assert any("remove a protected link" in item for item in gate["prohibitions"])
    assert gate["outcome_fields_used"] == []


def test_requirements_sufficiency_gate_freezes_an_unresolved_disposition_only() -> None:
    gate = contract.build_requirements_sufficiency_gate()
    contract.validate_content_digest(gate)
    assert gate["criteria_total"] == len(gate["criteria"]) == 9
    assert gate["criteria_passed"] == 8
    assert gate["pass"] is False
    assert gate["requirements_review_disposition_frozen"] is True
    assert gate["deployment_hard_scope_frozen"] is False
    assert gate["primary_classification"] == contract.PRIMARY_CLASSIFICATION
    failed = [row for row in gate["criteria"] if not row["pass"]]
    assert [row["criterion"] for row in failed] == [
        "every excluded hard-veto context has a defensible alternative treatment"
    ]


def test_complete_receipt_is_canonical_self_digesting_and_json_roundtrippable() -> None:
    first = contract.build_contract_receipt()
    second = contract.build_contract_receipt()
    assert first == second
    contract.validate_contract_receipt(first)
    encoded = contract.canonical_json_bytes(first)
    assert json.loads(encoded) == first
    assert contract.canonical_digest({key: value for key, value in first.items()
                                      if key != "content_digest"}) == first["content_digest"]
    assert first["scope_decision"]["scope_narrowing_authorized"] is False
    assert all(value is False for value in first["training_or_evaluation"].values())


def test_receipt_validator_detects_digest_and_recomputed_semantic_tampering() -> None:
    receipt = contract.build_contract_receipt()
    changed = copy.deepcopy(receipt)
    changed["start_commit"] = "0" * 40
    with pytest.raises(contract.ContractValidationError, match="digest"):
        contract.validate_contract_receipt(changed)
    _redigest(changed)
    with pytest.raises(contract.ContractValidationError, match="start commit"):
        contract.validate_contract_receipt(changed)

    changed = copy.deepcopy(receipt)
    changed["sources"]["sources"][0]["line_ranges"] = "1"
    _redigest(changed["sources"])
    _redigest(changed)
    with pytest.raises(contract.ContractValidationError, match="sources semantic content"):
        contract.validate_contract_receipt(changed)

    changed = copy.deepcopy(receipt)
    changed["classifications"]["secondary_exactly"].append("UNAUTHORIZED")
    _redigest(changed)
    with pytest.raises(contract.ContractValidationError, match="secondary"):
        contract.validate_contract_receipt(changed)


def test_no_builder_exposes_scientific_execution_or_scope_narrowing() -> None:
    receipt = contract.build_contract_receipt()
    assert receipt["requirements_only_barrier"]["outcome_fields_used"] == []
    assert receipt["requirements_only_barrier"]["sensor_or_model_performance_may_select_scope"] is False
    assert receipt["scope_decision"]["protected_links_removed"] == []
    assert receipt["scope_decision"]["protected_shapes_removed"] == []
    assert receipt["scope_decision"]["contact_labels_changed"] is False
    assert receipt["stage_b_gate"]["authorized"] is False
    assert receipt["classifications"]["other_secondary_classifications_authorized"] is False
