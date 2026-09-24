from __future__ import annotations

import copy
import json

import pytest

from lewm.safety import physical_graph_edge_handoff_qualification_v1_contract as V1
from lewm.safety import physical_graph_edge_handoff_qualification_v2_contract as C


def test_v2_identity_lineage_subjects_outputs_and_custody_binding() -> None:
    assert C.EXPERIMENT_ID == "PHYSICAL_GRAPH_EDGE_HANDOFF_QUALIFICATION_V2"
    assert C.SOURCE_PARENT_COMMIT == "3dfff6caec1c3162d8123c737d04bcdd42799653"
    assert C.CONTRACT_FREEZE_COMMIT_SUBJECT == (
        "Freeze corrected physical graph edge handoff qualification V2"
    )
    assert C.RESULT_COMMIT_SUBJECT == (
        "Evaluate corrected physical graph edge handoff qualification V2"
    )
    assert str(C.OUTPUT_ROOT).endswith("physical_graph_edge_handoff_qualification_v2")
    assert str(C.MATERIAL_ROOT).endswith("physical_graph_edge_handoff_qualification_v2_material")
    assert str(C.EXTERNAL_REGENERATION_RECEIPT).endswith(
        "physical_graph_edge_handoff_qualification_v2_regeneration_receipt.json"
    )
    assert C.OUTPUT_LEAF_COUNT == len(C.SUCCESS_OUTPUT_LEAVES) == 26
    assert len(set(C.SUCCESS_OUTPUT_LEAVES)) == 26
    assert C.REPRODUCTION_MISMATCH_LEAVES == (
        "contract.json",
        "v1_custody_and_nonreuse.json",
        "scientific_invariance_receipt.json",
        "v1_v2_first_eight_reproduction.json",
    )
    assert C.V1_CUSTODY_RECEIPT_BINDING == {
        "path": "/home/andrewknowles/RecoveryStorage/LeWMQuad-v3/physical_graph_edge_handoff_qualification_v1_custody_receipt.json",
        "bytes": 18403,
        "sha256": "bb4950d2bde0bf1971e643c746b15a28a1949b1ef3d70736bdaae9da45282d32",
    }


def test_every_v1_scientific_field_spec_and_source_dependency_is_exact() -> None:
    contract = C.build_contract()
    assert C.scientific_invariance_projection(contract) == C.V1_SCIENTIFIC_PROJECTION
    assert C.scientific_constant_projection() == C.V1_SCIENTIFIC_CONSTANT_PROJECTION
    assert C.V2_SCIENTIFIC_CONSTANTS_SHA256 == C.V1_SCIENTIFIC_CONSTANTS_SHA256
    assert "HANDOFF_GATE" in C.V1_SCIENTIFIC_CONSTANT_NAMES
    assert "SOURCE_DEPENDENCY_PATHS" in C.V1_SCIENTIFIC_CONSTANT_NAMES
    assert "EXPERIMENT_ID" not in C.V1_SCIENTIFIC_CONSTANT_NAMES
    assert C.build_candidate_specs() == V1.build_candidate_specs()
    assert C.build_prospective_pool_specs() == V1.build_prospective_pool_specs()
    assert tuple(C.SOURCE_DEPENDENCY_PATHS) == tuple(V1.SOURCE_DEPENDENCY_PATHS)
    assert contract["source_baseline_commit"] == V1.SOURCE_BASELINE_COMMIT
    assert contract["handoff_gate"] == V1.build_contract()["handoff_gate"]
    assert contract["metric_formulas"] == V1.build_contract()["metric_formulas"]
    assert contract["classification_precedence"] == list(V1.CLASSIFICATION_PRECEDENCE)
    assert contract["next_decisions"] == V1.NEXT_DECISION_BY_CLASSIFICATION
    assert all(spec["candidate_spec_id"].startswith("pgehq-v1-") for spec in C.build_candidate_specs())
    assert [spec["procedural_seed"] for spec in C.build_candidate_specs()] == [
        spec["procedural_seed"] for spec in V1.build_candidate_specs()
    ]
    assert C.validate_contract(contract) == contract
    assert C.SCIENTIFIC_INVARIANCE_AUTHORITY["scientific_constants_equal"] is True
    identity = C.V1_COMPATIBILITY_IDENTITY_AUTHORITY
    assert identity["official_experiment_id"] == C.EXPERIMENT_ID
    assert identity["frozen_inherited_scientific_identity_salt"] == C.V1_EXPERIMENT_ID
    assert identity["v1_scientific_completion_claimed"] is False
    assert identity["v2_execution_and_result_identity_claimed"] is True
    alignment = C.PORT_HEADING_IMPLEMENTATION_ALIGNMENT_AUTHORITY
    assert alignment["disposition"] == (
        "INHERITED_PORT_HEADING_IMPLEMENTATION_ALIGNED_TO_FROZEN_AUTHORITY"
    )
    assert alignment["changed_output_fields"] == ["directed_port_world[2]"]
    assert alignment["changes_frozen_scientific_design"] is False
    assert alignment["uses_outcome_values_to_choose_formula_or_parameter"] is False
    candidate_alignment = C.CANDIDATE_PORT_METRIC_IMPLEMENTATION_ALIGNMENT_AUTHORITY
    assert candidate_alignment["disposition"] == (
        "INHERITED_CANDIDATE_PORT_METRIC_IMPLEMENTATION_ALIGNED_TO_FROZEN_AUTHORITY"
    )
    assert candidate_alignment["changed_output_fields"] == [
        "port_progress_m",
        "lateral_error_m",
        "positive_port_progress",
    ]
    assert candidate_alignment["changes_frozen_scientific_design"] is False
    assert candidate_alignment[
        "uses_outcome_values_to_choose_formula_or_parameter"
    ] is False
    persisted = json.loads(C.canonical_json_bytes(contract))
    assert C.validate_contract(persisted) == persisted


def test_contract_rejects_scientific_and_allowed_identity_drift(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    contract = C.build_contract()
    scientific = copy.deepcopy(contract)
    scientific["handoff_gate"]["coverage_rate_minimum"] = 0.91
    scientific = C.attach_content_digest({
        key: value for key, value in scientific.items() if key != "content_digest"
    })
    with pytest.raises(C.PhysicalGraphEdgeHandoffV2ContractError):
        C.validate_contract(scientific)
    identity = copy.deepcopy(contract)
    identity["output"]["leaf_count"] = 25
    identity = C.attach_content_digest({
        key: value for key, value in identity.items() if key != "content_digest"
    })
    with pytest.raises(C.PhysicalGraphEdgeHandoffV2ContractError):
        C.validate_contract(identity)
    monkeypatch.setattr(C, "HANDOFF_GATE", {**C.HANDOFF_GATE, "coverage_rate_minimum": 0.91})
    with pytest.raises(
        C.PhysicalGraphEdgeHandoffV2ContractError,
        match="scientific constants",
    ):
        C.build_contract()


def test_raw_persisted_bytes_authority_and_ten_regressions_are_exact() -> None:
    authority = C.PERSISTED_ARRAY_HASH_AUTHORITY
    assert authority["digest_domain"] == (
        "exact C-contiguous array.tobytes(order='C') only"
    )
    assert authority["tolerance"] == 0
    defect = authority["identified_defect_field"]
    assert defect["digest_domain"] == "exact C-contiguous persisted bytes only"
    assert defect["dtype_str"] == "<f8" and defect["per_snapshot_shape"] == [3]
    assert defect["first_eight_expected_raw_bytes_sha256"] == (
        "9d908ecfb6b256def8b49a7c504e6c889c4b0e41fe6ce3e01863dd7b61a20aa0"
    )
    assert "dtype/shape header" in authority["forbidden_domains"][0]
    assert C.REGRESSION_REQUIREMENT_IDS == (
        "FLOAT64_EXACT_PERSISTED_BYTES_VALID",
        "FLOAT32_CAST_DIGEST_REJECTED",
        "PERSISTED_DTYPE_MISMATCH_REJECTED",
        "PERSISTED_SHAPE_MISMATCH_REJECTED",
        "PERSISTED_VALUE_MUTATION_REJECTED",
        "NONCONTIGUOUS_VIEW_C_CONTIGUOUS_NO_DTYPE_CAST",
        "SAVE_RELOAD_DIGEST_IDENTICAL",
        "OTHER_METADATA_BINDINGS_UNCHANGED",
        "V1_SCIENTIFIC_CONSTANTS_AND_SOURCE_PATHS_UNCHANGED",
        "FIRST_POOL_PRODUCTION_WRITER_VALIDATOR_PASS",
    )
    assert C.REGRESSION_GATE_AUTHORITY["required_before_simulator_creation"] is True
    assert C.V2_CORRECTION_RUNTIME_POLICY["v1_runtime_artifact_reuse"] is False


def test_first_eight_gate_and_mismatch_are_technical_not_scientific() -> None:
    authority = C.FIRST_EIGHT_REPRODUCTION_AUTHORITY
    assert authority["pool_indices"] == list(range(8))
    assert authority["required_before_pool_index"] == 8
    assert authority["mismatch_status"] == "V1_V2_PHYSICAL_REPRODUCTION_MISMATCH"
    assert authority["mismatch_is_technical_terminal_outside_scientific_classes"] is True
    assert authority["mismatch_output_leaves"] == list(C.REPRODUCTION_MISMATCH_LEAVES)
    assert authority["corrected_v2_previous_command_binding"] == (
        C.PERSISTED_ARRAY_HASH_AUTHORITY["identified_defect_field"]
    )
    assert "V1_V2_PHYSICAL_REPRODUCTION_MISMATCH" not in C.PRIMARY_CLASSIFICATIONS


def test_runtime_roundtrip_and_changed_path_source_closure_are_fail_closed() -> None:
    runtime = C.build_runtime_contract("1" * 40)
    assert runtime["source_parent_commit"] == C.V1_SOURCE_FREEZE_COMMIT
    assert runtime["v1_custody_receipt_binding"] == C.V1_CUSTODY_RECEIPT_BINDING
    assert C.validate_runtime_contract(runtime, source_freeze_commit="1" * 40) == runtime
    persisted = json.loads(C.canonical_json_bytes(runtime))
    assert C.validate_runtime_contract(
        persisted, source_freeze_commit="1" * 40
    ) == persisted
    assert len(C.TRACKED_SOURCE_PATHS) == len(set(C.TRACKED_SOURCE_PATHS)) == 15
    assert len(C.SOURCE_DEPENDENCY_PATHS) == 58
    assert len(C.SOURCE_CLOSURE_PATHS) == len(set(C.SOURCE_CLOSURE_PATHS)) == 71
    assert not any(path.startswith("docs/") for path in C.SOURCE_CLOSURE_PATHS)
    assert C.SOURCE_CLOSURE_PATHS[:8] == C.TRACKED_SOURCE_PATHS[7:]
    assert "lewm/safety/physical_graph_edge_handoff_qualification_v1_contract.py" in (
        C.V2_WRAPPER_DEPENDENCY_PATHS
    )
    assert (
        "lewm/tests/test_run_physical_graph_edge_handoff_qualification_v1.py"
        in C.V2_WRAPPER_DEPENDENCY_PATHS
    )
    tampered = copy.deepcopy(runtime)
    tampered["v1_custody_receipt_binding"]["bytes"] += 1
    tampered = C.attach_content_digest({
        key: value for key, value in tampered.items() if key != "content_digest"
    })
    with pytest.raises(C.PhysicalGraphEdgeHandoffV2ContractError):
        C.validate_runtime_contract(tampered)
