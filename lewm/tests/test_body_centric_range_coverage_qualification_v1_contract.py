from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest

from lewm.safety import body_centric_range_coverage_qualification_v1_contract as contract


def test_canonical_json_is_stable_and_rejects_ambiguous_values() -> None:
    assert contract.canonical_json_bytes({"z": 2, "a": [True, None, 1.25]}) == (
        b'{"a":[true,null,1.25],"z":2}'
    )
    assert contract.canonical_json_sha256({"b": 1, "a": 2}) == hashlib.sha256(
        b'{"a":2,"b":1}'
    ).hexdigest()
    with pytest.raises(contract.ContractError, match="non-finite"):
        contract.canonical_json_bytes({"bad": float("nan")})
    with pytest.raises(contract.ContractError, match="non-string"):
        contract.canonical_json_bytes({1: "ambiguous"})
    with pytest.raises(contract.ContractError, match="unsupported JSON type"):
        contract.canonical_json_bytes({"tuple": (1, 2)})


def test_contract_self_digest_and_copy_are_stable() -> None:
    first = contract.build_contract()
    second = contract.build_contract()
    assert first == second == contract.contract_receipt() == contract.CONTRACT
    assert first is not second
    declared = first.pop("contract_sha256")
    assert declared == contract.CONTRACT_SHA256
    assert contract.canonical_json_sha256(first) == declared
    assert hashlib.sha256(contract.contract_receipt_bytes()).hexdigest() == (
        contract.CONTRACT_RECEIPT_SHA256
    )
    assert contract.contract_receipt_bytes().endswith(b"\n")
    assert not contract.contract_receipt_bytes().endswith(b"\n\n")


def test_contract_freezes_all_conditions_modes_and_cross_product() -> None:
    receipt = contract.build_contract()
    assert tuple(row["id"] for row in receipt["conditions"]) == contract.CONDITION_IDS
    assert tuple(row["id"] for row in receipt["evidence_modes"]) == contract.EVIDENCE_MODE_IDS
    assert receipt["evaluation_matrix"] == [
        {"condition_id": condition_id, "evidence_mode": evidence_mode}
        for condition_id in contract.CONDITION_IDS
        for evidence_mode in contract.EVIDENCE_MODE_IDS
    ]
    assert len(receipt["evaluation_matrix"]) == 8


def test_frozen_corpus_lineage_and_role_isolation() -> None:
    receipt = contract.build_contract()
    bindings = receipt["predecessor_bindings"]
    assert bindings["experiment"] == "EXPLICIT_PER_LINK_GEOMETRIC_MICRO_STATE_UPPER_BOUND_V1"
    assert bindings["source_lineage_commit"] == "10b3a190d506830e6a87e04a0f1c832b92295bd7"
    assert bindings["completed_result_commit"] == "034c2fb902997ac29e2742fc4ddc2c28ad1706b6"
    assert bindings["corpus_logical_digest"] == "e41d9926cb7f0f9e1158d09a88b746547806411950b9cac7ebe58aa500a92223"
    assert bindings["corpus_index_sha256"] == "c1055724ebffd71c67b5424b4e447d223a828d3bc4da01369486bff73ad265f0"
    assert bindings["geometry_index_sha256"] == "67b473e6d0e0c422e4b2916b800399c98d5ec60383d398ff907fe2b8909a1d7f"
    assert bindings["role_split_sha256"] == "eb2b41ca3ca4d4f7d2d2fc41495944e306e39798ede8865dd0904fa6c3d88021"
    assert bindings["action_contract_sha256"] == "cf8df092e8eff61d04348ebfe22b5e6a0cd31b5f39a4e05e45242752b3e5dc06"
    assert bindings["repaired_row_ledger_sha256"] == "63726e042e793d06784236b9dcc37c3844c798b8f526d03e4f19517186d5cc94"
    assert bindings["frozen_state_count"] == 176
    assert bindings["frozen_current_and_successor_transition_count"] == 29470
    assert bindings["frozen_physics_frame_count"] == 1473500
    assert bindings["role_state_counts"] == {
        "training": 128,
        "internal_calibration": 24,
        "development_held_out": 24,
    }
    assert bindings["oracle_label_array"] == "frozen_contact_label"
    assert "authoritative physics-rate H1 target" in bindings["contact_authority"]
    assert bindings["action_authority"] == "unique deployable applied-action contract"
    assert bindings["route_authority"] == "deterministic H3 route scores"
    assert bindings["mutable_fields"] == []
    assert receipt["roles"]["internal_calibration"]["use"] == (
        "the only threshold-calibration role"
    )
    assert receipt["roles"]["development_held_out"]["use"] == (
        "evaluation only after all thresholds are frozen"
    )
    assert receipt["roles"]["untouched_g2"] == "FORBIDDEN_NOT_READ"


def test_failed_attempt_and_exact_sensor_materialization_amendment_are_frozen() -> None:
    receipt = contract.build_contract()
    assert receipt["schema_version"] == (
        "body_centric_range_coverage_qualification_v1.contract.v2"
    )
    amendment = receipt["prospective_execution_amendment"]
    assert amendment["id"] == "EXACT_SENSOR_MATERIALIZATION_MAP_AMENDMENT_V1"
    assert amendment["status"] == "FROZEN_BEFORE_REEXECUTION"
    failed = amendment["failed_attempt"]
    assert failed["source_freeze_commit"] == (
        "3ef985d7fcea8c26609fd6cc8a1d5e66507e9c1c"
    )
    assert failed["path"] == (
        "/home/andrewknowles/RecoveryStorage/LeWMQuad-v3/"
        "body_centric_range_coverage_qualification_v1__failed_3ef985d_"
        "copy_geometry_assertion"
    )
    assert failed["receipt_sha256"] == (
        "277e76da379f7b992ae5b5df27d7cdf88b8f5b56d63720bb765e2074549f51e4"
    )
    assert failed["file_manifest_sha256"] == (
        "e6244e5d3284e29ad17a949a93b0d21e4d94e93557d57c3cafcda25feabae0a8"
    )
    assert failed["terminal_absences"] == {
        "materialization_index": "ABSENT",
        "calibration_thresholds": "ABSENT",
        "heldout_metric_evaluation": "NOT_RUN",
        "result": "ABSENT",
    }
    custody = amendment["custody"]
    assert custody["prior_state_shards_reusable_after_refreeze"] is False
    assert custody["canonical_output_root_remains_unchanged"] is True
    assert custody["restart_policy"] == (
        "fresh preexecution receipt and materialisation from state zero"
    )

    maps = receipt["representative_maps"]
    action = maps["decision_action_copy_map"]
    assert action["contract_id"] == "DECISION_ACTION_COPY_MAP"
    assert action["implementation_type"] == "AppliedActionCopyMap"
    assert action["npz_array"] == "action_representative_transition"
    assert action["legacy_npz_alias"] == "representative_transition"
    assert action["sensor_reuse_authority"] is False
    assert action["representative_count"] == 13385
    assert "diagnostic only" in action["applied_action_copy_validation"]

    geometry = maps["exact_sensor_materialization_map"]
    assert geometry["contract_id"] == "EXACT_SENSOR_MATERIALIZATION_MAP"
    assert geometry["implementation_type"] == "ExactGeometryMaterializationMap"
    assert geometry["npz_array"] == "geometry_representative_transition"
    assert geometry["exact_array_fields"] == [
        "qpos",
        "link_transform",
        "geom_transform",
        "native_contact",
        "exact_contact",
        "frozen_contact_label",
    ]
    assert geometry["boundary_requirement"] == "identical boundary snapshot digest"
    assert "tolerance is forbidden" in geometry["equality"]
    assert geometry["representative_count"] == 13584
    assert geometry["exact_reused_transition_pairs"] == 15886
    assert geometry["independently_materialized_nonexact_action_copy_pairs"] == 199
    assert geometry["nonexact_action_copy_state_count"] == 81
    assert geometry["nonexact_action_copy_state_role_counts"] == {
        "training": 58,
        "internal_calibration": 12,
        "development_held_out": 11,
    }
    cardinality = maps["cardinality_identity"]
    assert cardinality["decision_action_representatives"] + 199 == (
        cardinality["exact_sensor_materialization_representatives"]
    )
    assert cardinality["exact_sensor_materialization_representatives"] + (
        cardinality["exact_sensor_reused_transition_pairs"]
    ) == cardinality["frozen_transition_count"] == 29470
    preflight = maps["full_corpus_structural_preflight"]
    assert preflight["state_count"] == 176
    assert preflight["failure_policy"] == "fail closed before materialisation"


def test_assumed_l2_sensor_and_mounts_are_exactly_bound() -> None:
    receipt = contract.build_contract()
    sensor = receipt["hardware_binding"]
    assert sensor["binding_class"] == "ASSUMED_GO2_HEAD_LIDAR_L2"
    assert sensor["secondary_classification_required"] == "ASSUMED_SENSOR_CONTRACT"
    assert sensor["selected_profile"] == "wide_negative_angle_360_x_96"
    assert sensor["profile_default_claim"] == "NOT_CLAIMED"
    assert sensor["fov"] == {
        "horizontal_deg": 360.0,
        "vertical_deg": 96.0,
        "elevation_min_deg": -6.0,
        "elevation_max_deg": 90.0,
        "elevation_interval_basis": (
            "inference from the documented 96 degree wide/negative-angle mode and "
            "documented negative-angle extent"
        ),
    }
    assert sensor["rates"]["effective_points_per_s"] == 64000
    assert sensor["rates"]["raw_sampling_per_s"] == 128000
    assert sensor["range"]["near_blind_region_m"] == 0.05
    assert sensor["range"]["maximum_m_at_90_percent_reflectivity"] == 30.0
    assert sensor["range"]["maximum_m_at_10_percent_reflectivity"] == 15.0
    assert sensor["coordinate_frame"]["imu_origin_in_lidar_m"] == [
        -0.007698,
        -0.014655,
        0.00667,
    ]

    platform = receipt["mounts"]["platform"]
    assert platform["parent_link"] == "base"
    assert platform["child_frame"] == "radar"
    assert platform["translation_m"] == [0.28945, 0.0, -0.046825]
    assert platform["rotation_rpy_rad"] == [0.0, 2.8782, 0.0]
    body = receipt["mounts"]["body_centric"]
    assert body["translation_m"] == [0.0, 0.0, 0.067]
    assert body["trunk_top_z_m"] + body["mechanical_clearance_m"] == pytest.approx(0.067)
    assert body["tuning"] == (
        "single prospectively selected mount; no contact-label or outcome tuning"
    )


def test_scan_phase_timing_dense_semantics_and_visibility_are_frozen() -> None:
    receipt = contract.build_contract()
    scan = receipt["realistic_scan_approximation"]
    assert scan["classification"] == "APPROXIMATED_REALISTIC_PLATFORM_SCAN"
    assert scan["effective_ray_rate_hz"] == 64000
    assert scan["transition_ray_count"] == 6400
    assert scan["timestamp_law"] == (
        "t_k = (k + 0.5) / 64000 seconds in the 100 ms window"
    )
    assert scan["azimuth_law"] == "-pi + 2*pi*frac(azimuth_phase + 5.55*t)"
    assert "216*t" in scan["elevation_law"]
    assert scan["randomness"] == (
        "none; phase hashing is deterministic and outcome-independent"
    )

    dense_platform = receipt["conditions"][2]
    dense_body = receipt["conditions"][3]
    assert dense_platform["elevation_min_deg"] == -6.0
    assert dense_platform["elevation_max_deg"] == 90.0
    assert dense_body["elevation_min_deg"] == -90.0
    assert dense_body["elevation_max_deg"] == 90.0
    assert dense_body["mount_search_count"] == 1
    assert "mathematical directional continuum" in dense_platform["ray_semantics"]
    assert "full-sphere" in dense_body["ray_semantics"]

    visibility = receipt["visibility_and_reduction"]
    assert "geom indices 1 and 2" in visibility["robot_self_occlusion"]
    housing = receipt["mounts"]["platform"]["emitting_housing_ray_policy"]
    assert housing["exempt_frozen_geom_indices"] == [1, 2]
    assert housing["scope"].startswith("exclude only geometry indices 1 and 2")
    assert visibility["self_return_clearance_use"] == "excluded from environment clearance"
    assert visibility["occluded_sector_semantics"] == "unsupported, never free"
    assert visibility["equal_distance_tie"] == "robot self-return wins conservatively"
    assert visibility["local_target_support_radius_m"] == 0.1
    assert "support and attribution only" in visibility["local_target_support_scope"]
    assert "shortest-arc normalized linear" in visibility["motion_compensation"]


def test_boundary_keyed_phase_api_has_a_frozen_known_answer() -> None:
    boundary_digest = "0123456789abcdef" * 4
    phases = contract.derive_l2_scan_phases(
        boundary_snapshot_digest=boundary_digest,
    )
    assert phases == {
        "phase_namespace": "BODY_CENTRIC_RANGE_COVERAGE_QUALIFICATION_V1/L2_PHASE_V1",
        "boundary_snapshot_digest": boundary_digest,
        "phase_digest_sha256": (
            "e2e4cc3ebefde748455314aa18b61c5ad946f153e83ce95672a02dcb16e5ed33"
        ),
        "horizontal_phase_cycles": 0.8863036778629901,
        "vertical_phase_cycles": 0.2707989611887673,
    }
    assert phases == contract.derive_l2_scan_phases(
        boundary_snapshot_digest=boundary_digest,
    )
    with pytest.raises(contract.ContractError, match="boundary_snapshot_digest"):
        contract.derive_l2_scan_phases(
            boundary_snapshot_digest="not-a-sha256",
        )
    with pytest.raises(contract.ContractError, match="lowercase"):
        contract.derive_l2_scan_phases(boundary_snapshot_digest="A" * 64)

    scan = contract.build_contract()["realistic_scan_approximation"]
    assert "every candidate action" in scan["candidate_invariance"]
    assert "successor/current" in scan["successor_boundary_rule"]
    assert "candidate identity and successor endpoint are excluded" in scan["candidate_invariance"]


def test_threshold_rule_gate_and_classifications_cannot_drift() -> None:
    receipt = contract.build_contract()
    calibration = receipt["threshold_calibration"]
    assert calibration["role"] == "internal_calibration"
    assert calibration["contact_rule"] == (
        "score <= threshold is predicted contact; exact ties are contact-positive"
    )
    assert calibration["eligibility"] == {
        "combined_current_successor_contact_recall_gte": 0.95,
        "combined_current_successor_false_negative_rate_lte": 0.05,
    }
    assert len(calibration["lexicographic_selection"]) == 7
    assert calibration["lexicographic_selection"][-1] == (
        "more conservative threshold (numerically larger clearance threshold)"
    )

    gate = receipt["immutable_gate"]
    assert gate["mode"] == "TRUE_FUTURE_OBSERVABILITY_CLOUD only"
    assert gate["current_contact_auc_gte"] == 0.9
    assert gate["successor_contact_auc_gte"] == 0.9
    assert gate["combined_contact_recall_gte"] == 0.95
    assert gate["combined_contact_false_negative_rate_lte"] == 0.05
    assert gate["safe_action_zero_nonzero_accuracy_gte"] == 0.9
    assert gate["safe_action_false_nonzero_rate_lte"] == 0.05
    assert gate["oracle_viable_states_retaining_action_gte"] == 18
    assert gate["oracle_nonviable_correct_abstentions_eq"] == 4
    assert gate["selected_immediate_contacts_eq"] == 0
    assert gate["selected_nonviable_successors_eq"] == 0
    assert gate["h3_route_progress_fraction_of_exact_geometry_gte"] == 0.8
    assert gate["normalized_viability_constrained_regret_lte"] == 0.2
    assert gate["best_admissible_top_3_gte"] == 0.75
    assert gate["no_family_collapse"]["required"] is True
    assert gate["weakening"] == "forbidden"

    assert tuple(receipt["classifications"]["primary_exactly_one"]) == (
        contract.PRIMARY_CLASSIFICATIONS
    )
    assert tuple(receipt["classifications"]["secondary"]) == (
        contract.SECONDARY_CLASSIFICATIONS
    )
    assert tuple(receipt["coverage_attribution"]["error_classes"]) == (
        contract.COVERAGE_ERROR_CLASSES
    )


def test_storage_roots_preflight_and_audit_selection_are_prospective() -> None:
    receipt = contract.build_contract()
    storage = receipt["storage"]
    expected_root = (
        "/home/andrewknowles/RecoveryStorage/LeWMQuad-v3/"
        "body_centric_range_coverage_qualification_v1"
    )
    assert storage["output_root"] == expected_root
    assert storage["cache_root"].startswith(expected_root + "/")
    assert storage["output_filesystem_required_type"] == "ext4"
    assert storage["output_must_be_different_device_from_workspace"] is True
    assert storage["capacity_unit"] == "decimal GB = 1,000,000,000 bytes"
    assert storage["minimum_output_free_bytes"] == 40_000_000_000
    assert storage["minimum_workspace_free_bytes"] == 20_000_000_000
    assert storage["temporary_storage_ceiling_bytes"] == 30_000_000_000
    assert storage["final_storage_ceiling_bytes"] == 20_000_000_000
    assert storage["evaluation_mode"] == "streaming"

    audit = receipt["raw_audit_subset"]
    assert audit["selection_depends_on_sensor_results"] is False
    assert "single lowest SHA-256" in audit["scientific_selection"]
    assert audit["other_raw_ray_or_point_persistence"] == "forbidden"
    assert "not a replacement finite angular scan" in audit["dense_continuum_rule"]


def test_contract_validation_rejects_tampering_even_when_resigned() -> None:
    tampered = contract.build_contract()
    tampered["mounts"]["body_centric"]["translation_m"][2] = 0.068
    tampered.pop("contract_sha256")
    tampered["contract_sha256"] = contract.canonical_json_sha256(tampered)
    with pytest.raises(contract.ContractError, match="prospectively frozen"):
        contract.validate_contract(tampered)

    bad_digest = contract.build_contract()
    bad_digest["contract_sha256"] = "0" * 64
    with pytest.raises(contract.ContractError, match="content SHA-256 mismatch"):
        contract.validate_contract(bad_digest)

    extra = contract.build_contract()
    extra["unexpected"] = True
    with pytest.raises(contract.ContractError, match="top-level keys"):
        contract.validate_contract(extra)


def test_output_schema_is_self_digesting_and_row_level(tmp_path: Path) -> None:
    schema = contract.build_output_schema()
    assert schema["schema_version"] == (
        "body_centric_range_coverage_qualification_v1.output.v2"
    )
    declared = schema.pop("output_schema_sha256")
    assert declared == contract.OUTPUT_SCHEMA_SHA256
    assert contract.canonical_json_sha256(schema) == declared
    assert {
        "transition_evidence",
        "per_link_evidence",
        "coverage_errors",
        "raw_audit_manifest",
    }.issubset(schema["files"])
    assert "minimum_observed_environment_clearance_m" in schema["files"][
        "per_link_evidence"
    ]["required_keys"]
    assert "error_class" in schema["files"]["coverage_errors"]["required_keys"]
    preexecution = schema["files"]["preexecution_receipt"]
    assert "exact_geometry_materialization_preflight" in preexecution["required_keys"]
    assert "prospective_execution_amendment_validation" in preexecution["required_keys"]
    index = schema["files"]["materialization_index"]
    assert "geometry_representatives" in index["required_keys"]
    assert "geometry_representatives" in index["record_required_keys"]
    state = schema["files"]["state_evidence"]
    assert state["npz_required_arrays"] == [
        "representative_transition",
        "action_representative_transition",
        "geometry_representative_transition",
    ]
    assert {
        "geometry_representatives",
        "geometry_current_representatives",
        "geometry_successor_representatives",
        "applied_action_copy_validation",
        "exact_geometry_materialization_validation",
        "representative_mappings",
    }.issubset(state["required_keys"])
    assert state["scan_transition_index_semantics"].startswith(
        "transition_index is the exact geometry-source transition"
    )
    assert "may be false" in state["validation_semantics"][
        "applied_action_copy_validation"
    ]
    assert "must pass" in state["validation_semantics"][
        "exact_geometry_materialization_validation"
    ]
    transition = schema["files"]["transition_evidence"]
    assert {
        "action_representative_transition_index",
        "geometry_representative_transition_index",
    }.issubset(transition["required_keys"])
    per_link = schema["files"]["per_link_evidence"]
    assert {
        "state_id",
        "transition_index",
        "action_representative_transition_index",
        "geometry_representative_transition_index",
    }.issubset(per_link["required_keys"])
    raw = schema["files"]["raw_audit_manifest"]
    assert {
        "source_representative_transition_uid",
        "source_action_representative_transition_uid",
        "source_geometry_representative_transition_uid",
    }.issubset(raw["required_keys"])
    assert "geometry_representatives" in schema["files"]["result"][
        "materialisation_count_required_keys"
    ]

    path = tmp_path / "nested" / "schema.json"
    assert contract.write_output_schema(path) == path
    assert path.read_bytes() == contract.output_schema_receipt_bytes()
    assert contract.load_and_validate_output_schema(path) == contract.OUTPUT_SCHEMA


def test_immutable_receipt_writer_is_idempotent_and_fail_closed(tmp_path: Path) -> None:
    path = tmp_path / "receipt" / "contract.json"
    assert contract.write_contract(path) == path
    first = path.read_bytes()
    assert first == contract.contract_receipt_bytes()
    assert json.loads(first) == contract.CONTRACT
    assert contract.write_contract(path) == path
    assert path.read_bytes() == first
    assert contract.load_and_validate_contract(path) == contract.CONTRACT

    path.write_bytes(b"{}\n")
    with pytest.raises(contract.ContractError, match="refusing to overwrite"):
        contract.write_contract(path)
    with pytest.raises(contract.ContractError, match="byte-identical"):
        contract.load_and_validate_contract(path)


def test_build_contract_does_not_share_mutable_state() -> None:
    changed = contract.build_contract()
    changed["conditions"][0]["azimuth_bins"] = 1
    changed["prohibitions"].append("invented")
    fresh = contract.build_contract()
    assert fresh["conditions"][0]["azimuth_bins"] == 180
    assert "invented" not in fresh["prohibitions"]
    assert contract.validate_contract(copy.deepcopy(fresh)) == fresh


def test_no_training_g2_jepa_memory_or_navigation_authority() -> None:
    receipt = contract.build_contract()
    prohibitions = set(receipt["prohibitions"])
    assert {
        "model training",
        "fresh panel or corpus collection",
        "opening or executing the JEPA predictor",
        "reading or opening untouched G2 evaluation",
        "retraining recurrent memory",
        "learned closed-loop navigation",
        "implementing memory, novelty, routing, or beacon capture",
    }.issubset(prohibitions)
    assert receipt["seeds"]["model_seed"] is None
    assert receipt["source_and_environment"]["known_runtime"]["tinyquadjepa_required"] is False
    assert receipt["source_and_environment"]["fixture_test"] == (
        "lewm/tests/test_body_centric_range_coverage_v1.py"
    )
