from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest

from lewm.safety import (
    minimum_multi_origin_body_range_coverage_qualification_v1_contract as contract,
)


def test_canonical_json_is_stable_and_fail_closed() -> None:
    assert contract.canonical_json_bytes({"z": 2, "a": [True, None, 1.25]}) == (
        b'{"a":[true,null,1.25],"z":2}'
    )
    assert contract.canonical_json_sha256({"b": 1, "a": 2}) == hashlib.sha256(
        b'{"a":2,"b":1}'
    ).hexdigest()
    with pytest.raises(contract.ContractError, match="non-finite"):
        contract.canonical_json_bytes({"bad": float("nan")})
    with pytest.raises(contract.ContractError, match="non-string"):
        contract.canonical_json_bytes({1: "bad"})
    with pytest.raises(contract.ContractError, match="unsupported JSON type"):
        contract.canonical_json_bytes({"bad": (1, 2)})


def test_contract_and_schema_are_self_digesting_independent_copies() -> None:
    first = contract.build_contract()
    second = contract.build_contract()
    assert first == second == contract.CONTRACT == contract.contract_receipt()
    assert first is not second
    declared = first.pop("contract_sha256")
    assert declared == contract.CONTRACT_SHA256
    assert contract.canonical_json_sha256(first) == declared
    assert hashlib.sha256(contract.contract_receipt_bytes()).hexdigest() == (
        contract.CONTRACT_RECEIPT_SHA256
    )
    assert contract.contract_receipt_bytes().endswith(b"\n")

    schema = contract.build_output_schema()
    schema_digest = schema.pop("output_schema_sha256")
    assert schema_digest == contract.OUTPUT_SCHEMA_SHA256
    assert contract.canonical_json_sha256(schema) == schema_digest


def test_predecessor_result_and_frozen_corpus_are_exact() -> None:
    receipt = contract.build_contract()
    predecessor = receipt["predecessor_bindings"]
    assert predecessor["experiment"] == "BODY_CENTRIC_RANGE_COVERAGE_QUALIFICATION_V1"
    assert predecessor["source_freeze_commit"] == (
        "6fb55dec810b8fb8337d4519096f17a294c78425"
    )
    assert predecessor["completed_result_commit"] == (
        "d9748abe0fad0a25face56801f6b0c5e699db92f"
    )
    assert predecessor["result_content_sha256"] == (
        "98699ac43046a4d1f425998217d5527637209ca233a872cf667e484123a967ac"
    )
    assert predecessor["result_file_sha256"] == (
        "8844c0ee5a8bcdd28f505d64a670b5dee595933af05a00290f327cb8f3702019"
    )
    assert predecessor["persistence_receipt_file_sha256"] == (
        "12bdb2e6e0e7b54a9ad1947126715ac85d367afcbfd403ca1c80ebd593f51a43"
    )
    assert predecessor["transition_evidence_sha256"] == (
        "c25c5a0c8bcea8cdad3e7f14946126ae53098ed6d14f90a04ace5d053e303991"
    )
    assert predecessor["per_link_evidence_sha256"] == (
        "78f5e21416b603fb6481968f2e6d70753cbb138dff49f85f3910ec1c12bae218"
    )
    assert predecessor["corpus_logical_digest"] == (
        "e41d9926cb7f0f9e1158d09a88b746547806411950b9cac7ebe58aa500a92223"
    )
    assert predecessor["corpus_index_sha256"] == (
        "c1055724ebffd71c67b5424b4e447d223a828d3bc4da01369486bff73ad265f0"
    )
    assert predecessor["action_contract_sha256"] == (
        "cf8df092e8eff61d04348ebfe22b5e6a0cd31b5f39a4e05e45242752b3e5dc06"
    )
    assert predecessor["repaired_row_ledger_sha256"] == (
        "63726e042e793d06784236b9dcc37c3844c798b8f526d03e4f19517186d5cc94"
    )
    assert predecessor["primary_classification"] == (
        "SINGLE_ORIGIN_RANGE_COVERAGE_NO_GO"
    )
    assert predecessor["passing_sensor_conditions"] == []
    assert predecessor["training_authorized"] is False
    assert predecessor["mutable_fields"] == []

    corpus = receipt["frozen_corpus"]
    assert corpus == {
        "states": 176,
        "transitions": 29470,
        "physics_frames": 1473500,
        "protected_links": 13,
        "protected_collision_shapes": 27,
        "state_role_action_transition_contact_and_h3_identities": "immutable",
        "oracle_label": "repaired frozen_contact_label",
        "action_authority": "unique deployable applied-action contract",
        "route_authority": "deterministic H3 route scores",
        "fresh_panel": "forbidden",
    }
    assert receipt["roles"]["untouched_g2"] == "FORBIDDEN_NOT_READ"
    maps = receipt["representative_maps"]
    assert maps["decision_action_copy_map"]["representatives"] == 13385
    assert maps["decision_action_copy_map"]["sensor_reuse_authority"] is False
    assert maps["exact_sensor_materialization_map"]["representatives"] == 13584
    assert maps["exact_sensor_materialization_map"]["numeric_tolerance"] == 0.0
    assert maps["exact_sensor_materialization_map"][
        "exact_reused_transition_pairs"
    ] == 15886
    assert maps["exact_sensor_materialization_map"][
        "realistic_scan_reuse_authority"
    ] is False
    assert "29,470 transition UIDs" in maps["exact_sensor_materialization_map"][
        "realistic_scan_exclusion"
    ]
    assert maps["mapping_change"] == "forbidden"


def test_mounts_keep_entire_assumed_l2_housing_outside_trunk() -> None:
    receipt = contract.build_contract()
    mounts = receipt["mounts"]
    assert mounts["trunk_collision_envelope"]["half_extents_m"] == [
        0.1881,
        0.04675,
        0.057,
    ]
    assert receipt["hardware_binding"]["housing"]["aabb_full_extents_m"] == [
        0.075,
        0.075,
        0.065,
    ]
    assert receipt["hardware_binding"]["housing"]["mass_kg"] == 0.230
    assert mounts["mechanical_clearance_m"] == 0.01

    assert mounts["HEAD_STOCK"]["translation_m"] == [0.28945, 0.0, -0.046825]
    assert mounts["HEAD_STOCK"]["rotation_rpy_rad"] == [0.0, 2.8782, 0.0]
    assert mounts["REAR_TOP_TRUNK"]["translation_m"] == [-0.1254, 0.0, 0.0995]
    assert mounts["LEFT_UPPER_FLANK"]["translation_m"] == [0.0, 0.09425, 0.0285]
    assert mounts["RIGHT_UPPER_FLANK"]["translation_m"] == [0.0, -0.09425, 0.0285]

    # The full housing, not merely the ray origin, clears on the declared axis.
    assert 0.0995 - 0.065 / 2 - 0.057 == pytest.approx(0.01)
    assert 0.09425 - 0.075 / 2 - 0.04675 == pytest.approx(0.01)
    assert -0.09425 + 0.075 / 2 - (-0.04675) == pytest.approx(-0.01)
    assert mounts["supplemental_mount_constraints"]["leg_mounts"] == "forbidden"
    assert mounts["supplemental_mount_constraints"]["outcome_tuning"] is False


def test_orientation_library_is_canonical_and_outcome_independent() -> None:
    library = contract.build_contract()["orientation_library"]
    assert tuple(library["mount_ids"]) == contract.SUPPLEMENTAL_MOUNT_IDS
    assert tuple(library["orientation_ids_in_tie_order"]) == contract.ORIENTATION_IDS
    assert library["pole_definitions"] == {
        "LEVEL": "sensor local +Z = body [0,0,+1]",
        "INVERTED": "sensor local +Z = body [0,0,-1]",
        "OUTWARD_DOWNWARD": (
            "sensor local +Z = normalize(horizontal_outward_body_vector + [0,0,-1])"
        ),
        "INWARD_DOWNWARD": (
            "sensor local +Z = normalize(-horizontal_outward_body_vector + [0,0,-1])"
        ),
    }
    assert "body +Y" in library["canonical_basis"]["sensor_x"]
    assert library["canonical_basis"]["sensor_y"] == "sensor_z cross sensor_x"
    selection = library["static_selection"]
    assert selection["labels_or_transition_outcomes_read"] == []
    assert "nominally FOV-and-range-eligible" in selection[
        "self_occlusion_denominator"
    ]
    assert selection["zero_nominal_eligible_witnesses"] == "fail closed"
    assert selection["final_tie_break"] == "orientation_ids_in_tie_order"
    assert selection["must_be_frozen_before_transition_sensor_materialization"] is True
    assert selection["selected_numeric_binding"] == str(
        contract.TRACKED_MOUNT_LIBRARY_PATH
    )
    assert selection["selected_numeric_binding_sha256"] == (
        "47028f5ca82e983995aac6dea989acba1dd049fcffb553bbbee464b788e5d7b8"
    )
    assert selection["static_witness_count"] == 842
    assert selection["frozen_selections"]["REAR_TOP_TRUNK"]["orientation_id"] == (
        "INVERTED"
    )
    assert selection["frozen_selections"]["LEFT_UPPER_FLANK"][
        "orientation_id"
    ] == "INWARD_DOWNWARD"


def test_static_mount_receipt_is_exactly_bound_and_selected_transforms_are_frozen() -> None:
    receipt = contract.load_and_validate_mount_library()
    assert contract.MOUNT_LIBRARY_RECEIPT_SHA256 == (
        "47028f5ca82e983995aac6dea989acba1dd049fcffb553bbbee464b788e5d7b8"
    )
    assert receipt["content_digest"] == (
        "85683ac020d9fff0ec245ca9d2bf99a3942ae506f2c8c8bce7a49028a24392e6"
    )
    assert receipt["witness_count"] == 842
    assert receipt["witness_digest"] == (
        "663de93b5cf840b7fe05788a1460702349094b24a882f7500c481e30d5a3f264"
    )
    assert receipt["outcome_fields_read"] == []
    assert receipt["contact_outcomes_used"] is False
    assert receipt["pass"] is True
    assert receipt["housing_occlusion_frame"] == (
        "BODY_AXIS_ALIGNED_TRUNK_FRAME_INDEPENDENT_OF_OPTICAL_RAY_FRAME"
    )
    clearance_rows = receipt["housing_clearance_validation"]
    assert [row["mount_id"] for row in clearance_rows] == list(contract.MOUNT_IDS)
    assert all(row["pass"] for row in clearance_rows)
    assert all(
        row["protected_primitive_count"] == 27
        and row["minimum_protected_geometry_clearance_m"]
        == pytest.approx(0.01, abs=2e-16)
        for row in clearance_rows[1:]
    )
    selections = {
        row["mount_id"]: row for row in receipt["orientation_selections"]
    }
    assert selections["HEAD_STOCK"]["selected_orientation_id"] == "STOCK"
    assert selections["REAR_TOP_TRUNK"]["selected_orientation_id"] == "INVERTED"
    assert selections["LEFT_UPPER_FLANK"]["selected_orientation_id"] == (
        "INWARD_DOWNWARD"
    )
    assert selections["RIGHT_UPPER_FLANK"]["selected_orientation_id"] == (
        "INWARD_DOWNWARD"
    )
    assert selections["REAR_TOP_TRUNK"]["selected_pose"]["quaternion_body_wxyz"] == [
        0.0,
        1.0,
        0.0,
        0.0,
    ]
    assert selections["LEFT_UPPER_FLANK"]["selected_pose"][
        "quaternion_body_wxyz"
    ] == [0.3826834323650898, 0.9238795325112867, 0.0, 0.0]
    assert selections["RIGHT_UPPER_FLANK"]["selected_pose"][
        "quaternion_body_wxyz"
    ] == [0.3826834323650898, -0.9238795325112867, -0.0, -0.0]
    for row in receipt["orientation_selections"]:
        for candidate in row["candidates"]:
            assert candidate["nominal_witness_count"] > 0
            assert candidate["zero_nominal_fail_closed"] is False
            assert candidate["self_occlusion_fraction"] == pytest.approx(
                candidate["self_occluded_witness_count"]
                / candidate["nominal_witness_count"]
            )


def test_layout_candidates_and_label_free_selection_are_frozen() -> None:
    receipt = contract.build_contract()
    candidates = receipt["layout_candidates"]
    assert tuple(candidates["pair_layout_ids_in_tie_order"]) == contract.PAIR_LAYOUT_IDS
    assert tuple(candidates["three_layout_ids_in_tie_order"]) == (
        contract.THREE_LAYOUT_IDS
    )
    assert candidates["candidate_counts"] == {
        "pair": 3,
        "three": 3,
        "all_four_diagnostic": 1,
    }
    assert candidates["all_four_diagnostic_layout"] == {
        "id": contract.ALL_FOUR_LAYOUT_ID,
        "mounts": list(contract.MOUNT_IDS),
        "primary_layout_candidate": False,
    }

    selection = receipt["layout_selection"]
    assert selection["role"] == "training"
    assert selection["state_count"] == 128
    assert selection["outcome_fields_used_by_layout_objective"] == []
    assert selection["contact_labels_used_for_layout_selection"] is False
    assert (
        selection["frozen_outcomes_read_only_for_corpus_custody_validation"]
        is True
    )
    assert selection["geometry_reuse_contract"] == {
        "id": "LABEL_FREE_LAYOUT_GEOMETRY_REUSE_MAP",
        "partition": "within each deployable applied-action copy group only",
        "exact_fields": ["qpos", "link_transform", "geom_transform"],
        "boundary_snapshot_digest_must_match": True,
        "numeric_tolerance": 0.0,
        "outcome_fields_used_by_reuse_or_objective": [],
        "explicitly_forbidden_authority": (
            "EXACT_SENSOR_MATERIALIZATION_MAP because its validation fields include "
            "contact outcomes"
        ),
        "coverage": "every transition in all 128 training-role states exactly once",
    }
    assert selection["lexicographic_selection"] == [
        "highest minimum_body_region_support",
        "highest p5_transition_support",
        "highest rear_limb_support",
        "highest calf_support",
        "highest overall_mean_support",
        "lowest self_occlusion",
        "earlier fixed layout ID order",
    ]
    assert selection["body_region_ids"] == [
        "TRUNK",
        "FRONT_LIMBS",
        "REAR_LIMBS",
        "HIPS_AND_THIGHS",
        "CALVES",
    ]
    assert "13x50 witnesses" in selection["metrics"]["p5_transition_support"]
    assert "every nominally eligible origin" in selection["metrics"]["self_occlusion"]
    assert selection["zero_nominal_support_denominator"].startswith("fail closed")
    assert selection["row_level_evidence"]["path"] == (
        "layout_selection/training_layout_evidence.jsonl.gz"
    )
    assert selection["row_level_evidence"][
        "forbidden_outcomes_persisted_as_null"
    ] == ["contact_label", "safe_action_count", "route_outcome"]
    assert "before calibration" in selection["selection_freeze"]


def test_condition_ids_modes_and_conditional_flow_are_exact() -> None:
    receipt = contract.build_contract()
    assert tuple(row["id"] for row in receipt["conditions"]) == contract.CONDITION_IDS
    assert contract.REGRESSION_CONDITION_IDS == (
        "REALISTIC_PLATFORM_SCAN",
        "DENSE_FOV_UPPER_BOUND_AT_PLATFORM_MOUNT",
    )
    assert contract.DUAL_CONDITION_IDS == (
        "DUAL_DENSE_SPHERICAL_ORIGIN_UPPER_BOUND",
        "DUAL_DENSE_L2_FOV_UPPER_BOUND",
        "DUAL_REALISTIC_L2_SCAN",
    )
    assert contract.THREE_CONDITION_IDS == (
        "THREE_DENSE_SPHERICAL_ORIGIN_UPPER_BOUND",
        "THREE_DENSE_L2_FOV_UPPER_BOUND",
        "THREE_REALISTIC_L2_SCAN",
    )
    assert contract.DIAGNOSTIC_CONDITION_IDS == (
        "ALL_FOUR_DENSE_SPHERICAL_DIAGNOSTIC",
    )
    assert len(receipt["evaluation_matrix"]) == 17
    diagnostic = receipt["conditions"][-1]
    assert diagnostic["evidence_modes"] == ["TRUE_FUTURE_OBSERVABILITY_CLOUD"]
    assert diagnostic["primary_classification_eligible"] is False

    flow = receipt["conditional_execution"]
    assert flow["dual_early_stop"]["condition"] == "DUAL_REALISTIC_L2_SCAN"
    assert flow["dual_early_stop"]["requires_complete_gate_pass"] is True
    assert "does not pass every" in flow["three_stage_required_when"]
    assert "THREE_DENSE_SPHERICAL" in flow["all_four_diagnostic_required_when"]
    assert flow["all_four_limits"]["mode"] == (
        "TRUE_FUTURE_OBSERVABILITY_CLOUD only"
    )
    assert flow["all_four_limits"]["realistic_four_origin_scan"] == "forbidden"
    assert flow["all_four_limits"]["primary_classification_authority"] is False


def test_scan_phase_binding_is_deterministic_independent_and_has_known_answer() -> None:
    digest = "0123456789abcdef" * 4
    head = contract.derive_multi_origin_scan_phases(
        contract_digest_sha256=digest,
        transition_uid="state-01/current/7",
        mount_id="HEAD_STOCK",
    )
    assert head == contract.derive_multi_origin_scan_phases(
        contract_digest_sha256=digest,
        transition_uid="state-01/current/7",
        mount_id="HEAD_STOCK",
    )
    assert head == {
        "phase_namespace": (
            "MINIMUM_MULTI_ORIGIN_BODY_RANGE_COVERAGE_QUALIFICATION_V1/L2_PHASE_V1"
        ),
        "contract_digest_sha256": digest,
        "transition_uid": "state-01/current/7",
        "mount_id": "HEAD_STOCK",
        "phase_digest_sha256": (
            "92c073ad1dd7f324ed1b78000bb61909d0473a3328306971d362f40e40600342"
        ),
        "horizontal_phase_cycles": 0.573249082340993,
        "vertical_phase_cycles": 0.9262003898727119,
    }
    rear = contract.derive_multi_origin_scan_phases(
        contract_digest_sha256=digest,
        transition_uid="state-01/current/7",
        mount_id="REAR_TOP_TRUNK",
    )
    assert rear["phase_digest_sha256"] != head["phase_digest_sha256"]
    with pytest.raises(contract.ContractError, match="lowercase"):
        contract.derive_multi_origin_scan_phases(
            contract_digest_sha256="A" * 64,
            transition_uid="x",
            mount_id="HEAD_STOCK",
        )
    with pytest.raises(contract.ContractError, match="non-empty"):
        contract.derive_multi_origin_scan_phases(
            contract_digest_sha256=digest,
            transition_uid="",
            mount_id="HEAD_STOCK",
        )
    with pytest.raises(contract.ContractError, match="mount_id"):
        contract.derive_multi_origin_scan_phases(
            contract_digest_sha256=digest,
            transition_uid="x",
            mount_id="UNKNOWN",
        )
    scan_contract = contract.build_contract()["scan_contract"]
    assert "29,470 transition UIDs" in scan_contract[
        "transition_identity_independence"
    ]
    assert scan_contract["realistic_scan_count_semantics"] == {
        "sensor_scans_per_condition_mode": (
            "29,470 transitions multiplied by the selected layout origin count"
        ),
        "unique_rendered_scans": (
            "exactly sensor_scans; render_cache_reused is false"
        ),
        "rays": "sensor_scans multiplied by 6,400 rays",
        "dense_conditions": (
            "have analytic witness/acquisition counts, not finite realistic scan counts, "
            "and may reuse the exact sensor materialization map"
        ),
    }


def test_fusion_support_provenance_and_error_classes_are_frozen() -> None:
    receipt = contract.build_contract()
    fusion = receipt["visibility_and_fusion"]
    assert "27-shape" in fusion["self_occlusion"]
    assert fusion["occluded_space"] == "unsupported, never free"
    housing = fusion["emitting_housing_ray_policy"]
    assert "own 75x75x65" in housing["own_emitter"]
    assert "every other" in housing["other_installed_supplemental_housings"]
    assert "body-axis-aligned" in housing["other_installed_supplemental_housings"]
    assert "rotates inside" in housing["optical_mechanical_decoupling"]
    assert housing["head_origin_robot_geom_exemption"] == [1, 2]
    assert "supplemental origin" in housing["supplemental_origin_head_rule"]
    assert "never alter" in housing["protected_geometry_effect"]
    assert "not approved hardware" in housing["limitation"]
    assert housing["outcome_tuning"] is False
    assert "set union" in fusion["fusion"]
    assert "at least one" in fusion["multi_origin_support"]
    assert "every supporting origin" in fusion["provenance"]
    dominance = fusion["realistic_matched_dense_l2_fov_compatibility"]
    assert dominance["same_object_witness_radius_m"] == 0.10
    assert "every origin" in dominance["dominance_assertion"]
    assert "DENSE_SPHERICAL" in dominance["nested_dominance_assertion"]
    assert dominance["clearance_monotonicity_tolerance_m"] == 1e-9
    assert "no greater" in dominance["clearance_monotonicity"]
    assert "every member transition UID" in dominance[
        "exact_geometry_copy_group_rule"
    ]
    assert "before any threshold calibration" in dominance["decision_timing"]
    assert dominance["outcome_fields_used"] == []
    assert "not angular-grid tuning" in dominance["scientific_status"]
    assert tuple(receipt["coverage_attribution"]["error_classes"]) == (
        contract.COVERAGE_ERROR_CLASSES
    )
    assert contract.COVERAGE_ERROR_CLASSES == (
        "INSUFFICIENT_ORIGIN_COUNT",
        "VERTICAL_FOV_LIMITATION",
        "SCAN_PATTERN_SPARSITY",
        "SCAN_TIMING_LIMITATION",
        "ROBOT_SELF_OCCLUSION",
        "NEAR_BLIND_REGION",
        "MOUNT_POSITION_LIMITATION",
        "POINT_FUSION_ERROR",
        "UNRESOLVED",
    )
    hierarchy = receipt["coverage_attribution"]["mutually_exclusive_hierarchy"]
    assert [row["class"] for row in hierarchy] == [
        "POINT_FUSION_ERROR",
        "NEAR_BLIND_REGION",
        "ROBOT_SELF_OCCLUSION",
        "VERTICAL_FOV_LIMITATION",
        "SCAN_TIMING_LIMITATION",
        "SCAN_PATTERN_SPARSITY",
        "INSUFFICIENT_ORIGIN_COUNT",
        "MOUNT_POSITION_LIMITATION",
        "UNRESOLVED",
    ]
    assert "ambiguous" in receipt["coverage_attribution"]["ambiguity_rule"]


def test_threshold_gate_classification_precedence_cannot_drift() -> None:
    receipt = contract.build_contract()
    calibration = receipt["threshold_calibration"]
    assert calibration["role"] == "internal_calibration"
    assert calibration["eligibility"] == {
        "combined_current_successor_contact_recall_gte": 0.95,
        "combined_current_successor_false_negative_rate_lte": 0.05,
    }
    assert len(calibration["lexicographic_selection"]) == 7
    assert calibration["freeze_before_heldout"] is True

    gate = receipt["immutable_gate"]
    assert gate["mode"] == "TRUE_FUTURE_OBSERVABILITY_CLOUD only"
    assert gate["current_contact_auc_gte"] == 0.90
    assert gate["successor_contact_auc_gte"] == 0.90
    assert gate["combined_contact_recall_gte"] == 0.95
    assert gate["combined_contact_false_negative_rate_lte"] == 0.05
    assert gate["safe_action_zero_nonzero_accuracy_gte"] == 0.90
    assert gate["safe_action_false_nonzero_rate_lte"] == 0.05
    assert gate["oracle_viable_denominator"] == 20
    assert gate["oracle_viable_states_retaining_action_gte"] == 18
    assert gate["oracle_nonviable_denominator"] == 4
    assert gate["oracle_nonviable_correct_abstentions_eq"] == 4
    assert gate["selected_immediate_contacts_eq"] == 0
    assert gate["selected_nonviable_successors_eq"] == 0
    assert gate["h3_route_progress_fraction_of_exact_geometry_gte"] == 0.80
    assert gate["normalized_viability_constrained_regret_lte"] == 0.20
    assert gate["best_admissible_top_3_gte"] == 0.75
    assert gate["no_family_collapse"]["required"] is True

    classes = receipt["classifications"]
    assert tuple(classes["primary_exactly_one"]) == contract.PRIMARY_CLASSIFICATIONS
    assert [row["classification"] for row in classes["precedence"]] == list(
        contract.PRIMARY_CLASSIFICATIONS
    )
    assert "never replaces" in classes["all_four_secondary_rule"]
    assert "FOUR_OR_MORE_ORIGINS_THEORETICALLY_REQUIRED" in classes["secondary"]
    assert classes["replanning_interface"] == "REPLANNING_INTERFACE_UNRESOLVED"


def test_compute_storage_environment_and_fixtures_are_frozen() -> None:
    receipt = contract.build_contract()
    benchmark = receipt["compute_benchmark"]
    assert benchmark["device"] == "CPU"
    assert benchmark["numeric_dtype"] == "float32"
    assert "already materialized float32 physics-step/protected-link" in benchmark[
        "timed_inputs"
    ]
    assert "raw ray clouds are not timed inputs" in benchmark["timed_inputs"]
    assert "structured per-link state and transition contact decision" in benchmark[
        "scope"
    ]
    assert benchmark["warmups"] == 30
    assert benchmark["timed_iterations_gte"] == 1000
    assert benchmark["classification"] == {
        "MULTI_ORIGIN_SET_REDUCTION_COMPUTE_SIGNAL": (
            "P99 <= 50 ms and maximum <= 80 ms"
        ),
        "MULTI_ORIGIN_SET_REDUCTION_COMPUTE_POSITIVE_TENDENCY": (
            "not SIGNAL, P99 <= 80 ms and maximum <= 100 ms"
        ),
        "MULTI_ORIGIN_SET_REDUCTION_COMPUTE_NO_GO": "otherwise",
    }
    assert benchmark["replanning_interface"] == "REPLANNING_INTERFACE_UNRESOLVED"
    benchmark_schema = contract.build_output_schema()["files"]["compute_benchmark"]
    assert {
        "numeric_binding",
        "timed_sample_unit",
        "includes",
        "includes_ray_generation",
        "includes_future_trajectory_acquisition",
    }.issubset(benchmark_schema["required_keys"])

    storage = receipt["storage"]
    assert storage["output_root"] == str(contract.OUTPUT_ROOT)
    assert storage["output_filesystem_required_type"] == "ext4"
    assert storage["output_must_be_different_device_from_workspace"] is True
    assert storage["minimum_output_free_bytes"] == 100_000_000_000
    assert storage["minimum_workspace_free_bytes"] == 20_000_000_000
    assert storage["temporary_storage_ceiling_bytes"] == 50_000_000_000
    assert storage["final_storage_ceiling_bytes"] == 25_000_000_000
    assert storage["evaluation_mode"] == "streaming"

    runtime = receipt["source_and_environment"]["known_runtime"]
    assert runtime["python"] == "3.12.3"
    assert runtime["genesis"] == "0.3.14"
    assert runtime["numpy"] == "2.4.6"
    assert runtime["scipy"] == "1.17.1"
    assert runtime["cpu_only"] is True
    assert runtime["tinyquadjepa_required"] is False
    assert runtime["training_packages_required"] is False
    assert "new contract, core, metrics" in receipt["source_and_environment"][
        "experiment_import_closure_required"
    ]

    required = receipt["fixtures"]["required"]
    assert len(required) == 20
    assert "one origin occluded while another observes" in required
    assert "complementary left/right flank coverage" in required
    assert "complementary head/rear coverage" in required
    assert "synchronized scan overlap" in required
    assert "independent phased scans" in required
    assert "byte-identical regeneration" in required
    raw = receipt["fixtures"]["raw_evidence"]
    assert raw["location"] == "fixture.core.raw_fixture_evidence"
    assert raw["schema"] == (
        "minimum_multi_origin_body_range_coverage_fixture_raw_evidence_v1"
    )
    assert raw["complete_raw_ray_queries"] == [
        "near_blind",
        "between_scan_samples",
        "one_origin_occluded_another_observes",
    ]
    assert set(raw["complete_reduced_origin_evidence"]) == {
        "one_origin_occluded_another_observes",
        "complementary_left_right_flank",
        "complementary_head_rear",
        "synchronized_scan_overlap",
    }
    assert raw["separate_raw_artifact"] is False

    fixture_file = contract.build_output_schema()["files"]["fixture_receipt"]
    assert "raw_fixture_evidence" in fixture_file["core_required_keys"]
    assert fixture_file["raw_fixture_evidence_schema"] == (
        "minimum_multi_origin_body_range_coverage_fixture_raw_evidence_v1"
    )
    assert set(fixture_file["raw_fixture_evidence_required_keys"]) == {
        "schema",
        "complete_raw_ray_queries",
        "complete_reduced_origin_evidence",
        "reconstructible_noncloud_inputs",
        "raw_fixture_cloud_policy",
        "content_digest",
    }
    assert set(fixture_file["reconstructible_noncloud_input_required_keys"]) == {
        "robot_primitives",
        "clear_query_points_world_xyz_m",
        "clear_query_result_m",
        "contact_queries",
        "scan_phase_inputs",
        "safe_successor_inputs",
        "threshold_tie_input",
        "h3_rows",
    }
    raw_queries = fixture_file["raw_ray_query_required_keys"]
    assert "robot_primitives" in raw_queries["near_blind"]
    assert "robot_primitives" in raw_queries["between_scan_samples"]
    assert {
        "blocked_direction_world",
        "observed_direction_world",
        "timestamps_s",
        "near_m",
        "far_m",
    }.issubset(raw_queries["one_origin_occluded_another_observes"])


def test_output_schema_persists_selection_fusion_hardware_and_conditional_flow() -> None:
    schema = contract.build_output_schema()
    assert "per_origin_coverage" not in schema["condition_metric_groups"]
    assert schema["coverage_support_path"] == (
        "result.coverage_support[condition_id]"
    )
    assert schema["coverage_support_groups"] == [
        "per_origin",
        "per_link",
        "per_region",
        "per_family",
    ]
    files = schema["files"]
    assert {
        "mount_library_receipt",
        "layout_selection_receipt",
        "training_layout_evidence",
        "execution_plan_receipt",
        "predecessor_regression_receipt",
        "materialization_index",
        "calibration_thresholds",
        "transition_evidence",
        "per_link_evidence",
        "coverage_errors",
        "raw_audit_manifest",
        "result",
        "persistence_receipt",
    }.issubset(files)
    assert files["layout_selection_receipt"]["relative_path"] == (
        "layout_selection/selected_layouts.json"
    )
    assert {
        "outcome_fields_used_by_layout_objective",
        "contact_labels_used_for_layout_selection",
        "frozen_outcomes_read_only_for_corpus_custody_validation",
        "label_free_geometry_reuse",
        "training_layout_evidence",
    }.issubset(files["layout_selection_receipt"]["required_keys"])
    assert files["layout_selection_receipt"][
        "label_free_geometry_reuse_required_keys"
    ] == [
        "fields",
        "boundary_snapshot_digest_required",
        "numeric_tolerance",
        "action_group_partition_only",
        "outcome_fields_accessed",
        "representatives",
    ]
    assert {"per_link_counts", "per_link_support"}.issubset(
        files["layout_selection_receipt"]["candidate_metric_required_keys"]
    )
    training_rows = files["training_layout_evidence"]
    assert training_rows["relative_path"] == (
        "layout_selection/training_layout_evidence.jsonl.gz"
    )
    assert training_rows["row_schema"] == (
        "minimum_multi_origin_training_layout_transition_evidence_v1"
    )
    assert training_rows["layout_ids"] == [
        *contract.PAIR_LAYOUT_IDS,
        *contract.THREE_LAYOUT_IDS,
    ]
    assert training_rows["body_region_ids"] == list(contract.BODY_REGION_IDS)
    assert training_rows["protected_link_cardinality"] == 13
    assert {"protected_link_counts", "protected_link_support"}.issubset(
        training_rows["layout_metric_required_keys"]
    )
    assert training_rows["forbidden_outcome_value_rule"] == {
        "contact_label": None,
        "safe_action_count": None,
        "route_outcome": None,
    }
    assert files["materialization_index"]["relative_path"] == (
        "materialization/{dual,three,diagnostic}_index.json"
    )
    assert {
        "transition_uid_by_index",
        "transition_uid_by_index_sha256",
    }.issubset(files["materialization_index"]["state_phase_receipt_required_keys"])
    assert "transition_index" in files["materialization_index"][
        "scan_receipt_required_keys"
    ]
    assert "every finite-scan receipt" in files["materialization_index"][
        "transition_uid_binding_semantics"
    ]
    assert files["calibration_thresholds"]["relative_path"] == (
        "calibration/{phase}_thresholds_frozen.json"
    )
    assert files["transition_evidence"]["relative_path"].endswith(".jsonl.gz")
    assert files["per_link_evidence"]["relative_path"].endswith(".jsonl.gz")
    assert files["coverage_errors"]["relative_path"].endswith(".jsonl.gz")
    assert "newly executed multi-origin" in files["transition_evidence"]["cardinality"]
    assert {
        "supporting_origin_ids",
        "responsible_origin_id",
        "responsible_acquisition_time_s",
        "support_acquisition_times_s_by_origin",
    }.issubset(files["per_link_evidence"]["required_keys"])
    assert files["coverage_errors"]["error_class_enum"] == list(
        contract.COVERAGE_ERROR_CLASSES
    )
    assert "experiment_import_closure" in files["environment_receipt"][
        "required_keys"
    ]
    assert {
        "action_representative_transition_index",
        "geometry_representative_transition_index",
    }.issubset(files["transition_evidence"]["required_keys"])
    assert {
        "action_representative_transition_index",
        "geometry_representative_transition_index",
    }.issubset(files["per_link_evidence"]["required_keys"])
    assert {
        "phase_digest_sha256_by_mount",
        "source_action_representative_transition_uid",
        "source_geometry_representative_transition_uid",
        "boundary_snapshot_digest",
    }.issubset(files["raw_audit_manifest"]["required_keys"])
    assert {
        "selected_dual_layout_id",
        "selected_three_layout_id",
        "executed_condition_ids",
        "materialization_indices",
        "threshold_freezes",
    }.issubset(files["execution_plan_receipt"]["required_keys"])
    assert {
        "state_records",
        "action_representatives",
        "geometry_representatives",
        "dense_analytic_target_query_counts",
        "support_dominance",
        "raw_audit_manifest_sha256",
    }.issubset(files["materialization_index"]["required_keys"])
    assert {
        "finite_scan_support_inherited",
        "per_origin_finite_scan_support_inherited",
        "per_origin_support",
        "per_origin_self_occluded",
    }.issubset(files["materialization_index"]["npz_required_arrays"])
    assert {
        "state_receipt_path",
        "state_receipt_sha256",
        "state_receipt_bytes",
    }.issubset(files["materialization_index"]["state_record_required_keys"])
    assert files["materialization_index"][
        "support_dominance_chain_key_format"
    ] == (
        "<REALISTIC_CONDITION_ID><=<MATCHED_DENSE_L2_FOV_CONDITION_ID>"
        "<=<MATCHED_DENSE_SPHERICAL_CONDITION_ID>"
    )
    assert files["materialization_index"][
        "support_dominance_required_keys"
    ] == [
        "condition_chain",
        "evidence_mode",
        "per_origin_queries_checked",
        "per_origin_subset_violations",
        "fused_queries_checked",
        "fused_subset_violations",
        "clearance_monotonicity_pairs_checked",
        "clearance_monotonicity_violations",
        "maximum_clearance_excess_m",
        "inherited_support_witnesses_by_origin",
        "pass",
    ]
    assert {
        "observation_support_by_origin",
        "self_occlusion_by_origin",
        "finite_scan_support_inherited_by_origin",
        "finite_scan_support_inherited_count",
    }.issubset(files["per_link_evidence"]["required_keys"])
    assert {
        "layout_selection",
        "execution_plan",
        "support_dominance",
        "condition_metrics",
        "gate_results",
        "hardware_accounting",
        "compute_classification",
        "primary_classification",
        "secondary_classifications",
        "result_content_sha256",
    }.issubset(files["result"]["required_keys"])


def test_tracked_contract_source_fixture_and_terminal_paths_are_frozen() -> None:
    receipt = contract.build_contract()["source_and_environment"]
    assert receipt["tracked_preregistration"] == str(
        contract.TRACKED_PREREGISTRATION_PATH
    )
    assert receipt["tracked_contract_receipt"] == str(
        contract.TRACKED_CONTRACT_RECEIPT_PATH
    )
    assert receipt["tracked_output_schema"] == str(
        contract.TRACKED_OUTPUT_SCHEMA_PATH
    )
    assert receipt["tracked_mount_library"] == str(
        contract.TRACKED_MOUNT_LIBRARY_PATH
    )
    assert receipt["tracked_source_closure"] == str(
        contract.TRACKED_SOURCE_CLOSURE_PATH
    )
    assert receipt["tracked_fixture"] == str(contract.TRACKED_FIXTURE_PATH)
    assert receipt["tracked_result"] == str(contract.TRACKED_RESULT_PATH)
    assert receipt["tracked_report"] == str(contract.TRACKED_REPORT_PATH)
    schema_files = contract.build_output_schema()["files"]
    assert schema_files["fixture_receipt"]["tracked_path"] == str(
        contract.TRACKED_FIXTURE_PATH
    )
    assert schema_files["result"]["tracked_path"] == str(
        contract.TRACKED_RESULT_PATH
    )
    assert schema_files["report"]["tracked_path"] == str(
        contract.TRACKED_REPORT_PATH
    )


def test_contract_validation_rejects_tampering_even_when_resigned() -> None:
    good = contract.build_contract()
    assert contract.validate_contract(good) == good

    tampered = contract.build_contract()
    tampered["mounts"]["REAR_TOP_TRUNK"]["translation_m"][2] = 0.1
    tampered.pop("contract_sha256")
    tampered["contract_sha256"] = contract.canonical_json_sha256(tampered)
    with pytest.raises(contract.ContractError, match="prospectively frozen"):
        contract.validate_contract(tampered)


def test_immutable_writers_and_loaders(tmp_path: Path) -> None:
    schema_path = tmp_path / "schema.json"
    assert contract.write_output_schema(schema_path) == schema_path
    assert schema_path.read_bytes() == contract.output_schema_receipt_bytes()
    assert contract.write_output_schema(schema_path) == schema_path
    assert contract.load_and_validate_output_schema(schema_path) == contract.OUTPUT_SCHEMA
    schema_path.write_bytes(b"{}\n")
    with pytest.raises(contract.ContractError, match="refusing to overwrite"):
        contract.write_output_schema(schema_path)

    contract_path = tmp_path / "contract.json"
    assert contract.write_contract(contract_path) == contract_path
    assert contract_path.read_bytes() == contract.contract_receipt_bytes()
    assert contract.write_contract(contract_path) == contract_path
    assert contract.load_and_validate_contract(contract_path) == contract.CONTRACT


def test_build_contract_does_not_share_mutable_state() -> None:
    changed = contract.build_contract()
    changed["conditions"][0]["layout"] = "invented"
    changed["prohibitions"].append("invented")
    fresh = contract.build_contract()
    assert fresh["conditions"][0]["layout"] == "HEAD_STOCK"
    assert "invented" not in fresh["prohibitions"]


def test_prohibitions_deny_training_g2_jepa_memory_and_scope_drift() -> None:
    prohibitions = set(contract.build_contract()["prohibitions"])
    assert {
        "model training",
        "fresh panel or corpus collection",
        "reading or opening untouched G2 evaluation",
        "opening or executing the JEPA predictor",
        "retraining recurrent memory",
        "learned closed-loop navigation",
        "implementing memory, novelty, routing, or beacon capture",
        "changing the 13 protected links or 27 protected collision shapes",
        "executing a realistic four-origin scan condition",
    }.issubset(prohibitions)
    assert contract.build_contract()["seeds"]["model_seed"] is None
