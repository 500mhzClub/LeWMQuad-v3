from __future__ import annotations

import copy

import pytest

from lewm.safety import physical_graph_edge_handoff_qualification_v4_contract as C


def test_exact_identity_subjects_roots_and_development_boundary() -> None:
    assert C.EXPERIMENT_ID == "PHYSICAL_GRAPH_EDGE_HANDOFF_QUALIFICATION_V4"
    assert C.SOURCE_PARENT_COMMIT == "5b08d433f2e69f6e4fe85c9b14696726e32d2ab1"
    assert C.V3_SOURCE_FREEZE_COMMIT == C.SOURCE_PARENT_COMMIT
    assert C.CONTRACT_FREEZE_COMMIT_SUBJECT == "Freeze tipped-state physical graph edge handoff qualification V4"
    assert C.RESULT_COMMIT_SUBJECT == "Evaluate tipped-state physical graph edge handoff qualification V4"
    assert str(C.OUTPUT_ROOT).endswith("physical_graph_edge_handoff_qualification_v4")
    assert str(C.MATERIAL_ROOT).endswith("physical_graph_edge_handoff_qualification_v4_material")
    assert str(C.EXTERNAL_REGENERATION_RECEIPT).endswith("physical_graph_edge_handoff_qualification_v4_regeneration_receipt.json")
    assert C.DEVELOPMENT_ONLY is True
    assert C.FINAL_EVALUATION_ELIGIBLE is False


def test_exact_taxonomy_precedence_and_hard_stop_boundary() -> None:
    assert C.STATE_DISPOSITIONS == (
        "QUALIFIED", "TEACHER_PHYSICS_CONTACT", "TEACHER_CROSSING_INVALID",
        "TEACHER_DID_NOT_LEAVE_SOURCE", "TEACHER_NO_POSITIVE_PROGRESS",
        "INITIAL_BOUNDARY_TIPPED", "RESTORATION_PROBE_TIPPED",
        "TEACHER_TERMINATED_UNSAFELY", "STATE_MATERIALISATION_CORRUPT",
        "STATE_NONDETERMINISTIC", "UNRESOLVED_STATE_FAILURE",
    )
    assert C.HARD_STOP_DISPOSITIONS == (
        "STATE_MATERIALISATION_CORRUPT", "STATE_NONDETERMINISTIC"
    )
    assert C.GLOBAL_HARD_STOP_REASONS == (
        "UNSUPPORTED_SNAPSHOT_SERIALIZATION",
        "SNAPSHOT_IDENTITY_REPRODUCTION_FAILURE",
    )
    assert C.DISPOSITION_PRECEDENCE[:7] == (
        "STATE_MATERIALISATION_CORRUPT", "STATE_NONDETERMINISTIC",
        "INITIAL_BOUNDARY_TIPPED", "RESTORATION_PROBE_TIPPED",
        "TEACHER_TERMINATED_UNSAFELY", "TEACHER_PHYSICS_CONTACT",
        "TEACHER_CROSSING_INVALID",
    )
    assert "UNRESOLVED_STATE_FAILURE" in C.DEFINED_NONQUALIFIED_DISPOSITIONS


def test_exact_success_and_panel_inadequate_inventories() -> None:
    assert len(C.SUCCESS_OUTPUT_LEAVES) == 27
    assert len(set(C.SUCCESS_OUTPUT_LEAVES)) == 27
    assert C.PANEL_INADEQUATE_OUTPUT_LEAVES == (
        "contract.json", "v1_v2_v3_custody_and_nonreuse.json",
        "scientific_invariance_receipt.json", "qualification_state_dispositions.jsonl",
        "panel_adequacy.json", "metrics.json", "result.json", "result.md",
        "file_hashes.json",
    )
    assert C.PANEL_INADEQUATE_OUTPUT_LEAF_COUNT == 9
    assert C.NEXT_DECISION_PANEL_INADEQUATE == "REVISE_GENERATOR_FOR_SHORTFALL_FAMILIES_KEEP_TEACHER_CONTRACT_FROZEN"


def test_exact_material_inventories_include_all_auxiliary_crosslinks() -> None:
    authority = C.MATERIAL_INVENTORY_AUTHORITY
    assert authority["panel_inadequate_exact_file_count"] == 514
    assert authority["success_exact_file_count"] == 805
    assert authority["success_root_files"] == [
        "material_contract.json", "prospective_pool.json",
        "teacher_selection.json", "panel_context.json", "encoding_receipt.json",
    ]
    assert authority["qualification_shard_count"] == 256
    assert authority["selected_shard_count_on_success"] == 64
    assert authority["fanout_shard_count_on_success"] == 64
    assert authority["repeat_shard_count_on_success"] == 16
    assert authority["no_unlisted_material_files"] is True


def test_result_authority_carries_complete_inherited_science_projection() -> None:
    authority = C.RESULT_PUBLICATION_AUTHORITY
    assert authority["success_downstream_scientific_metric_fields"] == sorted(
        C.V4_INHERITED_DOWNSTREAM_METRIC_FIELDS
    )
    for field in (
        "evidence_counts", "panel", "development", "heldout",
        "repeatability", "command_tracking", "runtime_environments",
        "stratified", "classification_input", "gate", "primary_classification",
        "secondary_classifications", "next_experiment",
    ):
        assert field in C.V4_INHERITED_DOWNSTREAM_METRIC_FIELDS
    assert authority["panel_inadequate_downstream_scientific_metrics"] is None


def test_initial_boundary_payload_is_exact_and_no_fabrication_is_authorized() -> None:
    assert set(C.INITIAL_BOUNDARY_PAYLOAD_AUTHORITY) == {
        "intended_base_pose_world", "base_pose_world", "base_twist_world",
        "joint_position", "joint_velocity", "previous_applied_command",
        "physics_contact", "sim_time_ns", "episode_step", "command_ticks",
        "policy_steps", "termination_flags",
    }
    assert C.INITIAL_BOUNDARY_PAYLOAD_AUTHORITY["intended_base_pose_world"] == {
        "dtype_str": "<f8", "shape": [7],
        "source": "contract-derived spawn xyz plus quaternion xyzw",
    }
    assert C.INITIAL_BOUNDARY_PAYLOAD_AUTHORITY["termination_flags"]["order"] == list(C.TERMINATION_FLAG_ORDER)
    assert C.STATE_DISPOSITION_AUTHORITY["zero_fill_or_fabricated_evidence_forbidden"] is True
    nonfinite = C.V4_TERMINAL_NONFINITE_AUTHORITY
    assert nonfinite["scope"] == "V4 terminal rejection evidence only"
    assert nonfinite["initial_allowed_members"] == [
        "base_pose_world", "base_twist_world", "joint_position",
        "joint_velocity",
    ]
    assert nonfinite["trace_allowed_members"] == [
        "base_pose_world", "base_twist_world", "joint_position",
        "joint_velocity",
    ]
    assert nonfinite["v3_snapshot_semantic_serializer_changed"] is False


def test_panel_authority_preserves_exact_64_state_design() -> None:
    authority = C.PANEL_ADEQUACY_AUTHORITY
    assert authority["candidate_count"] == 256
    assert authority["strata_per_family"] == 16
    assert authority["candidates_per_stratum"] == 4
    assert authority["required_selected_per_family"] == 16
    assert authority["required_panel_state_count"] == 64
    assert authority["role_assignment_after_adequacy"] == C.V3.FAMILY_ROLE_COUNTS


def test_v3_science_and_candidate_pool_are_exactly_invariant() -> None:
    contract = C.build_contract()
    assert C.scientific_invariance_projection(contract) == C.V3_SCIENTIFIC_PROJECTION
    assert C.build_candidate_specs() == C.V3.build_candidate_specs()
    assert C.scientific_constant_projection() == C.V3.scientific_constant_projection()
    assert C.SCIENTIFIC_INVARIANCE_AUTHORITY["historical_snapshot_or_probe_rerun"] is False


def test_pre_panel_binary64_correction_is_disclosed_and_requires_restart() -> None:
    authority = C.PRE_PANEL_ENGINEERING_CORRECTION_AUTHORITY
    assert C.validate_content_digest(authority) == authority
    assert authority["status"] == (
        "PARTIAL_QUALIFICATION_INVALIDATED_RESTART_REQUIRED"
    )
    assert authority["completed_pool_indices_at_detection"] == list(range(128))
    assert authority["qualification_material_pair_count_at_detection"] == 128
    assert authority["official_leaves_at_detection"] == [
        "contract.json", "scientific_invariance_receipt.json",
        "v1_v2_v3_custody_and_nonreuse.json",
    ]
    assert authority["detected_before_qualification_ledger_or_panel_construction"] is True
    assert authority["qualification_ledger_or_panel_adequacy_persisted"] is False
    assert authority["downstream_outcomes_opened"] is False
    assert len(authority["persisted_crossing_mismatch_pool_indices"]) == 19
    assert len(authority["unmaterialized_compact_projection_mismatch_pool_indices"]) == 38
    assert authority["dependent_fields_recomputed_from_canonical_projection"] == [
        "state_disposition.teacher_criteria.teacher_within_lateral_bounds",
        "panel_manifest.pool_qualification.teacher_within_lateral_bounds",
    ]
    assert authority["matched_frozen_producer_values_for_available_pool_count"] == 128
    assert authority[
        "corrected_full_raw_reduction_cross_runtime_exact_for_available_pool_count"
    ] == 128
    assert authority[
        "corrected_full_raw_reduction_aggregate_sha256_domain"
    ].endswith("including the canonical terminal LF")
    assert authority["corrected_full_raw_reduction_aggregate_sha256"] == (
        "a52b828c76286eb4e9d7ec6a0437db252fe93210c1057706a61f7cb3de97f746"
    )
    assert authority["cross_runtime_exact_match_required"] is True
    assert authority["existing_partial_material_reuse_authorized"] is False
    assert authority["file_hash_inequality_is_nonreuse_proof"] is False
    assert authority["restart_from_fresh_v4_roots_required"] is True
    assert authority["qualification_disposition_or_criterion_changed"] is False
    assert authority["mathematical_formula_or_tolerance_changed"] is False
    assert authority["v1_v2_v3_source_or_evidence_changed"] is False
    contract = C.build_contract()
    assert contract["pre_panel_engineering_correction_authority"] == authority
    assert C.scientific_invariance_projection(contract) == C.V3_SCIENTIFIC_PROJECTION
    assert C.SCIENTIFIC_INVARIANCE_AUTHORITY[
        "pre_panel_engineering_correction"
    ]["authority_content_digest"] == authority["content_digest"]
    manifests = authority["invalidated_partial_root_manifest_authority"]
    assert C.validate_content_digest(manifests) == manifests
    assert manifests["roots"]["official"] == {
        "path": str(C.OUTPUT_ROOT),
        "file_count": 3,
        "regular_file_apparent_bytes": 150396,
        "complete_root_projection_canonical_byte_count": 552,
        "complete_root_projection_sha256": (
            "c996f0f1242bc03756198d7eddfab69b6aeacf513150ffda13fdd567589a2766"
        ),
        "files_array_sha256": (
            "e9c8c9519a744a85b1cccc1746e7d7b473bcb67d999db820f5c469d486c49f20"
        ),
    }
    assert manifests["roots"]["material"] == {
        "path": str(C.MATERIAL_ROOT),
        "file_count": 258,
        "regular_file_apparent_bytes": 124565766,
        "complete_root_projection_canonical_byte_count": 35783,
        "complete_root_projection_sha256": (
            "2bd85555bc1e7f7e48039364273982925a0110921e130eee761f332896a02d76"
        ),
        "files_array_sha256": (
            "00346fd51dc42052143568e0b9bf26462182800eb1071cd8ec305f9772fc6cff"
        ),
    }


def test_contract_has_no_v3_first_eight_or_runtime_adoption_surface() -> None:
    contract = C.build_contract()
    forbidden = {
        "first_eight_reproduction_authority",
        "qualification_shard_augmentation_authority",
        "historical_snapshot_deserializer_authority",
        "v1_v2_custody_and_nonreuse_authority", "v3_runtime_policy",
        "v3_wrapper_dependency_paths",
    }
    assert forbidden.isdisjoint(contract)
    assert contract["external_historical_custody_authority"] == (
        C.EXTERNAL_HISTORICAL_CUSTODY_AUTHORITY
    )
    assert contract["scientific_invariance_authority"] == (
        C.SCIENTIFIC_INVARIANCE_AUTHORITY
    )
    assert contract["qualification_runtime_authority"] == (
        C.QUALIFICATION_RUNTIME_AUTHORITY
    )


def test_exact_v3_terminal_interpretation_and_sole_change_are_bound() -> None:
    assert C.V3_TERMINAL_DIAGNOSIS == (
        "TIPPED_BOUNDARY_STATE_DISPOSITION_UNSPECIFIED"
    )
    assert set(C.V3_TERMINAL_INTERPRETATION) == set(
        C.V3_TERMINAL_INTERPRETATION_FIELDS
    )
    assert C.V3_TERMINAL_INTERPRETATION[
        "snapshot_semantic_equivalence_passed"
    ] is True
    assert C.V3_TERMINAL_INTERPRETATION[
        "snapshot_behavioural_equivalence_passed"
    ] is True
    assert C.V3_TERMINAL_INTERPRETATION[
        "first_eight_reproduction_exact"
    ] is True
    assert C.V3_TERMINAL_INTERPRETATION[
        "raw_torch_snapshot_transport_bytes_are_scientific_evidence"
    ] is False
    assert C.V3_TERMINAL_INTERPRETATION[
        "prospective_panel_collection_stopped_because_tipped_state_disposition_unspecified"
    ] is True
    assert C.V3_TERMINAL_INTERPRETATION[
        "panel_fanout_ranker_or_heldout_outcomes_opened"
    ] is False
    assert C.V3_TERMINAL_INTERPRETATION[
        "scientific_handoff_result_produced"
    ] is False
    assert C.SCIENTIFIC_INVARIANCE_AUTHORITY["authorized_change_scope"] == (
        "Invalid tipped boundaries are explicitly recorded as nonqualified "
        "panel candidates instead of aborting the complete collection."
    )


def test_development_source_audit_disclosure_and_zero_counter_scopes_are_exact() -> None:
    disclosure = C.DEVELOPMENT_SOURCE_AUDIT_DISCLOSURE
    assert C.validate_content_digest(disclosure) == disclosure
    assert disclosure["disposition"] == (
        "ABORTED_DEVELOPMENT_TIME_IGNORE_RULE_BYPASS_ATTEMPT"
    )
    assert disclosure["matched_file_opens_or_reads"] == 0
    assert disclosure["printed_paths"] == 0
    assert disclosure["usable_output_items"] == 0
    assert disclosure["evidence_derived"] is False
    assert disclosure["sealed_content_accessed"] is False
    assert disclosure["scientific_outcomes_contaminated"] is False
    assert disclosure["replacement_audit_scope"] == (
        "FROZEN_SOURCE_CLOSURE_PATHS_ONLY_USING_IGNORE_HONORING_TOOLS"
    )
    assert C.EXTERNAL_HISTORICAL_CUSTODY_AUTHORITY["zero_counter_scope"] == (
        "EXTERNAL_HISTORICAL_CUSTODY_RECEIPT_BUILD_AND_EMISSION_PROCESS_ONLY"
    )
    assert C.V1_V2_V3_CUSTODY_AND_NONREUSE_AUTHORITY[
        "zero_counter_scope"
    ] == "V4_SCIENTIFIC_EXECUTION_PROCESS_ONLY"
    assert C.V4_RUNTIME_POLICY["zero_counter_scope"] == (
        "V4_SCIENTIFIC_EXECUTION_PROCESS_ONLY"
    )
    assert C.build_contract()["development_source_audit_disclosure"] == disclosure


def test_contract_and_runtime_roundtrip_fail_closed() -> None:
    contract = C.build_contract()
    assert C.validate_contract(contract) == contract
    binding = C.HISTORICAL_CUSTODY_RECEIPT_BINDING
    assert binding == {
        "path": str(C.HISTORICAL_CUSTODY_RECEIPT_PATH),
        "bytes": 95476,
        "sha256": (
            "0dfc1d39f8f9810d6a99cc69312a77935abf6f4bf71e10184ded3a49850c07cd"
        ),
    }
    runtime = C.build_runtime_contract("b" * 40, binding)
    assert C.validate_runtime_contract(runtime, source_freeze_commit="b" * 40, historical_custody_receipt_binding=binding) == runtime
    bad = copy.deepcopy(runtime)
    bad["v4_runtime_policy"]["models_trained"] = 1
    with pytest.raises(C.PhysicalGraphEdgeHandoffV4ContractError):
        C.validate_runtime_contract(bad)
    with pytest.raises(
        C.PhysicalGraphEdgeHandoffV4ContractError,
        match="invalidated V4 source freeze",
    ):
        C.build_runtime_contract(C.INVALIDATED_V4_SOURCE_FREEZE_COMMIT, binding)


def test_every_terminal_metadata_variant_binds_corrected_source_and_runtime() -> None:
    required = {"source_freeze_commit", "runtime_contract_content_digest"}
    for fields in (
        C.INITIAL_TIPPED_METADATA_FIELDS,
        C.PROBE_TIPPED_METADATA_FIELDS,
        C.TEACHER_TERMINAL_METADATA_FIELDS,
    ):
        assert required.issubset(fields)
    binding = C.STATE_DISPOSITION_AUTHORITY["terminal_source_runtime_binding"]
    assert binding["metadata_fields"] == [
        "source_freeze_commit", "runtime_contract_content_digest",
    ]
    assert binding["invalidated_source_freeze_commit_rejected"] == (
        C.INVALIDATED_V4_SOURCE_FREEZE_COMMIT
    )
    assert binding["invalidated_runtime_contract_content_digest_rejected"] == (
        C.INVALIDATED_V4_RUNTIME_CONTRACT_CONTENT_DIGEST
    )
    assert binding["all_256_terminal_bindings_must_be_equal"] is True
    assert binding["file_hash_inequality_is_nonreuse_proof"] is False


def test_persisted_array_authority_is_exactly_v4_scoped() -> None:
    authority = C.PERSISTED_ARRAY_HASH_AUTHORITY
    assert authority["npz_archive_comment_utf8"] == (
        "PHYSICAL_GRAPH_EDGE_HANDOFF_QUALIFICATION_V4:FRESH"
    )
    assert authority["npz_archive_comment_scope"] == (
        "every V4 material shard payload.npz governed by "
        "persisted_array_evidence"
    )
    assert authority["npz_archive_comment_semantics"].startswith(
        "V4-only deterministic container provenance"
    )
    assert all("V4" in scope for scope in authority["digest_field_scope"])
    assert authority["identified_defect_field"]["digest_domain"] == (
        "exact C-contiguous persisted bytes only"
    )
    assert "snapshot_behavioural_digest_v1" in C.PROBE_TIPPED_TRIAL_FIELDS
    assert "terminal x/y are finite" in (
        C.V4_TERMINAL_NONFINITE_AUTHORITY["terminal_region_membership_rule"]
    )


def test_tracked_allowlist_and_closure_are_unique_and_non_self_referential() -> None:
    assert len(C.TRACKED_SOURCE_PATHS) == 15
    assert len(C.TRACKED_SOURCE_PATHS) == len(set(C.TRACKED_SOURCE_PATHS))
    assert C.TRACKED_SOURCE_PATHS[4].endswith("_source_closure_2026-09-03.json")
    assert C.TRACKED_SOURCE_PATHS[4] not in C.SOURCE_CLOSURE_PATHS
    assert len(C.SOURCE_CLOSURE_PATHS) == len(set(C.SOURCE_CLOSURE_PATHS))
    assert tuple(C.SOURCE_DEPENDENCY_PATHS) == tuple(C.V3.SOURCE_DEPENDENCY_PATHS)
    assert set(C.TRACKED_SOURCE_PATHS[7:]).issubset(C.SOURCE_CLOSURE_PATHS)
    assert set(C.V3.SOURCE_CLOSURE_PATHS).issubset(C.SOURCE_CLOSURE_PATHS)
